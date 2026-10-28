"""Ingest OpenBesluitvorming meeting documents into Delta tables."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import current_timestamp

from polderbot.ingestion.models import MeetingDocument

_DEFAULT_CHUNK_SIZE = 2000
_DEFAULT_CHUNK_OVERLAP = 300


class MeetingDocumentMetadata(BaseModel):
    """Metadata for a document and its meeting-agenda context."""

    meeting_id: str
    meeting_title: str | None
    meeting_date: str | None
    agenda_item_titles: list[str | None]
    document_id: str
    document_title: str | None
    document_date: str | None
    document_name: str
    document_content_type: str | None
    document_source_url: str | None
    pdf_url: str | None


_MEETING_DOCUMENT_METADATA_SCHEMA = T.StructType(
    [
        T.StructField("meeting_id", T.StringType(), nullable=False),
        T.StructField("meeting_title", T.StringType(), nullable=True),
        T.StructField("meeting_date", T.StringType(), nullable=True),
        T.StructField("agenda_item_titles", T.ArrayType(T.StringType()), nullable=True),
        T.StructField("document_id", T.StringType(), nullable=False),
        T.StructField("document_title", T.StringType(), nullable=True),
        T.StructField("document_date", T.StringType(), nullable=True),
        T.StructField("document_name", T.StringType(), nullable=False),
        T.StructField("document_content_type", T.StringType(), nullable=True),
        T.StructField("document_source_url", T.StringType(), nullable=True),
        T.StructField("pdf_url", T.StringType(), nullable=True),
    ]
)

_CHUNK_SCHEMA = T.StructType(
    [
        *_MEETING_DOCUMENT_METADATA_SCHEMA.fields,
        T.StructField("chunk_id", T.StringType(), nullable=False),
        T.StructField("text", T.StringType(), nullable=False),
        T.StructField("id", T.StringType(), nullable=False),
    ]
)


def ingest_meeting_documents(
    spark: SparkSession,
    meeting_documents: Iterable[MeetingDocument],
    metadata_table: str,
    chunks_table: str,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[MeetingDocumentMetadata]:
    """Store meeting metadata and chunks of the API-provided Markdown.

    Args:
        spark: Spark session used to save Delta DataFrames.
        meeting_documents: Documents and meeting context to ingest.
        metadata_table: Fully qualified Delta table name for metadata.
        chunks_table: Fully qualified Delta table name for Markdown chunks.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by consecutive chunks.

    Returns:
        Metadata for the ingested meeting documents.

    Raises:
        ValueError: If a document has no non-empty Markdown text.

    """
    metadata: list[MeetingDocumentMetadata] = []
    chunk_rows: list[dict[str, object]] = []

    for meeting_document in meeting_documents:
        markdown_text = meeting_document.document.markdown_text
        if markdown_text is None or not markdown_text.strip():
            raise ValueError(f"Document {meeting_document.document.entity_id} does not have Markdown text")

        record = _build_meeting_document_metadata(meeting_document)
        metadata.append(record)
        for chunk_id, text in _extract_chunks(markdown_text, chunk_size, overlap):
            chunk_rows.append(
                {
                    **record.model_dump(),
                    "chunk_id": chunk_id,
                    "text": text,
                    "id": f"{record.document_id}_{chunk_id}",
                }
            )

    if not metadata:
        return metadata

    _save_rows(spark, [record.model_dump() for record in metadata], _MEETING_DOCUMENT_METADATA_SCHEMA, metadata_table)
    _save_rows(spark, chunk_rows, _CHUNK_SCHEMA, chunks_table)
    spark.sql(f"""
        ALTER TABLE {chunks_table}
        SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    return metadata


def _build_meeting_document_metadata(meeting_document: MeetingDocument) -> MeetingDocumentMetadata:
    """Build flat Delta metadata from a meeting document.

    Args:
        meeting_document: Document and meeting context to record.

    Returns:
        Metadata ready to be written to the Delta table.

    """
    document = meeting_document.document
    return MeetingDocumentMetadata(
        meeting_id=meeting_document.meeting.entity_id,
        meeting_title=meeting_document.meeting.title,
        meeting_date=meeting_document.meeting.date.isoformat() if meeting_document.meeting.date else None,
        agenda_item_titles=[agenda_item.title for agenda_item in meeting_document.agenda_item_path],
        document_id=document.entity_id,
        document_title=document.title,
        document_date=document.date.isoformat() if document.date else None,
        document_name=meeting_document.document_link.name,
        document_content_type=document.content_type or meeting_document.document_link.content_type,
        document_source_url=document.download_url or meeting_document.document_link.original_url or document.pdf_url,
        pdf_url=document.pdf_url,
    )


def _extract_chunks(markdown_text: str, chunk_size: int, overlap: int) -> list[tuple[str, str]]:
    """Split Markdown into fixed-size character windows.

    Args:
        markdown_text: Markdown text supplied by OpenBesluitvorming.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by consecutive chunks.

    Returns:
        Chunk identifiers and their Markdown content in document order.

    """
    chunks: list[tuple[str, str]] = []
    start = 0
    while start < len(markdown_text):
        end = min(start + chunk_size, len(markdown_text))
        chunks.append((str(len(chunks)), markdown_text[start:end]))
        if end == len(markdown_text):
            break
        start = end - overlap
    return chunks


def _save_rows(
    spark: SparkSession,
    rows: list[dict[str, object]],
    schema: T.StructType,
    table_name: str,
) -> None:
    """Append records with an ingestion timestamp to a Delta table.

    Args:
        spark: Spark session used to create and save the DataFrame.
        rows: Records to persist.
        schema: Schema for the records.
        table_name: Fully qualified Delta table name.

    """
    dataframe = spark.createDataFrame(rows, schema=schema).withColumn("ingest_ts", current_timestamp())
    dataframe.write.format("delta").mode("append").saveAsTable(table_name)
