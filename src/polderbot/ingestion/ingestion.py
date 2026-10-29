"""Ingest OpenBesluitvorming meeting documents into Delta tables."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

from polderbot.ingestion.models import MeetingDocument
from polderbot.schemas import MEETING_DOCUMENT_METADATA_SCHEMA


class MeetingDocumentMetadata(BaseModel):
    """Document Markdown and its meeting-agenda context."""

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
    markdown_text: str
    ingestion_batch_id: str | None = None


def ingest_meeting_documents(
    spark: SparkSession,
    meeting_documents: Iterable[MeetingDocument],
    metadata_table: str,
    *,
    ingestion_batch_id: str | None = None,
) -> list[MeetingDocumentMetadata]:
    """Store meeting metadata and the original API-provided Markdown.

    Args:
        spark: Spark session used to save Delta DataFrames.
        meeting_documents: Documents and meeting context to ingest.
        metadata_table: Fully qualified Delta table name for documents.
        ingestion_batch_id: Identifier for the batch; generated if omitted.

    Returns:
        Metadata for the ingested meeting documents.

    Raises:
        ValueError: If a document has no non-empty Markdown text.

    """
    metadata: list[MeetingDocumentMetadata] = []
    if ingestion_batch_id is None:
        ingestion_batch_id = str(uuid4())
    for meeting_document in meeting_documents:
        record = _build_meeting_document_metadata(meeting_document)
        record.ingestion_batch_id = ingestion_batch_id
        metadata.append(record)

    if not metadata:
        return metadata

    dataframe = spark.createDataFrame(
        [record.model_dump() for record in metadata], schema=MEETING_DOCUMENT_METADATA_SCHEMA
    ).withColumn("ingest_ts", current_timestamp())
    dataframe.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(metadata_table)
    return metadata


def _build_meeting_document_metadata(meeting_document: MeetingDocument) -> MeetingDocumentMetadata:
    """Build flat Delta metadata from a meeting document.

    Args:
        meeting_document: Document and meeting context to record.

    Returns:
        Metadata ready to be written to the Delta table.

    Raises:
        ValueError: If the document has no non-empty Markdown text.

    """
    document = meeting_document.document
    markdown_text = document.markdown_text
    if markdown_text is None or not markdown_text.strip():
        raise ValueError(f"Document {document.entity_id} does not have Markdown text")
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
        markdown_text=markdown_text,
    )
