"""Create searchable chunks from persisted meeting documents."""

import re
from uuid import uuid4

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp

from polderbot.schemas import CHUNK_METADATA_FIELDS, CHUNK_SCHEMA

_DEFAULT_CHUNK_SIZE = 400
_DEFAULT_CHUNK_OVERLAP = 50


def chunk_meeting_documents(
    spark: SparkSession,
    metadata_table: str,
    chunks_table: str,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ingestion_batch_id: str | None = None,
) -> int:
    """Append cleaned chunks for documents stored in a Delta table.

    Args:
        spark: Spark session used to read and write Delta tables.
        metadata_table: Fully qualified source documents table.
        chunks_table: Fully qualified destination chunks table.
        chunk_size: Maximum number of whitespace-delimited words per chunk.
        overlap: Number of words shared by consecutive chunks.
        ingestion_batch_id: Batch to process; defaults to the latest by ingestion timestamp.

    Returns:
        The number of chunks written.

    Raises:
        ValueError: If stored Markdown is invalid.

    """
    chunk_rows: list[dict[str, object]] = []
    documents = spark.table(metadata_table)
    if ingestion_batch_id is None:
        latest = (
            documents.orderBy(["ingest_ts", "ingestion_batch_id"], ascending=[False, False])
            .select("ingestion_batch_id")
            .first()
        )
        if latest is None:
            return 0
        ingestion_batch_id = latest.ingestion_batch_id
    documents = documents.where(col("ingestion_batch_id") == ingestion_batch_id)
    for row in documents.toLocalIterator():
        document = row.asDict()
        markdown_text = document["markdown_text"]
        if not isinstance(markdown_text, str) or not markdown_text.strip():
            raise ValueError(f"Document {document['document_id']} does not have Markdown text")
        metadata = {field.name: document[field.name] for field in CHUNK_METADATA_FIELDS}
        for chunk_number, text in _extract_chunks(_clean_text(markdown_text), chunk_size, overlap):
            chunk_rows.append(
                {
                    **metadata,
                    "chunk_number": chunk_number,
                    "text": text.strip(),
                    "id": str(uuid4()),
                }
            )

    if not chunk_rows:
        return 0

    dataframe = spark.createDataFrame(chunk_rows, schema=CHUNK_SCHEMA).withColumn("ingest_ts", current_timestamp())
    dataframe.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(chunks_table)
    spark.sql(f"""
        ALTER TABLE {chunks_table}
        SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    return len(chunk_rows)


def _clean_text(text: str) -> str:
    """Normalize line-break hyphenation and whitespace for search.

    Args:
        text: Raw Markdown text to normalize.

    Returns:
        Text with line-break hyphenation removed and whitespace collapsed.

    """
    dehyphenated_text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", dehyphenated_text).strip()


def _extract_chunks(markdown_text: str, chunk_size: int, overlap: int) -> list[tuple[str, str]]:
    """Split text into word windows while preserving its original whitespace.

    Args:
        markdown_text: Text to split.
        chunk_size: Maximum number of whitespace-delimited words per chunk.
        overlap: Number of words shared by consecutive chunks.

    Returns:
        Chunk numbers and their content in document order.

    """
    words = list(re.finditer(r"\S+", markdown_text))
    if not words:
        return [("0", markdown_text)] if markdown_text else []

    chunks: list[tuple[str, str]] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        text_start = 0 if start == 0 else words[start].start()
        text_end = words[end].start() if end < len(words) else len(markdown_text)
        chunks.append((str(len(chunks)), markdown_text[text_start:text_end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks
