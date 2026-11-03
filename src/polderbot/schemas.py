"""Delta table schemas for meeting documents and their chunks."""

from pyspark.sql import types as T

MEETING_DOCUMENT_METADATA_SCHEMA = T.StructType(
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
        T.StructField("markdown_text", T.StringType(), nullable=True),
        T.StructField("ingestion_batch_id", T.StringType(), nullable=True),
        T.StructField("ingest_ts", T.TimestampType(), nullable=True),
    ]
)

CHUNK_METADATA_FIELDS = tuple(
    field
    for field in MEETING_DOCUMENT_METADATA_SCHEMA.fields
    if field.name not in {"markdown_text", "ingestion_batch_id", "ingest_ts"}
)

CHUNK_SCHEMA = T.StructType(
    [
        *CHUNK_METADATA_FIELDS,
        T.StructField("chunk_number", T.StringType(), nullable=False),
        T.StructField("text", T.StringType(), nullable=False),
        T.StructField("id", T.StringType(), nullable=False),
        T.StructField("embedding", T.ArrayType(T.FloatType()), nullable=True),
        T.StructField("ingest_ts", T.TimestampType(), nullable=True),
    ]
)
