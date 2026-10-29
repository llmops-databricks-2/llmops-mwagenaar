"""Generate embeddings for stored document chunks."""

from pyspark.sql import SparkSession


def embed_chunks(spark: SparkSession, chunks_table: str, embedding_endpoint: str) -> None:
    """Generate embeddings for chunks that do not yet have one.

    Args:
        spark: Spark session used to query and update the Delta table.
        chunks_table: Fully qualified Delta table containing chunks.
        embedding_endpoint: Databricks endpoint to invoke with ai_query.

    """
    spark.sql(
        """
        MERGE INTO IDENTIFIER(:chunks_table) AS target
        USING (
            SELECT id, CAST(ai_query(:embedding_endpoint, text) AS ARRAY<FLOAT>) AS embedding
            FROM IDENTIFIER(:chunks_table)
            WHERE embedding IS NULL
        ) AS source
        ON target.id = source.id
        WHEN MATCHED AND target.embedding IS NULL THEN UPDATE SET target.embedding = source.embedding
        """,
        args={"chunks_table": chunks_table, "embedding_endpoint": embedding_endpoint},
    )
