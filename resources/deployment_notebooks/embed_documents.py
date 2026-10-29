# Databricks notebook source
"""Chunk previously ingested documents and generate embeddings."""

# COMMAND ----------

from databricks.sdk.runtime import dbutils
from pyspark.sql import SparkSession

from polderbot.config import load_config
from polderbot.embeddings.chunking import chunk_meeting_documents
from polderbot.embeddings.embedding import embed_chunks

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("ingestion_batch_id", "")
dbutils.widgets.text("ingested_count", "")
env = dbutils.widgets.get("env")
ingestion_batch_id = dbutils.widgets.get("ingestion_batch_id")
ingested_count = dbutils.widgets.get("ingested_count")

cfg = load_config("../../project_config.yml", env=env)
spark = SparkSession.builder.getOrCreate()
chunks_table = cfg.get_full_table_name(cfg.chunks_table)

# COMMAND ----------

if ingested_count == "0":
    count = 0
else:
    count = chunk_meeting_documents(
        spark=spark,
        metadata_table=cfg.get_full_table_name(cfg.metadata_table),
        chunks_table=chunks_table,
        ingestion_batch_id=ingestion_batch_id or None,
    )
print(f"Wrote {count} chunks to {chunks_table}.")

if ingested_count == "0" and not spark.catalog.tableExists(chunks_table):
    print(f"No chunks to embed in {chunks_table}.")
else:
    embed_chunks(spark, chunks_table, cfg.embedding_endpoint)
    print(f"Generated missing embeddings in {chunks_table}.")
