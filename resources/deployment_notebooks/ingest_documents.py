# Databricks notebook source
"""Ingest documents and their original Markdown into Delta."""

# COMMAND ----------

from uuid import uuid4

from databricks.sdk.runtime import dbutils
from pyspark.sql import SparkSession

from polderbot.config import load_config
from polderbot.ingestion.client import OpenBesluitvormingClient
from polderbot.ingestion.ingestion import ingest_meeting_documents

MEETING_ID = "meeting:notubiz:waterschap:wetterskip_fryslan:783113"

dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")

cfg = load_config("../../project_config.yml", env=env)
spark = SparkSession.builder.getOrCreate()
ingestion_batch_id = str(uuid4())

# COMMAND ----------

with OpenBesluitvormingClient() as openbesluitvorming_client:
    meeting_documents = openbesluitvorming_client.get_documents_with_context_for_meeting(
        MEETING_ID, ignore_missing_documents=True
    )
    metadata = ingest_meeting_documents(
        spark=spark,
        meeting_documents=meeting_documents[:3],  # TODO remove slicing for full ingestion
        metadata_table=cfg.get_full_table_name(cfg.metadata_table),
        ingestion_batch_id=ingestion_batch_id,
    )

dbutils.jobs.taskValues.set(key="ingestion_batch_id", value=ingestion_batch_id)
dbutils.jobs.taskValues.set(key="ingested_count", value=len(metadata))
print(f"Ingestion batch: {ingestion_batch_id}")
print(f"Ingested {len(metadata)} documents from {MEETING_ID} into {cfg.get_full_table_name(cfg.metadata_table)}.")
