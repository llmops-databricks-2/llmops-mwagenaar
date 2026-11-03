# Databricks notebook source
"""Ingest documents and their original Markdown into Delta."""

# COMMAND ----------

from uuid import uuid4

from databricks.sdk.runtime import dbutils
from pyspark.sql import SparkSession

from polderbot.config import load_config
from polderbot.ingestion.client import DEFAULT_SNAPSHOT_CURSOR, OpenBesluitvormingClient
from polderbot.ingestion.ingestion import get_last_ingested_meeting_id, ingest_meeting_documents

DEFAULT_MEETING_LIMIT = "1"

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("meeting_limit", DEFAULT_MEETING_LIMIT)
env = dbutils.widgets.get("env")
meeting_limit = int(dbutils.widgets.get("meeting_limit"))
if meeting_limit <= 0:
    raise ValueError("meeting_limit must be greater than zero")

cfg = load_config("../../project_config.yml", env=env)
spark = SparkSession.builder.getOrCreate()
ingestion_batch_id = str(uuid4())
metadata_table = cfg.get_full_table_name(cfg.metadata_table)
last_meeting_id = get_last_ingested_meeting_id(spark, metadata_table)

# COMMAND ----------

with OpenBesluitvormingClient() as openbesluitvorming_client:
    cursor = last_meeting_id or DEFAULT_SNAPSHOT_CURSOR
    ingested_meetings = 0
    meeting_documents = []
    while ingested_meetings < meeting_limit:
        remaining = meeting_limit - ingested_meetings
        meeting_ids = openbesluitvorming_client.get_next_meeting_ids(remaining, cursor=cursor)
        if not meeting_ids:
            break
        for meeting_id in meeting_ids:
            documents = openbesluitvorming_client.get_documents_with_context_for_meeting(
                meeting_id, ignore_missing_documents=True
            )
            if not documents:
                print(f"No documents for {meeting_id}; checking the next meeting.")
                continue
            meeting_documents.extend(documents)
            ingested_meetings += 1
        cursor = meeting_ids[-1]
        if len(meeting_ids) < remaining:
            break

    metadata = ingest_meeting_documents(
        spark=spark,
        meeting_documents=meeting_documents,
        metadata_table=metadata_table,
        ingestion_batch_id=ingestion_batch_id,
    )

dbutils.jobs.taskValues.set(key="ingestion_batch_id", value=ingestion_batch_id)
dbutils.jobs.taskValues.set(key="ingested_count", value=len(metadata))
print(f"Ingestion batch: {ingestion_batch_id}")
print(f"Ingested {len(metadata)} documents from {ingested_meetings} meetings into {metadata_table}.")
