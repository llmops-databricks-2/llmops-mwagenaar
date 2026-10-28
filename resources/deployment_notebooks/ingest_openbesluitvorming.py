# Databricks notebook source
"""Ingest one OpenBesluitvorming meeting's Markdown into Delta."""

# COMMAND ----------

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

# COMMAND ----------

with OpenBesluitvormingClient() as openbesluitvorming_client:
    meeting_documents = openbesluitvorming_client.get_documents_with_context_for_meeting(
        MEETING_ID, ignore_missing_documents=True
    )
    metadata = ingest_meeting_documents(
        spark=spark,
        meeting_documents=meeting_documents[:3],
        metadata_table=cfg.get_full_table_name(cfg.metadata_table),
        chunks_table=cfg.get_full_table_name(cfg.chunks_table),
    )

print(f"Ingested {len(metadata)} documents from {MEETING_ID} into {cfg.get_full_table_name(cfg.metadata_table)}.")
