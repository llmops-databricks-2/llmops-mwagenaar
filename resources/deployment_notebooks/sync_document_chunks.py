# Databricks notebook source
"""Synchronize embedded document chunks to Lakebase."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from pyspark.sql import SparkSession

from polderbot.config import load_config
from polderbot.synced_tables import sync_document_chunks

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("ingested_count", "")
env = dbutils.widgets.get("env")
ingested_count = dbutils.widgets.get("ingested_count")

cfg = load_config("../../project_config.yml", env=env)
spark = SparkSession.builder.getOrCreate()
source_table = cfg.get_full_table_name(cfg.chunks_table)

if ingested_count == "0" and not spark.catalog.tableExists(source_table):
    print(f"No chunks to synchronize in {source_table}.")
else:
    sync_document_chunks(config=cfg, env=env, workspace=WorkspaceClient())
