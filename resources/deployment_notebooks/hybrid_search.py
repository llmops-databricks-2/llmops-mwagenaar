# Databricks notebook source
"""Run one hybrid search over the synchronized document chunks."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from loguru import logger

from polderbot.config import load_config
from polderbot.synced_tables import get_document_chunks_synced_table_name
from polderbot.vector_search import LakebasePgVectorManager

_QUERY = "acties algemeen bestuur"
_NUM_RESULTS = 5
_TEXT_PREVIEW_LENGTH = 150
_RESULT_COLUMNS = ("id", "document_id", "document_title", "document_name", "meeting_title", "text")

dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")

cfg = load_config("../../project_config.yml", env=env)
pgvector = LakebasePgVectorManager(
    config=cfg,
    workspace=WorkspaceClient(),
    synced_table_name=get_document_chunks_synced_table_name(cfg, env),
    title_column="document_title",
    result_columns=_RESULT_COLUMNS,
)
hybrid_results = pgvector.hybrid_search(query=_QUERY, num_results=_NUM_RESULTS)

logger.info(f"Hybrid query: {_QUERY}")
for position, row in enumerate(hybrid_results, 1):
    logger.info(
        f"{position}. {row.get('document_title') or row['document_name']} | "
        f"score={float(row['score']):.4f} | "
        f"vector={float(row['vector_score']):.4f} | "
        f"text={float(row['text_score']):.4f}"
    )
    logger.info(f"   {row['text'][:_TEXT_PREVIEW_LENGTH]}")
