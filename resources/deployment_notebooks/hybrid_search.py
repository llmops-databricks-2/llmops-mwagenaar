# Databricks notebook source
"""Run an agent conversation using document search, Genie, and Lakebase memory."""

# COMMAND ----------

import asyncio
import os
from uuid import uuid4

import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from loguru import logger

from polderbot.agent import PolderAgent
from polderbot.config import load_config
from polderbot.memory import LakebaseMemory
from polderbot.synced_tables import get_document_chunks_synced_table_name
from polderbot.tools import create_mcp_tools, create_search_documents_tool
from polderbot.vector_search import LakebasePgVectorManager

nest_asyncio.apply()

# COMMAND ----------

_SYSTEM_PROMPT = (
    "You are an assistant for meetings and documents from Wetterskip Fryslân. "
    "Use the available tools to answer questions and cite your sources."
)

dbutils.widgets.text("env", "dev")
dbutils.widgets.text("genie_space_id", "")
env = dbutils.widgets.get("env")
genie_space_id = dbutils.widgets.get("genie_space_id").strip()

# COMMAND ----------

cfg = load_config("../../project_config.yml", env=env)
workspace = WorkspaceClient()
pgvector = LakebasePgVectorManager(
    config=cfg,
    workspace=workspace,
    synced_table_name=get_document_chunks_synced_table_name(cfg, env),
)

# COMMAND ----------
genie_url = f"{workspace.config.host.rstrip('/')}/api/2.0/mcp/genie/{genie_space_id}"
genie_tools = asyncio.run(create_mcp_tools(workspace, [genie_url]))

if not genie_tools:
    raise RuntimeError("The Genie MCP server did not provide any tools.")

all_tools = [create_search_documents_tool(pgvector), *genie_tools]
client_id = dbutils.secrets.get(catalog=cfg.catalog, schema=cfg.schema, key="databricks_client_id")  # type: ignore
client_secret = dbutils.secrets.get(catalog=cfg.catalog, schema=cfg.schema, key="databricks_client_secret")  # type: ignore
os.environ["LAKEBASE_SP_CLIENT_ID"] = client_id
os.environ["LAKEBASE_SP_CLIENT_SECRET"] = client_secret
os.environ["LAKEBASE_SP_HOST"] = workspace.config.host
memory = LakebaseMemory(project_id=cfg.lakebase_project_id, workspace=workspace)
memory.ensure_schema()
agent = PolderAgent(
    workspace=workspace,
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=_SYSTEM_PROMPT,
    tools=all_tools,
    memory=memory,
)

# COMMAND ----------
session_id = str(uuid4())
response = agent.generate_message(
    "What is the current status of Elahuizen according to the meeting documents?", session_id=session_id
)
logger.info(response)

# COMMAND ----------

follow_up_response = agent.generate_message(
    "What was my previous question?",
    session_id=session_id,
)
logger.info(follow_up_response)

# COMMAND ----------

genie_tools_response = agent.generate_message(
    "How many documents are available in total? Query the genie space for this information.",
    session_id=session_id,
)
logger.info(genie_tools_response)

# COMMAND ----------

new_session_id = str(uuid4())
new_conversation_response = agent.generate_message(
    "What was my previous question?",
    session_id=new_session_id,
)
logger.info(new_conversation_response)
