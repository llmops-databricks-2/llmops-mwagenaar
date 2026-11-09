"""Agent tools for document search and managed MCP servers."""

import json
from collections.abc import Callable
from datetime import datetime
from functools import partial

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from pydantic import AwareDatetime, BaseModel, Field

from polderbot.vector_search import LakebasePgVectorManager


class ToolInfo(BaseModel):
    """Tool information for agent integration.

    Attributes:
        name: Tool name.
        spec: JSON description in OpenAI Chat Completions format.
        exec_fn: Function that implements the tool logic.

    """

    name: str
    spec: dict
    exec_fn: Callable

    class Config:  # noqa: D106
        arbitrary_types_allowed = True


class SearchDocumentsArgs(BaseModel):
    """Arguments supplied by the model for document search."""

    query: str = Field(min_length=1, description="Text to search for in meeting documents.")
    num_results: int = Field(default=5, gt=0, description="Maximum number of matching chunks.")
    document_date_from: AwareDatetime | None = Field(
        default=None,
        description="Inclusive lower bound as an ISO 8601 datetime with timezone.",
    )
    document_date_to: AwareDatetime | None = Field(
        default=None,
        description="Inclusive upper bound as an ISO 8601 datetime with timezone.",
    )


def search_documents(
    query: str,
    num_results: int = 5,
    document_date_from: datetime | None = None,
    document_date_to: datetime | None = None,
    *,
    pgvector: LakebasePgVectorManager,
) -> str:
    """Search document chunks and return their metadata and scores as JSON.

    Args:
        query: Text to search for.
        num_results: Maximum number of matching chunks.
        document_date_from: Inclusive timezone-aware lower bound.
        document_date_to: Inclusive timezone-aware upper bound.
        pgvector: Manager configured for the document chunks synced table.

    Returns:
        JSON array containing the matching chunks and their scores.

    Raises:
        ValueError: If arguments are invalid.

    """
    args = SearchDocumentsArgs.model_validate(
        {
            "query": query,
            "num_results": num_results,
            "document_date_from": document_date_from,
            "document_date_to": document_date_to,
        }
    )
    results = pgvector.search_documents_hybrid(**args.model_dump())
    return json.dumps(results, ensure_ascii=False)


def create_search_documents_tool(pgvector: LakebasePgVectorManager) -> ToolInfo:
    """Create an OpenAI Chat Completions tool backed by document search.

    Args:
        pgvector: Manager configured for the document chunks synced table.

    Returns:
        Tool specification and callable for an agent tool registry.

    """
    return ToolInfo(
        name="search_documents",
        spec={
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search meeting documents using hybrid vector and keyword search.",
                "parameters": SearchDocumentsArgs.model_json_schema(),
            },
        },
        exec_fn=partial(search_documents, pgvector=pgvector),
    )


def create_managed_exec_fn(server_url: str, tool_name: str, w: WorkspaceClient) -> Callable[..., str]:
    """Create an execution function for an MCP tool.

    Args:
        server_url: MCP server URL.
        tool_name: Name of the tool.
        w: Databricks workspace client.

    Returns:
        Callable that executes the tool.

    """

    def exec_fn(**kwargs: dict) -> str:
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)

        return "".join([c.text for c in response.content])

    return exec_fn


async def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    """Create tools from MCP servers.

    Args:
        w: Databricks workspace client.
        url_list: List of MCP server URLs.

    Returns:
        List of ToolInfo objects.

    """
    tools = []

    for server_url in url_list:
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        mcp_tools = mcp_client.list_tools()

        for mcp_tool in mcp_tools:
            input_schema = mcp_tool.inputSchema.copy() if mcp_tool.inputSchema else {}
            tool_spec = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "parameters": input_schema,
                    "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
                },
            }
            exec_fn = create_managed_exec_fn(server_url, mcp_tool.name, w)
            tools.append(ToolInfo(name=mcp_tool.name, spec=tool_spec, exec_fn=exec_fn))

    return tools
