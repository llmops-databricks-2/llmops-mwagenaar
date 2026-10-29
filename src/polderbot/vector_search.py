"""Lakebase pgvector search over the arXiv chunks Synced Table."""

from __future__ import annotations

import os
import urllib.parse
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import EndpointType
from loguru import logger
from psycopg import sql

from polderbot.config import ProjectConfig

_BRANCH_NAME = "production"
_ARXIV_RESULT_COLUMNS = ("id", "arxiv_id", "title", "summary", "authors", "year", "month", "day", "text")


def get_lakebase_branch(project_id: str) -> str:
    """Get the production branch resource name for a Lakebase project.

    Args:
        project_id: Lakebase project identifier.

    Returns:
        Fully qualified Lakebase branch resource name.

    """
    return f"projects/{project_id}/branches/{_BRANCH_NAME}"


class LakebasePgVectorManager:
    """Search chunk embeddings in a Lakebase Synced Table with pgvector.

    The Synced Table is a read-only copy of arxiv_chunks_table, created with a
    type override so that `embedding` is a pgvector vector column. Full-text
    search uses an expression index, built on the same expression the queries use.
    Connections use a writable endpoint on the project's production branch.
    """

    synced_table_name = "arxiv_chunks_synced"

    def __init__(
        self,
        config: ProjectConfig | None = None,
        table_name: str | None = None,
        *,
        catalog: str | None = None,
        schema: str | None = None,
        lakebase_project_id: str | None = None,
        embedding_model: str | None = None,
        workspace: WorkspaceClient | None = None,
        synced_table_name: str | None = None,
        title_column: str = "title",
        result_columns: tuple[str, ...] = _ARXIV_RESULT_COLUMNS,
    ) -> None:
        self.config = config
        self.catalog = config.catalog if config else catalog
        self.schema = config.schema if config else schema
        self.lakebase_project_id = config.lakebase_project_id if config else lakebase_project_id
        self.embedding_model = embedding_model or (config.embedding_endpoint if config else None)
        if not self.catalog or not self.schema or not self.lakebase_project_id:
            raise ValueError("catalog, schema, and lakebase_project_id are required")
        if not self.embedding_model:
            raise ValueError("embedding_model is required")
        # Synced Tables land in a Postgres schema named after the Unity Catalog schema
        self.synced_table_name = synced_table_name or self.synced_table_name
        self.table_name = table_name or sql.Identifier(self.schema, self.synced_table_name).as_string()
        self._connection_string: str | None = None
        self.workspace = workspace
        self.result_columns = sql.SQL(", ").join(map(sql.Identifier, result_columns)).as_string()

        # Shared by the GIN index and the queries, so Postgres can use the index
        self.text_expr = (
            f"to_tsvector('english', coalesce({sql.Identifier(title_column).as_string()}, '') || ' ' || text)"
        )

    def _get_connection_string(self) -> str:
        """Create a short-lived Lakebase connection string."""
        if self._connection_string:
            return self._connection_string

        workspace = self.workspace or WorkspaceClient()
        postgres_api = workspace.postgres
        branch = get_lakebase_branch(self.lakebase_project_id)
        endpoint = next(
            (
                item
                for item in postgres_api.list_endpoints(parent=branch)
                if item.status and item.status.endpoint_type == EndpointType.ENDPOINT_TYPE_READ_WRITE
            ),
            None,
        )
        if not endpoint or not endpoint.name or not endpoint.status or not endpoint.status.hosts:
            raise ValueError(f"No writable Lakebase endpoint with a host found on {branch}")
        host = endpoint.status.hosts.host
        credential = postgres_api.generate_database_credential(endpoint=endpoint.name)
        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        username = client_id or workspace.current_user.me().user_name
        if not host or not credential.token or not username:
            raise ValueError("Missing Lakebase host, database credential, or user")

        user = urllib.parse.quote(username, safe="")
        token = urllib.parse.quote(credential.token, safe="")
        self._connection_string = f"postgresql://{user}:{token}@{host}:5432/databricks_postgres?sslmode=require"
        return self._connection_string

    def connection(self) -> psycopg.Connection:
        """Open a Lakebase connection using a fresh workspace credential."""
        self._connection_string = None
        return psycopg.connect(self._get_connection_string())

    def ensure_extension(self) -> None:
        """Enable lakebase_vector (installs pgvector with CASCADE).

        Must run before the Synced Table with a vector column is created: creating a
        synced table does not install extensions.
        """
        with self.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE")
        logger.info("lakebase_vector extension ready")

    def ensure_indexes(self) -> None:
        """Create a full-text index on the Synced Table.

        Synced Tables allow CREATE INDEX (but no writes). Idempotent: rerun it if
        the indexes are missing, for example after the Synced Table was recreated.
        """
        with self.connection() as conn:
            # Disabled: the lakebase_ann index (notebook 2.5, section 9) is the only
            # vector index on embedding, so ai_search sees a single index type.
            # conn.execute(
            #     f"""
            #     CREATE INDEX IF NOT EXISTS {self.synced_table_name}_embedding_hnsw
            #     ON {self.table_name}
            #     USING hnsw (embedding vector_cosine_ops)
            #     """
            # )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {sql.Identifier(f"{self.synced_table_name}_search_gin").as_string()}
                ON {self.table_name} USING gin (({self.text_expr}))
                """
            )
        logger.info(f"pgvector indexes ready on {self.table_name}")

    @staticmethod
    def _vector_literal(vector: list[float] | tuple[float, ...]) -> str:
        """Format a Python vector for PostgreSQL's vector input syntax."""
        return "[" + ",".join(str(float(value)) for value in vector) + "]"

    def embed_query(self, query: str) -> list[float]:
        """Generate a query embedding with the same model used for documents."""
        workspace = self.workspace or WorkspaceClient()
        client = workspace.serving_endpoints.get_open_ai_client()
        response = client.embeddings.create(model=self.embedding_model, input=query)
        if not response.data or not response.data[0].embedding:
            raise ValueError("The embedding endpoint returned no vector")
        return list(response.data[0].embedding)

    def _fetch(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                columns = [description.name for description in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def search(
        self,
        query: str,
        num_results: int = 5,
        year: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run cosine-similarity search with an optional year filter."""
        query_vector = self._vector_literal(self.embed_query(query))
        where = "WHERE embedding IS NOT NULL"
        params: list[Any] = [query_vector]
        if year is not None:
            where += " AND year = %s"
            params.append(year)
        params.extend([query_vector, num_results])
        sql = f"""
            SELECT {self.result_columns},
                   1 - (embedding <=> %s::vector) AS score
            FROM {self.table_name}
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        return self._fetch(sql, params)

    def hybrid_search(
        self,
        query: str,
        num_results: int = 5,
        year: int | None = None,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Combine cosine similarity and PostgreSQL full-text ranking."""
        query_vector = self._vector_literal(self.embed_query(query))
        text_weight = 1.0 - vector_weight
        where = "WHERE embedding IS NOT NULL"
        params: list[Any] = [query_vector, query]
        if year is not None:
            where += " AND year = %s"
            params.append(year)
        params.extend([vector_weight, text_weight, num_results])
        sql = f"""
            WITH ranked AS (
                SELECT {self.result_columns},
                       1 - (embedding <=> %s::vector) AS vector_score,
                       ts_rank_cd({self.text_expr}, plainto_tsquery('english', %s))
                           AS text_score
                FROM {self.table_name}
                {where}
            )
            SELECT *, (%s * vector_score + %s * text_score) AS score
            FROM ranked
            ORDER BY score DESC
            LIMIT %s
        """
        return self._fetch(sql, params)
