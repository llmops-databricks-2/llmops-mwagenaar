"""Persist session messages in Lakebase PostgreSQL."""

import json
import os
import urllib.parse
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import PostgresAPI
from psycopg_pool import ConnectionPool


class LakebaseMemory:
    """Persist session messages in Lakebase PostgreSQL."""

    def __init__(
        self,
        project_id: str,
        workspace: WorkspaceClient | None = None,
    ) -> None:
        """Initialize Lakebase configuration without opening a connection.

        Args:
            project_id: Lakebase project ID used for endpoint discovery.
            workspace: Optional workspace client for Lakebase API calls.

        """
        self.project_id = project_id
        self.workspace = workspace
        self._pool: ConnectionPool | None = None

    def ensure_schema(self) -> None:
        """Create the session table and index if they do not exist."""
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
                ON session_messages(session_id)
                """
            )

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Load a session's messages in ascending creation-time order.

        Args:
            session_id: Identifier of the session to load.

        Returns:
            The stored messages, or an empty list if no messages exist.

        Raises:
            psycopg.OperationalError: If a database operational error occurs;
                the connection pool is reset before the error is re-raised.

        """
        try:
            with self._get_pool().connection() as conn:
                result = conn.execute(
                    """
                    SELECT message_data FROM session_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (session_id,),
                ).fetchall()
                return [row[0] for row in result]
        except psycopg.OperationalError:
            self._reset_pool()
            raise

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append messages to a session.

        Args:
            session_id: Identifier of the session to update.
            messages: JSON-serializable messages to append.

        Raises:
            psycopg.OperationalError: If a database operational error occurs;
                the connection pool is reset before the error is re-raised.

        """
        try:
            with self._get_pool().connection() as conn:
                for msg in messages:
                    conn.execute(
                        "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
                        (session_id, json.dumps(msg)),
                    )
        except psycopg.OperationalError:
            self._reset_pool()
            raise

    def _get_pool(self) -> ConnectionPool:
        """Get the existing connection pool or create it on first use.

        Returns:
            The connection pool for this memory instance.

        """
        if self._pool is None:
            conn_string = self._get_connection_string()
            self._pool = ConnectionPool(conninfo=conn_string, min_size=1, max_size=5)
        return self._pool

    def _reset_pool(self) -> None:
        """Close and clear the pool so the next use obtains new credentials."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _get_connection_string(self) -> str:
        """Build a Lakebase connection string.

        Returns:
            A PostgreSQL connection string with a generated database credential.

        """
        client_id = os.environ.get("LAKEBASE_SP_CLIENT_ID") or os.environ.get("DATABRICKS_CLIENT_ID")
        workspace = self._get_workspace_client(client_id)
        pg_api = PostgresAPI(workspace.api_client)
        username = self._get_username(workspace, client_id)
        endpoint_name, host, database = self._get_endpoint_details(pg_api)
        pg_credential = pg_api.generate_database_credential(endpoint=endpoint_name)  # type: ignore

        return f"postgresql://{username}:{pg_credential.token}@{host}:5432/{database}?sslmode=require"

    def _get_workspace_client(self, client_id: str | None) -> WorkspaceClient:
        """Select dedicated service-principal or default workspace authentication.

        Args:
            client_id: The configured service-principal client ID, if any.

        Returns:
            The workspace client used for Lakebase API calls.

        """
        client_secret = os.environ.get("LAKEBASE_SP_CLIENT_SECRET") or os.environ.get("DATABRICKS_CLIENT_SECRET")
        host = os.environ.get("LAKEBASE_SP_HOST") or os.environ.get("DATABRICKS_HOST")

        if client_id and client_secret and host:
            return WorkspaceClient(host=host, client_id=client_id, client_secret=client_secret)

        return self.workspace or WorkspaceClient()

    def _get_username(self, workspace: WorkspaceClient, client_id: str | None) -> str:
        """Select the database username for the configured authentication.

        Args:
            workspace: The workspace client used to identify a local user.
            client_id: The configured service-principal client ID, if any.

        Returns:
            The client ID or URL-encoded workspace username.

        """
        if client_id:
            return client_id

        user = workspace.current_user.me()

        return urllib.parse.quote_plus(user.user_name)  # type: ignore

    def _get_endpoint_details(self, pg_api: PostgresAPI) -> tuple[str | None, str | None, str]:
        """Resolve the configured app endpoint or discover the project endpoint.

        Args:
            pg_api: The Lakebase PostgreSQL API client.

        Returns:
            The endpoint name, host, and database name. The name or host may
            be absent in the API response or environment.

        Raises:
            StopIteration: If endpoint discovery finds no branch or endpoint.

        """
        endpoint_name = os.environ.get("LAKEBASE_ENDPOINT")
        if endpoint_name:
            return (
                endpoint_name,
                os.environ.get("PGHOST"),
                os.environ.get("PGDATABASE", "databricks_postgres"),
            )
        project_parent = f"projects/{self.project_id}"
        default_branch = next(iter(pg_api.list_branches(parent=project_parent)))
        endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))  # type: ignore

        return endpoint.name, endpoint.status.hosts.host, "databricks_postgres"  # type: ignore
