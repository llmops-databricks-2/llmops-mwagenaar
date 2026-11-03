"""Synchronize embedded document chunks to a Lakebase Synced Table."""

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.pipelines import UpdateInfoState
from databricks.sdk.service.postgres import (
    NewPipelineSpec,
    SyncedTable,
    SyncedTableState,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecPgSpecificType,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    SyncedTableSyncedTableSpecTypeOverride,
)
from loguru import logger

from polderbot.config import ProjectConfig
from polderbot.vector_search import LakebasePgVectorManager, get_lakebase_branch

_EMBEDDING_DIMENSIONS = 1024
_DATABASE = "databricks_postgres"
_POLL_INTERVAL_SECONDS = 30
_SYNC_TIMEOUT_SECONDS = 1800
_ONLINE_STATES = {
    SyncedTableState.SYNCED_TABLE_ONLINE,
    SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
}
_FAILED_STATES = {
    SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED,
    SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED,
}


def get_document_chunks_synced_table_name(config: ProjectConfig, env: str) -> str:
    """Get the unqualified synced table name for an environment.

    Args:
        config: Project settings containing the chunks table name.
        env: Bundle target used to distinguish synced tables.

    Returns:
        The Lakebase synced table name.

    """
    return f"{config.chunks_table}_synced_{env}"


def sync_document_chunks(config: ProjectConfig, env: str, workspace: WorkspaceClient) -> None:
    """Create or refresh the environment's Lakebase copy of embedded chunks.

    Args:
        config: Project settings for the source and Lakebase project.
        env: Bundle target, used to distinguish tables in a shared project.
        workspace: Authenticated Databricks workspace client.

    Raises:
        ValueError: If the usage policy or required Lakebase endpoint details are missing.
        RuntimeError: If the synchronization fails.
        TimeoutError: If the synchronization does not finish in time.

    """
    if not config.usage_policy_id:
        raise ValueError("A usage_policy_id is required to create the sync pipeline")

    source_table = config.get_full_table_name(config.chunks_table)
    synced_table_name = get_document_chunks_synced_table_name(config, env)
    synced_table = config.get_full_table_name(synced_table_name)
    resource_name = f"synced_tables/{synced_table}"
    branch = get_lakebase_branch(config.lakebase_project_id)
    pgvector = LakebasePgVectorManager(
        config=config, workspace=workspace, synced_table_name=synced_table_name, title_column="document_title"
    )

    try:
        table = workspace.postgres.get_synced_table(name=resource_name)
    except NotFound:
        pgvector.ensure_extension()
        workspace.postgres.create_synced_table(
            synced_table=SyncedTable(
                spec=SyncedTableSyncedTableSpec(
                    source_table_full_name=source_table,
                    branch=branch,
                    postgres_database=_DATABASE,
                    primary_key_columns=["id"],
                    scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
                    create_database_objects_if_missing=True,
                    new_pipeline_spec=NewPipelineSpec(
                        budget_policy_id=config.usage_policy_id,
                        storage_catalog=config.catalog,
                        storage_schema=config.schema,
                    ),
                    type_overrides=[
                        SyncedTableSyncedTableSpecTypeOverride(
                            column_name="embedding",
                            pg_type=SyncedTableSyncedTableSpecPgSpecificType.PG_SPECIFIC_TYPE_VECTOR,
                            size=_EMBEDDING_DIMENSIONS,
                        )
                    ],
                )
            ),
            synced_table_id=synced_table,
        ).wait()
        logger.info(f"Created Synced Table: {synced_table}")
        _wait_for_sync(workspace, resource_name)
        pgvector.ensure_indexes()
        return

    if not table.status or not table.status.pipeline_id:
        raise ValueError(f"Synced Table {synced_table} has no pipeline ID")
    pipeline_id = table.status.pipeline_id
    update = workspace.pipelines.start_update(pipeline_id=pipeline_id)
    if not update.update_id:
        raise ValueError(f"Starting pipeline {pipeline_id} returned no update ID")
    logger.info(f"Started sync for {synced_table}: update {update.update_id}")
    _wait_for_sync(workspace, resource_name, update.update_id)
    pgvector.ensure_indexes()


def _wait_for_sync(
    workspace: WorkspaceClient,
    resource_name: str,
    update_id: str | None = None,
) -> None:
    """Wait for the initial sync or the specific pipeline update to finish.

    Args:
        workspace: Authenticated Databricks workspace client.
        resource_name: Synced Table resource name.
        update_id: Identifier of the update to wait for, if any.

    Raises:
        RuntimeError: If the Synced Table or update fails.
        TimeoutError: If the synchronization does not finish in time.

    """
    deadline = time.monotonic() + _SYNC_TIMEOUT_SECONDS
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Synced Table {resource_name}")
        status = workspace.postgres.get_synced_table(name=resource_name).status
        if not status:
            raise RuntimeError(f"Synced Table {resource_name} returned no status")
        if status.detailed_state in _FAILED_STATES:
            raise RuntimeError(f"Synced Table failed: {status.detailed_state} | {status.message}")
        if update_id:
            if not status.pipeline_id:
                raise RuntimeError(f"Synced Table {resource_name} returned no pipeline ID")
            update = workspace.pipelines.get_update(pipeline_id=status.pipeline_id, update_id=update_id).update
            if not update:
                raise RuntimeError(f"Pipeline update {update_id} returned no status")
            if update.state == UpdateInfoState.COMPLETED:
                logger.info(f"Sync completed: {update_id}")
                return
            if update.state in {UpdateInfoState.FAILED, UpdateInfoState.CANCELED}:
                raise RuntimeError(f"Sync {update_id} ended in state {update.state}")
        elif status.detailed_state in _ONLINE_STATES:
            logger.info(f"Synced Table is online: {resource_name}")
            return
        logger.info(f"Waiting for sync: {status.detailed_state}")
        time.sleep(_POLL_INTERVAL_SECONDS)
