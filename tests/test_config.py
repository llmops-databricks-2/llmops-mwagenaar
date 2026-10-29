"""Your first test: project_config.yml loads with the course's ProjectConfig for dev.

`ci.yml` runs every test under tests/ on each pull request and each push to main; add yours here.
"""

from pathlib import Path

from polderbot.config import ProjectConfig

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def test_project_config_loads_for_dev() -> None:
    """Load the provisioned settings without requiring optional lecture resources."""
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")

    assert cfg.full_volume_path == "llmops_dev.mwagenaar.arxiv_files"
    assert cfg.warehouse_id == "d0a9aaf5c9434b89"
    assert cfg.metadata_table == "documents_metadata"
    assert cfg.chunks_table == "document_chunks"
    assert cfg.get_full_table_name(cfg.metadata_table) == "llmops_dev.mwagenaar.documents_metadata"
