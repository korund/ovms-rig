"""Tests for remove stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ovms_rig.stages import remove


@pytest.fixture
def minimal_ovms_config(tmp_path: Path) -> Path:
    """Create minimal ovms.yaml for testing."""
    config_path = tmp_path / "ovms.yaml"
    config_path.write_text(
        """\
runtime:
  rest_port: 8080
  log_level: INFO
repository:
  qwen3-14b:
    hf: Qwen/Qwen3-14B-int8-ov
    task: text_generation
  qwen3-draft:
    hf: Qwen/Qwen3-0.6B-int8-ov
    task: text_generation
models:
  prod-main:
    source: qwen3-14b
    graph:
      device: GPU
      draft_model: null
      draft_device: null
  prod-draft:
    source: qwen3-draft
    graph:
      device: CPU
      draft_model: null
      draft_device: null
profiles:
  prod:
    active: false
    models:
      - prod-main
""",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def minimal_local_config(tmp_path: Path) -> Path:
    """Create minimal local.yaml for testing."""
    config_path = tmp_path / "local.yaml"
    models_path = tmp_path / "models"
    models_path.mkdir(exist_ok=True)
    config_path.write_text(
        f"models:\n  repository_path: {models_path}\n",
        encoding="utf-8",
    )
    return config_path


def test_remove_unknown_repository(minimal_ovms_config: Path, minimal_local_config: Path) -> None:
    """Test removing an unknown repository returns error."""
    ctx = {
        "config_path": str(minimal_ovms_config),
        "local_path": str(minimal_local_config),
        "ovms_path": None,
        "log_level": None,
        "repository_name": "nonexistent",
        "force": False,
    }
    assert remove.run(ctx) == 1


def test_remove_not_fetched(minimal_ovms_config: Path, minimal_local_config: Path) -> None:
    """Test removing a repository that was never fetched returns 0 (nothing to do)."""
    ctx = {
        "config_path": str(minimal_ovms_config),
        "local_path": str(minimal_local_config),
        "ovms_path": None,
        "log_level": None,
        "repository_name": "qwen3-14b",
        "force": False,
    }
    assert remove.run(ctx) == 0


def test_remove_happy_path(minimal_ovms_config: Path, minimal_local_config: Path) -> None:
    """Test successful removal of a fetched repository with no references."""
    models_path = Path(minimal_local_config).parent / "models"
    models_path.mkdir(exist_ok=True)

    # Create the directory structure that fetch would create for qwen3-draft.
    model_dir = models_path / "Qwen" / "Qwen3-0.6B-int8-ov"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create a test file in the model directory.
    test_file = model_dir / "model.safetensors"
    test_file.write_text("dummy model data", encoding="utf-8")

    # Create config.json with a mediapipe entry.
    config_json_path = models_path / "config.json"
    config_data = {
        "mediapipe_config_list": [
            {
                "name": "qwen3-draft",
                "base_path": str(model_dir),
                "graph_path": "graph.qwen3-draft.pbtxt",
            }
        ]
    }
    config_json_path.write_text(json.dumps(config_data), encoding="utf-8")

    ctx = {
        "config_path": str(minimal_ovms_config),
        "local_path": str(minimal_local_config),
        "ovms_path": None,
        "log_level": None,
        "repository_name": "qwen3-draft",
        "force": False,
    }

    # Perform the removal.
    assert remove.run(ctx) == 0

    # Verify directory is gone.
    assert not model_dir.exists()

    # Verify entry is removed from config.json.
    if config_json_path.exists():
        data = json.loads(config_json_path.read_text(encoding="utf-8"))
        entries = data.get("mediapipe_config_list", [])
        assert not any(e.get("name") == "qwen3-draft" for e in entries)


def test_remove_blocked_by_reference(
    minimal_ovms_config: Path, minimal_local_config: Path
) -> None:
    """Test that removal is blocked when repository is referenced by a profile."""
    models_path = Path(minimal_local_config).parent / "models"
    models_path.mkdir(exist_ok=True)

    # Create the directory structure.
    model_dir = models_path / "Qwen" / "Qwen3-14B-int8-ov"
    model_dir.mkdir(parents=True, exist_ok=True)
    test_file = model_dir / "model.safetensors"
    test_file.write_text("dummy model data", encoding="utf-8")

    ctx = {
        "config_path": str(minimal_ovms_config),
        "local_path": str(minimal_local_config),
        "ovms_path": None,
        "log_level": None,
        "repository_name": "qwen3-14b",
        "force": False,
    }

    # Should be blocked because prod profile has prod-main which sources qwen3-14b.
    assert remove.run(ctx) == 1

    # Directory should still exist.
    assert model_dir.exists()


def test_remove_blocked_with_force_override(
    minimal_ovms_config: Path, minimal_local_config: Path
) -> None:
    """Test that --force overrides reference blocking."""
    models_path = Path(minimal_local_config).parent / "models"
    models_path.mkdir(exist_ok=True)

    # Create the directory structure.
    model_dir = models_path / "Qwen" / "Qwen3-14B-int8-ov"
    model_dir.mkdir(parents=True, exist_ok=True)
    test_file = model_dir / "model.safetensors"
    test_file.write_text("dummy model data", encoding="utf-8")

    ctx = {
        "config_path": str(minimal_ovms_config),
        "local_path": str(minimal_local_config),
        "ovms_path": None,
        "log_level": None,
        "repository_name": "qwen3-14b",
        "force": True,  # Override the block.
    }

    # Should succeed with force.
    assert remove.run(ctx) == 0

    # Directory should be gone.
    assert not model_dir.exists()
