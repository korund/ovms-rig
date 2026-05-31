"""Tests for cleanup.py: removal of obsolete sibling-graph files.

Regression tests include the crash bug: cleanup must not crash when
repository contains dir-source models (hf field is None).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ovms_rig.config.schema import OvmsConfig, Profile
from ovms_rig.stages.activation.cleanup import cleanup_obsolete_sibling_graphs


@pytest.fixture
def base_config() -> dict:
    """Base config dict for constructing OvmsConfig instances."""
    return {
        "runtime": {"rest_port": 8000},
        "repository": {},
        "models": {},
        "profiles": {},
    }


def test_cleanup_with_hf_source(tmp_path: Path) -> None:
    """Cleanup works normally with hf-source models."""
    store = tmp_path / "store"
    store.mkdir()

    # Create model dir with a pristine graph and obsolete sibling.
    model_dir = store / "OpenVINO" / "Qwen3-14B"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text("pristine", encoding="utf-8")
    (model_dir / "graph.old_model.pbtxt").write_text("obsolete", encoding="utf-8")

    config_data = {
        "runtime": {"rest_port": 8000},
        "repository": {
            "qwen": {"hf": "OpenVINO/Qwen3-14B", "task": "text_generation"}
        },
        "models": {},
        "profiles": {},
    }
    ovms = OvmsConfig.model_validate(config_data)

    # Cleanup with only "qwen" in active_models should remove old_model sibling.
    cleaned = cleanup_obsolete_sibling_graphs(store, {"qwen"}, ovms)
    assert len(cleaned) == 1
    assert "graph.old_model.pbtxt" in cleaned[0]
    assert not (model_dir / "graph.old_model.pbtxt").exists()


def test_cleanup_with_dir_source_plain_model(tmp_path: Path) -> None:
    """Cleanup does not crash with dir-source plain (no task) models.

    Plain models have no graph.pbtxt or sibling graphs, so nothing is swept.
    This is a regression test for the crash bug: hf is None for dir models.
    """
    store = tmp_path / "store"
    store.mkdir()

    # Create a dir-source plain model directory.
    model_dir = store / "local" / "plain_model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_text("weights", encoding="utf-8")

    config_data = {
        "runtime": {"rest_port": 8000},
        "repository": {
            "plain": {"dir": "local/plain_model"}
        },
        "models": {},
        "profiles": {},
    }
    ovms = OvmsConfig.model_validate(config_data)

    # Cleanup should not crash even though dir-source has hf=None.
    cleaned = cleanup_obsolete_sibling_graphs(store, set(), ovms)
    assert len(cleaned) == 0  # No sibling graphs to clean.


def test_cleanup_with_dir_source_task_model(tmp_path: Path) -> None:
    """Cleanup works with dir-source task models that have sibling graphs.

    Task-based models (even dir-sourced) can have sibling graphs.
    Cleanup should find and remove obsolete ones.
    """
    store = tmp_path / "store"
    store.mkdir()

    # Create a dir-source task model directory with pristine and sibling graphs.
    model_dir = store / "local" / "task_model"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text("pristine", encoding="utf-8")
    (model_dir / "graph.active_model.pbtxt").write_text("active", encoding="utf-8")
    (model_dir / "graph.obsolete_model.pbtxt").write_text("obsolete", encoding="utf-8")

    config_data = {
        "runtime": {"rest_port": 8000},
        "repository": {
            "task_model": {"dir": "local/task_model", "task": "text_generation"}
        },
        "models": {},
        "profiles": {},
    }
    ovms = OvmsConfig.model_validate(config_data)

    # Cleanup with only "active_model" should remove "obsolete_model" sibling.
    cleaned = cleanup_obsolete_sibling_graphs(store, {"active_model"}, ovms)
    assert len(cleaned) == 1
    assert "graph.obsolete_model.pbtxt" in cleaned[0]
    assert not (model_dir / "graph.obsolete_model.pbtxt").exists()
    assert (model_dir / "graph.active_model.pbtxt").exists()


def test_cleanup_mixed_hf_and_dir_sources(tmp_path: Path) -> None:
    """Cleanup handles repositories with both hf and dir sources.

    Regression: previously, hitting a dir-source entry would crash.
    """
    store = tmp_path / "store"
    store.mkdir()

    # Create hf-source model.
    hf_model_dir = store / "OpenVINO" / "Qwen"
    hf_model_dir.mkdir(parents=True)
    (hf_model_dir / "graph.pbtxt").write_text("pristine", encoding="utf-8")
    (hf_model_dir / "graph.hf_model.pbtxt").write_text("active", encoding="utf-8")
    (hf_model_dir / "graph.old_hf.pbtxt").write_text("obsolete", encoding="utf-8")

    # Create dir-source model.
    dir_model_dir = store / "local" / "dir_model"
    dir_model_dir.mkdir(parents=True)
    (dir_model_dir / "graph.pbtxt").write_text("pristine", encoding="utf-8")
    (dir_model_dir / "graph.dir_model.pbtxt").write_text("active", encoding="utf-8")
    (dir_model_dir / "graph.old_dir.pbtxt").write_text("obsolete", encoding="utf-8")

    config_data = {
        "runtime": {"rest_port": 8000},
        "repository": {
            "qwen": {"hf": "OpenVINO/Qwen", "task": "text_generation"},
            "dir_mdl": {"dir": "local/dir_model", "task": "text_generation"},
        },
        "models": {},
        "profiles": {},
    }
    ovms = OvmsConfig.model_validate(config_data)

    # Cleanup with only the active models should remove both old_hf and old_dir.
    cleaned = cleanup_obsolete_sibling_graphs(
        store, {"hf_model", "dir_model"}, ovms
    )
    assert len(cleaned) == 2
    assert not (hf_model_dir / "graph.old_hf.pbtxt").exists()
    assert not (dir_model_dir / "graph.old_dir.pbtxt").exists()
    assert (hf_model_dir / "graph.hf_model.pbtxt").exists()
    assert (dir_model_dir / "graph.dir_model.pbtxt").exists()
