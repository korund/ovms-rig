"""Tests for apply rollback behavior (fail-fast + atomicity)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Minimal pbtxt content similar to real OVMS output.
GRAPH_PBTXT = """\
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            enable_prefix_caching: true,
            cache_size: 0,
        }
    }
"""

LOCAL_YAML = """\
runtime:
  ovms_path: null
models:
  repository_path: {store}
"""


def test_apply_rollback_restores_config_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Rollback restores config.json from snapshot after failure."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    # Create initial config.json with some content.
    config_json = store / "config.json"
    original_config = {
        "mediapipe_config_list": [
            {
                "name": "old_model",
                "base_path": "/old/path",
                "graph_path": "graph.old_model.pbtxt",
            }
        ]
    }
    config_json.write_text(json.dumps(original_config, indent=2), encoding="utf-8")

    # Create only main model directory (draft will be missing).
    model_dir = store / "OpenVINO" / "main-int8-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")

    # YAML with two models but only one directory exists.
    ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: DEBUG

repository:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation
  draft:
    hf: OpenVINO/draft-int8-ov
    task: text_generation

models:
  ep:
    source: main
    device: GPU
    graph: {}
  draft_ep:
    source: draft
    device: CPU
    graph: {}

profiles:
  default:
    models: [ep, draft_ep]
    active: true
"""
    cfg.write_text(ovms_yaml, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    from ovms_rig.stages.activation import apply
    ctx = {
        "config_path": str(cfg),
        "local_path": str(loc),
        "ovms_path": sys.executable,
        "log_level": None,
        "dry_run": False,
        "extras": [],
    }
    rc = apply.run(ctx)

    # Should fail.
    assert rc == 1

    # config.json should be restored to original content.
    assert config_json.exists()
    restored_config = json.loads(config_json.read_text(encoding="utf-8"))
    assert restored_config == original_config, "config.json should be restored from snapshot"


def test_apply_rollback_deletes_new_graphs_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Rollback deletes only graph files created in failed run, preserves existing graphs."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    # Create main model directory with an existing sibling graph.
    model_dir = store / "OpenVINO" / "main-int8-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
    # Create an existing sibling graph from previous activation.
    (model_dir / "graph.old_ep.pbtxt").write_text("existing graph", encoding="utf-8")

    # YAML with two models but only one directory exists.
    ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: DEBUG

repository:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation
  draft:
    hf: OpenVINO/draft-int8-ov
    task: text_generation

models:
  ep:
    source: main
    device: GPU
    graph: {}
  draft_ep:
    source: draft
    device: CPU
    graph: {}

profiles:
  default:
    models: [ep, draft_ep]
    active: true
"""
    cfg.write_text(ovms_yaml, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    from ovms_rig.stages.activation import apply
    ctx = {
        "config_path": str(cfg),
        "local_path": str(loc),
        "ovms_path": sys.executable,
        "log_level": None,
        "dry_run": False,
        "extras": [],
    }
    rc = apply.run(ctx)

    # Should fail.
    assert rc == 1

    # Existing sibling graph should be preserved.
    assert (model_dir / "graph.old_ep.pbtxt").exists(), "Existing graphs should be preserved"

    # New sibling graph should NOT be created.
    assert not (model_dir / "graph.ep.pbtxt").exists(), "New graph should not be created after rollback"
