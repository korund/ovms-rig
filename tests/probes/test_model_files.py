"""Tests for model_files probe."""

from __future__ import annotations

from pathlib import Path

from ovms_rig.config import Declaration, load_ovms
from ovms_rig.config.schema import LocalConfig, LocalModels, LocalRuntime
from ovms_rig.probes import model_files


OVMS_YAML_BASE = """
runtime:
  rest_port: 8000

repository:
  main:
    hf: org/main
    task: text_generation
  draft:
    hf: org/draft
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
  draft_model:
    source: draft
    graph:
      device: CPU
"""

OVMS_YAML_WITH_SPECULATIVE = """
runtime:
  rest_port: 8000

repository:
  main:
    hf: org/main
    task: text_generation
  draft:
    hf: org/draft
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
      draft_model: draft
      draft_device: CPU

profiles:
  default:
    models: [ep]
    active: true
"""

OVMS_YAML_NO_PROFILES = """
runtime:
  rest_port: 8000

repository:
  main:
    hf: org/main
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
"""

OVMS_YAML_MULTIPLE_PROFILES = """
runtime:
  rest_port: 8000

repository:
  main:
    hf: org/main
    task: text_generation
  draft:
    hf: org/draft
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
  draft_model:
    source: draft
    graph:
      device: CPU

profiles:
  default:
    models: [ep, draft_model]
    active: true
  bench:
    models: [ep]
    active: false
"""


def test_no_active_profile(tmp_path):
    """No active profile -> status=ok, summary contains 'no active profile'."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_NO_PROFILES, encoding="utf-8")
    ovms = load_ovms(cfg)
    decl = Declaration(
        ovms=ovms,
        local=LocalConfig(
            runtime=LocalRuntime(),
            models=LocalModels(repository_path=tmp_path / "store"),
        ),
    )

    result = model_files.check(decl)
    assert result.status == "ok"
    assert "no active profile" in result.summary


def test_all_files_present(tmp_path):
    """All model dirs present, all files present -> status=ok, no missing files."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_MULTIPLE_PROFILES, encoding="utf-8")
    ovms = load_ovms(cfg)
    store = tmp_path / "store"

    # Create model directories with required and optional files.
    ep_dir = store / "org" / "main"
    ep_dir.mkdir(parents=True)
    (ep_dir / "graph.pbtxt").write_text("graph", encoding="utf-8")
    (ep_dir / "generation_config.json").write_text("{}", encoding="utf-8")

    draft_dir = store / "org" / "draft"
    draft_dir.mkdir(parents=True)
    (draft_dir / "graph.pbtxt").write_text("graph", encoding="utf-8")
    (draft_dir / "generation_config.json").write_text("{}", encoding="utf-8")

    decl = Declaration(
        ovms=ovms,
        local=LocalConfig(
            runtime=LocalRuntime(),
            models=LocalModels(repository_path=store),
        ),
    )

    result = model_files.check(decl)
    assert result.status == "ok"
    assert result.details["missing_required"] == {}
    assert result.details["missing_optional"] == {}
    assert "2/2 models complete" in result.summary


def test_draft_model_deduplication(tmp_path):
    """Model with draft_model: both dirs present, both checked, deduplicated."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_WITH_SPECULATIVE, encoding="utf-8")
    ovms = load_ovms(cfg)
    store = tmp_path / "store"

    # Create both primary and draft model directories.
    ep_dir = store / "org" / "main"
    ep_dir.mkdir(parents=True)
    (ep_dir / "graph.pbtxt").write_text("graph", encoding="utf-8")
    (ep_dir / "generation_config.json").write_text("{}", encoding="utf-8")

    draft_dir = store / "org" / "draft"
    draft_dir.mkdir(parents=True)
    (draft_dir / "graph.pbtxt").write_text("graph", encoding="utf-8")
    (draft_dir / "generation_config.json").write_text("{}", encoding="utf-8")

    decl = Declaration(
        ovms=ovms,
        local=LocalConfig(
            runtime=LocalRuntime(),
            models=LocalModels(repository_path=store),
        ),
    )

    result = model_files.check(decl)
    # Both directories exist and are checked, but model name appears once (deduped).
    assert "ep" in result.details["checked"]
    assert result.details["checked"].count("ep") == 1
    assert result.details["missing_required"] == {}


def test_model_dir_not_present_skipped(tmp_path):
    """Model dir missing on disk -> silently skipped (not in checked/missing)."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_MULTIPLE_PROFILES, encoding="utf-8")
    ovms = load_ovms(cfg)
    store = tmp_path / "store"

    # Create only ep_dir; omit draft_model directory.
    ep_dir = store / "org" / "main"
    ep_dir.mkdir(parents=True)
    (ep_dir / "graph.pbtxt").write_text("graph", encoding="utf-8")
    (ep_dir / "generation_config.json").write_text("{}", encoding="utf-8")

    decl = Declaration(
        ovms=ovms,
        local=LocalConfig(
            runtime=LocalRuntime(),
            models=LocalModels(repository_path=store),
        ),
    )

    result = model_files.check(decl)
    # Only ep should be checked; draft_model directory is missing and skipped.
    assert result.details["checked"] == ["ep"]
    assert "draft_model" not in result.details["checked"]
    assert "draft_model" not in result.details["missing_required"]
    assert "draft_model" not in result.details["missing_optional"]


def test_graph_pbtxt_missing_error(tmp_path):
    """graph.pbtxt missing -> status=error, model in missing_required."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_MULTIPLE_PROFILES, encoding="utf-8")
    ovms = load_ovms(cfg)
    store = tmp_path / "store"

    # Create ep_dir with generation_config but no graph.pbtxt.
    ep_dir = store / "org" / "main"
    ep_dir.mkdir(parents=True)
    (ep_dir / "generation_config.json").write_text("{}", encoding="utf-8")

    # Create draft_dir with both files.
    draft_dir = store / "org" / "draft"
    draft_dir.mkdir(parents=True)
    (draft_dir / "graph.pbtxt").write_text("graph", encoding="utf-8")
    (draft_dir / "generation_config.json").write_text("{}", encoding="utf-8")

    decl = Declaration(
        ovms=ovms,
        local=LocalConfig(
            runtime=LocalRuntime(),
            models=LocalModels(repository_path=store),
        ),
    )

    result = model_files.check(decl)
    assert result.status == "error"
    assert result.details["missing_required"] == {"ep": ["graph.pbtxt"]}
    assert "1/2 models missing required files" in result.summary
    assert result.hint and "missing graph.pbtxt blocks OVMS load" in result.hint


def test_generation_config_missing_ok(tmp_path):
    """generation_config.json missing but graph.pbtxt present -> status=ok."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_MULTIPLE_PROFILES, encoding="utf-8")
    ovms = load_ovms(cfg)
    store = tmp_path / "store"

    # Create ep_dir with graph.pbtxt but no generation_config.json.
    ep_dir = store / "org" / "main"
    ep_dir.mkdir(parents=True)
    (ep_dir / "graph.pbtxt").write_text("graph", encoding="utf-8")

    # Create draft_dir with graph.pbtxt only.
    draft_dir = store / "org" / "draft"
    draft_dir.mkdir(parents=True)
    (draft_dir / "graph.pbtxt").write_text("graph", encoding="utf-8")

    decl = Declaration(
        ovms=ovms,
        local=LocalConfig(
            runtime=LocalRuntime(),
            models=LocalModels(repository_path=store),
        ),
    )

    result = model_files.check(decl)
    assert result.status == "ok"
    assert result.details["missing_required"] == {}
    assert result.details["missing_optional"] == {
        "ep": ["generation_config.json"],
        "draft_model": ["generation_config.json"],
    }
    assert result.hint and "missing generation_config.json drops sampling overrides" in result.hint
