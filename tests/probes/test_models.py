"""Tests for models probe."""

from __future__ import annotations

from ovms_rig.config import Declaration, load_ovms
from ovms_rig.config.schema import LocalConfig, LocalModels, LocalRuntime
from ovms_rig.probes import models
from pathlib import Path

OVMS_YAML_VALID = """
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

OVMS_YAML_MULTIPLE_MODELS = """
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
  draft_ep:
    source: draft
    graph:
      device: CPU

profiles:
  default:
    models: [ep]
    active: true
  bench:
    models: [ep, draft_ep]
    active: false
"""


def test_models_valid_source(tmp_path):
    """Valid source reference -> ok status."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_VALID, encoding="utf-8")
    ovms = load_ovms(cfg)
    local = LocalConfig(
        runtime=LocalRuntime(),
        models=LocalModels(repository_path=Path("/tmp/store")),
    )
    decl = Declaration(ovms=ovms, local=local)

    result = models.check(decl)
    assert result.status == "ok"
    assert "1 model(s)" in result.summary
    assert result.details["models"]["ep"]["source"] == "main"
    assert result.details["models"]["ep"]["source_status"] == "ok"


def test_models_orphan_not_in_any_profile(tmp_path):
    """Models not in any profile report empty profiles list."""
    ovms_yaml = """
runtime:
  rest_port: 8000

repository:
  main:
    hf: org/main
    task: text_generation
  main2:
    hf: org/main2
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
  orphan:
    source: main2
    graph:
      device: CPU

profiles:
  default:
    models: [ep]
    active: true
"""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(ovms_yaml, encoding="utf-8")
    ovms = load_ovms(cfg)
    local = LocalConfig(
        runtime=LocalRuntime(),
        models=LocalModels(repository_path=Path("/tmp/store")),
    )
    decl = Declaration(ovms=ovms, local=local)

    result = models.check(decl)
    assert result.status == "ok"
    # ep is in default profile
    assert result.details["models"]["ep"]["profiles"] == ["default"]
    # orphan is not in any profile
    assert result.details["models"]["orphan"]["profiles"] == []
