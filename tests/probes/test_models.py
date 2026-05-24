"""Tests for models probe."""

from __future__ import annotations

from ovms_rig.config import load_ovms
from ovms_rig.probes import models

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

    result = models.check(ovms)
    assert result.status == "ok"
    assert "1 model(s)" in result.summary
    assert result.details["models"]["ep"]["source"] == "main"
    assert result.details["models"]["ep"]["source_status"] == "ok"


def test_models_profile_membership(tmp_path):
    """Models correctly report which profiles contain them."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_MULTIPLE_MODELS, encoding="utf-8")
    ovms = load_ovms(cfg)

    result = models.check(ovms)
    assert result.status == "ok"
    # ep is in default and bench
    assert result.details["models"]["ep"]["profiles"] == ["default", "bench"]
    # draft_ep is only in bench
    assert result.details["models"]["draft_ep"]["profiles"] == ["bench"]
