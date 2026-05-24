"""Tests for profiles probe."""

from __future__ import annotations

from ovms_rig.config import load_ovms
from ovms_rig.probes import profiles

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

OVMS_YAML_WITH_PROFILES = """
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
  other:
    source: main2
    graph:
      device: CPU

profiles:
  default:
    models: [ep]
    active: true
  bench:
    models: [ep, other]
    active: false
"""


def test_profiles_none_declared(tmp_path):
    """No profiles declared -> ok status."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_NO_PROFILES, encoding="utf-8")
    ovms = load_ovms(cfg)

    result = profiles.check(ovms)
    assert result.status == "ok"
    assert "no profiles" in result.summary


def test_profiles_with_active(tmp_path):
    """One active profile -> ok, shows active name."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_WITH_PROFILES, encoding="utf-8")
    ovms = load_ovms(cfg)

    result = profiles.check(ovms)
    assert result.status == "ok"
    assert "default" in result.summary
    assert "active profile" in result.summary
    assert result.details["profiles"]["default"]["active"] is True
    assert result.details["profiles"]["bench"]["active"] is False


def test_profiles_model_membership(tmp_path):
    """Model membership correctly maps profiles."""
    cfg = tmp_path / "ovms.yaml"
    cfg.write_text(OVMS_YAML_WITH_PROFILES, encoding="utf-8")
    ovms = load_ovms(cfg)

    result = profiles.check(ovms)
    # ep is in both default and bench
    assert result.details["model_membership"]["ep"] == ["default", "bench"]
    # other is only in bench
    assert result.details["model_membership"]["other"] == ["bench"]
