"""Loader: schema validation + internal reference resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from ovms_rig.config import ConfigError, load_local, load_ovms

VALID_OVMS = """
runtime:
  rest_port: 8000
  log_level: INFO

repository:
  main:
    hf: org/main-int8-ov
    task: text_generation
  draft:
    hf: org/draft-int8-ov
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
      draft_model: draft
      draft_device: CPU
"""


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_ovms_happy_path(tmp_path: Path) -> None:
    cfg = load_ovms(_write(tmp_path / "ovms.yaml", VALID_OVMS))
    assert cfg.runtime.rest_port == 8000
    assert set(cfg.repository) == {"main", "draft"}
    assert cfg.models["ep"].graph.draft_model == "draft"


def test_load_ovms_dangling_model_reference(tmp_path: Path) -> None:
    bad = VALID_OVMS.replace("source: main", "source: ghost")
    with pytest.raises(ConfigError, match="unknown source 'ghost'"):
        load_ovms(_write(tmp_path / "ovms.yaml", bad))


def test_load_ovms_dangling_draft_reference(tmp_path: Path) -> None:
    bad = VALID_OVMS.replace("draft_model: draft", "draft_model: ghost")
    with pytest.raises(ConfigError, match="unknown draft_model 'ghost'"):
        load_ovms(_write(tmp_path / "ovms.yaml", bad))


def test_load_ovms_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    bad = VALID_OVMS + "\nstray: 1\n"
    with pytest.raises(ConfigError, match="schema validation failed"):
        load_ovms(_write(tmp_path / "ovms.yaml", bad))


def test_load_ovms_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_ovms(_write(tmp_path / "ovms.yaml", "runtime: [::"))


def test_load_local_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        load_local(tmp_path / "absent.yaml")


def test_load_local_minimal_yields_defaults(tmp_path: Path) -> None:
    # repository_path is required; everything else falls back to defaults.
    body = "models:\n  repository_path: C:/store\n"
    cfg = load_local(_write(tmp_path / "local.yaml", body))
    assert cfg.runtime.ovms_path is None
    assert str(cfg.models.repository_path).replace("\\", "/") == "C:/store"


def test_load_local_rejects_missing_repository_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="schema validation failed"):
        load_local(_write(tmp_path / "local.yaml", ""))


# Tests for profiles section
def test_load_ovms_profiles_missing_section_yields_empty_dict(tmp_path: Path) -> None:
    cfg = load_ovms(_write(tmp_path / "ovms.yaml", VALID_OVMS))
    assert cfg.profiles == {}


def test_load_ovms_profiles_parsing(tmp_path: Path) -> None:
    yaml_with_profiles = VALID_OVMS + """
profiles:
  default:
    models: [ep]
    active: true
  bench:
    models: [ep]
"""
    cfg = load_ovms(_write(tmp_path / "ovms.yaml", yaml_with_profiles))
    assert "default" in cfg.profiles
    assert "bench" in cfg.profiles
    assert cfg.profiles["default"].models == ["ep"]
    assert cfg.profiles["default"].active is True
    assert cfg.profiles["bench"].active is False


def test_load_ovms_profile_active_defaults_false(tmp_path: Path) -> None:
    yaml_with_profile = VALID_OVMS + """
profiles:
  test:
    models: [ep]
"""
    cfg = load_ovms(_write(tmp_path / "ovms.yaml", yaml_with_profile))
    assert cfg.profiles["test"].active is False


def test_load_ovms_profile_dangling_model_reference(tmp_path: Path) -> None:
    yaml_with_bad_profile = VALID_OVMS + """
profiles:
  test:
    models: [unknown_model]
"""
    with pytest.raises(ConfigError, match="profile 'test' references unknown model 'unknown_model'"):
        load_ovms(_write(tmp_path / "ovms.yaml", yaml_with_bad_profile))


def test_load_ovms_profile_multiple_active_rejected(tmp_path: Path) -> None:
    yaml_with_multiple_active = VALID_OVMS + """
profiles:
  default:
    models: [ep]
    active: true
  bench:
    models: [ep]
    active: true
"""
    with pytest.raises(ConfigError, match="at most one profile can be active.*got 2"):
        load_ovms(_write(tmp_path / "ovms.yaml", yaml_with_multiple_active))


def test_load_ovms_profile_single_active_ok(tmp_path: Path) -> None:
    yaml_with_one_active = VALID_OVMS + """
profiles:
  default:
    models: [ep]
    active: true
  bench:
    models: [ep]
"""
    cfg = load_ovms(_write(tmp_path / "ovms.yaml", yaml_with_one_active))
    active_count = sum(1 for p in cfg.profiles.values() if p.active)
    assert active_count == 1


def test_load_ovms_profile_zero_active_ok(tmp_path: Path) -> None:
    yaml_with_no_active = VALID_OVMS + """
profiles:
  default:
    models: [ep]
  bench:
    models: [ep]
"""
    cfg = load_ovms(_write(tmp_path / "ovms.yaml", yaml_with_no_active))
    active_count = sum(1 for p in cfg.profiles.values() if p.active)
    assert active_count == 0


def test_load_ovms_rejects_duplicate_source_per_model(tmp_path: Path) -> None:
    """Invariant: one source per model. Two models with same source rejected."""
    bad = """
runtime:
  rest_port: 8000

repository:
  main:
    hf: org/main-int8-ov
    task: text_generation

models:
  ep1:
    source: main
    graph:
      device: GPU
  ep2:
    source: main
    graph:
      device: CPU
"""
    # Error message references OVMS limitation and the conflicting entries.
    with pytest.raises(ConfigError, match="one source can be target|ep1|ep2|OVMS limitation"):
        load_ovms(_write(tmp_path / "ovms.yaml", bad))


def test_load_ovms_rejects_empty_yaml(tmp_path: Path) -> None:
    """Empty or whitespace-only YAML rejected (missing required runtime section)."""
    with pytest.raises(ConfigError, match="schema validation failed|required"):
        load_ovms(_write(tmp_path / "ovms.yaml", ""))
