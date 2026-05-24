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
