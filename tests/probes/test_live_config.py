"""Tests for live_config probe."""

from __future__ import annotations

import json
from pathlib import Path

from ovms_rig.config import Declaration, load_local, load_ovms
from ovms_rig.probes import live_config

OVMS_YAML = """
runtime:
  rest_port: 8000

repository:
  main:
    hf: org/main
    task: text_generation

models:
  ep:
    source: main
    device: GPU
    graph: {}

profiles:
  default:
    models: [ep]
    active: true
"""

OVMS_YAML_NO_ACTIVE = """
runtime:
  rest_port: 8000

repository:
  main:
    hf: org/main
    task: text_generation

models:
  ep:
    source: main
    device: GPU
    graph: {}

profiles:
  default:
    models: [ep]
    active: false
"""

LOCAL_YAML = """
runtime:
  ovms_path: null
models:
  repository_path: {store}
"""


def test_live_config_no_file_no_active_profile(tmp_path):
    """No config.json and no active profile -> ok."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    cfg.write_text(OVMS_YAML_NO_ACTIVE, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    ovms = load_ovms(cfg)
    local = load_local(loc)
    decl = Declaration(ovms=ovms, local=local)

    result = live_config.check(decl)
    assert result.status == "ok"
    assert "no active profile" in result.summary


def test_live_config_no_file_active_profile(tmp_path):
    """No config.json but active profile -> warn."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    cfg.write_text(OVMS_YAML, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    ovms = load_ovms(cfg)
    local = load_local(loc)
    decl = Declaration(ovms=ovms, local=local)

    result = live_config.check(decl)
    assert result.status == "warn"
    assert "config.json missing" in result.summary
    assert "default" in result.summary


def test_live_config_matches_active_profile(tmp_path):
    """config.json matches active profile models -> ok."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    cfg.write_text(OVMS_YAML, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    ovms = load_ovms(cfg)
    local = load_local(loc)
    decl = Declaration(ovms=ovms, local=local)

    # Create config.json with matching entry.
    config_json = store / "config.json"
    config_data = {
        "mediapipe_config_list": [
            {
                "name": "ep",
                "graph_path": "org/main/graph.ep.pbtxt",
            }
        ]
    }
    config_json.write_text(json.dumps(config_data), encoding="utf-8")

    result = live_config.check(decl)
    assert result.status == "ok"
    assert "OK: 1 model(s)" in result.summary
    assert result.details["active_profile"] == "default"
    # Verify probe computed matching sets (not just the count).
    assert set(result.details["expected_models"]) == {"ep"}
    assert set(result.details["live_models"]) == {"ep"}


def test_live_config_extra_model_in_live(tmp_path):
    """config.json has extra model not in active profile -> warn."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    cfg.write_text(OVMS_YAML, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    ovms = load_ovms(cfg)
    local = load_local(loc)
    decl = Declaration(ovms=ovms, local=local)

    # Create config.json with extra entry.
    config_json = store / "config.json"
    config_data = {
        "mediapipe_config_list": [
            {
                "name": "ep",
                "graph_path": "org/main/graph.ep.pbtxt",
            },
            {
                "name": "extra",
                "graph_path": "org/other/graph.extra.pbtxt",
            }
        ]
    }
    config_json.write_text(json.dumps(config_data), encoding="utf-8")

    result = live_config.check(decl)
    assert result.status == "warn"
    assert "mismatch" in result.summary
    assert "extra" in result.details["extra_in_live"]


def test_live_config_missing_model_in_live(tmp_path):
    """config.json missing model from active profile -> warn."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    cfg.write_text(OVMS_YAML, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    ovms = load_ovms(cfg)
    local = load_local(loc)
    decl = Declaration(ovms=ovms, local=local)

    # Create empty config.json.
    config_json = store / "config.json"
    config_data = {"mediapipe_config_list": []}
    config_json.write_text(json.dumps(config_data), encoding="utf-8")

    result = live_config.check(decl)
    assert result.status == "warn"
    assert "mismatch" in result.summary
    assert "ep" in result.details["missing_from_live"]


def test_live_config_active_profile_empty_models(tmp_path):
    """Active profile with empty models list -> ok with empty config.json."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    # YAML with active profile that has no models.
    ovms_yaml = """
runtime:
  rest_port: 8000

repository:
  main:
    hf: org/main
    task: text_generation

models:
  ep:
    source: main
    device: GPU
    graph: {}

profiles:
  empty_profile:
    models: []
    active: true
"""
    cfg.write_text(ovms_yaml, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    ovms = load_ovms(cfg)
    local = load_local(loc)
    decl = Declaration(ovms=ovms, local=local)

    # Create config.json with empty list (matches empty active profile).
    config_json = store / "config.json"
    config_data = {"mediapipe_config_list": []}
    config_json.write_text(json.dumps(config_data), encoding="utf-8")

    result = live_config.check(decl)
    assert result.status == "ok"
    assert "empty config" in result.summary or "OK" in result.summary
    assert result.details["active_profile"] == "empty_profile"
    assert set(result.details["expected_models"]) == set()
    assert set(result.details["live_models"]) == set()


PLAIN_YAML = """
runtime:
  rest_port: 8000

repository:
  doclayout:
    hf: pp-doclayout-m

models:
  layout:
    source: doclayout
    device: NPU

profiles:
  default:
    models: [layout]
    active: true
"""


def test_live_config_recognizes_plain_model_in_model_config_list(tmp_path):
    """A plain model lives in model_config_list; the probe must see it there."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    cfg.write_text(PLAIN_YAML, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    ovms = load_ovms(cfg)
    local = load_local(loc)
    decl = Declaration(ovms=ovms, local=local)

    config_json = store / "config.json"
    config_data = {
        "model_config_list": [
            {"config": {"name": "layout", "base_path": "/x", "target_device": "NPU"}}
        ],
        "mediapipe_config_list": [],
    }
    config_json.write_text(json.dumps(config_data), encoding="utf-8")

    result = live_config.check(decl)
    assert result.status == "ok"
    assert set(result.details["live_models"]) == {"layout"}
