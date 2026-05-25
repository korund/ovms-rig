"""Tests for mediapipe_config_list registry rendering."""

from __future__ import annotations

import json
from pathlib import Path

from ovms_rig.stages.activation.registry import render_mediapipe_entries


def test_render_replaces_existing_entries(tmp_path: Path) -> None:
    """Render rewrites mediapipe_config_list as exact projection of desired."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "mediapipe_config_list": [
            {"name": "stale", "base_path": "/old/path", "graph_path": "graph.stale.pbtxt"},
        ],
    }), encoding="utf-8")

    render_mediapipe_entries(config_path, {
        "new_model": (tmp_path / "new", "graph.new.pbtxt"),
    })

    data = json.loads(config_path.read_text(encoding="utf-8"))
    entries = data["mediapipe_config_list"]
    assert len(entries) == 1
    assert entries[0]["name"] == "new_model"
    assert entries[0]["graph_path"] == "graph.new.pbtxt"


def test_render_with_empty_desired_entries(tmp_path: Path) -> None:
    """Empty desired -> empty mediapipe_config_list."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "mediapipe_config_list": [
            {"name": "ep", "base_path": "/old/path", "graph_path": "graph.ep.pbtxt"},
        ],
    }), encoding="utf-8")

    render_mediapipe_entries(config_path, {})

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["mediapipe_config_list"] == []


def test_render_wipes_unknown_top_level_keys(tmp_path: Path) -> None:
    """Rig owns config.json: stray model_config_list / unknown keys are dropped.

    Previously the renderer preserved pre-existing top-level keys, which made
    config.json "sticky" -- a leftover model_config_list from a different tool
    would survive every apply. The declarative contract requires the file to
    be an exact projection of the active profile.
    """
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "model_config_list": [
            {"config": {"name": "leftover", "base_path": "/garbage"}},
        ],
        "monitoring": {"metrics": {"enable": True}},
        "mediapipe_config_list": [],
    }), encoding="utf-8")

    render_mediapipe_entries(config_path, {
        "m": (tmp_path / "m", "graph.m.pbtxt"),
    })

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data == {
        "model_config_list": [],
        "mediapipe_config_list": [
            {"name": "m", "base_path": str(tmp_path / "m"), "graph_path": "graph.m.pbtxt"},
        ],
    }


def test_render_creates_file_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    render_mediapipe_entries(config_path, {
        "m": (tmp_path / "m", "graph.m.pbtxt"),
    })
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["model_config_list"] == []
    assert len(data["mediapipe_config_list"]) == 1
