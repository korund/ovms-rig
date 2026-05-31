"""Tests for config.json registry rendering (mediapipe + plain lists)."""

from __future__ import annotations

import json
from pathlib import Path

from ovms_rig.stages.activation.registry import render_config


def test_render_replaces_existing_entries(tmp_path: Path) -> None:
    """Render rewrites mediapipe_config_list as exact projection of desired."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "mediapipe_config_list": [
            {"name": "stale", "base_path": "/old/path", "graph_path": "graph.stale.pbtxt"},
        ],
    }), encoding="utf-8")

    render_config(config_path, {
        "new_model": (tmp_path / "new", "graph.new.pbtxt"),
    }, {})

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

    render_config(config_path, {}, {})

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["mediapipe_config_list"] == []


def test_render_plain_model_entry(tmp_path: Path) -> None:
    """Plain models land in model_config_list with target_device."""
    config_path = tmp_path / "config.json"
    render_config(config_path, {}, {
        "doclayout": (tmp_path / "pp-doclayout", "NPU", None, None),
    })

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["mediapipe_config_list"] == []
    assert data["model_config_list"] == [
        {
            "config": {
                "name": "doclayout",
                "base_path": str(tmp_path / "pp-doclayout"),
                "target_device": "NPU",
            }
        },
    ]


def test_render_plain_model_with_plugin_config(tmp_path: Path) -> None:
    """plugin_config is emitted into the plain config object when present."""
    config_path = tmp_path / "config.json"
    render_config(config_path, {}, {
        "doclayout": (tmp_path / "d", "NPU", {"PERFORMANCE_HINT": "LATENCY"}, None),
    })

    cfg = json.loads(config_path.read_text(encoding="utf-8"))["model_config_list"][0]["config"]
    assert cfg["plugin_config"] == {"PERFORMANCE_HINT": "LATENCY"}


def test_render_plain_model_with_plain_options(tmp_path: Path) -> None:
    """plain dict is merged verbatim into the config object."""
    config_path = tmp_path / "config.json"
    render_config(config_path, {}, {
        "doclayout": (tmp_path / "d", "NPU", None, {"batch_size": 4, "nireq": 8}),
    })

    cfg = json.loads(config_path.read_text(encoding="utf-8"))["model_config_list"][0]["config"]
    assert cfg["batch_size"] == 4
    assert cfg["nireq"] == 8


def test_render_plain_model_with_both_plugin_and_plain(tmp_path: Path) -> None:
    """plugin_config and plain options both present in config."""
    config_path = tmp_path / "config.json"
    render_config(config_path, {}, {
        "doclayout": (tmp_path / "d", "NPU", {"CACHE_DIR": "/cache"}, {"batch_size": 2}),
    })

    cfg = json.loads(config_path.read_text(encoding="utf-8"))["model_config_list"][0]["config"]
    assert cfg["plugin_config"] == {"CACHE_DIR": "/cache"}
    assert cfg["batch_size"] == 2


def test_render_plain_model_rig_owned_keys_override_plain(tmp_path: Path) -> None:
    """Rig-owned keys (name, base_path, target_device) always win over plain.

    If a user passes a plain dict with a colliding rig-owned key, the rig's
    computed value is preserved. The plain dict is still merged for other
    fields (e.g. batch_size), enforcing the precedence rule without validation.
    """
    config_path = tmp_path / "config.json"
    computed_path = tmp_path / "computed"
    render_config(config_path, {}, {
        "model_a": (computed_path, "GPU", None,
                    {"base_path": "HACK", "batch_size": 4}),
    })

    cfg = json.loads(config_path.read_text(encoding="utf-8"))["model_config_list"][0]["config"]
    assert cfg["name"] == "model_a"
    assert cfg["base_path"] == str(computed_path)
    assert cfg["target_device"] == "GPU"
    assert cfg["batch_size"] == 4


def test_render_mixed_lists(tmp_path: Path) -> None:
    """Both lists populated independently in one render."""
    config_path = tmp_path / "config.json"
    render_config(
        config_path,
        {"qwen": (tmp_path / "qwen", "graph.qwen.pbtxt")},
        {"doclayout": (tmp_path / "doclayout", "NPU", None, None)},
    )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert len(data["mediapipe_config_list"]) == 1
    assert len(data["model_config_list"]) == 1


def test_render_wipes_unknown_top_level_keys(tmp_path: Path) -> None:
    """Rig owns config.json: stray keys are dropped.

    Previously the renderer preserved pre-existing top-level keys, which made
    config.json "sticky" -- a leftover entry from a different tool would survive
    every apply. The declarative contract requires the file to be an exact
    projection of the active profile.
    """
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "model_config_list": [
            {"config": {"name": "leftover", "base_path": "/garbage"}},
        ],
        "monitoring": {"metrics": {"enable": True}},
        "mediapipe_config_list": [],
    }), encoding="utf-8")

    render_config(config_path, {
        "m": (tmp_path / "m", "graph.m.pbtxt"),
    }, {})

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data == {
        "model_config_list": [],
        "mediapipe_config_list": [
            {"name": "m", "base_path": str(tmp_path / "m"), "graph_path": "graph.m.pbtxt"},
        ],
    }


def test_render_creates_file_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    render_config(config_path, {
        "m": (tmp_path / "m", "graph.m.pbtxt"),
    }, {})
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["model_config_list"] == []
    assert len(data["mediapipe_config_list"]) == 1
