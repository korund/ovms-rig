"""Tests for mediapipe_config_list registry rendering."""

from __future__ import annotations

import json
from pathlib import Path

from ovms_rig.stages.activation.registry import render_mediapipe_entries


def test_render_skips_entry_without_name(tmp_path: Path) -> None:
    """Render handles malformed entry without name gracefully."""
    config_path = tmp_path / "config.json"

    # Create config.json with entry missing name and some valid entries.
    config_data = {
        "mediapipe_config_list": [
            {
                "base_path": "/some/path",
                "graph_path": "graph.old.pbtxt",
                # Missing 'name' key
            },
            {
                "name": "existing",
                "base_path": "/old/path",
                "graph_path": "graph.existing.pbtxt",
            }
        ]
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    # Render with new desired entries.
    desired_entries = {
        "new_model": (tmp_path / "new", "graph.new.pbtxt"),
    }

    # Should not crash.
    render_mediapipe_entries(config_path, desired_entries)

    # Check result: should contain only desired entry (no malformed, no old).
    result_data = json.loads(config_path.read_text(encoding="utf-8"))
    result_entries = result_data["mediapipe_config_list"]

    assert len(result_entries) == 1
    assert result_entries[0]["name"] == "new_model"
    assert "graph.new.pbtxt" in result_entries[0]["graph_path"]


def test_render_with_empty_desired_entries(tmp_path: Path) -> None:
    """Render with empty desired_entries clears mediapipe_config_list."""
    config_path = tmp_path / "config.json"

    # Create config.json with some entries.
    config_data = {
        "mediapipe_config_list": [
            {
                "name": "ep",
                "base_path": "/old/path",
                "graph_path": "graph.ep.pbtxt",
            }
        ]
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    # Render with empty desired.
    render_mediapipe_entries(config_path, {})

    # Result should have empty list.
    result_data = json.loads(config_path.read_text(encoding="utf-8"))
    assert result_data["mediapipe_config_list"] == []
