"""Tests for the typed Graph and ModelEntry models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ovms_rig.config.schema import Graph, ModelEntry


def test_graph_rejects_unknown_field() -> None:
    # Locks the migration away from draft_source_model (replaced by draft_model
    # ref + flat-layout apply) and from any other stray ovms flag name.
    with pytest.raises(ValidationError):
        Graph(draft_source_model="org/draft")


def test_graph_rejects_device() -> None:
    # device moved to ModelEntry (shared deployment knob); it is no longer a
    # graph field and must be rejected there.
    with pytest.raises(ValidationError):
        Graph(device="GPU")


def test_graph_draft_model_without_device_rejected() -> None:
    with pytest.raises(ValidationError, match="draft_model and draft_device"):
        Graph(draft_model="draft")


def test_graph_draft_device_without_model_rejected() -> None:
    with pytest.raises(ValidationError, match="draft_model and draft_device"):
        Graph(draft_device="CPU")


def test_graph_both_draft_fields_set_ok() -> None:
    g = Graph(draft_model="draft", draft_device="CPU")
    assert g.draft_model == "draft"
    assert g.draft_device == "CPU"


def test_graph_no_draft_fields_ok() -> None:
    g = Graph()
    assert g.draft_model is None
    assert g.draft_device is None


def test_entry_device_is_required() -> None:
    # device is mandatory on the entry -- ovms cannot start without a target device.
    with pytest.raises(ValidationError, match="device"):
        ModelEntry(source="qwen", graph=Graph())


def test_entry_plugin_config_defaults_to_none() -> None:
    e = ModelEntry(source="qwen", device="GPU", graph=Graph())
    assert e.plugin_config is None


def test_entry_plugin_config_accepts_string_dict() -> None:
    e = ModelEntry(
        source="qwen",
        device="GPU",
        plugin_config={"CACHE_DIR": "C:/cache", "PERFORMANCE_HINT": "LATENCY"},
        graph=Graph(),
    )
    assert e.plugin_config == {"CACHE_DIR": "C:/cache", "PERFORMANCE_HINT": "LATENCY"}
