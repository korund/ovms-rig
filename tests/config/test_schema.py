"""Tests for the typed Graph model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ovms_rig.config.schema import Graph


def test_graph_rejects_unknown_field() -> None:
    # Locks the migration away from draft_source_model (replaced by draft_model
    # ref + flat-layout apply) and from any other stray ovms flag name.
    with pytest.raises(ValidationError):
        Graph(draft_source_model="org/draft")


def test_graph_device_is_required() -> None:
    # device is mandatory -- ovms cannot start without a target device.
    with pytest.raises(ValidationError, match="device"):
        Graph(max_num_seqs=256)


def test_graph_draft_model_without_device_rejected() -> None:
    with pytest.raises(ValidationError, match="draft_model and draft_device"):
        Graph(device="GPU", draft_model="draft")


def test_graph_draft_device_without_model_rejected() -> None:
    with pytest.raises(ValidationError, match="draft_model and draft_device"):
        Graph(device="GPU", draft_device="CPU")


def test_graph_both_draft_fields_set_ok() -> None:
    g = Graph(device="GPU", draft_model="draft", draft_device="CPU")
    assert g.draft_model == "draft"
    assert g.draft_device == "CPU"


def test_graph_no_draft_fields_ok() -> None:
    g = Graph(device="GPU")
    assert g.draft_model is None
    assert g.draft_device is None


def test_graph_plugin_config_defaults_to_none() -> None:
    g = Graph(device="GPU")
    assert g.plugin_config is None


def test_graph_plugin_config_accepts_string_dict() -> None:
    g = Graph(
        device="GPU",
        plugin_config={"CACHE_DIR": "C:/cache", "PERFORMANCE_HINT": "LATENCY"},
    )
    assert g.plugin_config == {"CACHE_DIR": "C:/cache", "PERFORMANCE_HINT": "LATENCY"}
