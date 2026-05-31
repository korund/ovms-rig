"""Tests for the typed Graph and ModelEntry models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ovms_rig.config.schema import Graph, ModelEntry, ModelIdentity


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


def test_entry_plain_defaults_to_none() -> None:
    e = ModelEntry(source="qwen", device="GPU", graph=Graph())
    assert e.plain is None


def test_entry_plain_accepts_object_dict() -> None:
    e = ModelEntry(
        source="mobilenet",
        device="CPU",
        plain={"batch_size": 4, "nireq": 8, "model_version_policy": "latest"},
    )
    assert e.plain == {"batch_size": 4, "nireq": 8, "model_version_policy": "latest"}


def test_entry_plain_rejects_both_graph_and_plain() -> None:
    with pytest.raises(ValidationError, match="plain and graph are mutually exclusive"):
        ModelEntry(
            source="model",
            device="GPU",
            graph=Graph(),
            plain={"batch_size": 4},
        )


def test_model_identity_hf_alone_ok() -> None:
    # hf-based model (traditional HuggingFace source) is valid.
    m = ModelIdentity(hf="OpenVINO/Qwen3-14B", task="text_generation")
    assert m.hf == "OpenVINO/Qwen3-14B"
    assert m.dir is None
    assert m.task == "text_generation"


def test_model_identity_dir_alone_ok() -> None:
    # dir-based model (local directory source) is valid.
    m = ModelIdentity(dir="local/models/qwen")
    assert m.dir == "local/models/qwen"
    assert m.hf is None
    assert m.task is None


def test_model_identity_dir_absolute_path_ok() -> None:
    # dir-based model with absolute path is valid.
    m = ModelIdentity(dir="/opt/models/mobilenet")
    assert m.dir == "/opt/models/mobilenet"
    assert m.hf is None


def test_model_identity_rejects_both_hf_and_dir() -> None:
    # Exactly one source kind is required; both is an error.
    with pytest.raises(ValidationError, match="exactly one of hf or dir must be set"):
        ModelIdentity(hf="OpenVINO/Qwen", dir="local/models/qwen")


def test_model_identity_rejects_neither_hf_nor_dir() -> None:
    # Neither hf nor dir is an error.
    with pytest.raises(ValidationError, match="exactly one of hf or dir must be set"):
        ModelIdentity(task="text_generation")
