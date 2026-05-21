"""Tests for the typed Graph model: pull/pbtxt bucket split."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ovms_rig.config.schema import GRAPH_PBTXT_FIELDS, GRAPH_PULL_FIELDS, Graph


def test_pull_flags_returns_only_pull_bucket() -> None:
    g = Graph(
        max_num_seqs=256,
        enable_prefix_caching=True,
        cache_size=0,
        device="GPU",
        draft_device="CPU",
        draft_model="draft",
    )
    flags = g.pull_flags()
    assert set(flags) == {"max_num_seqs", "enable_prefix_caching", "cache_size"}
    assert flags["max_num_seqs"] == 256
    assert flags["enable_prefix_caching"] is True
    assert flags["cache_size"] == 0


def test_pull_flags_skips_unset_fields() -> None:
    g = Graph(device="GPU")
    assert g.pull_flags() == {}


def test_graph_rejects_unknown_field() -> None:
    # Locks the migration away from draft_source_model (replaced by draft_model
    # ref + flat-layout apply) and from any other stray ovms flag name.
    with pytest.raises(ValidationError):
        Graph(draft_source_model="org/draft")


def test_graph_device_is_required() -> None:
    # device is mandatory -- ovms cannot start without a target device.
    with pytest.raises(ValidationError, match="device"):
        Graph(max_num_seqs=256)


def test_bucket_sets_are_disjoint_and_cover_all_fields() -> None:
    # Invariant: every Graph field belongs to exactly one bucket.
    declared = set(Graph.model_fields)
    assert GRAPH_PULL_FIELDS.isdisjoint(GRAPH_PBTXT_FIELDS)
    assert GRAPH_PULL_FIELDS | GRAPH_PBTXT_FIELDS == declared
