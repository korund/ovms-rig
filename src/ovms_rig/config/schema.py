"""Typed schema for the two declaration files (ovms.yaml + local.yaml).

This module only defines the shape; loading and reference resolution land
in the loader (next step). Schema is intentionally strict (extra="forbid")
so typos in YAML surface immediately as validation errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]
Device = Literal["CPU", "GPU", "NPU"]
# Task taxonomy as accepted by `ovms --pull --task`. Set per model when the
# model is generative (pull cannot infer it from the HF repo). Left unset for
# plain models (e.g. detection/classification ONNX or OpenVINO IR) which OVMS
# serves via model_config_list without a mediapipe graph and which are not
# pulled by `ovms --pull` at all -- their files are placed in the store directly.
Task = Literal[
    "text_generation",
    "embeddings",
    "rerank",
    "image_generation",
    "text2speech",
    "speech2text",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----- ovms.yaml -----

class Runtime(_Strict):
    rest_port: int = Field(ge=1, le=65535)
    log_level: LogLevel = "INFO"


class ModelIdentity(_Strict):
    # Model source kind: exactly one of hf or dir must be set.
    # hf: pulled via `ovms --pull` (HuggingFace coordinate org/repo).
    # dir: local directory holding weights (not fetched, user-managed).
    hf: str | None = None
    dir: str | None = None
    # None -> plain model (model_config_list, no graph, not pulled).
    task: Task | None = None
    revision: str | None = None

    @model_validator(mode="after")
    def _source_kind_is_exclusive(self) -> ModelIdentity:
        # Exactly one of hf or dir must be set. Both or neither is an error.
        # Structure supports adding a third kind (e.g., github) later: just add
        # a field and update this validator to count set fields.
        sources_set = sum([
            self.hf is not None,
            self.dir is not None,
        ])
        if sources_set != 1:
            raise ValueError(
                f"exactly one of hf or dir must be set; "
                f"got hf={self.hf!r}, dir={self.dir!r}"
            )
        return self


class Graph(_Strict):
    # All graph fields are patched into graph.pbtxt during apply. `ovms --pull`
    # at fetch time stays "dumb" -- it just downloads the model and generates a
    # template pbtxt with ovms defaults; the activation stage overrides values
    # in the sibling copy. Field names mirror LLMCalculatorOptions keys.
    max_num_seqs: int | None = Field(default=None, ge=1)
    enable_prefix_caching: bool | None = None
    # cache_size in GB; 0 means dynamic per OVMS convention.
    cache_size: int | None = Field(default=None, ge=0)
    dynamic_split_fuse: bool | None = None
    # KV cache precision is an OpenVINO plugin property, not an
    # LLMCalculatorOptions field; set it via plugin_config.KV_CACHE_PRECISION.

    draft_device: Device | None = None
    # Reference into ovms.yaml `repository:` keys. Resolved to a filesystem path
    # (draft_models_path in pbtxt) during apply, relative to the target's
    # graph.pbtxt directory.
    draft_model: str | None = None

    @model_validator(mode="after")
    def _draft_fields_are_paired(self) -> Graph:
        # draft_model and draft_device are a pair: speculative decoding needs
        # both a draft model to run and a device to run it on. Either declare
        # both or neither -- one without the other is always a mistake.
        has_model = self.draft_model is not None
        has_device = self.draft_device is not None
        if has_model != has_device:
            raise ValueError(
                "draft_model and draft_device must be set together "
                "(both define speculative decoding); "
                f"got draft_model={self.draft_model!r}, "
                f"draft_device={self.draft_device!r}"
            )
        return self


class ModelEntry(_Strict):
    source: str
    # Deployment knobs shared by every model type (mediapipe LLM graph and
    # plain model_config_list alike): the target device and OpenVINO device
    # properties. They live on the entry, not in `graph`, because OVMS applies
    # them regardless of how the model is registered.
    device: Device  # required; ovms cannot start without a target device
    # OpenVINO device properties forwarded to the device plugin. For the LLM
    # pipeline they are serialized into graph.pbtxt LLMCalculatorOptions.
    # Generic key/value bag -- we do not validate individual keys (CACHE_DIR,
    # PERFORMANCE_HINT, NUM_STREAMS, ...) since the set is plugin-defined.
    plugin_config: dict[str, str] | None = None
    # LLM/mediapipe pipeline tuning written into graph.pbtxt. Present only for
    # task-based (generative) models; absent for plain model_config_list models.
    graph: Graph | None = None
    # Plain model config options forwarded to model_config_list[].config. Present only
    # for plain (non-task) models. Generic key/value bag for OVMS model_config_list options
    # (batch_size, nireq, model_version_policy, etc.) -- we do not validate individual keys
    # since OVMS validates at load time.
    plain: dict[str, object] | None = None
    # Overrides merged into the model's generation_config.json at apply time.
    # Lives on the model entry (not the model identity) because it is a deployment-level
    # override, like device. Passthrough dict; values are not validated.
    generation: dict[str, int | float | bool | str | list[Any]] | None = None

    @model_validator(mode="after")
    def _plain_and_graph_are_exclusive(self) -> ModelEntry:
        # plain (model_config_list options) and graph (LLM pipeline tuning) are for
        # different model types and cannot coexist. Validation also happens in loader.py
        # but we enforce it here for schema consistency.
        if self.plain is not None and self.graph is not None:
            raise ValueError(
                "plain and graph are mutually exclusive "
                "(plain is for non-task models, graph is for task-based models)"
            )
        return self

    @property
    def draft_model(self) -> str | None:
        """Draft model reference, or None when no graph / no speculative decoding."""
        return self.graph.draft_model if self.graph is not None else None


class Profile(_Strict):
    models: list[str]
    active: bool = False


class OvmsConfig(_Strict):
    runtime: Runtime
    repository: dict[str, ModelIdentity]
    models: dict[str, ModelEntry] = Field(default_factory=dict)
    profiles: dict[str, Profile] = Field(default_factory=dict)


# ----- local.yaml -----

class LocalRuntime(_Strict):
    ovms_path: Path | None = None
    cache_dir: Path | None = None


class LocalModels(_Strict):
    # Required: every downstream stage (fetch, apply, start) needs a concrete
    # store path. Making it optional in schema and erroring later caused the
    # failure to surface mid-pipeline instead of at config load.
    repository_path: Path


class LocalConfig(_Strict):
    runtime: LocalRuntime = Field(default_factory=LocalRuntime)
    models: LocalModels
