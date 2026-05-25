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
KvCachePrecision = Literal["u8", "f16"]
Device = Literal["CPU", "GPU", "NPU"]
# Task taxonomy as accepted by `ovms --pull --task`. Required per model
# since pull cannot infer it from the HF repo.
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
    hf: str
    task: Task
    revision: str | None = None


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
    kv_cache_precision: KvCachePrecision | None = None

    device: Device  # required; ovms cannot start without a target device
    draft_device: Device | None = None
    # Reference into ovms.yaml `repository:` keys. Resolved to a filesystem path
    # (draft_models_path in pbtxt) during apply, relative to the target's
    # graph.pbtxt directory.
    draft_model: str | None = None
    # OpenVINO device properties forwarded to the LLM pipeline via
    # LLMCalculatorOptions.plugin_config (serialized as JSON in graph.pbtxt).
    # Generic key/value bag -- we do not validate individual keys (CACHE_DIR,
    # PERFORMANCE_HINT, NUM_STREAMS, ...) since the set is plugin-defined.
    plugin_config: dict[str, str] | None = None

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
    graph: Graph
    # Overrides merged into the model's generation_config.json at apply time.
    # Lives on the model entry (not the model identity) because it is a deployment-level
    # override, like graph.device. Passthrough dict; values are not validated.
    generation: dict[str, int | float | bool | str | list[Any]] | None = None


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
