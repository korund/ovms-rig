"""Typed schema for the two declaration files (ovms.yaml + local.yaml).

This module only defines the shape; loading and reference resolution land
in the loader (next step). Schema is intentionally strict (extra="forbid")
so typos in YAML surface immediately as validation errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    ovms_version: str
    rest_port: int = Field(ge=1, le=65535)
    log_level: LogLevel = "INFO"


class ModelIdentity(_Strict):
    hf: str
    task: Task
    revision: str | None = None


# Fields of `graph:` split by which stage consumes them.
# Pull-bucket: passed as `--<name> <value>` to `ovms --pull` at fetch time.
# Pbtxt-bucket: not pull flags; patched into graph.pbtxt at apply time.
# Field names mirror ovms flag names verbatim (snake_case).
GRAPH_PULL_FIELDS = frozenset({
    "max_num_seqs",
    "enable_prefix_caching",
    "cache_size",
    "dynamic_split_fuse",
    "kv_cache_precision",
})
GRAPH_PBTXT_FIELDS = frozenset({
    "device",
    "draft_device",
    "draft_model",
})


class Graph(_Strict):
    # Pull-bucket: forwarded to `ovms --pull` as CLI flags.
    max_num_seqs: int | None = Field(default=None, ge=1)
    enable_prefix_caching: bool | None = None
    # cache_size in GB; 0 means dynamic per OVMS convention.
    cache_size: int | None = Field(default=None, ge=0)
    dynamic_split_fuse: bool | None = None
    kv_cache_precision: KvCachePrecision | None = None

    # Pbtxt-only: patched into graph.pbtxt during apply.
    device: Device  # required; ovms cannot start without a target device
    draft_device: Device | None = None
    # Reference into ovms.yaml `models:` keys. Resolved to a filesystem path
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

    def pull_flags(self) -> dict[str, object]:
        return {name: getattr(self, name)
                for name in GRAPH_PULL_FIELDS
                if getattr(self, name) is not None}


class ServedEntry(_Strict):
    name: str
    model: str
    graph: Graph


class OvmsConfig(_Strict):
    runtime: Runtime
    models: dict[str, ModelIdentity]
    served: list[ServedEntry]


# ----- local.yaml -----

class LocalRuntime(_Strict):
    ovms_path: Path | None = None
    cache_dir: Path | None = None


class LocalModels(_Strict):
    repository_path: Path | None = None


class LocalConfig(_Strict):
    runtime: LocalRuntime = Field(default_factory=LocalRuntime)
    models: LocalModels = Field(default_factory=LocalModels)
