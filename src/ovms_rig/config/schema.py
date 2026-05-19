"""Typed schema for the two declaration files (ovms.yaml + local.yaml).

This module only defines the shape; loading and reference resolution land
in the loader (next step). Schema is intentionally strict (extra="forbid")
so typos in YAML surface immediately as validation errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----- ovms.yaml -----

class Runtime(_Strict):
    ovms_version: str
    rest_port: int = Field(ge=1, le=65535)
    log_level: LogLevel = "INFO"


class ModelIdentity(_Strict):
    hf: str
    revision: str | None = None


class ServedEntry(_Strict):
    name: str
    model: str
    # Pass-through bag for LLMCalculatorOptions; the loader does not interpret
    # individual fields. Validation of contents is OVMS's job at start time.
    graph: dict[str, Any] = Field(default_factory=dict)


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
