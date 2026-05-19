"""Load and validate ovms.yaml + local.yaml, resolve internal references."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from ovms_rig.config.schema import LocalConfig, OvmsConfig


class ConfigError(Exception):
    """Raised when a declaration file is unreadable, malformed, or inconsistent."""


def load_ovms(path: Path) -> OvmsConfig:
    raw = _read_yaml(path)
    try:
        cfg = OvmsConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"{path}: schema validation failed\n{e}") from e
    _check_references(cfg, source=path)
    return cfg


def load_local(path: Path) -> LocalConfig:
    raw = _read_yaml(path) or {}
    try:
        return LocalConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"{path}: schema validation failed\n{e}") from e


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"{path}: file not found")
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e


def _check_references(cfg: OvmsConfig, source: Path) -> None:
    known = set(cfg.models)
    targets: dict[str, str] = {}
    for entry in cfg.served:
        if entry.model not in known:
            raise ConfigError(
                f"{source}: served entry '{entry.name}' references unknown model "
                f"'{entry.model}' (declared models: {sorted(known)})"
            )
        # One model -> one target. The pull-bucket fields under served.graph
        # bake into the model's pbtxt; multiple served entries targeting the
        # same model would race over a single file.
        if entry.model in targets:
            raise ConfigError(
                f"{source}: model '{entry.model}' is target of both "
                f"'{targets[entry.model]}' and '{entry.name}'; "
                "one model can serve at most one entry"
            )
        targets[entry.model] = entry.name
        draft = entry.graph.draft_model
        if draft is not None and draft not in known:
            raise ConfigError(
                f"{source}: served entry '{entry.name}' references unknown "
                f"draft_model '{draft}' (declared models: {sorted(known)})"
            )
