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
    _check_profiles(cfg, source=path)
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
    known = set(cfg.repository)
    targets: dict[str, str] = {}
    for name, entry in cfg.models.items():
        if entry.source not in known:
            raise ConfigError(
                f"{source}: model '{name}' references unknown source "
                f"'{entry.source}' (declared models: {sorted(known)})"
            )
        # One model -> one target. The pull-bucket fields under model.graph
        # bake into the model's pbtxt; multiple model entries targeting the
        # same source would race over a single file.
        if entry.source in targets:
            raise ConfigError(
                f"{source}: source '{entry.source}' is target of both "
                f"'{targets[entry.source]}' and '{name}'; "
                "one source can be target of at most one model entry"
            )
        targets[entry.source] = name
        draft = entry.graph.draft_model
        if draft is not None and draft not in known:
            raise ConfigError(
                f"{source}: model '{name}' references unknown "
                f"draft_model '{draft}' (declared models: {sorted(known)})"
            )


def _check_profiles(cfg: OvmsConfig, source: Path) -> None:
    known_models = set(cfg.models)

    # Check that all profile.models references exist in cfg.models.
    for profile_name, profile in cfg.profiles.items():
        for model_name in profile.models:
            if model_name not in known_models:
                raise ConfigError(
                    f"{source}: profile '{profile_name}' references unknown model "
                    f"'{model_name}' (declared models: {sorted(known_models)})"
                )

    # Check that at most one profile is active.
    active_profiles = [name for name, profile in cfg.profiles.items() if profile.active]
    if len(active_profiles) > 1:
        raise ConfigError(
            f"{source}: at most one profile can be active, got {len(active_profiles)}: "
            f"{active_profiles}"
        )
