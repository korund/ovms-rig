"""Load and validate ovms.yaml + local.yaml, resolve internal references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from ovms_rig.config.schema import LocalConfig, LocalModels, LocalRuntime, OvmsConfig


@dataclass(frozen=True)
class Declaration:
    """Unified value object for all loaded configuration.

    Contains:
    - ovms: the primary declaration from ovms.yaml
    - local: per-host overrides from local.yaml
    - cli_override: optional CLI override for the ovms binary path
    """
    ovms: OvmsConfig
    local: LocalConfig
    cli_override: Path | None = None


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
    raw = _read_yaml(path)
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
        # One model -> one target. OVMS reads generation_config.json by fixed name
        # and does not support override via config.json. Multiple model entries
        # targeting the same source would conflict over a single shared config file.
        # This limitation may be lifted once OVMS adds generation_config override support.
        if entry.source in targets:
            raise ConfigError(
                f"{source}: source '{entry.source}' is target of both "
                f"'{targets[entry.source]}' and '{name}'; "
                "one source can be target of at most one model entry "
                "(OVMS limitation: generation_config.json is not overridable per entry)"
            )
        targets[entry.source] = name
        # graph (mediapipe LLM tuning) only applies to task-based models. A
        # plain source with a graph block is a declaration mistake.
        if cfg.repository[entry.source].task is None and entry.graph is not None:
            raise ConfigError(
                f"{source}: model '{name}' declares a graph block but its source "
                f"'{entry.source}' has no task; graph fields apply only to "
                "task-based (generative) models"
            )
        # plain (model_config_list options) only applies to non-task models.
        # task-based sources should not declare plain.
        if cfg.repository[entry.source].task is not None and entry.plain is not None:
            raise ConfigError(
                f"{source}: model '{name}' declares a plain block but its source "
                f"'{entry.source}' has a task; plain options apply only to "
                "non-task (plain) models"
            )
        draft = entry.draft_model
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


def load_declaration(
    config_path: Path,
    local_path: Path,
    cli_override: Path | None = None,
) -> Declaration:
    """Load and parse both declaration files, returning unified Declaration.

    Args:
        config_path: path to ovms.yaml
        local_path: path to local.yaml (missing file is OK, defaults are used)
        cli_override: optional CLI override for ovms binary path

    Returns:
        Declaration value object with ovms, local, and cli_override

    Raises:
        ConfigError: if either file is invalid or references cannot be resolved
    """
    ovms = load_ovms(config_path)
    # load_local requires file to exist; handle missing file gracefully here
    if local_path.exists():
        local = load_local(local_path)
    else:
        # File missing: use defaults (require repository_path with sensible default)
        try:
            local = LocalConfig(
                runtime=LocalRuntime(),
                models=LocalModels(repository_path=Path("models")),
            )
        except ValidationError as e:
            raise ConfigError(f"{local_path}: schema validation failed\n{e}") from e
    return Declaration(ovms=ovms, local=local, cli_override=cli_override)
