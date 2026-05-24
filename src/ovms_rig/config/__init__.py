"""Declaration subsystem: schema + loader for ovms.yaml and local.yaml.

The package re-exports the public API so callers can write
`from ovms_rig.config import OvmsConfig, load_ovms` without caring whether a
given symbol lives in `schema` or `loader`.
"""

from ovms_rig.config.loader import ConfigError, load_declaration, load_local, load_ovms
from ovms_rig.config.schema import (
    LocalConfig,
    LocalModels,
    LocalRuntime,
    LogLevel,
    ModelEntry,
    ModelIdentity,
    OvmsConfig,
    Profile,
    Runtime,
)

__all__ = [
    "ConfigError",
    "LocalConfig",
    "LocalModels",
    "LocalRuntime",
    "LogLevel",
    "ModelEntry",
    "ModelIdentity",
    "OvmsConfig",
    "Profile",
    "Runtime",
    "load_declaration",
    "load_local",
    "load_ovms",
]
