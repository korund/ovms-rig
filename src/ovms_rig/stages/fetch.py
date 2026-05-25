"""Pull a single repository entry into the store.

Idempotency: presence is decided by directory existence at the HF path
under the store (the layout `ovms --pull` produces). If the directory
already exists, the pull is skipped.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.config import (
    ConfigError,
    ModelIdentity,
    load_declaration,
)
from ovms_rig.env import build_env
from ovms_rig.probes import ovms_binary

logger = logging.getLogger(__name__)


def run(ctx: dict) -> int:
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    repository_name: str = ctx.get("repository_name", "")
    ovms_override = Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None
    cli_level: str | None = ctx.get("log_level")
    extras: list[str] = list(ctx.get("extras") or [])

    logging_setup.configure((cli_level or "INFO").upper())

    try:
        decl = load_declaration(config_path, local_path, cli_override=ovms_override)
        ovms = decl.ovms
        local = decl.local
    except ConfigError as e:
        logger.error("config load failed: %s", e)
        return 1

    level = (cli_level or ovms.runtime.log_level).upper()
    logging_setup.configure(level)

    # Check that repository_name exists in ovms.repository.
    if repository_name not in ovms.repository:
        logger.error("repository entry '%s' not found (available: %s)",
                     repository_name, sorted(ovms.repository.keys()))
        return 1

    binary, source, error = ovms_binary.resolve_validated(ovms_override, local)
    if error is not None:
        logger.error("ovms binary %s (source: %s); run `ovms-rig status` for details", error, source)
        return 1

    store = local.models.repository_path
    store.mkdir(parents=True, exist_ok=True)

    model_identity = ovms.repository[repository_name]
    dest = store / model_identity.hf

    # Idempotent: if already present, skip.
    if dest.is_dir():
        logger.info("[skip] '%s' already present at %s", repository_name, dest)
        return 0

    env = build_env(binary.parent)
    rc = _pull_one(binary, env, store, repository_name, model_identity, extras)
    return rc


def _pull_one(
    binary: Path,
    env: dict[str, str],
    store: Path,
    name: str,
    model: ModelIdentity,
    extras: list[str],
) -> int:
    args: list[str] = [
        str(binary),
        "--pull",
        "--source_model", model.hf,
        "--model_repository_path", str(store),
        "--task", model.task,
    ]
    if model.revision is not None:
        # ovms --pull (as of 2026.1.0) has no revision flag; HF_HUB git ref
        # cannot be expressed. Honor the declaration as documentation only.
        logger.warning(
            "[pull] '%s': revision '%s' declared but `ovms --pull` does not "
            "support pinning; pulling latest from HF main",
            name, model.revision,
        )
    args += extras

    logger.info("[pull] '%s' (%s)", name, model.hf)
    logger.debug("       cmd: %s", " ".join(args))
    proc = subprocess.run(args, env=env, check=False)
    if proc.returncode != 0:
        logger.error("[pull] '%s' failed with exit code %d", name, proc.returncode)
    return proc.returncode
