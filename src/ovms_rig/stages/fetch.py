"""Pull missing HF models into the store.

Idempotency: presence is decided by directory existence at the HF path
under the store (the layout `ovms --pull` produces). A model whose HF
directory is already present is skipped -- not re-pulled.

Layout policy: each declared model is pulled by its own `ovms --pull`
call, without `--draft_source_model`. Target and draft end up as siblings
under `<store>/<hf_org>/<hf_repo>/`. The target->draft binding is wired
later in apply (patching `draft_models_path` in the target's pbtxt).

Pull-bucket fields from `model_entry.graph` are forwarded as CLI flags only
when pulling a model that is a *source* of a model entry. Draft-only
models pull with no graph customization.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.config import (
    ConfigError,
    OvmsConfig,
    load_local,
    load_ovms,
)
from ovms_rig.config.schema import ModelEntry, ModelIdentity
from ovms_rig.env import build_env
from ovms_rig.probes import ovms_binary

logger = logging.getLogger(__name__)


def run(ctx: dict) -> int:
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    ovms_override = Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None
    cli_level: str | None = ctx.get("log_level")
    extras: list[str] = list(ctx.get("extras") or [])

    logging_setup.configure((cli_level or "INFO").upper())

    try:
        ovms = load_ovms(config_path)
        local = load_local(local_path)
    except ConfigError as e:
        logger.error("config load failed: %s", e)
        return 1

    level = (cli_level or ovms.runtime.log_level).upper()
    logging_setup.configure(level)

    binary, _src = ovms_binary.resolve(ovms_override, local)
    if binary is None or not binary.is_file():
        logger.error("ovms binary not resolved; run `ovms-rig status` for details")
        return 1

    store = local.models.repository_path
    store.mkdir(parents=True, exist_ok=True)

    targets = _targets_by_model(ovms)
    env = build_env(binary.parent)

    failures: list[str] = []
    for name in sorted(ovms.repository):
        model = ovms.repository[name]
        dest = store / model.hf
        if dest.is_dir():
            logger.info("[skip] %s (already present at %s)", name, dest)
            continue
        model_entry = targets.get(name)
        rc = _pull_one(binary, env, store, name, model, model_entry, extras)
        if rc != 0:
            failures.append(name)

    if failures:
        logger.error("fetch failed for: %s", ", ".join(failures))
        return 1
    return 0


def _targets_by_model(ovms: OvmsConfig) -> dict[str, ModelEntry]:
    return {entry.source: entry for name, entry in ovms.models.items()}


def _pull_one(
    binary: Path,
    env: dict[str, str],
    store: Path,
    name: str,
    model: ModelIdentity,
    model_entry: ModelEntry | None,
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
            "[pull] %s: revision '%s' declared but `ovms --pull` does not "
            "support pinning; pulling latest from HF main",
            name, model.revision,
        )
    if model_entry is not None:
        for key, value in model_entry.graph.pull_flags().items():
            args += [f"--{key}", _format(value)]
    args += extras

    logger.info("[pull] %s (%s)", name, model.hf)
    logger.debug("       cmd: %s", " ".join(args))
    proc = subprocess.run(args, env=env, check=False)
    if proc.returncode != 0:
        logger.error("[pull] %s failed with exit code %d", name, proc.returncode)
    return proc.returncode


def _format(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
