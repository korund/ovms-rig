"""Top-level entry point for the start stage."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.config import ConfigError, load_local, load_ovms
from ovms_rig.env import build_env
from ovms_rig.probes import ovms_binary
from ovms_rig.stages.start.precheck import run as precheck_run
from ovms_rig.stages.start.command import build as build_command
from ovms_rig.stages.start.signals import install as install_signals

logger = logging.getLogger(__name__)


def run(ctx: dict) -> int:
    """Precheck, build env, launch ovms, forward signals, return its exit code."""
    cli_level: str | None = ctx.get("log_level")
    logging_setup.configure((cli_level or "INFO").upper())

    rc = precheck_run(ctx)
    if rc != 0:
        return rc

    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    ovms_override = Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None
    extras: list[str] = list(ctx.get("extras") or [])

    try:
        ovms_cfg = load_ovms(config_path)
        local = load_local(local_path)
    except ConfigError as e:
        logger.error("config load failed: %s", e)
        return 1

    level = (cli_level or ovms_cfg.runtime.log_level).upper()
    logging_setup.configure(level)

    binary, _src = ovms_binary.resolve(ovms_override, local)
    if binary is None or not binary.is_file():
        logger.error("ovms binary not resolved; run `ovms-rig status` for details")
        return 1

    store = local.models.repository_path
    config_json = store / "config.json"
    env = build_env(binary.parent)

    # ovms silently ignores --cache_dir if the directory does not exist, which
    # makes misconfiguration invisible. Materialize it eagerly when declared
    # and not overridden via extras.
    extras_flags = {tok for tok in extras if tok.startswith("--")}
    cache_dir = local.runtime.cache_dir
    if cache_dir is not None and "--cache_dir" not in extras_flags:
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("created cache directory: %s", cache_dir)

    cmd = build_command(binary, config_json, ovms_cfg.runtime, local.runtime, extras)

    logger.info("starting ovms: %s", " ".join(cmd))

    # On win32 create a new process group so we can send CTRL_BREAK_EVENT.
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(cmd, env=env, **kwargs)
    install_signals(proc)

    proc.wait()
    return proc.returncode
