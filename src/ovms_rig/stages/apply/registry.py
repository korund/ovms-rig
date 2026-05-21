"""Wrap `ovms --add_to_config` for registering served endpoints.

Empirical findings (tested against OVMS 2026.1.0.72cc0624):
- Re-running `ovms --add_to_config` for an already-registered model name
  is a NO-OP: ovms returns exit code 0 and does not modify config.json.
  The entry is identified by the `name` field inside config.json's `model_config_list`.
- `ovms --add_to_config --config_path <path>` works: ovms reads and writes
  the specified config file regardless of its location. This is the mechanism
  used for dry-run: we point it at build/config.json instead of the live file.
- When config.json does not yet exist at the given path, ovms creates it.
- There is no error or warning on re-add; idempotency is baked in.

Therefore: no pre-check needed for existing entries. Just call --add_to_config
and let ovms handle it.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def add_to_config(
    binary: Path,
    env: dict[str, str],
    config_path: Path,
    model_name: str,
    model_path: Path,
    extras: list[str],
) -> int:
    """Call `ovms --add_to_config` to register model_name -> model_path.

    config_path: the config.json to create/update (may be in build/ for dry-run).
    Returns the subprocess exit code.
    """
    args = [
        str(binary),
        "--add_to_config",
        str(config_path),
        "--model_name", model_name,
        "--model_path", str(model_path),
    ] + extras

    logger.debug("[registry] cmd: %s", " ".join(args))
    proc = subprocess.run(args, env=env, check=False, capture_output=True, text=True)
    if proc.stdout:
        logger.debug("[registry] stdout: %s", proc.stdout.rstrip())
    if proc.stderr:
        logger.debug("[registry] stderr: %s", proc.stderr.rstrip())
    if proc.returncode != 0:
        logger.error(
            "[registry] ovms --add_to_config failed (rc=%d) for model '%s'",
            proc.returncode, model_name,
        )
    return proc.returncode


def is_registered(config_path: Path, model_name: str) -> bool:
    """Return True if model_name already appears in config.json.

    Used only for logging; ovms handles idempotency natively.
    """
    if not config_path.exists():
        return False
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        entries = data.get("model_config_list", [])
        return any(
            e.get("config", {}).get("name") == model_name for e in entries
        )
    except (json.JSONDecodeError, KeyError):
        return False
