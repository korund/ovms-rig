"""Resolve the ovms binary path by a fixed 4-step priority."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ovms_rig.config import LocalConfig
from ovms_rig.report import CheckResult

NAME = "ovms binary"


def resolve_validated(cli_override: Path | None, local: LocalConfig) -> tuple[Path | None, str, str | None]:
    """Resolve ovms binary and validate. Returns (path, source, error).

    path is whatever resolve() returned (Path or None if not found).
    error is None on success, otherwise a short reason string.
    """
    resolved, source = resolve(cli_override, local)

    if resolved is None:
        return None, source, "no path resolved"
    if not resolved.is_file():
        return resolved, source, "not a file"
    if not os.access(resolved, os.X_OK):
        return resolved, source, "not executable"

    return resolved, source, None


def check(cli_override: Path | None, local: LocalConfig) -> CheckResult:
    path, source, error = resolve_validated(cli_override, local)

    if error == "no path resolved":
        return CheckResult(
            name=NAME,
            status="error",
            summary="ovms binary not found",
            hint=(
                "pass --ovms-path, set runtime.ovms_path in local.yaml, "
                "or add ovms to PATH"
            ),
        )
    if error == "not a file":
        return CheckResult(
            name=NAME,
            status="error",
            summary=f"resolved path is not a file: {path}",
            details={"source": source},
        )
    if error == "not executable":
        return CheckResult(
            name=NAME,
            status="error",
            summary=f"resolved binary is not executable: {path}",
            details={"source": source},
        )

    return CheckResult(
        name=NAME,
        status="ok",
        summary=str(path),
        details={"source": source},
    )


def resolve(cli_override: Path | None, local: LocalConfig) -> tuple[Path | None, str]:
    if cli_override is not None:
        return cli_override, "cli"
    if local.runtime.ovms_path is not None:
        return local.runtime.ovms_path, "local.yaml"
    found = shutil.which("ovms")
    if found:
        return Path(found), "PATH"
    return None, "none"
