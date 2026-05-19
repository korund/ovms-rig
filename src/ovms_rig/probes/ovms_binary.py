"""Resolve the ovms binary path by a fixed 4-step priority."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ovms_rig.config import LocalConfig
from ovms_rig.report import CheckResult

NAME = "ovms binary"


def check(cli_override: Path | None, local: LocalConfig) -> CheckResult:
    resolved, source = resolve(cli_override, local)
    if resolved is None:
        return CheckResult(
            name=NAME,
            status="error",
            summary="ovms binary not found",
            hint=(
                "pass --ovms-path, set runtime.ovms_path in local.yaml, "
                "or add ovms to PATH"
            ),
        )
    if not resolved.is_file():
        return CheckResult(
            name=NAME,
            status="error",
            summary=f"resolved path is not a file: {resolved}",
            details={"source": source},
        )
    if not os.access(resolved, os.X_OK):
        return CheckResult(
            name=NAME,
            status="error",
            summary=f"resolved binary is not executable: {resolved}",
            details={"source": source},
        )
    return CheckResult(
        name=NAME,
        status="ok",
        summary=str(resolved),
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
