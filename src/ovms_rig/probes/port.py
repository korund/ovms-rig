"""Check whether the declared REST port can be bound on localhost."""

from __future__ import annotations

import socket

from ovms_rig.config import Declaration
from ovms_rig.report import CheckResult

NAME = "rest port"


def check(decl: Declaration) -> CheckResult:
    port = decl.ovms.runtime.rest_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            s.bind(("127.0.0.1", port))
        except OSError as e:
            return CheckResult(
                name=NAME,
                status="error",
                summary=f"port {port} is busy",
                details={"port": port, "errno": e.errno},
            )
    return CheckResult(name=NAME, status="ok", summary=f"port {port} is free")
