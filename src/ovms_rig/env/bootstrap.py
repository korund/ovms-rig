"""Build the OS environment for invoking the ovms binary.

OVMS ships with a self-contained runtime (own shared libs, optionally a
bundled Python interpreter). The binary needs PATH and LD_LIBRARY_PATH
pointed at the install root before it can locate its DLLs/sos and, in the
python-on variant, its bundled interpreter.

Install layout assumed (verified empirically + per upstream
docs/deploying_server_baremetal.md):

    win32 python-on:  <ovms>/, <ovms>/python/, <ovms>/python/Scripts/
    win32 python-off: <ovms>/
    linux python-on:  <ovms>/bin/, <ovms>/lib/, <ovms>/lib/python/
    linux python-off: <ovms>/bin/, <ovms>/lib/

The python-on variant is detected by the presence of the bundled-python
directory. Detection is filesystem-driven so the user never declares the
variant in config (which would create a source of drift).

Prepend semantics: PATH and LD_LIBRARY_PATH are *prepended*, not appended,
so OVMS's bundled shared libs take precedence over any same-named system
libs (e.g. libcurl).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PATHLIKE_KEYS = frozenset({"PATH", "LD_LIBRARY_PATH"})


def build_env(ovms_path: Path) -> dict[str, str]:
    """Return a full env dict (current os.environ + ovms overrides)."""
    env = os.environ.copy()
    for key, value in _overrides(ovms_path).items():
        if key in _PATHLIKE_KEYS:
            env[key] = _prepend(env.get(key), value)
        else:
            env[key] = value
    return env


def _overrides(ovms: Path) -> dict[str, str]:
    if sys.platform == "win32":
        return _win32(ovms)
    return _linux(ovms)


def _win32(ovms: Path) -> dict[str, str]:
    bundled_py = ovms / "python"
    if bundled_py.is_dir():
        return {
            "PATH": os.pathsep.join(
                [str(ovms), str(bundled_py), str(bundled_py / "Scripts")]
            ),
            "PYTHONHOME": str(bundled_py),
        }
    return {"PATH": str(ovms)}


def _linux(ovms: Path) -> dict[str, str]:
    bin_dir = ovms / "bin"
    lib_dir = ovms / "lib"
    bundled_py = lib_dir / "python"
    overrides = {
        "PATH": str(bin_dir),
        "LD_LIBRARY_PATH": str(lib_dir),
    }
    if bundled_py.is_dir():
        overrides["PYTHONPATH"] = str(bundled_py)
    return overrides


def _prepend(existing: str | None, addition: str) -> str:
    if not existing:
        return addition
    return f"{addition}{os.pathsep}{existing}"
