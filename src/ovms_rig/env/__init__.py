"""OS environment composition for invoking the ovms binary.

Used by fetch (for `ovms --pull`) and start (for `exec ovms`). Knows the
ovms install layout per platform and per variant (python-on/off); detects
variant from the filesystem, never asks for it explicitly.
"""

from ovms_rig.env.bootstrap import build_env

__all__ = ["build_env"]
