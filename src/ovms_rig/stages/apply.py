"""Patch live graph.pbtxt and config.json from the declaration; back up first.

With dry_run=True, write the proposed files to build/ and stop. Live files
are not touched and no backup is taken.
"""

from __future__ import annotations


def run(ctx: dict) -> int:
    raise NotImplementedError("apply stage not implemented yet")
