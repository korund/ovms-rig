"""Shared result type for status checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["ok", "warn", "error"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    summary: str
    details: dict = field(default_factory=dict)
    hint: str | None = None
