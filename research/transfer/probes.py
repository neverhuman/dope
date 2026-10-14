"""Risk-sketch status. An absent or failed sketch is not a passing probe."""

from __future__ import annotations


def sketch_status(ok: bool | None) -> str:
    if ok is None:
        return "absent"
    if ok is True:
        return "ok"
    if ok is False:
        return "failed"
    raise ValueError("invalid sketch")
