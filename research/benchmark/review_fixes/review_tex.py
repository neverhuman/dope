"""Small TeX helpers for review tables: a row-key guard."""

from __future__ import annotations


def assert_distinct(keys, label: str) -> None:
    """Refuse a table in which two printed rows carry the same identifying cells."""
    seen = set()
    for key in keys:
        if key in seen:
            raise ValueError(f"{label} prints two rows with the same identity: {key}")
        seen.add(key)
