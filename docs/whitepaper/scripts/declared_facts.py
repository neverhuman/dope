"""Counts and limits that renderers read from their declared sources instead of typing them.

The v2 interval level comes from the pre-declaration and the release byte cap from the
embedded release contract. Paths are relative to a repository root so fixtures can supply
their own copies; the module has no side effects and imports nothing outside the standard
library, so the claims renderer and the isolated paper build can both load it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PREDECLARE = "research/benchmark/review_fixes/predeclare_v2.json"
CONTRACT = "production/kpi-contract.json"


def read(name: str, root: Path = REPO) -> dict:
    return json.loads((root / name).read_text())


def interval_level(predeclare: dict) -> int:
    """Percent level of the pre-declared nested bootstrap interval."""
    match = re.search(r"percentile (\d+)% interval", predeclare["estimands"]["uncertainty"])
    if match is None:
        raise ValueError("predeclare_v2 does not state the interval level")
    return int(match.group(1))


def byte_cap(contract: dict) -> int:
    """The L3 release byte cap from the embedded release contract."""
    cap = contract["tier_byte_limits"]["l3"]
    if type(cap) is not int or cap <= 0:
        raise ValueError("release contract L3 byte cap is not a positive integer")
    return cap
