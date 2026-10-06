#!/usr/bin/env python3
"""Publication rows for the BeyondArena transfer panel.

A missing receipt, or a receipt that asks for a median, stays the words
``not measured``. This module does not fit, sample, or average cells.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from select_beyondarena_panel import build_lock

NOT_MEASURED = "not measured"
METHODS = (
    "DOPE",
    "GaussianCopula",
    "Chow-Liu",
    "independent_marginals",
    "CTGAN",
    "TVAE",
    "ARF",
    "Forest-Flow",
    "TabSyn",
    "TabDDPM",
)
AUDITORS = ("catboost", "linear", "mlp")
NULL_CLAIMS = ("mfs_v2", "ptf_v1", "release_safe", "superiority")
MEDIAN_KEYS = ("median", "median_retention", "dope_median", "dope_median_retention")
ROWS_PATH = Path(__file__).resolve().parent / "results" / "beyondarena-retention-rows.json"


def _selected(lock: dict) -> dict[str, str]:
    return {row["name"]: row["task"] for row in lock["datasets"]}


def _unmeasured(lock: dict, receipt_status: str, accepted_cells: int) -> dict:
    tasks = sorted({row["task"] for row in lock["datasets"]})
    blocks = []
    for task in tasks:
        for method in METHODS:
            for auditor in AUDITORS:
                blocks.append(
                    {
                        "auditor": auditor,
                        "median_retention": NOT_MEASURED,
                        "method": method,
                        "n": None,
                        "task": task,
                        "wins_ties_losses": NOT_MEASURED,
                    }
                )
    return {
        "accepted_cells": accepted_cells,
        "blocks": blocks,
        "configuration": "features12_steps2048",
        "fold": "inner_validation",
        "format": "dope-beyondarena-retention-rows",
        "median_rule": "not measured until the manuscript generator reads a complete block",
        "mfs_v2": None,
        "official_pmlb_tests_opened": False,
        "pooled_with_s3_100": False,
        "ptf_v1": None,
        "receipt": receipt_status,
        "release_safe": None,
        "selected_count": lock["selected_count"],
        "superiority": None,
        "version": 1,
    }


def _reject_median(payload: object, where: str) -> None:
    if not isinstance(payload, dict):
        return
    for key in MEDIAN_KEYS:
        if key in payload:
            raise ValueError(f"{where} must not carry a median")


def validate_receipt(receipt: dict, lock: dict) -> int:
    """Return the number of accepted cells. Raise when the receipt overclaims."""
    if not isinstance(receipt, dict):
        raise ValueError("retention receipt must be an object")
    _reject_median(receipt, "retention receipt")
    pooled = receipt.get("pooled_with_s3_100") is True
    pooled = pooled or receipt.get("pooled_with_public_headline") is True
    if pooled:
        raise ValueError("BeyondArena retention must not be pooled")
    opened = receipt.get("official_pmlb_tests_opened") is True
    opened = opened or receipt.get("official_tests_opened") is True
    if opened:
        raise ValueError("PMLB official tests stay sealed")
    if receipt.get("fold") not in (None, "inner_validation"):
        raise ValueError("wave 1 fold must be inner_validation")
    if receipt.get("configuration") not in (None, "features12_steps2048"):
        raise ValueError("displayed configuration is features12_steps2048")
    for key in NULL_CLAIMS:
        if key in receipt and receipt[key] is not None:
            raise ValueError(f"{key} stays null")
    cells = receipt.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("cells must be a list")
    selected = _selected(lock)
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("cell must be an object")
        _reject_median(cell, "cell")
        name = cell.get("family")
        if name not in selected:
            raise ValueError("cell family is outside the lock")
        if cell.get("task") not in (None, selected[name]):
            raise ValueError("cell task does not match the lock")
        if cell.get("method") not in METHODS:
            raise ValueError("cell method is outside the publication list")
        if cell.get("auditor") not in AUDITORS:
            raise ValueError("cell auditor is outside the publication list")
        if cell.get("status") not in ("ok", "unavailable", "uninformative"):
            raise ValueError("cell status is not recognized")
        retention = cell.get("retention")
        if retention is not None and (
            isinstance(retention, bool)
            or not isinstance(retention, (int, float))
            or not math.isfinite(retention)
        ):
            raise ValueError("cell retention must be a finite number or null")
    return len(cells)


def publish(receipt: dict | None = None, lock: dict | None = None) -> dict:
    """Rows a manuscript may read. Medians stay the words not measured."""
    panel = lock if lock is not None else build_lock()
    if receipt is None:
        return _unmeasured(panel, "absent", 0)
    accepted = validate_receipt(receipt, panel)
    return _unmeasured(panel, "accepted", accepted)


def main() -> None:
    ROWS_PATH.write_text(json.dumps(publish(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
