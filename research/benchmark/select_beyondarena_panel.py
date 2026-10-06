#!/usr/bin/env python3
"""Freeze the BeyondArena evaluation cohort.

The version-2 lock kept 12 families. This selector keeps every binary or
regression family with a recorded redistribution license, and drops a family
already in the PMLB 100. It does not fit, sample, or open an official test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "research" / "benchmark"
INVENTORY = BENCHMARK / "results" / "beyondarena-s3-inventory.json"
LINEAGE_RECORD = BENCHMARK / "results" / "s3-lineage-record.json"
LOCK_PATH = BENCHMARK / "results" / "beyondarena-panel.lock.json"

TASKS = ("binary_classification", "regression")
HISTORICAL_PILOT = (
    "airfoil_self_noise",
    "early_learning_predictors",
    "lending_club",
    "heart_disease_va_long_beach",
    "musk",
    "kick",
    "qsar_biodeg",
    "telemonitoring_parkinsons_biomedical_voice_measurements",
    "hotel_booking_demand",
    "wine_world_cost",
    "video_transcoding_time_prediction",
    "garments_worker_productivity",
)
DATASET_FIELDS = (
    "name",
    "source_family_id",
    "task",
    "stratum",
    "split_regime",
    "rows",
    "columns",
    "license_spdx",
    "objective_metric",
    "target",
    "parquet_key",
    "parquet_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def pmlb_display_names(record: dict | None = None) -> set[str]:
    """Names of the prepared PMLB lineages. The record must contain 100."""
    payload = record if record is not None else _load(LINEAGE_RECORD)
    names = {row["display_name"] for row in payload["rows"]}
    if len(names) != 100:
        raise ValueError(f"PMLB display names must be 100, found {len(names)}")
    return names


def classify(inventory: dict, pmlb_names: set[str]) -> dict:
    """Split the bucket into the eligible cohort and the two exclusion counts."""
    families = inventory["datasets"]
    if len(families) != inventory["dataset_count"]:
        raise ValueError("inventory dataset_count does not match the family list")
    eligible = []
    rights = 0
    other_task = 0
    for family in families:
        if not family.get("license_spdx"):
            rights += 1
        if family.get("task") not in TASKS:
            other_task += 1
        if family.get("task") not in TASKS or not family.get("license_spdx"):
            continue
        if family["name"] in pmlb_names:
            continue
        eligible.append({field: family[field] for field in DATASET_FIELDS})
    eligible.sort(key=lambda row: row["name"])
    return {
        "eligible": eligible,
        "exclusion_reason_counts": {
            "redistribution_rights": rights,
            "task_not_binary_or_regression": other_task,
        },
    }


def build_lock(inventory: dict | None = None, pmlb_names: set[str] | None = None) -> dict:
    inventory = inventory if inventory is not None else _load(INVENTORY)
    names = pmlb_names if pmlb_names is not None else pmlb_display_names()
    split = classify(inventory, names)
    eligible = split["eligible"]
    return {
        "format": "dope-beyondarena-panel-lock",
        "version": 3,
        "bucket": inventory["bucket"],
        "revision": inventory["revision"],
        "inventory": "research/benchmark/results/beyondarena-s3-inventory.json",
        "inventory_sha256": _sha256(INVENTORY),
        "bucket_family_count": inventory["dataset_count"],
        "eligible_count": len(eligible),
        "selected_count": len(eligible),
        "maximum_source_families": len(eligible),
        "exclusion_reason_counts": split["exclusion_reason_counts"],
        "selection_rule": (
            "binary or regression, recorded redistribution license, "
            "not already one of the 100 PMLB lineages; every eligible family; "
            "the version-2 cap of 12 is the historical prepare only"
        ),
        "historical_pilot_families": list(HISTORICAL_PILOT),
        "historical_pilot_cap": 12,
        "displayed_configuration": "features12_steps2048",
        "fit_seed": 11,
        "sample_seeds": [101, 211, 307],
        "size_multiplier": 4,
        "inner_split": "official_training_fold_0_repeat_0_grouped_or_stratified_80_20",
        "inner_seed": 1729,
        "wave": 1,
        "wave_1_fold": "inner_validation",
        "outer_fold_authorized": False,
        "frozen_before_beyondarena_fit": True,
        "official_tests_opened": False,
        "pooled_with_s3_100": False,
        "pooled_with_public_headline": False,
        "mfs_v2": None,
        "datasets": eligible,
    }


def main() -> None:
    lock = build_lock()
    LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
