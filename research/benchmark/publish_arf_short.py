"""Publish the completed ARF short-grid native-validation evidence.

This is a grid readout, not the eight-trial ARF selection. The parent round
must finish before the final native-tuned configuration can be selected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .arf_round import METHOD_LOCK, DATASET_LOCK, SOURCE, ROOT, source_files
from .arf_short_source import SOURCE_SHA256, materialize
from .manifest import digest
from .score import artifact_inventory, sha256


ROUND = ROOT / "arf-native-short-v1/round.lock.json"
RESULTS = Path(__file__).with_name("results")
ARTIFACT_FILES = ["model.json", "bounds.csv", "continuous.csv",
                  "categories.csv", "projection.json"]


def build(lock_path: Path = ROUND) -> dict:
    source = Path(__file__).parents[2] / "target/arf_short_round.py"
    materialize(source)
    lock = json.loads(lock_path.read_text())
    parent = ROOT / "arf-native-v1/round.lock.json"
    parent_lock = json.loads(parent.read_text())
    parent_first = (parent.parent / "jobs" / digest(parent_lock["jobs"][0])
                    / "attempt-0001/receipt.json")
    if (lock.get("format") != "dope-arf-native-validation-round"
            or lock.get("round_source_sha256") != SOURCE_SHA256
            or lock.get("round_source_sha256") != sha256(source)
            or lock.get("adapter_source_sha256") != sha256(
                Path(__file__).with_name("arf_native.py"))
            or lock.get("source_archive_sha256") != sha256(SOURCE)
            or lock.get("source_files") != source_files()
            or lock.get("dataset_lock_sha256") != sha256(DATASET_LOCK)
            or lock.get("method_lock_sha256") != sha256(METHOD_LOCK)
            or lock.get("parent_round_sha256") != sha256(parent)
            or lock.get("parent_timeout_receipt_sha256") != sha256(parent_first)
            or json.loads(parent_first.read_text())["status"] != "timeout"
            or lock.get("cumulative_tuning_trials_per_dataset") != 8
            or lock.get("tuning_trials_per_dataset") != 4
            or lock.get("native_objective", {}).get("name")
            != "heldout_forde_mean_log_density"
            or lock["native_objective"].get("direction") != "maximize"
            or lock.get("official_tests_opened") is not False
            or lock.get("mfs_v2") is not None or lock.get("ptf_v1") is not None
            or len(lock.get("jobs", [])) != 12):
        raise ValueError("frozen ARF refinement lineage changed")
    datasets = {"Adult": [], "California": [], "News": []}
    for job in lock["jobs"]:
        dataset = job["dataset"]
        if dataset not in datasets or job["trial"] not in range(4) \
                or job["config"] != lock["config_grid"][job["trial"]]:
            raise ValueError("ARF short-grid job differs from frozen matrix")
        attempt = lock_path.parent / "jobs" / digest(job) / "attempt-0001"
        path = attempt / "receipt.json"
        receipt = json.loads(path.read_text())
        if (receipt.get("format") != "dope-arf-native-attempt"
                or receipt.get("job") != job
                or receipt.get("round_sha256") != sha256(lock_path)
                or receipt.get("status") not in
                ("ok", "failed", "timeout", "invalid_evidence")
                or receipt.get("official_tests_opened") is not False
                or receipt.get("mfs_v2") is not None
                or receipt.get("ptf_v1") is not None):
            raise ValueError("ARF short-grid attempt identity changed")
        for name, expected in receipt["evidence_files"].items():
            if sha256(attempt / name) != expected:
                raise ValueError("ARF short-grid evidence changed")
        row = {"trial": job["trial"], "config": job["config"],
               "status": receipt["status"],
               "wall_seconds": receipt["wall_seconds"],
               "peak_rss_kib": receipt["peak_rss_kib"],
               "attempt_receipt_sha256": sha256(path),
               "native_kpi": None, "artifact_bytes": None,
               "within_l3_bytes": False}
        if receipt["status"] == "ok":
            metric_path = attempt / "native-kpi.json"
            metric = json.loads(metric_path.read_text())
            inventory, charged = artifact_inventory(attempt / "artifact",
                                                     ARTIFACT_FILES)
            value = metric["value"]
            if (metric["objective"] != lock["native_objective"]
                    or metric["dataset"] != dataset
                    or metric["trial"] != job["trial"]
                    or metric["validation_sha256"] != job["validation_sha256"]
                    or metric["official_tests_opened"] is not False
                    or receipt["native_kpi"] != value
                    or receipt["native_kpi_sha256"] != sha256(metric_path)
                    or receipt["artifact_inventory"] != inventory
                    or receipt["artifact_bytes"] != charged
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError("ARF native KPI or charged artifact changed")
            row.update(native_kpi=value, artifact_bytes=charged,
                       within_l3_bytes=charged <= 10_240,
                       native_kpi_receipt_sha256=sha256(metric_path))
        datasets[dataset].append(row)
    cells = []
    for dataset, rows in datasets.items():
        rows.sort(key=lambda row: row["trial"])
        if [row["trial"] for row in rows] != list(range(4)) \
                or sum(row["wall_seconds"] for row in rows) > 43_200:
            raise ValueError("ARF refinement matrix or budget is incomplete")
        successful = [row for row in rows if row["status"] == "ok"]
        leader = min(successful, key=lambda row: (
            -row["native_kpi"], row["artifact_bytes"],
            digest(row["config"]), row["trial"])) if successful else None
        cells.append({"dataset": dataset, "method": "ARF",
                      "objective": lock["native_objective"]["name"],
                      "grid_leader_trial": leader["trial"] if leader else None,
                      "grid_leader_native_kpi": leader["native_kpi"] if leader else None,
                      "trials": rows})
    return {"format": "dope-arf-short-grid-native-validation", "version": 1,
            "scope": "training_derived_validation_only",
            "round_lock_sha256": sha256(lock_path),
            "parent_round_sha256": sha256(parent),
            "publisher_source_sha256": sha256(Path(__file__)),
            "cells": cells,
            "aggregate": {"cells": 3, "trials": 12,
                          "successful_trials": sum(
                              row["status"] == "ok" for cell in cells
                              for row in cell["trials"]),
                          "within_l3_trials": sum(
                              row["within_l3_bytes"] for cell in cells
                              for row in cell["trials"]),
                          "trial_wall_seconds": sum(
                              row["wall_seconds"] for cell in cells
                              for row in cell["trials"])},
            "eight_trial_native_selection_complete": False,
            "official_tests_opened": False,
            "mfs_v2": None, "ptf_v1": None,
            "production_certified": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-lock", type=Path, default=ROUND)
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "arf-short-native-validation.json")
    args = parser.parse_args()
    report = build(args.round_lock)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2,
                                      allow_nan=False) + "\n")
    lines = ["# ARF short-grid native validation", "",
             "This is a training-derived validation readout for the four-trial "
             "refinement grid. The original four-trial round is outside this readout; "
             "the grid leader is not the final eight-trial native selection. "
             "No official test rows were opened. MFS-v2 and PTF-v1 are null.", "",
             "| Dataset | Short-grid leader | Held-out FORDE mean log density | "
             "Charged artifact bytes | Successful trials |", "|---|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        leader = next((row for row in cell["trials"]
                       if row["trial"] == cell["grid_leader_trial"]), None)
        kpi = f"{leader['native_kpi']:.6f}" if leader else "—"
        lines.append(f"| {cell['dataset']} | "
                     f"{cell['grid_leader_trial'] if leader else '—'} | "
                     f"{kpi} | "
                     f"{leader['artifact_bytes'] if leader else '—'} | "
                     f"{sum(row['status'] == 'ok' for row in cell['trials'])}/4 |")
    lines += ["", "The native KPI compares configurations within each dataset only. "
              "All 12 short-grid sampler artifacts exceed the 10,240-byte L3 cap. "
              "Restricted sampler factors and detailed attempt logs remain on benchmark scratch; "
              "the JSON contains their immutable hashes.", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
