"""Publish complete ARF eight-trial native-validation selections."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from .arf_combined_select import OUT as SELECTION, select
from .arf_round import OUT as ORIGINAL, SOURCE
from .manifest import digest
from .publish_arf_short import ROUND as SHORT, build as short_grid
from .score import sha256


HERE = Path(__file__).parent
RESULT = HERE / "results/arf-combined-native-validation.json"
CAP = 10_240


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def build() -> dict:
    original_lock_path = ORIGINAL / "round.lock.json"
    short_lock_path = SHORT
    original = read(original_lock_path)
    refinement = short_grid(short_lock_path)
    selections = select()
    if (original["source_archive_sha256"] != sha256(SOURCE)
            or original["native_objective"]["name"]
            != "heldout_forde_mean_log_density"
            or original["native_objective"]["direction"] != "maximize"
            or original["official_tests_opened"] is not False
            or refinement["aggregate"]["trials"] != 12
            or refinement["aggregate"]["successful_trials"] != 12
            or refinement["round_lock_sha256"] != sha256(short_lock_path)
            or refinement["official_tests_opened"] is not False
            or len(selections) != 3):
        raise ValueError("ARF frozen native grids changed")
    cells = []
    for choice in selections:
        dataset = choice["dataset"]
        path = SELECTION / f"{dataset}-selection.json"
        if (read(path) != choice
                or choice["format"]
                != "dope-arf-eight-trial-native-validation-selection"
                or choice["source_sha256"] != sha256(
                    HERE / "arf_combined_select.py")
                or choice["original_round_sha256"] != sha256(original_lock_path)
                or choice["refinement_round_sha256"] != sha256(short_lock_path)
                or choice["refinement_publisher_sha256"] != sha256(
                    HERE / "publish_arf_short.py")
                or choice["objective"] != original["native_objective"]
                or choice["tuning_attempts"] != 8
                or choice["tuning_wall_seconds"] > 43_200
                or choice["official_tests_opened"] is not False
                or choice["mfs_v2"] is not None
                or choice["ptf_v1"] is not None):
            raise ValueError("ARF eight-trial selection changed")
        trials = []
        for row in choice["trials"]:
            lock_path = (original_lock_path if row["round"] == "original"
                         else short_lock_path)
            attempt = lock_path.parent / "jobs" / digest(row["job"])
            attempt /= "attempt-0001/receipt.json"
            if (row["round_lock_sha256"] != sha256(lock_path)
                    or row["attempt_receipt_sha256"] != sha256(attempt)
                    or row["job"]["dataset"] != dataset
                    or row["status"] not in
                    ("ok", "failed", "timeout", "invalid_evidence")):
                raise ValueError("ARF native attempt receipt changed")
            trials.append({"round": row["round"], "trial": row["trial"],
                           "config": row["job"]["config"],
                           "status": row["status"],
                           "native_kpi": row["native_kpi"],
                           "artifact_bytes": row["artifact_bytes"],
                           "wall_seconds": row["wall_seconds"],
                           "attempt_receipt_sha256": row["attempt_receipt_sha256"]})
        if (len(trials) != 8
                or {row["trial"] for row in trials if row["round"] == "original"}
                != set(range(4))
                or {row["trial"] for row in trials if row["round"] == "short"}
                != set(range(4))
                or sum(row["wall_seconds"] for row in trials)
                != choice["tuning_wall_seconds"]):
            raise ValueError("ARF native trial matrix incomplete")
        selected = next((row for row in trials
                         if row["round"] == choice["selected_round"]
                         and row["trial"] == choice["selected_trial"]), None)
        if (selected is None or selected["status"] != "ok"
                or selected["native_kpi"] != choice["selected_native_kpi"]
                or selected["artifact_bytes"] != choice["selected_artifact_bytes"]
                or selected["config"] != choice["selected_config"]
                or choice["default_status"] != trials[0]["status"]):
            raise ValueError("ARF selected native KPI changed")
        cells.append({"dataset": dataset, "method": "ARF",
                      "objective": choice["objective"]["name"],
                      "direction": "maximize",
                      "default_status": choice["default_status"],
                      "selected_round": choice["selected_round"],
                      "selected_trial": choice["selected_trial"],
                      "selected_native_kpi": choice["selected_native_kpi"],
                      "selected_artifact_bytes": choice["selected_artifact_bytes"],
                      "selected_within_l3_bytes":
                      choice["selected_artifact_bytes"] <= CAP,
                      "tuning_attempts": 8,
                      "tuning_wall_seconds": choice["tuning_wall_seconds"],
                      "selection_receipt_sha256": sha256(path),
                      "trials": trials})
    aggregate = {"datasets": len(cells), "trials": sum(len(cell["trials"])
                  for cell in cells), "successful_trials": sum(
                  row["status"] == "ok" for cell in cells for row in cell["trials"]),
                 "timeout_trials": sum(row["status"] == "timeout"
                  for cell in cells for row in cell["trials"]),
                 "selected_within_l3_bytes": sum(cell["selected_within_l3_bytes"]
                  for cell in cells),
                 "tuning_wall_seconds": sum(cell["tuning_wall_seconds"]
                  for cell in cells)}
    if (aggregate["datasets"] != 3 or aggregate["trials"] != 24
            or aggregate["successful_trials"] != 16
            or aggregate["timeout_trials"] != 8
            or aggregate["selected_within_l3_bytes"] != 0):
        raise ValueError("ARF complete native grid aggregate changed")
    return {"format": "dope-arf-eight-trial-native-validation-report",
            "version": 1, "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "locks": {"original_round": sha256(original_lock_path),
                      "short_round": sha256(short_lock_path),
                      "author_source_archive": sha256(SOURCE),
                      "short_publisher_source": sha256(
                          HERE / "publish_arf_short.py"),
                      "combined_selector_source": sha256(
                          HERE / "arf_combined_select.py")},
            "l3_byte_cap": CAP, "cells": cells, "aggregate": aggregate,
            "notes": ["ARF tuning maximized frozen held-out FORDE mean log density from a labeled study implementation of the author's fitted factors; values compare configurations within each dataset only.",
                      "Four original Adult and four original News trials timed out under the unchanged 900-second cap; all twelve short refinements succeeded.",
                      "The selected sampler artifacts exceed L3 on all three datasets; this native validation grid contributes no release-safe DOPE win.",
                      "Restricted fitted factors and source rows remain on scratch; official tests stayed sealed.",
                      "Shared validation outcomes and release gates are not complete; MFS-v2 and PTF-v1 remain null."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-arf-eight-trial-native-validation-report"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("ARF published native report changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=("dataset", "default_status",
                            "selected_round", "selected_trial", "selected_native_kpi",
                            "selected_artifact_bytes", "selected_within_l3_bytes",
                            "tuning_attempts", "tuning_wall_seconds"),
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["cells"])
    RESULT.with_suffix(".csv").write_text(table.getvalue())
    lines = ["# ARF complete native-validation pilot", "",
             "Four original and four bounded-refinement trials were charged to "
             "each dataset. Selection used only ARF's held-out FORDE mean log "
             "density on training-derived validation rows. The likelihood "
             "evaluator is a labeled study implementation using the author's "
             "fitted factors. Values compare trials "
             "within a dataset, not methods or datasets.", "",
             "| Dataset | Default | Selected round / trial | Native log density | "
             "Artifact bytes | L3 bytes | Successful trials |",
             "|---|---|---|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        lines.append(f"| {cell['dataset']} | {cell['default_status']} | "
                     f"{cell['selected_round']} / {cell['selected_trial']} | "
                     f"{cell['selected_native_kpi']:.6f} | "
                     f"{cell['selected_artifact_bytes']:,} | "
                     f"{'yes' if cell['selected_within_l3_bytes'] else 'no'} | "
                     f"{sum(row['status'] == 'ok' for row in cell['trials'])}/8 |")
    lines += ["", "All eight original Adult/News attempts timed out under their "
              "frozen 900-second caps; their null outcomes remain in the JSON. "
              "All twelve short refinements succeeded, but the final native "
              "selections are over the 10,240-byte L3 limit. Restricted artifacts "
              "and detailed logs stay on scratch. Official tests remain sealed; "
              "shared validation metrics, MFS-v2, PTF-v1, and production "
              "certification are unavailable.", ""]
    RESULT.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
