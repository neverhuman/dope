"""Publish rights-safe compact-method validation receipts without test outcomes."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from . import runner, tune_density
from .score import artifact_inventory, sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
ROUND = SCRATCH / "compact-native-v1/round.lock.json"
METHODS = Path(__file__).with_name("method-locks") / "2026-09-30.json"
RESULTS = Path(__file__).with_name("results")


def build(round_path: Path = ROUND, methods_path: Path = METHODS,
          expected_jobs: int = 24) -> dict:
    locked = json.loads(round_path.read_text())
    entries = json.loads(methods_path.read_text())["methods"]
    if (locked["format"] != "dope-benchmark-compact-native-round"
            or locked["method_lock_sha256"] != sha256(methods_path)
            or locked["tuner_sha256"] != sha256(Path(tune_density.__file__))
            or len(locked["jobs"]) != expected_jobs):
        raise ValueError("compact validation matrix changed")
    cells = []
    for job in locked["jobs"]:
        root = round_path.parent / job["dataset"] / job["method"]
        selection_path = root / "selection.json"
        selected = json.loads(selection_path.read_text()) if selection_path.exists() else None
        if selected is not None:
            if (selected.get("round_sha256") != sha256(round_path)
                    or selected.get("stage") != job["stage"]
                    or selected.get("test_opened") is not False
                    or [trial["config"] for trial in selected["trials"]] != job["configurations"]):
                raise ValueError("native selection differs from matrix")
            choice = {"kind": "tuned", "values": selected["selected_config"],
                      "selection_path": str(selection_path),
                      "selection_sha256": sha256(selection_path)}
            runner.resolve_configuration({"final": True, "method": job["method"],
                                          "configuration": choice}, entries[job["method"]],
                                         job["dataset"], 100, SCRATCH)
        trials = []
        for index, config in enumerate(job["configurations"]):
            receipt_path = root / f"trial-{index:02d}/attempt.json"
            trial = json.loads(receipt_path.read_text())
            if selected is not None and selected["trials"][index] != trial:
                raise ValueError("native attempt changed")
            identity = trial["identity"]
            if (identity["round_sha256"] != sha256(round_path)
                    or identity["train_sha256"] != job["train_sha256"]
                    or identity["validation_sha256"] != job["validation_sha256"]
                    or identity["config"] != config
                    or identity["fit_seed"] != job["fit_seed"]):
                raise ValueError("native attempt identity changed")
            if trial["status"] == "ok":
                artifact = root / f"trial-{index:02d}/artifact"
                inventory, charged_bytes = artifact_inventory(
                    artifact, ["model.json", "projection.json"])
                metric = json.loads(Path(trial["metric_receipt_path"]).read_text())
                if (inventory != trial["artifact_inventory"]
                        or charged_bytes != trial["artifact_bytes"]
                        or metric["validation_sha256"] != job["validation_sha256"]):
                    raise ValueError("native artifact or KPI changed")
            trials.append({"index": index, "config": trial["config"],
                           "status": trial["status"],
                           "native_kpi": trial.get("native_kpi"),
                           "artifact_bytes": trial.get("artifact_bytes"),
                           "wall_seconds": trial["wall_seconds"],
                           "failure_type": trial.get("error_type"),
                           "attempt_receipt_path": str(receipt_path),
                           "attempt_receipt_sha256": sha256(receipt_path),
                           "metric_receipt_sha256": trial.get("metric_receipt_sha256")})
        if selected is None and any(trial["status"] == "ok" for trial in trials):
            raise ValueError("successful native trial has no selection")
        winner = trials[selected["selected_trial_index"]] if selected is not None else None
        cells.append({"stage": job["stage"], "dataset": job["dataset"],
                      "method": job["method"], "objective": "mean_log_density",
                      "selected_trial_index": selected["selected_trial_index"]
                      if selected is not None else None,
                      "selected_native_kpi": winner["native_kpi"] if winner else None,
                      "selected_artifact_bytes": winner["artifact_bytes"] if winner else None,
                      "selected_within_l3": winner["artifact_bytes"] <= 10_240
                      if winner else False,
                      "selection_receipt_path": str(selection_path) if winner else None,
                      "selection_receipt_sha256": sha256(selection_path) if winner else None,
                      "trials": trials})
    cells.sort(key=lambda cell: (cell["stage"], cell["dataset"], cell["method"]))
    aggregate = {}
    for method in tune_density.METHODS:
        group = [cell for cell in cells if cell["method"] == method]
        aggregate[method] = {"cells": len(group),
                             "trials": sum(len(cell["trials"]) for cell in group),
                             "failed_trials": sum(trial["status"] != "ok" for cell in group
                                                  for trial in cell["trials"]),
                             "selected_within_l3": sum(cell["selected_within_l3"]
                                                       for cell in group),
                             "trial_wall_seconds": sum(trial["wall_seconds"]
                                                       for cell in group
                                                       for trial in cell["trials"])}
    return {"format": "dope-benchmark-compact-native-validation", "version": 1,
            "scope": "validation_only_one_fit_seed", "round_lock_sha256": sha256(round_path),
            "method_lock_sha256": sha256(methods_path), "cells": cells,
            "aggregate": aggregate, "ptf_v1": None, "mfs_v2": None,
            "production_certified": False, "public_or_s3_test_opened": False}


def render_csv(report: dict) -> str:
    stream = io.StringIO(newline="")
    fields = ["stage", "dataset", "method", "trial", "selected", "config", "status",
              "native_kpi", "artifact_bytes", "wall_seconds", "failure_type",
              "attempt_receipt_sha256", "metric_receipt_sha256"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for cell in report["cells"]:
        for trial in cell["trials"]:
            writer.writerow({"stage": cell["stage"], "dataset": cell["dataset"],
                             "method": cell["method"], "trial": trial["index"],
                             "selected": trial["index"] == cell["selected_trial_index"],
                             "config": json.dumps(trial["config"], sort_keys=True),
                             **{field: trial[field] for field in fields[6:]}})
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-lock", type=Path, default=ROUND)
    parser.add_argument("--methods-lock", type=Path, default=METHODS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--expected-jobs", type=int, choices=(24, 200), default=24)
    parser.add_argument("--basename", default="compact-native-validation")
    args = parser.parse_args()
    if (args.expected_jobs, args.basename) not in ((24, "compact-native-validation"),
                                                   (200, "compact-native-all-validation")):
        parser.error("result basename does not match frozen matrix size")
    report = build(args.round_lock, args.methods_lock, args.expected_jobs)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"{args.basename}.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n")
    (args.results / f"{args.basename}.csv").write_text(render_csv(report))
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
