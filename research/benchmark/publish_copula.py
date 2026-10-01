"""Reconcile GaussianCopula validation tuning with immutable scratch receipts."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from . import runner, tune_copula
from .publish_compact_native import render_csv
from .score import artifact_inventory, sha256


RESULTS = Path(__file__).with_name("results")


def build(round_path: Path = tune_copula.ROOT / "round.lock.json",
          methods_path: Path = Path(__file__).with_name("method-locks") / "2026-09-30.json",
          expected_jobs: int = 12) -> dict:
    lock = json.loads(round_path.read_text())
    entry = json.loads(methods_path.read_text())["methods"]["GaussianCopula"]
    if (lock["format"] != "dope-benchmark-copula-native-round"
            or lock["tuner_sha256"] != sha256(Path(tune_copula.__file__))
            or lock["methods_sha256"] != sha256(methods_path)
            or len(lock["jobs"]) != expected_jobs):
        raise ValueError("copula validation round changed")
    cells = []
    for job in lock["jobs"]:
        root = round_path.parent / job["dataset"]
        selection_path = root / "selection.json"
        selected = json.loads(selection_path.read_text()) if selection_path.exists() else None
        if selected is not None:
            if (selected["round_sha256"] != sha256(round_path)
                    or selected["stage"] != job["stage"]
                    or selected["test_opened"] is not False
                    or [trial["config"] for trial in selected["trials"]]
                    != job["configurations"]):
                raise ValueError("copula selection differs from lock")
            runner.resolve_configuration(
                {"final": True, "method": "GaussianCopula",
                 "configuration": {"kind": "tuned", "values": selected["selected_config"],
                                   "selection_path": str(selection_path),
                                   "selection_sha256": sha256(selection_path)}},
                entry, job["dataset"], 100, tune_copula.SCRATCH)
        trials = []
        for index, config in enumerate(job["configurations"]):
            trial_root = root / f"trial-{index:02d}"
            receipt_path = trial_root / "attempt.json"
            trial = json.loads(receipt_path.read_text())
            if selected is not None and selected["trials"][index] != trial:
                raise ValueError("copula attempt changed")
            identity = trial["identity"]
            if (identity["round_sha256"] != sha256(round_path)
                    or identity["train_sha256"] != job["train_sha256"]
                    or identity["validation_sha256"] != job["validation_sha256"]
                    or identity["config"] != config
                    or identity["fit_seed"] != job["fit_seed"]):
                raise ValueError("copula attempt identity changed")
            for name, expected in trial["logs"].items():
                if sha256(trial_root / name) != expected:
                    raise ValueError("copula process log changed")
            if trial["status"] == "ok":
                artifact = trial_root / "artifact"
                inventory, charged = artifact_inventory(
                    artifact, ["model.json", "projection.json"])
                metric = json.loads(Path(trial["metric_receipt_path"]).read_text())
                if (inventory != trial["artifact_inventory"]
                        or charged != trial["artifact_bytes"]
                        or metric["validation_sha256"] != job["validation_sha256"]
                        or metric["implementation_sha256"] != lock["objective_sha256"]):
                    raise ValueError("copula artifact or KPI changed")
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
            raise ValueError("successful copula trial has no selection")
        winner = trials[selected["selected_trial_index"]] if selected is not None else None
        cells.append({"stage": job["stage"], "dataset": job["dataset"],
                      "method": "GaussianCopula", "objective": "mean_log_density",
                      "selected_trial_index": selected["selected_trial_index"]
                      if selected is not None else None,
                      "selected_native_kpi": winner["native_kpi"] if winner else None,
                      "selected_artifact_bytes": winner["artifact_bytes"] if winner else None,
                      "selected_within_l3": winner["artifact_bytes"] <= 10_240
                      if winner else False,
                      "selection_receipt_path": str(selection_path) if winner else None,
                      "selection_receipt_sha256": sha256(selection_path) if winner else None,
                      "trials": trials})
    cells.sort(key=lambda cell: (cell["stage"], cell["dataset"]))
    return {"format": "dope-benchmark-copula-native-validation", "version": 1,
            "scope": "validation_only_one_fit_seed", "round_lock_sha256": sha256(round_path),
            "method_lock_sha256": sha256(methods_path), "cells": cells,
            "aggregate": {"cells": len(cells),
                          "trials": sum(len(cell["trials"]) for cell in cells),
                          "failed_trials": sum(trial["status"] != "ok" for cell in cells
                                               for trial in cell["trials"]),
                          "selected_within_l3": sum(cell["selected_within_l3"]
                                                    for cell in cells),
                          "trial_wall_seconds": sum(trial["wall_seconds"] for cell in cells
                                                    for trial in cell["trials"])},
            "ptf_v1": None, "mfs_v2": None, "production_certified": False,
            "public_or_s3_test_opened": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-lock", type=Path,
                        default=tune_copula.ROOT / "round.lock.json")
    parser.add_argument("--expected-jobs", type=int, choices=(12, 100), default=12)
    parser.add_argument("--basename", default="copula-native-validation")
    args = parser.parse_args()
    if (args.expected_jobs, args.basename) not in ((12, "copula-native-validation"),
                                                   (100, "copula-native-all-validation")):
        parser.error("result basename does not match frozen matrix size")
    report = build(args.round_lock, expected_jobs=args.expected_jobs)
    path = RESULTS / f"{args.basename}.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (RESULTS / f"{args.basename}.csv").write_text(render_csv(report))
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
