"""Combine original and bounded-refinement ARF trials by native likelihood."""

from __future__ import annotations

import json
from pathlib import Path

from .arf_round import OUT as ORIGINAL, write_once
from .arf_select import select as select_original
from .manifest import digest
from .publish_arf_short import ROUND as SHORT, build as read_short
from .score import sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
OUT = ROOT / "arf-native-combined-v1"


def select() -> list[dict]:
    parent_lock_path = ORIGINAL / "round.lock.json"
    short_lock_path = SHORT
    parent_lock = json.loads(parent_lock_path.read_text())
    short_lock = json.loads(short_lock_path.read_text())
    if (short_lock["parent_round_sha256"] != sha256(parent_lock_path)
            or short_lock["native_objective"] != parent_lock["native_objective"]
            or short_lock["cumulative_tuning_trials_per_dataset"] != 8
            or short_lock["official_tests_opened"] is not False):
        raise ValueError("ARF refinement is not the frozen parent continuation")
    original = {row["dataset"]: row for row in select_original(parent_lock_path)}
    short = {row["dataset"]: row for row in read_short(short_lock_path)["cells"]}
    result = []
    for dataset in ("Adult", "California", "News"):
        parent = original[dataset]
        refinement = short[dataset]
        trials = []
        for round_name, rows, lock, lock_path in (
            ("original", parent["trials"], parent_lock, parent_lock_path),
            ("short", refinement["trials"], short_lock, short_lock_path),
        ):
            jobs = {job["trial"]: job for job in lock["jobs"]
                    if job["dataset"] == dataset}
            if len(jobs) != 4 or len(rows) != 4:
                raise ValueError("ARF eight-trial matrix is incomplete")
            for row in rows:
                trial = row["job"]["trial"] if round_name == "original" else row["trial"]
                job = jobs[trial]
                attempt = lock_path.parent / "jobs" / digest(job) / "attempt-0001/receipt.json"
                if (row["attempt_receipt_sha256"] != sha256(attempt)
                        or (round_name == "original" and row["job"] != job)
                        or (round_name == "short" and row["config"] != job["config"])):
                    raise ValueError("ARF combined trial receipt changed")
                trials.append({"round": round_name, "trial": trial,
                               "job": job, "status": row["status"],
                               "wall_seconds": row["wall_seconds"],
                               "native_kpi": row.get("native_kpi"),
                               "artifact_bytes": row.get("artifact_bytes"),
                               "attempt_receipt_sha256": sha256(attempt),
                               "round_lock_sha256": sha256(lock_path)})
        if len(trials) != 8 or sum(row["wall_seconds"] for row in trials) > 43_200:
            raise ValueError("ARF combined trial or wall budget exceeded")
        eligible = [row for row in trials if row["status"] == "ok"]
        chosen = min(eligible, key=lambda row: (
            -row["native_kpi"], row["artifact_bytes"],
            digest(row["job"]["config"]), row["round"], row["trial"])) if eligible else None
        selection = {"format": "dope-arf-eight-trial-native-validation-selection",
                     "version": 1, "dataset": dataset, "method": "ARF",
                     "objective": parent_lock["native_objective"],
                     "source_sha256": sha256(Path(__file__)),
                     "original_round_sha256": sha256(parent_lock_path),
                     "refinement_round_sha256": sha256(short_lock_path),
                     "original_selection_sha256": sha256(
                         parent_lock_path.parent / f"{dataset}-selection.json"),
                     "refinement_publisher_sha256": sha256(
                         Path(__file__).with_name("publish_arf_short.py")),
                     "tuning_attempts": len(trials),
                     "tuning_wall_seconds": sum(row["wall_seconds"] for row in trials),
                     "default_status": trials[0]["status"],
                     "selected_round": chosen["round"] if chosen else None,
                     "selected_trial": chosen["trial"] if chosen else None,
                     "selected_config": chosen["job"]["config"] if chosen else None,
                     "selected_native_kpi": chosen["native_kpi"] if chosen else None,
                     "selected_artifact_bytes": chosen["artifact_bytes"] if chosen else None,
                     "selected_attempt_receipt_sha256":
                     chosen["attempt_receipt_sha256"] if chosen else None,
                     "trials": trials, "official_tests_opened": False,
                     "mfs_v2": None, "ptf_v1": None,
                     "production_certified": False}
        path = OUT / f"{dataset}-selection.json"
        if path.exists():
            if json.loads(path.read_text()) != selection:
                raise ValueError("existing ARF combined selection changed")
        else:
            write_once(path, selection)
        result.append(selection)
    return result


if __name__ == "__main__":
    print(json.dumps([{"dataset": row["dataset"],
                       "selected_round": row["selected_round"],
                       "selected_trial": row["selected_trial"],
                       "selected_native_kpi": row["selected_native_kpi"]}
                      for row in select()], sort_keys=True))
