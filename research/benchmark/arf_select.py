"""Select author-ARF configurations only by frozen held-out FORDE likelihood."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .arf_round import OUT, check_lock, write_once
from .manifest import digest
from .score import artifact_inventory, sha256


def winner(trials: list[dict]) -> dict | None:
    eligible = [trial for trial in trials if trial["status"] == "ok"]
    if not eligible:
        return None
    return min(eligible, key=lambda trial: (
        -trial["native_kpi"], trial["artifact_bytes"],
        digest(trial["job"]["config"]), trial["job"]["trial"]))


def select(lock_path: Path = OUT / "round.lock.json") -> list[dict]:
    lock = check_lock(lock_path)
    result = []
    for dataset in ("Adult", "California", "News"):
        trials = []
        for job in lock["jobs"]:
            if job["dataset"] != dataset:
                continue
            attempt = lock_path.parent / "jobs" / digest(job) / "attempt-0001"
            path = attempt / "receipt.json"
            if not path.is_file():
                raise ValueError("ARF native trial has no immutable attempt receipt")
            receipt = json.loads(path.read_text())
            if (receipt.get("format") != "dope-arf-native-attempt"
                    or receipt.get("job") != job
                    or receipt.get("round_sha256") != sha256(lock_path)
                    or receipt.get("status") not in
                    ("ok", "failed", "timeout", "invalid_evidence")
                    or receipt.get("official_tests_opened") is not False
                    or receipt.get("mfs_v2") is not None
                    or receipt.get("ptf_v1") is not None):
                raise ValueError("ARF native attempt lineage changed")
            for name, expected in receipt["evidence_files"].items():
                if sha256(attempt / name) != expected:
                    raise ValueError("ARF native trial evidence changed")
            trial = {"job": job, "status": receipt["status"],
                     "wall_seconds": receipt["wall_seconds"],
                     "attempt_receipt_sha256": sha256(path)}
            if receipt["status"] == "ok":
                metric_path = attempt / "native-kpi.json"
                metric = json.loads(metric_path.read_text())
                inventory, charged = artifact_inventory(
                    attempt / "artifact",
                    ["model.json", "bounds.csv", "continuous.csv",
                     "categories.csv", "projection.json"])
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
                trial.update({"native_kpi": value, "artifact_bytes": charged,
                              "native_kpi_receipt_sha256": sha256(metric_path),
                              "artifact_inventory": inventory})
            trials.append(trial)
        if (len(trials) != 4 or len({trial["job"]["trial"] for trial in trials}) != 4
                or sum(trial["wall_seconds"] for trial in trials) > 43_200):
            raise ValueError("ARF native search exceeds frozen trial or wall budget")
        chosen = winner(trials)
        selection = {"format": "dope-arf-native-validation-selection", "version": 1,
                     "dataset": dataset, "method": "ARF",
                     "round_sha256": sha256(lock_path),
                     "selection_source_sha256": sha256(Path(__file__)),
                     "objective": lock["native_objective"],
                     "default_trial": 0,
                     "selected_trial": chosen["job"]["trial"] if chosen else None,
                     "selected_config": chosen["job"]["config"] if chosen else None,
                     "selected_native_kpi": chosen["native_kpi"] if chosen else None,
                     "selected_artifact_bytes": chosen["artifact_bytes"] if chosen else None,
                     "tuning_attempts": len(trials),
                     "tuning_wall_seconds": sum(trial["wall_seconds"] for trial in trials),
                     "trials": trials, "official_tests_opened": False,
                     "mfs_v2": None, "ptf_v1": None,
                     "production_certified": False}
        path = lock_path.parent / f"{dataset}-selection.json"
        if path.exists():
            if json.loads(path.read_text()) != selection:
                raise ValueError("existing ARF native selection changed")
        else:
            write_once(path, selection)
        result.append(selection)
    return result


if __name__ == "__main__":
    print(json.dumps([{"dataset": item["dataset"],
                       "selected_trial": item["selected_trial"],
                       "selected_native_kpi": item["selected_native_kpi"]}
                      for item in select()], sort_keys=True))
