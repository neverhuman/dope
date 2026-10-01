"""Bounded News seed-23 confirmation of the preregistered autoregressive candidate."""

from __future__ import annotations

import fcntl
import json
import os
import sys
from pathlib import Path

from research.benchmark import gpu_probe, pilot_metrics
from research.benchmark.dispatch_gpu_round import admitted
from research.benchmark.inventory_hosts import local
from research.benchmark.score import sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark")
OUT = ROOT / "pilot-24h/dope-news-autoreg-confirm-v1"
LOCK = OUT / "round.lock.json"
WORKER = ROOT / "pilot-24h/prepared-v2/worker/News"
BINARY = ROOT / "conditional-readout-v1/bin/dope-gpu-132f8467f56a79ec"
BASELINE = Path("/home/ubuntu/dope/.agent/worktrees/integration/research/benchmark/results/pilot24-interim.json")
CPU = list(range(32, 48))
JOB = {"dataset": "News", "candidate": "symbolic_autoregressive_residual",
       "seed": 23, "target_weight": 4.0, "structural_penalty": 0.0}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def freeze() -> Path:
    baseline = read(BASELINE)
    rows = [row for row in baseline["research_candidates"]
            if row["dataset"] == "News"
            and row["candidate"] == JOB["candidate"]
            and row["target_weight"] == JOB["target_weight"]
            and row["fit_seed"] == 11 and row["size"] == 1]
    worker = read(WORKER / "worker-manifest.json")
    if (len(rows) != 1 or rows[0]["retention"] <= 1.0
            or rows[0]["exact_row_copies"] != 0
            or rows[0]["artifact_bytes_research_packed"] > 10_240
            or baseline["official_tests_opened"] is not False
            or baseline["mfs_v2"] is not None
            or baseline["ptf_v1"] is not None
            or worker["dataset_id"] != "News"
            or (WORKER / "test.csv").exists()
            or any(sha256(WORKER / f"{part}.csv")
                   != worker["projected_hashes"][part]
                   for part in ("train", "validation"))
            or sha256(WORKER / "projection.json") != worker["projection_sha256"]):
        raise ValueError("News autoregressive confirmation trigger changed")
    pilot = read(ROOT / "pilot-24h/start-v2.json")
    lock = {"format": "dope-news-autoregressive-confirmation-round", "version": 1,
            "scope": "training_derived_validation_only",
            "trigger": "News seed-11 candidate exceeded 1.0 descriptive retention at n; confirm independently at fit seed 23 before any promotion.",
            "seed11_report_sha256": sha256(BASELINE),
            "seed11_metric_receipt_sha256": rows[0]["metric_receipt_sha256"],
            "source_sha256": sha256(Path(__file__)),
            "probe_source_sha256": sha256(Path(gpu_probe.__file__)),
            "metric_source_sha256": sha256(Path(pilot_metrics.__file__)),
            "gpu_binary_sha256": sha256(BINARY),
            "gpu_binary_path": str(BINARY),
            "train_sha256": worker["projected_hashes"]["train"],
            "validation_sha256": worker["projected_hashes"]["validation"],
            "projection_sha256": worker["projection_sha256"],
            "gpu_jobs": [JOB], "fit_seed": 23,
            "sample_seeds": [101, 211, 307],
            "size_multipliers": [1, 4],
            "cpu_slot": CPU, "gpu_fit_seconds_ceiling": 600,
            "gpu_vram_mib_ceiling": 16 * 1024,
            "scratch_ceiling_bytes": 200_000_000_000,
            "deadline_utc": pilot["deadline_utc"],
            "selection_policy": "confirmation_only_no_test_and_no_global_selection",
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}
    write_once(LOCK, lock)
    return LOCK


def check_lock() -> dict:
    lock = read(LOCK)
    if (lock["format"] != "dope-news-autoregressive-confirmation-round"
            or lock["source_sha256"] != sha256(Path(__file__))
            or lock["probe_source_sha256"] != sha256(Path(gpu_probe.__file__))
            or lock["gpu_binary_sha256"] != sha256(BINARY)
            or lock["seed11_report_sha256"] != sha256(BASELINE)
            or lock["gpu_jobs"] != [JOB]
            or lock["cpu_slot"] != CPU
            or lock["official_tests_opened"] is not False
            or lock["mfs_v2"] is not None or lock["ptf_v1"] is not None):
        raise ValueError("News autoregressive confirmation lock changed")
    return lock


def run() -> dict:
    lock = check_lock()
    if set(os.sched_getaffinity(0)) != set(CPU):
        raise ValueError("News GPU confirmation escaped CPU slot")
    snapshot = local()
    receipt_path = OUT / "dispatch-receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if (receipt["round_lock_sha256"] != sha256(LOCK)
                or receipt["source_sha256"] != sha256(Path(__file__))):
            raise ValueError("prior News confirmation dispatch changed")
        return receipt
    with (ROOT / "neural-native-v1/xbabe2.gpu-reservation").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = local()
        if not admitted(snapshot):
            receipt = {"format": "dope-news-autoregressive-confirmation-dispatch",
                       "version": 1, "status": "admission_blocked",
                       "reason": "gpu_or_host_resource_unavailable",
                       "host_snapshot": snapshot,
                       "round_lock_sha256": sha256(LOCK),
                       "source_sha256": sha256(Path(__file__)),
                       "fit_receipt_sha256": None,
                       "official_tests_opened": False,
                       "mfs_v2": None, "ptf_v1": None}
        else:
            try:
                fitted = gpu_probe.run(WORKER, BINARY, JOB["candidate"],
                                       JOB["seed"], JOB["target_weight"],
                                       JOB["structural_penalty"], OUT / "fits", LOCK)
                status, reason = fitted["status"], None
            except (OSError, ValueError, RuntimeError) as error:
                fitted = None
                status, reason = "failed", type(error).__name__
            receipt = {"format": "dope-news-autoregressive-confirmation-dispatch",
                       "version": 1, "status": status, "reason_type": reason,
                       "host_snapshot": snapshot,
                       "round_lock_sha256": sha256(LOCK),
                       "source_sha256": sha256(Path(__file__)),
                       "fit_receipt_path": fitted["receipt"] if fitted else None,
                       "fit_receipt_sha256": fitted["receipt_sha256"] if fitted else None,
                       "official_tests_opened": False,
                       "mfs_v2": None, "ptf_v1": None}
        write_once(receipt_path, receipt)
        return receipt


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("freeze", "run"):
        raise SystemExit("freeze|run")
    print(freeze() if sys.argv[1] == "freeze" else json.dumps(run(), sort_keys=True))
