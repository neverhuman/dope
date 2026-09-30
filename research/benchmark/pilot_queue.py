"""Bounded three-host pilot queue; all job data and receipts live on shared scratch."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .fetch_jope import LIMIT, used_bytes


ROSTER = ("dope", "GaussianCopula", "TabPC", "TabKDE", "TabSyn", "TabDiff", "AIM")
DATASETS = ("Adult", "California", "News")
FIT_SEEDS = (11, 23)
SAMPLE_SEEDS = (101, 211)
SIZES = (1, 2, 4, 8)
HOST_SLOTS = {"xbabe1": 2, "xbabe2": 2, "xbabe3": 2}
ROOT = Path("/mnt/fast-scratch/dope-benchmark")
PACKAGE = ROOT / "pilot-4h" / "package"
RESULTS = ROOT / "pilot-4h" / "results"


@contextmanager
def cpu_slot(host: str, pools: dict):
    slot = pools[host].get()
    try:
        yield slot
    finally:
        pools[host].put(slot)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("x") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_inputs(lock: dict) -> None:
    if not lock.get("complete") or {item["id"] for item in lock["datasets"]} != set(DATASETS):
        raise ValueError("pilot dataset lock is incomplete")
    for item in lock["datasets"]:
        worker = ROOT / "public-prepared" / "worker" / item["id"]
        for name in ("train", "validation"):
            if digest(worker / f"{name}.csv") != item["projected_files"][name]:
                raise ValueError(f"{item['id']} {name} digest mismatch")
        if digest(worker / "projection.json") != item["projection_sha256"]:
            raise ValueError(f"{item['id']} projection digest mismatch")
    if digest(PACKAGE / "research/benchmark/adapters.py") != next(
        entry["adapter_sha256"] for entry in METHODS["methods"].values() if entry["status"] == "locked"
    ):
        raise ValueError("deployed adapter digest differs from source lock")
    if digest(ROOT / "bin/dope-kernel") != METHODS["methods"]["dope"]["binary_sha256"]:
        raise ValueError("DOPE binary digest mismatch")


def host_for(dataset: str, method: str, seed: int) -> str:
    if method == "GaussianCopula":
        return "xbabe2"
    # The two 3090s were occupied at admission; DOPE is a CPU job.
    return ("xbabe1", "xbabe3")[(DATASETS.index(dataset) + FIT_SEEDS.index(seed)) % 2]


def matrix(methods: dict) -> list[dict]:
    return [
        {"dataset": dataset, "method": method, "fit_seed": seed,
         "sample_seeds": list(SAMPLE_SEEDS), "size_multipliers": list(SIZES),
         "host": host_for(dataset, method, seed) if methods["methods"][method]["status"] == "locked" else None,
         "method_status": methods["methods"][method]["status"],
         "track": "common_numeric"}
        for dataset in DATASETS for method in ROSTER for seed in FIT_SEEDS
    ]


def job_name(cell: dict) -> str:
    return f"{cell['dataset']}-{cell['method']}-{cell['fit_seed']}"


def run_cell(cell: dict, deadline_epoch: float, semaphores: dict, cpu_pools: dict,
             repair_digest_mismatch: bool = False) -> dict:
    name = job_name(cell)
    directory = ROOT / "pilot-4h"
    receipt_path = directory / "receipts" / f"{name}.json"
    if receipt_path.exists():
        original = json.loads(receipt_path.read_text())
        old_log = directory / "logs" / f"{name}.log"
        if not (repair_digest_mismatch and original["status"] == "failed"
                and old_log.exists() and "adapter source digest changed" in old_log.read_text()):
            return original
        receipt_path = directory / "repair-receipts" / f"{name}-attempt2.json"
        if receipt_path.exists():
            return json.loads(receipt_path.read_text())
    entry = METHODS["methods"][cell["method"]]
    start = time.time()
    common = {"format": "dope-four-hour-pilot-job", "dataset": cell["dataset"],
              "method": cell["method"], "fit_seed": cell["fit_seed"],
              "host": cell["host"], "source_status": entry["status"],
              "started_utc": datetime.fromtimestamp(start, timezone.utc).isoformat()}
    if entry["status"] != "locked":
        receipt = {**common, "status": "blocked", "reason": entry["status"],
                   "detail": entry.get("note", "Source, dependency, and adapter audit incomplete"),
                   "elapsed_seconds": 0.0}
        atomic_json(receipt_path, receipt)
        return receipt
    host = cell["host"]
    with semaphores[host], cpu_slot(host, cpu_pools) as slot:
        start = time.time()
        common["started_utc"] = datetime.fromtimestamp(start, timezone.utc).isoformat()
        common["cpu_affinity"] = f"{16 * slot}-{16 * slot + 15}"
        remaining = int(deadline_epoch - time.time())
        if remaining <= 0:
            receipt = {**common, "status": "not_started", "reason": "pilot_deadline",
                       "elapsed_seconds": 0.0}
            atomic_json(receipt_path, receipt)
            return receipt
        # The largest fixed-pilot eight-cell output allowance is below 3.2 GB;
        # 20 GB covers all six concurrent pilot slots.
        if used_bytes(ROOT) > LIMIT - 20_000_000_000:
            receipt = {**common, "status": "not_started", "reason": "scratch_ceiling",
                       "elapsed_seconds": 0.0}
            atomic_json(receipt_path, receipt)
            return receipt
        job = {"worker_dir": str(ROOT / "public-prepared" / "worker" / cell["dataset"]),
               "method": cell["method"], "fit_seed": cell["fit_seed"],
               "sample_seeds": list(SAMPLE_SEEDS), "size_multipliers": list(SIZES),
               "track": "common_numeric", "scratch_root": str(ROOT),
               "dope_binary": str(ROOT / "bin/dope-kernel")}
        job_path = directory / "jobs" / f"{name}.json"
        if not job_path.exists():
            atomic_json(job_path, job)
        python = ("/home/ubuntu/dope/.agent/worktrees/integration/target/bench-venv/bin/python"
                  if cell["method"] == "GaussianCopula" else "/usr/bin/python3")
        command = ["env", f"PYTHONPATH={PACKAGE}", "CUDA_VISIBLE_DEVICES=",
                   "OMP_NUM_THREADS=16", "OPENBLAS_NUM_THREADS=16", "MKL_NUM_THREADS=16",
                   python, "-m", "research.benchmark.runner", str(job_path),
                   str(PACKAGE / "research/benchmark/methods.lock.json"), str(RESULTS)]
        cap = min(remaining, 2700 if cell["method"] == "GaussianCopula" else 1200)
        command = ["timeout", "--signal=TERM", "--kill-after=20s", str(cap),
                   "/usr/bin/time", "-v", "taskset", "-c", common["cpu_affinity"], *command]
        if host != "xbabe2":
            command = ["ssh", "-o", "BatchMode=yes", host, shlex.join(command)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=cap + 35,
                                    cwd=str(PACKAGE) if host == "xbabe2" else None)
            log_path = directory / ("repair-logs" if repair_digest_mismatch else "logs") / f"{name}.log"
            log_path.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)
            child = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else []
            statuses = [value["status"] for value in child]
            status = "ok" if statuses and all(value == "ok" for value in statuses) else (
                "partial" if statuses else "timeout" if result.returncode == 124 else "failed")
            receipt = {**common, "status": status, "runner_exit_code": result.returncode,
                       "sample_statuses": statuses, "runner_receipt_count": len(child),
                       "elapsed_seconds": time.time() - start,
                       "log_sha256": digest(log_path), "log_path": str(log_path)}
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            receipt = {**common, "status": "timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed",
                       "error_type": type(error).__name__, "elapsed_seconds": time.time() - start}
        atomic_json(receipt_path, receipt)
        return receipt


def summary(cells: list[dict]) -> dict:
    from collections import Counter
    counts = Counter(cell["status"] for cell in cells)
    return {"format": "dope-four-hour-pilot-summary", "total_fit_cells": len(cells),
            "status_counts": dict(sorted(counts.items())),
            "by_method": {method: dict(Counter(cell["status"] for cell in cells if cell["method"] == method))
                          for method in ROSTER},
            "measured_job_seconds": sum(cell["elapsed_seconds"] for cell in cells)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-utc", required=True, help="ISO 8601 UTC timestamp")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repair-digest-mismatch", action="store_true")
    args = parser.parse_args()
    global METHODS
    METHODS = json.loads((PACKAGE / "research/benchmark/methods.lock.json").read_text())
    lock = json.loads((PACKAGE / "research/benchmark/pilot-datasets.lock.json").read_text())
    verify_inputs(lock)
    cells = matrix(METHODS)
    pilot = ROOT / "pilot-4h"
    for name in ("jobs", "logs", "receipts", "repair-logs", "repair-receipts"):
        (pilot / name).mkdir(parents=True, exist_ok=True)
    matrix_path = pilot / "matrix.json"
    if not matrix_path.exists():
        atomic_json(matrix_path, {"format": "dope-four-hour-pilot-matrix", "cells": cells})
    elif json.loads(matrix_path.read_text())["cells"] != cells:
        raise ValueError("pilot matrix changed after freeze")
    if not args.execute:
        print(json.dumps({"cells": len(cells), "runnable": sum(c["method_status"] == "locked" for c in cells)}))
        return
    deadline_epoch = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()
    semaphores = {host: threading.Semaphore(slots) for host, slots in HOST_SLOTS.items()}
    cpu_pools = {host: queue.Queue() for host in HOST_SLOTS}
    for host, pool in cpu_pools.items():
        for slot in range(HOST_SLOTS[host]):
            pool.put(slot)
    with concurrent.futures.ThreadPoolExecutor(max_workers=sum(HOST_SLOTS.values())) as pool:
        futures = [pool.submit(run_cell, cell, deadline_epoch, semaphores, cpu_pools,
                               args.repair_digest_mismatch) for cell in cells]
        receipts = [future.result() for future in concurrent.futures.as_completed(futures)]
    report = summary(receipts)
    atomic_json(pilot / ("queue-summary-repair.json" if args.repair_digest_mismatch
                         else "queue-summary.json"), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
