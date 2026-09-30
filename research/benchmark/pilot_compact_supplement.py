"""Run source-locked compact reference cells outside the fixed pilot matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .pilot_queue import (DATASETS, FIT_SEEDS, PACKAGE, ROOT, SAMPLE_SEEDS,
                          SIZES, atomic_json, digest)


METHODS = ("independent_marginals", "Chow-Liu")
BASE = ROOT / "pilot-4h" / "compact-supplement"


def dispatch(dataset: str, method: str, seed: int, deadline: float) -> dict:
    name = f"{dataset}-{method}-{seed}"
    receipt_path = BASE / "receipts" / f"{name}.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())
    job = {"worker_dir": str(ROOT / "public-prepared" / "worker" / dataset),
           "method": method, "fit_seed": seed, "sample_seeds": list(SAMPLE_SEEDS),
           "size_multipliers": list(SIZES), "track": "common_numeric",
           "scratch_root": str(ROOT), "dope_binary": str(ROOT / "bin/dope-kernel")}
    job_path = BASE / "jobs" / f"{name}.json"
    if job_path.exists():
        if json.loads(job_path.read_text()) != job:
            raise ValueError(f"supplement job changed: {name}")
    else:
        atomic_json(job_path, job)
    remaining = min(1200, int(deadline - time.time()))
    started = datetime.now(timezone.utc).isoformat()
    if remaining <= 0:
        receipt = {"name": name, "status": "not_started", "reason": "pilot_deadline",
                   "host": "xbabe1", "started_utc": started, "elapsed_seconds": 0.0}
    else:
        command = ["env", f"PYTHONPATH={PACKAGE}", "CUDA_VISIBLE_DEVICES=",
                   "OMP_NUM_THREADS=16", "OPENBLAS_NUM_THREADS=16", "MKL_NUM_THREADS=16",
                   "timeout", "--signal=TERM", "--kill-after=20s", str(remaining),
                   "/usr/bin/time", "-v", "taskset", "-c",
                   "0-15" if seed == 11 else "16-31", "/usr/bin/python3", "-m",
                   "research.benchmark.runner", str(job_path),
                   str(PACKAGE / "research/benchmark/methods.lock.json"), str(BASE / "results")]
        start = time.monotonic()
        try:
            process = subprocess.run(["ssh", "-o", "BatchMode=yes", "xbabe1", shlex.join(command)],
                                     capture_output=True, text=True, timeout=remaining + 35)
        except subprocess.TimeoutExpired as error:
            receipt = {"name": name, "status": "timeout", "host": "xbabe1",
                       "started_utc": started, "elapsed_seconds": time.monotonic() - start,
                       "error_type": type(error).__name__}
            atomic_json(receipt_path, receipt)
            return receipt
        log_path = BASE / "logs" / f"{name}.log"
        log_path.write_text(process.stdout + "\n--- stderr ---\n" + process.stderr)
        status = "timeout" if process.returncode == 124 else "failed"
        if process.returncode == 0:
            try:
                children = json.loads(process.stdout.strip().splitlines()[-1])
                status = "ok" if children and all(child["status"] == "ok" for child in children) else "partial"
            except (IndexError, json.JSONDecodeError):
                status = "failed"
        receipt = {"name": name, "status": status, "host": "xbabe1",
                   "started_utc": started, "elapsed_seconds": time.monotonic() - start,
                   "runner_exit_code": process.returncode, "log_sha256": digest(log_path)}
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    method_lock = json.loads((PACKAGE / "research/benchmark/methods.lock.json").read_text())
    if any(method_lock["methods"][method]["status"] != "locked" for method in METHODS):
        raise ValueError("supplement method is not source locked")
    cells = [{"dataset": dataset, "method": method, "fit_seed": seed, "host": "xbabe1"}
             for dataset in DATASETS for method in METHODS for seed in FIT_SEEDS]
    for name in ("jobs", "logs", "receipts", "results"):
        (BASE / name).mkdir(parents=True, exist_ok=True)
    matrix_path = BASE / "matrix.json"
    if matrix_path.exists():
        if json.loads(matrix_path.read_text())["cells"] != cells:
            raise ValueError("supplement matrix changed")
    else:
        atomic_json(matrix_path, {"format": "dope-compact-pilot-supplement", "cells": cells})
    if not args.execute:
        print(json.dumps({"cells": len(cells)}))
        return
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dispatch, cell["dataset"], cell["method"], cell["fit_seed"], deadline)
                   for cell in cells]
        receipts = [future.result() for future in concurrent.futures.as_completed(futures)]
    print(json.dumps({"fit_cells": len(receipts),
                      "status_counts": {status: sum(r["status"] == status for r in receipts)
                                        for status in sorted({r["status"] for r in receipts})}}, sort_keys=True))


if __name__ == "__main__":
    main()
