"""Bounded validation refinement using existing GPU conditional DOPE backends."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

from . import gpu_probe, pilot_metrics
from .dispatch_gpu_round import admitted
from .inventory_hosts import local
from .score import sha256
from .sdv_round import write_once


ROOT = Path("/mnt/fast-scratch/dope-benchmark/conditional-gpu-v1")
SCRATCH = ROOT.parent
WORKERS = SCRATCH / "s3-v1/prepared/worker"
BINARY = SCRATCH / "gpu-research-v1/bin/dope-gpu-c37c25bf4d1f138b"


def freeze(stage: str, root: Path, binary: Path, trigger: str) -> Path:
    if not root.resolve().is_relative_to(SCRATCH.resolve()):
        raise ValueError("conditional evidence must stay on benchmark scratch")
    cohort_path = SCRATCH / "s3-v1/gpu-cohort.lock.json"
    cohort = json.loads(cohort_path.read_text())
    jobs = [{"dataset": row["id"], "candidate": candidate, "seed": 11,
             "target_weight": 2.0, "structural_penalty": 0.0}
            for row in cohort[stage] for candidate in
            ("compact_neural_residual_symbolic", "symbolic_autoregressive_residual")]
    lock = {"format": "dope-conditional-gpu-validation-round", "version": 1,
            "validation_only": True, "stage": stage, "gpu_jobs": jobs,
            "cohort_sha256": sha256(cohort_path), "gpu_binary_sha256": sha256(binary),
            "gpu_binary_path": str(binary),
            "source_manifest_sha256": sha256(root / "source/files.json"),
            "probe_source_sha256": sha256(Path(gpu_probe.__file__)),
            "runner_source_sha256": sha256(Path(__file__)),
            "metric_source_sha256": sha256(Path(pilot_metrics.__file__)),
            "inputs": cohort[stage], "gpu_fit_compute_ceiling_seconds": 12 * 600,
            "trigger": trigger,
            "confirmation_rule": "For each family require six successful discovery fits, all charged artifacts within L3, zero exact train/validation row copies, and median CatBoost retention higher than the corresponding original GPU family. This is diagnostic promotion, not production PTF selection.",
            "ptf_v1": None, "mfs_v2": None}
    path = root / f"{stage}.lock.json"
    if path.exists():
        if json.loads(path.read_text()) != lock:
            raise ValueError("conditional GPU round changed")
    else:
        write_once(path, lock)
    return path


def run(stage: str, root: Path, binary: Path, trigger: str) -> None:
    import torch
    lock_path = freeze(stage, root, binary, trigger)
    locked = json.loads(lock_path.read_text())
    library = str(Path(torch.__file__).parent / "lib")
    env = {**os.environ, "LD_LIBRARY_PATH": library + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (SCRATCH / "neural-native-v1/xbabe2.gpu-reservation").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, job in enumerate(locked["gpu_jobs"]):
            output = root / stage / f"cell-{index:02d}.json"
            if output.exists():
                old = json.loads(output.read_text())
                fit_path = Path(old["fit"]["receipt"])
                if (old["job"] != job or old["round_sha256"] != sha256(lock_path)
                        or sha256(fit_path) != old["fit"]["receipt_sha256"]):
                    raise ValueError("resumed conditional identity changed")
                original = json.loads(fit_path.read_text())
                if original["artifact_sha256"] is not None and sha256(fit_path.with_suffix(".dpk")) != original["artifact_sha256"]:
                    raise ValueError("resumed conditional artifact changed")
                if old["common_validation"] is not None and (
                        sha256(fit_path.parent / "sample.csv") != old["sample_sha256"]
                        or sha256(fit_path.parent / "common-validation.json") != old["metric_sha256"]):
                    raise ValueError("resumed conditional metric changed")
                continue
            snapshot = local()
            if not admitted(snapshot):
                raise ValueError("conditional GPU admission blocked")
            worker = WORKERS / job["dataset"]
            reference = next(row for row in locked["inputs"] if row["id"] == job["dataset"])
            if any(sha256(worker / f"{part}.csv") != reference[f"{part}_sha256"]
                   for part in ("train", "validation")):
                raise ValueError("conditional GPU cohort input changed")
            fitted = gpu_probe.run(worker, binary, job["candidate"], 11, 2.0, 0.0,
                                   root / stage / "fits", lock_path)
            fit_path = Path(fitted["receipt"])
            receipt = json.loads(fit_path.read_text())
            result = {"job": job, "round_sha256": sha256(lock_path), "fit": fitted,
                      "host_admission": snapshot, "validation_only": True,
                      "ptf_v1": None, "mfs_v2": None, "common_validation": None}
            if fitted["status"] == "ok":
                sample = fit_path.parent / "sample.csv"
                command = [str(binary), "sample", "--kernel", str(fit_path.with_suffix(".dpk")),
                           "--rows", str(reference["train_rows"]), "--seed", "101", "--out", str(sample)]
                subprocess.run(command, env=env, capture_output=True, check=True, timeout=600)
                first = sha256(sample)
                repeated = fit_path.parent / "repeat.csv"
                subprocess.run(command[:-1] + [str(repeated)], env=env, capture_output=True, check=True, timeout=600)
                if first != sha256(repeated):
                    raise ValueError("conditional artifact sampling changed")
                affinity = os.sched_getaffinity(0)
                try:
                    os.sched_setaffinity(0, sorted(affinity)[:16])
                    metric = pilot_metrics.measure(worker / "train.csv", worker / "validation.csv", sample, "regression")
                finally:
                    os.sched_setaffinity(0, affinity)
                metric_path = fit_path.parent / "common-validation.json"
                write_once(metric_path, metric)
                result.update({"sample_sha256": first, "metric_sha256": sha256(metric_path),
                               "common_validation": metric, "artifact_bytes": receipt["artifact_bytes"]})
            write_once(output, result)
            print(json.dumps({"dataset": job["dataset"], "candidate": job["candidate"],
                              "status": fitted["status"], "bytes": receipt["artifact_bytes"],
                              "catboost_retention": result["common_validation"]["utility"]["catboost"].get("retention")
                              if result["common_validation"] else None}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("discovery", "confirmation"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--trigger", default="Prior joint micro-TVAE configurations lost utility to the symbolic default on all six confirmation datasets; test the existing conditional GPU target models.")
    args = parser.parse_args()
    run(args.stage, args.root, args.binary, args.trigger)
