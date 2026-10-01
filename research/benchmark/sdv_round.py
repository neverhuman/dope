"""Frozen CTGAN/TVAE validation research, with bounded isolated attempts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .fetch_jope import LIMIT, used_bytes
from .gpu_probe import gpu_snapshot
from .inventory_hosts import local
from .manifest import digest
from .score import artifact_inventory, sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
ROOT = SCRATCH / "neural-native-v1"
WORKERS = SCRATCH / "s3-v1/prepared/worker"


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def environment() -> dict:
    from importlib.metadata import version
    return {name: version(name) for name in (
        "ctgan", "rdt", "sdmetrics", "copulas", "numpy", "pandas", "scipy",
        "scikit-learn", "torch", "catboost", "Faker", "python-dateutil", "tqdm", "plotly")}


def check_lock(lock_path: Path) -> dict:
    lock = json.loads(lock_path.read_text())
    if lock.get("validation_only") is not True or lock.get("format") != "dope-sdv-validation-round":
        raise ValueError("invalid validation round")
    if environment() != lock["environment"]:
        raise ValueError("runtime differs from frozen environment")
    here = Path(__file__).parent
    for name, expected in lock["source_files"].items():
        if sha256(here / name) != expected:
            raise ValueError("research source changed")
    for name, expected in lock["dependency_files"].items():
        if sha256(ROOT / "deps" / name) != expected:
            raise ValueError("dependency source changed")
    return lock


def process(request: dict, prefix: Path, seconds: float, gpu: bool = False) -> dict:
    write_once(prefix.with_suffix(".request.json"), request)
    command = [sys.executable, "-m", "research.benchmark.sdv_adapter",
               str(prefix.with_suffix(".request.json"))]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0" if gpu else "", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "AWS_SHARED_CREDENTIALS_FILE": "/dev/null", "AWS_CONFIG_FILE": "/dev/null",
                "AWS_EC2_METADATA_DISABLED": "true"})
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        env.pop(name, None)
    start = last = time.monotonic()
    peak = 0
    energy = 0.0
    power_samples = 0
    reason = None
    with prefix.with_suffix(".stdout").open("w") as stdout, prefix.with_suffix(".stderr").open("w") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        while child.poll() is None:
            time.sleep(0.5)
            now = time.monotonic()
            if gpu and now - last >= 2:
                try:
                    sample = gpu_snapshot()
                    peak = max(peak, sample["used_mib"])
                    if sample["power_watts"] is not None:
                        energy += sample["power_watts"] * (now - last)
                        power_samples += 1
                except (OSError, ValueError, subprocess.SubprocessError):
                    reason = "resource_monitor_failed"
                last = now
                if peak > 16 * 1024:
                    reason = "gpu_memory_limit"
            if now - start >= seconds:
                reason = "timeout"
            if reason:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
    result = {"wall_seconds": time.monotonic() - start, "exit_code": child.returncode,
              "status": reason or ("ok" if child.returncode == 0 else "failed"),
              "peak_device_used_mib": peak if gpu else None,
              "energy_joules_estimate": energy if power_samples else None,
              "power_samples": power_samples,
              "stdout_sha256": sha256(prefix.with_suffix(".stdout")),
              "stderr_sha256": sha256(prefix.with_suffix(".stderr"))}
    if result["status"] == "ok":
        try:
            result["child"] = json.loads(prefix.with_suffix(".stdout").read_text().splitlines()[-1])
        except (ValueError, IndexError):
            result["status"] = "malformed_receipt"
    write_once(prefix.with_suffix(".receipt.json"), result)
    if result["status"] != "ok":
        raise RuntimeError(result["status"])
    return result


def verify_attempt(root: Path, receipt: dict) -> None:
    for name, expected in receipt["evidence_files"].items():
        if sha256(root / name) != expected:
            raise ValueError("resumed attempt evidence changed")
    if receipt["status"] == "ok":
        inventory, charged = artifact_inventory(root / "artifact", ["model.pt", "model.json", "projection.json"])
        if inventory != receipt["artifact_inventory"] or charged != receipt["artifact_bytes"]:
            raise ValueError("resumed artifact changed")


def run_job(lock_path: Path, job: dict, retry_failed: bool = False) -> dict:
    lock = check_lock(lock_path)
    if job not in lock["jobs"]:
        raise ValueError("job absent from frozen matrix")
    worker = WORKERS / job["dataset"]
    for part in ("train", "validation"):
        if sha256(worker / f"{part}.csv") != job[f"{part}_sha256"]:
            raise ValueError("worker partition changed")
    if sha256(worker / "projection.json") != job["projection_sha256"]:
        raise ValueError("worker projection changed")
    root = lock_path.parent / "jobs" / digest(job)
    root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(root.glob("attempt-*/receipt.json"))
    if attempts:
        old = json.loads(attempts[-1].read_text())
        if old["round_sha256"] != sha256(lock_path) or old["job"] != job:
            raise ValueError("retry identity changed")
        verify_attempt(attempts[-1].parent, old)
        if old["status"] == "ok" or not retry_failed:
            return old
    prior = []
    for p in (lock_path.parent / "jobs").glob("*/attempt-*/receipt.json"):
        item = json.loads(p.read_text())
        if (item["job"]["dataset"], item["job"]["method"]) == (job["dataset"], job["method"]):
            prior.append(item)
    if len(prior) >= 8 or sum(item["wall_seconds"] for item in prior) >= 43_200:
        raise ValueError("per-cell trial or wall budget exhausted")
    snapshot = local()
    if (snapshot["active_gpu_processes"] or snapshot["gpus"][0]["memory_free_mib"] < 17 * 1024
            or len(snapshot["allowed_cpus"]) < 16 or snapshot["memory"]["MemAvailable"] < 24 * 1024**3
            or snapshot["scratch_disk"]["free_bytes"] < 5_000_000_000
            or snapshot["load_average"][0] + 16 > 1.25 * len(snapshot["allowed_cpus"])
            or used_bytes(SCRATCH) + 1_000_000_000 > LIMIT):
        return {"status": "admission_blocked", "job": job}
    attempt = root / f"attempt-{len(attempts) + 1:04d}"
    attempt.mkdir()
    write_once(attempt / "host.json", snapshot)
    original_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, snapshot["allowed_cpus"][:16])
    started = time.monotonic()
    receipt = {"format": "dope-sdv-validation-attempt", "version": 1, "job": job,
               "round_sha256": sha256(lock_path), "host": snapshot["host"],
               "cpu_affinity": snapshot["allowed_cpus"][:16], "status": "failed",
               "validation_only": True, "ptf_v1": None, "mfs_v2": None}
    try:
        fitted = process({"action": "fit", "method": job["method"], "config": job["config"],
                          "train": str(worker / "train.csv"), "seed": job["fit_seed"],
                          "artifact": str(attempt / "artifact")}, attempt / "fit", 600, gpu=True)
        artifact = attempt / "artifact"
        (artifact / "projection.json").write_bytes((worker / "projection.json").read_bytes())
        inventory, size = artifact_inventory(artifact, ["model.pt", "model.json", "projection.json"])
        sample_request = {"action": "sample", "artifact": str(artifact), "rows": job["train_rows"],
                          "seed": job["sample_seed"], "output": str(attempt / "sample.csv")}
        sampled = process(sample_request, attempt / "sample", 600)
        repeated = process({**sample_request, "output": str(attempt / "repeat.csv")}, attempt / "repeat", 600)
        if sampled["child"]["sha256"] != repeated["child"]["sha256"]:
            raise ValueError("artifact sampling is not deterministic")
        native = process({"action": "efficacy", "synthetic": str(attempt / "sample.csv"),
                          "validation": str(worker / "validation.csv")}, attempt / "native", 600)
        receipt.update({"status": "ok", "artifact_inventory": inventory, "artifact_bytes": size,
                        "native_kpi": native["child"], "fit": fitted,
                        "sample": sampled, "sampling_repeated_sha256": repeated["child"]["sha256"]})
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        receipt["error_type"] = type(error).__name__
        fit_receipt = attempt / "fit.receipt.json"
        if fit_receipt.exists():
            receipt["fit"] = json.loads(fit_receipt.read_text())
            if receipt["fit"]["status"] != "ok":
                receipt["status"] = receipt["fit"]["status"]
    finally:
        os.sched_setaffinity(0, original_affinity)
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["evidence_files"] = {p.relative_to(attempt).as_posix(): sha256(p)
                                 for p in sorted(attempt.rglob("*")) if p.is_file()}
    write_once(attempt / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    lock = check_lock(args.lock)
    host = os.uname().nodename.split(".")[0]
    guard_path = ROOT / f"{host}.gpu-reservation"
    with guard_path.open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, job in enumerate(lock["jobs"]):
            if index % args.shards == args.shard:
                outcome = run_job(args.lock, job, args.retry_failed)
                print(json.dumps({"job": digest(job), "dataset": job["dataset"],
                                  "method": job["method"], "trial": job["trial"],
                                  "status": outcome["status"]}), flush=True)


if __name__ == "__main__":
    main()
