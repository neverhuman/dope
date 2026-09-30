"""Run a frozen research grid, one admitted GPU fit per host."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from .manifest import digest
from .score import sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark/gpu-research-v1")
PACKAGE = ROOT / "package"


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def on_host(host: str, command: list[str], timeout: int) -> subprocess.CompletedProcess:
    if host == os.uname().nodename:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                           host, shlex.join(command)], capture_output=True, text=True,
                          timeout=timeout)


def admitted(snapshot: dict) -> bool:
    return (len(snapshot["allowed_cpus"]) >= 16
            and snapshot["memory"]["MemAvailable"] >= 24 * 1024**3
            and snapshot["scratch_disk"]["free_bytes"] >= 5_000_000_000
            and not snapshot["active_gpu_processes"]
            and any(gpu["memory_free_mib"] >= 17 * 1024 for gpu in snapshot["gpus"])
            and snapshot["load_average"][0] + 16 <= 1.25 * len(snapshot["allowed_cpus"]))


def run_job(host: str, job: dict, lock_path: Path, binary: Path,
            retry_failed: bool, round_name: str) -> dict:
    key = digest(job)
    job_dir = ROOT / f"{round_name}-queue" / key
    job_dir.mkdir(parents=True, exist_ok=True)
    previous = sorted(path for path in job_dir.glob("attempt-*.json")
                      if re.fullmatch(r"attempt-\d{4}\.json", path.name))
    if previous:
        last = json.loads(previous[-1].read_text())
        if last["status"] == "ok" or not retry_failed:
            return {"job": job, "status": last["status"], "attempt_sha256": sha256(previous[-1])}
    attempt = len(previous) + 1
    prefix = job_dir / f"attempt-{attempt:04d}"
    inventory_command = ["env", f"PYTHONPATH={PACKAGE}", "python3", "-m",
                         "research.benchmark.inventory_hosts", "--local"]
    try:
        probe = on_host(host, inventory_command, 30)
        if probe.returncode != 0:
            raise ValueError("host inventory failed")
        snapshot = json.loads(probe.stdout)
        if snapshot["host"].split(".")[0] != host:
            raise ValueError("host identity mismatch")
        write_once(Path(f"{prefix}.inventory.json"), snapshot)
        if not admitted(snapshot):
            status, reason, child, output = "admission_blocked", "insufficient_reserved_resources", None, ""
        else:
            command = ["env", "-u", "AWS_ACCESS_KEY_ID", "-u", "AWS_SECRET_ACCESS_KEY",
                       "-u", "AWS_SESSION_TOKEN", "-u", "AWS_PROFILE",
                       "AWS_SHARED_CREDENTIALS_FILE=/dev/null", "AWS_CONFIG_FILE=/dev/null",
                       "AWS_EC2_METADATA_DISABLED=true", f"PYTHONPATH={PACKAGE}",
                       "python3", "-m", "research.benchmark.gpu_probe",
                       str(Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker") / job["dataset"]),
                       str(binary), job["candidate"], "--seed", str(job["seed"]),
                       "--target-weight", str(job["target_weight"]),
                       "--structural-penalty", str(job["structural_penalty"]),
                       "--output-root", str(ROOT / f"{round_name}-fits"),
                       "--round-lock", str(lock_path)]
            result = on_host(host, command, 700)
            output = result.stdout + "\n" + result.stderr
            child = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else None
            status = child["status"] if child else "failed"
            reason = None if child else "probe_exit_or_receipt_failure"
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        status, reason, child, output = "failed", type(error).__name__, None, ""
    log = Path(f"{prefix}.log")
    log.write_text(output)
    inventory = Path(f"{prefix}.inventory.json")
    receipt = {"format": "dope-benchmark-gpu-round-dispatch", "version": 1,
               "round_lock_sha256": sha256(lock_path), "job": job, "host": host,
               "attempt": attempt, "status": status, "reason_type": reason,
               "inventory_sha256": sha256(inventory) if inventory.exists() else None,
               "log_sha256": sha256(log), "child": child}
    receipt_path = Path(f"{prefix}.json")
    write_once(receipt_path, receipt)
    return {"job": job, "status": status, "attempt_sha256": sha256(receipt_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=ROOT / "round0.lock.json")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    round_name = f"round{lock['round']}"
    package_manifest = json.loads((ROOT / "package-manifest.json").read_text())
    for name, expected in package_manifest["files"].items():
        if sha256(PACKAGE / "research/benchmark" / name) != expected:
            raise ValueError("deployed research package changed")
    binary = ROOT / "bin" / f"dope-gpu-{lock['gpu_binary_sha256'][:16]}"
    if (sha256(binary) != lock["gpu_binary_sha256"]
            or package_manifest["files"]["gpu_probe.py"] != lock["probe_source_sha256"]):
        raise ValueError("GPU binary or probe differs from round lock")
    hosts = lock["gpu_hosts"]
    assignments = {host: [] for host in hosts}
    for index, job in enumerate(lock["gpu_jobs"]):
        assignments[hosts[index % len(hosts)]].append(job)
    def run_host(host: str) -> list[dict]:
        return [run_job(host, job, args.lock, binary, args.retry_failed, round_name)
                for job in assignments[host]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        batches = list(pool.map(run_host, hosts))
    outcomes = [item for batch in batches for item in batch]
    print(json.dumps({"jobs": len(outcomes), "status_counts": {
        status: sum(item["status"] == status for item in outcomes)
        for status in sorted({item["status"] for item in outcomes})}}, sort_keys=True))


if __name__ == "__main__":
    main()
