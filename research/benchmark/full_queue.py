"""Resumable three-host dispatcher for an admitted, frozen benchmark matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .admission import assess
from .fetch_jope import LIMIT, used_bytes
from .score import CONTRACT, artifact_inventory, sha256


HOSTS = ("xbabe1", "xbabe2", "xbabe3")
ROOT = Path("/mnt/fast-scratch/dope-benchmark")
SLOT_CORES = 16
MAX_SLOTS = 4
MIN_FREE_DISK = 5_000_000_000
MIN_FREE_MEMORY = 4_000_000_000
GPU_HEADROOM_MIB = 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def probe(host: str) -> dict:
    # The script is sent over stdin so probes do not depend on a mutable remote checkout.
    source = Path(__file__).with_name("inventory_hosts.py").read_text()
    command = ["python3", "-", "--local"]
    if host != "xbabe2":
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                   host, *command]
    result = subprocess.run(command, input=source, text=True, capture_output=True,
                            check=True, timeout=25)
    snapshot = json.loads(result.stdout)
    if snapshot["host"].split(".")[0] != host:
        raise ValueError("host inventory identity mismatch")
    return snapshot


def cpu_slots(snapshot: dict) -> list[tuple[int, ...]]:
    allowed = snapshot["allowed_cpus"]
    return [tuple(allowed[index:index + SLOT_CORES])
            for index in range(0, min(len(allowed), MAX_SLOTS * SLOT_CORES), SLOT_CORES)
            if len(allowed[index:index + SLOT_CORES]) == SLOT_CORES]


def available_slot(snapshot: dict, occupied: set[tuple[int, ...]], cell: dict,
                   reserved_memory_bytes: int = 0) -> tuple[int, ...] | None:
    allowed = snapshot["allowed_cpus"]
    # The load check accounts for workloads outside this queue, including owners' jobs.
    if snapshot["load_average"][0] + SLOT_CORES > len(allowed) * 1.25:
        return None
    needed = max(MIN_FREE_MEMORY, cell["memory_reservation_bytes"])
    if snapshot["memory"]["MemAvailable"] - reserved_memory_bytes < needed:
        return None
    if cell["requires_gpu"]:
        if snapshot["active_gpu_processes"]:
            return None
        if not any(gpu["memory_free_mib"] >= cell["gpu_vram_mib"] + GPU_HEADROOM_MIB
                   for gpu in snapshot["gpus"]):
            return None
    return next((slot for slot in cpu_slots(snapshot) if slot not in occupied), None)


def validate_cell(cell: dict, methods: dict) -> None:
    required = ("id", "dataset", "method", "panel", "track", "tier", "applicable",
                "fit_seed", "sample_seeds", "size_multipliers", "worker_dir",
                "scratch_reservation_bytes", "memory_reservation_bytes", "requires_gpu",
                "gpu_vram_mib", "timeout_seconds", "runtime_python", "configuration")
    if any(key not in cell for key in required):
        raise ValueError("matrix cell missing dispatch fields")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cell["id"]) or ".." in cell["id"]:
        raise ValueError("invalid cell identity")
    if cell["applicable"] and methods["methods"][cell["method"]]["status"] != "locked":
        raise ValueError("applicable method is not locked")
    if cell["track"] != "common_numeric":
        raise ValueError("author-faithful adapter is not available to this runner")
    if (cell["fit_seed"] not in CONTRACT["fit_seeds"]
            or cell["sample_seeds"] != CONTRACT["sample_seeds"]
            or cell["size_multipliers"] != CONTRACT["size_multipliers"]):
        raise ValueError("final cell has incomplete sample matrix")
    choice = cell["configuration"]
    if not isinstance(choice, dict) or choice.get("kind") not in ("default", "tuned"):
        raise ValueError("final cell has no frozen configuration")
    method = methods["methods"][cell["method"]]
    if method.get("group") == "dp":
        if cell.get("dp_epsilon") not in (1, 4, 10):
            raise ValueError("final DP cell has no frozen epsilon")
    elif "dp_epsilon" in cell:
        raise ValueError("DP epsilon on non-DP cell")
    if (cell["scratch_reservation_bytes"] <= 0 or cell["memory_reservation_bytes"] <= 0
            or cell["timeout_seconds"] <= 0 or cell["gpu_vram_mib"] < 0):
        raise ValueError("invalid cell resource reservation")
    if not Path(cell["worker_dir"]).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("worker data must stay on benchmark scratch")


def latest_attempt(attempt_dir: Path) -> dict | None:
    paths = sorted(attempt_dir.glob("attempt-*.json"))
    return json.loads(paths[-1].read_text()) if paths else None


def verify_success(receipt: dict, queue_root: Path) -> None:
    if receipt.get("status") != "ok":
        raise ValueError("queue attempt is not successful")
    name, attempt = receipt["cell_id"], receipt["attempt"]
    inventory = queue_root / "inventories" / f"{name}-attempt-{attempt:04d}.json"
    log = queue_root / "logs" / f"{name}-attempt-{attempt:04d}.log"
    if sha256(inventory) != receipt["inventory_sha256"] or sha256(log) != receipt["log_sha256"]:
        raise ValueError("successful attempt inventory or log changed")
    children = receipt["runner_receipts"]
    if not children or len({item["run_key"] for item in children}) != len(children):
        raise ValueError("successful attempt has missing or duplicate sample receipts")
    fit_keys = {item["fit_key"] for item in children}
    if len(fit_keys) != 1:
        raise ValueError("successful attempt has multiple fit identities")
    fit_dir = queue_root / "results" / fit_keys.pop()
    fit = json.loads((fit_dir / "fit-receipt.json").read_text())
    if fit.get("status") != "ok":
        raise ValueError("successful attempt has no valid fit receipt")
    inventory_now, _ = artifact_inventory(fit_dir / "artifact", fit["artifact_files"])
    if inventory_now != fit["artifact_inventory"]:
        raise ValueError("successful attempt artifact changed")
    if fit.get("fit_evidence_files"):
        evidence_now, _ = artifact_inventory(fit_dir / "fit_evidence", fit["fit_evidence_files"])
        if evidence_now != fit["fit_evidence_inventory"]:
            raise ValueError("successful attempt fit evidence changed")
    for child in children:
        if child["status"] != "ok":
            raise ValueError("successful attempt contains a failed sample")
        run_key = child["run_key"]
        stored = json.loads((fit_dir / f"{run_key}.receipt.json").read_text())
        if stored != child or sha256(fit_dir / f"{run_key}.csv") != child["sample_sha256"]:
            raise ValueError("successful attempt sample or receipt changed")


def execute(cell: dict, host: str, slot: tuple[int, ...], attempt: int,
            package: Path, results: Path, queue_root: Path, inventory: dict) -> dict:
    name = cell["id"]
    start = time.monotonic()
    job = {"worker_dir": cell["worker_dir"], "method": cell["method"],
           "fit_seed": cell["fit_seed"], "sample_seeds": cell["sample_seeds"],
           "size_multipliers": cell["size_multipliers"], "track": cell["track"],
           "final": True, "configuration": cell["configuration"],
           "scratch_root": str(ROOT), "dope_binary": str(ROOT / "bin/dope-kernel")}
    if "dp_epsilon" in cell:
        job["dp_epsilon"] = cell["dp_epsilon"]
    job_path = queue_root / "jobs" / f"{name}.json"
    if job_path.exists():
        if json.loads(job_path.read_text()) != job:
            raise ValueError("job changed after first attempt")
    else:
        write_once(job_path, job)
    inventory_path = queue_root / "inventories" / f"{name}-attempt-{attempt:04d}.json"
    write_once(inventory_path, inventory)
    env = ["env", "-u", "AWS_ACCESS_KEY_ID", "-u", "AWS_SECRET_ACCESS_KEY",
           "-u", "AWS_SESSION_TOKEN", "-u", "AWS_PROFILE", "-u", "AWS_DEFAULT_PROFILE",
           "-u", "AWS_ROLE_ARN", "-u", "AWS_WEB_IDENTITY_TOKEN_FILE",
           "-u", "AWS_CONTAINER_CREDENTIALS_FULL_URI", "-u", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
           "AWS_SHARED_CREDENTIALS_FILE=/dev/null", "AWS_CONFIG_FILE=/dev/null",
           "AWS_EC2_METADATA_DISABLED=true", f"HOME={queue_root / 'worker-home'}",
           f"PYTHONPATH={package}",
           f"CUDA_VISIBLE_DEVICES={'0' if cell['requires_gpu'] else ''}",
           "OMP_NUM_THREADS=16", "OPENBLAS_NUM_THREADS=16", "MKL_NUM_THREADS=16",
           "RAYON_NUM_THREADS=16"]
    command = ["timeout", "--signal=TERM", "--kill-after=20s", str(cell["timeout_seconds"]),
               "/usr/bin/time", "-v", "taskset", "-c", ",".join(map(str, slot)),
               *env, cell["runtime_python"], "-m", "research.benchmark.runner",
               str(job_path), str(package / "research/benchmark/methods.lock.json"), str(results)]
    if host != "xbabe2":
        command = ["ssh", "-o", "BatchMode=yes", host, shlex.join(command)]
    try:
        process = subprocess.run(command, capture_output=True, text=True,
                                 timeout=cell["timeout_seconds"] + 40)
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
        child = json.loads(stdout.strip().splitlines()[-1]) if returncode == 0 else []
        expected = len(cell["sample_seeds"]) * len(cell["size_multipliers"])
        status = ("ok" if len(child) == expected and all(item["status"] == "ok" for item in child)
                  else "partial" if child else "timeout" if returncode == 124 else "failed")
        error_type = None
    except subprocess.TimeoutExpired as error:
        stdout, stderr, returncode = error.stdout or b"", error.stderr or b"", 124
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        child, status, error_type = [], "timeout", "CoordinatorTimeout"
    except json.JSONDecodeError:
        child, status, error_type = [], "failed", "MalformedRunnerReceipt"
    log_path = queue_root / "logs" / f"{name}-attempt-{attempt:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout + "\n--- stderr ---\n" + stderr)
    receipt = {"format": "dope-benchmark-queue-attempt", "version": 1,
               "cell_id": name, "attempt": attempt, "host": host, "cpu_affinity": list(slot),
               "gpu_index": 0 if cell["requires_gpu"] else None,
               "status": status, "runner_exit_code": returncode,
               "runner_receipt_count": len(child), "runner_receipts": child,
               "elapsed_seconds": time.monotonic() - start, "completed_utc": utc_now(),
               "inventory_sha256": sha256(inventory_path), "log_sha256": sha256(log_path),
               "error_type": error_type}
    write_once(queue_root / "attempts" / name / f"attempt-{attempt:04d}.json", receipt)
    return receipt


def _run_locked(repo_root: Path, lock_root: Path, package: Path, queue_root: Path,
                retry_failed: bool, max_idle_seconds: int) -> dict:
    decision = assess(repo_root, lock_root)
    if not decision["admitted"]:
        raise ValueError("final benchmark admission blocked: " + ", ".join(decision["blockers"]))
    methods = json.loads((lock_root / "methods.lock.json").read_text())
    matrix = json.loads((lock_root / "method-dataset-matrix.lock.json").read_text())
    if sha256(package / "research/benchmark/methods.lock.json") != sha256(lock_root / "methods.lock.json"):
        raise ValueError("deployed method lock differs")
    cells = [cell for cell in matrix["cells"] if cell["applicable"]]
    if len({cell["id"] for cell in cells}) != len(cells):
        raise ValueError("duplicate cell identity")
    for cell in cells:
        validate_cell(cell, methods)
    if not queue_root.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("queue receipts must stay on benchmark scratch")
    queue_root.mkdir(parents=True, exist_ok=True)
    (queue_root / "worker-home").mkdir(exist_ok=True)
    pending = {cell["id"]: cell for cell in cells}
    for name in list(pending):
        previous = latest_attempt(queue_root / "attempts" / name)
        if previous and previous["status"] == "ok":
            expected = len(pending[name]["sample_seeds"]) * len(pending[name]["size_multipliers"])
            if previous["runner_receipt_count"] != expected:
                raise ValueError("successful attempt has incomplete sample matrix")
            verify_success(previous, queue_root)
        if previous and (previous["status"] == "ok" or not retry_failed):
            pending.pop(name)
    active: dict[concurrent.futures.Future, tuple[str, tuple[int, ...], dict, int]] = {}
    started = 0
    coordinator_failures = 0
    last_admission = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(HOSTS) * MAX_SLOTS) as executor:
        while pending or active:
            completed = [future for future in active if future.done()]
            for future in completed:
                host, slot, cell, attempt = active[future]
                try:
                    future.result()
                except (OSError, ValueError, RuntimeError, TimeoutError, KeyError,
                        subprocess.SubprocessError) as error:
                    coordinator_failures += 1
                    receipt_path = (queue_root / "attempts" / cell["id"]
                                    / f"attempt-{attempt:04d}.json")
                    if not receipt_path.exists():
                        write_once(receipt_path, {
                            "format": "dope-benchmark-queue-attempt", "version": 1,
                            "cell_id": cell["id"], "attempt": attempt, "host": host,
                            "cpu_affinity": list(slot), "status": "failed",
                            "error_type": type(error).__name__, "completed_utc": utc_now()})
                del active[future]
            admitted = False
            for name, cell in list(pending.items()):
                if len(active) >= len(HOSTS) * MAX_SLOTS:
                    break
                if used_bytes(ROOT) + sum(item[2]["scratch_reservation_bytes"] for item in active.values()) + cell["scratch_reservation_bytes"] > LIMIT:
                    continue
                for host in HOSTS:
                    try:
                        snapshot = probe(host)
                    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
                        continue
                    if snapshot["scratch_disk"]["free_bytes"] < max(MIN_FREE_DISK, cell["scratch_reservation_bytes"]):
                        continue
                    occupied = {slot for running_host, slot, _, _ in active.values() if running_host == host}
                    if cell["requires_gpu"] and any(running_host == host and running["requires_gpu"]
                                                     for running_host, _, running, _ in active.values()):
                        continue
                    reserved_memory = sum(running["memory_reservation_bytes"]
                                          for running_host, _, running, _ in active.values()
                                          if running_host == host)
                    slot = available_slot(snapshot, occupied, cell, reserved_memory)
                    if slot is None:
                        continue
                    attempt_dir = queue_root / "attempts" / name
                    attempt = len(list(attempt_dir.glob("attempt-*.json"))) + 1
                    future = executor.submit(execute, cell, host, slot, attempt,
                                             package, queue_root / "results", queue_root, snapshot)
                    active[future] = (host, slot, cell, attempt)
                    pending.pop(name)
                    started += 1
                    admitted = True
                    last_admission = time.monotonic()
                    break
            if not active and not pending:
                break
            if not admitted and time.monotonic() - last_admission >= max_idle_seconds:
                break
            time.sleep(20 if not completed and not admitted else 1)
    return {"format": "dope-benchmark-queue-summary", "version": 1,
            "started": started, "coordinator_failures": coordinator_failures,
            "pending": sorted(pending), "completed_utc": utc_now()}


def run(repo_root: Path, lock_root: Path, package: Path, queue_root: Path,
        retry_failed: bool = False, max_idle_seconds: int = 600) -> dict:
    decision = assess(repo_root, lock_root)
    if not decision["admitted"]:
        raise ValueError("final benchmark admission blocked: " + ", ".join(decision["blockers"]))
    if not queue_root.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("queue receipts must stay on benchmark scratch")
    queue_root.mkdir(parents=True, exist_ok=True)
    with (queue_root / "coordinator.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another benchmark coordinator holds this queue") from None
        result = _run_locked(repo_root, lock_root, package, queue_root,
                             retry_failed, max_idle_seconds)
        write_once(queue_root / "summaries" / f"summary-{time.time_ns()}.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--lock-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--package", type=Path, default=ROOT / "final/package")
    parser.add_argument("--queue-root", type=Path, default=ROOT / "final/queue")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-idle-seconds", type=int, default=600)
    args = parser.parse_args()
    try:
        result = run(args.repo_root, args.lock_root, args.package, args.queue_root,
                     args.retry_failed, args.max_idle_seconds)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
