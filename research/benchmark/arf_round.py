"""Bounded author-ARF native-likelihood pilot on training-derived validation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from . import arf_native
from .fetch_jope import LIMIT, used_bytes
from .inventory_hosts import local
from .manifest import digest
from .score import artifact_inventory, sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
ROOT = SCRATCH / "pilot-24h"
OUT = ROOT / "arf-native-v1"
WORKERS = ROOT / "prepared-v2/worker"
DATASET_LOCK = Path(__file__).with_name("pilot-24h-datasets.lock.json")
METHOD_LOCK = Path(__file__).with_name("methods.lock.json")
SOURCE = SCRATCH / "source-snapshots/arfpy-8b63c1b.tar.gz"
SOURCE_DIR = (SCRATCH / "source-snapshots/arfpy-unpacked/"
              "arfpy-8b63c1b3999981125b4af2828ff52cba8e29169d")
CAP = 10_240
TIMEOUT = 900
SLOT = list(range(48, 64))
CONFIGS = (
    {"num_trees": 30, "min_node_size": 5, "max_iters": 10, "alpha": 0.0},
    {"num_trees": 30, "min_node_size": 10, "max_iters": 10, "alpha": 0.0},
    {"num_trees": 30, "min_node_size": 5, "max_iters": 10, "alpha": 0.5},
    {"num_trees": 50, "min_node_size": 5, "max_iters": 10, "alpha": 0.5},
)


def source_files() -> dict[str, str]:
    expected = {}
    with tarfile.open(SOURCE, "r:gz") as archive:
        for name in ("arf.py", "utils.py"):
            members = [member for member in archive.getmembers()
                       if member.name.endswith("/arfpy/" + name) and member.isfile()]
            if len(members) != 1:
                raise ValueError("ARF source archive lacks unique author module")
            stream = archive.extractfile(members[0])
            if stream is None:
                raise ValueError("ARF author module cannot be read")
            digest_value = hashlib.sha256(stream.read()).hexdigest()
            if digest_value != sha256(SOURCE_DIR / "arfpy" / name):
                raise ValueError("extracted ARF source differs from pinned archive")
            expected[name] = digest_value
    return expected


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def runtime() -> dict:
    from importlib.metadata import version
    return {name: version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn")}


def freeze() -> Path:
    method = json.loads(METHOD_LOCK.read_text())["methods"]["ARF"]
    if (method["source_sha256"] != sha256(SOURCE) or method["license"] != "MIT"
            or method["upstream_commit"] != "8b63c1b3999981125b4af2828ff52cba8e29169d"):
        raise ValueError("ARF author source or rights changed")
    datasets = {row["id"]: row for row in json.loads(DATASET_LOCK.read_text())["datasets"]}
    jobs = []
    for dataset in ("Adult", "California", "News"):
        worker = WORKERS / dataset
        record = datasets[dataset]
        if ((worker / "test.csv").exists() or "evaluator" in worker.parts
                or any(sha256(worker / f"{part}.csv") != record["projected_files"][part]
                       for part in ("train", "validation"))
                or sha256(worker / "projection.json") != record["projection_sha256"]):
            raise ValueError("ARF worker differs from rights-cleared training partitions")
        for trial, config in enumerate(CONFIGS):
            jobs.append({"dataset": dataset, "trial": trial, "config": config,
                         "fit_seed": 11, "sample_seed": 101,
                         "train_sha256": record["projected_files"]["train"],
                         "validation_sha256": record["projected_files"]["validation"],
                         "projection_sha256": record["projection_sha256"],
                         "task": record["task"]})
    start = json.loads((ROOT / "start-v2.json").read_text())
    lock = {"format": "dope-arf-native-validation-round", "version": 1,
            "scope": "training_derived_validation_only", "official_tests_opened": False,
            "source_archive_sha256": sha256(SOURCE),
            "source_files": source_files(),
            "round_source_sha256": sha256(Path(__file__)),
            "adapter_source_sha256": sha256(Path(arf_native.__file__)),
            "dataset_lock_sha256": sha256(DATASET_LOCK),
            "method_lock_sha256": sha256(METHOD_LOCK),
            "pilot_start_sha256": sha256(ROOT / "start-v2.json"),
            "pilot_deadline_utc": start["deadline_utc"],
            "runtime": runtime(), "cpu_slot": SLOT,
            "fit_timeout_seconds": TIMEOUT,
            "tuning_trials_per_dataset": 4,
            "tuning_wall_seconds_per_dataset": 43_200,
            "scratch_reservation_bytes": 2_000_000_000,
            "native_objective": {"name": "heldout_forde_mean_log_density",
                                 "direction": "maximize",
                                 "implementation_sha256": sha256(Path(arf_native.__file__)),
                                 "tie_breaks": ["artifact_bytes_ascending",
                                                "config_sha256_ascending"]},
            "config_grid": list(CONFIGS), "jobs": jobs,
            "artifact_policy": "restricted_numeric_forde_factors_without_full_training_rows",
            "formal_dp_claim": False, "mfs_v2": None, "ptf_v1": None,
            "production_certified": False}
    path = OUT / "round.lock.json"
    write_once(path, lock)
    return path


def check_lock(path: Path) -> dict:
    lock = json.loads(path.read_text())
    if (lock.get("format") != "dope-arf-native-validation-round"
            or lock.get("round_source_sha256") != sha256(Path(__file__))
            or lock.get("adapter_source_sha256") != sha256(Path(arf_native.__file__))
            or lock.get("source_archive_sha256") != sha256(SOURCE)
            or lock.get("source_files") != source_files()
            or lock.get("dataset_lock_sha256") != sha256(DATASET_LOCK)
            or lock.get("method_lock_sha256") != sha256(METHOD_LOCK)
            or lock.get("pilot_start_sha256") != sha256(ROOT / "start-v2.json")
            or lock.get("runtime") != runtime()
            or lock.get("config_grid") != list(CONFIGS)
            or lock.get("cpu_slot") != SLOT
            or lock.get("official_tests_opened") is not False
            or len(lock.get("jobs", [])) != 12):
        raise ValueError("ARF native round lock changed")
    return lock


def worker(lock_path: Path, job_digest: str, attempt: Path) -> None:
    import numpy as np

    lock = check_lock(lock_path)
    job = next((item for item in lock["jobs"] if digest(item) == job_digest), None)
    if job is None:
        raise ValueError("ARF job absent from frozen matrix")
    if set(os.sched_getaffinity(0)) != set(SLOT):
        raise ValueError("ARF worker escaped reserved CPU slot")
    worker_dir = WORKERS / job["dataset"]
    if ((worker_dir / "test.csv").exists() or any(
            sha256(worker_dir / f"{part}.csv") != job[f"{part}_sha256"]
            for part in ("train", "validation"))
            or sha256(worker_dir / "projection.json") != job["projection_sha256"]):
        raise ValueError("ARF worker input changed")
    sys.path.insert(0, str(SOURCE_DIR))
    from arfpy.arf import arf

    train = np.loadtxt(worker_dir / "train.csv", delimiter=",", ndmin=2)
    validation = np.loadtxt(worker_dir / "validation.csv", delimiter=",", ndmin=2)
    fit_frame, valid_frame, categorical = arf_native.prepare_frames(train, validation)
    np.random.seed(job["fit_seed"])
    config = job["config"]
    started = time.monotonic()
    fitted = arf(fit_frame, num_trees=config["num_trees"], delta=0,
                 max_iters=config["max_iters"], early_stop=True, verbose=False,
                 min_node_size=config["min_node_size"],
                 random_state=job["fit_seed"], n_jobs=16)
    fitted.forde(dist="truncnorm", oob=False, alpha=config["alpha"])
    fit_seconds = time.monotonic() - started
    value = arf_native.heldout_mean_log_density(fitted, valid_frame)
    metric = {"format": "dope-arf-native-validation-kpi", "version": 1,
              "dataset": job["dataset"], "trial": job["trial"],
              "objective": lock["native_objective"], "value": value,
              "validation_sha256": job["validation_sha256"],
              "categorical_columns": categorical,
              "official_tests_opened": False}
    write_once(attempt / "native-kpi.json", metric)
    artifact = attempt / "artifact"
    files = arf_native.save_sampler(fitted, artifact)
    shutil.copyfile(worker_dir / "projection.json", artifact / "projection.json")
    files.append("projection.json")
    inventory, charged = artifact_inventory(artifact, files)
    loaded = arf_native.load_sampler(artifact)
    hashes = []
    for _ in range(2):
        np.random.seed(job["sample_seed"])
        sample = loaded.forge(128)
        values = sample.to_numpy(dtype=float)
        if values.shape != (128, train.shape[1]) or not np.isfinite(values).all():
            raise ValueError("ARF author sample is malformed")
        values = np.clip(values, 0.0, 1.0)
        if job["task"] == "binary":
            values[:, -1] = (values[:, -1] >= 0.5).astype(float)
        text = io.StringIO()
        np.savetxt(text, values, delimiter=",", fmt="%.17g")
        hashes.append(hashlib.sha256(text.getvalue().encode()).hexdigest())
    if hashes[0] != hashes[1]:
        raise ValueError("ARF artifact-only sampling is nondeterministic")
    result = {"format": "dope-arf-native-worker-result", "version": 1,
              "job": job, "round_sha256": sha256(lock_path), "fit_seconds": fit_seconds,
              "native_kpi": value, "native_kpi_sha256": sha256(attempt / "native-kpi.json"),
              "artifact_inventory": inventory, "artifact_bytes": charged,
              "sample_sha256": hashes[0], "repeat_sha256": hashes[1],
              "source_rows_required_for_sampling": False,
              "validation_only": True, "official_tests_opened": False,
              "mfs_v2": None, "ptf_v1": None, "production_certified": False}
    write_once(attempt / "worker-result.json", result)
    print(json.dumps({"native_kpi": value, "artifact_bytes": charged,
                      "sample_sha256": hashes[0]}), flush=True)


def run_job(lock_path: Path, lock: dict, job: dict) -> dict:
    worker_dir = WORKERS / job["dataset"]
    if ((worker_dir / "test.csv").exists() or any(
            sha256(worker_dir / f"{part}.csv") != job[f"{part}_sha256"]
            for part in ("train", "validation"))
            or sha256(worker_dir / "projection.json") != job["projection_sha256"]):
        raise ValueError("ARF worker input changed before admission")
    root = OUT / "jobs" / digest(job)
    prior = root / "attempt-0001" / "receipt.json"
    if prior.exists():
        value = json.loads(prior.read_text())
        if value["job"] != job or value["round_sha256"] != sha256(lock_path):
            raise ValueError("ARF attempt identity changed")
        for name, expected in value["evidence_files"].items():
            if sha256(prior.parent / name) != expected:
                raise ValueError("ARF attempt evidence changed")
        return value
    from datetime import datetime, timezone
    if datetime.now(timezone.utc) >= datetime.fromisoformat(lock["pilot_deadline_utc"]):
        return {"status": "deadline_blocked", "job": job}
    snapshot = local()
    if (not set(SLOT).issubset(snapshot["allowed_cpus"])
            or snapshot["memory"]["MemAvailable"] < 24 * 1024**3
            or snapshot["scratch_disk"]["free_bytes"] < 5_000_000_000
            or snapshot["load_average"][0] + 16 > 1.25 * len(snapshot["allowed_cpus"])
            or used_bytes(SCRATCH) + lock["scratch_reservation_bytes"] > LIMIT):
        return {"status": "admission_blocked", "job": job}
    attempt = root / "attempt-0001"
    attempt.mkdir(parents=True, exist_ok=False)
    write_once(attempt / "host.json", snapshot)
    stdout = attempt / "worker.stdout"
    stderr = attempt / "worker.stderr"
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(Path(__file__).parents[2]) + ":" + str(SOURCE_DIR),
                "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
                "AWS_SHARED_CREDENTIALS_FILE": "/dev/null", "AWS_CONFIG_FILE": "/dev/null",
                "AWS_EC2_METADATA_DISABLED": "true"})
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        env.pop(name, None)
    command = ["taskset", "-c", "48-63", sys.executable, "-m",
               "research.benchmark.arf_round", "worker", str(lock_path),
               digest(job), str(attempt)]
    started = time.monotonic()
    peak_rss_kib = 0
    reason = None
    with stdout.open("x") as out, stderr.open("x") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                 start_new_session=True)
        while child.poll() is None:
            time.sleep(0.5)
            try:
                for line in Path(f"/proc/{child.pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        peak_rss_kib = max(peak_rss_kib, int(line.split()[1]))
                        break
            except FileNotFoundError:
                pass
            if time.monotonic() - started >= TIMEOUT:
                reason = "timeout"
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
    result_path = attempt / "worker-result.json"
    status = reason or ("ok" if child.returncode == 0 and result_path.is_file() else "failed")
    receipt = {"format": "dope-arf-native-attempt", "version": 1,
               "job": job, "round_sha256": sha256(lock_path),
               "status": status, "exit_code": child.returncode,
               "wall_seconds": time.monotonic() - started,
               "peak_rss_kib": peak_rss_kib,
               "host": snapshot["host"], "cpu_affinity": SLOT,
               "source_archive_sha256": lock["source_archive_sha256"],
               "validation_only": True, "official_tests_opened": False,
               "mfs_v2": None, "ptf_v1": None, "production_certified": False}
    if status == "ok":
        try:
            result = json.loads(result_path.read_text())
            inventory, charged = artifact_inventory(attempt / "artifact",
                                                    ["model.json", "bounds.csv", "continuous.csv",
                                                     "categories.csv", "projection.json"])
            if (result["job"] != job or result["round_sha256"] != receipt["round_sha256"]
                    or result["artifact_inventory"] != inventory
                    or result["artifact_bytes"] != charged
                    or result["sample_sha256"] != result["repeat_sha256"]
                    or result["native_kpi_sha256"] != sha256(attempt / "native-kpi.json")):
                raise ValueError("ARF worker result changed before receipt")
            receipt.update({"native_kpi": result["native_kpi"],
                            "native_kpi_sha256": result["native_kpi_sha256"],
                            "artifact_inventory": inventory, "artifact_bytes": charged,
                            "sample_sha256": result["sample_sha256"],
                            "within_l3_bytes": charged <= CAP})
        except (OSError, ValueError, KeyError) as error:
            receipt["status"] = "invalid_evidence"
            receipt["error_type"] = type(error).__name__
    receipt["evidence_files"] = {
        path.relative_to(attempt).as_posix(): sha256(path)
        for path in sorted(attempt.rglob("*")) if path.is_file()}
    write_once(prior, receipt)
    return receipt


def dispatch(lock_path: Path) -> None:
    lock = check_lock(lock_path)
    guard = OUT / "cpu-48-63.reservation"
    with guard.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in lock["jobs"]:
            result = run_job(lock_path, lock, job)
            print(json.dumps({"dataset": job["dataset"], "trial": job["trial"],
                              "status": result["status"]}), flush=True)
            if result["status"] in ("deadline_blocked", "admission_blocked"):
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "dispatch", "worker"))
    parser.add_argument("lock", nargs="?", type=Path)
    parser.add_argument("job_digest", nargs="?")
    parser.add_argument("attempt", nargs="?", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        path = freeze()
        print(json.dumps({"round_sha256": sha256(path), "jobs": 12}))
    elif args.action == "dispatch":
        dispatch(args.lock or OUT / "round.lock.json")
    else:
        if args.lock is None or args.job_digest is None or args.attempt is None:
            parser.error("worker needs lock, job digest, and attempt path")
        worker(args.lock, args.job_digest, args.attempt)


if __name__ == "__main__":
    main()
