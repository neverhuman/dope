"""Idempotent, single-host dataset–method–fit-seed worker."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import adapters
from .fetch_jope import LIMIT, used_bytes
from .manifest import digest
from .score import CONTRACT, artifact_inventory, sha256


class AdapterFailure(Exception):
    def __init__(self, source_error_type: str):
        self.source_error_type = source_error_type
        super().__init__(source_error_type)


def _run_worker(command: list[str], request: dict, seconds: int) -> tuple[str, int]:
    if seconds <= 0:
        raise TimeoutError("adapter deadline")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    try:
        stdout, _ = process.communicate(json.dumps(request), timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise TimeoutError("adapter deadline") from None
    return stdout, process.returncode


def call_adapter(request: dict, seconds: int) -> dict:
    stdout, returncode = _run_worker(
        [sys.executable, "-m", "research.benchmark.adapter_worker"], request, seconds)
    try:
        result = json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise AdapterFailure("MalformedWorkerReceipt") from None
    if returncode != 0 or result.get("status") != "ok":
        raise AdapterFailure(result.get("error_type", "WorkerExit"))
    return result


def _write_once(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def _attempt_path(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.stem.split("-")[-1]) for path in root.glob("attempt-*.json")]
    return root / f"attempt-{max(numbers, default=0) + 1:04d}.json"


def reserve_scratch(root: Path, rows: int, columns: int) -> None:
    # Two simultaneous samples are held for determinism verification. The
    # numeric CSV upper allowance covers a full 17-digit float plus separators.
    allowance = 1_000_000_000 + 2 * 25 * rows * columns
    if used_bytes(root) + allowance > LIMIT:
        raise ValueError("benchmark scratch ceiling would be exceeded")


def check_numpy_runtime(entry: dict, final: bool = False) -> None:
    """Verify compact-method dependencies before any NumPy initializer runs."""
    from importlib.metadata import distribution, version
    if not final and entry["dependency_or_container_digest"] == f"numpy=={version('numpy')}":
        return  # Historical validation receipts retain their version identity.
    path = Path(__file__).with_name("numpy-runtime.lock.json")
    if entry["dependency_or_container_digest"] != sha256(path):
        raise ValueError("adapter dependency digest changed")
    lock = json.loads(path.read_text())
    if (not isinstance(lock, dict) or lock.get("format") != "dope-numpy-runtime-lock"
            or lock.get("versions") != {"numpy": version("numpy")}
            or entry.get("dependency_versions") != lock["versions"]):
        raise ValueError("adapter dependency version changed")
    installed = distribution("numpy")
    files = {}
    for name in ("numpy", "numpy.libs"):
        root = Path(installed.locate_file(name))
        for source in root.rglob("*"):
            if source.is_file() and (source.suffix in (".py", ".so") or ".so." in source.name):
                if source.is_symlink():
                    raise ValueError("adapter dependency source is symlinked")
                files[f"{name}/{source.relative_to(root).as_posix()}"] = sha256(source)
    expected = lock.get("files")
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in expected)
            or len({item["path"] for item in expected}) != len(expected)
            or not files or files != {item["path"]: item["sha256"] for item in expected}):
        raise ValueError("adapter dependency source changed")


def native_evidence_path(value: str, scratch_root: Path) -> Path:
    """Reject table/evaluator paths, including symlinks, before reading bytes."""
    path = Path(value)
    resolved = path.resolve()
    if (not resolved.is_relative_to(scratch_root.resolve())
            or "evaluator" in path.parts or "evaluator" in resolved.parts
            or path.suffix != ".json" or resolved.suffix != ".json"):
        raise ValueError("native evidence outside receipt-only benchmark scratch")
    return path


def resolve_configuration(job: dict, entry: dict, dataset: str, rows: int,
                          scratch_root: Path,
                          validation_sha256: str | None = None) -> tuple[dict, dict]:
    """Bind a final configuration to its validation selection and DP budget."""
    choice = job.get("configuration")
    if choice is None:
        if job.get("final"):
            raise ValueError("final job has no frozen configuration")
        return entry["default_config"], {"kind": "default"}
    if not isinstance(choice, dict) or choice.get("kind") not in ("default", "tuned"):
        raise ValueError("invalid configuration kind")
    kind = choice["kind"]
    if kind == "default":
        if set(choice) != {"kind"}:
            raise ValueError("default configuration has unexpected fields")
        config = dict(entry["default_config"])
        identity = {"kind": kind}
    else:
        if validation_sha256 is None:
            raise ValueError("tuned configuration requires worker validation digest")
        if set(choice) != {"kind", "values", "selection_path", "selection_sha256"}:
            raise ValueError("tuned configuration is missing selection evidence")
        config = choice["values"]
        default = entry["default_config"]
        search = entry["tuning_search_space"]
        if (not isinstance(config, dict) or set(config) != set(default)
                or not isinstance(search, dict)
                or any(value != default[key] and value not in search.get(key, [])
                       for key, value in config.items())):
            raise ValueError("tuned configuration outside locked search space")
        selection_path = native_evidence_path(choice["selection_path"], scratch_root)
        if sha256(selection_path) != choice["selection_sha256"]:
            raise ValueError("selection evidence digest mismatch")
        selection = json.loads(selection_path.read_text())
        if not isinstance(selection, dict):
            raise ValueError("invalid validation selection evidence")
        trials = selection.get("trials")
        objective = entry.get("native_objective")
        if (selection.get("format") != "dope-benchmark-validation-selection"
                or selection.get("partition") != "validation"
                or selection.get("dataset") != dataset
                or selection.get("method") != job["method"]
                or not isinstance(objective, dict)
                or objective.get("status") != "locked"
                or objective.get("direction") not in ("maximize", "minimize")
                or selection.get("objective") != objective
                or selection.get("selected_config") != config
                or selection.get("validation_sha256") != validation_sha256
                or selection.get("test_opened") is not False
                or not isinstance(trials, list) or not 1 <= len(trials) <= 8):
            raise ValueError("invalid validation selection evidence")
        successful = []
        elapsed = 0.0
        for index, trial in enumerate(trials):
            if not isinstance(trial, dict) or trial.get("status") not in ("ok", "timeout", "failed"):
                raise ValueError("invalid validation selection evidence")
            trial_config = trial.get("config")
            seconds = trial.get("wall_seconds")
            if (not isinstance(trial_config, dict) or set(trial_config) != set(default)
                    or any(value != default[key] and value not in search.get(key, [])
                           for key, value in trial_config.items())
                    or not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
                    or not 0 <= seconds <= 43200):
                raise ValueError("invalid validation selection evidence")
            elapsed += seconds
            attempt_path = native_evidence_path(trial.get("attempt_receipt_path", ""), scratch_root)
            if (attempt_path.name != "attempt.json"
                    or sha256(attempt_path) != trial.get("attempt_receipt_sha256")):
                raise ValueError("invalid native tuning attempt receipt")
            attempt = json.loads(attempt_path.read_text())
            if (not isinstance(attempt, dict) or not isinstance(attempt.get("identity"), dict)
                    or attempt != {key: value for key, value in trial.items()
                            if key not in ("attempt_receipt_path", "attempt_receipt_sha256")}
                    or attempt.get("identity", {}).get("dataset") != dataset
                    or attempt["identity"].get("method") != job["method"]
                    or attempt["identity"].get("method_source_sha256") != entry["source_sha256"]
                    or attempt["identity"].get("round_sha256") != selection.get("round_sha256")
                    or attempt["identity"].get("validation_sha256") != validation_sha256
                    or attempt["identity"].get("config") != trial_config):
                raise ValueError("native tuning attempt lineage mismatch")
            if trial["status"] != "ok":
                continue
            metric_path = native_evidence_path(trial.get("metric_receipt_path", ""), scratch_root)
            if (metric_path != attempt_path.parent / "native-metric.json"
                    or sha256(metric_path) != trial.get("metric_receipt_sha256")):
                raise ValueError("invalid native KPI receipt")
            metric = json.loads(metric_path.read_text())
            value = trial.get("native_kpi")
            artifact_bytes = trial.get("artifact_bytes")
            inventory = trial.get("artifact_inventory")
            if (not isinstance(inventory, list)
                    or sorted(item.get("path") for item in inventory if isinstance(item, dict))
                    != ["model.json", "projection.json"]):
                raise ValueError("invalid native artifact inventory")
            for name in ("model.json", "projection.json"):
                native_evidence_path(attempt_path.parent / "artifact" / name, scratch_root)
            actual_inventory, charged = artifact_inventory(
                attempt_path.parent / "artifact", ["model.json", "projection.json"])
            if (not isinstance(metric, dict) or metric.get("method") != job["method"]
                    or metric.get("partition") != "validation"
                    or metric.get("validation_sha256") != validation_sha256
                    or metric.get("objective") != objective.get("name")
                    or metric.get("implementation_sha256") != objective.get("implementation_sha256")
                    or metric.get("value") != value
                    or metric.get("artifact_sha256") != trial.get("artifact_sha256")
                    or trial.get("artifact_sha256") != sha256(attempt_path.parent / "artifact/model.json")
                    or actual_inventory != inventory or charged != artifact_bytes
                    or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not isinstance(artifact_bytes, int) or artifact_bytes < 0):
                raise ValueError("invalid native KPI receipt")
            successful.append((index, value, artifact_bytes, trial_config))
        if not successful or elapsed > 43200:
            raise ValueError("invalid validation selection evidence")
        sign = -1 if objective["direction"] == "maximize" else 1
        winner = min(successful, key=lambda row: (sign * row[1], row[2], digest(row[3]), row[0]))
        if selection.get("selected_trial_index") != winner[0] or config != winner[3]:
            raise ValueError("selected configuration is not the native KPI winner")
        identity = {"kind": kind, "selection_sha256": choice["selection_sha256"]}
        config = dict(config)
    if entry.get("group") == "dp" and job.get("final"):
        epsilon = job.get("dp_epsilon")
        if (epsilon not in (1, 4, 10)
                or epsilon not in entry["tuning_search_space"].get("epsilon", [])):
            raise ValueError("DP epsilon outside frozen budget")
        if kind == "tuned" and config.get("epsilon") != epsilon:
            raise ValueError("tuned selection differs from DP budget")
        config["epsilon"] = epsilon
        identity["dp_budget"] = {"epsilon": epsilon, "delta": min(1e-5, 1 / rows**2)}
    elif "dp_epsilon" in job:
        raise ValueError("DP epsilon on non-final or non-DP job")
    return config, identity


def run(job: dict, methods: dict, output_root: Path) -> list[dict]:
    method = job["method"]
    entry = methods["methods"][method]
    if (entry["status"] not in ("locked", "pilot_locked")
            or (entry["status"] == "pilot_locked" and job.get("pilot_only") is not True)
            or entry["adapter"] != method):
        raise ValueError("method is not source/config locked")
    adapter_path = (Path(__file__).with_name("sdv_adapter.py")
                    if method in ("CTGAN", "TVAE") else Path(adapters.__file__))
    if entry["adapter_sha256"] != sha256(adapter_path):
        raise ValueError("adapter source digest changed")
    if method in ("CTGAN", "TVAE"):
        from importlib.metadata import version
        runtime_lock = Path(__file__).with_name("sdv-runtime.lock.json")
        versions = json.loads(runtime_lock.read_text())
        if ({name: version(name) for name in versions} != versions
                or entry["dependency_or_container_digest"] != sha256(runtime_lock)
                or entry["worker_sha256"] != sha256(Path(__file__).with_name("adapter_worker.py"))
                or entry["source_sha256"] != sha256(Path(entry["source_archive"]))):
            raise ValueError("SDV source, worker, or dependency lock changed")
    if method in ("independent_marginals", "Chow-Liu"):
        check_numpy_runtime(entry, job.get("final") is True)
    if method == "GaussianCopula":
        import copulas
        import numpy as np
        import pandas as pd
        import scipy
        if entry["dependency_version"] != f"copulas=={copulas.__version__}":
            raise ValueError("copula dependency version changed")
        if entry["dependency_versions"] != {"numpy": np.__version__, "pandas": pd.__version__,
                                            "scipy": scipy.__version__}:
            raise ValueError("copula transitive dependency version changed")
    if method == "AIM":
        from importlib.metadata import version
        from . import aim_adapter
        runtime_lock = Path(__file__).with_name("aim-runtime.lock.json")
        versions = json.loads(runtime_lock.read_text())
        if ({name: version(name) for name in versions} != versions
                or entry["dependency_or_container_digest"] != sha256(runtime_lock)
                or entry["aim_adapter_sha256"] != sha256(Path(aim_adapter.__file__))
                or entry["source_sha256"] != sha256(Path(entry["source_archive"]))):
            raise ValueError("AIM source, adapter, or dependency lock changed")
    if job["fit_seed"] not in CONTRACT["fit_seeds"] or job["track"] not in ("common_numeric", "author_faithful"):
        raise ValueError("job outside frozen contract")
    worker = Path(job["worker_dir"])
    manifest = json.loads((worker / "worker-manifest.json").read_text())
    projection = json.loads((worker / "projection.json").read_text())
    if manifest["projection_sha256"] != sha256(worker / "projection.json"):
        raise ValueError("projection digest mismatch")
    for name in ("train", "validation"):
        if manifest["projected_hashes"][name] != sha256(worker / f"{name}.csv"):
            raise ValueError("worker table digest mismatch")
    scratch_root = Path(job.get("scratch_root", output_root.parent)).resolve()
    if job["track"] == "author_faithful":
        raise ValueError("author-faithful data preparation and adapter unavailable")
    n = manifest.get("train_rows")
    if n is None:
        with (worker / "train.csv").open() as stream:
            n = sum(1 for _ in stream)
    sample_seeds = job.get("sample_seeds", CONTRACT["sample_seeds"])
    multipliers = job.get("size_multipliers", CONTRACT["size_multipliers"])
    if (not sample_seeds or not multipliers or len(sample_seeds) != len(set(sample_seeds))
            or len(multipliers) != len(set(multipliers))
            or not set(sample_seeds).issubset(CONTRACT["sample_seeds"])
            or not set(multipliers).issubset(CONTRACT["size_multipliers"])):
        raise ValueError("sampling cells outside frozen contract")
    if job.get("final") and (sample_seeds != CONTRACT["sample_seeds"]
                             or multipliers != CONTRACT["size_multipliers"]):
        raise ValueError("final job requires the complete sample matrix")
    config, config_identity = resolve_configuration(job, entry, manifest["dataset_id"],
                                                    n, scratch_root,
                                                    manifest["projected_hashes"]["validation"])
    fit_identity = {
        "dataset": manifest["dataset_id"], "split": manifest["split_hashes"],
        "projection": manifest["projection_sha256"], "track": job["track"],
        "method": method, "method_source": entry["source_sha256"],
        "adapter_sha256": entry["adapter_sha256"],
        "adapter_worker_sha256": sha256(Path(__file__).with_name("adapter_worker.py")),
        "binary_sha256": entry.get("binary_sha256"),
        "config": config, "configuration": config_identity, "fit_seed": job["fit_seed"],
        "sample_seeds": sample_seeds, "size_multipliers": multipliers,
        "contract": sha256(Path(__file__).with_name("contract.json")),
        "runner_source": sha256(Path(__file__)),
    }
    fit_key = digest(fit_identity)
    directory = output_root / fit_key
    directory.mkdir(parents=True, exist_ok=True)
    artifact_dir = directory / "artifact"
    fit_receipt_path = directory / "fit-receipt.json"
    binary = Path(job["dope_binary"]) if method == "dope" else None
    if method == "dope" and sha256(binary) != entry["binary_sha256"]:
        raise ValueError("DOPE binary digest changed")
    if not fit_receipt_path.exists():
        attempts = sorted((directory / "fit-attempts").glob("attempt-*.json"))
        if attempts:
            previous = json.loads(attempts[-1].read_text())
            if previous["status"] == "ok":
                inventory, _ = artifact_inventory(artifact_dir, previous["artifact_files"])
                if previous["fit_key"] != fit_key or inventory != previous["artifact_inventory"]:
                    raise ValueError("successful fit attempt changed before resume")
                if previous.get("fit_evidence_files"):
                    evidence, _ = artifact_inventory(directory / "fit_evidence",
                                                     previous["fit_evidence_files"])
                    if evidence != previous["fit_evidence_inventory"]:
                        raise ValueError("successful fit evidence changed before resume")
                _write_once(fit_receipt_path, previous)
    if fit_receipt_path.exists():
        fit_receipt = json.loads(fit_receipt_path.read_text())
        if fit_receipt["fit_key"] != fit_key or fit_receipt["status"] != "ok":
            raise ValueError("existing fit receipt failed or mismatched")
        inventory, _ = artifact_inventory(artifact_dir, fit_receipt["artifact_files"])
        if inventory != fit_receipt["artifact_inventory"]:
            raise ValueError("artifact changed since fit receipt")
        if fit_receipt.get("fit_evidence_files"):
            evidence_inventory, _ = artifact_inventory(directory / "fit_evidence", fit_receipt["fit_evidence_files"])
            if evidence_inventory != fit_receipt["fit_evidence_inventory"]:
                raise ValueError("fit evidence changed since receipt")
    else:
        start = time.perf_counter()
        attempt_path = _attempt_path(directory / "fit-attempts")
        try:
            if os.uname().nodename == "xbabe2":
                reserve_scratch(scratch_root, 0, 0)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(worker / "projection.json", artifact_dir / "projection.json")
            files = call_adapter({"action": "fit", "method": method,
                                  "train": str(worker / "train.csv"), "metadata": projection,
                                  "config": config, "seed": job["fit_seed"],
                                  "artifact_dir": str(artifact_dir),
                                  "binary": str(binary) if binary else None},
                                 config.get("fit_timeout_seconds", 600 if method in ("CTGAN", "TVAE") else 1800))["files"]
            files = ["projection.json", *files]
            inventory, artifact_bytes = artifact_inventory(artifact_dir, files)
            fit_evidence_dir = directory / "fit_evidence"
            fit_evidence_files = sorted(path.name for path in fit_evidence_dir.iterdir()) if fit_evidence_dir.exists() else []
            fit_evidence_inventory = artifact_inventory(fit_evidence_dir, fit_evidence_files)[0] if fit_evidence_files else []
            fit_receipt = {"format": "dope-benchmark-fit-receipt", "status": "ok",
                           "fit_key": fit_key, "fit_identity": fit_identity,
                           "artifact_files": files, "artifact_inventory": inventory,
                           "fit_evidence_files": fit_evidence_files,
                           "fit_evidence_inventory": fit_evidence_inventory,
                           "artifact_bytes": artifact_bytes,
                           "fit_seconds": time.perf_counter() - start,
                           "host": os.uname().nodename,
                           "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                           "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
        except (AdapterFailure, TimeoutError, OSError, ValueError, KeyError,
                TypeError, subprocess.SubprocessError) as error:
            fit_receipt = {"format": "dope-benchmark-fit-receipt",
                           "status": "timeout" if isinstance(error, TimeoutError) else "failed",
                           "fit_key": fit_key, "fit_identity": fit_identity,
                           "fit_seconds": time.perf_counter() - start,
                           "error_type": getattr(error, "source_error_type", type(error).__name__),
                           "host": os.uname().nodename}
            for name in ("artifact", "fit_evidence"):
                staged = directory / name
                if staged.exists():
                    staged.rename(attempt_path.parent / f"{attempt_path.stem}-{name}")
        _write_once(attempt_path, fit_receipt)
        if fit_receipt["status"] != "ok":
            return [fit_receipt]
        _write_once(fit_receipt_path, fit_receipt)
    receipts = []
    for sample_seed in sample_seeds:
        for multiplier in multipliers:
            row_count = n * multiplier
            key = digest({**fit_identity, "sample_seed": sample_seed, "row_count": row_count})
            receipt_path = directory / f"{key}.receipt.json"
            output = directory / f"{key}.csv"
            if not receipt_path.exists():
                attempts = sorted((directory / "sample-attempts" / key).glob("attempt-*.json"))
                if attempts:
                    previous = json.loads(attempts[-1].read_text())
                    if previous["status"] == "ok":
                        if (previous["run_key"] != key
                                or sha256(output) != previous["sample_sha256"]):
                            raise ValueError("successful sample attempt changed before resume")
                        _write_once(receipt_path, previous)
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["status"] != "ok":
                    raise ValueError("existing sample receipt failed")
                if sha256(output) != receipt["sample_sha256"]:
                    raise ValueError("sample changed since receipt")
                receipts.append(receipt)
                continue
            start = time.perf_counter()
            attempt_path = _attempt_path(directory / "sample-attempts" / key)
            repeat = directory / f"{key}.repeat.csv"
            try:
                if os.uname().nodename == "xbabe2":
                    reserve_scratch(scratch_root, row_count, projection["output_features"] + 1)
                sample_request = {"action": "sample", "method": method,
                                  "artifact_dir": str(artifact_dir), "row_count": row_count,
                                  "seed": sample_seed, "binary": str(binary) if binary else None}
                sample_limit = config.get("sample_timeout_seconds", 600)
                call_adapter({**sample_request, "output": str(output)}, sample_limit)
                remaining = sample_limit - (time.perf_counter() - start)
                call_adapter({**sample_request, "output": str(repeat)}, int(remaining))
                first_hash, second_hash = sha256(output), sha256(repeat)
                repeat.unlink()
                receipt = {"format": "dope-benchmark-sample-receipt", "status": "ok",
                           "run_key": key, "fit_key": fit_key, "sample_seed": sample_seed,
                           "row_count": row_count, "sample_sha256": first_hash,
                           "artifact_sampling_verified": first_hash == second_hash,
                           "sample_seconds": time.perf_counter() - start,
                           "host": os.uname().nodename}
            except (AdapterFailure, TimeoutError, OSError, ValueError, KeyError,
                    TypeError, subprocess.SubprocessError) as error:
                receipt = {"format": "dope-benchmark-sample-receipt",
                           "status": "timeout" if isinstance(error, TimeoutError) else "failed",
                           "run_key": key, "fit_key": fit_key, "sample_seed": sample_seed,
                           "row_count": row_count, "sample_seconds": time.perf_counter() - start,
                           "error_type": getattr(error, "source_error_type", type(error).__name__),
                           "host": os.uname().nodename}
                for staged in (output, repeat):
                    if staged.exists():
                        staged.rename(attempt_path.parent / f"{attempt_path.stem}-{staged.name}")
            _write_once(attempt_path, receipt)
            if receipt["status"] == "ok":
                _write_once(receipt_path, receipt)
            receipts.append(receipt)
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("methods", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.output_root.resolve().is_relative_to(Path(__file__).resolve().parents[2]) or args.output_root.resolve().is_relative_to(Path("/tmp")):
        parser.error("bulk results must live outside the worktree and /tmp")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:16]))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "16"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print(json.dumps(run(json.loads(args.job.read_text()), json.loads(args.methods.read_text()),
                         args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
