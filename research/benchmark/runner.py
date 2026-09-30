"""Idempotent, single-host dataset–method–fit-seed worker."""

from __future__ import annotations

import argparse
import json
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


def resolve_configuration(job: dict, entry: dict, dataset: str, rows: int,
                          scratch_root: Path) -> tuple[dict, dict]:
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
        selection_path = Path(choice["selection_path"])
        if not selection_path.resolve().is_relative_to(scratch_root.resolve()):
            raise ValueError("selection evidence outside benchmark scratch")
        if sha256(selection_path) != choice["selection_sha256"]:
            raise ValueError("selection evidence digest mismatch")
        selection = json.loads(selection_path.read_text())
        trials = selection.get("trials")
        if (selection.get("format") != "dope-benchmark-validation-selection"
                or selection.get("partition") != "validation"
                or selection.get("dataset") != dataset
                or selection.get("method") != job["method"]
                or selection.get("selected_config") != config
                or not isinstance(trials, list) or len(trials) != 8
                or any(not isinstance(trial, dict)
                       or trial.get("status") not in ("ok", "timeout", "failed")
                       or not isinstance(trial.get("wall_seconds"), (int, float))
                       or not 0 <= trial["wall_seconds"] <= 43200 for trial in trials)
                or sum(trial["wall_seconds"] for trial in trials) > 43200):
            raise ValueError("invalid validation selection evidence")
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
    if entry["adapter_sha256"] != sha256(Path(adapters.__file__)):
        raise ValueError("adapter source digest changed")
    if method in ("independent_marginals", "Chow-Liu"):
        import numpy as np
        if entry["dependency_or_container_digest"] != f"numpy=={np.__version__}":
            raise ValueError("adapter dependency version changed")
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
                                                    n, scratch_root)
    fit_identity = {
        "dataset": manifest["dataset_id"], "split": manifest["split_hashes"],
        "projection": manifest["projection_sha256"], "track": job["track"],
        "method": method, "method_source": entry["source_sha256"],
        "adapter_sha256": entry["adapter_sha256"],
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
                                 config.get("fit_timeout_seconds", 1800))["files"]
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
        except Exception as error:
            fit_receipt = {"format": "dope-benchmark-fit-receipt",
                           "status": "timeout" if isinstance(error, TimeoutError) else "failed",
                           "fit_key": fit_key, "fit_identity": fit_identity,
                           "fit_seconds": time.perf_counter() - start,
                           "error_type": getattr(error, "source_error_type", type(error).__name__),
                           "host": os.uname().nodename}
            for name in ("artifact", "fit_evidence"):
                partial = directory / name
                if partial.exists():
                    partial.rename(attempt_path.parent / f"{attempt_path.stem}-{name}")
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
            except Exception as error:
                receipt = {"format": "dope-benchmark-sample-receipt",
                           "status": "timeout" if isinstance(error, TimeoutError) else "failed",
                           "run_key": key, "fit_key": fit_key, "sample_seed": sample_seed,
                           "row_count": row_count, "sample_seconds": time.perf_counter() - start,
                           "error_type": getattr(error, "source_error_type", type(error).__name__),
                           "host": os.uname().nodename}
                for partial in (output, repeat):
                    if partial.exists():
                        partial.rename(attempt_path.parent / f"{attempt_path.stem}-{partial.name}")
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
