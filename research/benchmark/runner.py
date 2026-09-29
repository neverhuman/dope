"""Idempotent, single-host dataset–method–fit-seed worker."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from . import adapters
from .fetch_jope import LIMIT, used_bytes
from .manifest import digest
from .score import CONTRACT, artifact_inventory, sha256


def _write_once(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


@contextmanager
def deadline(seconds: int):
    if seconds <= 0:
        raise ValueError("deadline must be positive")
    def expired(_signum, _frame):
        raise TimeoutError("job deadline")
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def reserve_scratch(root: Path, rows: int, columns: int) -> None:
    # Two simultaneous samples are held for determinism verification. The
    # numeric CSV upper allowance covers a full 17-digit float plus separators.
    allowance = 1_000_000_000 + 2 * 25 * rows * columns
    if used_bytes(root) + allowance > LIMIT:
        raise ValueError("benchmark scratch ceiling would be exceeded")


def run(job: dict, methods: dict, output_root: Path) -> list[dict]:
    method = job["method"]
    entry = methods["methods"][method]
    if entry["status"] != "locked" or entry["adapter"] != method:
        raise ValueError("method is not source/config locked")
    if entry["adapter_sha256"] != sha256(Path(adapters.__file__)):
        raise ValueError("adapter source digest changed")
    if method == "independent_marginals":
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
    if job["track"] == "author_faithful" and method != "dope":
        raise ValueError("author-faithful adapter unavailable")
    if job["track"] == "author_faithful" and method == "dope":
        # DOPE accepts the same numeric projection in both tracks.
        pass
    n = manifest.get("train_rows")
    if n is None:
        with (worker / "train.csv").open() as stream:
            n = sum(1 for _ in stream)
    fit_identity = {
        "dataset": manifest["dataset_id"], "split": manifest["split_hashes"],
        "projection": manifest["projection_sha256"], "track": job["track"],
        "method": method, "method_source": entry["source_sha256"],
        "adapter_sha256": entry["adapter_sha256"],
        "binary_sha256": entry.get("binary_sha256"),
        "config": entry["default_config"], "fit_seed": job["fit_seed"],
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
        try:
            if os.uname().nodename == "xbabe2":
                reserve_scratch(scratch_root, 0, 0)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(worker / "projection.json", artifact_dir / "projection.json")
            with deadline(entry["default_config"].get("fit_timeout_seconds", 1800)):
                files = adapters.fit(method, worker / "train.csv", projection,
                                     entry["default_config"], job["fit_seed"], artifact_dir, binary)
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
                           "error_type": type(error).__name__, "host": os.uname().nodename}
        _write_once(fit_receipt_path, fit_receipt)
        if fit_receipt["status"] != "ok":
            return [fit_receipt]
    receipts = []
    sample_seeds = job.get("sample_seeds", CONTRACT["sample_seeds"])
    multipliers = job.get("size_multipliers", CONTRACT["size_multipliers"])
    if (not sample_seeds or not multipliers or len(sample_seeds) != len(set(sample_seeds))
            or len(multipliers) != len(set(multipliers))
            or not set(sample_seeds).issubset(CONTRACT["sample_seeds"])
            or not set(multipliers).issubset(CONTRACT["size_multipliers"])):
        raise ValueError("sampling cells outside frozen contract")
    for sample_seed in sample_seeds:
        for multiplier in multipliers:
            row_count = n * multiplier
            key = digest({**fit_identity, "sample_seed": sample_seed, "row_count": row_count})
            receipt_path = directory / f"{key}.receipt.json"
            output = directory / f"{key}.csv"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["status"] == "ok" and sha256(output) != receipt["sample_sha256"]:
                    raise ValueError("sample changed since receipt")
                receipts.append(receipt)
                continue
            start = time.perf_counter()
            try:
                if os.uname().nodename == "xbabe2":
                    reserve_scratch(scratch_root, row_count, projection["output_features"] + 1)
                with deadline(entry["default_config"].get("sample_timeout_seconds", 600)):
                    adapters.sample(method, artifact_dir, row_count, sample_seed, output, binary)
                    repeat = directory / f"{key}.repeat.csv"
                    adapters.sample(method, artifact_dir, row_count, sample_seed, repeat, binary)
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
                           "error_type": type(error).__name__, "host": os.uname().nodename}
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(json.dumps(run(json.loads(args.job.read_text()), json.loads(args.methods.read_text()),
                         args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
