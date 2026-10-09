"""Holdout privacy and GBDT C2ST on published sample CSVs.

Reads the committed ledgers and the already written sample files. Does not
open a file named test.csv. A finished cell is skipped, so a stopped job
can resume. An unresolved sample path is recorded as unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.benchmark.review_fixes.source_pins import authenticated_published_bytes

EXPANDED_PATH = HERE.parent / "expanded_validation_metrics.py"
WORKERS = Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker")
SEEDS = (101, 211, 307)
PUBLISHED = (
    ("density-matched-population-validation.json", "DOPE", "features12_steps2048"),
    ("density-matched-population-validation.json", "GaussianCopula", "native_selected"),
    ("density-matched-population-validation.json", "Chow-Liu", "native_selected"),
    ("density-matched-population-validation.json", "independent_marginals", "native_selected"),
    ("sdv-matched-population-validation.json", "CTGAN", "native_selected"),
    ("sdv-matched-population-validation.json", "TVAE", "native_selected"),
    ("arf-matched-population-validation.json", "ARF", "author_default"),
    ("arf-matched-population-validation.json", "ARF", "native_selected"),
    ("s3-matched-forest-confirmation-validation.json", "ForestDiffusion/Forest-Flow", "native_selected"),
)


def _resolve_sample_csv(cell: dict):
    spec = importlib.util.spec_from_file_location("review_fix_sample_paths", HERE / "sample_paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_sample_csv(cell)


def _load_expanded():
    spec = importlib.util.spec_from_file_location("expanded_validation_metrics", EXPANDED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _memory_fraction() -> float:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        info[key] = int(rest.split()[0])
    return info["MemAvailable"] / info["MemTotal"]


def _wait_for_memory() -> None:
    while _memory_fraction() < 0.15:
        time.sleep(20)


def _read_table(path: Path) -> np.ndarray:
    if path.name == "test.csv":
        raise ValueError("privacy scoring refuses the official test partition")
    with path.open() as handle:
        token = handle.readline().split(",")[0].strip()
    skip = 0
    try:
        float(token)
    except ValueError:
        skip = 1
    table = np.loadtxt(path, delimiter=",", skiprows=skip, ndmin=2)
    if table.ndim != 2 or not np.isfinite(table).all() or table.shape[1] < 2:
        raise ValueError("malformed numeric table")
    return table


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _median(summary):
    if isinstance(summary, dict) and _finite(summary.get("median")):
        return float(summary["median"])
    return None


def _measured_value(block):
    if isinstance(block, dict) and block.get("status") == "ok" and _finite(block.get("value")):
        return float(block["value"])
    return None


def _nndr_median(block):
    if not isinstance(block, dict) or block.get("status") != "ok":
        return None
    value = block.get("value")
    if isinstance(value, dict):
        return _median(value)
    return _measured_value(block)


def _expected_sample_sha(cell: dict, csv_path: str | None) -> str | None:
    evidence = cell.get("metric_receipt") if cell.get("method") == "DOPE" else cell.get("sample_evidence")
    if not isinstance(evidence, dict):
        return None
    sample_sha = evidence.get("sample_sha256")
    if isinstance(sample_sha, str):
        return sample_sha
    # SDV binds a CSV directly. A forest receipt digest is not a sample digest.
    if isinstance(csv_path, str) and evidence.get("path") == csv_path:
        return evidence.get("sha256")
    return None


def _valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _authenticated_sample_sha(path: Path, job: dict) -> str:
    expected = job.get("expected_sha256")
    if not _valid_sha(expected):
        raise ValueError("published sample sha pin missing")
    if path.name == "test.csv":
        raise ValueError("official test sample refused")
    if _sha(path) != expected:
        raise ValueError("published sample sha mismatch")
    return expected


def _existing_problem(record: dict, job: dict) -> str | None:
    for field in ("dataset", "method", "configuration", "size", "sample_seed"):
        if record.get(field) != job.get(field):
            return "existing privacy cell identity mismatch"
    if record.get("status") == "ok":
        expected = job.get("expected_sha256")
        if not _valid_sha(expected):
            return "existing privacy cell published pin missing"
        if record.get("expected_sha256") != expected or record.get("synthetic_sha256") != expected:
            return "existing privacy cell requires sample-pin revalidation"
    return None


def _validate_existing(path: Path, job: dict) -> None:
    record = json.loads(path.read_text())
    problem = _existing_problem(record, job)
    if problem is not None:
        raise ValueError(problem)
    if record.get("status") == "ok":
        csv_path = job.get("csv")
        if not isinstance(csv_path, str):
            raise ValueError("published sample sha pin missing")
        _authenticated_sample_sha(Path(csv_path), job)


def _index(results: Path) -> list[dict]:
    captured = authenticated_published_bytes(results, (spec[0] for spec in PUBLISHED))
    cache = {}
    jobs = []
    for filename, method, configuration in PUBLISHED:
        if filename not in cache:
            cache[filename] = json.loads(captured[filename])["cells"]
        for cell in cache[filename]:
            if cell.get("method") != method or cell.get("configuration") != configuration:
                continue
            if cell.get("fit_seed") not in (None, 11):
                continue
            size = cell.get("size_multiplier")
            seed = cell.get("sample_seed")
            dataset = cell.get("dataset")
            if size not in (1, 4) or seed not in SEEDS or not isinstance(dataset, str):
                continue
            csv_path = _resolve_sample_csv(cell)
            jobs.append({
                "dataset": dataset,
                "method": method,
                "configuration": configuration,
                "size": size,
                "sample_seed": seed,
                "csv": csv_path,
                "source": filename,
                "expected_sha256": _expected_sample_sha(cell, csv_path),
            })
    return jobs


def _out_path(out_dir: Path, job: dict) -> Path:
    name = f"size{job['size']}-seed{job['sample_seed']}.json"
    return out_dir / job["method"] / job["configuration"] / job["dataset"] / name


def _score_one(job: dict) -> dict:
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        _validate_existing(out, job)
        return {"status": "skipped", "path": str(out)}
    payload = {
        "dataset": job["dataset"],
        "method": job["method"],
        "configuration": job["configuration"],
        "size": job["size"],
        "sample_seed": job["sample_seed"],
        "csv": job.get("csv"),
        "source": job.get("source"),
        "expected_sha256": job.get("expected_sha256"),
        "official_tests_opened": False,
        "formal_dp": False,
        "hipaa_deidentification": False,
    }
    csv_path = job.get("csv")
    if not isinstance(csv_path, str):
        payload.update(status="unavailable", reason="sample_csv_unresolved")
        _write(out, payload)
        return {"status": "unavailable", "path": str(out)}
    path = Path(csv_path)
    if path.name == "test.csv" or not path.is_file():
        payload.update(status="unavailable", reason="sample_csv_absent")
        _write(out, payload)
        return {"status": "unavailable", "path": str(out)}
    try:
        synthetic_sha256 = _authenticated_sample_sha(path, job)
        _wait_for_memory()
        expanded = _load_expanded()
        root = WORKERS / job["dataset"]
        fit = _read_table(root / "train.csv")
        validation = _read_table(root / "validation.csv")
        synthetic = _read_table(path)
        if fit.shape[1] != validation.shape[1] or fit.shape[1] != synthetic.shape[1]:
            raise ValueError("table width mismatch")
        kinds = ["continuous"] * fit.shape[1]
        privacy = expanded.privacy(fit, validation, synthetic, kinds)
        detection = expanded.c2st(validation, synthetic, "catboost")
        payload.update(
            status="ok",
            reason=None,
            synthetic_sha256=synthetic_sha256,
            rows={"fit": int(fit.shape[0]), "validation": int(validation.shape[0]), "synthetic": int(synthetic.shape[0])},
            dcr_fit_median=_median(privacy.get("dcr_fit")),
            dcr_validation_median=_median(privacy.get("dcr_validation")),
            nndr_fit_median=_nndr_median(privacy.get("nndr_fit")),
            distance_mia_auc=_measured_value(privacy.get("distance_mia")),
            distance_mia_status=(privacy.get("distance_mia") or {}).get("status"),
            c2st_catboost_auc=_measured_value(detection),
            c2st_rows_per_class=detection.get("rows_per_class") if isinstance(detection, dict) else None,
            c2st_status=detection.get("status") if isinstance(detection, dict) else None,
        )
    except (ValueError, RuntimeError, OSError) as error:
        payload.update(status="unavailable", reason=type(error).__name__)
    _write(out, payload)
    return {"status": payload["status"], "path": str(out)}


def _write(path: Path, payload: dict) -> None:
    staging = path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(payload, sort_keys=True) + "\n")
    staging.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--workers-parallel", type=int, default=2)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if os.uname().nodename != args.host:
        raise SystemExit(f"refusing to run on {os.uname().nodename}")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    jobs = _index(args.results)
    if args.index_only:
        args.out.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            out = _out_path(args.out, job)
            if out.exists():
                _validate_existing(out, job)
        text = "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs)
        (args.out / "index.jsonl").write_text(text)
        resolved = sum(isinstance(job.get("csv"), str) for job in jobs)
        print(f"indexed {len(jobs)} resolved {resolved}", flush=True)
        return
    pending = []
    for job in jobs:
        out = _out_path(args.out, job)
        job = dict(job)
        job["out"] = str(out)
        if out.exists():
            _validate_existing(out, job)
        else:
            pending.append(job)
    print(f"jobs {len(jobs)} pending {len(pending)}", flush=True)
    if not pending:
        return
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=args.workers_parallel, mp_context=mp.get_context("spawn")) as pool:
        for result in pool.map(_score_one, pending):
            print(result["status"], result.get("path", ""), flush=True)


if __name__ == "__main__":
    main()
