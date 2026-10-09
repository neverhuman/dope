"""Predictor-only and real-bootstrap controls.

The definitions are fixed in predeclare.json. This script never reads a file
named test.csv. A finished cell is skipped, so a stopped job can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT_PATH = HERE.parent / "pilot_metrics.py"
FIT_SEEDS = (11, 23, 37, 53, 71)
SAMPLE_SEEDS = (101, 211, 307)
SPLIT_SEEDS = (2027, 2999, 4099, 8191)


def _load_pilot():
    spec = importlib.util.spec_from_file_location("pilot_metrics", PILOT_PATH)
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
        raise ValueError("controls refuse the official test partition")
    table = np.loadtxt(path, delimiter=",", ndmin=2)
    if table.ndim != 2 or not np.isfinite(table).all() or table.shape[1] < 2:
        raise ValueError("malformed numeric table")
    return table


def _write_table(path: Path, table: np.ndarray) -> str:
    np.savetxt(path, table, delimiter=",")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _real_bootstrap(train: np.ndarray, sample_seed: int) -> np.ndarray:
    rng = np.random.default_rng(sample_seed)
    pick = rng.integers(0, len(train), size=4 * len(train))
    return train[pick]


def _fit_rows(train: np.ndarray, fit_seed: int) -> np.ndarray:
    if fit_seed == 11:
        return train
    rng = np.random.default_rng(fit_seed)
    pick = rng.integers(0, len(train), size=len(train))
    return train[pick]


def _predictor_rows(train: np.ndarray, task: str, fit_seed: int, sample_seed: int, size: int) -> np.ndarray:
    fitted = _fit_rows(train, fit_seed)
    x_fit, y_fit = fitted[:, :-1], fitted[:, -1]
    rng = np.random.default_rng([fit_seed, sample_seed, size])
    rows = size * len(train)
    x_new = np.column_stack([
        rng.choice(x_fit[:, column], size=rows, replace=True) for column in range(x_fit.shape[1])
    ])
    if task == "binary":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=fit_seed)
        model.fit(x_fit, y_fit)
        probability = np.clip(model.predict_proba(x_new)[:, 1], 1e-6, 1 - 1e-6)
        y_new = (rng.random(rows) < probability).astype(float)
    elif task == "regression":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(x_fit, y_fit)
        residual = y_fit - model.predict(x_fit)
        sigma = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
        y_new = model.predict(x_new) + rng.normal(0.0, sigma, size=rows)
    else:
        raise ValueError("unknown task")
    return np.column_stack([x_new, y_new])


def _resplit(train: np.ndarray, validation: np.ndarray, task: str, split_seed: int):
    pool = np.vstack([train, validation])
    rng = np.random.default_rng(split_seed)
    if task == "binary":
        chosen = []
        held = []
        labels = pool[:, -1]
        for label in np.unique(labels):
            group = np.flatnonzero(labels == label)
            rng.shuffle(group)
            cut = int(0.8 * len(group))
            if cut <= 0 or cut >= len(group):
                return None
            chosen.append(group[:cut])
            held.append(group[cut:])
        return pool[np.concatenate(chosen)], pool[np.concatenate(held)]
    order = rng.permutation(len(pool))
    cut = int(0.8 * len(pool))
    if cut <= 1 or cut >= len(pool):
        return None
    return pool[order[:cut]], pool[order[cut:]]


def _score(pilot, train_path, valid_path, synthetic_path, task: str) -> dict:
    _wait_for_memory()
    report = pilot.measure(train_path, valid_path, synthetic_path, task)
    return {
        "null_loss": report["null_loss"],
        "utility": report["utility"],
        "rows": report["rows"],
        "marginal_ks_mean": report["marginal_ks_mean"],
        "pair_correlation_fidelity": report["pair_correlation_fidelity"],
        "c2st_auc": report["c2st_auc"],
    }


def _emit(out: Path, payload: dict) -> None:
    staging = out.with_suffix(".json.tmp")
    staging.write_text(json.dumps(payload, sort_keys=True) + "\n")
    staging.replace(out)


def _one(job: dict) -> dict:
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return {"status": "skipped", "path": str(out)}
    try:
        return _one_body(job, out)
    except Exception as error:
        payload = {
            "status": "failed",
            "reason": type(error).__name__,
            "dataset": job.get("dataset"),
            "kind": job.get("kind"),
            "fit_seed": job.get("fit_seed"),
            "sample_seed": job.get("sample_seed"),
            "size": job.get("size"),
            "split_seed": job.get("split_seed"),
            "metrics": None,
            "official_tests_opened": False,
        }
        _emit(out, payload)
        return {"status": "failed", "path": str(out), "reason": type(error).__name__}


def _one_body(job: dict, out: Path) -> dict:
    pilot = _load_pilot()
    train = _read_table(Path(job["train"]))
    valid = _read_table(Path(job["validation"]))
    task = job["task"]
    if job["kind"] == "split":
        split = _resplit(train, valid, task, job["split_seed"])
        if split is None:
            payload = {
                "status": "unavailable",
                "reason": "split_too_small",
                "dataset": job["dataset"],
                "kind": job["kind"],
                "fit_seed": job.get("fit_seed"),
                "sample_seed": job["sample_seed"],
                "size": job.get("size"),
                "split_seed": job.get("split_seed"),
                "metrics": None,
                "official_tests_opened": False,
            }
            _emit(out, payload)
            return payload
        train, valid = split
        train_path = out.with_suffix(".train.csv")
        valid_path = out.with_suffix(".validation.csv")
        _write_table(train_path, train)
        _write_table(valid_path, valid)
    else:
        train_path = Path(job["train"])
        valid_path = Path(job["validation"])
    if job["kind"] == "real_bootstrap_4n":
        synthetic = _real_bootstrap(train, job["sample_seed"])
    else:
        synthetic = _predictor_rows(train, task, job["fit_seed"], job["sample_seed"], job["size"])
    synthetic_path = out.with_suffix(".synthetic.csv")
    digest = _write_table(synthetic_path, synthetic)
    try:
        metrics = _score(pilot, train_path, valid_path, synthetic_path, task)
        status = "ok"
        reason = None
    except (ValueError, RuntimeError) as error:
        metrics = None
        status = "failed"
        reason = type(error).__name__
    payload = {
        "status": status,
        "reason": reason,
        "dataset": job["dataset"],
        "kind": job["kind"],
        "fit_seed": job.get("fit_seed"),
        "sample_seed": job["sample_seed"],
        "size": job.get("size"),
        "split_seed": job.get("split_seed"),
        "synthetic_sha256": digest,
        "metrics": metrics,
        "official_tests_opened": False,
    }
    _emit(out, payload)
    return {"status": status, "path": str(out)}


def _jobs(workers: Path, datasets: list[str], out_dir: Path, include_splits: bool) -> list[dict]:
    jobs = []
    for dataset in datasets:
        root = workers / dataset
        projection = json.loads((root / "projection.json").read_text())
        common = {
            "dataset": dataset,
            "train": str(root / "train.csv"),
            "validation": str(root / "validation.csv"),
            "task": projection["task"],
        }
        for sample_seed in SAMPLE_SEEDS:
            jobs.append({
                **common,
                "kind": "real_bootstrap_4n",
                "sample_seed": sample_seed,
                "size": 4,
                "out": str(out_dir / dataset / f"real-bootstrap-sample{sample_seed}.json"),
            })
            for fit_seed in FIT_SEEDS:
                for size in (1, 4):
                    jobs.append({
                        **common,
                        "kind": "predictor_only",
                        "fit_seed": fit_seed,
                        "sample_seed": sample_seed,
                        "size": size,
                        "out": str(out_dir / dataset / f"predictor-fit{fit_seed}-sample{sample_seed}-size{size}.json"),
                    })
        if include_splits:
            for split_seed in SPLIT_SEEDS:
                for sample_seed in SAMPLE_SEEDS:
                    for size in (1, 4):
                        jobs.append({
                            **common,
                            "kind": "split",
                            "fit_seed": 11,
                            "sample_seed": sample_seed,
                            "size": size,
                            "split_seed": split_seed,
                            "out": str(out_dir / dataset / f"split{split_seed}-sample{sample_seed}-size{size}.json"),
                        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--lineage-record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--workers-parallel", type=int, default=4)
    parser.add_argument("--include-split-seeds", action="store_true")
    args = parser.parse_args()
    if os.uname().nodename != args.host:
        raise SystemExit(f"refusing to run on {os.uname().nodename}")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    record = json.loads(args.lineage_record.read_text())
    datasets = sorted(row["dataset"] for row in record["rows"])
    jobs = _jobs(args.workers, datasets, args.out, args.include_split_seeds)
    pending = [job for job in jobs if not Path(job["out"]).exists()]
    print(f"jobs {len(jobs)} pending {len(pending)}", flush=True)
    if not pending:
        return
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=args.workers_parallel, mp_context=mp.get_context("spawn")) as pool:
        for result in pool.map(_one, pending):
            print(result["status"], result.get("path", ""), flush=True)


if __name__ == "__main__":
    main()
