"""One sealed official-test scoring run.

The predeclaration hash and DOPE_RF_UNSEAL=1 are required. If a start marker
or a finished panel already exists, the script exits before it builds a test
path. pilot_metrics.measure is not used, because that function refuses a file
named test.csv. The auditor hyperparameters match pilot_metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(REPO))
from research.benchmark.review_fixes.sample_paths import resolve_sample_csv
from research.benchmark.review_fixes.stats import cluster_of, median_ci

PILOT_PATH = HERE.parent / "pilot_metrics.py"
RESULTS = REPO / "research" / "benchmark" / "results"
WORKERS = Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker")
EVALUATOR = Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/evaluator")
AUDITORS = ("catboost", "linear", "mlp")
SEEDS = (101, 211, 307)


def _pilot():
    spec = importlib.util.spec_from_file_location("pilot_metrics_sealed", PILOT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_numeric(path: Path) -> np.ndarray:
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


def _retention(pilot, train, test, synthetic, task: str, seed: int = 1729) -> dict:
    x_real, y_real = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    x_synth, y_synth = synthetic[:, :-1], synthetic[:, -1]
    if task == "binary" and not set(np.unique(y_synth)).issubset({0.0, 1.0}):
        raise ValueError("synthetic binary target is not binary")
    null_prediction = float(np.mean(y_real))
    if task == "binary":
        from sklearn.metrics import log_loss
        probability = float(np.clip(null_prediction, 1e-9, 1 - 1e-9))
        null_loss = float(log_loss(y_test, np.full(len(y_test), probability), labels=[0, 1]))
    else:
        from sklearn.metrics import mean_squared_error
        null_loss = float(mean_squared_error(y_test, np.full(len(y_test), null_prediction)))
    utility = {}
    for name in AUDITORS:
        try:
            real_model = pilot._model(name, task, seed).fit(x_real, y_real)
            synth_model = pilot._model(name, task, seed).fit(x_synth, y_synth)
            trtr = pilot._loss(real_model, x_test, y_test, task)
            tstr = pilot._loss(synth_model, x_test, y_test, task)
            gap = null_loss - trtr
            informative = gap >= 0.01 * abs(null_loss) and gap != 0
            utility[name] = {
                "trtr_loss": trtr,
                "tstr_loss": tstr,
                "informative": informative,
                "retention": (null_loss - tstr) / gap if informative else None,
            }
        except (ValueError, RuntimeError) as error:
            utility[name] = {"status": "failed", "error_type": type(error).__name__}
    return {"null_loss": null_loss, "utility": utility, "rows": {
        "train": int(len(train)), "test": int(len(test)), "synthetic": int(len(synthetic)),
    }}


def _published_csvs() -> dict:
    """Train-only sample files already written by the published campaigns."""
    found = {}
    specs = (
        ("density-matched-population-validation.json", "DOPE", "features12_steps2048"),
        ("density-matched-population-validation.json", "GaussianCopula", "native_selected"),
        ("density-matched-population-validation.json", "Chow-Liu", "native_selected"),
        ("density-matched-population-validation.json", "independent_marginals", "native_selected"),
    )
    for filename, method, configuration in specs:
        document = json.loads((RESULTS / filename).read_text())
        for cell in document["cells"]:
            if cell.get("method") != method or cell.get("configuration") != configuration:
                continue
            if cell.get("size_multiplier") not in (1, 4) or cell.get("sample_seed") not in SEEDS:
                continue
            found[(method, configuration, cell["dataset"], cell["size_multiplier"], cell["sample_seed"])] = resolve_sample_csv(cell)
    return found


def _control_csv(controls: Path, dataset: str, method: str, size: int, seed: int) -> Path | None:
    if method == "real_bootstrap_4n" and size == 4:
        path = controls / dataset / f"real-bootstrap-sample{seed}.synthetic.csv"
    elif method == "predictor_only_fit_seed_11":
        path = controls / dataset / f"predictor-fit11-sample{seed}-size{size}.synthetic.csv"
    else:
        return None
    return path if path.is_file() else None


def _gate(args) -> dict:
    out = args.out
    if (out / "panel.json").exists() or (out / "started.json").exists():
        raise SystemExit("sealed output already exists; refusing a second run")
    if os.environ.get("DOPE_RF_UNSEAL") != "1":
        raise SystemExit("refusing: DOPE_RF_UNSEAL=1 is not set")
    if not args.controls.is_dir() or not any(args.controls.glob("*/*.synthetic.csv")):
        raise SystemExit("refusing to unseal before train-only control samples exist")
    if args.predeclare.name == "test.csv":
        raise SystemExit("predeclare path is not a test file")
    blob = subprocess.check_output(["git", "hash-object", str(args.predeclare)], text=True).strip()
    if blob != args.expect_git_blob:
        raise SystemExit("predeclare git blob does not match --expect-git-blob")
    digest = _sha256(args.predeclare)
    if digest != args.expect_sha256:
        raise SystemExit("predeclare sha256 does not match --expect-sha256")
    document = json.loads(args.predeclare.read_text())
    if document.get("sealed_test", {}).get("status") != "not_unsealed":
        raise SystemExit("predeclare sealed_test.status is no longer not_unsealed")
    out.mkdir(parents=True, exist_ok=True)
    started = out / "started.json"
    payload = {
        "started_unix": time.time(),
        "predeclaration_sha256": digest,
        "predeclaration_git_blob": blob,
        "official_tests_opened": False,
    }
    fd = os.open(started, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predeclare", type=Path, required=True)
    parser.add_argument("--expect-sha256", required=True)
    parser.add_argument("--expect-git-blob", required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    if os.uname().nodename != args.host:
        raise SystemExit(f"refusing to run on {os.uname().nodename}")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    document = _gate(args)
    frozen = document["sealed_test"]["configuration_frozen_by_this_file"]
    pilot = _pilot()
    names = {
        row["dataset"]: row["display_name"]
        for row in json.loads((RESULTS / "s3-lineage-record.json").read_text())["rows"]
    }
    published = _published_csvs()
    methods = [("DOPE", frozen["profile"])] + [
        (name, "native_selected") for name in ("GaussianCopula", "Chow-Liu", "independent_marginals")
    ]
    control_methods = ("real_bootstrap_4n", "predictor_only_fit_seed_11")
    cells = []
    opened = False
    for dataset in sorted(names):
        projection_path = WORKERS / dataset / "projection.json"
        train_path = WORKERS / dataset / "train.csv"
        test_path = EVALUATOR / dataset / "test.csv"
        if not projection_path.is_file() or not train_path.is_file() or not test_path.is_file():
            cells.append({"dataset": dataset, "status": "unavailable", "reason": "partition_missing"})
            continue
        task = json.loads(projection_path.read_text())["task"]
        train = _read_numeric(train_path)
        if not opened:
            opened = True
        test = _read_numeric(test_path)
        for method, configuration in methods:
            for size in frozen["sizes"]:
                for seed in frozen["sample_seeds"]:
                    csv_path = published.get((method, configuration, dataset, size, seed))
                    cells.append(_one_cell(
                        pilot, dataset, method, configuration, size, seed, csv_path, train, test, task,
                    ))
        for method in control_methods:
            sizes = (4,) if method == "real_bootstrap_4n" else tuple(frozen["sizes"])
            for size in sizes:
                for seed in frozen["sample_seeds"]:
                    csv_path = _control_csv(args.controls, dataset, method, size, seed)
                    cells.append(_one_cell(
                        pilot, dataset, method, method, size, seed,
                        str(csv_path) if csv_path else None, train, test, task,
                    ))
    summary = _summarize(cells, names)
    payload = {
        "format": "dope-review-fix-sealed-once",
        "version": 1,
        "official_tests_opened": opened,
        "unseal_count": 1 if opened else 0,
        "predeclaration_sha256": _sha256(args.predeclare),
        "frozen_configuration": frozen,
        "cells": cells,
        "summary": summary,
        "claims": {"mfs_v2": None, "ptf_v1": None, "release_safe_l3": None, "superiority": None, "formal_dp": False},
    }
    temporary = args.out / "panel.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out / "panel.json")
    for row in summary:
        if row["method"] == "DOPE" and row["auditor"] == "catboost":
            print(
                f"sealed DOPE CatBoost size {row['size']}n median {row['median']} "
                f"on {row['n']} informative lineages"
            )


def _one_cell(pilot, dataset, method, configuration, size, seed, csv_path, train, test, task) -> dict:
    base = {
        "dataset": dataset, "method": method, "configuration": configuration,
        "size": size, "sample_seed": seed, "official_tests_opened": True,
    }
    if not csv_path or not Path(csv_path).is_file():
        return {**base, "status": "absent", "reason": "train_only_sample_missing"}
    try:
        synthetic = _read_numeric(Path(csv_path))
        if synthetic.shape[1] != train.shape[1] or test.shape[1] != train.shape[1]:
            return {**base, "status": "unavailable", "reason": "width_mismatch"}
        metrics = _retention(pilot, train, test, synthetic, task)
    except (ValueError, RuntimeError, OSError) as error:
        return {**base, "status": "failed", "reason": type(error).__name__}
    return {**base, "status": "ok", "metrics": metrics, "synthetic_sha256": _sha256(Path(csv_path))}


def _summarize(cells: list[dict], names: dict) -> list[dict]:
    grouped = {}
    for cell in cells:
        if cell.get("status") != "ok":
            continue
        key = (cell["method"], cell["configuration"], cell["size"], cell["dataset"])
        grouped.setdefault(key, {})[cell["sample_seed"]] = cell
    rows = []
    methods = sorted({(key[0], key[1], key[2]) for key in grouped})
    for method, configuration, size in methods:
        for auditor in AUDITORS:
            values = {}
            for (method_i, configuration_i, size_i, dataset), seeds in grouped.items():
                if (method_i, configuration_i, size_i) != (method, configuration, size):
                    continue
                if set(seeds) != set(SEEDS):
                    continue
                triple = []
                keep = True
                for seed in SEEDS:
                    block = ((seeds[seed].get("metrics") or {}).get("utility") or {}).get(auditor) or {}
                    retention = block.get("retention")
                    if block.get("informative") is not True or not isinstance(retention, (int, float)):
                        keep = False
                        break
                    triple.append(float(retention))
                if keep:
                    values[dataset] = float(sorted(triple)[1])
            keys = sorted(values)
            summary = median_ci(
                [values[key] for key in keys],
                f"sealed|{method}|{configuration}|{size}|{auditor}",
                [cluster_of(names.get(key, key), key) for key in keys],
            )
            summary.update({"method": method, "configuration": configuration, "size": size, "auditor": auditor})
            rows.append(summary)
    return rows


if __name__ == "__main__":
    main()
