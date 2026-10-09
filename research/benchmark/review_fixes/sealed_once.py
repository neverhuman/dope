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
REGISTRY = RESULTS / "review-fixes-sealed-v1"
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


def _published_samples() -> list[dict]:
    """Train-only published samples. Fit seed 11 only. Duplicates are refused."""
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
            if cell.get("fit_seed") not in (None, 11):
                continue
            if cell.get("size_multiplier") not in (1, 4) or cell.get("sample_seed") not in SEEDS:
                continue
            dataset = cell.get("dataset")
            if not isinstance(dataset, str):
                continue
            key = (method, configuration, dataset, cell["size_multiplier"], cell["sample_seed"])
            if key in found:
                raise SystemExit(f"duplicate published sample {key}")
            evidence = cell.get("metric_receipt") if method == "DOPE" else cell.get("sample_evidence")
            expected = evidence.get("sample_sha256") if isinstance(evidence, dict) else None
            csv_path = resolve_sample_csv(cell) if cell.get("status") == "ok" else None
            found[key] = {
                "method": method,
                "configuration": configuration,
                "dataset": dataset,
                "size": cell["size_multiplier"],
                "sample_seed": cell["sample_seed"],
                "csv": csv_path,
                "expected_sha256": expected if isinstance(expected, str) else None,
                "published_status": cell.get("status"),
            }
    return list(found.values())


def _committed_blob(path: Path) -> str:
    relative = path.resolve().relative_to(REPO.resolve()).as_posix()
    try:
        recorded = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=REPO, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise SystemExit("predeclare is not in committed HEAD") from error
    working = subprocess.check_output(["git", "hash-object", str(path)], cwd=REPO, text=True).strip()
    if recorded != working:
        raise SystemExit("working-tree predeclare differs from committed HEAD")
    return recorded


def _control_record(controls: Path, dataset: str, method: str, size: int, seed: int) -> dict:
    if method == "real_bootstrap_4n" and size == 4:
        stem = f"real-bootstrap-sample{seed}"
    elif method == "predictor_only_fit_seed_11":
        stem = f"predictor-fit11-sample{seed}-size{size}"
    else:
        raise SystemExit(f"undeclared control {method}")
    cell_path = controls / dataset / f"{stem}.json"
    csv_path = controls / dataset / f"{stem}.synthetic.csv"
    if not cell_path.is_file() or not csv_path.is_file():
        raise SystemExit(f"missing control sample {dataset} {method} size {size} seed {seed}")
    cell = json.loads(cell_path.read_text())
    if cell.get("status") != "ok" or cell.get("kind") == "split":
        raise SystemExit(f"control sample is not an ok primary cell {cell_path.name}")
    expected = cell.get("synthetic_sha256")
    if not isinstance(expected, str):
        raise SystemExit(f"control sample has no synthetic_sha256 {cell_path.name}")
    return {"csv": str(csv_path), "expected_sha256": expected}


def _preflight(controls: Path, datasets: list[str]) -> list[dict]:
    """Hash every declared train-only sample. Do not open a test file."""
    manifest = []
    problems = []
    for sample in _published_samples():
        csv_path = sample.get("csv")
        expected = sample.get("expected_sha256")
        if sample.get("published_status") != "ok" or not isinstance(csv_path, str):
            manifest.append({**sample, "file_sha256": None, "ready": False, "reason": "published_sample_absent"})
            continue
        path = Path(csv_path)
        if path.name == "test.csv" or not path.is_file():
            problems.append(f"published ok sample missing {sample['method']} {sample['dataset']} {sample['size']} {sample['sample_seed']}")
            continue
        digest = _sha256(path)
        if expected is not None and digest != expected:
            problems.append(f"published sample sha mismatch {sample['method']} {sample['dataset']}")
            continue
        manifest.append({**sample, "file_sha256": digest, "ready": True, "reason": None})
    for dataset in datasets:
        for method in ("real_bootstrap_4n", "predictor_only_fit_seed_11"):
            sizes = (4,) if method == "real_bootstrap_4n" else (1, 4)
            for size in sizes:
                for seed in SEEDS:
                    record = _control_record(controls, dataset, method, size, seed)
                    digest = _sha256(Path(record["csv"]))
                    if digest != record["expected_sha256"]:
                        problems.append(f"control sha mismatch {dataset} {method} {size} {seed}")
                        continue
                    manifest.append({
                        "method": method,
                        "configuration": method,
                        "dataset": dataset,
                        "size": size,
                        "sample_seed": seed,
                        "csv": record["csv"],
                        "expected_sha256": record["expected_sha256"],
                        "file_sha256": digest,
                        "ready": True,
                        "reason": None,
                    })
    if problems:
        raise SystemExit("preflight refused before unsealing: " + "; ".join(problems[:8]))
    return manifest


def _gate(args, manifest: list[dict]) -> dict:
    if args.out.resolve() != REGISTRY.resolve():
        raise SystemExit("sealed output must be research/benchmark/results/review-fixes-sealed-v1")
    if (REGISTRY / "panel.json").exists() or (REGISTRY / "started.json").exists():
        raise SystemExit("sealed registry already exists; refusing a second run")
    if os.environ.get("DOPE_RF_UNSEAL") != "1":
        raise SystemExit("refusing: DOPE_RF_UNSEAL=1 is not set")
    if args.predeclare.name == "test.csv":
        raise SystemExit("predeclare path is not a test file")
    blob = _committed_blob(args.predeclare)
    if blob != args.expect_git_blob:
        raise SystemExit("predeclare git blob does not match --expect-git-blob")
    digest = _sha256(args.predeclare)
    if digest != args.expect_sha256:
        raise SystemExit("predeclare sha256 does not match --expect-sha256")
    document = json.loads(args.predeclare.read_text())
    if document.get("sealed_test", {}).get("status") != "not_unsealed":
        raise SystemExit("predeclare sealed_test.status is no longer not_unsealed")
    ready = sum(1 for row in manifest if row.get("ready"))
    if ready == 0:
        raise SystemExit("preflight found no train-only samples")
    REGISTRY.mkdir(parents=True, exist_ok=True)
    started = REGISTRY / "started.json"
    payload = {
        "started_unix": time.time(),
        "predeclaration_sha256": digest,
        "predeclaration_git_blob": blob,
        "manifest_rows": len(manifest),
        "ready_rows": ready,
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
    if os.environ.get("DOPE_RF_UNSEAL") != "1":
        raise SystemExit("refusing: DOPE_RF_UNSEAL=1 is not set")
    if args.out.resolve() != REGISTRY.resolve():
        raise SystemExit("sealed output must be research/benchmark/results/review-fixes-sealed-v1")
    if (REGISTRY / "panel.json").exists() or (REGISTRY / "started.json").exists():
        raise SystemExit("sealed registry already exists; refusing a second run")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    names = {
        row["dataset"]: row["display_name"]
        for row in json.loads((RESULTS / "s3-lineage-record.json").read_text())["rows"]
    }
    manifest = _preflight(args.controls, sorted(names))
    document = _gate(args, manifest)
    frozen = document["sealed_test"]["configuration_frozen_by_this_file"]
    pilot = _pilot()
    indexed = {
        (row["method"], row["configuration"], row["dataset"], row["size"], row["sample_seed"]): row
        for row in manifest
    }
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
                    sample = indexed.get((method, configuration, dataset, size, seed))
                    cells.append(_one_cell(pilot, dataset, method, configuration, size, seed, sample, train, test, task))
        for method in control_methods:
            sizes = (4,) if method == "real_bootstrap_4n" else tuple(frozen["sizes"])
            for size in sizes:
                for seed in frozen["sample_seeds"]:
                    sample = indexed.get((method, method, dataset, size, seed))
                    cells.append(_one_cell(pilot, dataset, method, method, size, seed, sample, train, test, task))
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


def _one_cell(pilot, dataset, method, configuration, size, seed, sample, train, test, task) -> dict:
    base = {
        "dataset": dataset, "method": method, "configuration": configuration,
        "size": size, "sample_seed": seed, "official_tests_opened": True,
    }
    if not sample or not sample.get("ready"):
        return {**base, "status": "absent", "reason": (sample or {}).get("reason") or "train_only_sample_missing"}
    csv_path = sample.get("csv")
    expected = sample.get("file_sha256")
    if not isinstance(csv_path, str) or not isinstance(expected, str):
        return {**base, "status": "absent", "reason": "train_only_sample_missing"}
    path = Path(csv_path)
    digest = _sha256(path)
    stored = sample.get("expected_sha256")
    if digest != expected or (isinstance(stored, str) and digest != stored):
        return {**base, "status": "failed", "reason": "sample_sha256_mismatch"}
    try:
        synthetic = _read_numeric(path)
        if synthetic.shape[1] != train.shape[1] or test.shape[1] != train.shape[1]:
            return {**base, "status": "unavailable", "reason": "width_mismatch", "synthetic_sha256": digest}
        metrics = _retention(pilot, train, test, synthetic, task)
    except (ValueError, RuntimeError, OSError) as error:
        return {**base, "status": "failed", "reason": type(error).__name__, "synthetic_sha256": digest}
    return {**base, "status": "ok", "metrics": metrics, "synthetic_sha256": digest}


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
