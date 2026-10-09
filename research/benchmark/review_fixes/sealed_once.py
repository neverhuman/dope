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
import math
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
APPROVED_HOST = "xbabe3"
CANONICAL_REGISTRY = Path("/mnt/fast-scratch/dope-benchmark/review-fixes-lane-b/sealed-v1")
REGISTRY = CANONICAL_REGISTRY
FREEZE_COMMIT = "c36d7f21b8ce62af81bc1f538e332c180e22eb95"
FREEZE_BLOB = "a0691b1efc848add288e17b22759169c515d4328"
FREEZE_SHA256 = "921865c374bc4f0d6dd88f3450409ea3c35a84ff9e8ce206d5772b7710dca3e5"
WORKERS = Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker")
EVALUATOR = Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/evaluator")
AUDITORS = ("catboost", "linear", "mlp")
SEEDS = (101, 211, 307)
PUBLISHED_SPECS = (
    ("DOPE", "features12_steps2048"),
    ("GaussianCopula", "native_selected"),
    ("Chow-Liu", "native_selected"),
    ("independent_marginals", "native_selected"),
)


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
    _assert_ledger_pins()
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


def _assert_frozen_constants(document: dict) -> None:
    frozen = document["sealed_test"]["configuration_frozen_by_this_file"]
    if document["sealed_test"]["status"] != "not_unsealed":
        raise SystemExit("predeclare sealed_test.status is no longer not_unsealed")
    if tuple(frozen["auditors"]) != AUDITORS:
        raise SystemExit("auditor set does not match the frozen configuration")
    if tuple(frozen["sample_seeds"]) != SEEDS:
        raise SystemExit("sample seeds do not match the frozen configuration")
    if tuple(frozen["sizes"]) != (1, 4):
        raise SystemExit("sizes do not match the frozen configuration")
    if frozen["fit_seed"] != 11 or frozen["auditor_seed"] != 1729:
        raise SystemExit("fit seed or auditor seed does not match the frozen configuration")
    if frozen["profile"] != "features12_steps2048" or frozen["method"] != "DOPE":
        raise SystemExit("profile does not match the frozen configuration")
    expected = (
        "GaussianCopula",
        "Chow-Liu",
        "independent_marginals",
        "real_bootstrap_4n",
        "predictor_only_fit_seed_11",
    )
    if tuple(frozen["comparators"]) != expected:
        raise SystemExit("comparators do not match the frozen configuration")


def _assert_original_freeze(path: Path, expect_sha256: str, expect_git_blob: str) -> dict:
    """Bind the run to the original committed predeclare, not a later edit."""
    if expect_sha256 != FREEZE_SHA256 or expect_git_blob != FREEZE_BLOB:
        raise SystemExit("expect pins are not the original predeclare freeze")
    if _sha256(path) != FREEZE_SHA256:
        raise SystemExit("predeclare sha256 is not the original freeze")
    if _committed_blob(path) != FREEZE_BLOB:
        raise SystemExit("HEAD predeclare differs from the original freeze")
    recorded = subprocess.check_output(
        ["git", "rev-parse", f"{FREEZE_COMMIT}:research/benchmark/review_fixes/predeclare.json"],
        cwd=REPO,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if recorded != FREEZE_BLOB:
        raise SystemExit("freeze commit does not contain the original predeclare blob")
    document = json.loads(path.read_text())
    _assert_frozen_constants(document)
    return document


def _csv_shape(path: Path) -> tuple[int, int]:
    if path.name == "test.csv":
        raise ValueError("shape check refuses a test file")
    rows = 0
    cols = None

    def consume(line: str) -> None:
        nonlocal rows, cols
        if not line.strip():
            return
        parts = [part.strip() for part in line.rstrip("\n").split(",")]
        width = len(parts)
        if cols is None:
            cols = width
        elif width != cols:
            raise ValueError("ragged table")
        for part in parts:
            if not math.isfinite(float(part)):
                raise ValueError("nonfinite table")
        rows += 1

    with path.open() as handle:
        first = handle.readline()
        if not first:
            raise ValueError("empty table")
        try:
            float(first.split(",")[0].strip())
        except ValueError:
            first = None
        if first is not None:
            consume(first)
        for line in handle:
            consume(line)
    if cols is None or rows < 1:
        raise ValueError("empty table")
    return rows, cols


def _shape_problem(sample_rows: int, sample_cols: int, train_rows: int, train_cols: int, size: int) -> str | None:
    if sample_cols != train_cols:
        return "width"
    if sample_rows != train_rows * size:
        return "row_count"
    if sample_rows < 1 or sample_cols < 2:
        return "empty"
    return None


def cell_path(registry: Path, dataset: str, method: str, configuration: str, size: int, seed: int) -> Path:
    return registry / "cells" / f"{dataset}__{method}__{configuration}__{size}__{seed}.json"


def partition_path(registry: Path, dataset: str) -> Path:
    return registry / "cells" / f"{dataset}__partition_missing.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def registry_mode(registry: Path, expect_sha256: str, expect_git_blob: str) -> str:
    """A fresh registry is the only mode. An existing start marker is not a resume."""
    if not isinstance(expect_sha256, str) or not isinstance(expect_git_blob, str):
        raise SystemExit("sealed freeze pin is missing")
    if (registry / "panel.json").exists():
        raise SystemExit("sealed registry already exists; refusing a second run")
    if (registry / "started.json").exists():
        raise SystemExit("sealed start marker already exists; refusing to reopen official tests")
    return "fresh"


def _control_identity(cell: dict, dataset: str, method: str, size: int, seed: int) -> str | None:
    expected_kind = "real_bootstrap_4n" if method == "real_bootstrap_4n" else "predictor_only"
    if cell.get("dataset") != dataset or cell.get("kind") != expected_kind:
        return "control_identity"
    if cell.get("sample_seed") != seed or cell.get("size") != size:
        return "control_identity"
    if cell.get("split_seed") not in (None,):
        return "control_identity"
    if method == "predictor_only_fit_seed_11" and cell.get("fit_seed") != 11:
        return "control_identity"
    if method == "real_bootstrap_4n" and cell.get("fit_seed") not in (None,):
        return "control_identity"
    if cell.get("status") != "ok":
        return "control_not_ok"
    return None


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
    identity = _control_identity(cell, dataset, method, size, seed)
    if identity is not None:
        raise SystemExit(f"{identity} {cell_path.name}")
    expected = cell.get("synthetic_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise SystemExit(f"control sample has no synthetic_sha256 {cell_path.name}")
    return {"csv": str(csv_path), "expected_sha256": expected}


def _assert_ledger_pins() -> None:
    panel_path = RESULTS / "review-fixes-receipts-v1" / "panel.json"
    if not panel_path.is_file():
        raise SystemExit("receipt panel pin file is missing")
    pins = json.loads(panel_path.read_text()).get("sources") or {}
    names = (
        "s3-lineage-record.json",
        "density-matched-population-validation.json",
        "sdv-matched-population-validation.json",
        "arf-matched-population-validation.json",
        "s3-matched-forest-confirmation-validation.json",
    )
    for name in names:
        recorded = pins.get(name)
        if not isinstance(recorded, str) or _sha256(RESULTS / name) != recorded:
            raise SystemExit(f"source drift {name}")


def _missing_published_grid(manifest: list[dict], datasets: list[str]) -> list[str]:
    present = {
        (row.get("method"), row.get("configuration"), row.get("dataset"), row.get("size"), row.get("sample_seed"))
        for row in manifest
        if row.get("method") in {spec[0] for spec in PUBLISHED_SPECS}
    }
    missing = []
    for dataset in datasets:
        for method, configuration in PUBLISHED_SPECS:
            for size in (1, 4):
                for seed in SEEDS:
                    if (method, configuration, dataset, size, seed) not in present:
                        missing.append(f"{method} {dataset} size {size} seed {seed}")
    return missing


def _published_problem(sample: dict) -> str | None:
    """A published ok cell with no 64-hex sample pin cannot be scored."""
    if sample.get("published_status") != "ok" or not isinstance(sample.get("csv"), str):
        return None
    expected = sample.get("expected_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return (
            f"published sample has no sha pin {sample.get('method')} {sample.get('dataset')} "
            f"{sample.get('size')} {sample.get('sample_seed')}"
        )
    return None


def _preflight(controls: Path, datasets: list[str]) -> list[dict]:
    """Hash every declared train-only sample. Do not open a test file."""
    manifest = []
    problems = []
    train_shapes: dict[str, tuple[int, int]] = {}
    for sample in _published_samples():
        csv_path = sample.get("csv")
        expected = sample.get("expected_sha256")
        if sample.get("published_status") != "ok" or not isinstance(csv_path, str):
            manifest.append({**sample, "file_sha256": None, "ready": False, "reason": "published_sample_absent"})
            continue
        problem = _published_problem(sample)
        if problem is not None:
            problems.append(problem)
            continue
        path = Path(csv_path)
        if path.name == "test.csv" or not path.is_file():
            problems.append(f"published ok sample missing {sample['method']} {sample['dataset']} {sample['size']} {sample['sample_seed']}")
            continue
        digest = _sha256(path)
        if digest != expected:
            problems.append(f"published sample sha mismatch {sample['method']} {sample['dataset']}")
            continue
        dataset = sample["dataset"]
        try:
            sample_rows, sample_cols = _csv_shape(path)
            if dataset not in train_shapes:
                train_shapes[dataset] = _csv_shape(WORKERS / dataset / "train.csv")
            train_rows, train_cols = train_shapes[dataset]
        except (OSError, ValueError) as error:
            problems.append(
                f"published sample shape {sample['method']} {dataset} {type(error).__name__}"
            )
            continue
        shape = _shape_problem(sample_rows, sample_cols, train_rows, train_cols, int(sample["size"]))
        if shape is not None:
            problems.append(f"published sample {shape} {sample['method']} {dataset} size {sample['size']}")
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
                    try:
                        sample_rows, sample_cols = _csv_shape(Path(record["csv"]))
                        if dataset not in train_shapes:
                            train_shapes[dataset] = _csv_shape(WORKERS / dataset / "train.csv")
                        train_rows, train_cols = train_shapes[dataset]
                    except (OSError, ValueError) as error:
                        problems.append(f"control sample shape {dataset} {method} {type(error).__name__}")
                        continue
                    shape = _shape_problem(sample_rows, sample_cols, train_rows, train_cols, size)
                    if shape is not None:
                        problems.append(f"control sample {shape} {dataset} {method} size {size}")
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
    problems.extend(_missing_published_grid(manifest, datasets))
    if problems:
        raise SystemExit("preflight refused before unsealing: " + "; ".join(problems[:8]))
    return manifest


def _gate(args, manifest: list[dict], registry: Path | None = None) -> dict:
    registry = REGISTRY if registry is None else registry
    if args.out.resolve() != registry.resolve():
        raise SystemExit(
            "sealed output must be /mnt/fast-scratch/dope-benchmark/review-fixes-lane-b/sealed-v1"
        )
    if (registry / "panel.json").exists() or (registry / "started.json").exists():
        raise SystemExit("sealed registry already exists; refusing a second run")
    if os.environ.get("DOPE_RF_UNSEAL") != "1":
        raise SystemExit("refusing: DOPE_RF_UNSEAL=1 is not set")
    if getattr(args, "host", None) != APPROVED_HOST:
        raise SystemExit(f"sealed host must be {APPROVED_HOST}")
    if args.predeclare.name == "test.csv":
        raise SystemExit("predeclare path is not a test file")
    document = _assert_original_freeze(args.predeclare, args.expect_sha256, args.expect_git_blob)
    digest = FREEZE_SHA256
    blob = FREEZE_BLOB
    ready = sum(1 for row in manifest if row.get("ready"))
    if ready == 0:
        raise SystemExit("preflight found no train-only samples")
    registry.mkdir(parents=True, exist_ok=True)
    started = registry / "started.json"
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
    if os.uname().nodename != APPROVED_HOST or args.host != APPROVED_HOST:
        raise SystemExit(f"sealed host must be {APPROVED_HOST}")
    if os.environ.get("DOPE_RF_UNSEAL") != "1":
        raise SystemExit("refusing: DOPE_RF_UNSEAL=1 is not set")
    if args.out.resolve() != REGISTRY.resolve():
        raise SystemExit(
            "sealed output must be /mnt/fast-scratch/dope-benchmark/review-fixes-lane-b/sealed-v1"
        )
    mode = registry_mode(REGISTRY, args.expect_sha256, args.expect_git_blob)
    if mode != "fresh":
        raise SystemExit("sealed registry mode must be fresh")
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
        jobs = []
        for method, configuration in methods:
            for size in frozen["sizes"]:
                for seed in frozen["sample_seeds"]:
                    jobs.append((method, configuration, size, seed))
        for method in control_methods:
            sizes = (4,) if method == "real_bootstrap_4n" else tuple(frozen["sizes"])
            for size in sizes:
                for seed in frozen["sample_seeds"]:
                    jobs.append((method, method, size, seed))
        missing_marker = partition_path(REGISTRY, dataset)
        stored_missing = read_checkpoint(missing_marker)
        if stored_missing is not None:
            cells.append(stored_missing)
            continue
        projection_path = WORKERS / dataset / "projection.json"
        train_path = WORKERS / dataset / "train.csv"
        test_path = EVALUATOR / dataset / "test.csv"
        if not projection_path.is_file() or not train_path.is_file() or not test_path.is_file():
            cell = {"dataset": dataset, "status": "unavailable", "reason": "partition_missing"}
            write_json(missing_marker, cell)
            cells.append(cell)
            continue
        pending = [
            job for job in jobs
            if read_checkpoint(cell_path(REGISTRY, dataset, job[0], job[1], job[2], job[3])) is None
        ]
        if not pending:
            for method, configuration, size, seed in jobs:
                cells.append(read_checkpoint(cell_path(REGISTRY, dataset, method, configuration, size, seed)))
            continue
        task = json.loads(projection_path.read_text())["task"]
        train = _read_numeric(train_path)
        test = _read_numeric(test_path)
        opened = True
        for method, configuration, size, seed in jobs:
            path = cell_path(REGISTRY, dataset, method, configuration, size, seed)
            stored = read_checkpoint(path)
            if stored is None:
                sample = indexed.get((method, configuration, dataset, size, seed))
                stored = _one_cell(
                    pilot, dataset, method, configuration, size, seed, sample, train, test, task,
                )
                write_json(path, stored)
            cells.append(stored)
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
