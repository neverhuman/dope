from __future__ import annotations

import contextlib
import hashlib
import html
import json
import math
import shutil
import socket
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

from .eval import COMPONENT_WEIGHTS, rbcs, score_synthetic_dataset
from .io import load_real_split
from .kernel import fit_kernel_from_arrays, sample_kernel
from .utils import canonical_dumps, clip01, normalize_task, qfloat


SEED = 1729
KPI_FIELDS = (
    "rbcs",
    "fidelity_real",
    "transfer_utility",
    "feature_importance",
    "distribution_copula",
    "residual_or_calibration",
    "prediction_agreement",
)
DEPENDENCE_KINDS = ("auto", "ind", "gauss_copula")
TARGET_KINDS = ("linear_sparse", "lift_logit", "tree_piecewise")
EMPQ_KINDS = (None, 9, 17, 33)
EventCallback = Callable[[dict[str, Any]], None]
SUPPORTED_TUNER_TASKS = ("regression", "binary")
BINARY_SHORTLIST_TOP_K = 2
BINARY_LABEL_MODES = ("shortlist", "all-fast")
SYNTHETIC_LABEL_MODES = ("anchor-consistency", "fast-relabel")
SCORER_VERSION = 3
ANCHOR_CONSISTENCY_LABEL_WEIGHT = 0.25
FAST_RELABEL_SYNTHETIC_WEIGHT = 0.75
REAL_LABEL_WEIGHT = 1.0
TRUST_GAP_LIMIT = 0.01
TRUST_AUDIT_DELTA_LIMIT = 0.02
TRUST_REQUIRED_SEEDS = 3
TRUST_FULL_KPI_AUDIT_TARGET = 200
TRAINING_MIXES = {
    "real-first": (0.8, 0.2),
    "synthetic-heavy": (0.2, 0.8),
    "real-only": (1.0, 0.0),
}
ALL_DATASETS_PREP_SECONDS_CAP = 900.0
ALL_DATASETS_PREP_FRACTION = 0.15
ALL_DATASETS_MIN_TRAIN_SECONDS = 60.0


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    task: str
    dataset_id: str


@dataclass(frozen=True)
class TunerExample:
    spec: DatasetSpec
    features: np.ndarray
    kpis: np.ndarray
    best_index: int
    records: tuple[dict[str, Any], ...]
    source: str = "real"
    anchor_dataset_id: str | None = None
    label_quality: dict[str, Any] = field(default_factory=dict)
    augmentation_kernel: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    sample_rows: dict[str, int] = field(default_factory=dict)
    label_weight: float = REAL_LABEL_WEIGHT


@dataclass(frozen=True)
class SmokeScoreCache:
    real_score: float
    real_pred: np.ndarray
    real_importance: np.ndarray


@dataclass(frozen=True)
class BinaryDiagnostics:
    prevalence: float
    class_balance: float
    target_corr_mean: float
    target_corr_max: float
    feature_corr_mean: float
    feature_corr_max: float
    logistic_score: float
    prediction_std: float
    rows: int
    cols: int


def parse_tasks(value: str | list[str] | tuple[str, ...]) -> list[str]:
    raw = value if isinstance(value, (list, tuple)) else str(value).split(",")
    tasks: list[str] = []
    for item in raw:
        task = normalize_task(str(item))
        if task not in tasks:
            tasks.append(task)
    if not tasks:
        raise ValueError("at least one task is required")
    return tasks


def select_smoke_datasets(
    corpus: str | Path,
    tasks: str | list[str] | tuple[str, ...] = "regression,binary",
    max_datasets: int = 8,
    seed: int = SEED,
) -> list[DatasetSpec]:
    del seed
    root = Path(corpus)
    requested = parse_tasks(tasks)
    max_datasets = int(max_datasets)
    if max_datasets <= 0:
        raise ValueError("max_datasets must be positive")

    by_task: dict[str, list[DatasetSpec]] = {task: [] for task in requested}
    seen: set[Path] = set()
    quota = max(1, math.ceil(max_datasets / len(requested)))

    def add(spec: DatasetSpec | None) -> None:
        if spec is None or spec.task not in by_task:
            return
        resolved = spec.path.resolve()
        if resolved in seen:
            return
        by_task[spec.task].append(spec)
        seen.add(resolved)

    for spec in _manifest_specs(root, requested):
        add(spec)
        if all(len(items) >= quota for items in by_task.values()):
            break

    if not all(len(items) >= quota for items in by_task.values()):
        for spec in _scan_specs(root, requested):
            add(spec)
            if all(len(items) >= quota for items in by_task.values()):
                break

    selected: list[DatasetSpec] = []
    queues = {task: list(items) for task, items in by_task.items()}
    while len(selected) < max_datasets and any(queues.values()):
        for task in requested:
            if queues[task] and len(selected) < max_datasets:
                selected.append(queues[task].pop(0))

    if not selected:
        raise FileNotFoundError(f"no train.csv/test.csv datasets found under {root}")
    return selected


def select_all_supported_datasets(
    corpus: str | Path,
    tasks: str | list[str] | tuple[str, ...] = "regression,binary",
    seed: int = SEED,
) -> list[DatasetSpec]:
    root = Path(corpus)
    requested = [task for task in parse_tasks(tasks) if task in SUPPORTED_TUNER_TASKS]
    if not requested:
        raise ValueError("at least one supported task is required")

    selected: list[DatasetSpec] = []
    seen: set[Path] = set()

    def add(spec: DatasetSpec) -> None:
        resolved = spec.path.resolve()
        if resolved in seen:
            return
        selected.append(spec)
        seen.add(resolved)

    rng = np.random.default_rng(seed)

    for spec in _balanced_task_dir_specs(root, requested, rng):
        add(spec)

    if not selected:
        for spec in _balanced_specs_by_task(list(_manifest_specs(root, requested)), requested, rng):
            add(spec)

    if not selected:
        raise FileNotFoundError(f"no train.csv/test.csv datasets found under {root}")
    return selected


def _balanced_task_dir_specs(root: Path, tasks: list[str], rng: np.random.Generator) -> Iterator[DatasetSpec]:
    queues = {task: _shuffled_specs(list(_task_dir_specs(root, [task])), rng) for task in tasks}
    while any(queues.values()):
        for task in tasks:
            if queues[task]:
                yield queues[task].pop(0)


def _balanced_specs_by_task(specs: list[DatasetSpec], tasks: list[str], rng: np.random.Generator) -> Iterator[DatasetSpec]:
    queues = {task: _shuffled_specs([spec for spec in specs if spec.task == task], rng) for task in tasks}
    while any(queues.values()):
        for task in tasks:
            if queues[task]:
                yield queues[task].pop(0)


def _shuffled_specs(specs: list[DatasetSpec], rng: np.random.Generator) -> list[DatasetSpec]:
    if len(specs) <= 1:
        return list(specs)
    order = rng.permutation(len(specs))
    return [specs[int(idx)] for idx in order]


def _task_dir_specs(root: Path, tasks: list[str]) -> Iterator[DatasetSpec]:
    for task in tasks:
        task_dir = root / task
        if not task_dir.exists():
            continue
        for child in sorted(task_dir.iterdir(), key=lambda item: item.name):
            if child.name.startswith("."):
                continue
            if child.is_dir() and (child / "train.csv").exists() and (child / "test.csv").exists():
                yield DatasetSpec(path=child, task=task, dataset_id=child.name)


def _manifest_specs(root: Path, tasks: list[str]) -> Iterator[DatasetSpec]:
    manifest = root / "manifest.json"
    if not manifest.exists():
        return
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    rows = sorted(
        payload.get("datasets", []),
        key=lambda item: (
            str(item.get("task_type", "")),
            str(item.get("dataset_hash", "")),
            str(item.get("output_dir", "")),
        ),
    )
    for row in rows:
        try:
            task = normalize_task(str(row.get("task_type", "")))
        except ValueError:
            continue
        if task not in tasks:
            continue
        dataset_id = str(row.get("dataset_hash") or "")
        for path in _candidate_dataset_paths(root, task, dataset_id, row.get("output_dir")):
            spec = _spec_from_dir(path, fallback_task=task, dataset_id=dataset_id)
            if spec is not None:
                yield spec
                break


def _scan_specs(root: Path, tasks: list[str]) -> Iterator[DatasetSpec]:
    for task in tasks:
        task_dir = root / task
        if task_dir.exists():
            for child in sorted(task_dir.iterdir(), key=lambda item: item.name):
                spec = _spec_from_dir(child, fallback_task=task)
                if spec is not None:
                    yield spec

    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.name.startswith(".") or child.name in set(tasks):
            continue
        spec = _spec_from_dir(child)
        if spec is not None and spec.task in tasks:
            yield spec


def _candidate_dataset_paths(root: Path, task: str, dataset_id: str, output_dir: Any) -> Iterator[Path]:
    if dataset_id:
        yield root / task / dataset_id
        yield root / dataset_id
    if output_dir:
        output_path = Path(str(output_dir))
        yield output_path
        yield root / output_path.name


def _spec_from_dir(path: Path, fallback_task: str | None = None, dataset_id: str | None = None) -> DatasetSpec | None:
    if not path.is_dir() or not (path / "train.csv").exists() or not (path / "test.csv").exists():
        return None
    meta_path = path / "meta.json"
    task = fallback_task
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            task = normalize_task(str(meta.get("task_type", task or "")))
            dataset_id = str(meta.get("dataset_hash") or dataset_id or path.name)
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    if task is None:
        return None
    return DatasetSpec(path=path, task=normalize_task(task), dataset_id=dataset_id or path.name)


def candidate_configs(task: str) -> list[dict[str, Any]]:
    task = normalize_task(task)
    target_options = ["linear_sparse", "tree_piecewise"] if task == "regression" else ["lift_logit", "tree_piecewise"]
    return [
        {"dependence_kind": dep, "target_kind": target, "empq_k": empq_k}
        for dep in DEPENDENCE_KINDS
        for target in target_options
        for empq_k in EMPQ_KINDS
    ]


def prepare_tuner_examples(
    specs: list[DatasetSpec],
    row_cap: int = 64,
    seed: int = SEED,
    event_callback: EventCallback | None = None,
    full_kpi_every: int = 0,
    full_kpi_top_k: int = 3,
    deadline: float | None = None,
    binary_label_mode: str = "shortlist",
    label_modes_by_dataset: dict[str, str] | None = None,
    full_kpi_dataset_keys: set[str] | None = None,
    example_cache_dir: str | Path | None = None,
    adaptive_relabel: bool = False,
) -> list[TunerExample]:
    _validate_binary_label_mode(binary_label_mode)
    examples = []
    for dataset_index, spec in enumerate(specs):
        if _deadline_reached(deadline) and examples:
            _emit_event(
                event_callback,
                {
                    "type": "time_limit_reached",
                    "phase": "dataset_selection",
                    "datasets_prepared": len(examples),
                    "datasets_requested": len(specs),
                },
            )
            break
        spec_key = _spec_identity(spec)
        spec_label_mode = (label_modes_by_dataset or {}).get(spec_key, binary_label_mode)
        _validate_binary_label_mode(spec_label_mode)
        full_kpi = (
            spec_key in (full_kpi_dataset_keys or set())
            or (int(full_kpi_every) > 0 and (dataset_index == 0 or (dataset_index + 1) % int(full_kpi_every) == 0))
        )
        dataset_seed = seed + dataset_index * 997
        examples.append(
            _prepare_tuner_example_cached(
                spec,
                row_cap=row_cap,
                seed=dataset_seed,
                event_callback=event_callback,
                dataset_index=dataset_index,
                total_datasets=len(specs),
                full_kpi=full_kpi,
                full_kpi_top_k=full_kpi_top_k,
                binary_label_mode=spec_label_mode,
                example_cache_dir=example_cache_dir,
                adaptive_relabel=adaptive_relabel,
            )
        )
    return examples


def _prepare_tuner_example_cached(
    spec: DatasetSpec,
    row_cap: int,
    seed: int,
    event_callback: EventCallback | None,
    dataset_index: int,
    total_datasets: int | None,
    full_kpi: bool,
    full_kpi_top_k: int,
    binary_label_mode: str,
    example_cache_dir: str | Path | None,
    adaptive_relabel: bool,
) -> TunerExample:
    if example_cache_dir is None:
        return prepare_tuner_example(
            spec,
            row_cap=row_cap,
            seed=seed,
            event_callback=event_callback,
            dataset_index=dataset_index,
            total_datasets=total_datasets,
            full_kpi=full_kpi,
            full_kpi_top_k=full_kpi_top_k,
            binary_label_mode=binary_label_mode,
            adaptive_relabel=adaptive_relabel,
        )
    cache_key = _example_cache_key(
        spec,
        row_cap=row_cap,
        seed=seed,
        binary_label_mode=binary_label_mode,
        full_kpi=full_kpi,
        full_kpi_top_k=full_kpi_top_k,
    )
    cache_path = _example_cache_path(example_cache_dir, cache_key)
    cached = _load_cached_example(cache_path, cache_key)
    if cached is not None:
        _emit_event(
            event_callback,
            {
                "type": "example_cache_hit",
                "dataset_index": int(dataset_index),
                "dataset": _spec_json(spec),
                "cache_key": cache_key,
                "cache_path": str(cache_path),
                "binary_label_mode": binary_label_mode if spec.task == "binary" else "all-fast",
                "full_kpi": bool(full_kpi),
            },
        )
        return cached
    _emit_event(
        event_callback,
        {
            "type": "example_cache_miss",
            "dataset_index": int(dataset_index),
            "dataset": _spec_json(spec),
            "cache_key": cache_key,
            "cache_path": str(cache_path),
            "binary_label_mode": binary_label_mode if spec.task == "binary" else "all-fast",
            "full_kpi": bool(full_kpi),
        },
    )
    example = prepare_tuner_example(
        spec,
        row_cap=row_cap,
        seed=seed,
        event_callback=event_callback,
        dataset_index=dataset_index,
        total_datasets=total_datasets,
        full_kpi=full_kpi,
        full_kpi_top_k=full_kpi_top_k,
        binary_label_mode=binary_label_mode,
        adaptive_relabel=adaptive_relabel,
    )
    _write_cached_example(cache_path, cache_key, example)
    _emit_event(
        event_callback,
        {
            "type": "example_cache_write",
            "dataset_index": int(dataset_index),
            "dataset": _spec_json(spec),
            "cache_key": cache_key,
            "cache_path": str(cache_path),
        },
    )
    return example


def prepare_tuner_example(
    spec: DatasetSpec,
    row_cap: int = 64,
    seed: int = SEED,
    event_callback: EventCallback | None = None,
    dataset_index: int = 0,
    total_datasets: int | None = None,
    full_kpi: bool = False,
    full_kpi_top_k: int = 3,
    binary_label_mode: str = "shortlist",
    adaptive_relabel: bool = False,
) -> TunerExample:
    _validate_binary_label_mode(binary_label_mode)
    _emit_event(
        event_callback,
        {
            "type": "dataset_start",
            "dataset_index": int(dataset_index),
            "total_datasets": total_datasets,
            "dataset": _spec_json(spec),
            "binary_label_mode": binary_label_mode if spec.task == "binary" else "all-fast",
            "full_kpi": bool(full_kpi),
        },
    )
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, _ = load_real_split(spec.path, spec.task)
    X_train_full = _clean(X_train_raw)
    y_train_full = _clean_y(y_train_raw, spec.task)
    X_test_full = _clean(X_test_raw)
    y_test_full = _clean_y(y_test_raw, spec.task)
    train_cap = _effective_sample_cap(y_train_full, row_cap, spec.task, "train")
    test_requested_cap = max(32, row_cap // 2)
    test_cap = _effective_sample_cap(y_test_full, test_requested_cap, spec.task, "test")
    X_train, y_train = _cap_xy(X_train_full, y_train_full, train_cap, spec.task, seed)
    X_test, y_test = _cap_xy(X_test_full, y_test_full, test_cap, spec.task, seed + 1)
    _warn_binary_sample_quality(event_callback, spec, dataset_index, "train", y_train_full, y_train)
    _warn_binary_sample_quality(event_callback, spec, dataset_index, "test", y_test_full, y_test)
    dataset_features = _dataset_feature_vector(X_train, y_train, X_test, y_test, spec.task)
    _emit_event(
        event_callback,
        {
            "type": "dataset_loaded",
            "dataset_index": int(dataset_index),
            "dataset": _spec_json(spec),
            "rows": {"train": int(len(X_train)), "test": int(len(X_test))},
            "cols": int(X_train.shape[1] + 1),
            "task": spec.task,
        },
    )

    rows = []
    features = []
    candidate_assets: list[dict[str, Any] | None] = []
    configs = candidate_configs(spec.task)
    smoke_cache = _build_smoke_score_cache(X_train, y_train, X_test, y_test, spec.task, seed)
    binary_diagnostics = _binary_diagnostics(X_train, y_train, X_test, y_test, smoke_cache) if spec.task == "binary" else None
    proxy_rows: dict[int, dict[str, Any]] = {}
    shortlist_indexes = set(range(len(configs)))
    if binary_diagnostics is not None:
        proxy_rows = {
            candidate_index: _binary_proxy_candidate_row(
                candidate_index,
                config,
                binary_diagnostics,
                rows=len(X_train),
                cols=X_train.shape[1] + 1,
            )
            for candidate_index, config in enumerate(configs)
        }
        if binary_label_mode == "all-fast":
            shortlist_indexes = set(range(len(configs)))
        else:
            shortlist_indexes = set(
                sorted(proxy_rows, key=lambda idx: float(proxy_rows[idx].get("rbcs", 0.0)), reverse=True)[
                    : min(len(configs), BINARY_SHORTLIST_TOP_K)
                ]
            )
        _emit_event(
            event_callback,
            {
                "type": "binary_diagnostics",
                "dataset_index": int(dataset_index),
                "dataset": _spec_json(spec),
                "diagnostics": _binary_diagnostics_json(binary_diagnostics),
            },
        )
        _emit_event(
            event_callback,
            {
                "type": "binary_shortlist",
                "dataset_index": int(dataset_index),
                "dataset": _spec_json(spec),
                "label_mode": binary_label_mode,
                "candidate_total": len(configs),
                "shortlist_top_k": len(shortlist_indexes),
                "shortlisted_candidates": sorted(shortlist_indexes),
            },
        )
    for candidate_index, config in enumerate(configs):
        if candidate_index not in shortlist_indexes:
            row = proxy_rows[candidate_index]
            candidate_assets.append(None)
            proxy_kernel = _binary_proxy_kernel(config, int(row["kernel_bytes"]), X_train.shape[1] + 1)
            feature_vec = _candidate_feature_vector(config, proxy_kernel, len(X_train), X_train.shape[1] + 1)
            _emit_event(
                event_callback,
                {
                    "type": "candidate_score",
                    "dataset_index": int(dataset_index),
                    "candidate_index": int(candidate_index),
                    "dataset": _spec_json(spec),
                    **_event_candidate_fields(row),
                },
            )
            rows.append(row)
            features.append(np.concatenate([dataset_features, feature_vec]))
            continue
        _emit_event(
            event_callback,
            {
                "type": "candidate_fit_start",
                "dataset_index": int(dataset_index),
                "candidate_index": int(candidate_index),
                "candidate_total": len(configs),
                "dataset": _spec_json(spec),
                "config": dict(config),
            },
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                kernel = fit_kernel_from_arrays(
                    X_train,
                    y_train,
                    spec.task,
                    seed=seed,
                    dependence_kind=str(config["dependence_kind"]),
                    target_kind=str(config["target_kind"]),
                    empq_k=config["empq_k"],
                )
                synth = sample_kernel(kernel, len(X_train), seed=seed)
                report = _score_synthetic_dataset_smoke(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    synth[:, :-1],
                    synth[:, -1],
                    spec.task,
                    seed,
                    cache=smoke_cache,
                )
            row = _candidate_metric_row(
                candidate_index,
                config,
                kernel,
                report,
                rows=len(X_train),
                cols=X_train.shape[1] + 1,
                score_mode="fast",
            )
            if candidate_index in proxy_rows:
                _attach_proxy_quality(row, proxy_rows[candidate_index], "fast")
            candidate_assets.append({"kernel": kernel, "synth": synth, "config": config})
            feature_vec = _candidate_feature_vector(config, kernel, len(X_train), X_train.shape[1] + 1)
            _emit_event(
                event_callback,
                {
                    "type": "candidate_score",
                    "dataset_index": int(dataset_index),
                    "candidate_index": int(candidate_index),
                    "dataset": _spec_json(spec),
                    **_event_candidate_fields(row),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive for long unattended runs
            row = _failed_candidate_record(candidate_index, config, exc)
            candidate_assets.append(None)
            feature_vec = np.zeros(_candidate_feature_dim(), dtype=np.float32)
            _emit_event(
                event_callback,
                {
                    "type": "candidate_failed",
                    "dataset_index": int(dataset_index),
                    "candidate_index": int(candidate_index),
                    "dataset": _spec_json(spec),
                    "config": dict(config),
                    "error": str(exc),
                },
            )
        rows.append(row)
        features.append(np.concatenate([dataset_features, feature_vec]))

    if full_kpi and rows:
        scorable_indexes = [idx for idx, asset in enumerate(candidate_assets) if asset is not None]
        top_indexes = sorted(scorable_indexes, key=lambda idx: float(rows[idx].get("rbcs", 0.0)), reverse=True)[
            : max(0, int(full_kpi_top_k))
        ]
        for candidate_index in top_indexes:
            asset = candidate_assets[candidate_index]
            if asset is None:
                continue
            _emit_event(
                event_callback,
                {
                    "type": "full_kpi_start",
                    "dataset_index": int(dataset_index),
                    "candidate_index": int(candidate_index),
                    "dataset": _spec_json(spec),
                    "config": dict(asset["config"]),
                    "fast_rbcs": rows[candidate_index].get("rbcs"),
                },
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                synth = asset["synth"]
                full_report = score_synthetic_dataset(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    synth[:, :-1],
                    synth[:, -1],
                    spec.task,
                    seed,
                )
            full_row = _candidate_metric_row(
                candidate_index,
                asset["config"],
                asset["kernel"],
                full_report,
                rows=len(X_train),
                cols=X_train.shape[1] + 1,
                score_mode="full",
            )
            _merge_full_metric_row(rows[candidate_index], full_row)
            if candidate_index in proxy_rows:
                _attach_proxy_quality(rows[candidate_index], proxy_rows[candidate_index], "full")
            _emit_event(
                event_callback,
                {
                    "type": "full_kpi_score",
                    "dataset_index": int(dataset_index),
                    "candidate_index": int(candidate_index),
                    "dataset": _spec_json(spec),
                    **_event_candidate_fields(rows[candidate_index]),
                },
            )

    if adaptive_relabel and rows:
        next_row_cap = _adaptive_relabel_row_cap(row_cap, y_train_full, y_test_full, spec.task, rows)
        if next_row_cap is not None:
            _emit_event(
                event_callback,
                {
                    "type": "adaptive_relabel",
                    "dataset_index": int(dataset_index),
                    "dataset": _spec_json(spec),
                    "from_row_cap": int(row_cap),
                    "to_row_cap": int(next_row_cap),
                    "reason": "top_candidates_close",
                },
            )
            return prepare_tuner_example(
                spec,
                row_cap=next_row_cap,
                seed=seed,
                event_callback=event_callback,
                dataset_index=dataset_index,
                total_datasets=total_datasets,
                full_kpi=full_kpi,
                full_kpi_top_k=full_kpi_top_k,
                binary_label_mode=binary_label_mode,
                adaptive_relabel=False,
            )

    kpis = [[float(row[field]) for field in KPI_FIELDS] for row in rows]
    kpi_array = np.asarray(kpis, dtype=np.float32)
    best_index = int(np.argmax(kpi_array[:, 0]))
    label_quality = _label_quality_report(
        spec,
        y_train_full,
        y_test_full,
        y_train,
        y_test,
        requested_row_cap=row_cap,
        effective_train_cap=train_cap,
        effective_test_cap=test_cap,
        records=rows,
        sample_seed=seed,
        binary_label_mode=binary_label_mode,
    )
    augmentation_kernel = _augmentation_kernel_from_assets(rows, candidate_assets)
    _emit_event(
        event_callback,
        {
            "type": "label_quality",
            "dataset_index": int(dataset_index),
            "dataset": _spec_json(spec),
            "quality": label_quality,
        },
    )
    _emit_event(
        event_callback,
        {
            "type": "dataset_complete",
            "dataset_index": int(dataset_index),
            "dataset": _spec_json(spec),
            "best_candidate_index": best_index,
            "label_quality": label_quality,
            **_event_candidate_fields(rows[best_index]),
        },
    )
    return TunerExample(
        spec=spec,
        features=np.asarray(features, dtype=np.float32),
        kpis=kpi_array,
        best_index=best_index,
        records=tuple(rows),
        source="real",
        label_quality=label_quality,
        augmentation_kernel=augmentation_kernel,
        sample_rows={"train": int(len(X_train)), "test": int(len(X_test))},
        label_weight=REAL_LABEL_WEIGHT,
    )


def _candidate_metric_row(
    candidate_index: int,
    config: dict[str, Any],
    kernel: dict[str, Any],
    report: dict[str, Any],
    rows: int,
    cols: int,
    score_mode: str,
) -> dict[str, Any]:
    kernel_bytes = int(kernel["kernel_bytes"])
    metric_row = {
        "candidate_index": int(candidate_index),
        "config": dict(config),
        "kernel_bytes": kernel_bytes,
        "rbcs": rbcs(report["fidelity_real"], kernel_bytes, rows, cols),
        "fidelity_real": report["fidelity_real"],
        "score_mode": score_mode,
        **report["components"],
    }
    metric_row["kernel_summary"] = _kernel_summary(kernel, config, metric_row)
    return metric_row


def _merge_full_metric_row(row: dict[str, Any], full_row: dict[str, Any]) -> None:
    for field in KPI_FIELDS:
        row[f"fast_{field}"] = row[field]
    row["fast_kernel_bytes"] = row["kernel_bytes"]
    row["fast_score_mode"] = row.get("score_mode", "fast")
    for field in ("rbcs", "fidelity_real", *KPI_FIELDS[2:], "kernel_bytes", "score_mode", "kernel_summary"):
        row[field] = full_row[field]
    row["full_kpi_delta"] = qfloat(float(row["rbcs"]) - float(row["fast_rbcs"]))


def _failed_candidate_record(candidate_index: int, config: dict[str, Any], exc: Exception) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_index": int(candidate_index),
        "config": dict(config),
        "kernel_bytes": 0,
        "rbcs": 0.0,
        "fidelity_real": 0.0,
        "score_mode": "failed",
        "error": str(exc),
        "kernel_summary": {
            "canonical_program": "",
            "config": dict(config),
            "kernel_bytes": 0,
            "description_bits": 0,
            "dependence": {"kind": config.get("dependence_kind")},
            "target": {"kind": config.get("target_kind")},
            "marginals": {"count": 0, "kinds": {}},
            "top_kpi_components": [],
        },
    }
    for field in KPI_FIELDS[2:]:
        row[field] = 0.0
    return row


def _event_candidate_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": row.get("config"),
        "score_mode": row.get("score_mode"),
        "rbcs": row.get("rbcs"),
        "fidelity_real": row.get("fidelity_real"),
        "transfer_utility": row.get("transfer_utility"),
        "feature_importance": row.get("feature_importance"),
        "distribution_copula": row.get("distribution_copula"),
        "residual_or_calibration": row.get("residual_or_calibration"),
        "prediction_agreement": row.get("prediction_agreement"),
        "kernel_bytes": row.get("kernel_bytes"),
        "full_kpi_delta": row.get("full_kpi_delta"),
        "proxy_rbcs": row.get("proxy_rbcs"),
        "proxy_vs_fast_delta": row.get("proxy_vs_fast_delta"),
        "proxy_vs_full_delta": row.get("proxy_vs_full_delta"),
        "kernel_summary": row.get("kernel_summary"),
    }


def _attach_proxy_quality(row: dict[str, Any], proxy_row: dict[str, Any], target: str) -> None:
    row["proxy_rbcs"] = proxy_row.get("rbcs")
    row["proxy_fidelity_real"] = proxy_row.get("fidelity_real")
    try:
        row[f"proxy_vs_{target}_delta"] = qfloat(float(row["rbcs"]) - float(proxy_row["rbcs"]))
    except (KeyError, TypeError, ValueError):
        return


def _kernel_summary(kernel: dict[str, Any], config: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "canonical_program": str(kernel.get("program", "")),
        "config": dict(config),
        "kernel_bytes": int(kernel.get("kernel_bytes", 0)),
        "description_bits": int(kernel.get("description_bits", 0)),
        "description_bits_per_cell": qfloat(float(kernel.get("description_bits_per_cell", 0.0))),
        "dependence": _dependence_summary(kernel.get("dependence", {})),
        "target": _target_summary(kernel.get("target", {})),
        "marginals": _marginal_summary(kernel.get("marginals", [])),
        "top_kpi_components": _top_kpi_components(metrics or {}),
    }


def _dependence_summary(dependence: dict[str, Any]) -> dict[str, Any]:
    params = dependence.get("params", {}) if isinstance(dependence, dict) else {}
    summary = {"kind": dependence.get("kind") if isinstance(dependence, dict) else None}
    if isinstance(params, dict) and "rank" in params:
        summary["rank"] = int(params["rank"])
    return summary


def _target_summary(target: dict[str, Any]) -> dict[str, Any]:
    params = target.get("params", {}) if isinstance(target, dict) else {}
    summary = {"kind": target.get("kind") if isinstance(target, dict) else None}
    if isinstance(params, dict):
        for key in ("coef", "thresholds", "leaves"):
            value = params.get(key)
            if isinstance(value, (list, tuple)):
                summary[f"{key}_count"] = len(value)
    return summary


def _marginal_summary(marginals: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = Counter(str(item.get("kind", "unknown")) for item in marginals if isinstance(item, dict))
    return {"count": len(marginals), "kinds": dict(sorted(kinds.items()))}


def _top_kpi_components(metrics: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    items = []
    for field in KPI_FIELDS:
        if field in metrics and isinstance(metrics[field], (int, float)):
            items.append({"name": field, "value": qfloat(float(metrics[field]))})
    return sorted(items, key=lambda item: item["value"], reverse=True)[:limit]


def _candidate_feature_dim() -> int:
    return len(DEPENDENCE_KINDS) + len(TARGET_KINDS) + len(EMPQ_KINDS) + 3


def _emit_event(callback: EventCallback | None, record: dict[str, Any]) -> None:
    if callback is not None:
        callback(record)


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _prepare_deadline(
    start_time: float,
    max_seconds: float | None,
    all_datasets: bool,
    prep_max_seconds: float | None = None,
) -> float | None:
    if max_seconds is None or float(max_seconds) <= 0:
        if prep_max_seconds is not None and float(prep_max_seconds) > 0:
            return start_time + float(prep_max_seconds)
        return None
    if prep_max_seconds is not None and float(prep_max_seconds) > 0:
        max_prep = float(prep_max_seconds)
        train_reserve = ALL_DATASETS_MIN_TRAIN_SECONDS if float(max_seconds) > ALL_DATASETS_MIN_TRAIN_SECONDS else 0.0
        return start_time + min(max_prep, max(1.0, float(max_seconds) - train_reserve))
    if not all_datasets:
        return start_time + float(max_seconds)
    max_seconds = float(max_seconds)
    if max_seconds <= ALL_DATASETS_MIN_TRAIN_SECONDS * 2:
        prep_seconds = max_seconds / 2.0
    else:
        prep_seconds = min(ALL_DATASETS_PREP_SECONDS_CAP, max(ALL_DATASETS_MIN_TRAIN_SECONDS, max_seconds * ALL_DATASETS_PREP_FRACTION))
        prep_seconds = min(prep_seconds, max_seconds - ALL_DATASETS_MIN_TRAIN_SECONDS)
    return start_time + max(1.0, prep_seconds)


def _build_smoke_score_cache(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int,
) -> SmokeScoreCache:
    real_score, real_pred, _ = _smoke_predict_score(X_train, y_train, X_test, y_test, task, seed)
    real_importance = _linear_importance(X_train, y_train, task, seed) if X_train.shape[1] > 1 else np.ones(X_train.shape[1], dtype=float)
    return SmokeScoreCache(real_score=float(real_score), real_pred=np.asarray(real_pred, dtype=float), real_importance=np.asarray(real_importance, dtype=float))


def _binary_diagnostics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cache: SmokeScoreCache,
) -> BinaryDiagnostics:
    del X_test, y_test
    prevalence = float(np.mean(y_train)) if len(y_train) else 0.0
    class_balance = 1.0 - min(1.0, abs(prevalence - 0.5) / 0.5)
    target_corr = _target_corr_summary(X_train, y_train)
    feature_corr = _feature_corr_summary(X_train)
    return BinaryDiagnostics(
        prevalence=qfloat(prevalence),
        class_balance=qfloat(class_balance),
        target_corr_mean=qfloat(target_corr[0]),
        target_corr_max=qfloat(target_corr[1]),
        feature_corr_mean=qfloat(feature_corr[0]),
        feature_corr_max=qfloat(feature_corr[1]),
        logistic_score=qfloat(cache.real_score),
        prediction_std=qfloat(float(np.std(cache.real_pred)) if len(cache.real_pred) else 0.0),
        rows=int(len(y_train)),
        cols=int(X_train.shape[1] + 1),
    )


def _binary_diagnostics_json(diagnostics: BinaryDiagnostics) -> dict[str, Any]:
    return {
        "prevalence": diagnostics.prevalence,
        "class_balance": diagnostics.class_balance,
        "target_corr_mean": diagnostics.target_corr_mean,
        "target_corr_max": diagnostics.target_corr_max,
        "feature_corr_mean": diagnostics.feature_corr_mean,
        "feature_corr_max": diagnostics.feature_corr_max,
        "logistic_score": diagnostics.logistic_score,
        "prediction_std": diagnostics.prediction_std,
        "rows": diagnostics.rows,
        "cols": diagnostics.cols,
    }


def _binary_proxy_candidate_row(
    candidate_index: int,
    config: dict[str, Any],
    diagnostics: BinaryDiagnostics,
    rows: int,
    cols: int,
) -> dict[str, Any]:
    report = _binary_proxy_report(config, diagnostics)
    kernel_bytes = _binary_proxy_kernel_bytes(config, cols)
    metric_row: dict[str, Any] = {
        "candidate_index": int(candidate_index),
        "config": dict(config),
        "kernel_bytes": int(kernel_bytes),
        "rbcs": rbcs(report["fidelity_real"], int(kernel_bytes), rows, cols),
        "fidelity_real": report["fidelity_real"],
        "score_mode": "proxy",
        "binary_proxy": _binary_diagnostics_json(diagnostics),
        **report["components"],
    }
    metric_row["kernel_summary"] = _kernel_summary(_binary_proxy_kernel(config, int(kernel_bytes), cols), config, metric_row)
    return metric_row


def _binary_proxy_report(config: dict[str, Any], diagnostics: BinaryDiagnostics) -> dict[str, Any]:
    corr = float(diagnostics.target_corr_max)
    mean_corr = float(diagnostics.target_corr_mean)
    feature_corr = float(diagnostics.feature_corr_mean)
    balance = float(diagnostics.class_balance)
    logistic = float(diagnostics.logistic_score)
    dep = str(config.get("dependence_kind"))
    target = str(config.get("target_kind"))

    dep_score = {
        "ind": 1.0 - min(1.0, feature_corr),
        "gauss_copula": 0.65 + 0.35 * min(1.0, max(feature_corr, float(diagnostics.feature_corr_max))),
        "auto": 0.7 + 0.3 * min(1.0, max(feature_corr, corr)),
    }.get(dep, 0.6)
    lift_score = 0.45 + 0.35 * logistic + 0.20 * corr
    tree_score = 0.45 + 0.25 * corr + 0.20 * balance + 0.10 * (1.0 - abs(logistic - 0.5) / 0.5)
    target_score = lift_score if target == "lift_logit" else tree_score

    components = {
        "transfer_utility": 0.45 * logistic + 0.30 * target_score + 0.25 * balance,
        "feature_importance": 0.55 * corr + 0.25 * mean_corr + 0.20 * target_score,
        "distribution_copula": 0.70 * dep_score + 0.30 * (1.0 - min(1.0, feature_corr)),
        "residual_or_calibration": 0.65 * balance + 0.35 * target_score,
        "prediction_agreement": 0.55 * logistic + 0.25 * target_score + 0.20 * min(1.0, float(diagnostics.prediction_std) * 4.0),
    }
    clipped = {key: qfloat(float(np.clip(value, 0.0, 1.0))) for key, value in components.items()}
    fidelity = sum(COMPONENT_WEIGHTS[key] * clipped[key] for key in COMPONENT_WEIGHTS)
    return {"fidelity_real": qfloat(fidelity), "components": clipped}


def _binary_proxy_kernel_bytes(config: dict[str, Any], cols: int) -> int:
    dep = str(config.get("dependence_kind"))
    target = str(config.get("target_kind"))
    p = max(1, int(cols) - 1)
    dep_bytes = {"ind": 640 + 20 * p, "gauss_copula": 980 + 36 * p, "auto": 1100 + 40 * p}.get(dep, 900 + 24 * p)
    target_bytes = {"lift_logit": 860 + 32 * p, "tree_piecewise": 1240 + 48 * p}.get(target, 900 + 32 * p)
    return int(dep_bytes + target_bytes)


def _binary_proxy_kernel(config: dict[str, Any], kernel_bytes: int, cols: int) -> dict[str, Any]:
    p = max(0, int(cols) - 1)
    description_bits = int(max(1, kernel_bytes * 5))
    return {
        "program": f"(proxy-decision task=binary dep={config.get('dependence_kind')} target={config.get('target_kind')})",
        "kernel_bytes": int(kernel_bytes),
        "description_bits": description_bits,
        "description_bits_per_cell": qfloat(description_bits / max(1, p + 1)),
        "dependence": {"kind": config.get("dependence_kind")},
        "target": {"kind": config.get("target_kind")},
        "marginals": [{"kind": "proxy"} for _ in range(p)],
    }


def _score_synthetic_dataset_smoke(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
    cache: SmokeScoreCache | None = None,
) -> dict[str, Any]:
    X_train = _clean(X_train)
    y_train = _clean_y(y_train, task)
    X_test = _clean(X_test)
    y_test = _clean_y(y_test, task)
    X_synth = _clean(X_synth)
    y_synth = _clean_y(y_synth, task)
    components = {
        "transfer_utility": _smoke_transfer_utility(
            X_train,
            y_train,
            X_test,
            y_test,
            X_synth,
            y_synth,
            task,
            seed,
            real_score=cache.real_score if cache is not None else None,
        ),
        "feature_importance": _smoke_feature_importance(
            X_train,
            y_train,
            X_synth,
            y_synth,
            task,
            seed,
            real_importance=cache.real_importance if cache is not None else None,
        ),
        "distribution_copula": _smoke_distribution_copula(X_train, y_train, X_synth, y_synth),
        "residual_or_calibration": _smoke_residual_or_calibration(X_train, y_train, X_synth, y_synth, task, seed),
        "prediction_agreement": _smoke_prediction_agreement(
            X_train,
            y_train,
            X_test,
            X_synth,
            y_synth,
            task,
            seed,
            real_pred=cache.real_pred if cache is not None else None,
        ),
    }
    fidelity = sum(COMPONENT_WEIGHTS[key] * components[key] for key in COMPONENT_WEIGHTS)
    return {
        "fidelity_real": qfloat(fidelity),
        "components": {key: qfloat(value) for key, value in components.items()},
    }


def _smoke_model(task: str, seed: int) -> Ridge | LogisticRegression:
    if task == "regression":
        return Ridge(alpha=1.0)
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=seed)


def _smoke_predict_score(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int,
) -> tuple[float, np.ndarray, Any]:
    model = _smoke_model(task, seed)
    if task == "binary" and len(np.unique(y_fit)) < 2:
        pred = np.full(len(y_test), float(np.mean(y_fit)))
        return _binary_score(y_test, pred), pred, None
    model.fit(X_fit, y_fit)
    if task == "regression":
        pred = _clean(model.predict(X_test))
        return _regression_score(y_test, pred), pred, model
    pred = model.predict_proba(X_test)[:, -1]
    return _binary_score(y_test, pred), pred, model


def _regression_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    scale = max(float(np.std(y_true)), 0.05)
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    return float(np.clip(1.0 - rmse / scale, 0.0, 1.0))


def _binary_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) >= 2:
        try:
            return float(np.clip(roc_auc_score(y_true, pred), 0.0, 1.0))
        except ValueError:
            pass
    return float(accuracy_score(y_true, pred >= 0.5))


def _smoke_transfer_utility(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
    real_score: float | None = None,
) -> float:
    if real_score is None:
        real_score, _, _ = _smoke_predict_score(X_train, y_train, X_test, y_test, task, seed)
    synth_score, _, _ = _smoke_predict_score(X_synth, y_synth, X_test, y_test, task, seed)
    return float(np.clip(1.0 - min(1.0, abs(real_score - synth_score) / max(abs(real_score), 0.1)), 0.0, 1.0))


def _smoke_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
    real_importance: np.ndarray | None = None,
) -> float:
    if X_train.shape[1] <= 1:
        return 1.0
    real = real_importance if real_importance is not None else _linear_importance(X_train, y_train, task, seed)
    synth = _linear_importance(X_synth, y_synth, task, seed)
    corr = stats.spearmanr(real, synth).correlation
    corr_score = 0.5 if not np.isfinite(corr) else (float(corr) + 1.0) / 2.0
    k = max(1, min(5, X_train.shape[1] // 3 if X_train.shape[1] >= 3 else 1))
    overlap = len(set(np.argsort(-real)[:k]) & set(np.argsort(-synth)[:k])) / k
    return float(np.clip(0.65 * corr_score + 0.35 * overlap, 0.0, 1.0))


def _linear_importance(X: np.ndarray, y: np.ndarray, task: str, seed: int) -> np.ndarray:
    if task == "binary" and len(np.unique(y)) < 2:
        return np.zeros(X.shape[1], dtype=float)
    model = _smoke_model(task, seed)
    model.fit(X, y)
    coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1])), dtype=float).ravel()
    imp = np.abs(coef)
    total = float(np.sum(imp))
    return imp / total if total > 1e-9 else imp


def _smoke_distribution_copula(X_train: np.ndarray, y_train: np.ndarray, X_synth: np.ndarray, y_synth: np.ndarray) -> float:
    real = np.column_stack([X_train, y_train])
    synth = np.column_stack([X_synth, y_synth])
    p = real.shape[1]
    marginal = []
    for j in range(p):
        ks = 1.0 - float(stats.ks_2samp(real[:, j], synth[:, j]).statistic)
        wd = 1.0 - min(1.0, float(stats.wasserstein_distance(real[:, j], synth[:, j])) / 0.25)
        marginal.append(0.6 * ks + 0.4 * wd)
    corr = _feature_corr_similarity(real, synth)
    return float(np.clip(0.65 * float(np.mean(marginal)) + 0.35 * corr, 0.0, 1.0))


def _feature_corr_similarity(real: np.ndarray, synth: np.ndarray) -> float:
    if real.shape[1] <= 1:
        return 1.0
    real_keep = np.std(real, axis=0) > 1e-9
    synth_keep = np.std(synth, axis=0) > 1e-9
    keep = real_keep & synth_keep
    if int(np.sum(keep)) <= 1:
        return 1.0
    cr = np.nan_to_num(np.corrcoef(real[:, keep], rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    cs = np.nan_to_num(np.corrcoef(synth[:, keep], rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.clip(1.0 - float(np.mean(np.abs(cr - cs))) / 1.5, 0.0, 1.0))


def _smoke_residual_or_calibration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
) -> float:
    if task == "binary":
        prevalence = 1.0 - min(1.0, abs(float(np.mean(y_train)) - float(np.mean(y_synth))) / 0.5)
        return float(np.clip(prevalence, 0.0, 1.0))
    real_resid = _smoke_residuals(X_train, y_train, seed)
    synth_resid = _smoke_residuals(X_synth, y_synth, seed)
    q_real = np.quantile(real_resid, [0.1, 0.25, 0.5, 0.75, 0.9])
    q_synth = np.quantile(synth_resid, [0.1, 0.25, 0.5, 0.75, 0.9])
    std_diff = abs(float(np.std(real_resid)) - float(np.std(synth_resid))) / max(float(np.std(real_resid)), 0.05)
    q_diff = float(np.mean(np.abs(q_real - q_synth))) / 0.25
    return float(np.clip(1.0 - min(1.0, 0.5 * std_diff + 0.5 * q_diff), 0.0, 1.0))


def _smoke_residuals(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    _, pred, _ = _smoke_predict_score(X, y, X, y, "regression", seed)
    return y - pred


def _smoke_prediction_agreement(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
    real_pred: np.ndarray | None = None,
) -> float:
    dummy = np.zeros(len(X_test), dtype=float)
    if real_pred is None:
        _, real_pred, _ = _smoke_predict_score(X_train, y_train, X_test, dummy, task, seed)
    _, synth_pred, _ = _smoke_predict_score(X_synth, y_synth, X_test, dummy, task, seed)
    diff = float(np.mean(np.abs(real_pred - synth_pred)))
    return float(np.clip(1.0 - min(1.0, diff / 0.5), 0.0, 1.0))


def _clean(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(clip01(values), nan=0.0, posinf=1.0, neginf=0.0)


def _clean_y(values: np.ndarray, task: str) -> np.ndarray:
    y = _clean(values)
    return (y >= 0.5).astype(float) if task == "binary" else y


def _effective_sample_cap(source_y: np.ndarray, requested_cap: int, task: str, split: str) -> int:
    cap = int(requested_cap)
    if cap <= 0 or len(source_y) <= cap:
        return cap
    floor = 64 if split == "train" else 32
    cap = min(len(source_y), max(cap, floor))
    if task == "binary" and len(np.unique(source_y)) == 2:
        minority = int(min(np.sum(source_y < 0.5), np.sum(source_y >= 0.5)))
        if minority > 0:
            cap = min(len(source_y), max(cap, min(len(source_y), minority * 2)))
    return int(cap)


def _cap_xy(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    task: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_rows = int(max_rows)
    if max_rows <= 0 or len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    if task == "binary" and len(np.unique(y)) == 2 and max_rows >= 2:
        zeros = np.flatnonzero(y < 0.5)
        ones = np.flatnonzero(y >= 0.5)
        one_count = max(1, int(round(max_rows * len(ones) / len(y))))
        zero_count = max_rows - one_count
        zero_count = min(len(zeros), max(1, zero_count))
        one_count = min(len(ones), max(1, max_rows - zero_count))
        idx = np.concatenate(
            [
                rng.choice(zeros, size=zero_count, replace=False),
                rng.choice(ones, size=one_count, replace=False),
            ]
        )
    else:
        idx = rng.choice(np.arange(len(y)), size=max_rows, replace=False)
    if task == "binary" and len(np.unique(y)) == 2 and len(np.unique(y[idx])) < 2:
        missing_value = 1.0 if np.all(y[idx] < 0.5) else 0.0
        replacement_pool = np.flatnonzero(y >= 0.5) if missing_value >= 0.5 else np.flatnonzero(y < 0.5)
        if len(replacement_pool):
            idx[int(rng.integers(0, len(idx)))] = int(rng.choice(replacement_pool))
    idx = np.sort(idx)
    return X[idx], y[idx]


def _repair_binary_targets(y: np.ndarray, seed: int) -> np.ndarray:
    repaired = _clean_y(y, "binary")
    if len(repaired) >= 2 and len(np.unique(repaired)) < 2:
        rng = np.random.default_rng(seed)
        repaired = repaired.copy()
        repaired[int(rng.integers(0, len(repaired)))] = 1.0 - float(repaired[0])
    return repaired


def _warn_binary_sample_quality(
    event_callback: EventCallback | None,
    spec: DatasetSpec,
    dataset_index: int,
    split: str,
    source_y: np.ndarray,
    sampled_y: np.ndarray,
) -> None:
    if spec.task != "binary":
        return
    source_classes = len(np.unique(source_y))
    sampled_classes = len(np.unique(sampled_y))
    if source_classes < 2:
        message = f"binary {split} source for {spec.dataset_id} has one class after cleaning"
    elif sampled_classes < 2:
        message = f"binary {split} sample for {spec.dataset_id} has one class; increase row_cap"
    else:
        return
    _emit_event(
        event_callback,
        {
            "type": "warning",
            "dataset_index": int(dataset_index),
            "dataset": _spec_json(spec),
            "split": split,
            "message": message,
            "source_classes": int(source_classes),
            "sampled_classes": int(sampled_classes),
        },
    )


def _label_quality_report(
    spec: DatasetSpec,
    y_train_full: np.ndarray,
    y_test_full: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    requested_row_cap: int,
    effective_train_cap: int,
    effective_test_cap: int,
    records: list[dict[str, Any]],
    sample_seed: int,
    binary_label_mode: str,
) -> dict[str, Any]:
    score_modes = Counter(str(record.get("score_mode", "unknown")) for record in records)
    full_kpi_count = sum(1 for record in records if isinstance(record.get("full_kpi_delta"), (int, float)))
    fast_count = sum(1 for record in records if str(record.get("score_mode")) in {"fast", "full"})
    proxy_count = int(score_modes.get("proxy", 0))
    report: dict[str, Any] = {
        "source": "real",
        "scorer_version": SCORER_VERSION,
        "sample_seed": int(sample_seed),
        "label_mode": binary_label_mode if spec.task == "binary" else "all-fast",
        "requested_row_cap": int(requested_row_cap),
        "effective_row_cap": {"train": int(effective_train_cap), "test": int(effective_test_cap)},
        "source_rows": {"train": int(len(y_train_full)), "test": int(len(y_test_full))},
        "sampled_rows": {"train": int(len(y_train)), "test": int(len(y_test))},
        "candidate_count": int(len(records)),
        "score_modes": dict(sorted(score_modes.items())),
        "fast_label_count": int(fast_count),
        "proxy_label_count": int(proxy_count),
        "proxy_label_share": qfloat(proxy_count / max(1, len(records))),
        "full_kpi_count": int(full_kpi_count),
        "full_kpi_audited": bool(full_kpi_count > 0),
        "full_vs_fast_kpi_delta": _delta_summary(record.get("full_kpi_delta") for record in records),
        "proxy_vs_fast_delta": _delta_summary(record.get("proxy_vs_fast_delta") for record in records),
        "proxy_vs_full_delta": _delta_summary(record.get("proxy_vs_full_delta") for record in records),
    }
    if spec.task == "binary":
        report["source_classes"] = {"train": int(len(np.unique(y_train_full))), "test": int(len(np.unique(y_test_full)))}
        report["sampled_classes"] = {"train": int(len(np.unique(y_train))), "test": int(len(np.unique(y_test)))}
        report["class_balance"] = {"train": qfloat(_binary_class_balance(y_train)), "test": qfloat(_binary_class_balance(y_test))}
        report["prevalence"] = {
            "train": qfloat(float(np.mean(y_train)) if len(y_train) else 0.0),
            "test": qfloat(float(np.mean(y_test)) if len(y_test) else 0.0),
        }
    return report


def _delta_summary(values: Any) -> dict[str, Any]:
    arr = np.asarray([float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))], dtype=float)
    if len(arr) == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(len(arr)),
        "mean": qfloat(float(np.mean(arr))),
        "min": qfloat(float(np.min(arr))),
        "max": qfloat(float(np.max(arr))),
    }


def _adaptive_relabel_row_cap(
    row_cap: int,
    y_train_full: np.ndarray,
    y_test_full: np.ndarray,
    task: str,
    records: list[dict[str, Any]],
    margin: float = 0.01,
) -> int | None:
    if len(records) < 2:
        return None
    scored = sorted((float(record.get("rbcs") or 0.0) for record in records), reverse=True)
    if scored[0] - scored[1] > float(margin):
        return None
    current = int(row_cap)
    source_rows = min(int(len(y_train_full)), int(len(y_test_full) * 2) if len(y_test_full) else int(len(y_train_full)))
    for candidate_cap in (512, 1024):
        if current < candidate_cap and source_rows > current:
            return min(candidate_cap, max(current + 1, source_rows))
    if task == "binary" and current < len(y_train_full):
        return min(len(y_train_full), max(current + 1, 512))
    return None


def _binary_class_balance(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    prevalence = float(np.mean(y))
    return 1.0 - min(1.0, abs(prevalence - 0.5) / 0.5)


def _augmentation_kernel_from_assets(rows: list[dict[str, Any]], candidate_assets: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    choices: list[tuple[float, dict[str, Any]]] = []
    for row, asset in zip(rows, candidate_assets):
        if asset is None or not isinstance(asset.get("kernel"), dict):
            continue
        choices.append((float(row.get("rbcs") or 0.0), asset["kernel"]))
    if not choices:
        return None
    return max(choices, key=lambda item: item[0])[1]


def _dataset_feature_vector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
) -> np.ndarray:
    X = _clean(X_train)
    y = _clean_y(y_train, task)
    n = max(1, X.shape[0])
    p = X.shape[1]
    means = X.mean(axis=0) if p else np.array([0.0])
    stds = X.std(axis=0) if p else np.array([0.0])
    unique_fracs = np.array([len(np.unique(np.round(X[:, j], 6))) / n for j in range(p)]) if p else np.array([0.0])
    target_entropy = _target_entropy(y, task)
    x_corr = _feature_corr_summary(X)
    y_corr = _target_corr_summary(X, y)
    shift = _mean_abs_shift(X_train, X_test, y_train, y_test)
    values = [
        1.0 if task == "regression" else 0.0,
        1.0 if task == "binary" else 0.0,
        min(1.0, math.log2(n + 1.0) / 12.0),
        min(1.0, math.log2(p + 1.0) / 10.0),
        float(np.mean(y)),
        float(np.std(y)),
        target_entropy,
        float(np.mean(means)),
        float(np.std(means)),
        float(np.mean(stds)),
        float(np.std(stds)),
        float(np.mean(X <= 1e-9)) if p else 0.0,
        float(np.mean(X >= 1.0 - 1e-9)) if p else 0.0,
        float(np.mean(unique_fracs)),
        x_corr[0],
        x_corr[1],
        y_corr[0],
        y_corr[1],
        shift,
    ]
    return np.asarray([qfloat(v) for v in values], dtype=np.float32)


def _target_entropy(y: np.ndarray, task: str) -> float:
    if task == "binary":
        p = float(np.clip(np.mean(y), 1e-9, 1.0 - 1e-9))
        return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))
    hist, _ = np.histogram(y, bins=8, range=(0.0, 1.0))
    prob = hist.astype(float) / max(1.0, float(hist.sum()))
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)) / 3.0) if len(prob) else 0.0


def _feature_corr_summary(X: np.ndarray) -> tuple[float, float]:
    if X.shape[1] <= 1 or X.shape[0] <= 2:
        return 0.0, 0.0
    keep = np.std(X, axis=0) > 1e-9
    if int(np.sum(keep)) <= 1:
        return 0.0, 0.0
    corr = np.nan_to_num(np.corrcoef(X[:, keep], rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])
    return float(np.mean(vals)), float(np.max(vals)) if len(vals) else 0.0


def _target_corr_summary(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if X.shape[1] == 0 or X.shape[0] <= 2 or np.std(y) <= 1e-9:
        return 0.0, 0.0
    vals = []
    for j in range(X.shape[1]):
        if np.std(X[:, j]) <= 1e-9:
            vals.append(0.0)
        else:
            vals.append(abs(float(np.corrcoef(X[:, j], y)[0, 1])))
    vals = np.nan_to_num(np.asarray(vals), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(vals)), float(np.max(vals)) if len(vals) else 0.0


def _mean_abs_shift(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    if len(X_test) == 0:
        return 0.0
    train = np.column_stack([_clean(X_train), _clean(y_train)])
    test = np.column_stack([_clean(X_test), _clean(y_test)])
    return float(np.clip(np.mean(np.abs(train.mean(axis=0) - test.mean(axis=0))) / 0.5, 0.0, 1.0))


def _candidate_feature_vector(config: dict[str, Any], kernel: dict[str, Any], rows: int, cols: int) -> np.ndarray:
    dep = str(config["dependence_kind"])
    target = str(config["target_kind"])
    empq_k = config.get("empq_k")
    kernel_bytes = int(kernel.get("kernel_bytes", 0))
    description_bits = int(kernel.get("description_bits", 0))
    bytes_per_cell = kernel_bytes / max(1.0, float(rows * cols))
    values: list[float] = []
    values.extend(1.0 if dep == item else 0.0 for item in DEPENDENCE_KINDS)
    values.extend(1.0 if target == item else 0.0 for item in TARGET_KINDS)
    values.extend(1.0 if empq_k == item else 0.0 for item in EMPQ_KINDS)
    values.extend(
        [
            min(1.0, math.log2(max(1.0, float(kernel_bytes))) / 20.0),
            min(1.0, math.log2(max(1.0, bytes_per_cell)) / 8.0),
            min(1.0, math.log2(max(1.0, float(description_bits))) / 18.0),
        ]
    )
    return np.asarray(values, dtype=np.float32)


class FrozenFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, seed: int = SEED) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        generator = torch.Generator()
        generator.manual_seed(seed)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.proj.weight, generator=generator)
            nn.init.zeros_(self.proj.bias)
            self.norm.weight.fill_(1.0)
            self.norm.bias.zero_()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.norm(self.proj(features)))


class KernelTuner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, kpi_dim: int = len(KPI_FIELDS), seed: int = SEED) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = FrozenFeatureEncoder(input_dim, hidden_dim=hidden_dim, seed=seed)
        self.ranking_head = _head(hidden_dim, 1)
        self.kpi_head = _head(hidden_dim, kpi_dim)
        self.loss_head = _head(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = features.shape[:-1]
        flat = features.reshape(-1, features.shape[-1])
        encoded = self.encoder(flat)
        return {
            "ranking_logits": self.ranking_head(encoded).reshape(*shape),
            "kpi": self.kpi_head(encoded).reshape(*shape, len(KPI_FIELDS)),
            "candidate_loss": self.loss_head(encoded).reshape(*shape),
        }

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for module in (self.ranking_head, self.kpi_head, self.loss_head):
            yield from module.parameters()


def _head(hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, out_dim))


def tuner_batch_loss(
    model: KernelTuner,
    features: torch.Tensor,
    kpis: torch.Tensor,
    best_index: torch.Tensor,
    example_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    outputs = model(features)
    logits = outputs["ranking_logits"]
    probs = F.softmax(logits, dim=1)
    rbcs = kpis[:, :, 0]
    best_rbcs = kpis[:, :, 0].amax(dim=1, keepdim=True)
    loss_target = torch.clamp(best_rbcs - rbcs, min=0.0, max=1.0)
    weights = _normalized_example_weight(example_weight, features.shape[0], features.device)
    ranking_loss = _weighted_mean(torch.sum(probs * loss_target, dim=1), weights)
    kpi_per_example = F.mse_loss(torch.sigmoid(outputs["kpi"]), kpis, reduction="none").mean(dim=(1, 2))
    kpi_loss = _weighted_mean(kpi_per_example, weights)
    agg_per_example = F.mse_loss(torch.sigmoid(outputs["candidate_loss"]), loss_target, reduction="none").mean(dim=1)
    agg_loss = _weighted_mean(agg_per_example, weights)
    logit_penalty = torch.mean(logits.square())
    hard_ranking_nll = _weighted_mean(F.cross_entropy(logits, best_index, reduction="none"), weights)
    total = ranking_loss + 0.35 * kpi_loss + 0.15 * agg_loss + 0.001 * logit_penalty
    return {
        "loss": total,
        "ranking_loss": ranking_loss.detach(),
        "hard_ranking_nll": hard_ranking_nll.detach(),
        "kpi_loss": kpi_loss.detach(),
        "aggregation_loss": agg_loss.detach(),
        "logit_penalty": logit_penalty.detach(),
    }


def _normalized_example_weight(weight: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor | None:
    if weight is None:
        return None
    weights = torch.as_tensor(weight, dtype=torch.float32, device=device).reshape(-1)
    if int(weights.numel()) != int(batch_size):
        raise ValueError("example_weight must have one value per example")
    return torch.clamp(weights, min=0.0)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    denom = torch.clamp(weights.sum(), min=1e-9)
    return torch.sum(values * weights) / denom


class MetricsSink:
    def __init__(self, out_dir: str | Path, resume: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.events_path = self.out_dir / "events.jsonl"
        self.summary_path = self.out_dir / "summary.json"
        self.report_path = self.out_dir / "report.html"
        self.trust_report_path = self.out_dir / "trust_report.json"
        self.metrics: list[dict[str, Any]] = _read_jsonl(self.metrics_path) if resume else []
        self.events: list[dict[str, Any]] = _read_jsonl(self.events_path) if resume else []
        if not resume:
            self.metrics_path.write_text("", encoding="utf-8")
            self.events_path.write_text("", encoding="utf-8")
        else:
            self.metrics_path.touch(exist_ok=True)
            self.events_path.touch(exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        self.metrics.append(record)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_dumps(record) + "\n")

    def write_event(self, record: dict[str, Any]) -> None:
        self.events.append(record)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_dumps(record) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(canonical_dumps(summary) + "\n", encoding="utf-8")

    def write_report(self, summary: dict[str, Any]) -> None:
        self.report_path.write_text(render_report_html(summary, self.metrics, self.events), encoding="utf-8")

    def write_trust_report(self, report: dict[str, Any]) -> None:
        self.trust_report_path.write_text(canonical_dumps(report) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _validate_binary_label_mode(value: str) -> None:
    if value not in BINARY_LABEL_MODES:
        choices = ", ".join(BINARY_LABEL_MODES)
        raise ValueError(f"binary_label_mode must be one of: {choices}")


def _validate_synthetic_label_mode(value: str) -> None:
    if value not in SYNTHETIC_LABEL_MODES:
        choices = ", ".join(SYNTHETIC_LABEL_MODES)
        raise ValueError(f"synthetic_label_mode must be one of: {choices}")


def _spec_identity(spec: DatasetSpec) -> str:
    return f"{spec.task}:{spec.path.resolve()}:{spec.dataset_id}"


def _example_cache_key(
    spec: DatasetSpec,
    row_cap: int,
    seed: int,
    binary_label_mode: str,
    full_kpi: bool,
    full_kpi_top_k: int,
) -> str:
    payload = {
        "scorer_version": SCORER_VERSION,
        "dataset": _spec_json(spec),
        "dataset_signature": _dataset_signature(spec),
        "seed": int(seed),
        "row_cap": int(row_cap),
        "candidate_space": candidate_configs(spec.task),
        "label_mode": binary_label_mode if spec.task == "binary" else "all-fast",
        "full_kpi": bool(full_kpi),
        "full_kpi_top_k": int(full_kpi_top_k),
    }
    digest = hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()
    return f"v{SCORER_VERSION}-{digest}"


def _dataset_signature(spec: DatasetSpec) -> dict[str, Any]:
    files = []
    for name in ("train.csv", "test.csv", "meta.json"):
        path = spec.path / name
        if not path.exists():
            files.append({"name": name, "exists": False})
            continue
        stat = path.stat()
        files.append(
            {
                "name": name,
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return {"path": str(spec.path.resolve()), "files": files}


def _example_cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    root = Path(cache_dir)
    return root / cache_key[:8] / f"{cache_key}.pt"


def _load_cached_example(path: Path, cache_key: str) -> TunerExample | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("cache_key") != cache_key:
        return None
    spec_payload = payload.get("spec")
    if not isinstance(spec_payload, dict):
        return None
    try:
        spec = DatasetSpec(
            path=Path(str(spec_payload["path"])),
            task=normalize_task(str(spec_payload["task"])),
            dataset_id=str(spec_payload["dataset_id"]),
        )
        features = np.asarray(payload["features"], dtype=np.float32)
        kpis = np.asarray(payload["kpis"], dtype=np.float32)
        records = tuple(dict(row) for row in payload.get("records", ()))
    except (KeyError, TypeError, ValueError):
        return None
    return TunerExample(
        spec=spec,
        features=features,
        kpis=kpis,
        best_index=int(payload.get("best_index", 0)),
        records=records,
        source=str(payload.get("source", "real")),
        anchor_dataset_id=payload.get("anchor_dataset_id"),
        label_quality=dict(payload.get("label_quality") or {}),
        augmentation_kernel=payload.get("augmentation_kernel") if isinstance(payload.get("augmentation_kernel"), dict) else None,
        sample_rows=dict(payload.get("sample_rows") or {}),
        label_weight=float(payload.get("label_weight", REAL_LABEL_WEIGHT)),
    )


def _write_cached_example(path: Path, cache_key: str, example: TunerExample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "dope-kernel-tuner-example-cache",
        "version": 1,
        "cache_key": cache_key,
        "scorer_version": SCORER_VERSION,
        "spec": _spec_json(example.spec),
        "features": example.features,
        "kpis": example.kpis,
        "best_index": int(example.best_index),
        "records": [dict(row) for row in example.records],
        "source": example.source,
        "anchor_dataset_id": example.anchor_dataset_id,
        "label_quality": dict(example.label_quality),
        "augmentation_kernel": example.augmentation_kernel,
        "sample_rows": dict(example.sample_rows),
        "label_weight": float(example.label_weight),
    }
    torch.save(payload, path)


def train_tuner(
    corpus: str | Path,
    tasks: str | list[str] | tuple[str, ...] = "regression,binary",
    max_datasets: int | None = 8,
    all_datasets: bool = False,
    epochs: int = 3,
    batch_size: int = 4,
    device: str = "cuda",
    out: str | Path = "runs/kernel_tuner_smoke",
    tui: bool = False,
    seed: int = SEED,
    row_cap: int = 64,
    max_seconds: float | None = None,
    prep_max_seconds: float | None = None,
    target_prepared_datasets: int | None = None,
    checkpoint_every: float = 300.0,
    full_kpi_every: int = 0,
    full_kpi_top_k: int = 3,
    full_kpi_audit_count: int = 0,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    report_html: bool = True,
    resume: bool = False,
    preset: str = "smoke",
    synthetic_mode: str = "real-first",
    synthetic_label_mode: str = "anchor-consistency",
    synthetic_fast_relabel_limit: int = 1,
    binary_label_mode: str = "shortlist",
    validation_binary_label_mode: str = "all-fast",
    example_cache_dir: str | Path | None = None,
    adaptive_relabel: bool = False,
) -> dict[str, Any]:
    start_time = time.monotonic()
    deadline = start_time + float(max_seconds) if max_seconds is not None and float(max_seconds) > 0 else None
    prep_deadline = _prepare_deadline(start_time, max_seconds, all_datasets, prep_max_seconds=prep_max_seconds)
    _seed_everything(seed)
    resolved_device = _resolve_device(device)
    real_batch_fraction, synthetic_batch_fraction = _training_mix_fractions(synthetic_mode)
    _validate_synthetic_label_mode(synthetic_label_mode)
    _validate_binary_label_mode(binary_label_mode)
    _validate_binary_label_mode(validation_binary_label_mode)
    sink = MetricsSink(out, resume=resume)
    previous_run_identity = _read_previous_trust_identity(sink.trust_report_path)
    if example_cache_dir is None:
        example_cache_dir = Path(out) / "example_cache"
    run_info = _run_info(resolved_device, corpus, tasks, max_datasets, row_cap, out, all_datasets=all_datasets)
    config = {
        "preset": preset,
        "seed": int(seed),
        "all_datasets": bool(all_datasets),
        "max_datasets": None if max_datasets is None else int(max_datasets),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "row_cap": int(row_cap),
        "max_seconds": None if max_seconds is None else float(max_seconds),
        "requested_prep_max_seconds": None if prep_max_seconds is None else float(prep_max_seconds),
        "target_prepared_datasets": None if target_prepared_datasets is None else int(target_prepared_datasets),
        "checkpoint_every": float(checkpoint_every),
        "full_kpi_every": int(full_kpi_every),
        "full_kpi_top_k": int(full_kpi_top_k),
        "full_kpi_audit_count": int(full_kpi_audit_count),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "report_html": bool(report_html),
        "resume": bool(resume),
        "synthetic_mode": synthetic_mode,
        "synthetic_label_mode": synthetic_label_mode,
        "synthetic_fast_relabel_limit": int(synthetic_fast_relabel_limit),
        "binary_label_mode": binary_label_mode,
        "validation_binary_label_mode": validation_binary_label_mode,
        "example_cache_dir": str(example_cache_dir),
        "adaptive_relabel": bool(adaptive_relabel),
        "real_batch_fraction": qfloat(real_batch_fraction),
        "synthetic_batch_fraction": qfloat(synthetic_batch_fraction),
        "prep_max_seconds": qfloat(prep_deadline - start_time) if prep_deadline is not None else None,
    }
    dashboard = TunerDashboard(run_info | {"config": config})
    live = None

    def emit(record: dict[str, Any]) -> None:
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": qfloat(time.monotonic() - start_time),
            **record,
        }
        sink.write_event(event)
        dashboard.apply(event)
        if live is not None:
            live.update(dashboard.render())

    emit({"type": "run_start", **run_info, "config": config})
    if synthetic_mode == "synthetic-heavy":
        emit(
            {
                "type": "warning",
                "message": "synthetic-heavy is experimental; best checkpoints and trust decisions use real validation only",
            }
        )
    last_record: dict[str, Any] | None = None
    stopped_by = "epochs"
    step = 0
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    checkpoint_dir = Path(out)
    best_checkpoint_path = checkpoint_dir / "best_tuner.pt"
    latest_checkpoint_path = checkpoint_dir / "latest_tuner.pt"

    with _live_dashboard(tui, dashboard) as live:
        emit(
            {
                "type": "dataset_selection_start",
                "corpus": str(corpus),
                "tasks": parse_tasks(tasks),
                "max_datasets": None if max_datasets is None else int(max_datasets),
                "all_datasets": bool(all_datasets),
                "synthetic_mode": synthetic_mode,
            }
        )
        if all_datasets:
            specs = select_all_supported_datasets(corpus, tasks=tasks, seed=seed)
        else:
            if max_datasets is None:
                raise ValueError("max_datasets is required unless all_datasets is enabled")
            specs = select_smoke_datasets(corpus, tasks=tasks, max_datasets=int(max_datasets), seed=seed)
        selected_before_target = len(specs)
        specs = _limit_specs_for_target(specs, target_prepared_datasets)
        train_specs, val_specs = _train_val_spec_split(specs, seed)
        val_spec_keys = {_spec_identity(spec) for spec in val_specs}
        full_kpi_keys = _full_kpi_audit_keys(val_specs, int(full_kpi_audit_count))
        label_modes_by_dataset = {key: validation_binary_label_mode for key in val_spec_keys}
        prep_specs = _interleave_train_val_specs(train_specs, val_specs)
        emit(
            {
                "type": "dataset_selection_complete",
                "selected_datasets": [_spec_json(spec) for spec in specs],
                "selected_before_target": int(selected_before_target),
                "target_prepared_datasets": None if target_prepared_datasets is None else int(target_prepared_datasets),
                "train_datasets": [_spec_json(spec) for spec in train_specs],
                "validation_datasets": [_spec_json(spec) for spec in val_specs],
                "full_kpi_audit_datasets": [_spec_json(spec) for spec in val_specs if _spec_identity(spec) in full_kpi_keys],
            }
        )
        examples = prepare_tuner_examples(
            prep_specs,
            row_cap=row_cap,
            seed=seed,
            event_callback=emit,
            full_kpi_every=full_kpi_every,
            full_kpi_top_k=full_kpi_top_k,
            deadline=prep_deadline,
            binary_label_mode=binary_label_mode,
            label_modes_by_dataset=label_modes_by_dataset,
            full_kpi_dataset_keys=full_kpi_keys,
            example_cache_dir=example_cache_dir,
            adaptive_relabel=adaptive_relabel,
        )
        if not examples:
            raise RuntimeError("no tuner examples were prepared before the time limit")

        input_dim = int(examples[0].features.shape[1])
        model = KernelTuner(input_dim=input_dim, seed=seed).to(resolved_device)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3, weight_decay=1e-4)
        train_examples = [example for example in examples if _spec_identity(example.spec) not in val_spec_keys]
        val_examples = [example for example in examples if _spec_identity(example.spec) in val_spec_keys]
        if not val_examples and len(examples) > 1:
            train_examples, val_examples = _train_val_split(examples, seed)
        if not train_examples:
            train_examples = examples
        best_record = _best_candidate_record(examples)

        if resume:
            resume_payload = _load_resume_checkpoint(latest_checkpoint_path, resolved_device)
            if resume_payload is None:
                emit({"type": "warning", "message": f"resume requested but {latest_checkpoint_path} does not exist"})
            else:
                if int(resume_payload.get("input_dim", -1)) != input_dim:
                    raise ValueError(f"resume checkpoint input_dim does not match current run ({resume_payload.get('input_dim')} != {input_dim})")
                model.load_state_dict(resume_payload["model_state_dict"])
                start_epoch = int(resume_payload.get("epoch", 0)) + 1
                step = int(resume_payload.get("step", 0))
                best_val_loss = float(resume_payload.get("best_val_loss", best_val_loss))
                best_epoch = int(resume_payload.get("best_epoch", best_epoch))
                epochs_since_best = int(resume_payload.get("epochs_since_best", 0))
                emit(
                    {
                        "type": "resume_loaded",
                        "path": str(latest_checkpoint_path),
                        "start_epoch": start_epoch,
                        "step": step,
                        "best_val_loss": qfloat(best_val_loss) if math.isfinite(best_val_loss) else None,
                        "best_epoch": int(best_epoch),
                        "epochs_since_best": int(epochs_since_best),
                    }
                )

        last_checkpoint_time = time.monotonic()
        for epoch in range(start_epoch, int(epochs) + 1):
            if _deadline_reached(deadline):
                stopped_by = "max_seconds"
                emit({"type": "time_limit_reached", "phase": "training", "epoch": epoch, "step": step})
                break
            model.train()
            losses = []
            epoch_real_examples = 0
            epoch_synthetic_examples = 0
            emit({"type": "epoch_start", "epoch": epoch, "step": step})
            for batch in _training_batches(
                train_examples,
                int(batch_size),
                synthetic_mode,
                seed + epoch,
                synthetic_label_mode=synthetic_label_mode,
                synthetic_fast_relabel_limit=synthetic_fast_relabel_limit,
            ):
                if _deadline_reached(deadline):
                    stopped_by = "max_seconds"
                    emit({"type": "time_limit_reached", "phase": "training_step", "epoch": epoch, "step": step})
                    break
                step += 1
                real_batch_size = sum(1 for example in batch if example.source == "real")
                synthetic_batch_size = len(batch) - real_batch_size
                epoch_real_examples += real_batch_size
                epoch_synthetic_examples += synthetic_batch_size
                emit(
                    {
                        "type": "train_step_start",
                        "epoch": epoch,
                        "step": step,
                        "batch_size": len(batch),
                        "real_batch_size": int(real_batch_size),
                        "synthetic_batch_size": int(synthetic_batch_size),
                        "synthetic_mode": synthetic_mode,
                        "synthetic_label_mode": synthetic_label_mode,
                    }
                )
                features, kpis, best_index = _batch_tensors(batch, resolved_device)
                weights = _batch_weights(batch, resolved_device)
                optimizer.zero_grad(set_to_none=True)
                loss_parts = tuner_batch_loss(model, features, kpis, best_index, example_weight=weights)
                loss_parts["loss"].backward()
                optimizer.step()
                loss_value = float(loss_parts["loss"].detach().cpu())
                losses.append(loss_value)
                emit(
                    {
                        "type": "train_step",
                        "epoch": epoch,
                        "step": step,
                        "loss": qfloat(loss_value),
                        "ranking_loss": qfloat(float(loss_parts["ranking_loss"].cpu())),
                        "hard_ranking_nll": qfloat(float(loss_parts["hard_ranking_nll"].cpu())),
                        "kpi_loss": qfloat(float(loss_parts["kpi_loss"].cpu())),
                        "aggregation_loss": qfloat(float(loss_parts["aggregation_loss"].cpu())),
                        "logit_penalty": qfloat(float(loss_parts["logit_penalty"].cpu())),
                        "real_batch_size": int(real_batch_size),
                        "synthetic_batch_size": int(synthetic_batch_size),
                        "synthetic_mode": synthetic_mode,
                        "synthetic_label_mode": synthetic_label_mode,
                        "mean_label_weight": qfloat(float(weights.mean().detach().cpu())),
                    }
                )
                if float(checkpoint_every) > 0 and time.monotonic() - last_checkpoint_time >= float(checkpoint_every):
                    _save_tuner_checkpoint(
                        latest_checkpoint_path,
                        model,
                        input_dim,
                        examples,
                        tasks,
                        config,
                        last_record,
                        epoch,
                        step,
                        best_val_loss,
                    )
                    last_checkpoint_time = time.monotonic()
                    emit({"type": "checkpoint", "checkpoint_kind": "latest", "path": str(latest_checkpoint_path), "epoch": epoch, "step": step})
            if stopped_by == "max_seconds" and not losses:
                break

            optimization_loss = float(np.mean(losses)) if losses else 0.0
            emit({"type": "validation_start", "epoch": epoch, "step": step})
            train_report = _evaluate_model(model, train_examples, resolved_device)
            val_report = _evaluate_model(model, val_examples or train_examples, resolved_device)
            train_loss = float(train_report["loss"])
            val_loss = float(val_report["loss"])
            improved = val_loss < (best_val_loss - float(early_stop_min_delta))
            if improved:
                best_val_loss = val_loss
                best_epoch = int(epoch)
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            gap = _loss_gap(train_loss, val_loss)
            last_record = {
                "type": "epoch",
                "epoch": epoch,
                "step": step,
                "optimization_loss": qfloat(optimization_loss),
                "train_eval_loss": qfloat(train_loss),
                "val_eval_loss": qfloat(val_loss),
                "train_selection_regret": qfloat(train_report["selection_regret"]),
                "val_selection_regret": qfloat(val_report["selection_regret"]),
                "train_ranking_nll": qfloat(train_report["ranking_nll"]),
                "val_ranking_nll": qfloat(val_report["ranking_nll"]),
                "train_kpi_mse": qfloat(train_report["kpi_mse"]),
                "val_kpi_mse": qfloat(val_report["kpi_mse"]),
                "train_loss": qfloat(train_loss),
                "val_loss": qfloat(val_loss),
                "loss_gap": qfloat(gap),
                "loss_gap_ratio": qfloat(_loss_gap_ratio(train_loss, val_loss)),
                "generalization_status": _generalization_status(train_loss, val_loss, epochs_since_best),
                "train_rank_accuracy": qfloat(train_report["rank_accuracy"]),
                "val_rank_accuracy": qfloat(val_report["rank_accuracy"]),
                "rank_accuracy": qfloat(val_report["rank_accuracy"]),
                "best_val_loss": qfloat(best_val_loss),
                "best_epoch": int(best_epoch),
                "epochs_since_best": int(epochs_since_best),
                "optimization_real_examples": int(epoch_real_examples),
                "optimization_synthetic_examples": int(epoch_synthetic_examples),
                "train_eval_source": "real",
                "train_eval_real_examples": int(len(train_examples)),
                "val_eval_source": "held_out_real" if val_examples else "train_real_fallback",
                "val_eval_real_examples": int(len(val_examples or train_examples)),
                "val_eval_synthetic_examples": 0,
                "datasets": len(examples),
                **_flatten_best(best_record),
                **run_info,
            }
            sink.write(last_record)
            emit({"type": "epoch_metrics", **last_record})
            if improved:
                _save_tuner_checkpoint(
                    best_checkpoint_path,
                    model,
                    input_dim,
                    examples,
                    tasks,
                    config,
                    last_record,
                    epoch,
                    step,
                    best_val_loss,
                    best_epoch=best_epoch,
                    epochs_since_best=epochs_since_best,
                )
                emit({"type": "checkpoint", "checkpoint_kind": "best", "path": str(best_checkpoint_path), "epoch": epoch, "step": step})
            _save_tuner_checkpoint(
                latest_checkpoint_path,
                model,
                input_dim,
                examples,
                tasks,
                config,
                last_record,
                epoch,
                step,
                best_val_loss,
                best_epoch=best_epoch,
                epochs_since_best=epochs_since_best,
            )
            if int(early_stop_patience) > 0 and epochs_since_best >= int(early_stop_patience):
                stopped_by = "early_stop"
                emit(
                    {
                        "type": "early_stop",
                        "epoch": epoch,
                        "step": step,
                        "best_epoch": int(best_epoch),
                        "best_val_loss": qfloat(best_val_loss),
                        "epochs_since_best": int(epochs_since_best),
                        "patience": int(early_stop_patience),
                    }
                )
                break
            if stopped_by == "max_seconds":
                break

        if last_record is None:
            train_report = _evaluate_model(model, train_examples, resolved_device)
            val_report = _evaluate_model(model, val_examples or train_examples, resolved_device)
            train_loss = float(train_report["loss"])
            val_loss = float(val_report["loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = max(start_epoch - 1, 0)
            last_record = {
                "type": "epoch",
                "epoch": max(start_epoch - 1, 0),
                "step": step,
                "optimization_loss": 0.0,
                "train_eval_loss": qfloat(train_loss),
                "val_eval_loss": qfloat(val_loss),
                "train_selection_regret": qfloat(train_report["selection_regret"]),
                "val_selection_regret": qfloat(val_report["selection_regret"]),
                "train_ranking_nll": qfloat(train_report["ranking_nll"]),
                "val_ranking_nll": qfloat(val_report["ranking_nll"]),
                "train_kpi_mse": qfloat(train_report["kpi_mse"]),
                "val_kpi_mse": qfloat(val_report["kpi_mse"]),
                "train_loss": qfloat(train_loss),
                "val_loss": qfloat(val_loss),
                "loss_gap": qfloat(_loss_gap(train_loss, val_loss)),
                "loss_gap_ratio": qfloat(_loss_gap_ratio(train_loss, val_loss)),
                "generalization_status": _generalization_status(train_loss, val_loss, epochs_since_best),
                "train_rank_accuracy": qfloat(train_report["rank_accuracy"]),
                "val_rank_accuracy": qfloat(val_report["rank_accuracy"]),
                "rank_accuracy": qfloat(val_report["rank_accuracy"]),
                "best_val_loss": qfloat(best_val_loss),
                "best_epoch": int(best_epoch),
                "epochs_since_best": int(epochs_since_best),
                "optimization_real_examples": 0,
                "optimization_synthetic_examples": 0,
                "train_eval_source": "real",
                "train_eval_real_examples": int(len(train_examples)),
                "val_eval_source": "held_out_real" if val_examples else "train_real_fallback",
                "val_eval_real_examples": int(len(val_examples or train_examples)),
                "val_eval_synthetic_examples": 0,
                "datasets": len(examples),
                **_flatten_best(best_record),
                **run_info,
            }
            sink.write(last_record)
            emit({"type": "epoch_metrics", **last_record})

        _save_tuner_checkpoint(
            latest_checkpoint_path,
            model,
            input_dim,
            examples,
            tasks,
            config,
            last_record,
            int(last_record.get("epoch", 0)),
            step,
            best_val_loss,
            best_epoch=best_epoch,
            epochs_since_best=epochs_since_best,
        )
        if not best_checkpoint_path.exists():
            _save_tuner_checkpoint(
                best_checkpoint_path,
                model,
                input_dim,
                examples,
                tasks,
                config,
                last_record,
                int(last_record.get("epoch", 0)),
                step,
                best_val_loss,
                best_epoch=best_epoch,
                epochs_since_best=epochs_since_best,
            )
        emit({"type": "checkpoint", "checkpoint_kind": "latest", "path": str(latest_checkpoint_path), "epoch": last_record.get("epoch"), "step": step})

    summary = {
        "format": "dope-kernel-tuner",
        "version": 2,
        "preset": preset,
        "seed": seed,
        "all_datasets": bool(all_datasets),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "row_cap": int(row_cap),
        "max_datasets": None if max_datasets is None else int(max_datasets),
        "max_seconds": None if max_seconds is None else float(max_seconds),
        "requested_prep_max_seconds": None if prep_max_seconds is None else float(prep_max_seconds),
        "target_prepared_datasets": None if target_prepared_datasets is None else int(target_prepared_datasets),
        "checkpoint_every": float(checkpoint_every),
        "full_kpi_every": int(full_kpi_every),
        "full_kpi_top_k": int(full_kpi_top_k),
        "full_kpi_audit_count": int(full_kpi_audit_count),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "synthetic_mode": synthetic_mode,
        "synthetic_label_mode": synthetic_label_mode,
        "synthetic_fast_relabel_limit": int(synthetic_fast_relabel_limit),
        "binary_label_mode": binary_label_mode,
        "validation_binary_label_mode": validation_binary_label_mode,
        "example_cache_dir": str(example_cache_dir),
        "adaptive_relabel": bool(adaptive_relabel),
        "real_batch_fraction": qfloat(real_batch_fraction),
        "synthetic_batch_fraction": qfloat(synthetic_batch_fraction),
        "prep_max_seconds": qfloat(prep_deadline - start_time) if prep_deadline is not None else None,
        "best_val_loss": qfloat(best_val_loss) if math.isfinite(best_val_loss) else None,
        "best_epoch": int(best_epoch),
        "epochs_since_best": int(epochs_since_best),
        "stopped_by": stopped_by,
        "elapsed_seconds": qfloat(time.monotonic() - start_time),
        "out": str(out),
        "checkpoint": str(best_checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "metrics_jsonl": str(sink.metrics_path),
        "events_jsonl": str(sink.events_path),
        "report_html": str(sink.report_path) if report_html else None,
        "trust_report_json": str(sink.trust_report_path),
        "prepared_real_datasets": int(len(examples)),
        "selected_before_target": int(selected_before_target),
        "train_real_datasets": int(len(train_examples)),
        "val_real_datasets": int(len(val_examples)),
        "selected_datasets": [_spec_json(example.spec) for example in examples],
        "candidate_configs": {task: candidate_configs(task) for task in parse_tasks(tasks)},
        "kernel_summaries": _top_kernel_summaries(examples),
        "checkpoints": [event for event in sink.events if event.get("type") == "checkpoint"],
        "final": last_record or {},
        "host": run_info["host"],
        "device": run_info["device"],
        "gpu": run_info["gpu"],
        "disk": run_info["disk"],
        "max_cuda_memory_mb": _max_cuda_memory_mb(resolved_device),
    }
    trust_report = _build_trust_report(
        examples=examples,
        train_examples=train_examples,
        val_examples=val_examples,
        config=config,
        final_record=last_record or {},
        run_info=run_info,
        previous_run_identity=previous_run_identity,
        target_prepared_datasets=target_prepared_datasets,
        selected_before_target=selected_before_target,
    )
    summary["trust"] = trust_report
    sink.write_trust_report(trust_report)
    emit({"type": "trust_summary", "trust": trust_report})
    emit(
        {
            "type": "run_complete",
            "summary": {
                key: summary[key]
                for key in ("out", "checkpoint", "latest_checkpoint", "stopped_by", "elapsed_seconds", "trust_report_json")
            }
            | {"trust_status": trust_report.get("trust_status"), "failing_gates": trust_report.get("failing_gates")},
        }
    )
    sink.write_summary(summary)
    if report_html:
        sink.write_report(summary)
    return summary


def _build_trust_report(
    examples: list[TunerExample],
    train_examples: list[TunerExample],
    val_examples: list[TunerExample],
    config: dict[str, Any],
    final_record: dict[str, Any],
    run_info: dict[str, Any],
    previous_run_identity: str | None,
    target_prepared_datasets: int | None,
    selected_before_target: int,
) -> dict[str, Any]:
    run_identity = _trust_run_identity(config, examples)
    label_modes = Counter(str(example.label_quality.get("label_mode", "unknown")) for example in examples)
    prepared_by_task = Counter(example.spec.task for example in examples)
    train_by_task = Counter(example.spec.task for example in train_examples)
    val_by_task = Counter(example.spec.task for example in val_examples)
    audited_examples = [example for example in val_examples if bool(example.label_quality.get("full_kpi_audited"))]
    proxy_count = sum(int(example.label_quality.get("proxy_label_count") or 0) for example in examples)
    candidate_count = sum(int(example.label_quality.get("candidate_count") or len(example.records)) for example in examples)
    full_delta = _delta_summary(
        record.get("full_kpi_delta")
        for example in audited_examples
        for record in example.records
    )
    audit_mean_delta = full_delta.get("mean")
    audit_agrees = isinstance(audit_mean_delta, (int, float)) and abs(float(audit_mean_delta)) <= TRUST_AUDIT_DELTA_LIMIT
    baselines = _baseline_regret_report(train_examples, val_examples)
    train_regret = _number_or_none(final_record.get("train_selection_regret"))
    val_regret = _number_or_none(final_record.get("val_selection_regret"))
    regret_gap = abs(float(val_regret) - float(train_regret)) if train_regret is not None and val_regret is not None else None
    required_audit = min(TRUST_FULL_KPI_AUDIT_TARGET, len(val_examples)) if config.get("preset") == "trust" else 0
    gates = {
        "held_out_real_validation": bool(val_examples) and str(final_record.get("val_eval_source")) == "held_out_real",
        "train_val_regret_gap": regret_gap is not None and regret_gap <= TRUST_GAP_LIMIT,
        "beats_random_candidate": val_regret is not None
        and baselines.get("random_candidate_regret") is not None
        and float(val_regret) < float(baselines["random_candidate_regret"]),
        "beats_best_fixed_per_task": val_regret is not None
        and baselines.get("best_fixed_per_task_regret") is not None
        and float(val_regret) < float(baselines["best_fixed_per_task_regret"]),
        "seed_stability": False,
        "full_kpi_audit_count": len(audited_examples) >= required_audit,
        "full_kpi_fast_agreement": required_audit == 0 or audit_agrees,
    }
    if not val_examples:
        gates["full_kpi_audit_count"] = False
    failing_gates = [name for name, passed in gates.items() if not passed]
    audit_trusted = str(config.get("preset")) == "trust" and not failing_gates
    target = int(target_prepared_datasets) if target_prepared_datasets is not None else None
    if target is None:
        preparation_status = "not_targeted"
    elif len(examples) >= min(target, int(selected_before_target)):
        preparation_status = "target_met" if int(selected_before_target) >= target else "max_feasible_corpus_prepared"
    else:
        preparation_status = "budget_stopped_before_target"
    return {
        "format": "dope-kernel-tuner-trust-report",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_identity": run_identity,
        "previous_run_identity": previous_run_identity,
        "stale_run_identity": previous_run_identity is not None and previous_run_identity != run_identity,
        "trust_status": "audit trusted" if audit_trusted else "experimental",
        "audit_trusted": bool(audit_trusted),
        "failing_gates": failing_gates,
        "gates": gates,
        "thresholds": {
            "train_val_regret_gap": TRUST_GAP_LIMIT,
            "full_kpi_fast_delta": TRUST_AUDIT_DELTA_LIMIT,
            "required_stability_seeds": TRUST_REQUIRED_SEEDS,
            "trust_full_kpi_audit_target": TRUST_FULL_KPI_AUDIT_TARGET,
        },
        "prepared": {
            "real_datasets": int(len(examples)),
            "target_prepared_datasets": target,
            "selected_before_target": int(selected_before_target),
            "preparation_status": preparation_status,
            "by_task": dict(sorted(prepared_by_task.items())),
        },
        "split": {
            "train_real_datasets": int(len(train_examples)),
            "val_real_datasets": int(len(val_examples)),
            "train_by_task": dict(sorted(train_by_task.items())),
            "val_by_task": dict(sorted(val_by_task.items())),
            "validation_source": "held_out_real" if val_examples else "train_real_fallback",
        },
        "labels": {
            "scorer_version": SCORER_VERSION,
            "label_modes": dict(sorted(label_modes.items())),
            "binary_label_mode": config.get("binary_label_mode"),
            "validation_binary_label_mode": config.get("validation_binary_label_mode"),
            "candidate_count": int(candidate_count),
            "proxy_label_count": int(proxy_count),
            "proxy_label_share": qfloat(proxy_count / max(1, candidate_count)),
            "full_kpi_audited_datasets": int(len(audited_examples)),
            "full_kpi_audit_required": int(required_audit),
            "full_vs_fast_kpi_delta": full_delta,
        },
        "synthetic": {
            "synthetic_mode": config.get("synthetic_mode"),
            "synthetic_label_mode": config.get("synthetic_label_mode"),
            "real_batch_fraction": config.get("real_batch_fraction"),
            "synthetic_batch_fraction": config.get("synthetic_batch_fraction"),
            "optimization_real_examples": final_record.get("optimization_real_examples"),
            "optimization_synthetic_examples": final_record.get("optimization_synthetic_examples"),
            "validation_synthetic_examples": final_record.get("val_eval_synthetic_examples"),
        },
        "metrics": {
            "train_selection_regret": train_regret,
            "val_selection_regret": val_regret,
            "train_val_regret_gap": qfloat(regret_gap) if regret_gap is not None else None,
            "fast_validation_regret": val_regret,
            "full_kpi_audited_delta_mean": audit_mean_delta,
            "rank_accuracy": final_record.get("rank_accuracy"),
        },
        "baselines": baselines,
        "seed_stability": {
            "seeds_evaluated": 1,
            "required": TRUST_REQUIRED_SEEDS,
            "passed": False,
            "evaluated_seed": config.get("seed"),
        },
        "run": {
            "host": run_info.get("host"),
            "device": run_info.get("device"),
            "corpus": run_info.get("corpus"),
            "tasks": run_info.get("tasks"),
            "preset": config.get("preset"),
            "max_seconds": config.get("max_seconds"),
            "prep_max_seconds": config.get("prep_max_seconds"),
            "example_cache_dir": config.get("example_cache_dir"),
        },
    }


def _trust_run_identity(config: dict[str, Any], examples: list[TunerExample]) -> str:
    payload = {
        "scorer_version": SCORER_VERSION,
        "config": {
            key: config.get(key)
            for key in (
                "preset",
                "seed",
                "tasks",
                "row_cap",
                "binary_label_mode",
                "validation_binary_label_mode",
                "synthetic_mode",
                "synthetic_label_mode",
                "full_kpi_top_k",
                "full_kpi_audit_count",
                "target_prepared_datasets",
            )
        },
        "datasets": [_spec_json(example.spec) for example in examples],
        "candidate_configs": {task: candidate_configs(task) for task in sorted({example.spec.task for example in examples})},
    }
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def _read_previous_trust_identity(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    value = payload.get("run_identity") if isinstance(payload, dict) else None
    return str(value) if value else None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return qfloat(float(value))
    return None


def _baseline_regret_report(train_examples: list[TunerExample], val_examples: list[TunerExample]) -> dict[str, Any]:
    eval_examples = val_examples or train_examples
    if not eval_examples:
        return {
            "random_candidate_regret": None,
            "best_fixed_per_task_regret": None,
            "best_fixed_per_task_index": {},
        }
    random_regret = float(np.mean([_random_candidate_regret(example) for example in eval_examples]))
    fixed_indexes = _best_fixed_candidate_indexes(train_examples or eval_examples)
    fixed_regrets = []
    for example in eval_examples:
        index = fixed_indexes.get(example.spec.task, int(example.best_index))
        fixed_regrets.append(_candidate_regret(example, int(index)))
    return {
        "random_candidate_regret": qfloat(random_regret),
        "best_fixed_per_task_regret": qfloat(float(np.mean(fixed_regrets))) if fixed_regrets else None,
        "best_fixed_per_task_index": {task: int(index) for task, index in sorted(fixed_indexes.items())},
    }


def _random_candidate_regret(example: TunerExample) -> float:
    rbcs_values = np.asarray(example.kpis[:, 0], dtype=float)
    if len(rbcs_values) == 0:
        return 0.0
    return float(np.clip(float(np.max(rbcs_values)) - float(np.mean(rbcs_values)), 0.0, 1.0))


def _best_fixed_candidate_indexes(examples: list[TunerExample]) -> dict[str, int]:
    by_task: dict[str, list[TunerExample]] = {}
    for example in examples:
        by_task.setdefault(example.spec.task, []).append(example)
    indexes: dict[str, int] = {}
    for task, task_examples in by_task.items():
        candidate_count = min(example.kpis.shape[0] for example in task_examples)
        if candidate_count <= 0:
            continue
        means = []
        for candidate_index in range(candidate_count):
            means.append(float(np.mean([example.kpis[candidate_index, 0] for example in task_examples])))
        indexes[task] = int(np.argmax(np.asarray(means, dtype=float)))
    return indexes


def _candidate_regret(example: TunerExample, candidate_index: int) -> float:
    rbcs_values = np.asarray(example.kpis[:, 0], dtype=float)
    if len(rbcs_values) == 0:
        return 0.0
    candidate_index = max(0, min(int(candidate_index), len(rbcs_values) - 1))
    return float(np.clip(float(np.max(rbcs_values)) - float(rbcs_values[candidate_index]), 0.0, 1.0))


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device)


def _run_info(
    device: torch.device,
    corpus: str | Path,
    tasks: Any,
    max_datasets: int | None,
    row_cap: int,
    out: str | Path,
    all_datasets: bool = False,
) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        props = torch.cuda.get_device_properties(index)
        gpu = {
            "index": int(index),
            "name": props.name,
            "total_memory_mb": int(props.total_memory // (1024 * 1024)),
        }
    return {
        "host": socket.gethostname(),
        "device": str(device),
        "gpu": gpu,
        "corpus": str(corpus),
        "tasks": parse_tasks(tasks),
        "max_datasets": None if max_datasets is None else int(max_datasets),
        "all_datasets": bool(all_datasets),
        "row_cap": int(row_cap),
        "disk": _disk_info(out),
    }


def _disk_info(path: str | Path) -> dict[str, Any]:
    probe = Path(path)
    if not probe.exists():
        probe = probe.parent if probe.parent.exists() else Path.cwd()
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return {"path": str(probe), "total_mb": None, "free_mb": None}
    return {
        "path": str(probe),
        "total_mb": int(usage.total // (1024 * 1024)),
        "free_mb": int(usage.free // (1024 * 1024)),
    }


def _train_val_split(examples: list[TunerExample], seed: int) -> tuple[list[TunerExample], list[TunerExample]]:
    if len(examples) <= 1:
        return examples, []
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(examples)))
    val_count = max(1, len(examples) // 4)
    val_idx = set(order[:val_count])
    train = [example for idx, example in enumerate(examples) if idx not in val_idx]
    val = [example for idx, example in enumerate(examples) if idx in val_idx]
    return train or examples, val


def _limit_specs_for_target(specs: list[DatasetSpec], target_prepared_datasets: int | None) -> list[DatasetSpec]:
    if target_prepared_datasets is None or int(target_prepared_datasets) <= 0:
        return list(specs)
    return list(specs[: int(target_prepared_datasets)])


def _train_val_spec_split(specs: list[DatasetSpec], seed: int) -> tuple[list[DatasetSpec], list[DatasetSpec]]:
    if len(specs) <= 1:
        return list(specs), []
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[DatasetSpec]] = {}
    for spec in specs:
        by_task.setdefault(spec.task, []).append(spec)
    val_keys: set[str] = set()
    for task_specs in by_task.values():
        if len(task_specs) <= 1:
            continue
        order = rng.permutation(len(task_specs))
        count = max(1, len(task_specs) // 4)
        for idx in order[:count]:
            val_keys.add(_spec_identity(task_specs[int(idx)]))
    if not val_keys:
        order = rng.permutation(len(specs))
        val_keys.add(_spec_identity(specs[int(order[0])]))
    train = [spec for spec in specs if _spec_identity(spec) not in val_keys]
    val = [spec for spec in specs if _spec_identity(spec) in val_keys]
    return train or list(specs), val


def _interleave_train_val_specs(train_specs: list[DatasetSpec], val_specs: list[DatasetSpec]) -> list[DatasetSpec]:
    ordered: list[DatasetSpec] = []
    train_queue = list(train_specs)
    val_queue = list(val_specs)
    while val_queue or train_queue:
        if val_queue:
            ordered.append(val_queue.pop(0))
        for _ in range(3):
            if train_queue:
                ordered.append(train_queue.pop(0))
    return ordered


def _full_kpi_audit_keys(val_specs: list[DatasetSpec], full_kpi_audit_count: int) -> set[str]:
    if full_kpi_audit_count <= 0:
        return set()
    return {_spec_identity(spec) for spec in val_specs[: int(full_kpi_audit_count)]}


def _training_mix_fractions(synthetic_mode: str) -> tuple[float, float]:
    if synthetic_mode not in TRAINING_MIXES:
        choices = ", ".join(sorted(TRAINING_MIXES))
        raise ValueError(f"synthetic_mode must be one of: {choices}")
    return TRAINING_MIXES[synthetic_mode]


def _training_mix_counts(batch_size: int, synthetic_mode: str) -> tuple[int, int]:
    batch_size = max(1, int(batch_size))
    real_fraction, synthetic_fraction = _training_mix_fractions(synthetic_mode)
    if synthetic_fraction <= 0.0 or batch_size == 1:
        return batch_size, 0
    real_count = int(round(batch_size * real_fraction))
    real_count = max(1, min(batch_size - 1, real_count))
    return real_count, batch_size - real_count


def _training_batches(
    real_examples: list[TunerExample],
    batch_size: int,
    synthetic_mode: str,
    seed: int,
    synthetic_label_mode: str = "anchor-consistency",
    synthetic_fast_relabel_limit: int = 1,
) -> Iterator[list[TunerExample]]:
    _validate_synthetic_label_mode(synthetic_label_mode)
    real_count, synthetic_count = _training_mix_counts(batch_size, synthetic_mode)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(real_examples)))
    anchors = [example for example in real_examples if example.augmentation_kernel is not None]
    for start in range(0, len(order), real_count):
        batch = [real_examples[idx] for idx in order[start : start + real_count]]
        if synthetic_count and anchors:
            batch.extend(_sample_synthetic_examples(anchors, synthetic_count, rng, synthetic_label_mode, synthetic_fast_relabel_limit))
        yield batch


def _sample_synthetic_examples(
    anchors: list[TunerExample],
    count: int,
    rng: np.random.Generator,
    synthetic_label_mode: str = "anchor-consistency",
    synthetic_fast_relabel_limit: int = 1,
) -> list[TunerExample]:
    synthetic: list[TunerExample] = []
    if count <= 0 or not anchors:
        return synthetic
    fast_relabels_remaining = max(0, int(synthetic_fast_relabel_limit)) if synthetic_label_mode == "fast-relabel" else 0
    for _ in range(int(count)):
        anchor = anchors[int(rng.integers(0, len(anchors)))]
        mode = "fast-relabel" if fast_relabels_remaining > 0 else "anchor-consistency"
        example = _synthetic_example_from_anchor(anchor, int(rng.integers(0, 2**31 - 1)), synthetic_label_mode=mode)
        if mode == "fast-relabel" and example is not None:
            fast_relabels_remaining -= 1
        if example is not None:
            synthetic.append(example)
    return synthetic


def _synthetic_example_from_anchor(anchor: TunerExample, seed: int, synthetic_label_mode: str = "anchor-consistency") -> TunerExample | None:
    _validate_synthetic_label_mode(synthetic_label_mode)
    if anchor.augmentation_kernel is None:
        return None
    train_rows = max(2, int(anchor.sample_rows.get("train", 0) or 0))
    test_rows = max(2, int(anchor.sample_rows.get("test", 0) or train_rows // 2 or 2))
    try:
        sampled = sample_kernel(anchor.augmentation_kernel, train_rows + test_rows, seed=seed)
    except Exception:
        return None
    X = _clean(sampled[:, :-1])
    y = _clean_y(sampled[:, -1], anchor.spec.task)
    X_train = X[:train_rows]
    y_train = y[:train_rows]
    X_test = X[train_rows : train_rows + test_rows]
    y_test = y[train_rows : train_rows + test_rows]
    if anchor.spec.task == "binary":
        y_train = _repair_binary_targets(y_train, seed + 1)
        y_test = _repair_binary_targets(y_test, seed + 2)
    if synthetic_label_mode == "fast-relabel":
        return _fast_relabel_synthetic_example(anchor, X_train, y_train, X_test, y_test, seed)
    dataset_features = _dataset_feature_vector(X_train, y_train, X_test, y_test, anchor.spec.task)
    rows = []
    features = []
    for record in anchor.records:
        config = dict(record.get("config") or {})
        if not config:
            continue
        kernel_summary = record.get("kernel_summary") if isinstance(record.get("kernel_summary"), dict) else {}
        kernel_like = {
            "kernel_bytes": int(record.get("kernel_bytes") or 0),
            "description_bits": int(kernel_summary.get("description_bits") or max(1, int(record.get("kernel_bytes") or 0) * 8)),
        }
        features.append(np.concatenate([dataset_features, _candidate_feature_vector(config, kernel_like, train_rows, X_train.shape[1] + 1)]))
        synthetic_record = dict(record)
        synthetic_record["label_source"] = "anchor_consistency"
        synthetic_record["anchor_dataset_id"] = anchor.spec.dataset_id
        synthetic_record["label_weight"] = ANCHOR_CONSISTENCY_LABEL_WEIGHT
        rows.append(synthetic_record)
    if len(rows) != len(anchor.records):
        return None
    return TunerExample(
        spec=DatasetSpec(path=anchor.spec.path, task=anchor.spec.task, dataset_id=f"{anchor.spec.dataset_id}::synthetic"),
        features=np.asarray(features, dtype=np.float32),
        kpis=np.asarray(anchor.kpis[: len(rows)], dtype=np.float32),
        best_index=int(anchor.best_index),
        records=tuple(rows),
        source="synthetic",
        anchor_dataset_id=anchor.spec.dataset_id,
        label_quality={
            "source": "synthetic",
            "label_source": "anchor_consistency",
            "synthetic_label_mode": "anchor-consistency",
            "anchor_dataset_id": anchor.spec.dataset_id,
            "label_weight": ANCHOR_CONSISTENCY_LABEL_WEIGHT,
            "authoritative": False,
            "sampled_rows": {"train": int(train_rows), "test": int(test_rows)},
            "class_balance": {
                "train": qfloat(_binary_class_balance(y_train)) if anchor.spec.task == "binary" else None,
                "test": qfloat(_binary_class_balance(y_test)) if anchor.spec.task == "binary" else None,
            },
        },
        augmentation_kernel=anchor.augmentation_kernel,
        sample_rows={"train": int(train_rows), "test": int(test_rows)},
        label_weight=ANCHOR_CONSISTENCY_LABEL_WEIGHT,
    )


def _fast_relabel_synthetic_example(
    anchor: TunerExample,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> TunerExample | None:
    dataset_features = _dataset_feature_vector(X_train, y_train, X_test, y_test, anchor.spec.task)
    configs = candidate_configs(anchor.spec.task)
    smoke_cache = _build_smoke_score_cache(X_train, y_train, X_test, y_test, anchor.spec.task, seed)
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    for candidate_index, config in enumerate(configs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                kernel = fit_kernel_from_arrays(
                    X_train,
                    y_train,
                    anchor.spec.task,
                    seed=seed,
                    dependence_kind=str(config["dependence_kind"]),
                    target_kind=str(config["target_kind"]),
                    empq_k=config["empq_k"],
                )
                synth = sample_kernel(kernel, len(X_train), seed=seed + candidate_index + 1)
                report = _score_synthetic_dataset_smoke(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    synth[:, :-1],
                    synth[:, -1],
                    anchor.spec.task,
                    seed,
                    cache=smoke_cache,
                )
            row = _candidate_metric_row(
                candidate_index,
                config,
                kernel,
                report,
                rows=len(X_train),
                cols=X_train.shape[1] + 1,
                score_mode="fast",
            )
            row["label_source"] = "synthetic_fast_relabel"
            row["anchor_dataset_id"] = anchor.spec.dataset_id
            row["label_weight"] = FAST_RELABEL_SYNTHETIC_WEIGHT
            feature_vec = _candidate_feature_vector(config, kernel, len(X_train), X_train.shape[1] + 1)
        except Exception as exc:  # pragma: no cover - defensive for long runs
            row = _failed_candidate_record(candidate_index, config, exc)
            row["label_source"] = "synthetic_fast_relabel"
            row["anchor_dataset_id"] = anchor.spec.dataset_id
            row["label_weight"] = FAST_RELABEL_SYNTHETIC_WEIGHT
            feature_vec = np.zeros(_candidate_feature_dim(), dtype=np.float32)
        rows.append(row)
        features.append(np.concatenate([dataset_features, feature_vec]))
    if not rows:
        return None
    kpis = np.asarray([[float(row[field]) for field in KPI_FIELDS] for row in rows], dtype=np.float32)
    best_index = int(np.argmax(kpis[:, 0]))
    return TunerExample(
        spec=DatasetSpec(path=anchor.spec.path, task=anchor.spec.task, dataset_id=f"{anchor.spec.dataset_id}::synthetic-fast"),
        features=np.asarray(features, dtype=np.float32),
        kpis=kpis,
        best_index=best_index,
        records=tuple(rows),
        source="synthetic",
        anchor_dataset_id=anchor.spec.dataset_id,
        label_quality={
            "source": "synthetic",
            "label_source": "synthetic_fast_relabel",
            "synthetic_label_mode": "fast-relabel",
            "anchor_dataset_id": anchor.spec.dataset_id,
            "label_weight": FAST_RELABEL_SYNTHETIC_WEIGHT,
            "authoritative": True,
            "sampled_rows": {"train": int(len(X_train)), "test": int(len(X_test))},
            "score_modes": dict(Counter(str(row.get("score_mode", "unknown")) for row in rows)),
        },
        augmentation_kernel=anchor.augmentation_kernel,
        sample_rows={"train": int(len(X_train)), "test": int(len(X_test))},
        label_weight=FAST_RELABEL_SYNTHETIC_WEIGHT,
    )


def _batches(examples: list[TunerExample], batch_size: int, seed: int) -> Iterator[list[TunerExample]]:
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(examples)))
    for start in range(0, len(order), max(1, batch_size)):
        yield [examples[idx] for idx in order[start : start + max(1, batch_size)]]


def _batch_tensors(examples: list[TunerExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(np.stack([example.features for example in examples]), dtype=torch.float32, device=device)
    kpis = torch.as_tensor(np.stack([example.kpis for example in examples]), dtype=torch.float32, device=device)
    best = torch.as_tensor([example.best_index for example in examples], dtype=torch.long, device=device)
    return features, kpis, best


def _batch_weights(examples: list[TunerExample], device: torch.device) -> torch.Tensor:
    return torch.as_tensor([float(example.label_weight) for example in examples], dtype=torch.float32, device=device)


@torch.no_grad()
def _evaluate_model(model: KernelTuner, examples: list[TunerExample], device: torch.device) -> dict[str, float]:
    model.eval()
    features, kpis, best_index = _batch_tensors(examples, device)
    outputs = model(features)
    logits = outputs["ranking_logits"]
    rbcs = kpis[:, :, 0]
    predicted_index = torch.argmax(logits, dim=1)
    best_rbcs = rbcs.gather(1, best_index.unsqueeze(1)).squeeze(1)
    selected_rbcs = rbcs.gather(1, predicted_index.unsqueeze(1)).squeeze(1)
    selection_regret = torch.clamp(best_rbcs - selected_rbcs, min=0.0, max=1.0)
    ranking_nll = F.cross_entropy(logits, best_index)
    kpi_mse = F.mse_loss(torch.clamp(outputs["kpi"], 0.0, 1.0), kpis)
    best_target = rbcs.amax(dim=1, keepdim=True)
    loss_target = best_target - rbcs
    aggregation_mse = F.mse_loss(torch.clamp(outputs["candidate_loss"], 0.0, 1.0), loss_target)
    accuracy = torch.mean((torch.argmax(logits, dim=1) == best_index).float())
    return {
        "loss": float(selection_regret.mean().cpu()),
        "selection_regret": float(selection_regret.mean().cpu()),
        "ranking_nll": float(ranking_nll.cpu()),
        "kpi_mse": float(kpi_mse.cpu()),
        "aggregation_mse": float(aggregation_mse.cpu()),
        "rank_accuracy": float(accuracy.cpu()),
    }


def _loss_gap(train_loss: float, val_loss: float) -> float:
    return max(0.0, float(val_loss) - float(train_loss))


def _loss_gap_ratio(train_loss: float, val_loss: float) -> float:
    return _loss_gap(train_loss, val_loss) / max(0.05, abs(float(train_loss)))


def _generalization_status(train_loss: float, val_loss: float, epochs_since_best: int) -> str:
    ratio = _loss_gap_ratio(train_loss, val_loss)
    if epochs_since_best >= 5 and ratio > 1.0:
        return "overfit"
    if ratio > 1.0:
        return "watch"
    if epochs_since_best > 0:
        return "worsening"
    return "improving"


def _best_candidate_record(examples: list[TunerExample]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for example in examples:
        for record in example.records:
            item = {"dataset": _spec_json(example.spec), **record}
            if best is None or float(item["rbcs"]) > float(best["rbcs"]):
                best = item
    if best is None:
        raise RuntimeError("no candidate records were produced")
    return best


def _flatten_best(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "rbcs": record["rbcs"],
        "fidelity_real": record["fidelity_real"],
        "transfer_utility": record["transfer_utility"],
        "feature_importance": record["feature_importance"],
        "distribution_copula": record["distribution_copula"],
        "residual_or_calibration": record["residual_or_calibration"],
        "prediction_agreement": record["prediction_agreement"],
        "kernel_bytes": record["kernel_bytes"],
        "best_candidate_config": record["config"],
        "best_candidate_score_mode": record.get("score_mode"),
        "best_full_kpi_delta": record.get("full_kpi_delta"),
        "best_kernel_summary": record.get("kernel_summary"),
        "best_dataset": record["dataset"],
    }


def _spec_json(spec: DatasetSpec) -> dict[str, Any]:
    return {"path": str(spec.path), "task": spec.task, "dataset_id": spec.dataset_id}


def _max_cuda_memory_mb(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    index = torch.cuda.current_device() if device.index is None else device.index
    return int(torch.cuda.max_memory_allocated(index) // (1024 * 1024))


def _save_tuner_checkpoint(
    path: Path,
    model: KernelTuner,
    input_dim: int,
    examples: list[TunerExample],
    tasks: Any,
    config: dict[str, Any],
    last_record: dict[str, Any] | None,
    epoch: int,
    step: int,
    best_val_loss: float,
    best_epoch: int = 0,
    epochs_since_best: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "dope-kernel-tuner",
        "version": 2,
        "model_state_dict": model.state_dict(),
        "input_dim": int(input_dim),
        "kpi_fields": KPI_FIELDS,
        "config": dict(config),
        "selected_datasets": [_spec_json(example.spec) for example in examples],
        "candidate_configs": {task: candidate_configs(task) for task in parse_tasks(tasks)},
        "metrics": {"last": last_record or {}, "best_val_loss": qfloat(best_val_loss) if math.isfinite(best_val_loss) else None},
        "kernel_summaries": _top_kernel_summaries(examples),
        "epoch": int(epoch),
        "step": int(step),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_since_best": int(epochs_since_best),
    }
    torch.save(payload, path)


def _load_resume_checkpoint(path: Path, device: torch.device) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility
        return torch.load(path, map_location=device)


def _top_kernel_summaries(examples: list[TunerExample], limit: int = 16) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        for record in example.records:
            summary = record.get("kernel_summary")
            if not isinstance(summary, dict):
                continue
            rows.append(
                {
                    "dataset": _spec_json(example.spec),
                    "candidate_index": int(record.get("candidate_index", 0)),
                    "rbcs": record.get("rbcs"),
                    "fidelity_real": record.get("fidelity_real"),
                    "score_mode": record.get("score_mode"),
                    "full_kpi_delta": record.get("full_kpi_delta"),
                    **summary,
                }
            )
    return sorted(rows, key=lambda item: float(item.get("rbcs") or 0.0), reverse=True)[:limit]


@contextlib.contextmanager
def _live_dashboard(enabled: bool, dashboard: "TunerDashboard") -> Iterator[Any | None]:
    if not enabled:
        yield None
        return
    try:
        from rich.live import Live
    except ImportError:
        yield None
        return
    with Live(dashboard.render(), refresh_per_second=4, screen=False) as live:
        yield live


class TunerDashboard:
    def __init__(self, run_info: dict[str, Any] | None = None) -> None:
        self.run_info = run_info or {}
        self.last_event: dict[str, Any] = {"type": "starting", "elapsed_seconds": 0.0}
        self.active_dataset: dict[str, Any] | None = None
        self.datasets_done = 0
        self.candidates_seen = 0
        self.leaderboard: list[dict[str, Any]] = []
        self._leaderboard_keys: set[tuple[Any, ...]] = set()
        self.optimization_losses: list[float] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_rank_accuracy: list[float] = []
        self.val_rank_accuracy: list[float] = []
        self.rbcs_values: list[float] = []
        self.full_deltas: list[float] = []
        self.checkpoint: dict[str, Any] | None = None
        self.warnings: list[str] = []
        self.kernel_summary: dict[str, Any] | None = None
        self.latest_epoch: dict[str, Any] = {}
        self.full_kpi_audited_datasets = 0
        self.proxy_label_count = 0
        self.candidate_label_count = 0
        self.trust_report: dict[str, Any] = {}

    def apply(self, record: dict[str, Any]) -> None:
        self.last_event = record
        event_type = str(record.get("type", ""))
        if event_type == "dataset_selection_complete" and isinstance(record.get("selected_datasets"), list):
            self.run_info["max_datasets"] = len(record["selected_datasets"])
        if event_type in {"dataset_start", "dataset_loaded", "dataset_complete"}:
            dataset = record.get("dataset")
            if isinstance(dataset, dict):
                self.active_dataset = dataset
            if event_type == "dataset_complete":
                self.datasets_done = max(self.datasets_done, int(record.get("dataset_index", -1)) + 1)
        if event_type == "label_quality" and isinstance(record.get("quality"), dict):
            quality = record["quality"]
            if quality.get("full_kpi_audited"):
                self.full_kpi_audited_datasets += 1
            self.proxy_label_count += int(quality.get("proxy_label_count") or 0)
            self.candidate_label_count += int(quality.get("candidate_count") or 0)
        if event_type == "trust_summary" and isinstance(record.get("trust"), dict):
            self.trust_report = dict(record["trust"])
        if event_type in {"candidate_score", "full_kpi_score", "dataset_complete"}:
            if event_type in {"candidate_score", "full_kpi_score"}:
                self.candidates_seen += 1
            self._add_leader(record)
            summary = record.get("kernel_summary")
            if isinstance(summary, dict):
                self.kernel_summary = summary
            if isinstance(record.get("rbcs"), (int, float)):
                self.rbcs_values.append(float(record["rbcs"]))
            if isinstance(record.get("full_kpi_delta"), (int, float)):
                self.full_deltas.append(float(record["full_kpi_delta"]))
        if event_type in {"epoch_metrics", "epoch"}:
            self.latest_epoch = dict(record)
            if isinstance(record.get("optimization_loss"), (int, float)):
                self.optimization_losses.append(float(record["optimization_loss"]))
            train_loss = record.get("train_eval_loss", record.get("train_loss"))
            val_loss = record.get("val_eval_loss", record.get("val_loss"))
            if isinstance(train_loss, (int, float)):
                self.train_losses.append(float(train_loss))
            if isinstance(val_loss, (int, float)):
                self.val_losses.append(float(val_loss))
            if isinstance(record.get("train_rank_accuracy"), (int, float)):
                self.train_rank_accuracy.append(float(record["train_rank_accuracy"]))
            if isinstance(record.get("val_rank_accuracy"), (int, float)):
                self.val_rank_accuracy.append(float(record["val_rank_accuracy"]))
            elif isinstance(record.get("rank_accuracy"), (int, float)):
                self.val_rank_accuracy.append(float(record["rank_accuracy"]))
        if event_type == "train_step" and isinstance(record.get("loss"), (int, float)):
            self.optimization_losses.append(float(record["loss"]))
        if event_type == "checkpoint":
            self.checkpoint = record
        if event_type in {"warning", "candidate_failed"}:
            self.warnings.append(str(record.get("message") or record.get("error") or record))
            self.warnings = self.warnings[-6:]

    def render(self) -> Any:
        try:
            from rich.bar import Bar
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
            from rich.syntax import Syntax
            from rich.table import Table
            from rich.text import Text
        except ImportError:
            return str(self.last_event)

        layout = Layout(name="root")
        layout.split_column(Layout(name="top", size=6), Layout(name="body"), Layout(name="bottom", size=8))
        layout["top"].split_row(
            Layout(self._host_panel(Panel, Table), name="host"),
            Layout(self._time_panel(Panel, Table, Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn), name="time"),
            Layout(self._checkpoint_panel(Panel, Table), name="checkpoint"),
        )
        layout["body"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(
            Layout(self._dataset_panel(Panel, Table, Progress, TextColumn, BarColumn), name="dataset", size=7),
            Layout(self._kernel_panel(Panel, Table, Syntax), name="kernel"),
        )
        layout["right"].split_column(
            Layout(self._metric_panel(Panel, Table, Bar, Text), name="metrics", ratio=2),
            Layout(self._leaderboard_panel(Panel, Table, Bar), name="leaderboard", ratio=1),
        )
        layout["bottom"].split_row(
            Layout(self._delta_panel(Panel, Table, Bar, Text), name="delta"),
            Layout(self._warnings_panel(Panel, Text), name="warnings"),
        )
        return layout

    def _add_leader(self, record: dict[str, Any]) -> None:
        if not isinstance(record.get("rbcs"), (int, float)):
            return
        dataset = record.get("dataset") if isinstance(record.get("dataset"), dict) else {}
        item = {
            "dataset_id": dataset.get("dataset_id", "-") if isinstance(dataset, dict) else "-",
            "task": dataset.get("task", "-") if isinstance(dataset, dict) else "-",
            "candidate_index": record.get("candidate_index"),
            "rbcs": float(record["rbcs"]),
            "fidelity_real": record.get("fidelity_real"),
            "kernel_bytes": record.get("kernel_bytes"),
            "score_mode": record.get("score_mode"),
            "config": record.get("config"),
        }
        key = (item["dataset_id"], item["candidate_index"], item["score_mode"])
        if key in self._leaderboard_keys:
            self.leaderboard = [row for row in self.leaderboard if (row.get("dataset_id"), row.get("candidate_index"), row.get("score_mode")) != key]
        self._leaderboard_keys.add(key)
        self.leaderboard.append(item)
        self.leaderboard = sorted(self.leaderboard, key=lambda row: float(row.get("rbcs") or 0.0), reverse=True)[:8]
        self._leaderboard_keys = {(row.get("dataset_id"), row.get("candidate_index"), row.get("score_mode")) for row in self.leaderboard}

    def _host_panel(self, Panel: Any, Table: Any) -> Any:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan")
        table.add_column()
        gpu = self.run_info.get("gpu")
        disk = self.run_info.get("disk") or {}
        table.add_row("host", _display(self.run_info.get("host")))
        table.add_row("device", _display(self.run_info.get("device")))
        table.add_row("gpu", _display(gpu.get("name") if isinstance(gpu, dict) else gpu))
        table.add_row("disk free", f"{disk.get('free_mb', '-')} MB")
        return Panel(table, title="Host / GPU / Disk", border_style="cyan")

    def _time_panel(
        self,
        Panel: Any,
        Table: Any,
        Progress: Any,
        TextColumn: Any,
        BarColumn: Any,
        TimeElapsedColumn: Any,
        TimeRemainingColumn: Any,
    ) -> Any:
        config = self.run_info.get("config") if isinstance(self.run_info.get("config"), dict) else {}
        elapsed = float(self.last_event.get("elapsed_seconds") or 0.0)
        max_seconds = config.get("max_seconds") if isinstance(config, dict) else None
        total = float(max_seconds) if isinstance(max_seconds, (int, float)) and max_seconds else None
        progress = Progress(
            TextColumn("[bold]elapsed"),
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
            TimeRemainingColumn() if total else TextColumn(""),
            expand=True,
        )
        progress.add_task("elapsed", total=total, completed=min(elapsed, total) if total else elapsed)
        table = Table.grid()
        table.add_row(progress)
        table.add_row(f"event: {_display(self.last_event.get('type'))}")
        return Panel(table, title="Elapsed / ETA", border_style="green")

    def _checkpoint_panel(self, Panel: Any, Table: Any) -> Any:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold green")
        table.add_column()
        checkpoint = self.checkpoint or {}
        table.add_row("kind", _display(checkpoint.get("checkpoint_kind")))
        table.add_row("epoch", _display(checkpoint.get("epoch")))
        table.add_row("step", _display(checkpoint.get("step")))
        table.add_row("path", _display(checkpoint.get("path")))
        return Panel(table, title="Checkpoint", border_style="green" if checkpoint else "yellow")

    def _dataset_panel(self, Panel: Any, Table: Any, Progress: Any, TextColumn: Any, BarColumn: Any) -> Any:
        total = int(self.run_info.get("max_datasets") or 0) or None
        progress = Progress(TextColumn("[bold]datasets"), BarColumn(bar_width=None), TextColumn("{task.completed}/{task.total}"), expand=True)
        progress.add_task("datasets", total=total, completed=self.datasets_done)
        dataset = self.active_dataset or {}
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold magenta")
        table.add_column()
        table.add_row("queue", progress)
        table.add_row("active", _display(dataset.get("dataset_id")))
        table.add_row("task", _display(dataset.get("task")))
        table.add_row("path", _display(dataset.get("path")))
        table.add_row("candidates", str(self.candidates_seen))
        table.add_row("prepared real", str(self.datasets_done))
        table.add_row("full-KPI audited", str(self.full_kpi_audited_datasets))
        proxy_share = self.proxy_label_count / max(1, self.candidate_label_count)
        table.add_row("proxy label share", f"{proxy_share:.3f}")
        return Panel(table, title="Dataset Queue / Active Dataset", border_style="magenta")

    def _kernel_panel(self, Panel: Any, Table: Any, Syntax: Any) -> Any:
        summary = self.kernel_summary or {}
        program = str(summary.get("canonical_program") or "(waiting)")
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan")
        table.add_column()
        table.add_row("config", _display_compact(summary.get("config"), 120))
        table.add_row("bytes", _display(summary.get("kernel_bytes")))
        table.add_row("dependence", _display_compact(summary.get("dependence"), 80))
        table.add_row("target", _display_compact(summary.get("target"), 80))
        table.add_row("marginals", _display_compact(summary.get("marginals"), 100))
        table.add_row("top KPI", _display_compact(summary.get("top_kpi_components"), 160))
        table.add_row("program", Syntax(_truncate_text(program, 420), "lisp", word_wrap=True, background_color="default"))
        return Panel(table, title="Realtime Dataset Kernel", border_style="cyan")

    def _leaderboard_panel(self, Panel: Any, Table: Any, Bar: Any) -> Any:
        table = Table(title=None, expand=True)
        table.add_column("rank", style="bold")
        table.add_column("dataset")
        table.add_column("mode")
        table.add_column("RBCS")
        table.add_column("bytes")
        table.add_column("bar")
        for idx, row in enumerate(self.leaderboard, start=1):
            score = float(row.get("rbcs") or 0.0)
            color = "green" if idx == 1 else "magenta"
            table.add_row(
                str(idx),
                str(row.get("dataset_id")),
                str(row.get("score_mode")),
                f"{score:.4f}",
                _display(row.get("kernel_bytes")),
                Bar(1.0, 0.0, float(np.clip(score, 0.0, 1.0)), width=18, color=color),
            )
        return Panel(table, title="Candidate Leaderboard", border_style="magenta")

    def _metric_panel(self, Panel: Any, Table: Any, Bar: Any, Text: Any) -> Any:
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(style="bold green")
        table.add_column(justify="right")
        table.add_column(justify="right")
        table.add_column(justify="right")
        table.add_column()
        table.add_column()
        table.add_row(Text("signal", style="bold"), Text("now", style="bold"), Text("best", style="bold"), Text("last20", style="bold"), Text("state", style="bold"), Text("level", style="bold"))
        for label, values, lower_is_better in (
            ("optimizer loss", self.optimization_losses, True),
            ("train eval regret", self.train_losses, True),
            ("val eval regret", self.val_losses, True),
            ("train rank acc", self.train_rank_accuracy, False),
            ("val rank acc", self.val_rank_accuracy, False),
            ("candidate RBCS", self.rbcs_values, False),
        ):
            latest, lo, hi, bar_value = _metric_summary(values, lower_is_better)
            best = lo if lower_is_better else hi
            table.add_row(
                label,
                _display(latest),
                _display(best),
                _window_delta_text(values, lower_is_better=lower_is_better, Text=Text),
                _trend_text(values, lower_is_better=lower_is_better, Text=Text),
                Bar(1.0, 0.0, bar_value, width=16, color=_metric_color(values, lower_is_better)),
            )
        return Panel(table, title="Learning Health", border_style="green")

    def _delta_panel(self, Panel: Any, Table: Any, Bar: Any, Text: Any) -> Any:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold yellow")
        table.add_column()
        epoch = self.latest_epoch
        gap = epoch.get("loss_gap")
        ratio = epoch.get("loss_gap_ratio")
        status = str(epoch.get("generalization_status") or "-")
        status_style = {"improving": "green", "worsening": "red", "watch": "yellow", "overfit": "red"}.get(status, "white")
        latest_delta = self.full_deltas[-1] if self.full_deltas else None
        config = self.run_info.get("config") if isinstance(self.run_info.get("config"), dict) else {}
        trust_status = str(self.trust_report.get("trust_status") or config.get("trust_status") or "experimental")
        trust_style = "green" if trust_status == "audit trusted" else "yellow"
        table.add_row("epoch", _display(epoch.get("epoch")))
        table.add_row("best val", f"{_display(epoch.get('best_val_loss'))} @ epoch {_display(epoch.get('best_epoch'))}")
        table.add_row("since best", _display(epoch.get("epochs_since_best")))
        table.add_row("train-val gap", f"{_display(gap)} ({_display(ratio)}x)")
        table.add_row("status", Text(status, style=status_style))
        table.add_row("trust", Text(trust_status, style=trust_style))
        table.add_row("synthetic labels", _display(config.get("synthetic_label_mode")))
        table.add_row("full-vs-fast delta", _display(latest_delta))
        if latest_delta is not None:
            magnitude = min(abs(float(latest_delta or 0.0)), 1.0)
            color = "green" if latest_delta >= 0 else "yellow"
            table.add_row("delta magnitude", Bar(1.0, 0.0, magnitude, width=30, color=color))
        return Panel(table, title="Generalization / Best Checkpoint", border_style=status_style)

    def _warnings_panel(self, Panel: Any, Text: Any) -> Any:
        color = "red" if self.warnings else "green"
        body = Text("\n".join(self.warnings) if self.warnings else "no warnings", style=color)
        return Panel(body, title="Warnings", border_style=color)


def _dashboard(record: dict[str, Any]) -> Any:
    dashboard = TunerDashboard(record if record.get("host") else None)
    dashboard.apply(record)
    return dashboard.render()


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value)
    if value is None:
        return "-"
    return str(value)


def _display_compact(value: Any, limit: int = 120) -> str:
    return _truncate_text(_display(value), limit)


def _truncate_text(value: str, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _metric_summary(values: list[float], lower_is_better: bool) -> tuple[float | None, float | None, float | None, float]:
    arr = _finite_metric_values(values)
    if len(arr) == 0:
        return None, None, None, 0.0
    latest = float(arr[-1])
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    spread = hi - lo
    if spread <= 1e-12:
        bar_value = 1.0 if (latest <= lo if lower_is_better else latest >= hi) else 0.0
    else:
        normalized = (latest - lo) / spread
        bar_value = 1.0 - normalized if lower_is_better else normalized
    return qfloat(latest), qfloat(lo), qfloat(hi), float(np.clip(bar_value, 0.0, 1.0))


def _trend_label(values: list[float], lower_is_better: bool) -> str:
    arr = _finite_metric_values(values)
    if len(arr) < 2:
        return "-"
    delta = float(arr[-1] - arr[0])
    if abs(delta) <= 1e-9:
        return "flat"
    improving = delta < 0 if lower_is_better else delta > 0
    return "improving" if improving else "worsening"


def _trend_text(values: list[float], lower_is_better: bool, Text: Any) -> Any:
    label = _trend_label(values, lower_is_better)
    style = "green" if label == "improving" else "red" if label == "worsening" else "yellow"
    arrow = "↓" if label == "improving" and lower_is_better else "↑" if label == "improving" else "↑" if label == "worsening" and lower_is_better else "↓" if label == "worsening" else "→"
    return Text(f"{arrow} {label}", style=style)


def _window_delta_text(values: list[float], lower_is_better: bool, Text: Any, window: int = 20) -> Any:
    arr = _finite_metric_values(values)
    if len(arr) < 2:
        return Text("-", style="yellow")
    recent = arr[-min(window, len(arr)) :]
    delta = float(recent[-1] - recent[0])
    improving = delta < 0 if lower_is_better else delta > 0
    style = "green" if improving else "red" if abs(delta) > 1e-9 else "yellow"
    sign = "+" if delta > 0 else ""
    return Text(f"{sign}{qfloat(delta)}", style=style)


def _metric_color(values: list[float], lower_is_better: bool) -> str:
    label = _trend_label(values[-20:], lower_is_better)
    if label == "improving":
        return "green"
    if label == "worsening":
        return "red"
    return "yellow"


def _finite_metric_values(values: list[float]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def render_report_html(summary: dict[str, Any], metrics: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    candidate_rows = sorted(
        [event for event in events if event.get("type") in {"candidate_score", "full_kpi_score", "dataset_complete"} and isinstance(event.get("rbcs"), (int, float))],
        key=lambda event: float(event.get("rbcs") or 0.0),
        reverse=True,
    )[:25]
    checkpoints = [event for event in events if event.get("type") == "checkpoint"]
    kernel_rows = summary.get("kernel_summaries", []) if isinstance(summary.get("kernel_summaries"), list) else []
    selected = summary.get("selected_datasets", []) if isinstance(summary.get("selected_datasets"), list) else []
    final = summary.get("final", {}) if isinstance(summary.get("final"), dict) else {}
    config = {
        key: summary.get(key)
        for key in (
            "preset",
            "seed",
            "all_datasets",
            "max_datasets",
            "epochs",
            "batch_size",
            "row_cap",
            "max_seconds",
            "checkpoint_every",
            "full_kpi_every",
            "full_kpi_top_k",
            "early_stop_patience",
            "early_stop_min_delta",
            "synthetic_mode",
            "real_batch_fraction",
            "synthetic_batch_fraction",
            "best_val_loss",
            "best_epoch",
            "epochs_since_best",
            "stopped_by",
            "elapsed_seconds",
        )
    }
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Dope Kernel Tuner Report</title>",
            "<style>",
            "body{font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#172026;background:#f7f9fb;}",
            "section{margin:0 0 24px 0;padding:16px;background:#fff;border:1px solid #d9e2ec;border-radius:8px;}",
            "h1,h2{margin:0 0 12px 0;} table{border-collapse:collapse;width:100%;font-size:13px;} th,td{border-bottom:1px solid #e6edf3;padding:6px 8px;text-align:left;vertical-align:top;}",
            "th{background:#eef5f9;} code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;} pre{white-space:pre-wrap;word-break:break-word;background:#0f1720;color:#d7e4ee;padding:10px;border-radius:6px;}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.muted{color:#52616f}.curve{max-width:100%;height:120px;background:#fbfdff;border:1px solid #e6edf3;border-radius:6px;}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Dope Kernel Tuner Report</h1>",
            f'<p class="muted">Generated {html.escape(datetime.now(timezone.utc).isoformat())}</p>',
            "<section><h2>Run Config</h2>" + _html_table([config], list(config.keys())) + "</section>",
            "<section><h2>Final Metrics</h2>" + _html_table([final], list(final.keys())[:16]) + "</section>",
            '<section><h2>Loss / KPI Curves</h2><div class="grid">'
            + _curve_block("optimization_loss", [row.get("optimization_loss") for row in metrics])
            + _curve_block("train_eval_loss", [row.get("train_eval_loss", row.get("train_loss")) for row in metrics])
            + _curve_block("val_eval_loss", [row.get("val_eval_loss", row.get("val_loss")) for row in metrics])
            + _curve_block("train_ranking_nll", [row.get("train_ranking_nll") for row in metrics])
            + _curve_block("val_ranking_nll", [row.get("val_ranking_nll") for row in metrics])
            + _curve_block("train_rank_accuracy", [row.get("train_rank_accuracy") for row in metrics])
            + _curve_block("val_rank_accuracy", [row.get("val_rank_accuracy", row.get("rank_accuracy")) for row in metrics])
            + _curve_block("loss_gap", [row.get("loss_gap") for row in metrics])
            + _curve_block("candidate_rbcs", [row.get("rbcs") for row in candidate_rows])
            + "</div></section>",
            "<section><h2>Selected Datasets</h2>" + _html_table(selected, ["dataset_id", "task", "path"]) + "</section>",
            "<section><h2>Candidate Leaderboard</h2>"
            + _html_table(candidate_rows, ["dataset", "candidate_index", "score_mode", "rbcs", "fidelity_real", "kernel_bytes", "full_kpi_delta", "config"])
            + "</section>",
            "<section><h2>Best Kernel Summaries</h2>" + _kernel_summary_table(kernel_rows) + "</section>",
            "<section><h2>Checkpoints</h2>" + _html_table(checkpoints, ["time", "checkpoint_kind", "epoch", "step", "path"]) + "</section>",
            "</body></html>",
        ]
    )


def _curve_block(name: str, values: list[Any]) -> str:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return f"<div><strong>{html.escape(name)}</strong>{_svg_curve(numeric)}</div>"


def _svg_curve(values: list[float], width: int = 520, height: int = 120) -> str:
    if not values:
        return '<div class="curve muted">no data</div>'
    lo = min(values)
    hi = max(values)
    spread = hi - lo if hi > lo else 1.0
    points = []
    count = max(1, len(values) - 1)
    for idx, value in enumerate(values):
        x = idx / count * (width - 16) + 8
        y = height - 8 - ((value - lo) / spread * (height - 16))
        points.append(f"{x:.2f},{y:.2f}")
    return (
        f'<svg class="curve" viewBox="0 0 {width} {height}" role="img" aria-label="metric curve">'
        f'<polyline fill="none" stroke="#0f766e" stroke-width="2" points="{" ".join(points)}"/>'
        f'<text x="8" y="16" fill="#52616f" font-size="11">min {lo:.4g} max {hi:.4g}</text></svg>'
    )


def _html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return '<p class="muted">no records</p>'
    columns = [column for column in columns if any(column in row for row in rows)]
    if not columns:
        columns = sorted({key for row in rows for key in row.keys()})[:12]
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_html_value(row.get(column))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _kernel_summary_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">no kernel summaries</p>'
    table_rows = []
    for row in rows[:16]:
        dataset = row.get("dataset") if isinstance(row.get("dataset"), dict) else {}
        table_rows.append(
            {
                "dataset_id": dataset.get("dataset_id") if isinstance(dataset, dict) else None,
                "task": dataset.get("task") if isinstance(dataset, dict) else None,
                "candidate_index": row.get("candidate_index"),
                "score_mode": row.get("score_mode"),
                "rbcs": row.get("rbcs"),
                "kernel_bytes": row.get("kernel_bytes"),
                "dependence": row.get("dependence"),
                "target": row.get("target"),
                "marginals": row.get("marginals"),
                "program": row.get("canonical_program"),
            }
        )
    return _html_table(table_rows, ["dataset_id", "task", "candidate_index", "score_mode", "rbcs", "kernel_bytes", "dependence", "target", "marginals", "program"])


def _html_value(value: Any) -> str:
    if isinstance(value, float):
        return html.escape(f"{value:.6g}")
    if isinstance(value, dict):
        dataset_id = value.get("dataset_id")
        if dataset_id is not None and set(value.keys()) <= {"path", "task", "dataset_id"}:
            return html.escape(str(dataset_id))
        return "<code>" + html.escape(json.dumps(value, sort_keys=True)) + "</code>"
    if isinstance(value, list):
        return "<code>" + html.escape(json.dumps(value)) + "</code>"
    if value is None:
        return '<span class="muted">-</span>'
    text = str(value)
    if len(text) > 360:
        text = text[:357] + "..."
    return html.escape(text)
