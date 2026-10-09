"""Cohort, shared map, and median-of-three null for the MFS-v3 panel.

The null tables and their linear losses are fixed before any foundation-model
forward. Official test files are refused. A caller cannot pass a distance into
the null selection.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from research.benchmark.representation import (
    NORMALIZER,
    REFUSED_ENCODERS,
    TABULAR_AUDITORS,
    TABULAR_PROTOCOL,
    artifact_has_cleartext,
    column_hash,
    midrank_quantile,
    salt_id,
)

FIT_SEED = 11
SIZE = 4
SAMPLE_SEEDS = (101, 211, 307)
DENSITY_SPECS = (
    ("DOPE", "features12_steps2048"),
    ("GaussianCopula", "native_selected"),
    ("Chow-Liu", "native_selected"),
    ("independent_marginals", "native_selected"),
)
VECTOR_PIN = {
    "kumo_tabular_l": "row_project_output_mean_labeled_rows",
    "mitra_v2": "final_layer_norm_input_mean_self_query",
    "tabicl2": "tf_icl_output_mean_self_query",
}
KUMO_ESTIMATORS = 16
MODEL_FILES = {
    "kumo_tabular_l": "models/nvidia__Kumo-Tabular/large/regressor.pt",
    "mitra_v2": "models/autogluon__mitra-regressor-2/model.safetensors",
    "tabicl2": "models/jingang__TabICL/tabicl-regressor-v2-20260212.ckpt",
}
WEIGHT_ROOT = Path("/home/ubuntu/jopedime_probe_exp/ofm-20261008")
INFORMATIVE_FRACTION = 0.01
BYTE_CAP = 10240
VRAM_START_MIB = 10240
VRAM_STOP_MIB = 8192
OTHER_GROWTH_MIB = 2048
MEM_AVAILABLE_FLOOR = 0.15
_INTERNAL_COLUMN = re.compile(r"c\d+")


def refuse_test_path(path: Path) -> None:
    if path.name == "test.csv" or "test.csv" in path.parts:
        raise ValueError("official test files stay sealed")


def load_numeric_table(path: Path) -> np.ndarray:
    refuse_test_path(Path(path))
    table = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError("a mapped table needs rows and a target column")
    if not np.isfinite(table).all():
        raise ValueError("a mapped table must be finite")
    return table


def file_sha256(path: Path) -> str:
    refuse_test_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_pointer(cell: dict) -> tuple[Path, str] | None:
    for key in ("sample_evidence", "metric_receipt"):
        block = cell.get(key)
        if not isinstance(block, dict):
            continue
        metric = block.get("metric_path")
        digest = block.get("sample_sha256")
        if not isinstance(metric, str) or not isinstance(digest, str):
            continue
        metric_path = Path(metric)
        if not metric_path.name.endswith(".metric.json"):
            continue
        csv_path = metric_path.with_name(metric_path.name[: -len(".metric.json")] + ".csv")
        refuse_test_path(csv_path)
        return csv_path, digest
    return None


def _catboost_informative(cell: dict) -> bool:
    utility = (cell.get("utility") or {}).get("catboost") or {}
    retention = utility.get("retention")
    return (
        cell.get("status") == "ok"
        and utility.get("informative") is True
        and isinstance(retention, (int, float))
        and not isinstance(retention, bool)
    )


def density_lineages(record: dict) -> list[str]:
    """Complete paired lineages, the same informative rule as the CatBoost table."""
    grouped: dict[tuple[str, str], dict[str, dict[int, dict]]] = {}
    for method, configuration in DENSITY_SPECS:
        grouped[(method, configuration)] = {}
    for cell in record["cells"]:
        key = (cell.get("method"), cell.get("configuration"))
        if key not in grouped:
            continue
        if cell.get("fit_seed") != FIT_SEED or cell.get("size_multiplier") != SIZE:
            continue
        if cell.get("sample_seed") not in SAMPLE_SEEDS:
            continue
        grouped[key].setdefault(cell["dataset"], {})[cell["sample_seed"]] = cell
    present = None
    for key in DENSITY_SPECS:
        datasets = {
            dataset
            for dataset, by_seed in grouped[key].items()
            if set(by_seed) == set(SAMPLE_SEEDS)
            and all(_catboost_informative(by_seed[seed]) for seed in SAMPLE_SEEDS)
        }
        present = datasets if present is None else present & datasets
    return sorted(present or ())


def density_cells(record: dict, lineages: list[str]) -> list[dict]:
    wanted = set(lineages)
    found = []
    for cell in record["cells"]:
        if dict(DENSITY_SPECS).get(cell.get("method")) != cell.get("configuration"):
            continue
        if cell.get("dataset") not in wanted:
            continue
        if cell.get("fit_seed") != FIT_SEED or cell.get("size_multiplier") != SIZE:
            continue
        if cell.get("sample_seed") not in SAMPLE_SEEDS or cell.get("status") != "ok":
            continue
        pointer = sample_pointer(cell)
        if pointer is None:
            continue
        csv_path, digest = pointer
        fit_receipt = cell.get("fit_receipt")
        if isinstance(fit_receipt, dict) and isinstance(fit_receipt.get("path"), str):
            fit_receipt_path = fit_receipt["path"]
        elif isinstance(cell.get("fit_attempt_receipt_path"), str):
            fit_receipt_path = cell["fit_attempt_receipt_path"]
        else:
            fit_receipt_path = None
        found.append(
            {
                "artifact_bytes": cell.get("charged_artifact_bytes"),
                "block": "density",
                "fit_receipt_path": fit_receipt_path,
                "configuration": cell["configuration"],
                "counts_as_dope_win": False,
                "dataset": cell["dataset"],
                "fit_seed": FIT_SEED,
                "method": cell["method"],
                "official_tests_opened": False,
                "sample_csv": str(csv_path),
                "sample_seed": cell["sample_seed"],
                "sample_sha256": digest,
                "size_multiplier": SIZE,
                "superiority": None,
                "worker_dir": f"/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker/{cell['dataset']}",
            }
        )
    found.sort(key=lambda item: (item["dataset"], item["method"], item["sample_seed"]))
    return found


def cohort_document(record: dict) -> dict:
    lineages = density_lineages(record)
    return {
        "auditors": list(TABULAR_AUDITORS),
        "block": "density",
        "cells": density_cells(record, lineages),
        "counts_as_dope_win": False,
        "fit_seed": FIT_SEED,
        "format": "dope-mfs-v3-panel-cohort",
        "lineages": lineages,
        "model_files": MODEL_FILES,
        "normalizer": NORMALIZER,
        "official_tests_opened": False,
        "protocol": TABULAR_PROTOCOL,
        "refused_encoders": list(REFUSED_ENCODERS),
        "sample_seeds": list(SAMPLE_SEEDS),
        "size_multiplier": SIZE,
        "superiority": None,
        "vector_pin": VECTOR_PIN,
        "version": 1,
    }


def verify_cell_hash(cell: dict) -> bool:
    path = Path(cell["sample_csv"])
    refuse_test_path(path)
    return file_sha256(path) == cell["sample_sha256"]


def fit_map(fit: np.ndarray) -> list[np.ndarray]:
    if not np.isfinite(fit).all() or fit.ndim != 2:
        raise ValueError("the map is fit on finite rows")
    return [np.sort(fit[:, column].astype(np.float64)) for column in range(fit.shape[1])]


def apply_map(fitted: list[np.ndarray], table: np.ndarray) -> np.ndarray:
    if table.shape[1] != len(fitted):
        raise ValueError("the shared map has one fit column per table column")
    mapped = np.empty(table.shape, dtype=np.float64)
    for column, ordered in enumerate(fitted):
        values = midrank_quantile(ordered.tolist(), table[:, column].astype(np.float64).tolist())
        if values is None:
            raise ValueError("the shared map rejected a non-finite column")
        mapped[:, column] = values
    return mapped


def null_seed(dataset: str, method: str, configuration: str, sample_seed: int, index: int) -> int:
    text = f"{dataset}|{method}|{configuration}|{sample_seed}|{index}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def independent_table(fit: np.ndarray, rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    drawn = np.empty((rows, fit.shape[1]), dtype=np.float64)
    choice = rng.integers(0, fit.shape[0], size=(rows, fit.shape[1]))
    for column in range(fit.shape[1]):
        drawn[:, column] = fit[choice[:, column], column]
    return drawn


def linear_mse(context: np.ndarray, query: np.ndarray) -> float:
    x_fit, y_fit = context[:, :-1], context[:, -1]
    x_query, y_query = query[:, :-1], query[:, -1]
    design = np.column_stack([np.ones(len(x_fit)), x_fit])
    coefficient, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
    prediction = np.column_stack([np.ones(len(x_query)), x_query]) @ coefficient
    return float(np.mean((prediction - y_query) ** 2))


def build_nulls(fit: np.ndarray, holdout: np.ndarray, rows: int, seeds: tuple[int, int, int]):
    """Return three nulls, their linear losses, and the matched index.

    The first table is the independent null. The matched null is the median
    linear loss. Losses are computed here, before any encoder runs.
    """
    from research.benchmark.mfs_v3_score import _v3_contract

    specification = _v3_contract()["representation"]
    if (specification["matched_null"] != "median_linear_mse_of_three"
            or specification["matched_null_selection"]["candidate_indices"] != [0, 1, 2]
            or len(seeds) != 3 or len(set(seeds)) != 3):
        raise ValueError("matched null requires three declared distinct candidates")
    tables = [independent_table(fit, rows, seed) for seed in seeds]
    losses = [linear_mse(table, holdout) for table in tables]
    if not np.isfinite(losses).all():
        raise ValueError("matched null losses must be finite")
    match_index = sorted(range(3), key=lambda index: (losses[index], index))[1]
    return tables, losses, 0, match_index


def exact_row_matches(fit: np.ndarray, synthetic: np.ndarray) -> int:
    keys = {row.tobytes() for row in np.ascontiguousarray(fit)}
    return sum(row.tobytes() in keys for row in np.ascontiguousarray(synthetic))


def _squared_nearest(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    nearest = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], 256):
        block = query[start : start + 256]
        delta = block[:, None, :] - reference[None, :, :]
        nearest[start : start + len(block)] = np.min(np.sum(delta * delta, axis=2), axis=1)
    return nearest


def distinct_nearest_quantile_squared(fit: np.ndarray) -> float | None:
    """Declared quantile of nearest-other distances among distinct real rows."""
    from research.benchmark.mfs_v3_score import _v3_contract

    gates = _v3_contract()["release_gates"]
    quantile = gates["near_copy_quantile"]
    if (gates["near_copy_floor"] != "real_distinct_row_quantile"
            or gates["near_copy_quantile_interpolation"] != "linear"
            or type(quantile) not in (int, float) or not 0 <= quantile <= 1):
        raise ValueError("invalid declared near-copy quantile")
    if fit.ndim != 2 or not np.isfinite(fit).all():
        return None
    unique = np.unique(np.ascontiguousarray(fit), axis=0)
    if unique.shape[0] < 2:
        return None
    nearest = []
    for index in range(unique.shape[0]):
        delta = unique[index] - unique
        distance = np.sum(delta * delta, axis=1)
        distance[index] = np.inf
        nearest.append(float(distance.min()))
    return float(np.quantile(nearest, quantile, method="linear"))


def near_copy_ok(fit: np.ndarray, synthetic: np.ndarray) -> bool:
    """Require every synthetic distance to meet the contract's real-row quantile."""
    if synthetic.ndim != 2 or not len(synthetic) or not np.isfinite(synthetic).all():
        return False
    floor = distinct_nearest_quantile_squared(fit)
    if floor is None or not np.isfinite(floor):
        return False
    return bool(np.all(_squared_nearest(fit, synthetic) >= floor))


def retention(null_loss: float, real_loss: float, synthetic_loss: float) -> float | None:
    gap = null_loss - real_loss
    if gap < INFORMATIVE_FRACTION * abs(null_loss) or gap == 0.0:
        return None
    return (null_loss - synthetic_loss) / gap


def _projection_names(projection_path: Path) -> list[str] | None:
    refuse_test_path(projection_path)
    if not projection_path.is_file():
        return None
    payload = json.loads(projection_path.read_text())
    names = [name for name in list(payload.get("input_columns") or []) if isinstance(name, str)]
    target = payload.get("target")
    if isinstance(target, str):
        names.append(target)
    return names or None


def column_name_hashes(projection_path: Path, salt: bytes) -> list[str] | None:
    names = _projection_names(projection_path)
    if names is None:
        return None
    return [column_hash(salt, name) for name in names]


def cleartext_absent(artifact: Path, projection_path: Path) -> bool | None:
    """Scan for source text. Internal projection codes are not source text."""
    refuse_test_path(artifact)
    names = _projection_names(projection_path)
    if names is None or not artifact.is_file():
        return None
    needles = [name.encode() for name in names if not _INTERNAL_COLUMN.fullmatch(name)]
    if not needles:
        return None
    return not artifact_has_cleartext(artifact.read_bytes(), needles)


def gpu_stop_reason(
    free_mib: float,
    mem_available_fraction: float,
    other_used_mib: float,
    other_baseline_mib: float,
) -> str | None:
    if free_mib < VRAM_STOP_MIB:
        return "free_vram"
    if mem_available_fraction < MEM_AVAILABLE_FLOOR:
        return "mem_available"
    if other_used_mib > other_baseline_mib + OTHER_GROWTH_MIB:
        return "other_grew"
    return None


def gpu_may_start(free_mib: float, mem_available_fraction: float) -> bool:
    return free_mib >= VRAM_START_MIB and mem_available_fraction >= MEM_AVAILABLE_FLOOR


def public_salt_id(salt: bytes) -> str:
    return salt_id(salt)


def pilot_cells(cells: list[dict], lineages: list[str]) -> list[dict]:
    """Two lineages, four density methods, sample seed 101."""
    chosen = set(lineages[:2])
    return [
        cell
        for cell in cells
        if cell["dataset"] in chosen and cell["sample_seed"] == SAMPLE_SEEDS[0]
    ]
