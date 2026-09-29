from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9
FLOAT_DECIMALS = 6


def clip01(values: Any) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def qfloat(value: Any, decimals: int = FLOAT_DECIMALS) -> float:
    value = float(value)
    if not math.isfinite(value):
        value = 0.0
    rounded = round(value, decimals)
    return 0.0 if rounded == -0.0 else rounded


def qlist(values: Any, decimals: int = FLOAT_DECIMALS) -> list[float]:
    return [qfloat(v, decimals) for v in np.asarray(values, dtype=float).ravel()]


def qmatrix(values: Any, decimals: int = FLOAT_DECIMALS) -> list[list[float]]:
    array = np.asarray(values, dtype=float)
    return [[qfloat(v, decimals) for v in row] for row in array]


def canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_bytes(obj: Any) -> bytes:
    return canonical_dumps(obj).encode("utf-8")


def stable_hash_bytes(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def stable_dataset_seed(array: np.ndarray, default_seed: int = 1729) -> int:
    payload = np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"dope-dk").digest()
    return sklearn_seed(int.from_bytes(digest, "little") ^ int(default_seed))


def sklearn_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)


def write_headerless_csv(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(clip01(array))
    df.to_csv(path, header=False, index=False, float_format="%.10g")


def read_headerless_csv(path: str | Path) -> np.ndarray:
    data = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if data.ndim != 2:
        raise ValueError(f"{path} must be a 2D headerless CSV")
    if data.shape[1] < 1:
        raise ValueError(f"{path} must contain at least one column")
    return clip01(data)


def normalize_task(task: str) -> str:
    task = task.lower().strip()
    aliases = {
        "reg": "regression",
        "regression": "regression",
        "binary": "binary",
        "bin": "binary",
        "classification": "binary",
        "clf": "binary",
    }
    if task not in aliases:
        raise ValueError("task must be regression or binary")
    return aliases[task]


def infer_task_from_target(y: np.ndarray) -> str:
    y = clip01(y)
    rounded = np.round(y, 6)
    unique = np.unique(rounded)
    if len(unique) <= 2 and np.all(np.isin(unique, [0.0, 1.0])):
        return "binary"
    return "regression"


def bits_for_jsonish(value: Any) -> int:
    """Deterministic MDL-style bit estimate for quantized parameters."""
    if isinstance(value, dict):
        return 8 * sum(len(str(k)) for k in value) + sum(bits_for_jsonish(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(bits_for_jsonish(v) for v in value)
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(1, int(math.ceil(math.log2(abs(value) + 2))))
    if isinstance(value, float):
        return 16
    if value is None:
        return 1
    return 8 * len(str(value))


def finite_corrcoef(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[1] == 1:
        return np.ones((1, 1), dtype=float)
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def nearest_correlation(matrix: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, min_eig)
    repaired = (vecs * vals) @ vecs.T
    diag = np.sqrt(np.maximum(np.diag(repaired), min_eig))
    repaired = repaired / np.outer(diag, diag)
    repaired = np.clip((repaired + repaired.T) / 2.0, -0.999, 0.999)
    np.fill_diagonal(repaired, 1.0)
    return repaired
