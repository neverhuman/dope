from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .utils import clip01, infer_task_from_target, read_headerless_csv


def split_xy(data: np.ndarray, task: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    data = clip01(data)
    if data.shape[1] < 2:
        raise ValueError("datasets must contain at least one feature and one target column")
    X = data[:, :-1]
    y = data[:, -1]
    task = infer_task_from_target(y) if task is None else task
    if task == "binary":
        y = (y >= 0.5).astype(float)
    return X, y, task


def load_train(dataset_dir: str | Path, task: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing train.csv in {dataset_dir}")
    return split_xy(read_headerless_csv(path), task)


def load_optional_test(dataset_dir: str | Path, task: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path(dataset_dir) / "test.csv"
    if not path.exists():
        return None
    X, y, _ = split_xy(read_headerless_csv(path), task)
    return X, y


def canonical_feature_order(X: np.ndarray) -> np.ndarray:
    """Target-free anonymous permutation by distributional column statistics."""
    X = clip01(X)
    keys = []
    n = max(1, X.shape[0])
    for j in range(X.shape[1]):
        col = X[:, j]
        quantiles = np.quantile(col, [0.1, 0.25, 0.5, 0.75, 0.9])
        key = (
            round(float(np.mean(col)), 6),
            round(float(np.std(col)), 6),
            round(float(np.mean(col <= 1e-9)), 6),
            round(float(np.mean(col >= 1.0 - 1e-9)), 6),
            round(float(len(np.unique(np.round(col, 6))) / n), 6),
            *(round(float(q), 6) for q in quantiles),
            j,
        )
        keys.append((key, j))
    return np.array([j for _, j in sorted(keys)], dtype=int)


def apply_order(X: np.ndarray, order: np.ndarray) -> np.ndarray:
    return clip01(X)[:, order]


def load_real_split(dataset_dir: str | Path, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train, _ = load_train(dataset_dir, task)
    order = canonical_feature_order(X_train)
    X_train = apply_order(X_train, order)
    maybe_test = load_optional_test(dataset_dir, task)
    if maybe_test is not None:
        X_test, y_test = maybe_test
        return X_train, y_train, apply_order(X_test, order), y_test, order

    stratify = y_train if task == "binary" and len(np.unique(y_train)) > 1 else None
    X_fit, X_test, y_fit, y_test = train_test_split(
        X_train,
        y_train,
        test_size=max(1, int(round(0.25 * len(y_train)))) / len(y_train),
        random_state=1729,
        stratify=stratify,
    )
    return X_fit, y_fit, X_test, y_test, order
