"""Transfer-in-the-loop backend selection, inside the fit rows only.

DOPE's built-in tournament ranks candidates by train-only proxies (stored-target fit,
driver agreement, a joint-fidelity proxy). Those proxies do not see transfer, and the
best backend differs from table to table. This module chooses among a small portfolio
of DOPE backends by the quantity we actually want: K-fold train-on-synthetic,
test-on-held-out-fit-rows retention, with two cheap auditors from different families.

No holdout or test row is read. The selected backend is then compiled on all fit rows
by the caller. See docs/math.md (section 7, R1) for why this should help and
examples/quickstart/results/ablation.md for what it did on two open tables.
"""

from __future__ import annotations

import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dope_tables import TableEncoder, read_dope_samples

PORTFOLIO = ("chow_liu", "compact_neural_residual_symbolic", "sparse_gaussian_copula", "search")


@dataclass
class Selection:
    backend: str
    scores: dict[str, float]
    model_bytes: dict[str, float]


def _quick_score(train: pd.DataFrame, synthetic: pd.DataFrame, valid: pd.DataFrame,
                 target: str, task: str, positive) -> float:
    """Worst of a linear and a gradient-boosted auditor's retention on held-out fit rows."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    categorical = [c for c in train.columns if c != target and not pd.api.types.is_numeric_dtype(train[c])]
    numeric = [c for c in train.columns if c != target and c not in categorical]
    binary = task == "binary"

    def frame(df):
        X = df.drop(columns=[target]).copy()
        for column in categorical:
            X[column] = X[column].astype(object).where(X[column].notna(), "__missing__").astype(str)
        return X

    def labels(df):
        return (df[target] == positive).to_numpy(dtype=int) if binary else df[target].to_numpy(dtype=float)

    def loss(truth, prediction):
        if binary:
            p = np.clip(prediction, 1e-6, 1 - 1e-6)
            return float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p)))
        return float(np.mean((truth - prediction) ** 2))

    def linear():
        prep = ColumnTransformer([
            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
             categorical)])
        return make_pipeline(prep, LogisticRegression(max_iter=1000) if binary else Ridge(alpha=1.0))

    def boosted():
        prep = ColumnTransformer([
            ("n", "passthrough", numeric),
            ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                 encoded_missing_value=-1), categorical)])
        model = (HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor)(
            max_iter=150, random_state=0)
        return make_pipeline(prep, model)

    y_valid = labels(valid)
    y_train = labels(train)
    null = loss(y_valid, np.full(y_valid.shape, y_train.mean()))
    X_valid = frame(valid)
    scores = []
    for build in (linear, boosted):
        losses = []
        for table in (train, synthetic):
            y = labels(table)
            if binary and np.unique(y).size < 2:
                losses.append(loss(y_valid, np.full(y_valid.shape, float(y.mean()))))
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = build().fit(frame(table), y)
                prediction = model.predict_proba(X_valid)[:, 1] if binary else model.predict(X_valid)
            losses.append(loss(y_valid, np.asarray(prediction, dtype=float)))
        gain = null - losses[0]
        if gain > 0.01 * abs(null):
            scores.append((null - losses[1]) / gain)
    return min(scores) if scores else float("nan")


def select_backend(kernel: str, fit: pd.DataFrame, target: str, *, positive=None, encoder_options=None,
                   portfolio=PORTFOLIO, folds: int = 3, seed: int = 11) -> Selection:
    """Pick the portfolio backend with the best mean K-fold transfer score on the fit rows."""
    encoder_options = dict(encoder_options or {})
    order = np.random.default_rng(seed).permutation(len(fit))
    parts = np.array_split(order, folds)
    scores = {backend: [] for backend in portfolio}
    sizes = {backend: [] for backend in portfolio}
    with tempfile.TemporaryDirectory() as tmp:
        for fold, held in enumerate(parts):
            keep = np.setdiff1d(order, held)
            train = fit.iloc[keep].reset_index(drop=True)
            valid = fit.iloc[held].reset_index(drop=True)
            encoder = TableEncoder(target, positive=positive, **encoder_options).fit(train)
            work = Path(tmp) / f"fold{fold}"
            encoder.write_dataset_dir(train, work)
            for backend in portfolio:
                artifact = work / f"{backend}.dpk"
                command = [kernel, "compile", "--dataset-dir", str(work), "--task", encoder.task,
                           "--out", str(artifact), "--seed", str(seed), "--deadline-seconds", "120"]
                if backend != "search":
                    command += ["--candidate", backend]
                if subprocess.run(command, capture_output=True).returncode != 0:
                    continue
                samples = work / f"{backend}.csv"
                if subprocess.run([kernel, "sample", "--kernel", str(artifact), "--rows", str(len(train)),
                                   "--out", str(samples), "--seed", str(7 + fold)],
                                  capture_output=True).returncode != 0:
                    continue
                synthetic = encoder.inverse_transform(read_dope_samples(samples), seed=seed + fold)
                score = _quick_score(train, synthetic, valid, target, encoder.task, positive
                                     if encoder.task != "binary" else encoder.classes[1])
                if np.isfinite(score):
                    scores[backend].append(score)
                    sizes[backend].append(artifact.stat().st_size)
    mean = {b: float(np.mean(v)) for b, v in scores.items() if len(v) == folds}
    if not mean:
        raise RuntimeError("no portfolio backend completed every fold")
    best = max(mean, key=lambda b: (round(mean[b], 3), -float(np.mean(sizes[b]))))
    return Selection(best, mean, {b: float(np.mean(v)) for b, v in sizes.items() if v})
