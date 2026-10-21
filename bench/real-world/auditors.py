"""Auditor families for retention: linear, boosted trees (CatBoost), MLP.

Each auditor is trained on one table (real or synthetic) and scored on the
same held-out real test rows. Preprocessing is fitted on the auditor's own
training table: median imputation and standardization for numeric
columns, one-hot encoding (unknown levels ignored) for categorical ones.
Losses: MSE in original target units (regression), log-loss
(classification), as in the TDL definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FAMILIES = ("linear", "gbdt", "mlp")


def _features(train: pd.DataFrame) -> ColumnTransformer:
    cat = [c for c in train.columns if train[c].dtype == object]
    num = [c for c in train.columns if c not in cat]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
    ], sparse_threshold=0.0)


def _model(family: str, task: str, seed: int, n: int):
    if family == "linear":
        return Ridge(alpha=1.0) if task == "regression" else LogisticRegression(C=1.0, max_iter=3000)
    if family == "gbdt":
        from catboost import CatBoostClassifier, CatBoostRegressor
        kw = dict(iterations=500, depth=6, random_seed=seed, verbose=0, thread_count=2, allow_writing_files=False)
        return CatBoostRegressor(**kw) if task == "regression" else CatBoostClassifier(**kw)
    if family == "mlp":
        # Small tables: full-batch L-BFGS with weight decay (Adam with a 10%
        # early-stopping split is unstable below a few thousand rows).
        if n < 2000:
            kw = dict(hidden_layer_sizes=(128, 128), solver="lbfgs", alpha=1e-2, max_iter=300, random_state=seed)
        else:
            kw = dict(hidden_layer_sizes=(128, 128), early_stopping=True, max_iter=150, random_state=seed)
        return MLPRegressor(**kw) if task == "regression" else MLPClassifier(**kw)
    raise ValueError(family)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).where(df[c].notna(), np.nan)
    return df


def predict(family: str, task: str, train: pd.DataFrame, target: str, test: pd.DataFrame,
            classes: list, seed: int) -> np.ndarray:
    """Fit on `train`, return predictions (regression) or class probabilities
    aligned to `classes` (classification) for `test`."""
    train = train.dropna(subset=[target])
    x, y = _clean(train.drop(columns=[target])), train[target]
    xt = _clean(test.drop(columns=[target]))
    if task == "classification":
        y = y.astype(str)
        present = sorted(y.unique())
        if len(present) < 2:  # a degenerate table predicts its only class
            p = np.full((len(test), len(classes)), 1e-7)
            p[:, classes.index(present[0]) if present and present[0] in classes else 0] = 1.0
            return p / p.sum(1, keepdims=True)
    else:
        y = y.astype(float)
    pre = _features(x)
    model = make_pipeline(pre, _model(family, task, seed, len(x)))
    model.fit(x, y)
    if task == "regression":
        return model.predict(xt)
    proba = model.predict_proba(xt)
    out = np.full((len(test), len(classes)), 1e-7)
    for k, c in enumerate(model.classes_):
        if c in classes:
            out[:, classes.index(c)] = proba[:, k]
    return out / out.sum(1, keepdims=True)


def loss(task: str, pred: np.ndarray, y: pd.Series, classes: list) -> float:
    if task == "regression":
        return float(np.mean((pred - y.astype(float).values) ** 2))
    idx = np.array([classes.index(v) for v in y.astype(str)])
    p = np.clip(pred[np.arange(len(idx)), idx], 1e-7, 1.0)
    return float(-np.mean(np.log(p)))


def null_loss(task: str, train: pd.DataFrame, target: str, test: pd.DataFrame, classes: list) -> float:
    y = train[target].dropna()
    if task == "regression":
        return loss(task, np.full(len(test), y.astype(float).mean()), test[target], classes)
    freq = y.astype(str).value_counts()
    prior = np.array([freq.get(c, 0) + 0.5 for c in classes], float)
    prior /= prior.sum()
    return loss(task, np.tile(prior, (len(test), 1)), test[target], classes)
