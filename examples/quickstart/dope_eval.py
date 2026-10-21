"""Retention: how much of a real-trained model's gain a synthetic-trained model keeps.

For an auditor (a learning recipe) and a holdout of untouched real rows:

    retention = (L_null - L_synthetic) / (L_null - L_real)

``L_null`` is the loss of a predictor that learns nothing from the table (the fit-row
mean for regression, the fit-row base rate for a binary target), ``L_real`` the loss of
the auditor trained on the real fit rows, and ``L_synthetic`` the loss of the same
recipe trained only on synthetic rows. 1.0 means the synthetic rows taught the model
everything the real rows did; 0 means nothing; values below 0 mean the synthetic rows
were harmful. A cell whose real gain is under 1% of the null loss is uninformative and
reported as ``None``, never as 0.

Losses are mean squared error in original target units (regression) and log loss
(binary). This is the same estimand as the paper's retention; the recipes here are
smaller and are a local check, not the benchmark.
"""

from __future__ import annotations

import importlib.util
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

INFORMATIVE_FRACTION = 0.01


@dataclass
class AuditorResult:
    auditor: str
    null_loss: float
    real_loss: float
    synthetic_loss: float

    @property
    def retention(self) -> float | None:
        gain = self.null_loss - self.real_loss
        if not gain > INFORMATIVE_FRACTION * abs(self.null_loss):
            return None
        return (self.null_loss - self.synthetic_loss) / gain


def _has_catboost() -> bool:
    return importlib.util.find_spec("catboost") is not None


def default_auditors() -> tuple[str, ...]:
    """Three families, as in the paper: boosted trees, linear, neural network."""
    return ("catboost" if _has_catboost() else "gbdt", "linear", "mlp")


def _auditor(name: str, task: str, categorical: list[str], numeric: list[str], seed: int):
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    binary = task == "binary"
    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        model = (CatBoostClassifier if binary else CatBoostRegressor)(
            iterations=400, depth=6, learning_rate=0.08, random_seed=seed, verbose=0,
            thread_count=2, allow_writing_files=False)
        return ("catboost", model)
    dense = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    onehot = make_pipeline(SimpleImputer(strategy="most_frequent"),
                           OneHotEncoder(handle_unknown="ignore", min_frequency=5))
    if name == "gbdt":
        prep = ColumnTransformer([
            ("num", "passthrough", numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1), categorical)])
        model = (HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor)(
            max_iter=300, learning_rate=0.08, random_state=seed)
        return ("sklearn", make_pipeline(prep, model))
    prep = ColumnTransformer([("num", dense, numeric), ("cat", onehot, categorical)])
    if name == "linear":
        model = LogisticRegression(max_iter=2000) if binary else Ridge(alpha=1.0)
        return ("sklearn", make_pipeline(prep, model))
    if name == "mlp":
        if binary:
            model = MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, max_iter=400, random_state=seed)
        else:
            model = TransformedTargetRegressor(
                MLPRegressor(hidden_layer_sizes=(64,), early_stopping=True, max_iter=400, random_state=seed),
                transformer=StandardScaler())
        return ("sklearn", make_pipeline(prep, model))
    raise ValueError(f"unknown auditor {name!r}")


def _frame(df: pd.DataFrame, target: str, categorical: list[str]) -> pd.DataFrame:
    X = df.drop(columns=[target]).copy()
    for column in categorical:
        X[column] = X[column].astype(object).where(X[column].notna(), "__missing__").astype(str)
    return X


def _labels(df: pd.DataFrame, target: str, task: str, positive) -> np.ndarray:
    if task == "binary":
        return (df[target] == positive).to_numpy(dtype=int)
    return df[target].to_numpy(dtype=float)


def _loss(task: str, truth: np.ndarray, prediction: np.ndarray) -> float:
    if task == "binary":
        p = np.clip(prediction, 1e-6, 1 - 1e-6)
        return float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p)))
    return float(np.mean((truth - prediction) ** 2))


def retention_report(real_fit: pd.DataFrame, holdout: pd.DataFrame, synthetic: pd.DataFrame,
                     target: str, task: str, *, positive=None, auditors=None,
                     seed: int = 1729) -> list[AuditorResult]:
    """Train each auditor on real and on synthetic rows; score both on the holdout."""
    categorical = [c for c in real_fit.columns if c != target and not (
        pd.api.types.is_numeric_dtype(real_fit[c]) and not pd.api.types.is_bool_dtype(real_fit[c]))]
    numeric = [c for c in real_fit.columns if c != target and c not in categorical]
    if task == "binary" and positive is None:
        positive = sorted(pd.unique(real_fit[target]).tolist(), key=lambda v: (str(type(v)), str(v)))[1]
    y_hold = _labels(holdout, target, task, positive)
    y_fit = _labels(real_fit, target, task, positive)
    null = _loss(task, y_hold, np.full(y_hold.shape, y_fit.mean()))
    X_hold = _frame(holdout, target, categorical)
    results = []
    for name in auditors or default_auditors():
        losses = {}
        for arm, table in (("real", real_fit), ("synthetic", synthetic)):
            kind, model = _auditor(name, task, categorical, numeric, seed)
            X = _frame(table, target, categorical)
            y = _labels(table, target, task, positive)
            if task == "binary" and np.unique(y).size < 2:
                losses[arm] = _loss(task, y_hold, np.full(y_hold.shape, float(y.mean())))
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "catboost":
                    model.fit(X, y, cat_features=categorical)
                else:
                    model.fit(X, y)
                prediction = model.predict_proba(X_hold)[:, 1] if task == "binary" else model.predict(X_hold)
            losses[arm] = _loss(task, y_hold, np.asarray(prediction, dtype=float))
        results.append(AuditorResult(name, null, losses["real"], losses["synthetic"]))
    return results


def worst_retention(results: list[AuditorResult]) -> float | None:
    """Descriptive minimum over informative local auditors; not a TDL qualification."""
    values = [r.retention for r in results if r.retention is not None and math.isfinite(r.retention)]
    return min(values) if values else None


def local_summary(results: list[AuditorResult]) -> dict:
    """Keep the diagnostic minimum separate from complete required-auditor evidence."""
    values = [r.retention for r in results]
    complete = bool(values) and all(v is not None and math.isfinite(v) for v in values)
    return {"worst_retention": worst_retention(results),
            "retention_aggregation": "minimum_over_informative_local_auditors",
            "required_auditors_complete": complete,
            "tdl_qualified": False}
