from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .io import load_real_split
from .kernel import fit_kernel_from_arrays, sample_kernel
from .utils import EPS, canonical_dumps, clip01, qfloat


COMPONENT_WEIGHTS = {
    "transfer_utility": 0.35,
    "feature_importance": 0.25,
    "distribution_copula": 0.20,
    "residual_or_calibration": 0.10,
    "prediction_agreement": 0.10,
}


def evaluate_kernel(real_dir: str | Path, kernel: dict[str, Any], seed: int = 1729) -> dict[str, Any]:
    task = kernel["shell"]["task"]
    X_train, y_train, X_test, y_test, _ = load_real_split(real_dir, task)
    synth = sample_kernel(kernel, len(X_train), seed=seed)
    candidate = score_synthetic_dataset(X_train, y_train, X_test, y_test, synth[:, :-1], synth[:, -1], task, seed)
    candidate["kernel_bytes"] = int(kernel.get("kernel_bytes", len(canonical_dumps(kernel).encode("utf-8"))))
    candidate["rbcs"] = rbcs(candidate["fidelity_real"], candidate["kernel_bytes"], len(X_train), X_train.shape[1] + 1)

    baselines = baseline_reports(X_train, y_train, X_test, y_test, task, kernel, seed)
    return {
        "format": "dope-kernel-eval",
        "version": 1,
        "task": task,
        "rows": {"train": int(len(X_train)), "test": int(len(X_test))},
        "cols": int(X_train.shape[1] + 1),
        "component_weights": COMPONENT_WEIGHTS,
        "kernel": candidate,
        "baselines": baselines,
        "baseline_median_rbcs": qfloat(float(np.median([b["rbcs"] for b in baselines.values()]))),
        "beats_baseline_median": bool(candidate["rbcs"] > float(np.median([b["rbcs"] for b in baselines.values()]))),
        "notes": [
            "RBCS = Fidelity_real - 0.03 * log2(kernel_bytes / rows / cols).",
            "CatBoost, LightGBM, and XGBoost teacher families are not serialized; sklearn families provide local V1 metrics.",
        ],
    }


def baseline_reports(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    kernel: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    independent = fit_kernel_from_arrays(X_train, y_train, task, seed=seed, dependence_kind="ind")
    gaussian = fit_kernel_from_arrays(X_train, y_train, task, seed=seed, dependence_kind="gauss_copula")

    for name, baseline_kernel in [("independent_marginal", independent), ("gaussian_copula", gaussian)]:
        synth = sample_kernel(baseline_kernel, len(X_train), seed=seed)
        report = score_synthetic_dataset(X_train, y_train, X_test, y_test, synth[:, :-1], synth[:, -1], task, seed)
        report["kernel_bytes"] = int(baseline_kernel["kernel_bytes"])
        report["rbcs"] = rbcs(report["fidelity_real"], report["kernel_bytes"], len(X_train), X_train.shape[1] + 1)
        reports[name] = report

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X_train), size=len(X_train))
    X_boot = X_train[idx]
    y_boot = y_train[idx]
    report = score_synthetic_dataset(X_train, y_train, X_test, y_test, X_boot, y_boot, task, seed)
    bootstrap_bytes = int(np.column_stack([X_train, y_train]).nbytes)
    report["kernel_bytes"] = bootstrap_bytes
    report["rbcs"] = rbcs(report["fidelity_real"], bootstrap_bytes, len(X_train), X_train.shape[1] + 1)
    reports["row_bootstrap"] = report
    return reports


def rbcs(fidelity_real: float, kernel_bytes: int, rows: int, cols: int) -> float:
    ratio = max(float(kernel_bytes) / max(1.0, float(rows * cols)), EPS)
    return qfloat(float(fidelity_real) - 0.03 * float(np.log2(ratio)))


def score_synthetic_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
) -> dict[str, Any]:
    X_train = clip01(X_train)
    y_train = (clip01(y_train) >= 0.5).astype(float) if task == "binary" else clip01(y_train)
    X_test = clip01(X_test)
    y_test = (clip01(y_test) >= 0.5).astype(float) if task == "binary" else clip01(y_test)
    X_synth = clip01(X_synth)
    y_synth = (clip01(y_synth) >= 0.5).astype(float) if task == "binary" else clip01(y_synth)

    components = {
        "transfer_utility": transfer_utility(X_train, y_train, X_test, y_test, X_synth, y_synth, task, seed),
        "feature_importance": feature_importance_score(X_train, y_train, X_synth, y_synth, task, seed),
        "distribution_copula": distribution_copula_score(X_train, y_train, X_synth, y_synth),
        "residual_or_calibration": residual_or_calibration_score(X_train, y_train, X_synth, y_synth, task, seed),
        "prediction_agreement": prediction_agreement_score(X_train, y_train, X_test, X_synth, y_synth, task, seed),
    }
    fidelity = sum(COMPONENT_WEIGHTS[key] * components[key] for key in COMPONENT_WEIGHTS)
    return {
        "fidelity_real": qfloat(float(fidelity)),
        "components": {key: qfloat(value) for key, value in components.items()},
    }


def _reg_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    scale = max(float(np.std(y_true)), 0.05)
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    return float(np.clip(1.0 - rmse / scale, 0.0, 1.0))


def _clf_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) >= 2:
        try:
            return float(np.clip(roc_auc_score(y_true, proba), 0.0, 1.0))
        except ValueError:
            pass
    return float(accuracy_score(y_true, proba >= 0.5))


def _models(task: str, seed: int) -> list[Any]:
    if task == "regression":
        return [
            Ridge(alpha=1.0),
            RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1),
            ExtraTreesRegressor(n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1),
        ]
    return [
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
        RandomForestClassifier(n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1),
        ExtraTreesClassifier(n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1),
    ]


def _fit_predict_score(model: Any, X_fit: np.ndarray, y_fit: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, task: str) -> float:
    if task == "binary" and len(np.unique(y_fit)) < 2:
        proba = np.full(len(y_test), float(np.mean(y_fit)))
        return _clf_score(y_test, proba)
    model.fit(X_fit, y_fit)
    if task == "regression":
        return _reg_score(y_test, clip01(model.predict(X_test)))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, -1]
    else:
        proba = clip01(model.decision_function(X_test))
    return _clf_score(y_test, proba)


def transfer_utility(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
) -> float:
    scores = []
    for model in _models(task, seed)[:2]:
        real_score = _fit_predict_score(model, X_train, y_train, X_test, y_test, task)
        synth_model = _clone_model(model, seed)
        synth_score = _fit_predict_score(synth_model, X_synth, y_synth, X_test, y_test, task)
        scores.append(1.0 - min(1.0, abs(real_score - synth_score) / max(abs(real_score), 0.1)))
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _clone_model(model: Any, seed: int) -> Any:
    params = model.get_params()
    if "random_state" in params:
        params["random_state"] = seed
    return model.__class__(**params)


def _importance(model: Any, X: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    if task == "binary" and len(np.unique(y)) < 2:
        return np.zeros(X.shape[1], dtype=float)
    model.fit(X, y)
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        imp = np.zeros(X.shape[1], dtype=float)
    if np.sum(imp) > EPS:
        imp = imp / np.sum(imp)
    return imp


def feature_importance_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
) -> float:
    p = X_train.shape[1]
    if p <= 1:
        return 1.0
    scores = []
    for model in _models(task, seed):
        real = _importance(_clone_model(model, seed), X_train, y_train, task)
        synth = _importance(_clone_model(model, seed), X_synth, y_synth, task)
        corr = stats.spearmanr(real, synth).correlation
        corr_score = 0.5 if not np.isfinite(corr) else (float(corr) + 1.0) / 2.0
        k = max(1, min(5, p // 3 if p >= 3 else 1))
        top_real = set(np.argsort(-real)[:k])
        top_synth = set(np.argsort(-synth)[:k])
        overlap = len(top_real & top_synth) / k
        scores.append(0.65 * corr_score + 0.35 * overlap)
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def distribution_copula_score(X_train: np.ndarray, y_train: np.ndarray, X_synth: np.ndarray, y_synth: np.ndarray) -> float:
    real = clip01(np.column_stack([X_train, y_train]))
    synth = clip01(np.column_stack([X_synth, y_synth]))
    p = real.shape[1]
    ks_scores = []
    w_scores = []
    for j in range(p):
        ks_scores.append(1.0 - float(stats.ks_2samp(real[:, j], synth[:, j]).statistic))
        w = float(stats.wasserstein_distance(real[:, j], synth[:, j]))
        w_scores.append(1.0 - min(1.0, w / 0.25))
    corr_score = _corr_similarity(real, synth)
    mi_score = _mi_graph_similarity(real, synth)
    sliced = _sliced_wasserstein_score(real, synth)
    return float(np.clip(0.30 * np.mean(ks_scores) + 0.20 * np.mean(w_scores) + 0.20 * corr_score + 0.15 * mi_score + 0.15 * sliced, 0.0, 1.0))


def _corr_similarity(real: np.ndarray, synth: np.ndarray) -> float:
    if real.shape[1] <= 1:
        return 1.0
    cr = np.nan_to_num(np.corrcoef(real, rowvar=False), nan=0.0)
    cs = np.nan_to_num(np.corrcoef(synth, rowvar=False), nan=0.0)
    diff = float(np.mean(np.abs(cr - cs)))
    return float(np.clip(1.0 - diff / 1.5, 0.0, 1.0))


def _mi_graph_similarity(real: np.ndarray, synth: np.ndarray, bins: int = 8) -> float:
    p = real.shape[1]
    if p <= 1:
        return 1.0
    real_mi = []
    synth_mi = []
    for i in range(p):
        for j in range(i + 1, p):
            real_mi.append(_binned_mi(real[:, i], real[:, j], bins))
            synth_mi.append(_binned_mi(synth[:, i], synth[:, j], bins))
    if not real_mi:
        return 1.0
    corr = stats.spearmanr(real_mi, synth_mi).correlation
    return 0.5 if not np.isfinite(corr) else float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def _binned_mi(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    hist, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    total = float(np.sum(hist))
    if total <= 0:
        return 0.0
    pxy = hist / total
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    expected = px @ py
    mask = (pxy > 0) & (expected > 0)
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))


def _sliced_wasserstein_score(real: np.ndarray, synth: np.ndarray, projections: int = 16) -> float:
    rng = np.random.default_rng(1729)
    scores = []
    for _ in range(projections):
        direction = rng.normal(size=real.shape[1])
        direction = direction / max(float(np.linalg.norm(direction)), EPS)
        scores.append(stats.wasserstein_distance(real @ direction, synth @ direction))
    return float(np.clip(1.0 - min(1.0, float(np.mean(scores)) / 0.25), 0.0, 1.0))


def residual_or_calibration_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
) -> float:
    if task == "binary":
        prevalence = 1.0 - min(1.0, abs(float(np.mean(y_train)) - float(np.mean(y_synth))) / 0.5)
        nn_score = _nearest_lift_score(X_train, y_train, X_synth, y_synth)
        return float(np.clip(0.55 * prevalence + 0.45 * nn_score, 0.0, 1.0))

    real_resid = _linear_residuals(X_train, y_train)
    synth_resid = _linear_residuals(X_synth, y_synth)
    q_real = np.quantile(real_resid, [0.1, 0.25, 0.5, 0.75, 0.9])
    q_synth = np.quantile(synth_resid, [0.1, 0.25, 0.5, 0.75, 0.9])
    std_diff = abs(float(np.std(real_resid)) - float(np.std(synth_resid))) / max(float(np.std(real_resid)), 0.05)
    q_diff = float(np.mean(np.abs(q_real - q_synth))) / 0.25
    return float(np.clip(1.0 - min(1.0, 0.5 * std_diff + 0.5 * q_diff), 0.0, 1.0))


def _linear_residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return y - clip01(model.predict(X))


def _nearest_lift_score(X_real: np.ndarray, y_real: np.ndarray, X_synth: np.ndarray, y_synth: np.ndarray) -> float:
    if len(X_real) == 0 or len(X_synth) == 0:
        return 0.0
    k = min(10, len(X_synth))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_synth)
    idx = nn.kneighbors(X_real, return_distance=False)
    local = y_synth[idx].mean(axis=1)
    if np.std(local) <= EPS or np.std(y_real) <= EPS:
        return 1.0 - min(1.0, abs(float(np.mean(local)) - float(np.mean(y_real))) / 0.5)
    corr = stats.spearmanr(local, y_real).correlation
    return 0.5 if not np.isfinite(corr) else float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def prediction_agreement_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    task: str,
    seed: int,
) -> float:
    scores = []
    for model in _models(task, seed)[:2]:
        if task == "binary" and (len(np.unique(y_train)) < 2 or len(np.unique(y_synth)) < 2):
            real_pred = np.full(len(X_test), float(np.mean(y_train)))
            synth_pred = np.full(len(X_test), float(np.mean(y_synth)))
        else:
            real_model = _clone_model(model, seed)
            synth_model = _clone_model(model, seed)
            real_model.fit(X_train, y_train)
            synth_model.fit(X_synth, y_synth)
            if task == "regression":
                real_pred = clip01(real_model.predict(X_test))
                synth_pred = clip01(synth_model.predict(X_test))
            else:
                real_pred = real_model.predict_proba(X_test)[:, -1]
                synth_pred = synth_model.predict_proba(X_test)[:, -1]
        diff = float(np.mean(np.abs(real_pred - synth_pred)))
        scores.append(1.0 - min(1.0, diff / 0.5))
    return float(np.clip(np.mean(scores), 0.0, 1.0))
