from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .marginals import fit_marginal, inverse_marginal
from .utils import EPS, bits_for_jsonish, clip01, qfloat, qlist, sklearn_seed


def _safe_split(X: np.ndarray, y: np.ndarray, task: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y) < 12:
        return X, X, y, y
    stratify = y if task == "binary" and len(np.unique(y)) > 1 else None
    seed = sklearn_seed(seed)
    try:
        return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(X, y, test_size=0.25, random_state=seed)


def _sparse_terms(coef: np.ndarray, max_terms: int = 24) -> list[list[float]]:
    coef = np.asarray(coef, dtype=float).ravel()
    if coef.size == 0:
        return []
    order = np.argsort(-np.abs(coef))
    keep = [int(i) for i in order[: min(max_terms, len(order))] if abs(coef[i]) >= 1e-5]
    return [[int(i), qfloat(coef[i])] for i in sorted(keep)]


def _linear_predict(params: dict[str, Any], X: np.ndarray) -> np.ndarray:
    pred = np.full(X.shape[0], float(params["intercept"]), dtype=float)
    for idx, coef in params.get("terms", []):
        pred += float(coef) * X[:, int(idx)]
    return pred


def _fit_linear_regression(X: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    params = {
        "intercept": qfloat(model.intercept_),
        "terms": _sparse_terms(model.coef_),
    }
    pred = clip01(_linear_predict(params, X))
    resid = y - pred
    target = {"kind": "linear_sparse", "params": params}
    residual = {
        "kind": "homo",
        "params": {
            "sigma": qfloat(float(np.std(resid))),
            "residual_quantiles": qlist(np.quantile(resid, [0.05, 0.25, 0.5, 0.75, 0.95])),
        },
    }
    return target, residual


def _fit_logit(X: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(np.unique(y)) < 2:
        p = qfloat(float(np.mean(y)))
        return {"kind": "lift_logit", "params": {"intercept": qfloat(np.log((p + EPS) / (1 - p + EPS))), "terms": []}}, {
            "kind": "bernoulli_cal",
            "params": {"base_rate": p},
        }

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    params = {
        "intercept": qfloat(model.intercept_[0]),
        "terms": _sparse_terms(model.coef_[0]),
    }
    proba = expit(_linear_predict(params, X))
    target = {"kind": "lift_logit", "params": params}
    residual = {
        "kind": "bernoulli_cal",
        "params": {
            "base_rate": qfloat(float(np.mean(y))),
            "mean_prob": qfloat(float(np.mean(proba))),
            "log_loss": qfloat(float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]))),
        },
    }
    return target, residual


def _tree_common(model: DecisionTreeRegressor | DecisionTreeClassifier, values: np.ndarray) -> dict[str, Any]:
    tree = model.tree_
    return {
        "children_left": [int(v) for v in tree.children_left],
        "children_right": [int(v) for v in tree.children_right],
        "feature": [int(v) for v in tree.feature],
        "threshold": qlist(tree.threshold),
        "value": qlist(values),
        "node_count": int(tree.node_count),
    }


def _fit_tree_regression(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    max_leaf_nodes = int(np.clip(len(y) // 20, 4, 24))
    min_samples_leaf = int(np.clip(len(y) // 80, 2, 20))
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=sklearn_seed(seed),
    )
    model.fit(X, y)
    values = model.tree_.value[:, 0, 0]
    params = _tree_common(model, values)
    leaf_ids = model.apply(X)
    pred = values[leaf_ids]
    leaf_sigma = []
    for node in range(model.tree_.node_count):
        mask = leaf_ids == node
        sigma = float(np.std(y[mask] - pred[mask])) if np.any(mask) else 0.0
        leaf_sigma.append(qfloat(sigma))
    params["leaf_sigma"] = leaf_sigma
    target = {"kind": "tree_piecewise", "params": params}
    residual = {
        "kind": "hetero",
        "params": {"mean_leaf_sigma": qfloat(float(np.mean(leaf_sigma)))},
    }
    return target, residual


def _fit_tree_binary(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(np.unique(y)) < 2:
        p = float(np.mean(y))
        return {"kind": "tree_piecewise", "params": {"constant_prob": qfloat(p)}}, {
            "kind": "bernoulli_cal",
            "params": {"base_rate": qfloat(p)},
        }
    max_leaf_nodes = int(np.clip(len(y) // 25, 4, 20))
    min_samples_leaf = int(np.clip(len(y) // 70, 2, 20))
    model = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=sklearn_seed(seed),
    )
    model.fit(X, y)
    values = []
    class_index = list(model.classes_).index(1.0) if 1.0 in model.classes_ else len(model.classes_) - 1
    for raw in model.tree_.value[:, 0, :]:
        denom = max(float(np.sum(raw)), EPS)
        values.append(float(raw[class_index]) / denom)
    params = _tree_common(model, np.asarray(values))
    target = {"kind": "tree_piecewise", "params": params}
    residual = {
        "kind": "bernoulli_cal",
        "params": {"base_rate": qfloat(float(np.mean(y)))},
    }
    return target, residual


def _tree_predict(params: dict[str, Any], X: np.ndarray) -> np.ndarray:
    if "constant_prob" in params:
        return np.full(X.shape[0], float(params["constant_prob"]), dtype=float)
    left = params["children_left"]
    right = params["children_right"]
    feature = params["feature"]
    threshold = params["threshold"]
    value = params["value"]
    out = np.empty(X.shape[0], dtype=float)
    for i, row in enumerate(X):
        node = 0
        while left[node] != -1:
            f = int(feature[node])
            node = left[node] if row[f] <= float(threshold[node]) else right[node]
        out[i] = float(value[node])
    return out


def _score_candidate(
    target: dict[str, Any],
    residual: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
) -> float:
    pred = predict_target(target, X_val, task)
    bits = bits_for_jsonish(target) + bits_for_jsonish(residual)
    penalty = 0.00002 * bits / max(1, len(y_train) * max(1, X_train.shape[1]))
    if task == "regression":
        if np.std(y_val) <= EPS:
            score = 1.0 - float(np.mean(np.abs(pred - y_val)))
        else:
            score = float(r2_score(y_val, pred))
        return score - penalty
    if len(np.unique(y_val)) < 2:
        score = 1.0 - float(np.mean(np.abs(pred - y_val)))
    else:
        score = -float(log_loss(y_val, np.clip(pred, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return score - penalty


def fit_target(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    seed: int,
    target_kind: str = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    X = clip01(X)
    y = (clip01(y) >= 0.5).astype(float) if task == "binary" else clip01(y)

    if X.shape[1] == 0:
        marginal = fit_marginal(y)
        target = {"kind": "target_marginal", "params": marginal}
        residual = {"kind": "homo" if task == "regression" else "bernoulli_cal", "params": {}}
        return _with_bits(target), _with_bits(residual)

    X_train, X_val, y_train, y_val = _safe_split(X, y, task, seed)

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if task == "regression":
        if target_kind in {"auto", "linear_sparse"}:
            candidates.append(_fit_linear_regression(X, y))
        if target_kind in {"auto", "tree_piecewise", "TREEPW"}:
            candidates.append(_fit_tree_regression(X, y, seed))
    else:
        if target_kind in {"auto", "lift_logit", "linear_sparse"}:
            candidates.append(_fit_logit(X, y))
        if target_kind in {"auto", "tree_piecewise", "TREEPW"}:
            candidates.append(_fit_tree_binary(X, y, seed))

    if not candidates:
        raise ValueError(f"unsupported target kind {target_kind!r} for task {task}")

    scored = [
        (_score_candidate(target, residual, X_train, y_train, X_val, y_val, task), target, residual)
        for target, residual in candidates
    ]
    _, target, residual = max(scored, key=lambda item: item[0])
    return _with_bits(target), _with_bits(residual)


def _with_bits(obj: dict[str, Any]) -> dict[str, Any]:
    obj = dict(obj)
    obj["bits"] = bits_for_jsonish(obj)
    return obj


def predict_target(target: dict[str, Any], X: np.ndarray, task: str) -> np.ndarray:
    kind = target["kind"]
    params = target["params"]
    if kind == "linear_sparse":
        return clip01(_linear_predict(params, X))
    if kind == "lift_logit":
        return clip01(expit(_linear_predict(params, X)))
    if kind == "tree_piecewise":
        return clip01(_tree_predict(params, X))
    if kind == "target_marginal":
        u = np.linspace(EPS, 1.0 - EPS, X.shape[0])
        return clip01(inverse_marginal(params, u))
    raise ValueError(f"unsupported target kind: {kind}")


def sample_target(
    target: dict[str, Any],
    residual: dict[str, Any],
    X: np.ndarray,
    task: str,
    rng: np.random.Generator,
) -> np.ndarray:
    pred = predict_target(target, X, task)
    if task == "binary":
        return (rng.uniform(0.0, 1.0, size=len(pred)) <= pred).astype(float)

    if residual["kind"] == "hetero" and target["kind"] == "tree_piecewise" and "leaf_sigma" in target["params"]:
        sigma = _tree_leaf_sigma(target["params"], X)
    else:
        sigma = np.full(len(pred), float(residual.get("params", {}).get("sigma", 0.0)), dtype=float)
    noise = rng.normal(0.0, sigma)
    return clip01(pred + noise)


def _tree_leaf_sigma(params: dict[str, Any], X: np.ndarray) -> np.ndarray:
    if "constant_prob" in params:
        return np.zeros(X.shape[0], dtype=float)
    left = params["children_left"]
    right = params["children_right"]
    feature = params["feature"]
    threshold = params["threshold"]
    leaf_sigma = params.get("leaf_sigma", [0.0] * len(left))
    out = np.empty(X.shape[0], dtype=float)
    for i, row in enumerate(X):
        node = 0
        while left[node] != -1:
            f = int(feature[node])
            node = left[node] if row[f] <= float(threshold[node]) else right[node]
        out[i] = float(leaf_sigma[node])
    return out
