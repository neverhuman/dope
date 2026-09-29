from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .utils import EPS, bits_for_jsonish, clip01, qfloat, qlist


def _discrete_params(col: np.ndarray) -> tuple[list[float], list[float]]:
    values, counts = np.unique(np.round(col, 6), return_counts=True)
    probs = counts.astype(float) / max(1, counts.sum())
    order = np.argsort(values)
    return qlist(values[order]), qlist(probs[order])


def _beta_moments(col: np.ndarray) -> dict[str, float] | None:
    col = np.clip(np.asarray(col, dtype=float), 1e-5, 1.0 - 1e-5)
    mean = float(np.mean(col))
    var = float(np.var(col))
    max_var = mean * (1.0 - mean)
    if var <= 1e-8 or max_var <= 1e-8 or var >= max_var:
        return None
    common = max_var / var - 1.0
    alpha = float(np.clip(mean * common, 0.05, 500.0))
    beta = float(np.clip((1.0 - mean) * common, 0.05, 500.0))
    return {"alpha": qfloat(alpha), "beta": qfloat(beta)}


def _beta_ks(col: np.ndarray, params: dict[str, float]) -> float:
    sorted_col = np.sort(np.clip(col, 1e-5, 1.0 - 1e-5))
    if len(sorted_col) == 0:
        return 0.0
    pred = stats.beta.cdf(sorted_col, params["alpha"], params["beta"])
    emp = (np.arange(len(sorted_col)) + 0.5) / len(sorted_col)
    return float(np.max(np.abs(pred - emp)))


def _empq_params(col: np.ndarray, k: int | None = None) -> dict[str, Any]:
    n = len(col)
    if k is None:
        k = int(np.clip(round(np.sqrt(max(n, 1))) * 2 + 1, 9, 33))
    qs = np.linspace(0.0, 1.0, k)
    values = np.quantile(clip01(col), qs)
    values = np.maximum.accumulate(values)
    return {"k": int(k), "qs": qlist(qs), "values": qlist(values)}


def fit_marginal(col: np.ndarray, empq_k: int | None = None) -> dict[str, Any]:
    col = clip01(col)
    rounded = np.round(col, 6)
    unique = np.unique(rounded)
    zero_frac = float(np.mean(col <= 1e-9))
    one_frac = float(np.mean(col >= 1.0 - 1e-9))

    if len(unique) == 1:
        values, probs = _discrete_params(col)
        marginal = {"kind": "spike_grid", "params": {"values": values, "probs": probs}}
    elif len(unique) <= 2 and np.all(np.isin(unique, [0.0, 1.0])):
        marginal = {"kind": "bern", "params": {"p": qfloat(float(np.mean(col >= 0.5)))}}
    elif len(unique) <= 12 or max(zero_frac, one_frac) > 0.25:
        values, probs = _discrete_params(col)
        kind = "ordinal" if len(unique) <= 8 else "spike_grid"
        marginal = {"kind": kind, "params": {"values": values, "probs": probs}}
    elif zero_frac > 0.03 and np.sum(col > 1e-9) >= 8:
        positive = col[col > 1e-9]
        beta = _beta_moments(positive)
        if beta is None or _beta_ks(positive, beta) > 0.12:
            marginal = {
                "kind": "empq",
                "params": _empq_params(col, empq_k),
            }
        else:
            beta["zero_prob"] = qfloat(zero_frac)
            marginal = {"kind": "zi_beta", "params": beta}
    else:
        beta = _beta_moments(col)
        if beta is not None and _beta_ks(col, beta) <= 0.08:
            marginal = {"kind": "beta", "params": beta}
        else:
            marginal = {"kind": "empq", "params": _empq_params(col, empq_k)}

    marginal["bits"] = bits_for_jsonish(marginal)
    return marginal


def fit_marginals(X: np.ndarray, empq_k: int | None = None) -> list[dict[str, Any]]:
    return [fit_marginal(X[:, j], empq_k=empq_k) for j in range(X.shape[1])]


def inverse_marginal(marginal: dict[str, Any], u: np.ndarray) -> np.ndarray:
    kind = marginal["kind"]
    params = marginal["params"]
    u = np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)

    if kind == "bern":
        return (u <= float(params["p"])).astype(float)

    if kind in {"spike_grid", "ordinal"}:
        values = np.asarray(params["values"], dtype=float)
        probs = np.asarray(params["probs"], dtype=float)
        probs = probs / max(float(np.sum(probs)), EPS)
        cdf = np.cumsum(probs)
        idx = np.searchsorted(cdf, u, side="left")
        return values[np.clip(idx, 0, len(values) - 1)]

    if kind == "beta":
        values = stats.beta.ppf(u, float(params["alpha"]), float(params["beta"]))
        return clip01(np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0))

    if kind == "zi_beta":
        zero_prob = float(params["zero_prob"])
        out = np.zeros_like(u, dtype=float)
        mask = u > zero_prob
        scaled = np.clip((u[mask] - zero_prob) / max(1.0 - zero_prob, EPS), EPS, 1.0 - EPS)
        out[mask] = stats.beta.ppf(scaled, float(params["alpha"]), float(params["beta"]))
        return clip01(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0))

    if kind == "empq":
        qs = np.asarray(params["qs"], dtype=float)
        values = np.asarray(params["values"], dtype=float)
        return clip01(np.interp(u, qs, values))

    raise ValueError(f"unsupported marginal kind: {kind}")


def sample_marginals(marginals: list[dict[str, Any]], U: np.ndarray) -> np.ndarray:
    if not marginals:
        return np.empty((U.shape[0], 0), dtype=float)
    cols = [inverse_marginal(marginal, U[:, j]) for j, marginal in enumerate(marginals)]
    return clip01(np.column_stack(cols))
