from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .utils import bits_for_jsonish, finite_corrcoef, nearest_correlation, qmatrix


def rank_uniforms(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    U = np.empty((n, p), dtype=float)
    for j in range(p):
        ranks = stats.rankdata(X[:, j], method="average")
        U[:, j] = ranks / (n + 1.0)
    return np.clip(U, 1e-6, 1.0 - 1e-6)


def fit_dependence(X: np.ndarray, kind: str = "auto", shrink: float = 0.04) -> dict[str, Any]:
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    if p <= 1 or kind == "ind":
        dep = {"kind": "ind", "params": {}}
        dep["bits"] = bits_for_jsonish(dep)
        return dep

    U = rank_uniforms(X)
    Z = stats.norm.ppf(U)
    corr = finite_corrcoef(Z)
    corr = (1.0 - shrink) * corr + shrink * np.eye(p)
    corr = nearest_correlation(corr)

    offdiag = corr[~np.eye(p, dtype=bool)]
    if kind == "auto" and float(np.max(np.abs(offdiag))) < 0.035:
        dep = {"kind": "ind", "params": {}}
    else:
        dep = {
            "kind": "gauss_copula",
            "params": {
                "rank": int(np.linalg.matrix_rank(corr)),
                "corr": qmatrix(corr),
            },
        }
    dep["bits"] = bits_for_jsonish(dep)
    return dep


def sample_uniforms(dependence: dict[str, Any], rows: int, p: int, rng: np.random.Generator) -> np.ndarray:
    kind = dependence["kind"]
    if p == 0:
        return np.empty((rows, 0), dtype=float)
    if kind == "ind":
        return rng.uniform(0.0, 1.0, size=(rows, p))
    if kind == "gauss_copula":
        corr = np.asarray(dependence["params"]["corr"], dtype=float)
        corr = nearest_correlation(corr)
        normals = rng.multivariate_normal(np.zeros(p), corr, size=rows, check_valid="ignore")
        return np.clip(stats.norm.cdf(normals), 1e-9, 1.0 - 1e-9)
    raise ValueError(f"unsupported dependence kind for sampling: {kind}")
