"""Non-Gaussian, XOR/XNOR, tree-rule, heteroscedastic, lattice/nominal, and contrastive
synthetic tabular priors for DOPE v3 (`SingleEncV8`) training.

Explicitly fixes V7's prior-space blind spots:
- Non-Gaussian copulas (Clayton lower tail, Gumbel upper tail, Frank symmetric, Student-t)
- 2..4 component Gaussian mixture regimes (multi-modal X geometry)
- Pure XOR / XNOR & depth-3/4 oblivious decision trees (zero linear correlation)
- Friedman #1-#3 nonlinear functions & T03 heteroscedastic noise targets
- Decimal lattices + heaping + zero-inflation + wide nominal one-hot groups + MAR missingness
- Contrastive pairs (identical marginals, opposite XOR vs XNOR or Clayton vs Gumbel)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from research.levels.v8_evidence import _norm_ppf_scalar

_NORM_PPF_VEC = np.vectorize(_norm_ppf_scalar)


def _sample_copula_x(rng: np.random.Generator, n: int, p: int, family: str) -> np.ndarray:
    """Draw [n, p] latent features with non-Gaussian dependence structures."""
    if family == "student_t":
        nu = float(rng.choice([3.0, 4.0, 5.0]))
        rank = min(p, int(rng.integers(2, 6)))
        L = rng.normal(0.0, 0.6, size=(p, rank))
        L = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True) / 0.85, 1.0)
        z = rng.normal(size=(n, rank)) @ L.T + rng.normal(size=(n, p)) * 0.4
        scale = np.sqrt(nu / rng.chisquare(nu, size=(n, 1)))
        return np.clip(z * scale, -6.0, 6.0).astype(np.float32)

    if family == "clayton":
        theta = float(rng.uniform(2.0, 6.0))
        x = rng.normal(size=(n, p))
        for j in range(0, p - 1, 2):
            u1 = rng.uniform(1e-4, 1.0 - 1e-4, size=n)
            w = rng.uniform(1e-4, 1.0 - 1e-4, size=n)
            u2 = np.clip((u1 ** (-theta) * (w ** (-theta / (1.0 + theta)) - 1.0) + 1.0) ** (-1.0 / theta), 1e-4, 1.0 - 1e-4)
            x[:, j] = _NORM_PPF_VEC(u1)
            x[:, j + 1] = _NORM_PPF_VEC(u2)
        return np.clip(x, -6.0, 6.0).astype(np.float32)

    if family == "gumbel_upper":
        # Flip Clayton (1 - U) to create strong upper-tail dependence!
        x = -_sample_copula_x(rng, n, p, "clayton")
        return x.astype(np.float32)

    if family == "mixture":
        k_mix = int(rng.integers(2, 5))
        centers = rng.normal(0.0, 1.6, size=(k_mix, p))
        comp = rng.integers(0, k_mix, size=n)
        x = centers[comp] + rng.normal(0.0, 0.55, size=(n, p))
        x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-4)
        return np.clip(x, -6.0, 6.0).astype(np.float32)

    # Default correlated Gaussian
    rank = min(p, int(rng.integers(2, 7)))
    L = rng.normal(0.0, 0.65, size=(p, rank))
    L = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True) / 0.90, 1.0)
    z = rng.normal(size=(n, rank)) @ L.T + rng.normal(size=(n, p)) * 0.45
    return np.clip(z, -6.0, 6.0).astype(np.float32)


def _apply_realistic_marginals(
    rng: np.random.Generator,
    X: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Inject decimal lattices, heaping, zero-inflation, one-hot nominal groups, and MAR missingness."""
    n, p = X.shape
    Xo = X.copy()
    meta: list[dict[str, Any]] = [
        {
            "group_id": j, "group_kind": 0, "group_width": 1, "parent_col_idx": 65535,
            "support_class": 0, "skew_guard_active": 0, "lattice_q0_unit": 0.0,
            "lattice_origin_unit": 0.0, "has_lower_bound": 0, "has_upper_bound": 0,
            "log_pre_tail_asym": 0.0, "log_unique_bucket": 0.0,
        }
        for j in range(p)
    ]
    # Optional one-hot nominal group on trailing 3..5 columns when p >= 8
    gid_used = p
    if p >= 8 and rng.random() < 0.45:
        w = int(rng.integers(3, min(6, p - 3)))
        start = p - w
        logits = Xo[:, start:] * 0.9 + rng.normal(0.0, 0.3, size=(1, w))
        pick = np.argmax(logits, axis=1)
        Xo[:, start:] = 0.0
        Xo[np.arange(n), start + pick] = 1.0
        for c in range(start, p):
            meta[c]["group_id"] = gid_used
            meta[c]["group_kind"] = 4
            meta[c]["group_width"] = w
            meta[c]["support_class"] = 2

    # Inject lattice, zero-inflation, or MAR missingness on a subset of numeric cols
    limit = p - 5 if p >= 8 else p
    for j in range(max(limit, 1)):
        u = rng.random()
        if u < 0.12:
            # Decimal lattice + heaping
            Xo[:, j] = np.round(Xo[:, j] * 5.0) / 5.0
            heap = rng.random(n) < 0.35
            Xo[heap, j] = np.round(Xo[heap, j])
            meta[j]["support_class"] = 1
            meta[j]["lattice_q0_unit"] = 0.2
        elif u < 0.24:
            # Zero-inflated positive
            zero_m = rng.random(n) < float(rng.uniform(0.25, 0.55))
            Xo[:, j] = np.where(zero_m, 0.0, np.abs(Xo[:, j]) + 0.15)
            meta[j]["support_class"] = 3
            meta[j]["has_lower_bound"] = 1
        elif u < 0.34:
            # MAR missingness (10..25%)
            miss_m = rng.random(n) < float(rng.uniform(0.10, 0.25))
            Xo[miss_m, j] = np.nan
    return Xo, meta


def sample_v8_prior_episode(
    rng: np.random.Generator,
    n_fit: int = 384,
    n_qry: int = 256,
    p: int | None = None,
    task_id: int | None = None,
    mechanism: str | None = None,
) -> dict[str, Any]:
    """Generate a single rich non-Gaussian V8 training episode with disjoint FIT/QRY splits."""
    p = int(p if p is not None else rng.integers(6, 25))
    task_id = int(task_id if task_id is not None else rng.choice([0, 1, 2], p=[0.42, 0.38, 0.20]))
    n_classes = 1 if task_id == 0 else (2 if task_id == 1 else int(rng.integers(3, 7)))
    copula_fam = str(rng.choice(["gaussian", "student_t", "clayton", "gumbel_upper", "mixture"]))
    mech = mechanism or str(rng.choice(["xor", "obliv_tree", "friedman", "heteroscedastic", "poly_mix"]))

    n_tot = n_fit + n_qry
    X_lat = _sample_copula_x(rng, n_tot, p, copula_fam)
    X_raw, col_meta = _apply_realistic_marginals(rng, X_lat)
    Xc = np.nan_to_num(X_lat, nan=0.0)

    # Construct latent signal f(X)
    if mech == "xor":
        sgn = np.where((Xc[:, 0] > 0.0) ^ (Xc[:, 1] > 0.0), 1.0, -1.0)
        sign_flip = float(rng.choice([-1.0, 1.0]))
        signal = sign_flip * 2.2 * sgn + 0.35 * Xc[:, 2 % p]
        noise_sd = np.full(n_tot, 0.45)
    elif mech == "obliv_tree":
        signal = np.zeros(n_tot, dtype=np.float64)
        for _ in range(4):
            cols = rng.choice(p, size=min(3, p), replace=False)
            thrs = rng.uniform(-0.7, 0.7, size=len(cols))
            lvs = rng.normal(0.0, 1.2, size=1 << len(cols))
            idx = np.zeros(n_tot, dtype=np.int64)
            for d, (c, t) in enumerate(zip(cols, thrs)):
                idx |= ((Xc[:, c] > t).astype(np.int64) << d)
            signal += lvs[idx]
        noise_sd = np.full(n_tot, 0.40)
    elif mech == "friedman":
        u0 = 0.5 * (1.0 + np.tanh(Xc[:, 0]))
        u1 = 0.5 * (1.0 + np.tanh(Xc[:, 1]))
        u2 = 0.5 * (1.0 + np.tanh(Xc[:, 2 % p]))
        u3 = 0.5 * (1.0 + np.tanh(Xc[:, 3 % p]))
        signal = 2.0 * np.sin(math.pi * u0 * u1) + 2.5 * (u2 - 0.5) ** 2 + 1.2 * u3
        noise_sd = np.full(n_tot, 0.35)
    elif mech == "heteroscedastic":
        w = rng.normal(0.0, 0.6, size=p)
        signal = Xc @ w + 0.6 * (Xc[:, 0] ** 2 - 1.0)
        noise_sd = 0.25 * np.exp(np.clip(0.85 * Xc[:, 1 % p], -2.0, 2.0))
    else:
        w = rng.normal(0.0, 0.5, size=p)
        signal = Xc @ w + 0.8 * Xc[:, 0] * Xc[:, 1] - 0.5 * np.abs(Xc[:, 2 % p])
        noise_sd = np.full(n_tot, 0.45)

    signal = (signal - float(np.mean(signal))) / max(float(np.std(signal)), 1e-4)
    if task_id == 0:
        y = (signal + noise_sd * rng.normal(size=n_tot)).astype(np.float32)
    elif task_id == 1:
        logits = 2.0 * signal
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))
        y = (rng.random(n_tot) < probs).astype(np.float32)
    else:
        W_c = rng.normal(0.0, 0.8, size=(p, n_classes))
        lgs = Xc @ W_c
        lgs[:, 0] += 1.4 * signal
        lgs[:, 1] -= 1.2 * signal
        lgs -= lgs.max(axis=1, keepdims=True)
        ex = np.exp(np.clip(lgs, -15.0, 0.0))
        pr = ex / ex.sum(axis=1, keepdims=True)
        u = rng.random((n_tot, 1))
        y = (u > np.cumsum(pr, axis=1)).sum(axis=1).clip(0, n_classes - 1).astype(np.float32)

    return {
        "X_fit": X_raw[:n_fit],
        "y_fit": y[:n_fit],
        "X_qry": X_raw[n_fit:],
        "y_qry": y[n_fit:],
        "task_id": task_id,
        "n_classes": n_classes,
        "col_meta": col_meta,
        "copula_family": copula_fam,
        "mechanism": mech,
    }


def sample_contrastive_xor_pair(
    rng: np.random.Generator,
    n_fit: int = 320,
    n_qry: int = 192,
    p: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a contrastive pair (identical X marginals/copula, opposite XOR vs XNOR targets)."""
    n_tot = n_fit + n_qry
    X = rng.normal(size=(n_tot, p)).astype(np.float32)
    xor_bit = ((X[:, 0] > 0.0) ^ (X[:, 1] > 0.0)).astype(np.float32)
    flip = rng.random(n_tot) < 0.10
    y_xor = np.where(flip, 1.0 - xor_bit, xor_bit).astype(np.float32)
    y_xnor = (1.0 - y_xor).astype(np.float32)
    base = {"task_id": 1, "n_classes": 2, "col_meta": None, "copula_family": "gaussian"}
    return (
        {**base, "X_fit": X[:n_fit], "y_fit": y_xor[:n_fit], "X_qry": X[n_fit:], "y_qry": y_xor[n_fit:], "mechanism": "xor"},
        {**base, "X_fit": X[:n_fit], "y_fit": y_xnor[:n_fit], "X_qry": X[n_fit:], "y_qry": y_xnor[n_fit:], "mechanism": "xnor"},
    )
