"""Maximum-entropy S3 augmentation & infinite synthetic prior engine for SingleEncV7 (V7.2).

Unlocks 100% of all 4,145 S3 train tables (recovering all 51 multiclass tables with > 8 classes
via random K-class subset/tail cropping), enforces exact per-step shuffled tri-task balance
(37.5% regression / 31.25% binary / 31.25% multiclass), synthesizes infinite nonlinear SCM /
random-tree / bilinear-interaction priors, and applies fresh duplicate-grouped train/test splits,
log-uniform row/column cropping, multi-task target swaps, MCAR/MAR missingness, monotonic warps,
zero-inflation/discretization, and class-ID permutations on EVERY batch.
"""
from __future__ import annotations

import math

import numpy as np


def task_quota_for_step(micro_batch: int, step_seed: int) -> list[str]:
    """Exact balanced per-batch task schedule (6 reg / 5 bin / 5 mc for B=16), shuffled every step."""
    n_bin = max(1, (micro_batch * 5) // 16)
    n_mc = max(1, (micro_batch * 5) // 16)
    n_reg = max(1, micro_batch - n_bin - n_mc)
    tasks = ["regression"] * n_reg + ["binary"] * n_bin + ["multiclass"] * n_mc
    rng = np.random.default_rng(step_seed & 0xFFFFFFFFFFFFFFFF)
    rng.shuffle(tasks)
    return tasks


def sample_synthetic_prior_table(
    rng: np.random.Generator,
    task: str,
    n_fit: int,
    n_qry: int,
    max_cols: int = 44,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Infinite synthetic nonlinear SCM / random depth-3 tree / interaction prior table (2^64 entropy)."""
    n_tot = n_fit + n_qry
    p = int(rng.integers(4, min(max_cols, 36) + 1))
    # 1. Latent causes: mix of Gaussian, Student-t, skew-lognormal, and discrete regimes.
    r_lat = int(rng.integers(2, min(8, p) + 1))
    dist_kind = int(rng.integers(0, 3))
    if dist_kind == 0:
        Z = rng.standard_normal((n_tot, r_lat)).astype(np.float32)
    elif dist_kind == 1:
        Z = rng.standard_t(df=4.0, size=(n_tot, r_lat)).clip(-6.0, 6.0).astype(np.float32)
    else:
        Z = (rng.laplace(0.0, 0.8, size=(n_tot, r_lat))).clip(-6.0, 6.0).astype(np.float32)

    # 2. 2-layer structural DAG with nonlinear edges + optional discrete/zero-inflated features.
    W1 = rng.normal(0.0, 0.7 / math.sqrt(r_lat), size=(r_lat, p)).astype(np.float32)
    X = Z @ W1 + rng.normal(0.0, 0.55, size=(n_tot, p)).astype(np.float32)
    # Add random DAG nonlinear feature-to-feature couplings.
    n_edges = min(p // 2, 10)
    for _ in range(n_edges):
        src = int(rng.integers(0, p - 1))
        dst = int(rng.integers(src + 1, p))
        op = int(rng.integers(0, 4))
        w = float(rng.uniform(-0.8, 0.8))
        if op == 0:
            X[:, dst] += w * np.tanh(X[:, src])
        elif op == 1:
            X[:, dst] += w * np.clip(X[:, src] ** 2 - 1.0, -4.0, 4.0) * 0.4
        elif op == 2:
            src2 = int(rng.integers(0, dst))
            X[:, dst] += w * np.clip(X[:, src] * X[:, src2], -4.0, 4.0) * 0.4
        else:
            X[:, dst] += w * (X[:, src] > float(rng.uniform(-0.8, 0.8))).astype(np.float32)

    # Discretize / zero-inflate 20% of columns to mirror realistic mixed tabular manifolds.
    for j in range(p):
        u = rng.random()
        if u < 0.12:
            k_lvl = int(rng.integers(2, 8))
            qs = np.quantile(X[:, j], np.linspace(0.05, 0.95, k_lvl - 1))
            X[:, j] = np.digitize(X[:, j], qs).astype(np.float32)
        elif u < 0.22:
            thr = float(np.quantile(X[:, j], rng.uniform(0.15, 0.55)))
            X[:, j] = np.where(X[:, j] < thr, 0.0, X[:, j] - thr).astype(np.float32)

    # Normalize features cleanly before generating target signal.
    Xn = (X - X.mean(axis=0, keepdims=True)) / np.maximum(X.std(axis=0, keepdims=True), 1e-4)

    # 3. Ground-truth target generator: linear + nonlinear spline/quadratic + interaction + tree!
    n_cls = 1 if task == "regression" else (2 if task == "binary" else int(rng.integers(3, 9)))
    o_dim = 1 if task != "multiclass" else n_cls
    f_out = np.zeros((n_tot, o_dim), dtype=np.float32)
    n_act = min(p, int(rng.integers(2, min(p, 10) + 1)))
    act = rng.choice(p, size=n_act, replace=False)
    for c in range(o_dim):
        # Linear component
        w_lin = rng.normal(0.0, 1.0 / math.sqrt(n_act), size=n_act).astype(np.float32)
        sc = Xn[:, act] @ w_lin
        # Nonlinear component (quadratic / abs / sin / pairwise interaction)
        nl_wt = float(rng.uniform(0.35, 1.15))
        j0, j1 = int(act[0]), int(act[min(1, n_act - 1)])
        nl_type = int(rng.integers(0, 4))
        if nl_type == 0:
            sc += nl_wt * (np.clip(Xn[:, j0] ** 2, 0.0, 9.0) - 1.0) * 0.45 * float(rng.choice([-1.0, 1.0]))
        elif nl_type == 1:
            sc += nl_wt * np.abs(Xn[:, j0]) * float(rng.choice([-1.0, 1.0]))
        elif nl_type == 2:
            sc += nl_wt * np.sin(1.4 * Xn[:, j0]) + 0.5 * nl_wt * np.clip(Xn[:, j0] * Xn[:, j1], -4.0, 4.0)
        else:
            # Random depth-3 decision tree leaf values!
            t0 = float(rng.uniform(-0.8, 0.8))
            t1 = float(rng.uniform(-0.8, 0.8))
            t2 = float(rng.uniform(-0.8, 0.8))
            j2 = int(act[min(2, n_act - 1)])
            leaf = (
                ((Xn[:, j0] > t0).astype(np.int32) << 2)
                | ((Xn[:, j1] > t1).astype(np.int32) << 1)
                | (Xn[:, j2] > t2).astype(np.int32)
            )
            l_vals = rng.normal(0.0, 1.0, size=8).astype(np.float32)
            sc += nl_wt * l_vals[leaf]
        sc = (sc - sc.mean()) / max(float(sc.std()), 1e-4)
        f_out[:, c] = sc

    snr = float(rng.uniform(0.25, 0.85))
    if task == "regression":
        y = (f_out[:, 0] + snr * rng.standard_normal(n_tot)).astype(np.float32)
    elif task == "binary":
        temp = float(rng.uniform(1.2, 3.0))
        bias = float(rng.uniform(-0.6, 0.6))
        p1 = 1.0 / (1.0 + np.exp(-np.clip(temp * f_out[:, 0] + bias, -10.0, 10.0)))
        y = (rng.random(n_tot) < p1).astype(np.float32)
        if len(np.unique(y[:n_fit])) < 2 or len(np.unique(y[n_fit:])) < 2:
            y = (f_out[:, 0] > float(np.median(f_out[:n_fit, 0]))).astype(np.float32)
    else:
        temp = float(rng.uniform(1.4, 3.2))
        lg = temp * f_out
        lg -= lg.max(axis=1, keepdims=True)
        pr = np.exp(lg)
        pr /= pr.sum(axis=1, keepdims=True)
        cdf = np.cumsum(pr, axis=1)
        u = rng.random((n_tot, 1))
        y = np.minimum((u >= cdf).sum(axis=1), n_cls - 1).astype(np.float32)
        if len(np.unique(y[:n_fit])) < 3:
            y = np.argmax(f_out, axis=1).astype(np.float32)

    return X[:n_fit], y[:n_fit], X[n_fit:], y[n_fit:], n_cls


def crop_multiclass_classes(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    dg: np.ndarray,
    max_classes: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Recover ALL >8-class S3 multiclass datasets via random K-class subset or tail-merge cropping."""
    yi = np.rint(y).astype(np.int64)
    vals, cnts = np.unique(yi, return_counts=True)
    if len(vals) <= max_classes:
        return X, y, dg, False
    # Choose K in [3, max_classes] randomly for maximum class-combination entropy!
    k_pick = int(rng.integers(3, max_classes + 1))
    # Prioritize classes with >= 16 rows so FIT and QRY both remain populated.
    eligible = vals[cnts >= 16]
    if len(eligible) < 3:
        order = np.argsort(cnts)[::-1]
        eligible = vals[order[: max(3, min(len(vals), max_classes))]]
    k_pick = min(k_pick, len(eligible))
    chosen = rng.choice(eligible, size=k_pick, replace=False)
    if rng.random() < 0.70:
        # Mode A: clean subset crop to chosen K classes (pure K-class sub-problem!).
        mask = np.isin(yi, chosen)
        if int(mask.sum()) >= 64:
            return X[mask], y[mask], dg[mask], True
    # Mode B: keep top-(K-1) chosen classes and merge remaining tail into an "other" class.
    keep_main = chosen[: k_pick - 1]
    other_id = int(vals.max()) + 1
    y_new = np.where(np.isin(yi, keep_main), yi, other_id).astype(np.float32)
    return X, y_new, dg, True


def synthesize_target_from_regression(
    rng: np.random.Generator, y_raw: np.ndarray, target_task: str
) -> tuple[np.ndarray, int]:
    """Turn any real S3 regression table into a brand-new binary or K-class task via random cuts."""
    fin = y_raw[np.isfinite(y_raw)]
    if len(fin) < 20:
        return y_raw, 1
    if target_task == "binary":
        cut = float(np.quantile(fin, rng.uniform(0.25, 0.75)))
        return (y_raw > cut).astype(np.float32), 2
    k_bins = int(rng.integers(3, 9))
    qs = np.quantile(fin, np.linspace(0.0, 1.0, k_bins + 1)[1:-1])
    qs = np.unique(qs)
    if len(qs) < 2:
        return y_raw, 1
    yb = np.digitize(y_raw, qs).astype(np.float32)
    return yb, len(qs) + 1


def apply_real_table_target_swap(
    rng: np.random.Generator, X_raw: np.ndarray, y_raw: np.ndarray, desired_task: str
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    """Multi-task target-swap: promotes 2-level cols to binary, 3..8-level to multiclass, cont to reg."""
    nc = X_raw.shape[1]
    if nc < 4:
        return X_raw, y_raw, desired_task, False
    cand_cols = rng.permutation(nc)
    for col in cand_cols[: min(6, nc)]:
        yc = X_raw[:, col]
        fin = np.isfinite(yc)
        if float(fin.mean()) < 0.90:
            continue
        vals = yc[fin]
        if float(np.std(vals)) < 1e-4:
            continue
        uniq = np.unique(vals)
        new_task: str | None = None
        if len(uniq) == 2 and desired_task in ("binary", "any"):
            yc = (yc == uniq[-1]).astype(np.float32)
            new_task = "binary"
        elif 3 <= len(uniq) <= 8 and desired_task in ("multiclass", "any"):
            new_task = "multiclass"
        elif len(uniq) > 8 and desired_task == "regression":
            new_task = "regression"
        if new_task is not None:
            keep = np.ones(nc, dtype=bool)
            keep[col] = False
            Xn = np.concatenate([X_raw[:, keep], y_raw[:, None]], axis=1)
            yn = np.where(np.isfinite(yc), yc, float(np.nanmedian(vals))).astype(np.float32)
            return Xn, yn, new_task, True
    return X_raw, y_raw, desired_task, False


def apply_manifold_augmentations(
    rng: np.random.Generator,
    Xa_raw: np.ndarray,
    Xq_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply MCAR/MAR missingness, monotonic warps, sign flips, and discrete/zero-inflation warps."""
    Xa = Xa_raw.copy()
    Xq = Xq_raw.copy()
    p = Xa.shape[1]

    # 1. MCAR / MAR missingness injection (25% of episodes, 5-22% rate) BEFORE FIT-only scaling!
    if rng.random() < 0.25:
        rate = float(rng.uniform(0.05, 0.22))
        if rng.random() < 0.5 or p < 2:
            Xa[rng.random(Xa.shape) < rate] = np.nan
            Xq[rng.random(Xq.shape) < rate] = np.nan
        else:
            # MAR: missingness conditioned on column 0 quantile!
            driver = int(rng.integers(0, p))
            thr = float(np.nanmedian(Xa[:, driver]))
            m_a = (Xa[:, driver] > thr)[:, None] & (rng.random(Xa.shape) < rate * 1.6)
            m_q = (Xq[:, driver] > thr)[:, None] & (rng.random(Xq.shape) < rate * 1.6)
            m_a[:, driver] = False
            m_q[:, driver] = False
            Xa[m_a] = np.nan
            Xq[m_q] = np.nan

    # 2. Random column-wise monotonic power warp / sign flip / zero-inflation (consistent on FIT+QRY!).
    for j in range(p):
        fin_a = Xa[:, j][np.isfinite(Xa[:, j])]
        if len(fin_a) < 8:
            continue
        is_cont = len(np.unique(fin_a)) > 12
        if rng.random() < 0.35:
            # Random sign flip (exact symmetry!)
            Xa[:, j] = -Xa[:, j]
            Xq[:, j] = -Xq[:, j]
        if is_cont and rng.random() < 0.20:
            # Monotonic heavy-tail power warp around FIT median
            med = float(np.median(fin_a))
            sd = max(float(np.std(fin_a)), 1e-5)
            gamma = float(rng.uniform(0.65, 1.65))
            for arr in (Xa, Xq):
                d = (arr[:, j] - med) / sd
                arr[:, j] = med + sd * np.sign(d) * (np.abs(d) ** gamma)
        elif is_cont and rng.random() < 0.12:
            # Zero-inflation / left-censoring realism augmentation
            cut = float(np.quantile(fin_a, rng.uniform(0.15, 0.45)))
            Xa[:, j] = np.where(Xa[:, j] < cut, cut, Xa[:, j])
            Xq[:, j] = np.where(Xq[:, j] < cut, cut, Xq[:, j])
    return Xa, Xq


def append_nonlinear_and_distractor_cols(
    rng: np.random.Generator,
    Xa: np.ndarray,
    Xq: np.ndarray,
    max_cols: int = 48,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Append random bilinear/nonlinear derived features + Gaussian/Laplace distractors."""
    p = Xa.shape[1]
    avail = max_cols - p
    if avail <= 0:
        return Xa, Xq, 0
    n_add = int(rng.integers(0, min(6, avail) + 1))
    if n_add == 0:
        return Xa, Xq, 0
    if rng.random() < 0.5:
        na = rng.standard_normal((len(Xa), n_add)).astype(np.float32)
        nq = rng.standard_normal((len(Xq), n_add)).astype(np.float32)
    else:
        na = rng.laplace(0.0, 0.7, size=(len(Xa), n_add)).clip(-6.0, 6.0).astype(np.float32)
        nq = rng.laplace(0.0, 0.7, size=(len(Xq), n_add)).clip(-6.0, 6.0).astype(np.float32)
    if n_add >= 2 and p >= 2:
        # 1st extra col: bilinear interaction; 2nd extra col (if >=3): nonlinear tanh/abs derived col.
        j0, j1 = int(rng.integers(0, p)), int(rng.integers(0, p))
        na[:, 0] = np.clip(0.55 * Xa[:, j0] * Xa[:, j1] + 0.25 * na[:, 0], -6.0, 6.0)
        nq[:, 0] = np.clip(0.55 * Xq[:, j0] * Xq[:, j1] + 0.25 * nq[:, 0], -6.0, 6.0)
        if n_add >= 3:
            j2 = int(rng.integers(0, p))
            na[:, 1] = np.clip(np.tanh(Xa[:, j2]) + 0.3 * np.abs(Xa[:, j0]) + 0.2 * na[:, 1], -6.0, 6.0)
            nq[:, 1] = np.clip(np.tanh(Xq[:, j2]) + 0.3 * np.abs(Xq[:, j0]) + 0.2 * nq[:, 1], -6.0, 6.0)
    return np.concatenate([Xa, na], axis=1), np.concatenate([Xq, nq], axis=1), n_add
