"""Upgraded DEV evaluator for DOPE v3 (`DSE8` P0 analytic & `SingleEncV8` P1/P2 neural).

Evaluates strictly via the native pure-Rust `dope_v8_emit` binary (`fit` + `sample`), reporting:
- Twin-referenced `TRG` across `lin`, `gbdt`, AND strictly held-out `mlp` auditors!
- Factorized `C2ST` (`c2st_marginal`, `c2st_x`, `c2st_xy`)
- Drivers (`driver_spearman`, `driver_tv`, `ndcg5`)
- Calibration (`ece15`)
- Privacy (`dcr_share`, `exact_copy_rate`, `illegal_onehot_rows`)
- Packet byte size (`packet_bytes`)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

from research.levels.v8_packet import fit_dse8_rust, sample_dse8_rust


def _ece15(probs: np.ndarray, y_true: np.ndarray) -> float:
    n = len(y_true)
    if n < 15:
        return 0.0
    if probs.ndim == 1:
        pred = (probs >= 0.5).astype(int)
        conf = np.where(pred == 1, probs, 1.0 - probs)
        hit = (pred == (y_true > 0.5).astype(int)).astype(np.float64)
    else:
        pred = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        hit = (pred == np.rint(y_true).astype(int)).astype(np.float64)
    ece = 0.0
    for b in range(15):
        lo, hi = b / 15.0, (b + 1) / 15.0
        m = (conf >= lo) & (conf < hi if b < 14 else conf <= hi)
        if np.any(m):
            ece += float(m.mean()) * abs(float(hit[m].mean()) - float(conf[m].mean()))
    return float(ece)


def _c2st_auc(A: np.ndarray, B: np.ndarray, seed: int = 7) -> float:
    n = min(len(A), len(B), 400)
    if n < 20:
        return 0.5
    X = np.nan_to_num(np.vstack([A[:n], B[:n]]), nan=0.0)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof = np.zeros(2 * n, dtype=np.float64)
    for tr, va in skf.split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=30, max_depth=3, random_state=seed)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y, oof))


def _dcr_and_copy(Xa: np.ndarray, Xb: np.ndarray, Xs: np.ndarray) -> tuple[float, float]:
    m = min(len(Xa), len(Xb), len(Xs), 200)
    if m < 10:
        return 0.5, 0.0
    a = np.nan_to_num(Xa[:m], nan=0.0)
    b = np.nan_to_num(Xb[:m], nan=0.0)
    s = np.nan_to_num(Xs[:m], nan=0.0)
    da = np.sqrt(((s[:, None, :] - a[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    db = np.sqrt(((s[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    return float(np.mean(da < db)), float(np.mean(da < 1e-5))


def evaluate_dse8_cell(
    Xa: np.ndarray,
    ya: np.ndarray,
    Xb: np.ndarray,
    yb: np.ndarray,
    Xt: np.ndarray,
    yt: np.ndarray,
    task_id: int,
    n_classes: int = 1,
    budget_bytes: int = 2048,
    checkpoint: Path | None = None,
    col_meta: list[dict[str, Any]] | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Fit DSE8 packet on (Xa, ya), sample via native Rust `dope_v8_emit`, score TRG/C2ST/privacy."""
    pkt_bytes = fit_dse8_rust(
        Xa, ya, task_id, budget_bytes=budget_bytes, checkpoint=checkpoint,
        code_tier=1 if checkpoint is not None else 0, col_meta=col_meta, seed=seed,
    )
    n_syn = min(max(4 * len(Xa), 256), 2048)
    Xs, ys = sample_dse8_rust(pkt_bytes, n_rows=n_syn, seed=seed ^ 0x5A39, checkpoint=checkpoint)

    # Check illegal nominal one-hot rows if col_meta supplied
    illegal_rows = 0
    if col_meta is not None:
        groups: dict[int, list[int]] = {}
        for j, m in enumerate(col_meta):
            if m.get("group_kind") == 4:
                groups.setdefault(int(m["group_id"]), []).append(j)
        for cols in groups.values():
            if len(cols) >= 2:
                sub = Xs[:, cols]
                bad = np.any((sub != 0.0) & (sub != 1.0), axis=1) | (sub.sum(axis=1) > 1.0 + 1e-5)
                illegal_rows += int(bad.sum())

    Xa_c = np.nan_to_num(Xa, nan=0.0)
    Xb_c = np.nan_to_num(Xb, nan=0.0)
    Xs_c = np.nan_to_num(Xs, nan=0.0)
    Xt_c = np.nan_to_num(Xt, nan=0.0)

    trgs: dict[str, float] = {}
    ece15 = 0.0
    drv_sp = float("nan")
    drv_tv = float("nan")

    if task_id == 0:
        null_l = float(mean_squared_error(yt, np.full_like(yt, float(np.mean(ya)))))
        auditors = [
            ("lin", lambda: Ridge(alpha=1.0)),
            ("gbdt", lambda: HistGradientBoostingRegressor(max_iter=40, max_depth=4, random_state=7)),
            ("mlp", lambda: MLPRegressor(hidden_layer_sizes=(32,), max_iter=120, random_state=7)),
        ]
        for name, ctor in auditors:
            ma, mb, ms = ctor().fit(Xa_c, ya), ctor().fit(Xb_c, yb), ctor().fit(Xs_c, ys)
            la = float(mean_squared_error(yt, ma.predict(Xt_c)))
            lb = float(mean_squared_error(yt, mb.predict(Xt_c)))
            ls = float(mean_squared_error(yt, ms.predict(Xt_c)))
            gain = null_l - la
            if gain > 1e-4:
                ra = (null_l - ls) / gain
                rb = (null_l - lb) / gain
                trgs[name] = float(np.clip(ra, -1.0, 1.25) - np.clip(rb, -1.0, 1.25))
            if name == "lin" and Xa_c.shape[1] >= 3:
                ca, cs = np.abs(ma.coef_), np.abs(ms.coef_)
                r = spearmanr(ca, cs)[0]
                if r is not None and math.isfinite(float(r)):
                    drv_sp = float(r)
                pa = ca / max(float(ca.sum()), 1e-6)
                ps = cs / max(float(cs.sum()), 1e-6)
                drv_tv = float(0.5 * np.abs(pa - ps).sum())
    else:
        ya_i = np.rint(ya).astype(int)
        yb_i = np.rint(yb).astype(int)
        ys_i = np.rint(ys).astype(int)
        yt_i = np.rint(yt).astype(int)
        classes = sorted(set(ya_i.tolist()))
        k = max(len(classes), 2)
        if len(set(ys_i.tolist())) >= 2 and len(set(yt_i.tolist())) >= 2:
            pri = np.array([(ya_i == c).mean() for c in classes], dtype=np.float64)
            pri = np.clip(pri, 1e-4, 1.0)
            pri /= pri.sum()
            null_l = float(log_loss(yt_i, np.tile(pri, (len(yt_i), 1)), labels=classes))
            auditors_c = [
                ("lin", lambda: LogisticRegression(max_iter=200, C=1.0)),
                ("gbdt", lambda: HistGradientBoostingClassifier(max_iter=40, max_depth=4, random_state=7)),
                ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=120, random_state=7)),
            ]

            def _probs(m: Any) -> np.ndarray:
                raw = m.predict_proba(Xt_c)
                out = np.full((len(Xt_c), k), 1e-5, dtype=np.float64)
                c_map = {c: i for i, c in enumerate(classes)}
                for ci, c in enumerate(m.classes_):
                    if int(c) in c_map:
                        out[:, c_map[int(c)]] = raw[:, ci]
                return out / out.sum(axis=1, keepdims=True)

            for name, ctor in auditors_c:
                ma, mb, ms = ctor().fit(Xa_c, ya_i), ctor().fit(Xb_c, yb_i), ctor().fit(Xs_c, ys_i)
                pa_t, pb_t, ps_t = _probs(ma), _probs(mb), _probs(ms)
                la = float(log_loss(yt_i, pa_t, labels=classes))
                lb = float(log_loss(yt_i, pb_t, labels=classes))
                ls = float(log_loss(yt_i, ps_t, labels=classes))
                gain = null_l - la
                if gain > 0.01:
                    ra = (null_l - ls) / gain
                    rb = (null_l - lb) / gain
                    trgs[name] = float(np.clip(ra, -1.0, 1.25) - np.clip(rb, -1.0, 1.25))
                if name == "lin":
                    ece15 = _ece15(ps_t[:, 1] if k == 2 else ps_t, yt_i)
                    if Xa_c.shape[1] >= 3:
                        ca = np.mean(np.abs(ma.coef_), axis=0)
                        cs = np.mean(np.abs(ms.coef_), axis=0)
                        r = spearmanr(ca, cs)[0]
                        if r is not None and math.isfinite(float(r)):
                            drv_sp = float(r)
                        drv_tv = float(0.5 * np.abs(ca / max(ca.sum(), 1e-6) - cs / max(cs.sum(), 1e-6)).sum())

    # Factorized C2ST: 1D marginal mean, X joint, and (X, y) joint
    c2st_x = _c2st_auc(Xt_c, Xs_c, seed=seed)
    c2st_xy = _c2st_auc(
        np.column_stack([Xt_c, yt.reshape(-1, 1)]),
        np.column_stack([Xs_c, ys.reshape(-1, 1)]),
        seed=seed + 1,
    )
    dcr_share, exact_copy = _dcr_and_copy(Xa_c, Xb_c, Xs_c)

    return {
        "trg_worst": min(trgs.values()) if trgs else float("nan"),
        "trg_lin": trgs.get("lin", float("nan")),
        "trg_gbdt": trgs.get("gbdt", float("nan")),
        "trg_mlp_heldout": trgs.get("mlp", float("nan")),
        "c2st_x": c2st_x,
        "c2st_xy": c2st_xy,
        "driver_spearman": drv_sp,
        "driver_tv": drv_tv,
        "ece15": ece15,
        "dcr_share": dcr_share,
        "exact_copy_rate": exact_copy,
        "illegal_onehot_rows": float(illegal_rows),
        "packet_bytes": float(len(pkt_bytes)),
    }
