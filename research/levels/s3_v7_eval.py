"""Zero-safetensors-dependency held-out DEV evaluator for SingleEncV7 (strict MIA <= 0.55 guard).

Evaluates strictly on the isolated 30-table DEV hash list (`ag-dev-tables.txt`) and NEVER touches
the frozen CONFIRM panel. Uses only numpy, scipy, and scikit-learn (`n_jobs=1` for 1-thread/vCPU
compliance under the 72-vCPU ASR safety envelope).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass
try:
    from sklearn.ensemble import HistGradientBoostingClassifier as _GBC
    from sklearn.ensemble import HistGradientBoostingRegressor as _GBR
    _GB_KW: dict[str, Any] = {"max_iter": 40, "max_depth": 4, "random_state": 7}
except ImportError:
    _GBC, _GBR = GradientBoostingClassifier, GradientBoostingRegressor
    _GB_KW = {"n_estimators": 40, "max_depth": 4, "random_state": 7}
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def eval_regression(
    Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray,
    Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray,
) -> dict[str, float]:
    null_mse = float(mean_squared_error(yt, np.full_like(yt, float(np.mean(ya)))))
    rets, pars = [], []
    drv = float("nan")
    for name, ctor in [
        ("ridge", lambda: Ridge(alpha=1.0)),
        ("hgb", lambda: _GBR(**_GB_KW)),
    ]:
        ma, mb, ms = ctor().fit(Xa, ya), ctor().fit(Xb, yb), ctor().fit(Xs, ys)
        la = float(mean_squared_error(yt, ma.predict(Xt)))
        lb = float(mean_squared_error(yt, mb.predict(Xt)))
        ls = float(mean_squared_error(yt, ms.predict(Xt)))
        gain = null_mse - la
        if gain > 1e-4:
            rets.append((null_mse - ls) / gain)
            pars.append(1.0 if ls <= la + max(abs(lb - la) * 1.25, 0.03 * gain) else 0.0)
        if name == "ridge" and Xa.shape[1] >= 3:
            c = spearmanr(np.abs(ma.coef_), np.abs(ms.coef_))[0]
            if c is not None and math.isfinite(float(c)):
                drv = float(c)
    return {
        "retention": float(np.median(rets)) if rets else float("nan"),
        "parity": float(np.mean(pars)) if pars else float("nan"),
        "driver_spearman": drv,
    }


def eval_binary(
    Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray,
    Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray,
) -> dict[str, float]:
    ya_i, yb_i, yt_i = (ya > 0.5).astype(int), (yb > 0.5).astype(int), (yt > 0.5).astype(int)
    ys_i = (ys > 0.5).astype(int)
    if len(np.unique(ya_i)) < 2 or len(np.unique(yt_i)) < 2 or len(np.unique(ys_i)) < 2:
        return {"retention": float("nan"), "parity": float("nan"), "driver_spearman": float("nan")}
    p1 = float(np.clip(ya_i.mean(), 1e-4, 1.0 - 1e-4))
    null_ll = float(log_loss(yt_i, np.full(len(yt_i), p1), labels=[0, 1]))
    rets, pars = [], []
    drv = float("nan")
    for name, ctor in [
        ("logit", lambda: LogisticRegression(max_iter=200, C=1.0)),
        ("hgb", lambda: _GBC(**_GB_KW)),
    ]:
        ma, mb, ms = ctor().fit(Xa, ya_i), ctor().fit(Xb, yb_i), ctor().fit(Xs, ys_i)
        la = float(log_loss(yt_i, np.clip(ma.predict_proba(Xt)[:, 1], 1e-5, 1.0 - 1e-5), labels=[0, 1]))
        lb = float(log_loss(yt_i, np.clip(mb.predict_proba(Xt)[:, 1], 1e-5, 1.0 - 1e-5), labels=[0, 1]))
        ls = float(log_loss(yt_i, np.clip(ms.predict_proba(Xt)[:, 1], 1e-5, 1.0 - 1e-5), labels=[0, 1]))
        gain = null_ll - la
        if gain > 0.01:
            rets.append((null_ll - ls) / gain)
            pars.append(1.0 if ls <= la + max(abs(lb - la) * 1.25, 0.03 * gain) else 0.0)
        if name == "logit" and Xa.shape[1] >= 3:
            c = spearmanr(np.abs(ma.coef_[0]), np.abs(ms.coef_[0]))[0]
            if c is not None and math.isfinite(float(c)):
                drv = float(c)
    return {
        "retention": float(np.median(rets)) if rets else float("nan"),
        "parity": float(np.mean(pars)) if pars else float("nan"),
        "driver_spearman": drv,
    }


def eval_multiclass(
    Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray,
    Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray, k: int,
) -> dict[str, float]:
    ya_i = np.clip(np.rint(ya).astype(int), 0, k - 1)
    yb_i = np.clip(np.rint(yb).astype(int), 0, k - 1)
    ys_i = np.clip(np.rint(ys).astype(int), 0, k - 1)
    yt_i = np.clip(np.rint(yt).astype(int), 0, k - 1)
    if len(np.unique(ya_i)) < 2 or len(np.unique(ys_i)) < 2:
        return {"retention": float("nan"), "parity": float("nan"), "driver_spearman": float("nan")}
    pri = np.bincount(ya_i, minlength=k).astype(np.float64) + 0.5
    pri /= pri.sum()
    labels = list(range(k))
    null_ll = float(log_loss(yt_i, np.tile(pri, (len(yt_i), 1)), labels=labels))
    ma = LogisticRegression(max_iter=200, C=1.0).fit(Xa, ya_i)
    mb = LogisticRegression(max_iter=200, C=1.0).fit(Xb, yb_i)
    ms = LogisticRegression(max_iter=200, C=1.0).fit(Xs, ys_i)

    def full_probs(clf: Any) -> np.ndarray:
        raw = clf.predict_proba(Xt)
        out = np.full((len(Xt), k), 1e-5, dtype=np.float64)
        for ci, c in enumerate(clf.classes_):
            out[:, int(c)] = raw[:, ci]
        return out / out.sum(axis=1, keepdims=True)

    la = float(log_loss(yt_i, full_probs(ma), labels=labels))
    lb = float(log_loss(yt_i, full_probs(mb), labels=labels))
    ls = float(log_loss(yt_i, full_probs(ms), labels=labels))
    gain = null_ll - la
    if gain <= 0.02:
        return {"retention": float("nan"), "parity": float("nan"), "driver_spearman": float("nan")}
    drv = float("nan")
    if Xa.shape[1] >= 3 and ma.coef_.shape[1] == ms.coef_.shape[1]:
        c = spearmanr(np.mean(np.abs(ma.coef_), axis=0), np.mean(np.abs(ms.coef_), axis=0))[0]
        if c is not None and math.isfinite(float(c)):
            drv = float(c)
    return {
        "retention": (null_ll - ls) / gain,
        "parity": 1.0 if ls <= la + max(abs(lb - la) * 1.25, 0.03 * gain) else 0.0,
        "driver_spearman": drv,
    }


def eval_realism_and_privacy(
    Xa: np.ndarray, Xb: np.ndarray, Xs: np.ndarray, Xt: np.ndarray,
) -> dict[str, float]:
    """Empirical C2ST, MargAUC, DCR, MIA (<= 0.55 guard), exact-copy excess (non-DP / non-HIPAA)."""
    m = min(len(Xt), len(Xs), 180)
    X_all = np.vstack([Xt[:m], Xs[:m]])
    y_all = np.concatenate([np.zeros(m, dtype=int), np.ones(m, dtype=int)])
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)
    oof = np.zeros(len(y_all), dtype=np.float64)
    for tr, va in skf.split(X_all, y_all):
        clf = RandomForestClassifier(n_estimators=24, max_depth=5, random_state=17, n_jobs=1)
        clf.fit(X_all[tr], y_all[tr])
        oof[va] = clf.predict_proba(X_all[va])[:, 1]
    c2st = float(roc_auc_score(y_all, oof))
    c2st = max(c2st, 1.0 - c2st)

    marg_aucs = []
    for j in range(min(Xt.shape[1], 12)):
        col_r, col_s = Xt[:m, j], Xs[:m, j]
        if float(np.std(col_r)) < 1e-7 and float(np.std(col_s)) < 1e-7:
            marg_aucs.append(0.5)
            continue
        vals = np.concatenate([col_r, col_s])
        lbl = np.concatenate([np.zeros(len(col_r)), np.ones(len(col_s))])
        auc_j = float(roc_auc_score(lbl, vals))
        marg_aucs.append(max(auc_j, 1.0 - auc_j))
    marg_auc = float(np.median(marg_aucs)) if marg_aucs else 0.5

    mp = min(len(Xa), len(Xb), len(Xs), 120)
    Sa, Ra, Rb = Xs[:mp], Xa[:mp], Xb[:mp]
    da_mat = np.sqrt(((Sa[:, None, :] - Ra[None, :, :]) ** 2).sum(axis=2))
    db_mat = np.sqrt(((Sa[:, None, :] - Rb[None, :, :]) ** 2).sum(axis=2))
    da, db = da_mat.min(axis=1), db_mat.min(axis=1)
    dcr_share = float(np.mean((da + 1e-9 < db).astype(np.float64) + 0.5 * (np.abs(da - db) <= 1e-9)))
    d_mem, d_non = da_mat.min(axis=0), db_mat.min(axis=0)
    mia_y = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))])
    mia_auc = float(roc_auc_score(mia_y, -np.concatenate([d_mem, d_non])))
    mia_auc = max(mia_auc, 1.0 - mia_auc)
    ex_excess = max(0.0, float(np.mean(da <= 1e-6)) - float(np.mean(db <= 1e-6)))

    return {
        "c2st_auc": c2st,
        "marg_auc": marg_auc,
        "realism_pass": 1.0 if c2st <= 0.70 else 0.0,
        "dcr_share": dcr_share,
        "mia_auc": mia_auc,
        "exact_copy_excess": ex_excess,
        "privacy_pass": 1.0 if (dcr_share <= 0.55 and mia_auc <= 0.55 and ex_excess <= 0.005) else 0.0,
    }
