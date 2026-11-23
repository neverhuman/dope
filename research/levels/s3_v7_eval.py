"""Zero-safetensors-dependency held-out DEV evaluator for SingleEncV7 (strict MIA <= 0.55 guard).

Evaluates strictly on the isolated 30-table DEV hash list (`ag-dev-tables.txt`) and NEVER touches
the frozen CONFIRM panel. Uses only numpy, scipy, and scikit-learn (`n_jobs=1` for 1-thread/vCPU
compliance under the 72-vCPU ASR safety envelope).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import struct
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    part.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
        fh.flush()


def save_v7_safetensors(model: Any, out_path: Path) -> tuple[int, str]:
    """Write canonical HF safetensors v1 (F32, rank <= 2, 8-byte aligned JSON header, zero deps)."""
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    sd = model.state_dict()
    for name in sorted(sd):
        t = sd[name].detach().cpu().to(torch.float32).contiguous()
        if t.ndim > 2:
            raise ValueError(f"v7_tensor_rank_violation:{name}:ndim={t.ndim}")
        raw = t.numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hdr_bytes += b" " * ((8 - (len(hdr_bytes) % 8)) % 8)
    payload = struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + b"".join(blobs)
    part = out_path.with_suffix(".safetensors.part")
    part.write_bytes(payload)
    part.replace(out_path)
    return len(payload), hashlib.sha256(payload).hexdigest()


def save_v7_resume_state(
    out_path: Path,
    base_model: Any,
    ema_model: Any,
    opt: Any,
    step: int,
    opt_steps: int,
    total_unique: int,
    best_dev_kpi: float,
    meta: dict[str, Any],
) -> None:
    """Atomically save full training state (model + EMA + AdamW + RNG + step) for clean extension."""
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "step": int(step),
        "opt_steps": int(opt_steps),
        "total_unique": int(total_unique),
        "best_dev_kpi": float(best_dev_kpi),
        "meta": dict(meta),
        "base_model": {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
        "ema_model": {k: v.detach().cpu() for k, v in ema_model.state_dict().items()},
        "optimizer": opt.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    part = out_path.with_suffix(out_path.suffix + ".part")
    torch.save(payload, part)
    part.replace(out_path)


def load_v7_resume_state(
    resume_path: Path,
    base_model: Any,
    ema_model: Any,
    opt: Any,
    expected_arch: tuple[int, int, int, int],
) -> tuple[int, int, int, float]:
    """Load full training state (model + EMA + AdamW + RNG + step) with architecture guard."""
    import torch

    state = torch.load(resume_path, map_location="cpu", weights_only=False)
    meta = state.get("meta") or {}
    saved_arch = (
        int(meta.get("d_model", -1)),
        int(meta.get("n_layers", -1)),
        int(meta.get("n_heads", -1)),
        int(meta.get("h_target", -1)),
    )
    if saved_arch != expected_arch:
        raise ValueError(f"resume_arch_mismatch:saved={saved_arch}:expected={expected_arch}")
    base_model.load_state_dict(state["base_model"], strict=True)
    ema_model.load_state_dict(state["ema_model"], strict=True)
    opt.load_state_dict(state["optimizer"])
    if "torch_rng" in state:
        torch.set_rng_state(state["torch_rng"])
    if "numpy_rng" in state:
        np.random.set_state(state["numpy_rng"])
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    return (
        int(state["step"]),
        int(state.get("opt_steps", state["step"])),
        int(state.get("total_unique", 0)),
        float(state.get("best_dev_kpi", -999.0)),
    )
