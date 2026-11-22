"""Continuous unseen-family Main-KPI Evaluator + Best Checkpoint & JopeDime k=3 Valley Soup Promoter.

Evaluates BOTH `DopeSingleEncV6` (strict zero-raw-row single-encoding decode!) and `DopeFlowV6`
checkpoints on held-out family-disjoint S3 validation tables across ALL 4 tasks:
  - Regression, Binary, Multiclass, Multi-Label
  - Downstream auditors: Ridge + Logistic + HistGradientBoosting (TSTR vs TRTR parity)
  - Realism: RF C2ST AUC + Median Column MargAUC
  - Driver alignment: Spearman feature-importance correlation
  - Empirical Privacy (non-DP / non-HIPAA): DCR Share, Distance MIA AUC, Twin-B Exact-Copy Excess
Automatically preserves `best_main_kpi.safetensors`, builds & evaluates JopeDime's `k=3` valley
parameter soup (`best_soup_k3.safetensors`), updates `checkpoints/ledger.jsonl`, and writes
`curriculum_regret.json` so the live trainer dynamically upsamples hard tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any
import warnings

import numpy as np
from safetensors.torch import load_file, save_file
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch

from .s3_dynamic_augment import (
    S3CorpusStore,
    _duplicate_grouped_split,
    _fit_only_nc2_scale,
    _synthesize_multilabel_targets,
)
from .v5_fidelity_net import DopeFidelityV5
from .v6_single_encoding import DopeSingleEncV6

warnings.filterwarnings("ignore")


def _eval_regression(Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray, Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray) -> dict[str, float]:
    null_mse = float(mean_squared_error(yt, np.full_like(yt, float(np.mean(ya)))))
    rets, pars = [], []
    drv = float("nan")
    for name, ctor in [
        ("ridge", lambda: Ridge(alpha=1.0)),
        ("hgb", lambda: HistGradientBoostingRegressor(max_iter=40, max_depth=4, random_state=7)),
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
            c = spearmanr(np.abs(ma.coef_), np.abs(ms.coef_)).statistic
            if c is not None and math.isfinite(float(c)):
                drv = float(c)
    return {
        "retention": float(np.median(rets)) if rets else float("nan"),
        "parity": float(np.mean(pars)) if pars else float("nan"),
        "driver_spearman": drv,
    }


def _eval_binary(Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray, Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray) -> dict[str, float]:
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
        ("hgb", lambda: HistGradientBoostingClassifier(max_iter=40, max_depth=4, random_state=7)),
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
            c = spearmanr(np.abs(ma.coef_[0]), np.abs(ms.coef_[0])).statistic
            if c is not None and math.isfinite(float(c)):
                drv = float(c)
    return {
        "retention": float(np.median(rets)) if rets else float("nan"),
        "parity": float(np.mean(pars)) if pars else float("nan"),
        "driver_spearman": drv,
    }


def _eval_multiclass(Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray, Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray, k: int) -> dict[str, float]:
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
        c = spearmanr(np.mean(np.abs(ma.coef_), axis=0), np.mean(np.abs(ms.coef_), axis=0)).statistic
        if c is not None and math.isfinite(float(c)):
            drv = float(c)
    return {
        "retention": (null_ll - ls) / gain,
        "parity": 1.0 if ls <= la + max(abs(lb - la) * 1.25, 0.03 * gain) else 0.0,
        "driver_spearman": drv,
    }


def _eval_realism_and_privacy(Xa: np.ndarray, Xb: np.ndarray, Xs: np.ndarray, Xt: np.ndarray) -> dict[str, float]:
    m = min(len(Xt), len(Xs), 180)
    X_all = np.vstack([Xt[:m], Xs[:m]])
    y_all = np.concatenate([np.zeros(m, dtype=int), np.ones(m, dtype=int)])
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)
    oof = np.zeros(len(y_all), dtype=np.float64)
    for tr, va in skf.split(X_all, y_all):
        clf = RandomForestClassifier(n_estimators=24, max_depth=5, random_state=17, n_jobs=2)
        clf.fit(X_all[tr], y_all[tr])
        oof[va] = clf.predict_proba(X_all[va])[:, 1]
    c2st = float(roc_auc_score(y_all, oof))
    c2st = max(c2st, 1.0 - c2st)

    # Median single-column MargAUC (2-fold depth-3 tree / quantile distinguishability)
    marg_aucs = []
    for j in range(min(Xt.shape[1], 12)):
        col_r, col_s = Xt[:m, j], Xs[:m, j]
        if float(np.std(col_r)) < 1e-7 and float(np.std(col_s)) < 1e-7:
            marg_aucs.append(0.5)
            continue
        # Rank AUC between real and synthetic marginals
        vals = np.concatenate([col_r, col_s])
        lbl = np.concatenate([np.zeros(len(col_r)), np.ones(len(col_s))])
        auc_j = float(roc_auc_score(lbl, vals))
        marg_aucs.append(max(auc_j, 1.0 - auc_j))
    marg_auc = float(np.median(marg_aucs)) if marg_aucs else 0.5

    # Empirical Privacy (explicitly non-DP / non-HIPAA): DCR share + Distance MIA + Twin-B exact-copy excess
    mp = min(len(Xa), len(Xb), len(Xs), 120)
    Sa, Ra, Rb = Xs[:mp], Xa[:mp], Xb[:mp]
    da_mat = np.sqrt(((Sa[:, None, :] - Ra[None, :, :]) ** 2).sum(axis=2))
    db_mat = np.sqrt(((Sa[:, None, :] - Rb[None, :, :]) ** 2).sum(axis=2))
    da = da_mat.min(axis=1)
    db = db_mat.min(axis=1)
    dcr_share = float(np.mean((da + 1e-9 < db).astype(np.float64) + 0.5 * (np.abs(da - db) <= 1e-9)))
    # Distance-based MIA: can min distance to synthetic separate member Ra from non-member Rb?
    d_mem = da_mat.min(axis=0)
    d_non = db_mat.min(axis=0)
    mia_y = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))])
    mia_s = -np.concatenate([d_mem, d_non])
    mia_auc = float(roc_auc_score(mia_y, mia_s))
    mia_auc = max(mia_auc, 1.0 - mia_auc)
    # Twin-B exact-copy excess
    ex_a = float(np.mean(da <= 1e-6))
    ex_b = float(np.mean(db <= 1e-6))
    ex_excess = max(0.0, ex_a - ex_b)

    return {
        "c2st_auc": c2st,
        "marg_auc": marg_auc,
        "realism_pass": 1.0 if c2st <= 0.70 else 0.0,
        "dcr_share": dcr_share,
        "mia_auc": mia_auc,
        "exact_copy_excess": ex_excess,
        "privacy_pass": 1.0 if (dcr_share <= 0.55 and mia_auc <= 0.60 and ex_excess <= 0.005) else 0.0,
    }


def _load_model(ckpt_path: Path, meta: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, str]:
    arch = str(meta.get("arch", "flow_v6"))
    if arch == "single_enc_v6":
        model: torch.nn.Module = DopeSingleEncV6(
            d_model=int(meta.get("d_model", 384)),
            n_layers=int(meta.get("n_layers", 8)),
            n_heads=int(meta.get("n_heads", 8)),
            global_dim=int(meta.get("global_dim", 128)),
            col_dim=int(meta.get("col_dim", 24)),
        ).to(device)
    else:
        model = DopeFidelityV5(
            d_model=int(meta.get("d_model", 256)),
            n_layers=int(meta.get("n_layers", 6)),
            n_heads=int(meta.get("n_heads", 8)),
            latent_dim=int(meta.get("latent_dim", 128)),
        ).to(device)
    model.load_state_dict(load_file(str(ckpt_path), device=str(device)), strict=True)
    model.eval()
    return model, arch


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    meta: dict[str, Any],
    store: S3CorpusStore,
    device: torch.device,
    max_per_task: int = 16,
    eval_role: str = "step_ema",
) -> dict[str, Any]:
    model, arch = _load_model(ckpt_path, meta, device)
    rng = np.random.default_rng(20261122)
    by_task_ret: dict[str, list[float]] = {"regression": [], "binary": [], "multiclass": [], "multilabel": []}
    parity_vals, drv_vals, c2st_vals, marg_vals, real_pass = [], [], [], [], []
    dcr_vals, mia_vals, ex_vals, priv_pass, enc_bytes = [], [], [], [], []

    for task in ("regression", "binary", "multiclass"):
        for idx in store.by_task.get(task, [])[:max_per_task]:
            tbl = store.load_table(idx)
            X_raw, y_raw, dg = tbl["X"], tbl["y"], tbl["dup_groups"]
            if len(y_raw) < 90:
                continue
            X_raw = X_raw[:, : min(X_raw.shape[1], 48)]
            ab_idx, t_idx = _duplicate_grouped_split(dg, y_raw, task, 0.75, rng)
            if len(ab_idx) < 60 or len(t_idx) < 20:
                continue
            a_rel, b_rel = _duplicate_grouped_split(dg[ab_idx], y_raw[ab_idx], task, 0.53, rng)
            a_idx, b_idx, t_idx = ab_idx[a_rel][:384], ab_idx[b_rel][:384], t_idx[:256]
            if len(a_idx) < 24 or len(b_idx) < 24 or len(t_idx) < 16:
                continue

            Xa, Xt, mf_a, _ = _fit_only_nc2_scale(X_raw[a_idx], X_raw[t_idx])
            _, Xb, _, _ = _fit_only_nc2_scale(X_raw[a_idx], X_raw[b_idx])
            ya, yb, yt = y_raw[a_idx].copy(), y_raw[b_idx].copy(), y_raw[t_idx].copy()
            k_cls = int(tbl["n_classes"])
            if task == "regression":
                mu, sd = float(np.mean(ya)), max(float(np.std(ya)), 1e-5)
                ya, yb, yt = np.clip((ya - mu) / sd, -6, 6), np.clip((yb - mu) / sd, -6, 6), np.clip((yt - mu) / sd, -6, 6)
            elif task == "multiclass":
                u = np.unique(np.rint(ya).astype(int))
                if len(u) < 3:
                    continue
                k_cls = len(u)
                mp = {int(c): i for i, c in enumerate(u)}
                ya = np.array([mp.get(int(round(v)), 0) for v in ya], dtype=np.float32)
                yb = np.array([mp.get(int(round(v)), 0) for v in yb], dtype=np.float32)
                yt = np.array([mp.get(int(round(v)), 0) for v in yt], dtype=np.float32)

            xf_t = torch.as_tensor(Xa, dtype=torch.float32, device=device)
            yf_t = torch.as_tensor(ya, dtype=torch.float32, device=device)
            mf_t = torch.as_tensor(mf_a, dtype=torch.float32, device=device)
            gen = torch.Generator(device=device).manual_seed(1337 + idx)
            n_syn = max(len(Xa) * 2, 256)

            if arch == "single_enc_v6":
                enc = model.encode_to_single_latent(xf_t, yf_t, mf_t, task, k_cls)
                syn = model.decode_from_single_latent(enc, task, k_cls, n_syn, gen)
                enc_bytes.append(float(model.encoding_bytes_fp16(Xa.shape[1])))
            else:
                enc = model.encode_table(xf_t, yf_t, task, k_cls, mf_t)
                syn = model.synthesize_differentiable(xf_t, yf_t, enc, task, k_cls, n_syn, gen)
            Xs = syn["X_syn"].cpu().numpy()
            ys_raw = syn["y_syn"].cpu().numpy()
            if task == "binary":
                p_b = np.clip(ys_raw.reshape(-1), 1e-4, 1.0 - 1e-4)
                ys = (rng.random(len(p_b)) < p_b).astype(np.float32)
                if len(np.unique(ys)) < 2:
                    ys = (p_b >= float(np.quantile(p_b, np.clip(1.0 - float((ya > 0.5).mean()), 0.05, 0.95)))).astype(np.float32)
            elif task == "multiclass":
                pr = np.clip(ys_raw, 1e-6, 1.0)
                pr = pr / pr.sum(axis=1, keepdims=True)
                ys = (np.cumsum(pr, axis=1) >= rng.random((len(pr), 1))).argmax(axis=1).astype(np.float32)
            else:
                ys = ys_raw

            if task == "regression":
                res = _eval_regression(Xa, ya, Xb, yb, Xs, ys, Xt, yt)
            elif task == "binary":
                res = _eval_binary(Xa, ya, Xb, yb, Xs, ys, Xt, yt)
            else:
                res = _eval_multiclass(Xa, ya, Xb, yb, Xs, ys, Xt, yt, k_cls)
            if math.isfinite(res["retention"]):
                by_task_ret[task].append(res["retention"])
            if math.isfinite(res["parity"]):
                parity_vals.append(res["parity"])
            if math.isfinite(res["driver_spearman"]):
                drv_vals.append(res["driver_spearman"])

            rp = _eval_realism_and_privacy(Xa, Xb, Xs, Xt)
            c2st_vals.append(rp["c2st_auc"])
            marg_vals.append(rp["marg_auc"])
            real_pass.append(rp["realism_pass"])
            dcr_vals.append(rp["dcr_share"])
            mia_vals.append(rp["mia_auc"])
            ex_vals.append(rp["exact_copy_excess"])
            priv_pass.append(rp["privacy_pass"])

            if task == "regression" and len(by_task_ret["multilabel"]) < 10 and Xa.shape[1] >= 4:
                X_all_ml = np.vstack([Xa, Xb, Xt])
                ym_all, _, n_lbl = _synthesize_multilabel_targets(X_all_ml, Xt, np.zeros(Xa.shape[1], dtype=bool), rng)
                yma, ymb, ymt = ym_all[: len(Xa)], ym_all[len(Xa) : len(Xa) + len(Xb)], ym_all[len(Xa) + len(Xb) :]
                yma_t = torch.as_tensor(yma, dtype=torch.float32, device=device)
                if arch == "single_enc_v6":
                    enc_ml = model.encode_to_single_latent(xf_t, yma_t, mf_t, "multilabel", n_lbl)
                    syn_ml = model.decode_from_single_latent(enc_ml, "multilabel", n_lbl, n_syn, gen)
                else:
                    enc_ml = model.encode_table(xf_t, yma_t, "multilabel", n_lbl, mf_t)
                    syn_ml = model.synthesize_differentiable(xf_t, yma_t, enc_ml, "multilabel", n_lbl, n_syn, gen)
                Xs_ml, ys_ml_p = syn_ml["X_syn"].cpu().numpy(), np.clip(syn_ml["y_syn"].cpu().numpy(), 1e-4, 1.0 - 1e-4)
                ys_ml = (rng.random(ys_ml_p.shape) < ys_ml_p).astype(np.float32)
                lbl_rets = []
                for li in range(n_lbl):
                    r_li = _eval_binary(Xa, yma[:, li], Xb, ymb[:, li], Xs_ml, ys_ml[:, li], Xt, ymt[:, li])
                    if math.isfinite(r_li["retention"]):
                        lbl_rets.append(r_li["retention"])
                if lbl_rets:
                    by_task_ret["multilabel"].append(float(np.mean(lbl_rets)))

    all_rets = [r for vals in by_task_ret.values() for r in vals]
    med = lambda xs: round(float(np.median(xs)), 4) if xs else None
    mean = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    return {
        "time": round(time.time(), 3), "step": int(meta.get("step", 0)), "arch": arch,
        "checkpoint": ckpt_path.name, "eval_role": eval_role,
        "main_kpi_retention_median": med(all_rets), "main_kpi_retention_mean": mean(all_rets),
        "regression_retention": med(by_task_ret["regression"]), "binary_retention": med(by_task_ret["binary"]),
        "multiclass_retention": med(by_task_ret["multiclass"]), "multilabel_retention": med(by_task_ret["multilabel"]),
        "parity_coverage": mean(parity_vals), "realism_coverage": mean(real_pass), "privacy_pass_coverage": mean(priv_pass),
        "driver_spearman_median": med(drv_vals), "c2st_auc_median": med(c2st_vals), "marg_auc_median": med(marg_vals),
        "dcr_share_median": med(dcr_vals), "mia_auc_median": med(mia_vals), "exact_copy_excess_median": med(ex_vals),
        "encoding_bytes_median": med(enc_bytes), "tables_evaluated": len(all_rets), "formal_dp_state": "not_claimed",
    }


def _update_regret_curriculum(run_dir: Path, row: dict[str, Any]) -> None:
    """Emit JopeDime-style per-task regret multipliers so the live trainer boosts lagging tasks."""
    mults: dict[str, float] = {}
    for task, key in [("regression", "regression_retention"), ("binary", "binary_retention"), ("multiclass", "multiclass_retention")]:
        v = row.get(key)
        if v is None:
            mults[task] = 1.5
        else:
            gap = max(0.0, 0.95 - float(v))
            mults[task] = round(min(1.0 + gap * 6.0, 5.0), 3)
    payload = {"updated_at": round(time.time(), 3), "step": row["step"], "task_multipliers": mults, "hash_multipliers": {}}
    tmp = run_dir / ".curriculum_regret.json.tmp"
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, run_dir / "curriculum_regret.json")


def _promote_best_and_valley_soup(
    run_dir: Path,
    ckpt: Path,
    meta: dict[str, Any],
    row: dict[str, Any],
    store: S3CorpusStore,
    device: torch.device,
    max_per_task: int,
    eval_log: Path,
) -> None:
    """Preserve `best_main_kpi.safetensors` permanently & build JopeDime k=3 valley soup."""
    score = row.get("main_kpi_retention_median")
    if score is None:
        return
    ckpt_dir = run_dir / "checkpoints"
    best_json = ckpt_dir / "best_main_kpi.json"
    prev_best = -1e9
    if best_json.is_file():
        try:
            prev_best = float(json.loads(best_json.read_text(encoding="utf-8")).get("main_kpi_retention_median", -1e9))
        except (OSError, ValueError):
            prev_best = -1e9

    if float(score) > prev_best:
        best_file = ckpt_dir / "best_main_kpi.safetensors"
        tmp = ckpt_dir / ".best_main_kpi.safetensors.tmp"
        shutil.copy2(ckpt, tmp)
        os.replace(tmp, best_file)
        sha = hashlib.sha256(best_file.read_bytes()).hexdigest()
        bmeta = {**row, "role": "best_main_kpi", "source_checkpoint": ckpt.name, "sha256": sha, "bytes": best_file.stat().st_size, "delta_vs_prev": round(float(score) - prev_best, 4) if prev_best > -1e8 else None}
        best_json.write_text(json.dumps(bmeta, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with (ckpt_dir / "ledger.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**bmeta, "checkpoint": "best_main_kpi.safetensors"}, sort_keys=True) + "\n")
        print(f"  ★ NEW BEST MAIN KPI @ step={row['step']}: {score:.4f} (saved best_main_kpi.safetensors)", flush=True)

    # Maintain top-3 distinct-valley checkpoints (min step gap 150) -> JopeDime k=3 valley soup!
    vp = ckpt_dir / "valleys.json"
    valleys: list[dict[str, Any]] = []
    if vp.is_file():
        try:
            valleys = list(json.loads(vp.read_text(encoding="utf-8")).get("valleys", []))
        except (OSError, ValueError):
            valleys = []
    step = int(row["step"])
    too_close = [v for v in valleys if abs(int(v["step"]) - step) < 150]
    if too_close:
        if float(score) > max(float(v["score"]) for v in too_close):
            valleys = [v for v in valleys if abs(int(v["step"]) - step) >= 150]
            valleys.append({"step": step, "score": float(score), "checkpoint": ckpt.name})
    else:
        valleys.append({"step": step, "score": float(score), "checkpoint": ckpt.name})
    valleys = [v for v in sorted(valleys, key=lambda x: -float(x["score"])) if (ckpt_dir / v["checkpoint"]).is_file()][:3]
    vp.write_text(json.dumps({"valleys": valleys}, indent=2) + "\n", encoding="utf-8")

    if len(valleys) >= 2 and any(v["checkpoint"] == ckpt.name for v in valleys):
        states = [load_file(str(ckpt_dir / v["checkpoint"]), device="cpu") for v in valleys]
        soup = {k: sum(s[k].float() for s in states) / float(len(states)) for k in states[0]}
        soup_path = ckpt_dir / "best_soup_k3.safetensors"
        stmp = ckpt_dir / ".best_soup_k3.safetensors.tmp"
        save_file({k: v.contiguous() for k, v in soup.items()}, str(stmp))
        os.replace(stmp, soup_path)
        srow = evaluate_checkpoint(soup_path, {**meta, "step": step}, store, device, max_per_task=max_per_task, eval_role=f"valley_soup_k{len(valleys)}")
        srow["valley_steps"] = [int(v["step"]) for v in valleys]
        with eval_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(srow, sort_keys=True) + "\n")
        with (ckpt_dir / "ledger.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**srow, "role": "valley_soup", "sha256": hashlib.sha256(soup_path.read_bytes()).hexdigest(), "bytes": soup_path.stat().st_size}, sort_keys=True) + "\n")
        print(f"  🥣 VALLEY SOUP k={len(valleys)} ({srow['valley_steps']}) MAIN_KPI={srow['main_kpi_retention_median']} mc={srow['multiclass_retention']} c2st={srow['c2st_auc_median']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out family-disjoint Main-KPI evaluator + valley soup promoter")
    ap.add_argument("--corpus-dir", default="/home/ubuntu/dope-data/s3-v250-corpus")
    ap.add_argument("--run-dir", default="/home/ubuntu/dope-data/runs/dope_v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-per-task", type=int, default=16)
    ap.add_argument("--poll-s", type=float, default=15.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    store = S3CorpusStore(Path(args.corpus_dir), split="val", cache_cap=256)
    run_dir = Path(args.run_dir)
    eval_log = run_dir / "eval.jsonl"
    ptr_file = run_dir / "checkpoints" / "latest.json"
    last_step = -1

    while True:
        if ptr_file.exists():
            try:
                meta = json.loads(ptr_file.read_text(encoding="utf-8"))
                step = int(meta.get("step", -1))
                ckpt = run_dir / "checkpoints" / meta["checkpoint"]
                if step > last_step and ckpt.exists():
                    t0 = time.time()
                    row = evaluate_checkpoint(ckpt, meta, store, device, max_per_task=args.max_per_task)
                    row["eval_wall_s"] = round(time.time() - t0, 2)
                    with eval_log.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                    print(
                        f"[EVAL {row['arch']} step={step}] MAIN_KPI={row['main_kpi_retention_median']} "
                        f"(reg={row['regression_retention']}, bin={row['binary_retention']}, "
                        f"mc={row['multiclass_retention']}, ml={row['multilabel_retention']}) "
                        f"par={row['parity_coverage']} c2st={row['c2st_auc_median']} marg={row['marg_auc_median']} "
                        f"drv={row['driver_spearman_median']} dcr={row['dcr_share_median']} mia={row['mia_auc_median']} ({row['eval_wall_s']}s)",
                        flush=True,
                    )
                    _update_regret_curriculum(run_dir, row)
                    _promote_best_and_valley_soup(run_dir, ckpt, meta, row, store, device, args.max_per_task, eval_log)
                    last_step = step
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                print(f"[EVAL warn] {exc}", flush=True)
        if args.once:
            break
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
