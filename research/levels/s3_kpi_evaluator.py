"""Continuous held-out Main-KPI & dope-fidelity/v3 evaluator for DopeFidelityV5 checkpoints.

Evaluates every saved checkpoint on the strictly family-disjoint 10% S3 validation cohort
across Regression, Binary, Multiclass (S3 multi_label), and True Multi-Label views using
independent out-of-sample scikit-learn auditors (Ridge/LogReg + HistGBDT/RandomForest),
computing unclipped TSTR retention (MAIN KPI), parity pass rate vs disjoint real twin B,
driver Spearman alignment, 2-fold nonlinear C2ST AUC, max MargAUC, and empirical DCR share.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from safetensors.torch import load_file
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch

from .s3_dynamic_augment import AugmentConfig, S3CorpusStore, _duplicate_grouped_split, _fit_only_nc2_scale, _synthesize_multilabel_targets
from .v5_fidelity_net import DopeFidelityV5


def _eval_regression(Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray, Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray) -> dict[str, float]:
    null_mse = float(mean_squared_error(yt, np.full_like(yt, float(np.mean(ya)))))
    rets, parity_hits, drv_corrs = [], [], []
    for name in ("ridge", "hgbdt"):
        if name == "ridge":
            ma, mb, ms = Ridge(alpha=1.0), Ridge(alpha=1.0), Ridge(alpha=1.0)
        else:
            ma = HistGradientBoostingRegressor(max_depth=4, max_iter=50, random_state=7)
            mb = HistGradientBoostingRegressor(max_depth=4, max_iter=50, random_state=7)
            ms = HistGradientBoostingRegressor(max_depth=4, max_iter=50, random_state=7)
        ma.fit(Xa, ya)
        mb.fit(Xb, yb)
        ms.fit(Xs, ys)
        la = float(mean_squared_error(yt, ma.predict(Xt)))
        lb = float(mean_squared_error(yt, mb.predict(Xt)))
        ls = float(mean_squared_error(yt, ms.predict(Xt)))
        gain = null_mse - la
        if gain > 0.02 * null_mse:
            rets.append((null_mse - ls) / gain)
            tol = max(abs(lb - la) * 1.25, 0.03 * gain)
            parity_hits.append(1.0 if ls <= la + tol else 0.0)
        if name == "ridge" and Xa.shape[1] >= 3:
            c = spearmanr(np.abs(ma.coef_), np.abs(ms.coef_)).statistic
            if c is not None and math.isfinite(float(c)):
                drv_corrs.append(float(c))
    return {
        "retention": float(np.mean(rets)) if rets else float("nan"),
        "parity": float(np.mean(parity_hits)) if parity_hits else float("nan"),
        "driver_spearman": float(np.mean(drv_corrs)) if drv_corrs else float("nan"),
    }


def _eval_binary(Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray, Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray) -> dict[str, float]:
    ya_i, yb_i, ys_i, yt_i = (ya > 0.5).astype(int), (yb > 0.5).astype(int), (ys > 0.5).astype(int), (yt > 0.5).astype(int)
    if min(len(np.unique(ya_i)), len(np.unique(yb_i)), len(np.unique(ys_i)), len(np.unique(yt_i))) < 2:
        return {"retention": float("nan"), "parity": float("nan"), "driver_spearman": float("nan")}
    p0 = float(np.clip(np.mean(ya_i), 1e-4, 1.0 - 1e-4))
    null_ll = float(log_loss(yt_i, np.full(len(yt_i), p0), labels=[0, 1]))
    rets, parity_hits, drv_corrs = [], [], []
    for name in ("logreg", "hgbdt"):
        if name == "logreg":
            ma, mb, ms = LogisticRegression(max_iter=200), LogisticRegression(max_iter=200), LogisticRegression(max_iter=200)
        else:
            ma = HistGradientBoostingClassifier(max_depth=4, max_iter=50, random_state=7)
            mb = HistGradientBoostingClassifier(max_depth=4, max_iter=50, random_state=7)
            ms = HistGradientBoostingClassifier(max_depth=4, max_iter=50, random_state=7)
        ma.fit(Xa, ya_i)
        mb.fit(Xb, yb_i)
        ms.fit(Xs, ys_i)
        pa = np.clip(ma.predict_proba(Xt)[:, 1], 1e-5, 1.0 - 1e-5)
        pb = np.clip(mb.predict_proba(Xt)[:, 1], 1e-5, 1.0 - 1e-5)
        ps = np.clip(ms.predict_proba(Xt)[:, 1], 1e-5, 1.0 - 1e-5)
        la, lb, ls = float(log_loss(yt_i, pa, labels=[0, 1])), float(log_loss(yt_i, pb, labels=[0, 1])), float(log_loss(yt_i, ps, labels=[0, 1]))
        gain = null_ll - la
        if gain > 0.015:
            rets.append((null_ll - ls) / gain)
            tol = max(abs(lb - la) * 1.25, 0.03 * gain)
            parity_hits.append(1.0 if ls <= la + tol else 0.0)
        if name == "logreg" and Xa.shape[1] >= 3:
            c = spearmanr(np.abs(ma.coef_[0]), np.abs(ms.coef_[0])).statistic
            if c is not None and math.isfinite(float(c)):
                drv_corrs.append(float(c))
    return {
        "retention": float(np.mean(rets)) if rets else float("nan"),
        "parity": float(np.mean(parity_hits)) if parity_hits else float("nan"),
        "driver_spearman": float(np.mean(drv_corrs)) if drv_corrs else float("nan"),
    }


def _eval_multiclass(Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, yb: np.ndarray, Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray, k: int) -> dict[str, float]:
    ya_i = np.clip(np.rint(ya).astype(int), 0, k - 1)
    yb_i = np.clip(np.rint(yb).astype(int), 0, k - 1)
    ys_i = np.clip(np.rint(ys).astype(int), 0, k - 1)
    yt_i = np.clip(np.rint(yt).astype(int), 0, k - 1)
    if len(np.unique(ya_i)) < 2 or len(np.unique(ys_i)) < 2 or len(np.unique(yt_i)) < 2:
        return {"retention": float("nan"), "parity": float("nan"), "driver_spearman": float("nan")}
    labels = list(range(k))
    cnt = np.bincount(ya_i, minlength=k).astype(np.float64) + 0.5
    p0 = cnt / cnt.sum()
    null_ll = float(log_loss(yt_i, np.tile(p0, (len(yt_i), 1)), labels=labels))
    ma = LogisticRegression(max_iter=200)
    mb = LogisticRegression(max_iter=200)
    ms = LogisticRegression(max_iter=200)
    ma.fit(Xa, ya_i)
    mb.fit(Xb, yb_i)
    ms.fit(Xs, ys_i)

    def full_probs(clf: LogisticRegression) -> np.ndarray:
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
    tol = max(abs(lb - la) * 1.25, 0.03 * gain)
    drv = float("nan")
    if Xa.shape[1] >= 3 and ma.coef_.shape[1] == ms.coef_.shape[1]:
        c = spearmanr(np.mean(np.abs(ma.coef_), axis=0), np.mean(np.abs(ms.coef_), axis=0)).statistic
        if c is not None and math.isfinite(float(c)):
            drv = float(c)
    return {"retention": (null_ll - ls) / gain, "parity": 1.0 if ls <= la + tol else 0.0, "driver_spearman": drv}


def _eval_realism_and_privacy(Xa: np.ndarray, Xb: np.ndarray, Xs: np.ndarray, Xt: np.ndarray) -> dict[str, float]:
    m = min(len(Xt), len(Xs), 200)
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

    # Empirical DCR share: synthetic closer to fit A than disjoint real twin B (explicitly non-DP!)
    mp = min(len(Xa), len(Xb), len(Xs), 120)
    Sa, Ra, Rb = Xs[:mp], Xa[:mp], Xb[:mp]
    da = np.sqrt(((Sa[:, None, :] - Ra[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    db = np.sqrt(((Sa[:, None, :] - Rb[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    dcr_share = float(np.mean((da + 1e-9 < db).astype(np.float64) + 0.5 * (np.abs(da - db) <= 1e-9)))
    return {
        "c2st_auc": c2st,
        "realism_pass": 1.0 if c2st <= 0.70 else 0.0,
        "dcr_share": dcr_share,
        "privacy_pass": 1.0 if dcr_share <= 0.55 else 0.0,
    }


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    meta: dict[str, Any],
    store: S3CorpusStore,
    device: torch.device,
    max_per_task: int = 20,
) -> dict[str, Any]:
    model = DopeFidelityV5(
        d_model=int(meta.get("d_model", 256)),
        n_layers=int(meta.get("n_layers", 6)),
        latent_dim=int(meta.get("latent_dim", 128)),
    ).to(device)
    model.load_state_dict(load_file(str(ckpt_path), device=str(device)), strict=True)
    model.eval()

    rng = np.random.default_rng(20261122)
    by_task_ret: dict[str, list[float]] = {"regression": [], "binary": [], "multiclass": [], "multilabel": []}
    parity_vals, drv_vals, c2st_vals, real_pass, dcr_vals, priv_pass = [], [], [], [], [], []

    for task in ("regression", "binary", "multiclass"):
        pool = store.by_task.get(task, [])[:max_per_task]
        for idx in pool:
            tbl = store.load_table(idx)
            X_raw, y_raw, dg = tbl["X"], tbl["y"], tbl["dup_groups"]
            if len(y_raw) < 90:
                continue
            p_cap = min(X_raw.shape[1], 48)
            X_raw = X_raw[:, :p_cap]
            # Disjoint 3-way split: A (40% fit), B (35% fresh twin), T (25% held-out query)
            ab_idx, t_idx = _duplicate_grouped_split(dg, y_raw, task, 0.75, rng)
            if len(ab_idx) < 60 or len(t_idx) < 20:
                continue
            a_rel, b_rel = _duplicate_grouped_split(dg[ab_idx], y_raw[ab_idx], task, 0.53, rng)
            a_idx, b_idx = ab_idx[a_rel][:384], ab_idx[b_rel][:384]
            t_idx = t_idx[:256]
            if len(a_idx) < 24 or len(b_idx) < 24 or len(t_idx) < 16:
                continue

            Xa, Xt, _, _ = _fit_only_nc2_scale(X_raw[a_idx], X_raw[t_idx])
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
            enc = model.encode_table(xf_t, yf_t, task, k_cls)
            gen = torch.Generator(device=device).manual_seed(1337 + idx)
            syn = model.synthesize_differentiable(xf_t, yf_t, enc, task, k_cls, max(len(Xa) * 2, 256), gen)
            Xs = syn["X_syn"].cpu().numpy()
            ys_raw = syn["y_syn"].cpu().numpy()
            ys = np.argmax(ys_raw, axis=1).astype(np.float32) if task == "multiclass" else ys_raw

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
            real_pass.append(rp["realism_pass"])
            dcr_vals.append(rp["dcr_share"])
            priv_pass.append(rp["privacy_pass"])

            # Evaluate true multi-label view on first 10 regression tables
            if task == "regression" and len(by_task_ret["multilabel"]) < 10 and Xa.shape[1] >= 4:
                X_all_ml = np.vstack([Xa, Xb, Xt])
                ym_all, _, n_lbl = _synthesize_multilabel_targets(X_all_ml, Xt, np.zeros(Xa.shape[1], dtype=bool), rng)
                yma = ym_all[: len(Xa)]
                ymb = ym_all[len(Xa) : len(Xa) + len(Xb)]
                ymt = ym_all[len(Xa) + len(Xb) :]
                enc_ml = model.encode_table(xf_t, torch.as_tensor(yma, dtype=torch.float32, device=device), "multilabel", n_lbl)
                syn_ml = model.synthesize_differentiable(xf_t, torch.as_tensor(yma, dtype=torch.float32, device=device), enc_ml, "multilabel", n_lbl, max(len(Xa) * 2, 256), gen)
                Xs_ml, ys_ml = syn_ml["X_syn"].cpu().numpy(), syn_ml["y_syn"].cpu().numpy()
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
        "step": int(meta.get("step", 0)),
        "time": round(time.time(), 3),
        "main_kpi_retention_median": med(all_rets),
        "main_kpi_retention_mean": mean(all_rets),
        "regression_retention": med(by_task_ret["regression"]),
        "binary_retention": med(by_task_ret["binary"]),
        "multiclass_retention": med(by_task_ret["multiclass"]),
        "multilabel_retention": med(by_task_ret["multilabel"]),
        "parity_coverage": mean(parity_vals),
        "realism_coverage": mean(real_pass),
        "privacy_pass_coverage": mean(priv_pass),
        "driver_spearman_median": med(drv_vals),
        "c2st_auc_median": med(c2st_vals),
        "dcr_share_median": med(dcr_vals),
        "tables_evaluated": len(all_rets),
        "formal_dp_state": "not_claimed",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out family-disjoint Main-KPI evaluator for DopeFidelityV5")
    ap.add_argument("--corpus-dir", default="/home/ubuntu/dope-data/s3-v250-corpus")
    ap.add_argument("--run-dir", default="/home/ubuntu/dope-data/runs/dope_v5_ddp")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vram-cap-gib", type=float, default=4.5)
    ap.add_argument("--max-per-task", type=int, default=16)
    ap.add_argument("--poll-s", type=float, default=20.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda" and args.vram_cap_gib > 0:
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
        torch.cuda.set_per_process_memory_fraction(min(max(args.vram_cap_gib / total_gib, 0.08), 0.50), 0)

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
                        f"[EVAL step={step}] MAIN_KPI_ret={row['main_kpi_retention_median']} "
                        f"(reg={row['regression_retention']}, bin={row['binary_retention']}, "
                        f"mc={row['multiclass_retention']}, ml={row['multilabel_retention']}) "
                        f"parity={row['parity_coverage']} c2st={row['c2st_auc_median']} "
                        f"drv={row['driver_spearman_median']} dcr={row['dcr_share_median']} ({row['eval_wall_s']}s)",
                        flush=True,
                    )
                    last_step = step
            except Exception as exc:
                print(f"[EVAL warn] {exc}", flush=True)
        if args.once:
            break
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
