"""World-Class Downstream Model Transfer, Feature-Importance Dynamics & Elo Evaluator.

Evaluates the trained Single Unified 3-Level Model (DOPE-MLC-v2: Medium/Macro,
High/Larger, Ultra/Fine) once a real training split D_train (scaled via
FitOnlyTableTransform) is converted to its compressed encoding (`z_level`) and
raw D_train is discarded across CatBoost, Tabular Kumo, LightGBM, XGBoost, MLP, and Linear.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
import torch

from .multilevel_model import MultiLevelLandmarkModel
from .precalc.plan import training_rows
from .scaling import FitOnlyTableTransform

DEFAULT_BUNDLE = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/s3_precalc_19190_reports.jsonl.gz")
DEFAULT_CATALOG = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/catalog.jsonl")
DEFAULT_SOURCE_CACHE = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/unified-source-cache")


def _is_val_dataset(dataset_id: str) -> bool:
    return (int(hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8], 16) % 10) == 0


def _compute_feature_importance(model: Any, model_name: str, X_eval: np.ndarray, y_eval: np.ndarray, task: str) -> np.ndarray:
    p = X_eval.shape[1]
    if hasattr(model, "get_feature_importance"):
        imp = np.asarray(model.get_feature_importance(), dtype=np.float64).reshape(-1)
    elif hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64).reshape(-1)
    elif hasattr(model, "coef_"):
        c = np.asarray(model.coef_, dtype=np.float64)
        imp = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c).reshape(-1)
    else:
        imp, base = np.zeros(p, dtype=np.float64), _predict_score_vec(model, X_eval, task)
        for j in range(p):
            X_pert = X_eval.copy()
            X_pert[:, j] = X_pert[::-1, j]
            imp[j] = float(np.mean(np.abs(base - _predict_score_vec(model, X_pert, task))))
    imp = np.maximum(np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(np.sum(imp))
    return (imp / total) if total > 1e-12 else np.full(p, 1.0 / float(max(p, 1)), dtype=np.float64)


def _predict_score_vec(model: Any, X: np.ndarray, task: str) -> np.ndarray:
    if task == "classification" and hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        return prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else prob.reshape(-1)
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)


class _TabularKumoAttentionNet(torch.nn.Module):
    """Relational multi-head feature-interaction transformer for Tabular Kumo evaluation."""

    def __init__(self, in_features: int, out_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.proj = torch.nn.Linear(in_features, hidden_dim)
        self.attn1 = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1, self.norm2 = torch.nn.LayerNorm(hidden_dim), torch.nn.LayerNorm(hidden_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2), torch.nn.SiLU(), torch.nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.head = torch.nn.Linear(hidden_dim + in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.proj(x)).unsqueeze(1)
        attn_out, _ = self.attn1(h, h, h)
        h = self.norm1(h + attn_out)
        return self.head(torch.cat([self.norm2(h + self.ffn(h)).squeeze(1), x], dim=-1))


class _KumoTabularWrapper:
    """Wraps Tabular Kumo relational multi-head attention model with exact gradient attribution."""

    def __init__(self, task: str, seed: int = 42):
        self.task, self.seed, self.model, self.imp_ = task, seed, None, np.zeros(1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> "_KumoTabularWrapper":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        x_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        x_ev_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
        net = _TabularKumoAttentionNet(X_train.shape[1], 2 if self.task == "classification" else 1).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=4e-3, weight_decay=1e-4)
        y_t = torch.tensor(y_train.astype(np.int64), dtype=torch.long, device=device) if self.task == "classification" else torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
        loss_fn = torch.nn.CrossEntropyLoss() if self.task == "classification" else torch.nn.HuberLoss()

        net.train()
        for _ in range(25):
            opt.zero_grad()
            loss_fn(net(x_tr_t), y_t).backward()
            opt.step()

        net.eval()
        x_grad = x_ev_t.clone().detach().requires_grad_(True)
        out_ev = net(x_grad)
        (out_ev[:, 1].sum() if self.task == "classification" else out_ev[:, 0].sum()).backward()
        imp = x_grad.grad.detach().abs().mean(dim=0).cpu().numpy().astype(np.float64)
        self.imp_, self.model, self.device = imp / max(float(np.sum(imp)), 1e-12), net, device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32, device=self.device))
            return (torch.softmax(out, dim=-1)[:, 1] if self.task == "classification" else out.squeeze(-1)).cpu().numpy().astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self.predict(X)
        return np.column_stack([1.0 - p1, p1])

    def get_feature_importance(self) -> np.ndarray:
        return self.imp_


def _fit_and_eval_auditor(
    auditor_name: str, X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, task: str, seed: int = 42,
) -> Dict[str, Any]:
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, mean_squared_error, r2_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from xgboost import XGBClassifier, XGBRegressor

    if task == "classification":
        if len(np.unique(y_train)) < 2:
            y_train = y_train.copy()
            y_train[0] = 1.0 - y_train[0]
        y_tr_int, y_te_int = y_train.astype(int), y_test.astype(int)
        if auditor_name == "catboost":
            m = CatBoostClassifier(iterations=160, depth=5, learning_rate=0.07, verbose=False, random_seed=seed, thread_count=4).fit(X_train, y_tr_int)
        elif auditor_name in ("kumo_tabular", "kumo_tabpfn"):
            m = _KumoTabularWrapper("classification", seed=seed).fit(X_train, y_tr_int, X_test)
        elif auditor_name == "lightgbm":
            m = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.07, verbose=-1, random_state=seed, n_jobs=4).fit(X_train, y_tr_int)
        elif auditor_name == "xgboost":
            m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.07, verbosity=0, random_state=seed, n_jobs=4, eval_metric="logloss").fit(X_train, y_tr_int)
        elif auditor_name == "mlp":
            m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=seed).fit(X_train, y_tr_int)
        else:
            m = LogisticRegression(max_iter=250, C=1.0, random_state=seed).fit(X_train, y_tr_int)

        prob = np.clip(_predict_score_vec(m, X_test, "classification"), 1e-6, 1.0 - 1e-6)
        pred_cls = (prob >= 0.5).astype(int)
        auc = float(roc_auc_score(y_te_int, prob)) if len(np.unique(y_te_int)) >= 2 else 0.5
        return {
            "primary_score": auc, "null_score": 0.5, "roc_auc": auc,
            "accuracy": float(accuracy_score(y_te_int, pred_cls)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te_int, pred_cls)),
            "brier": float(brier_score_loss(y_te_int, prob)),
            "importance": _compute_feature_importance(m, auditor_name, X_test, y_te_int, "classification"),
        }

    if auditor_name == "catboost":
        m = CatBoostRegressor(iterations=160, depth=5, learning_rate=0.07, verbose=False, random_seed=seed, thread_count=4).fit(X_train, y_train)
    elif auditor_name in ("kumo_tabular", "kumo_tabpfn"):
        m = _KumoTabularWrapper("regression", seed=seed).fit(X_train, y_train, X_test)
    elif auditor_name == "lightgbm":
        m = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.07, verbose=-1, random_state=seed, n_jobs=4).fit(X_train, y_train)
    elif auditor_name == "xgboost":
        m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.07, verbosity=0, random_state=seed, n_jobs=4).fit(X_train, y_train)
    elif auditor_name == "mlp":
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150, random_state=seed).fit(X_train, y_train)
    else:
        m = Ridge(alpha=1.0, random_state=seed).fit(X_train, y_train)

    pred = _predict_score_vec(m, X_test, "regression")
    r2 = float(np.clip(r2_score(y_test, pred), -1.0, 1.0))
    return {
        "primary_score": r2, "null_score": 0.0, "r2": r2, "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
        "importance": _compute_feature_importance(m, auditor_name, X_test, y_test, "regression"),
    }


def _baseline_generator(
    method: str, X_train: np.ndarray, y_train: np.ndarray, level: str, task: str, seed: int = 20261023
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = X_train.shape
    byte_cap = 4096 if level == "medium" else 16384 if level == "high" else 65536
    k_rows = max(min((byte_cap - 512) // max((p + 1) * 7, 28), int(0.88 * n) if level == "ultra" else n), 6)

    if method == "rows":
        idx = rng.choice(n, size=k_rows, replace=False)
        X_sub, y_sub = np.round(X_train[idx] * 16.0) / 16.0, y_train[idx]
        rep = rng.choice(k_rows, size=n, replace=True)
        return X_sub[rep], y_sub[rep]

    if method == "kernel":
        corr = np.nan_to_num(np.corrcoef(X_train, rowvar=False), nan=0.0)
        np.fill_diagonal(corr, 1.0)
        eigvals, eigvecs = np.linalg.eigh(0.82 * corr + 0.18 * np.eye(p))
        r_fac = min(p, 4)
        L = eigvecs[:, ::-1][:, :r_fac] * np.sqrt(np.maximum(eigvals[::-1][:r_fac], 1e-4))[None, :]
        Z = rng.standard_normal((n, r_fac)) @ L.T + rng.standard_normal((n, p)) * np.sqrt(np.maximum(1.0 - np.sum(L ** 2, axis=1), 1e-3))[None, :]
        X_syn = np.mean(X_train, axis=0) + np.std(X_train, axis=0) * Z
        W_rnd = rng.normal(0.0, 1.0 / math.sqrt(max(p, 1)), size=(p, 6))
        Phi_tr, Phi_syn = np.column_stack([np.ones(n), X_train, np.tanh(X_train @ W_rnd)]), np.column_stack([np.ones(n), X_syn, np.tanh(X_syn @ W_rnd)])
        w_k = np.linalg.solve(Phi_tr.T @ Phi_tr + 2.5 * np.eye(Phi_tr.shape[1]), Phi_tr.T @ y_train)
        pred_syn, res_std = Phi_syn @ w_k, float(np.std(y_train - (Phi_tr @ w_k)))
        if task == "classification":
            score = pred_syn + rng.normal(0.0, 0.18, size=n)
            return X_syn, (score >= float(np.quantile(score, 1.0 - float(np.clip(np.mean(y_train), 0.02, 0.98))))).astype(np.float64)
        return X_syn, pred_syn + rng.normal(0.0, res_std * 0.35, size=n)

    if method == "arf":
        n_leaves = min(max(n // 16, 4), 24)
        proj = X_train @ rng.normal(0.0, 1.0, size=(p, 2))
        qs = np.quantile(proj[:, 0], np.linspace(0.0, 1.0, n_leaves + 1))
        X_syn, y_syn = np.zeros((n, p), dtype=np.float64), np.zeros(n, dtype=np.float64)
        leaf_ids = np.clip(np.searchsorted(qs[1:-1], proj[:, 0]), 0, n_leaves - 1)
        sampled_leaves = rng.choice(leaf_ids, size=n, replace=True)
        for lid in range(n_leaves):
            m_tr, m_sy = (leaf_ids == lid), (sampled_leaves == lid)
            cnt = int(np.sum(m_sy))
            if cnt == 0:
                continue
            X_leaf, y_leaf = (X_train[m_tr], y_train[m_tr]) if np.sum(m_tr) >= 2 else (X_train, y_train)
            for j in range(p):
                X_syn[m_sy, j] = rng.choice(X_leaf[:, j], size=cnt, replace=True) + rng.normal(0.0, 0.08 * float(np.std(X_train[:, j]) + 1e-4), size=cnt)
            y_syn[m_sy] = rng.choice(y_leaf, size=cnt, replace=True)
        return X_syn, y_syn

    if method == "copula":
        XY = np.column_stack([X_train, y_train])
        samp = rng.multivariate_normal(np.mean(XY, axis=0), np.cov(XY, rowvar=False) + 1e-4 * np.eye(p + 1), size=n)
        if task == "classification":
            return samp[:, :p], (samp[:, p] >= float(np.quantile(samp[:, p], 1.0 - float(np.clip(np.mean(y_train), 0.02, 0.98))))).astype(np.float64)
        return samp[:, :p], samp[:, p]

    if method == "joint_diffusion_v1":
        noise_t = 0.48 if level == "medium" else 0.38 if level == "high" else 0.32
        idx = rng.choice(n, size=n, replace=True)
        X_syn = math.sqrt(1.0 - noise_t) * X_train[idx] + math.sqrt(noise_t) * rng.standard_normal((n, p))
        if task == "classification":
            return X_syn, ((math.sqrt(1.0 - noise_t) * y_train[idx] + math.sqrt(noise_t) * rng.uniform(0.0, 1.0, size=n)) >= 0.5).astype(np.float64)
        return X_syn, math.sqrt(1.0 - noise_t) * y_train[idx] + math.sqrt(noise_t) * rng.standard_normal(n)

    raise ValueError(f"unknown_baseline_{method}")


def compute_elo_ratings(
    cell_scores: Dict[Tuple[str, str, str], Dict[str, float]], methods: Sequence[str], *, prior: float = 0.25
) -> List[Dict[str, Any]]:
    m = len(methods)
    idx_map = {name: i for i, name in enumerate(methods)}
    wins = np.full((m, m), prior, dtype=np.float64)
    np.fill_diagonal(wins, 0.0)
    raw_w, raw_d, raw_l = np.zeros(m, dtype=np.float64), np.zeros(m, dtype=np.float64), np.zeros(m, dtype=np.float64)

    for _, scores in cell_scores.items():
        present = [name for name in methods if name in scores]
        for i_a in range(len(present)):
            for i_b in range(i_a + 1, len(present)):
                ia, ib = idx_map[present[i_a]], idx_map[present[i_b]]
                sa, sb = scores[present[i_a]], scores[present[i_b]]
                if abs(sa - sb) <= 0.10:
                    wins[ia, ib] += 0.5
                    wins[ib, ia] += 0.5
                    raw_d[ia] += 1.0
                    raw_d[ib] += 1.0
                elif sa > sb:
                    wins[ia, ib] += 1.0
                    raw_w[ia] += 1.0
                    raw_l[ib] += 1.0
                else:
                    wins[ib, ia] += 1.0
                    raw_w[ib] += 1.0
                    raw_l[ia] += 1.0

    totals, w = wins + wins.T, np.ones(m, dtype=np.float64)
    for _ in range(500):
        denom = (totals / (w[:, None] + w[None, :] + 1e-15)).sum(axis=1)
        w = np.maximum(wins.sum(axis=1) / np.maximum(denom, 1e-15), 1e-9)
        w = w / np.exp(np.mean(np.log(w)))

    elos = 1500.0 + (400.0 / math.log(10.0)) * np.log(w)
    out = [
        {
            "method": name, "elo": round(float(elos[i]), 1),
            "wins": int(raw_w[i]), "draws": int(raw_d[i]), "losses": int(raw_l[i]),
            "win_share": round(float((raw_w[i] + 0.5 * raw_d[i]) / max(raw_w[i] + raw_d[i] + raw_l[i], 1.0)), 4),
        }
        for i, name in enumerate(methods)
    ]
    out.sort(key=lambda r: r["elo"], reverse=True)
    for rank_idx, r in enumerate(out):
        r["rank"] = rank_idx + 1
    return out


def run_world_class_evaluation(checkpoint_path: Path, output_dir: Path, *, n_val_datasets: int = 24) -> Dict[str, Any]:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLevelLandmarkModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cat_tasks: Dict[str, str] = {}
    with DEFAULT_CATALOG.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("id") or r.get("dataset_hash"):
                cat_tasks[r.get("id") or r.get("dataset_hash")] = r["task"]

    val_records, seen_reg, seen_bin = [], set(), set()
    with gzip.open(DEFAULT_BUNDLE, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sc = rec["scope"]
            ds_id = sc["dataset_id"]
            if not _is_val_dataset(ds_id) or sc["split"] != "outer5-iid-fold0-v1":
                continue
            t = cat_tasks.get(ds_id, sc["task"])
            if t == "regression" and len(seen_reg) < (n_val_datasets // 2) and ds_id not in seen_reg and 150 <= sc["rows"] <= 2000 and 5 <= sc["columns"] <= 45:
                seen_reg.add(ds_id)
                val_records.append((t, rec))
            elif t == "binary" and len(seen_bin) < (n_val_datasets // 2) and ds_id not in seen_bin and 150 <= sc["rows"] <= 2000 and 5 <= sc["columns"] <= 45:
                seen_bin.add(ds_id)
                val_records.append((t, rec))
            if len(seen_reg) >= (n_val_datasets // 2) and len(seen_bin) >= (n_val_datasets // 2):
                break

    auditors = ("catboost", "kumo_tabpfn", "lightgbm", "xgboost", "mlp", "linear")
    methods = ("dope_mlc_v2", "rows", "arf", "kernel", "copula", "joint_diffusion_v1")
    levels = ("medium", "high", "ultra")

    by_method_level: Dict[str, Dict[str, List[float]]] = {
        f"{m}_{lvl}": {"retention": [], "driver_D": [], "spearman_rho": [], "top3_overlap": [], "btr_auc": [], "acc": [], "bacc": [], "roc_auc": [], "r2": []}
        for m in methods for lvl in levels
    }
    by_auditor_ultra: Dict[str, Dict[str, List[float]]] = {
        aud: {"retention": [], "driver_D": [], "spearman_rho": [], "top3_overlap": [], "real_score": [], "syn_score": [], "acc_real": [], "acc_syn": []}
        for aud in auditors
    }
    unlimited_scaling_check: Dict[str, List[float]] = {"1x_N_train": [], "2x_N_train": [], "5x_N_train": []}
    cell_scores_for_elo: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    for cat_task, rec in val_records:
        sc = rec["scope"]
        ds_id = sc["dataset_id"]
        csv_path = DEFAULT_SOURCE_CACHE / f"{sc['train_sha256']}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
        if sc.get("has_header", False) and rows:
            rows = rows[1:]
        n_total = len(rows)
        fit_idx = training_rows(n_total, sc["physical_table_id"], sc["split"], int(sc.get("split_seed", 20261023)))
        fit_set = set(fit_idx)
        test_idx = [i for i in range(n_total) if i not in fit_set]

        X_tr_raw, y_tr_raw = [rows[i][:-1] for i in fit_idx], [rows[i][-1] for i in fit_idx]
        X_te_raw, y_te_raw = [rows[i][:-1] for i in test_idx], [rows[i][-1] for i in test_idx]
        task_name = "classification" if cat_task == "binary" else "regression"
        scaler = FitOnlyTableTransform.fit(X_tr_raw, y_tr_raw, task=task_name)
        X_tr, y_tr = scaler.transform(X_tr_raw, y_tr_raw)
        X_te, y_te = scaler.transform(X_te_raw, y_te_raw)
        p_use = min(X_tr.shape[1], 64)
        X_tr, X_te = X_tr[:, :p_use], X_te[:, :p_use]

        oracle_res = {aud: _fit_and_eval_auditor(aud, X_tr, y_tr, X_te, y_te, task_name) for aud in auditors}

        for lvl in levels:
            enc = model.encode_table(X_tr, y_tr, rec["report"], task=task_name, level=lvl)
            if lvl == "ultra":
                for mult_name, mult in (("1x_N_train", 1), ("2x_N_train", 2), ("5x_N_train", 5)):
                    Xs_m, ys_m = model.generate_synthetic(enc, n_rows=len(X_tr) * mult, seed=100 + mult)
                    cb_m = _fit_and_eval_auditor("catboost", Xs_m, ys_m, X_te, y_te, task_name)
                    denom = max(oracle_res["catboost"]["primary_score"] - oracle_res["catboost"]["null_score"], 0.03)
                    unlimited_scaling_check[mult_name].append(float(np.clip((cb_m["primary_score"] - oracle_res["catboost"]["null_score"]) / denom, 0.0, 1.15)))

            for m_name in methods:
                X_syn, y_syn = model.generate_synthetic(enc, n_rows=len(X_tr), seed=20261023) if m_name == "dope_mlc_v2" else _baseline_generator(m_name, X_tr, y_tr, lvl, task_name, seed=20261023)
                aud_list = auditors if (lvl == "ultra" or m_name in ("dope_mlc_v2", "rows", "joint_diffusion_v1")) else ("catboost", "lightgbm", "linear")
                for aud in aud_list:
                    syn_r, ora_r = _fit_and_eval_auditor(aud, X_syn, y_syn, X_te, y_te, task_name), oracle_res[aud]
                    denom = max(ora_r["primary_score"] - ora_r["null_score"], 0.03)
                    if ora_r["primary_score"] - ora_r["null_score"] < 0.02:
                        continue
                    R = float(np.clip((syn_r["primary_score"] - ora_r["null_score"]) / denom, 0.0, 1.10))
                    imp_real, imp_syn = ora_r["importance"], syn_r["importance"]
                    D = float(np.clip(1.0 - 0.5 * np.sum(np.abs(imp_real - imp_syn)), 0.0, 1.0))
                    rho_val, _ = spearmanr(imp_real, imp_syn)
                    rho = float(rho_val) if math.isfinite(float(rho_val)) else 0.0
                    k_top = min(3, len(imp_real))
                    top_real, top_syn = set(np.argsort(imp_real)[::-1][:k_top].tolist()), set(np.argsort(imp_syn)[::-1][:k_top].tolist())
                    top3_ov = float(len(top_real & top_syn)) / float(max(k_top, 1))
                    btr = 100.0 * R * (0.85 + 0.15 * D)

                    key = f"{m_name}_{lvl}"
                    for k_metric, v_metric in (("retention", R), ("driver_D", D), ("spearman_rho", rho), ("top3_overlap", top3_ov), ("btr_auc", btr)):
                        by_method_level[key][k_metric].append(v_metric)
                    if task_name == "classification":
                        by_method_level[key]["acc"].append(syn_r["accuracy"])
                        by_method_level[key]["bacc"].append(syn_r["balanced_accuracy"])
                        by_method_level[key]["roc_auc"].append(syn_r["roc_auc"])
                    else:
                        by_method_level[key]["r2"].append(syn_r["r2"])

                    if aud in ("catboost", "lightgbm", "linear"):
                        cell_scores_for_elo.setdefault((ds_id, lvl, aud), {})[m_name] = btr

                    if m_name == "dope_mlc_v2" and lvl == "ultra":
                        for k_metric, v_metric in (("retention", R), ("driver_D", D), ("spearman_rho", rho), ("top3_overlap", top3_ov), ("real_score", ora_r["primary_score"]), ("syn_score", syn_r["primary_score"])):
                            by_auditor_ultra[aud][k_metric].append(v_metric)
                        if task_name == "classification":
                            by_auditor_ultra[aud]["acc_real"].append(ora_r["accuracy"])
                            by_auditor_ultra[aud]["acc_syn"].append(syn_r["accuracy"])

    elo_overall = compute_elo_ratings(cell_scores_for_elo, methods)
    elo_by_level = {lvl: compute_elo_ratings({k: v for k, v in cell_scores_for_elo.items() if k[1] == lvl}, methods) for lvl in levels}

    method_level_summary = {
        key: {
            "mean_retention_R": round(float(np.mean(stats["retention"])), 4),
            "mean_driver_fidelity_D": round(float(np.mean(stats["driver_D"])), 4),
            "mean_feature_importance_spearman_rho": round(float(np.mean(stats["spearman_rho"])), 4),
            "mean_top3_driver_overlap": round(float(np.mean(stats["top3_overlap"])), 4),
            "mean_btr_auc": round(float(np.mean(stats["btr_auc"])), 2),
            "mean_binary_accuracy": round(float(np.mean(stats["acc"])), 4) if stats["acc"] else None,
            "mean_binary_balanced_accuracy": round(float(np.mean(stats["bacc"])), 4) if stats["bacc"] else None,
            "mean_binary_roc_auc": round(float(np.mean(stats["roc_auc"])), 4) if stats["roc_auc"] else None,
            "mean_regression_r2": round(float(np.mean(stats["r2"])), 4) if stats["r2"] else None,
        }
        for key, stats in by_method_level.items() if stats["retention"]
    }
    auditor_summary = {
        aud: {
            "mean_real_test_score": round(float(np.mean(stats["real_score"])), 4),
            "mean_synthetic_transfer_test_score": round(float(np.mean(stats["syn_score"])), 4),
            "mean_retention_R": round(float(np.mean(stats["retention"])), 4),
            "mean_driver_fidelity_D": round(float(np.mean(stats["driver_D"])), 4),
            "mean_feature_importance_spearman_rho": round(float(np.mean(stats["spearman_rho"])), 4),
            "mean_top3_driver_overlap": round(float(np.mean(stats["top3_overlap"])), 4),
            "mean_binary_real_accuracy": round(float(np.mean(stats["acc_real"])), 4) if stats["acc_real"] else None,
            "mean_binary_syn_accuracy": round(float(np.mean(stats["acc_syn"])), 4) if stats["acc_syn"] else None,
        }
        for aud, stats in by_auditor_ultra.items() if stats["retention"]
    }
    receipt = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "datasets_evaluated": len(val_records),
        "elo_leaderboard_overall": elo_overall,
        "elo_leaderboard_by_level": elo_by_level,
        "method_by_level_summary": method_level_summary,
        "world_class_auditors_at_ultra": auditor_summary,
        "unlimited_synthetic_expansion_retention": {k: round(float(np.mean(v)), 4) for k, v in unlimited_scaling_check.items() if v},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "world_class_transfer_and_elo_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
