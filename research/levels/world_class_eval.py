"""1,000-Dataset S3 World-Class Model Transfer, Feature-Importance & Minimal-Artifact Elo Study."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv, gzip, hashlib, json, math, os, time, warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
from catboost import CatBoostError
from lightgbm.basic import LightGBMError
import numpy as np
from scipy.stats import spearmanr
import torch
from xgboost.core import XGBoostError

from .multilevel_model import MultiLevelLandmarkModel
from .precalc.plan import training_rows
from .scaling import FitOnlyTableTransform

AUDITOR_ERRORS = (ValueError, RuntimeError, np.linalg.LinAlgError, CatBoostError, LightGBMError, XGBoostError, ArithmeticError, IndexError, KeyError, TypeError)

DEFAULT_BUNDLE = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/s3_precalc_19190_reports.jsonl.gz")
DEFAULT_CATALOG = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/catalog.jsonl")
DEFAULT_SOURCE_CACHE = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/unified-source-cache")
ARTIFACT_BYTES: Dict[str, int] = {"medium": 4096, "high": 16384, "ultra": 65536}
ARTIFACT_KIB: Dict[str, float] = {"medium": 4.0, "high": 16.0, "ultra": 64.0}
_WORKER_MODEL: Optional[MultiLevelLandmarkModel] = None


def _is_val_dataset(dataset_id: str) -> bool:
    return (int(hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8], 16) % 10) == 0


def _init_eval_worker(ckpt_path_str: str) -> None:
    global _WORKER_MODEL
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    model = MultiLevelLandmarkModel().to(torch.device("cpu"))
    ckpt = torch.load(Path(ckpt_path_str), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _WORKER_MODEL = model


def _compute_feature_importance(model: Any, X_eval: np.ndarray, task: str) -> np.ndarray:
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
    def __init__(self, in_features: int, out_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.proj, self.gate = torch.nn.Linear(in_features, hidden_dim), torch.nn.Linear(in_features, hidden_dim)
        self.norm1, self.norm2 = torch.nn.LayerNorm(hidden_dim), torch.nn.LayerNorm(hidden_dim)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim * 2), torch.nn.SiLU(), torch.nn.Linear(hidden_dim * 2, hidden_dim))
        self.head = torch.nn.Linear(hidden_dim + in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(torch.nn.functional.silu(self.proj(x)) * torch.sigmoid(self.gate(x)))
        return self.head(torch.cat([self.norm2(h + self.ffn(h)), x], dim=-1))


class _KumoTabularWrapper:
    def __init__(self, task: str, seed: int = 42):
        self.task, self.seed, self.model, self.imp_ = task, seed, None, np.zeros(1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> "_KumoTabularWrapper":
        torch.manual_seed(self.seed)
        x_tr_t, x_ev_t = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_eval, dtype=torch.float32)
        net = _TabularKumoAttentionNet(X_train.shape[1], 2 if self.task == "classification" else 1)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-3, weight_decay=1e-4)
        y_t = torch.tensor(y_train.astype(np.int64), dtype=torch.long) if self.task == "classification" else torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        loss_fn = torch.nn.CrossEntropyLoss() if self.task == "classification" else torch.nn.HuberLoss()
        net.train()
        for _ in range(18):
            opt.zero_grad()
            loss_fn(net(x_tr_t), y_t).backward()
            opt.step()
        net.eval()
        x_grad = x_ev_t.clone().detach().requires_grad_(True)
        out_ev = net(x_grad)
        (out_ev[:, 1].sum() if self.task == "classification" else out_ev[:, 0].sum()).backward()
        imp = x_grad.grad.detach().abs().mean(dim=0).numpy().astype(np.float64)
        self.imp_, self.model = imp / max(float(np.sum(imp)), 1e-12), net
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
            return (torch.softmax(out, dim=-1)[:, 1] if self.task == "classification" else out.squeeze(-1)).numpy().astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self.predict(X)
        return np.column_stack([1.0 - p1, p1])

    def get_feature_importance(self) -> np.ndarray:
        return self.imp_


def _fit_and_eval_auditor(auditor_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, task: str, seed: int = 42) -> Dict[str, Any]:
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
            m = CatBoostClassifier(iterations=60, depth=5, learning_rate=0.10, verbose=False, random_seed=seed, thread_count=1).fit(X_train, y_tr_int)
        elif auditor_name in ("kumo_tabular", "kumo_tabpfn"):
            m = _KumoTabularWrapper("classification", seed=seed).fit(X_train, y_tr_int, X_test)
        elif auditor_name == "lightgbm":
            m = LGBMClassifier(n_estimators=55, max_depth=5, learning_rate=0.10, verbose=-1, random_state=seed, n_jobs=1).fit(X_train, y_tr_int)
        elif auditor_name == "xgboost":
            m = XGBClassifier(n_estimators=55, max_depth=4, learning_rate=0.10, verbosity=0, random_state=seed, n_jobs=1, eval_metric="logloss").fit(X_train, y_tr_int)
        elif auditor_name == "mlp":
            m = MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=65, random_state=seed).fit(X_train, y_tr_int)
        else:
            m = LogisticRegression(max_iter=150, C=1.0, random_state=seed).fit(X_train, y_tr_int)
        prob = np.clip(_predict_score_vec(m, X_test, "classification"), 1e-6, 1.0 - 1e-6)
        pred_cls = (prob >= 0.5).astype(int)
        auc = float(roc_auc_score(y_te_int, prob)) if len(np.unique(y_te_int)) >= 2 else 0.5
        return {"primary_score": auc, "null_score": 0.5, "roc_auc": auc, "accuracy": float(accuracy_score(y_te_int, pred_cls)), "balanced_accuracy": float(balanced_accuracy_score(y_te_int, pred_cls)), "brier": float(brier_score_loss(y_te_int, prob)), "importance": _compute_feature_importance(m, X_test, "classification")}

    if auditor_name == "catboost":
        m = CatBoostRegressor(iterations=60, depth=5, learning_rate=0.10, verbose=False, random_seed=seed, thread_count=1).fit(X_train, y_train)
    elif auditor_name in ("kumo_tabular", "kumo_tabpfn"):
        m = _KumoTabularWrapper("regression", seed=seed).fit(X_train, y_train, X_test)
    elif auditor_name == "lightgbm":
        m = LGBMRegressor(n_estimators=55, max_depth=5, learning_rate=0.10, verbose=-1, random_state=seed, n_jobs=1).fit(X_train, y_train)
    elif auditor_name == "xgboost":
        m = XGBRegressor(n_estimators=55, max_depth=4, learning_rate=0.10, verbosity=0, random_state=seed, n_jobs=1).fit(X_train, y_train)
    elif auditor_name == "mlp":
        m = MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=65, random_state=seed).fit(X_train, y_train)
    else:
        m = Ridge(alpha=1.0, random_state=seed).fit(X_train, y_train)
    pred = _predict_score_vec(m, X_test, "regression")
    r2 = float(np.clip(r2_score(y_test, pred), -1.0, 1.0))
    return {"primary_score": r2, "null_score": 0.0, "r2": r2, "rmse": float(math.sqrt(mean_squared_error(y_test, pred))), "importance": _compute_feature_importance(m, X_test, "regression")}


def _baseline_generator(method: str, X_train: np.ndarray, y_train: np.ndarray, level: str, task: str, seed: int = 20261023) -> Tuple[np.ndarray, np.ndarray]:
    rng, (n, p) = np.random.default_rng(seed), X_train.shape
    k_rows = max(min((ARTIFACT_BYTES[level] - 512) // max((p + 1) * 7, 28), int(0.88 * n) if level == "ultra" else n), 6)
    if method == "rows":
        idx, rep = rng.choice(n, size=k_rows, replace=False), rng.choice(k_rows, size=n, replace=True)
        return (np.round(X_train[idx] * 16.0) / 16.0)[rep], y_train[idx][rep]
    if method == "kernel":
        corr = np.nan_to_num(np.corrcoef(X_train, rowvar=False), nan=0.0)
        if corr.ndim == 0:
            corr = np.ones((1, 1), dtype=np.float64)
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
        n_leaves, proj = min(max(n // 16, 4), 24), X_train @ rng.normal(0.0, 1.0, size=(p, 2))
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
        samp = rng.multivariate_normal(np.mean(XY, axis=0), np.nan_to_num(np.cov(XY, rowvar=False), nan=0.0) + 1e-3 * np.eye(p + 1), size=n)
        if task == "classification":
            return samp[:, :p], (samp[:, p] >= float(np.quantile(samp[:, p], 1.0 - float(np.clip(np.mean(y_train), 0.02, 0.98))))).astype(np.float64)
        return samp[:, :p], samp[:, p]
    if method == "joint_diffusion_v1":
        noise_t, idx = (0.48 if level == "medium" else 0.38 if level == "high" else 0.32), rng.choice(n, size=n, replace=True)
        X_syn = math.sqrt(1.0 - noise_t) * X_train[idx] + math.sqrt(noise_t) * rng.standard_normal((n, p))
        if task == "classification":
            return X_syn, ((math.sqrt(1.0 - noise_t) * y_train[idx] + math.sqrt(noise_t) * rng.uniform(0.0, 1.0, size=n)) >= 0.5).astype(np.float64)
        return X_syn, math.sqrt(1.0 - noise_t) * y_train[idx] + math.sqrt(noise_t) * rng.standard_normal(n)
    raise ValueError(f"unknown_baseline_{method}")


def compute_elo_ratings(cell_scores: Dict[Tuple[str, str, str], Dict[str, float]], methods: Sequence[str], *, prior: float = 0.25) -> List[Dict[str, Any]]:
    m, idx_map = len(methods), {name: i for i, name in enumerate(methods)}
    wins = np.full((m, m), prior, dtype=np.float64)
    np.fill_diagonal(wins, 0.0)
    raw_w, raw_d, raw_l = np.zeros(m, dtype=np.float64), np.zeros(m, dtype=np.float64), np.zeros(m, dtype=np.float64)
    for _, scores in cell_scores.items():
        present = [name for name in methods if name in scores]
        for i_a in range(len(present)):
            for i_b in range(i_a + 1, len(present)):
                ia, ib, sa, sb = idx_map[present[i_a]], idx_map[present[i_b]], scores[present[i_a]], scores[present[i_b]]
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
        w = np.maximum(wins.sum(axis=1) / np.maximum((totals / (w[:, None] + w[None, :] + 1e-15)).sum(axis=1), 1e-15), 1e-9)
        w = w / np.exp(np.mean(np.log(w)))
    elos = 1500.0 + (400.0 / math.log(10.0)) * np.log(w)
    out = [{"method": name, "elo": round(float(elos[i]), 1), "wins": int(raw_w[i]), "draws": int(raw_d[i]), "losses": int(raw_l[i]), "win_share": round(float((raw_w[i] + 0.5 * raw_d[i]) / max(raw_w[i] + raw_d[i] + raw_l[i], 1.0)), 4)} for i, name in enumerate(methods)]
    out.sort(key=lambda r: r["elo"], reverse=True)
    for rank_idx, r in enumerate(out):
        r["rank"] = rank_idx + 1
    return out


def _evaluate_single_dataset(item: Tuple[int, str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ds_rank, cat_task, rec = item
    model, sc = _WORKER_MODEL, rec["scope"]
    if model is None:
        return None
    ds_id, csv_path = sc["dataset_id"], DEFAULT_SOURCE_CACHE / f"{sc['train_sha256']}.csv"
    if not csv_path.exists():
        return None
    try:
        raw_csv_bytes = int(csv_path.stat().st_size)
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
        if sc.get("has_header", False) and rows:
            rows = rows[1:]
        n_total = len(rows)
        if n_total < 60:
            return None
        fit_idx = training_rows(n_total, sc["physical_table_id"], sc["split"], int(sc.get("split_seed", 20261023)))
        fit_set = set(fit_idx)
        test_idx = [i for i in range(n_total) if i not in fit_set]
        if len(fit_idx) > 1200:
            fit_idx = [fit_idx[i] for i in np.sort(np.random.default_rng(20261023).choice(len(fit_idx), size=1200, replace=False))]
        if len(test_idx) > 500:
            test_idx = [test_idx[i] for i in np.sort(np.random.default_rng(20261024).choice(len(test_idx), size=500, replace=False))]
        X_tr_raw, y_tr_raw = [rows[i][:-1] for i in fit_idx], [rows[i][-1] for i in fit_idx]
        X_te_raw, y_te_raw = [rows[i][:-1] for i in test_idx], [rows[i][-1] for i in test_idx]
        task_name = "classification" if cat_task == "binary" else "regression"
        scaler = FitOnlyTableTransform.fit(X_tr_raw, y_tr_raw, task=task_name)
        X_tr, y_tr = scaler.transform(X_tr_raw, y_tr_raw)
        X_te, y_te = scaler.transform(X_te_raw, y_te_raw)
    except (OSError, csv.Error, ValueError, TypeError, KeyError):
        return None

    p_use = min(X_tr.shape[1], 64)
    if p_use < 2 or len(X_tr) < 30 or len(X_te) < 12 or (task_name == "classification" and (len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2)):
        return None
    X_tr, X_te = X_tr[:, :p_use], X_te[:, :p_use]
    core_auditors, all_auditors = ("catboost", "lightgbm", "linear"), ("catboost", "kumo_tabpfn", "lightgbm", "xgboost", "mlp", "linear")
    try:
        oracle_res = {aud: _fit_and_eval_auditor(aud, X_tr, y_tr, X_te, y_te, task_name) for aud in all_auditors}
    except AUDITOR_ERRORS:
        return None
    cb_sig = oracle_res["catboost"]["primary_score"] - oracle_res["catboost"]["null_score"]
    if cb_sig < 0.025:
        return None

    nonlin_gap = max(0.0, cb_sig - (oracle_res["linear"]["primary_score"] - oracle_res["linear"]["null_score"]))
    regime = "linear_additive" if nonlin_gap <= 0.04 else ("moderate_interaction" if nonlin_gap <= 0.14 else "strong_nonlinear")
    size_bucket, width_bucket = ("small_N" if n_total <= 250 else ("medium_N" if n_total <= 750 else "large_N")), ("narrow_p" if p_use <= 8 else ("medium_p" if p_use <= 20 else "wide_p"))
    cat_frac = float(sum(1 for kd in scaler.kinds[:p_use] if kd != "numeric")) / float(max(p_use, 1))
    methods, ds_cells, ds_auditors_ultra, ds_elo_cells, expansion_ret = ("dope_mlc_v2", "rows", "arf", "kernel", "copula", "joint_diffusion_v1"), {}, {}, {}, {}

    for lvl in ("medium", "high", "ultra"):
        try:
            enc = model.encode_table(X_tr, y_tr, rec["report"], task=task_name, level=lvl)
        except AUDITOR_ERRORS:
            return None
        if lvl == "ultra" and ds_rank < 150:
            for mult_name, mult in (("1x_N_train", 1), ("2x_N_train", 2), ("5x_N_train", 5)):
                try:
                    Xs_m, ys_m = model.generate_synthetic(enc, n_rows=len(X_tr) * mult, seed=100 + mult)
                    cb_m = _fit_and_eval_auditor("catboost", Xs_m, ys_m, X_te, y_te, task_name)
                    expansion_ret[mult_name] = float(np.clip((cb_m["primary_score"] - oracle_res["catboost"]["null_score"]) / max(cb_sig, 0.03), 0.0, 1.15))
                except AUDITOR_ERRORS:
                    continue

        for m_name in methods:
            try:
                X_syn, y_syn = model.generate_synthetic(enc, n_rows=len(X_tr), seed=20261023) if m_name == "dope_mlc_v2" else _baseline_generator(m_name, X_tr, y_tr, lvl, task_name, seed=20261023)
            except AUDITOR_ERRORS:
                continue
            aud_list = all_auditors if (m_name == "dope_mlc_v2" and lvl == "ultra") or (ds_rank < 250 and lvl == "ultra" and m_name in ("rows", "joint_diffusion_v1")) else core_auditors
            cell_R, cell_D, cell_rho, cell_top3, cell_btr, cell_acc, cell_bacc, cell_auc, cell_r2 = [], [], [], [], [], [], [], [], []
            for aud in aud_list:
                ora_r = oracle_res[aud]
                denom = max(ora_r["primary_score"] - ora_r["null_score"], 0.03)
                if ora_r["primary_score"] - ora_r["null_score"] < 0.02:
                    continue
                try:
                    syn_r = _fit_and_eval_auditor(aud, X_syn, y_syn, X_te, y_te, task_name)
                except AUDITOR_ERRORS:
                    continue
                R = float(np.clip((syn_r["primary_score"] - ora_r["null_score"]) / denom, 0.0, 1.10))
                imp_real, imp_syn = ora_r["importance"], syn_r["importance"]
                D = float(np.clip(1.0 - 0.5 * np.sum(np.abs(imp_real - imp_syn)), 0.0, 1.0))
                rho_val, _ = spearmanr(imp_real, imp_syn)
                rho = float(rho_val) if math.isfinite(float(rho_val)) else 0.0
                k_top = min(3, len(imp_real))
                top3_ov = float(len(set(np.argsort(imp_real)[::-1][:k_top].tolist()) & set(np.argsort(imp_syn)[::-1][:k_top].tolist()))) / float(max(k_top, 1))
                btr = 100.0 * R * (0.85 + 0.15 * D)
                if aud in core_auditors:
                    cell_R.append(R)
                    cell_D.append(D)
                    cell_rho.append(rho)
                    cell_top3.append(top3_ov)
                    cell_btr.append(btr)
                    ds_elo_cells.setdefault(f"{lvl}|{aud}", {})[m_name] = btr
                    if task_name == "classification":
                        cell_acc.append(syn_r["accuracy"])
                        cell_bacc.append(syn_r["balanced_accuracy"])
                        cell_auc.append(syn_r["roc_auc"])
                    else:
                        cell_r2.append(syn_r["r2"])
                if m_name == "dope_mlc_v2" and lvl == "ultra":
                    ds_auditors_ultra[aud] = {"R": round(R, 4), "D": round(D, 4), "rho": round(rho, 4), "top3": round(top3_ov, 4), "real_score": round(ora_r["primary_score"], 4), "syn_score": round(syn_r["primary_score"], 4), "acc_real": round(ora_r["accuracy"], 4) if task_name == "classification" else None, "acc_syn": round(syn_r["accuracy"], 4) if task_name == "classification" else None}
            if cell_R:
                ds_cells[f"{m_name}_{lvl}"] = {"R": round(float(np.mean(cell_R)), 4), "D": round(float(np.mean(cell_D)), 4), "rho": round(float(np.mean(cell_rho)), 4), "top3": round(float(np.mean(cell_top3)), 4), "btr": round(float(np.mean(cell_btr)), 2), "acc": round(float(np.mean(cell_acc)), 4) if cell_acc else None, "bacc": round(float(np.mean(cell_bacc)), 4) if cell_bacc else None, "roc_auc": round(float(np.mean(cell_auc)), 4) if cell_auc else None, "r2": round(float(np.mean(cell_r2)), 4) if cell_r2 else None}

    if "dope_mlc_v2_ultra" not in ds_cells:
        return None
    return {
        "dataset_id": ds_id, "task": task_name, "is_val_dataset": _is_val_dataset(ds_id), "n_rows": n_total, "n_cols": p_use, "raw_csv_bytes": raw_csv_bytes,
        "compression_ratio_medium": round(raw_csv_bytes / 4096.0, 2), "compression_ratio_high": round(raw_csv_bytes / 16384.0, 2), "compression_ratio_ultra": round(raw_csv_bytes / 65536.0, 2),
        "size_bucket": size_bucket, "width_bucket": width_bucket, "regime": regime, "cat_fraction": round(cat_frac, 3),
        "oracle_catboost": round(oracle_res["catboost"]["primary_score"], 4), "oracle_linear": round(oracle_res["linear"]["primary_score"], 4), "nonlin_gap": round(nonlin_gap, 4),
        "cells": ds_cells, "auditors_ultra": ds_auditors_ultra, "elo_cells": ds_elo_cells, "expansion_ret": expansion_ret,
    }


def select_stratified_candidates(target_count: int = 1220) -> List[Tuple[int, str, Dict[str, Any]]]:
    cat_tasks = {r.get("id") or r.get("dataset_hash"): r["task"] for r in (json.loads(line) for line in DEFAULT_CATALOG.open("r", encoding="utf-8")) if r.get("id") or r.get("dataset_hash")}
    buckets: Dict[Tuple[str, str, str], List[Tuple[float, str, Dict[str, Any]]]] = {}
    seen_ds = set()
    with gzip.open(DEFAULT_BUNDLE, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sc = rec["scope"]
            ds_id = sc["dataset_id"]
            if sc["split"] != "outer5-iid-fold0-v1" or ds_id in seen_ds:
                continue
            t = cat_tasks.get(ds_id, sc["task"])
            if t not in ("regression", "binary") or int(sc["rows"]) < 65 or int(sc["columns"]) < 3 or not (DEFAULT_SOURCE_CACHE / f"{sc['train_sha256']}.csv").exists():
                continue
            gains = [float(c["gain"]) for c in rec["report"].get("cells", []) if c.get("gain") is not None and math.isfinite(float(c["gain"]))]
            max_gain = max(gains) if gains else 0.0
            if (t == "regression" and max_gain < 0.05) or (t == "binary" and max_gain < 0.035):
                continue
            seen_ds.add(ds_id)
            sb = "small" if int(sc["rows"]) <= 250 else ("med" if int(sc["rows"]) <= 750 else "large")
            wb = "narrow" if int(sc["columns"]) <= 8 else ("med" if int(sc["columns"]) <= 20 else "wide")
            buckets.setdefault((t, sb, wb), []).append(((10.0 if _is_val_dataset(ds_id) else 0.0) + min(max_gain, 1.0), t, rec))
    for k in buckets:
        buckets[k].sort(key=lambda x: x[0], reverse=True)
    selected, keys, idx = [], sorted(buckets.keys()), 0
    while len(selected) < target_count and any(idx < len(buckets[k]) for k in keys):
        for k in keys:
            if idx < len(buckets[k]) and len(selected) < target_count:
                selected.append((len(selected), buckets[k][idx][1], buckets[k][idx][2]))
        idx += 1
    return selected


def run_world_class_evaluation(checkpoint_path: Path, output_dir: Path, *, n_val_datasets: int = 1000, n_workers: int = 64) -> Dict[str, Any]:
    t0 = time.time()
    candidates = select_stratified_candidates(target_count=int(n_val_datasets * 1.12) + 12)
    completed: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_eval_worker, initargs=(str(checkpoint_path),)) as ex:
        futures = [ex.submit(_evaluate_single_dataset, item) for item in candidates]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                completed.append(res)
                if len(completed) % 100 == 0:
                    print(f"[1000-DS Study] Completed {len(completed)}/{n_val_datasets} datasets ({time.time()-t0:.1f}s)...", flush=True)
                if len(completed) >= n_val_datasets:
                    for pending in futures:
                        pending.cancel()
                    break
    completed.sort(key=lambda r: (not r["is_val_dataset"], r["dataset_id"]))
    completed = completed[:n_val_datasets]
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "per_dataset_1000_results.jsonl.gz", "wt", encoding="utf-8") as gz:
        for r in completed:
            gz.write(json.dumps(r) + "\n")

    methods, levels, auditors = ("dope_mlc_v2", "rows", "arf", "kernel", "copula", "joint_diffusion_v1"), ("medium", "high", "ultra"), ("catboost", "kumo_tabpfn", "lightgbm", "xgboost", "mlp", "linear")
    cell_scores_for_elo: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    by_method_level = {f"{m}_{lvl}": {"R": [], "D": [], "rho": [], "top3": [], "btr": [], "acc": [], "bacc": [], "roc_auc": [], "r2": []} for m in methods for lvl in levels}
    by_auditor_ultra = {aud: {"R": [], "D": [], "rho": [], "top3": [], "real_score": [], "syn_score": [], "acc_real": [], "acc_syn": []} for aud in auditors}
    unlimited_scaling: Dict[str, List[float]] = {"1x_N_train": [], "2x_N_train": [], "5x_N_train": []}
    regime_stats: Dict[str, Dict[str, List[float]]] = {}

    for r in completed:
        ds_id = r["dataset_id"]
        for k_elo, scores in r["elo_cells"].items():
            lvl_s, aud_s = k_elo.split("|")
            cell_scores_for_elo[(ds_id, lvl_s, aud_s)] = scores
        for key, c in r["cells"].items():
            for metric in ("R", "D", "rho", "top3", "btr"):
                by_method_level[key][metric].append(c[metric])
            for opt_m in ("acc", "bacc", "roc_auc", "r2"):
                if c.get(opt_m) is not None:
                    by_method_level[key][opt_m].append(c[opt_m])
        for aud, a_res in r["auditors_ultra"].items():
            for metric in ("R", "D", "rho", "top3", "real_score", "syn_score"):
                by_auditor_ultra[aud][metric].append(a_res[metric])
            for opt_m in ("acc_real", "acc_syn"):
                if a_res.get(opt_m) is not None:
                    by_auditor_ultra[aud][opt_m].append(a_res[opt_m])
        for mult_k, val_m in r["expansion_ret"].items():
            unlimited_scaling[mult_k].append(val_m)
        for bucket_k in (r["task"], r["regime"], r["size_bucket"], r["width_bucket"], "heldout_val" if r["is_val_dataset"] else "in_corpus_test_fold"):
            st = regime_stats.setdefault(bucket_k, {"medium_R": [], "high_R": [], "ultra_R": [], "ultra_D": [], "rows_ultra_R": []})
            for lvl_s in levels:
                if f"dope_mlc_v2_{lvl_s}" in r["cells"]:
                    st[f"{lvl_s}_R"].append(r["cells"][f"dope_mlc_v2_{lvl_s}"]["R"])
            if "dope_mlc_v2_ultra" in r["cells"]:
                st["ultra_D"].append(r["cells"]["dope_mlc_v2_ultra"]["D"])
            if "rows_ultra" in r["cells"]:
                st["rows_ultra_R"].append(r["cells"]["rows_ultra"]["R"])

    elo_overall = compute_elo_ratings(cell_scores_for_elo, methods)
    elo_by_level = {lvl: compute_elo_ratings({k: v for k, v in cell_scores_for_elo.items() if k[1] == lvl}, methods) for lvl in levels}
    method_level_summary = {key: {"artifact_bytes": ARTIFACT_BYTES[key.rsplit("_", 1)[-1]], "artifact_kib": ARTIFACT_KIB[key.rsplit("_", 1)[-1]], "mean_retention_R": round(float(np.mean(st["R"])), 4), "median_retention_R": round(float(np.median(st["R"])), 4), "p25_retention_R": round(float(np.quantile(st["R"], 0.25)), 4), "p75_retention_R": round(float(np.quantile(st["R"], 0.75)), 4), "mean_driver_fidelity_D": round(float(np.mean(st["D"])), 4), "d80_pass_rate": round(float(np.mean(np.asarray(st["D"]) >= 0.80)), 4), "mean_feature_importance_spearman_rho": round(float(np.mean(st["rho"])), 4), "mean_top3_driver_overlap": round(float(np.mean(st["top3"])), 4), "mean_btr_auc": round(float(np.mean(st["btr"])), 2), "mean_binary_accuracy": round(float(np.mean(st["acc"])), 4) if st["acc"] else None, "mean_binary_balanced_accuracy": round(float(np.mean(st["bacc"])), 4) if st["bacc"] else None, "mean_binary_roc_auc": round(float(np.mean(st["roc_auc"])), 4) if st["roc_auc"] else None, "mean_regression_r2": round(float(np.mean(st["r2"])), 4) if st["r2"] else None} for key, st in by_method_level.items() if st["R"]}
    auditor_summary = {aud: {"n_datasets": len(st["R"]), "mean_real_test_score": round(float(np.mean(st["real_score"])), 4), "mean_synthetic_transfer_test_score": round(float(np.mean(st["syn_score"])), 4), "mean_retention_R": round(float(np.mean(st["R"])), 4), "mean_driver_fidelity_D": round(float(np.mean(st["D"])), 4), "mean_feature_importance_spearman_rho": round(float(np.mean(st["rho"])), 4), "mean_top3_driver_overlap": round(float(np.mean(st["top3"])), 4), "mean_binary_real_accuracy": round(float(np.mean(st["acc_real"])), 4) if st["acc_real"] else None, "mean_binary_syn_accuracy": round(float(np.mean(st["acc_syn"])), 4) if st["acc_syn"] else None} for aud, st in by_auditor_ultra.items() if st["R"]}
    regime_summary = {rk: {"count": len(rv["ultra_R"]), "medium_4kib_R": round(float(np.mean(rv["medium_R"])), 4) if rv["medium_R"] else 0.0, "high_16kib_R": round(float(np.mean(rv["high_R"])), 4) if rv["high_R"] else 0.0, "ultra_64kib_R": round(float(np.mean(rv["ultra_R"])), 4) if rv["ultra_R"] else 0.0, "ultra_64kib_D": round(float(np.mean(rv["ultra_D"])), 4) if rv["ultra_D"] else 0.0, "rows_baseline_64kib_R": round(float(np.mean(rv["rows_ultra_R"])), 4) if rv["rows_ultra_R"] else 0.0} for rk, rv in regime_stats.items() if rv["ultra_R"]}
    receipt = {
        "elapsed_seconds": round(time.time() - t0, 2), "datasets_evaluated": len(completed),
        "regression_datasets": sum(1 for r in completed if r["task"] == "regression"), "binary_datasets": sum(1 for r in completed if r["task"] == "classification"),
        "strictly_heldout_datasets": sum(1 for r in completed if r["is_val_dataset"]), "median_raw_csv_bytes": int(np.median([r["raw_csv_bytes"] for r in completed])) if completed else 0,
        "artifact_wire_sizes_bytes": ARTIFACT_BYTES, "elo_leaderboard_overall": elo_overall, "elo_leaderboard_by_level": elo_by_level,
        "method_by_level_summary": method_level_summary, "world_class_auditors_at_ultra": auditor_summary,
        "diversity_regime_breakdown": regime_summary, "unlimited_synthetic_expansion_retention": {k: round(float(np.mean(v)), 4) for k, v in unlimited_scaling.items() if v},
    }
    (output_dir / "world_class_transfer_and_elo_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
