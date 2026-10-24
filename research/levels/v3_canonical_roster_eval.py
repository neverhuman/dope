"""Bounded Canonical 6-Auditor, Raw-Receiver & Equal-Wire Comparator Evaluation for DOPE-MLC-v3."""
from __future__ import annotations

import hashlib, math, struct, time, warnings, zlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
from catboost import CatBoostClassifier, CatBoostError, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm.basic import LightGBMError
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
import torch
from xgboost import XGBClassifier, XGBRegressor
from xgboost.core import XGBoostError

from .budget_planner import TIER_BYTE_CAPS, pack_compact_scaler_sidecar, unpack_compact_scaler_sidecar
from .scaling import FitOnlyTableTransform
from .v3_ledger_replay import _compute_retention_stats

V3_OUT_DIR = Path("/home/ubuntu/dope/target/s3-multilevel-training-20261023/v3")
V3_MANIFEST_PATH = V3_OUT_DIR / "v3_corpus_exposure_and_split_manifest.json"
DEFAULT_V2_CKPT = Path("/home/ubuntu/dope/target/s3-multilevel-training-20261023/dope_mlc_v2_disjoint1000_best.pt")

AUDITOR_ERRORS = (
    ValueError, RuntimeError, np.linalg.LinAlgError, CatBoostError,
    LightGBMError, XGBoostError, ArithmeticError, IndexError, KeyError, TypeError,
)
CANONICAL_6_AUDITORS = ("catboost", "lightgbm", "xgboost", "mlp", "linear", "local_gated_attention")
CORE_COMPARATOR_AUDITORS = ("catboost", "lightgbm", "linear")

CANONICAL_V3_AUDITOR_PROTOCOL_SPEC: Dict[str, Any] = {
    "protocol_id": "dope.v3.canonical_6aud_bounded_cpu_protocol.v1",
    "is_identical_to_historical_v2_world_class_eval_wrapper": False,
    "preprocessing": {
        "max_train_rows": 340, "max_test_rows": 160, "max_p_features": 36,
        "split_seed_train_subsample": 20261023, "split_seed_test_subsample": 20261024,
        "transform": "FitOnlyTableTransform_train_fit_only",
        "label_integrity_policy": "readonly_authentic_labels_constant_predictor_or_invertible_contiguous_bijection_never_perturb_flip_or_invent",
    },
    "auditors": {
        "catboost": {
            "regression": {"iterations": 25, "depth": 4, "learning_rate": 0.14, "random_seed": 42, "thread_count": 1},
            "classification": {"iterations": 25, "depth": 4, "learning_rate": 0.14, "random_seed": 42, "thread_count": 1},
            "multiclass": {"iterations": 24, "depth": 4, "learning_rate": 0.15, "loss_function": "MultiClass", "random_seed": 42, "thread_count": 1},
        },
        "lightgbm": {
            "regression": {"n_estimators": 25, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.14, "random_state": 42, "n_jobs": 1},
            "classification": {"n_estimators": 25, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.14, "random_state": 42, "n_jobs": 1},
            "multiclass": {"n_estimators": 24, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.14, "random_state": 42, "n_jobs": 1},
        },
        "xgboost": {
            "regression": {"n_estimators": 25, "max_depth": 4, "learning_rate": 0.14, "tree_method": "hist", "random_state": 42, "n_jobs": 1},
            "classification": {"n_estimators": 25, "max_depth": 4, "learning_rate": 0.14, "tree_method": "hist", "eval_metric": "logloss", "random_state": 42, "n_jobs": 1},
            "multiclass": {"n_estimators": 24, "max_depth": 4, "learning_rate": 0.14, "tree_method": "hist", "eval_metric": "mlogloss", "random_state": 42, "n_jobs": 1},
        },
        "mlp": {"hidden_layer_sizes": [32, 16], "max_iter": 28, "random_state": 42},
        "linear": {"regression_ridge_alpha": 1.0, "classification_logreg_max_iter": 100, "classification_logreg_C": 1.0, "random_state": 42},
        "local_gated_attention": {
            "class_name": "LocalGatedAttentionAuditor", "network_class": "_LocalGatedAttentionNet",
            "is_official_kumo_or_tabpfn": False, "hidden_dim": 32, "adamw_steps": 12,
            "learning_rate": 0.006, "weight_decay": 0.0001, "seed": 42,
        },
    },
}


def _read_proc_stat_totals() -> Tuple[int, int]:
    try:
        parts = [int(x) for x in Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]]
        return sum(parts), parts[3] + (parts[4] if len(parts) > 4 else 0)
    except OSError:
        return 0, 0


def measure_host_cpu_percent(window_s: float = 0.15) -> float:
    t1, i1 = _read_proc_stat_totals()
    if t1 <= 0:
        return 0.0
    time.sleep(max(window_s, 0.02))
    t2, i2 = _read_proc_stat_totals()
    return float(np.clip(100.0 * (1.0 - float(max(i2 - i1, 0)) / float(max(t2 - t1, 1))), 0.0, 100.0))


class _LocalGatedAttentionNet(torch.nn.Module):
    """Local PyTorch SiLU-gated attention-residual network (never official KumoTabular or TabPFN)."""

    def __init__(self, in_features: int, out_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.proj, self.gate = torch.nn.Linear(in_features, hidden_dim), torch.nn.Linear(in_features, hidden_dim)
        self.norm1, self.norm2 = torch.nn.LayerNorm(hidden_dim), torch.nn.LayerNorm(hidden_dim)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim * 2), torch.nn.SiLU(), torch.nn.Linear(hidden_dim * 2, hidden_dim))
        self.head = torch.nn.Linear(hidden_dim + in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(torch.nn.functional.silu(self.proj(x)) * torch.sigmoid(self.gate(x)))
        return self.head(torch.cat([self.norm2(h + self.ffn(h)), x], dim=-1))


class LocalGatedAttentionAuditor:
    """From-scratch per-table local gated-attention auditor (`is_official_kumo_or_tabpfn = False`)."""

    def __init__(self, task: str, n_classes: int = 2, seed: int = 42):
        self.task, self.n_classes, self.seed = task, max(int(n_classes), 2), seed
        self.model: Optional[_LocalGatedAttentionNet] = None
        self.imp_ = np.zeros(1, dtype=np.float64)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, compute_importance: bool = True) -> "LocalGatedAttentionAuditor":
        cfg = CANONICAL_V3_AUDITOR_PROTOCOL_SPEC["auditors"]["local_gated_attention"]
        torch.manual_seed(self.seed)
        x_tr = torch.tensor(np.nan_to_num(X_train, nan=0.0), dtype=torch.float32)
        net = _LocalGatedAttentionNet(x_tr.shape[1], 1 if self.task == "regression" else self.n_classes, hidden_dim=int(cfg["hidden_dim"]))
        opt = torch.optim.AdamW(net.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1) if self.task == "regression" else torch.tensor(np.clip(np.rint(y_train).astype(np.int64), 0, self.n_classes - 1), dtype=torch.long)
        loss_fn: torch.nn.Module = torch.nn.HuberLoss() if self.task == "regression" else torch.nn.CrossEntropyLoss()
        net.train()
        for _ in range(int(cfg["adamw_steps"])):
            opt.zero_grad()
            loss_fn(net(x_tr), y_t).backward()
            opt.step()
        net.eval()
        if compute_importance:
            x_grad = torch.tensor(np.nan_to_num(X_eval, nan=0.0), dtype=torch.float32, requires_grad=True)
            out_ev = net(x_grad)
            (out_ev[:, 0].sum() if self.task == "regression" else out_ev.abs().sum()).backward()
            imp = x_grad.grad.detach().abs().mean(dim=0).numpy().astype(np.float64)
            self.imp_ = imp / max(float(np.sum(imp)), 1e-12)
        else:
            self.imp_ = np.full(x_tr.shape[1], 1.0 / float(max(x_tr.shape[1], 1)), dtype=np.float64)
        self.model = net
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        with torch.no_grad():
            out = self.model(torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32))
            if self.task == "regression":
                return out.squeeze(-1).numpy().astype(np.float64)
            if self.task == "classification":
                return torch.softmax(out, dim=-1)[:, 1].numpy().astype(np.float64)
            return torch.argmax(out, dim=-1).numpy().astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        with torch.no_grad():
            return torch.softmax(self.model(torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32)), dim=-1).numpy().astype(np.float64)

    def get_feature_importance(self) -> np.ndarray:
        return self.imp_


def _extract_feature_importance(model: Any, X_eval: np.ndarray, task: str) -> np.ndarray:
    p = X_eval.shape[1]
    if hasattr(model, "get_feature_importance"):
        imp = np.asarray(model.get_feature_importance(), dtype=np.float64).reshape(-1)
    elif hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64).reshape(-1)
    elif hasattr(model, "coef_"):
        c = np.asarray(model.coef_, dtype=np.float64)
        imp = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c).reshape(-1)
    elif hasattr(model, "coefs_") and len(model.coefs_) > 0:
        imp = np.mean(np.abs(np.asarray(model.coefs_[0], dtype=np.float64)), axis=1).reshape(-1)
    else:
        imp = np.ones(p, dtype=np.float64)
    imp = np.maximum(np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if imp.size != p:
        imp = np.resize(imp, p)
    tot = float(np.sum(imp))
    return (imp / tot) if tot > 1e-12 else np.full(p, 1.0 / float(max(p, 1)), dtype=np.float64)


def _compute_driver_fidelity(imp_real: np.ndarray, imp_syn: np.ndarray) -> Dict[str, float]:
    p = min(imp_real.size, imp_syn.size)
    if p <= 1:
        return {"spearman_rho": 1.0, "top3_overlap": 1.0, "driver_fidelity_D": 1.0}
    r_v, s_v = imp_real[:p], imp_syn[:p]
    rho_raw = float(spearmanr(r_v, s_v).statistic) if (np.std(r_v) > 1e-12 and np.std(s_v) > 1e-12) else 0.0
    if not math.isfinite(rho_raw):
        rho_raw = 0.0
    k = min(3, p)
    top3 = float(len(set(np.argsort(r_v)[-k:].tolist()).intersection(set(np.argsort(s_v)[-k:].tolist())))) / float(k)
    d_score = 0.5 * (0.5 * (rho_raw + 1.0)) + 0.5 * top3
    return {"spearman_rho": round(rho_raw, 5), "top3_overlap": round(top3, 5), "driver_fidelity_D": round(float(np.clip(d_score, 0.0, 1.0)), 5)}


def _fit_discrete_learner(auditor_name: str, X_tr: np.ndarray, y_tr_contig: np.ndarray, X_te: np.ndarray, n_cls_fit: int, seed: int, compute_importance: bool) -> Any:
    spec = CANONICAL_V3_AUDITOR_PROTOCOL_SPEC["auditors"]
    mode_k = "classification" if n_cls_fit == 2 else "multiclass"
    if auditor_name == "catboost":
        return CatBoostClassifier(verbose=False, **{**spec["catboost"][mode_k], "random_seed": seed}).fit(X_tr, y_tr_contig)
    if auditor_name == "lightgbm":
        return LGBMClassifier(verbose=-1, **{**spec["lightgbm"][mode_k], "random_state": seed}).fit(X_tr, y_tr_contig)
    if auditor_name == "xgboost":
        return XGBClassifier(verbosity=0, **{**spec["xgboost"][mode_k], "random_state": seed}).fit(X_tr, y_tr_contig)
    if auditor_name == "mlp":
        return MLPClassifier(hidden_layer_sizes=tuple(spec["mlp"]["hidden_layer_sizes"]), max_iter=int(spec["mlp"]["max_iter"]), random_state=seed).fit(X_tr, y_tr_contig)
    if auditor_name == "local_gated_attention":
        return LocalGatedAttentionAuditor(mode_k, n_classes=n_cls_fit, seed=seed).fit(X_tr, y_tr_contig, X_te, compute_importance=compute_importance)
    return LogisticRegression(max_iter=int(spec["linear"]["classification_logreg_max_iter"]), C=float(spec["linear"]["classification_logreg_C"]), random_state=seed).fit(X_tr, y_tr_contig)


def fit_and_score_canonical_auditor(
    auditor_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    task: str, n_classes: int = 2, seed: int = 42, compute_importance: bool = True,
) -> Dict[str, Any]:
    """Fit and score a canonical auditor without ever perturbing, flipping, or inventing labels."""
    X_tr = np.nan_to_num(np.asarray(X_train, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(np.asarray(X_test, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_tr_ro = np.nan_to_num(np.asarray(y_train, dtype=np.float64).reshape(-1), nan=0.0).copy()
    y_te_ro = np.nan_to_num(np.asarray(y_test, dtype=np.float64).reshape(-1), nan=0.0).copy()
    y_tr_ro.flags.writeable = False
    y_te_ro.flags.writeable = False
    sha_tr_pre, sha_te_pre = hashlib.sha256(y_tr_ro.tobytes()).hexdigest(), hashlib.sha256(y_te_ro.tobytes()).hexdigest()
    p_cols = X_tr.shape[1]
    unif_imp = np.full(p_cols, 1.0 / float(max(p_cols, 1)), dtype=np.float64)
    spec = CANONICAL_V3_AUDITOR_PROTOCOL_SPEC["auditors"]

    if task == "regression":
        if float(np.ptp(y_tr_ro)) < 1e-12:
            pred = np.full(len(y_te_ro), float(y_tr_ro[0]), dtype=np.float64)
            score = float(np.clip(r2_score(y_te_ro, pred), -1.0, 1.0)) if float(np.ptp(y_te_ro)) >= 1e-12 else (1.0 if abs(float(y_tr_ro[0]) - float(y_te_ro[0])) < 1e-9 else 0.0)
            imp, lbl_st = unif_imp, "constant_target_train_constant_predictor"
        else:
            if auditor_name == "catboost":
                m: Any = CatBoostRegressor(verbose=False, **{**spec["catboost"]["regression"], "random_seed": seed}).fit(X_tr, y_tr_ro)
            elif auditor_name == "lightgbm":
                m = LGBMRegressor(verbose=-1, **{**spec["lightgbm"]["regression"], "random_state": seed}).fit(X_tr, y_tr_ro)
            elif auditor_name == "xgboost":
                m = XGBRegressor(verbosity=0, **{**spec["xgboost"]["regression"], "random_state": seed}).fit(X_tr, y_tr_ro)
            elif auditor_name == "mlp":
                m = MLPRegressor(hidden_layer_sizes=tuple(spec["mlp"]["hidden_layer_sizes"]), max_iter=int(spec["mlp"]["max_iter"]), random_state=seed).fit(X_tr, y_tr_ro)
            elif auditor_name == "local_gated_attention":
                m = LocalGatedAttentionAuditor("regression", seed=seed).fit(X_tr, y_tr_ro, X_te, compute_importance=compute_importance)
            else:
                m = Ridge(alpha=float(spec["linear"]["regression_ridge_alpha"]), random_state=seed).fit(X_tr, y_tr_ro)
            pred = np.nan_to_num(np.asarray(m.predict(X_te), dtype=np.float64).reshape(-1), nan=0.0)
            score = float(np.clip(r2_score(y_te_ro, pred), -1.0, 1.0))
            imp = _extract_feature_importance(m, X_te, "regression") if compute_importance else np.zeros(p_cols)
            lbl_st = "authentic_continuous_target"
        assert hashlib.sha256(y_tr_ro.tobytes()).hexdigest() == sha_tr_pre and hashlib.sha256(y_te_ro.tobytes()).hexdigest() == sha_te_pre
        return {"score": score, "chance": 0.0, "floor": 0.05, "importance": imp, "train_label_status": lbl_st, "present_train_classes": 0, "missing_train_classes_count": 0, "label_sha256_verified": True}

    if task == "classification":
        y_tr_i, y_te_i = (y_tr_ro >= 0.5).astype(np.int64), (y_te_ro >= 0.5).astype(np.int64)
        y_tr_i.flags.writeable = False
        y_te_i.flags.writeable = False
        present = np.unique(y_tr_i)
        if len(present) < 2:
            p1 = np.full(len(y_te_i), float(present[0]) if len(present) == 1 else 0.0, dtype=np.float64)
            score = float(roc_auc_score(y_te_i, p1)) if len(np.unique(y_te_i)) >= 2 else 0.5
            imp, lbl_st = unif_imp, "single_class_train_constant_predictor"
        else:
            m = _fit_discrete_learner(auditor_name, X_tr, y_tr_i, X_te, 2, seed, compute_importance)
            prob = np.asarray(m.predict_proba(X_te), dtype=np.float64)
            p1 = np.clip(np.nan_to_num(prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else prob.reshape(-1), nan=0.5), 1e-6, 1.0 - 1e-6)
            score = float(roc_auc_score(y_te_i, p1)) if len(np.unique(y_te_i)) >= 2 else 0.5
            imp = _extract_feature_importance(m, X_te, "classification") if compute_importance else np.zeros(p_cols)
            lbl_st = "complete_binary_support"
        assert hashlib.sha256(y_tr_ro.tobytes()).hexdigest() == sha_tr_pre and hashlib.sha256(y_te_ro.tobytes()).hexdigest() == sha_te_pre
        return {"score": score, "chance": 0.5, "floor": 0.03, "importance": imp, "train_label_status": lbl_st, "present_train_classes": int(len(present)), "missing_train_classes_count": int(2 - len(present)), "label_sha256_verified": True}

    k_cls = max(int(n_classes), 3)
    y_tr_c = np.clip(np.rint(y_tr_ro).astype(np.int64), 0, k_cls - 1)
    y_te_c = np.clip(np.rint(y_te_ro).astype(np.int64), 0, k_cls - 1)
    y_tr_c.flags.writeable = False
    y_te_c.flags.writeable = False
    present = np.unique(y_tr_c)
    k_pres = int(len(present))
    if k_pres < 2:
        pred_c = np.full(len(y_te_c), int(present[0]) if k_pres == 1 else 0, dtype=np.int64)
        imp, lbl_st = unif_imp, "single_class_train_constant_predictor"
    else:
        y_tr_contig = np.searchsorted(present, y_tr_c).astype(np.int64)
        y_tr_contig.flags.writeable = False
        m = _fit_discrete_learner(auditor_name, X_tr, y_tr_contig, X_te, k_pres, seed, compute_importance)
        raw_pred = np.asarray(m.predict(X_te))
        pred_contig = (raw_pred >= 0.5).astype(np.int64) if (raw_pred.ndim == 1 and np.issubdtype(raw_pred.dtype, np.floating) and k_pres == 2) else np.clip(np.rint(raw_pred).astype(np.int64).reshape(-1), 0, k_pres - 1)
        pred_c = present[pred_contig]
        imp = _extract_feature_importance(m, X_te, "multiclass") if compute_importance else np.zeros(p_cols)
        lbl_st = "complete_multiclass_support" if k_pres == k_cls else "partial_class_support_invertible_bijection"
    score = 0.5 * (float(balanced_accuracy_score(y_te_c, pred_c)) + float(accuracy_score(y_te_c, pred_c)))
    assert hashlib.sha256(y_tr_ro.tobytes()).hexdigest() == sha_tr_pre and hashlib.sha256(y_te_ro.tobytes()).hexdigest() == sha_te_pre
    return {"score": score, "chance": 1.0 / float(k_cls), "floor": 0.03, "importance": imp, "train_label_status": lbl_st, "present_train_classes": k_pres, "missing_train_classes_count": int(k_cls - k_pres), "label_sha256_verified": True}


def pack_rows_complete_wire(
    X_train_s: np.ndarray, y_train_s: np.ndarray, scaler: FitOnlyTableTransform, level: str, seed: int = 20261024
) -> bytes:
    """Serialize a budget-capped complete wire packet (CSC1 sidecar + quantized row subset) for `rows_wire`."""
    cap, (n, p) = TIER_BYTE_CAPS[level], X_train_s.shape
    sc_bytes = pack_compact_scaler_sidecar(scaler, p)
    rng = np.random.default_rng(seed)
    lo, hi, best_wire = 2, max(min(n, max(int(0.88 * n) if level == "ultra" else n, 4)), 2), b""
    while lo <= hi:
        mid = (lo + hi) // 2
        idx = rng.choice(n, size=min(mid, n), replace=False)
        x_q = (np.rint(np.clip(X_train_s[idx], -6.0, 6.0) * 32.0)).astype(np.int8).tobytes()
        y_q = np.asarray(y_train_s[idx], dtype=np.float16).tobytes()
        pkt = b"RWW1" + zlib.compress(struct.pack("<HHH", len(idx), p, len(sc_bytes)) + sc_bytes + x_q + y_q, level=9)
        if len(pkt) <= cap or mid == 2:
            best_wire, lo = pkt, mid + 1
        else:
            hi = mid - 1
    return best_wire


def unpack_rows_complete_wire(wire_bytes: bytes, n_rows: int, seed: int = 20261024) -> Tuple[np.ndarray, np.ndarray, FitOnlyTableTransform]:
    if wire_bytes[:4] != b"RWW1":
        raise ValueError("invalid_rows_wire_magic")
    inner = zlib.decompress(wire_bytes[4:])
    k_rows, p, len_sc = struct.unpack_from("<HHH", inner, 0)
    off, x_len = 6 + len_sc, k_rows * p
    sc = unpack_compact_scaler_sidecar(inner[6:off])
    X_sub = np.frombuffer(inner[off : off + x_len], dtype=np.int8).astype(np.float64).reshape(k_rows, p) / 32.0
    y_sub = np.frombuffer(inner[off + x_len : off + x_len + k_rows * 2], dtype=np.float16).astype(np.float64)
    rep = np.random.default_rng(seed + 17).choice(k_rows, size=n_rows, replace=True)
    return X_sub[rep], y_sub[rep], sc


def pack_copula_complete_wire(
    X_train_s: np.ndarray, y_train_s: np.ndarray, scaler: FitOnlyTableTransform, level: str, task: str, n_classes: int = 2
) -> bytes:
    """Serialize a budget-capped complete wire packet (CSC1 sidecar + Gaussian/low-rank copula) for `copula_wire`."""
    cap, (n, p) = TIER_BYTE_CAPS[level], X_train_s.shape
    sc_bytes = pack_compact_scaler_sidecar(scaler, p)
    XY = np.column_stack([X_train_s, y_train_s.reshape(-1, 1)])
    mu = np.mean(XY, axis=0).astype(np.float16)
    cov = np.nan_to_num(np.cov(XY, rowvar=False), nan=0.0) + 1e-3 * np.eye(p + 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = np.maximum(eigvals[order], 1e-5), eigvecs[:, order]
    t_code, best_pkt = (0 if task == "regression" else (1 if task == "classification" else 2)), b""
    for r_rank in [min(p + 1, 24), min(p + 1, 12), min(p + 1, 6), min(p + 1, 3), 1]:
        L_r = (eigvecs[:, :r_rank] * np.sqrt(eigvals[:r_rank])[None, :]).astype(np.float16)
        diag_rem = np.sqrt(np.maximum(np.diag(cov) - np.sum(L_r.astype(np.float64) ** 2, axis=1), 1e-5)).astype(np.float16)
        y_q = np.quantile(y_train_s, np.linspace(0.0, 1.0, 17)).astype(np.float16)
        inner = struct.pack("<BBBBHHH", t_code, max(n_classes, 2), r_rank, 0, p, len(sc_bytes), 17) + sc_bytes + mu.tobytes() + L_r.tobytes() + diag_rem.tobytes() + y_q.tobytes()
        best_pkt = b"CPW1" + zlib.compress(inner, level=9)
        if len(best_pkt) <= cap:
            break
    return best_pkt


def unpack_copula_complete_wire(wire_bytes: bytes, n_rows: int, seed: int = 20261024) -> Tuple[np.ndarray, np.ndarray, FitOnlyTableTransform]:
    if wire_bytes[:4] != b"CPW1":
        raise ValueError("invalid_copula_wire_magic")
    inner = zlib.decompress(wire_bytes[4:])
    t_code, n_cls, r_rank, _, p, len_sc, n_yq = struct.unpack_from("<BBBBHHH", inner, 0)
    off, d = 10 + len_sc, p + 1
    sc = unpack_compact_scaler_sidecar(inner[10:off])
    mu = np.frombuffer(inner[off : off + d * 2], dtype=np.float16).astype(np.float64)
    off += d * 2
    L_r = np.frombuffer(inner[off : off + d * r_rank * 2], dtype=np.float16).astype(np.float64).reshape(d, r_rank)
    off += d * r_rank * 2
    diag_rem = np.frombuffer(inner[off : off + d * 2], dtype=np.float16).astype(np.float64)
    off += d * 2
    y_q = np.frombuffer(inner[off : off + n_yq * 2], dtype=np.float16).astype(np.float64)
    rng = np.random.default_rng(seed + 31)
    Z = rng.standard_normal((n_rows, r_rank)) @ L_r.T + rng.standard_normal((n_rows, d)) * diag_rem[None, :] + mu[None, :]
    u_ranks = (np.argsort(np.argsort(Z[:, p])) + 0.5) / float(max(n_rows, 1))
    y_mapped = np.interp(u_ranks, np.linspace(0.0, 1.0, len(y_q)), y_q)
    if t_code == 1:
        y_syn = (y_mapped >= 0.5).astype(np.float64)
    elif t_code == 2:
        k_c = max(int(n_cls), 3)
        y_syn = np.clip(np.rint(y_mapped), 0, k_c - 1).astype(np.float64)
    else:
        y_syn = y_mapped
    return Z[:, :p], y_syn, sc


def evaluate_canonical_oracle(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
    task: str, n_classes: int = 2, auditors: Sequence[str] = CANONICAL_6_AUDITORS,
) -> Dict[str, Any]:
    """Fit canonical auditors on real training data and evaluate on held-out test data."""
    out: Dict[str, Any] = {}
    for aud in auditors:
        try:
            res = fit_and_score_canonical_auditor(aud, X_tr, y_tr, X_te, y_te, task=task, n_classes=n_classes, seed=42, compute_importance=True)
            gain = float(res["score"] - res["chance"])
            out[aud] = {
                "status": "completed" if gain >= float(res["floor"]) else "low_oracle_gain",
                "real_score": round(float(res["score"]), 5), "chance": round(float(res["chance"]), 5),
                "floor": float(res["floor"]), "oracle_gain": round(gain, 5),
                "train_label_status": res["train_label_status"], "present_train_classes": res["present_train_classes"],
                "missing_train_classes_count": res["missing_train_classes_count"],
                "label_sha256_verified": res["label_sha256_verified"],
                "importance": res["importance"], "is_official_kumo_or_tabpfn": False,
            }
        except AUDITOR_ERRORS as exc:
            out[aud] = {"status": "failed_auditor", "error": type(exc).__name__, "is_official_kumo_or_tabpfn": False}
    return out


def score_synthetic_against_oracle(
    oracle: Dict[str, Any], X_syn: np.ndarray, y_syn: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
    task: str, n_classes: int = 2, auditors: Sequence[str] = CANONICAL_6_AUDITORS, compute_drivers: bool = True,
) -> Dict[str, Any]:
    """Train canonical auditors on synthetic rows and score retention + driver fidelity against real oracle."""
    per_aud: Dict[str, Any] = {}
    unclipped_all, clipped_all, d_all, rho_all, top3_all, unclipped_nf, clipped_nf = [], [], [], [], [], [], []
    floor_cnt, clip_hi_cnt, clip_lo_cnt, fail_cnt, partial_or_deg_cnt = 0, 0, 0, 0, 0
    n_expected = len(auditors)
    for aud in auditors:
        o_info = oracle.get(aud, {})
        if o_info.get("status") == "failed_auditor" or "real_score" not in o_info:
            per_aud[aud] = {"status": "failed_oracle", "error": o_info.get("error", "oracle_unavailable"), "is_official_kumo_or_tabpfn": False}
            fail_cnt += 1
            continue
        try:
            s_res = fit_and_score_canonical_auditor(aud, X_syn, y_syn, X_te, y_te, task=task, n_classes=n_classes, seed=42, compute_importance=compute_drivers)
            r_stats = _compute_retention_stats(float(o_info["real_score"]), float(s_res["score"]), float(o_info["chance"]), float(o_info["floor"]))
            drv = _compute_driver_fidelity(o_info["importance"], s_res["importance"]) if compute_drivers else {"spearman_rho": 0.0, "top3_overlap": 0.0, "driver_fidelity_D": 0.0}
            lbl_st = s_res["train_label_status"]
            if lbl_st not in ("authentic_continuous_target", "complete_binary_support", "complete_multiclass_support"):
                partial_or_deg_cnt += 1
            rec = {
                "status": "completed" if not r_stats["denom_floor_applied"] else "completed_with_denom_floor",
                "train_label_status": lbl_st, "present_train_classes": s_res["present_train_classes"],
                "missing_train_classes_count": s_res["missing_train_classes_count"],
                "label_sha256_verified": s_res["label_sha256_verified"],
                **r_stats, **drv, "is_official_kumo_or_tabpfn": False,
            }
            if aud == "local_gated_attention":
                rec["auditor_display_name"] = "LocalGatedAttentionNet"
                rec["raw_v2_alias_key"] = "kumo_tabpfn"
            per_aud[aud] = rec
            unclipped_all.append(r_stats["unclipped_R"])
            clipped_all.append(r_stats["clipped_R"])
            if compute_drivers:
                d_all.append(drv["driver_fidelity_D"])
                rho_all.append(drv["spearman_rho"])
                top3_all.append(drv["top3_overlap"])
            if r_stats["denom_floor_applied"]:
                floor_cnt += 1
            else:
                unclipped_nf.append(r_stats["unclipped_R"])
                clipped_nf.append(r_stats["clipped_R"])
            clip_hi_cnt += int(bool(r_stats["clip_high_applied"]))
            clip_lo_cnt += int(bool(r_stats["clip_low_applied"]))
        except AUDITOR_ERRORS as exc:
            per_aud[aud] = {"status": "failed_auditor", "error": type(exc).__name__, "is_official_kumo_or_tabpfn": False}
            fail_cnt += 1
    cc_clip = round(float(np.mean(clipped_all)), 5) if clipped_all else None
    cc_unclip = round(float(np.mean(unclipped_all)), 5) if unclipped_all else None
    return {
        "per_auditor": per_aud, "expected_auditors_denom": n_expected,
        "completed_auditors": len(clipped_all), "failed_auditors": fail_cnt,
        "missing_or_error_rate": round(float(fail_cnt) / float(max(n_expected, 1)), 5),
        "partial_or_degenerate_label_auditors": partial_or_deg_cnt,
        "complete_case_mean_unclipped_R": cc_unclip, "complete_case_mean_clipped_R": cc_clip,
        "sensitivity_zero_imputed_all_denom_mean_clipped_R": round(float(np.sum(clipped_all)) / float(max(n_expected, 1)), 5),
        "mean_unclipped_R": cc_unclip, "mean_clipped_R": cc_clip,
        "mean_nonfloor_unclipped_R": round(float(np.mean(unclipped_nf)), 5) if unclipped_nf else None,
        "mean_nonfloor_clipped_R": round(float(np.mean(clipped_nf)), 5) if clipped_nf else None,
        "mean_driver_fidelity_D": round(float(np.mean(d_all)), 5) if d_all else None,
        "mean_spearman_rho": round(float(np.mean(rho_all)), 5) if rho_all else None,
        "mean_top3_overlap": round(float(np.mean(top3_all)), 5) if top3_all else None,
        "denom_floor_count": floor_cnt, "clip_high_count": clip_hi_cnt, "clip_low_count": clip_lo_cnt,
    }


if __name__ == "__main__":
    from .v3_canonical_runner import main

    main()
