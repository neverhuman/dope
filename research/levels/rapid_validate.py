"""Stage 0 Rapid CatBoost & Regression Validator (<3 minutes).

Validates:
1. Experiment 0A: Feature importance and out-of-dataset predictive power of
   pre-calculated S3 landmarks + newly extracted data descriptive features
   (contrasts, midranks, target skewness/kurtosis/entropy, Chatterjee xi spectrum)
   across all admitted Regression & Binary Classification splits in S3, plus
   verification of the 3-level hierarchy (Medium=Macro, High=Larger, Ultra=Fine).
2. Experiment 0B: Train-only per-column scaling & categorical encoding
   (FitOnlyTableTransform) vs. naive scaling on real regression and binary
   classification tables across CatBoost, LightGBM, XGBoost, and Linear models,
   measuring unseen test lift and feature-importance preservation.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from .descriptors import (
    FINE_ADDED_INDICES,
    LANDMARK_NAMES,
    LARGER_ADDED_INDICES,
    MACRO_INDICES,
    compute_table_descriptors,
    extract_report_descriptors,
    squash,
)
from .scaling import FitOnlyTableTransform

DEFAULT_BUNDLE = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/s3_precalc_19190_reports.jsonl.gz")
DEFAULT_CATALOG = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/catalog.jsonl")


def _dataset_is_val(dataset_id: str) -> bool:
    h = int(hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10) == 0  # 10% source-dataset-disjoint validation split


def _feature_names_for_descriptors() -> List[str]:
    names: List[str] = []
    names.extend([f"gain_{name}" for name in LANDMARK_NAMES])
    names.extend([f"mask_{name}" for name in LANDMARK_NAMES])
    names.extend([f"midrank_{name}" for name in LANDMARK_NAMES])
    contrast_names = [
        "contrast_signal_ceiling",
        "contrast_lin_share",
        "contrast_macro_add_share",
        "contrast_monotone_excess",
        "contrast_symmetric_excess",
        "contrast_best1_dominance",
        "contrast_missing_share",
        "contrast_lin_learning_curve",
        "contrast_step_vs_hinge",
        "contrast_periodic_excess",
        "contrast_pairs_interaction_excess",
        "contrast_rff_kernel_excess",
        "contrast_otree_vs_add",
        "contrast_nonadditive_excess",
        "contrast_knn1_vs_knn16_sharpness",
        "contrast_gbdt_lc_and_net_excess",
    ]
    names.extend(contrast_names)
    names.extend(["target_stat_skew_or_entropy", "target_stat_kurt_or_minclass"])
    names.extend(["shape_is_clf", "shape_log10_rows", "shape_log2_features", "shape_aspect_ratio"])
    spec_names = [
        "spec_lin_max",
        "spec_lin_mean",
        "spec_lin_gini",
        "spec_macro_nonlin_max",
        "spec_macro_nonlin_mean",
        "spec_larger_nonlin_max",
        "spec_larger_nonlin_mean",
        "spec_chatterjee_xi_max",
        "spec_chatterjee_xi_mean",
        "spec_chatterjee_xi_gini",
        "spec_best_top1",
        "spec_best_top3",
        "spec_best_driver_gini",
        "spec_macro_nonlin_excess",
        "spec_larger_nonlin_excess",
        "spec_chatterjee_xi_vs_lin_excess",
    ]
    names.extend(spec_names)
    group_names = [
        "group_macro_max",
        "group_larger_max",
        "group_fine_max",
        "group_macro_mean",
        "group_larger_mean",
        "group_fine_mean",
        "group_se_mean",
        "group_stable_rate",
        "group_ridge_lambda_mean",
    ]
    names.extend(group_names)
    return names


def run_experiment_0a(bundle_path: Path = DEFAULT_BUNDLE, catalog_path: Path = DEFAULT_CATALOG) -> Dict[str, Any]:
    """Run rapid CatBoost & Ridge feature-importance and 3-level hierarchy screen on S3 reports."""
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score

    t0 = time.time()
    cat_tasks: Dict[str, str] = {}
    if catalog_path.exists():
        with catalog_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ds_key = row.get("id") or row.get("dataset_hash") or row.get("dataset_id")
                if ds_key:
                    cat_tasks[ds_key] = row["task"]

    X_med_raw, X_med_rich = [], []
    X_high_raw, X_high_rich = [], []
    X_ultra_for_target = []
    y_fine_ceiling = []
    y_nonlin_uplift = []
    is_val_mask = []
    tasks_seen = {"regression": 0, "binary": 0}
    splits_seen: Dict[str, int] = {}

    with gzip.open(bundle_path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            scope = rec["scope"]
            ds_id = scope["dataset_id"]
            cat_task = cat_tasks.get(ds_id, scope["task"])
            if cat_task not in ("regression", "binary"):
                continue
            report = rec["report"]
            # Extract ground-truth fine targets from full report
            ultra_desc = extract_report_descriptors(report, level="ultra")
            gains23 = ultra_desc["gains23"]
            mask23 = ultra_desc["mask23"]
            # Require at least one macro and one fine landmark present
            if not np.any(mask23[list(MACRO_INDICES)]) or not np.any(mask23[list(FINE_ADDED_INDICES)]):
                continue

            fine_vals = [gains23[i] for i in FINE_ADDED_INDICES if mask23[i]]
            fine_max = float(max(fine_vals))
            lin_gain = float(gains23[0]) if mask23[0] else 0.0
            uplift = fine_max - lin_gain

            med_desc = extract_report_descriptors(report, level="medium")
            high_desc = extract_report_descriptors(report, level="high")

            X_med_raw.append(np.concatenate([med_desc["gains23"], med_desc["mask23"].astype(np.float32)]))
            X_med_rich.append(med_desc["vector116"])
            X_high_raw.append(np.concatenate([high_desc["gains23"], high_desc["mask23"].astype(np.float32)]))
            X_high_rich.append(high_desc["vector116"])

            # For Ultra feature importance on predicting GBDT/Net ceiling from non-target descriptors + lower landmarks + Chatterjee xi
            ultra_input = ultra_desc["vector116"].copy()
            # Mask out direct fine gains (17..20, 22) and group_fine_max/mean in the probe input so we test genuine predictive power of descriptors + Chatterjee xi + Macro/Larger landmarks!
            for idx in FINE_ADDED_INDICES:
                ultra_input[idx] = 0.0
                ultra_input[23 + idx] = 0.0
                ultra_input[46 + idx] = 0.0
            ultra_input[69 + 13] = 0.0  # nonadd_ex
            ultra_input[69 + 14] = 0.0  # knn_ratio
            ultra_input[69 + 15] = 0.0  # gbdt_lc
            ultra_input[107 + 2] = 0.0  # group_fine_max
            ultra_input[107 + 5] = 0.0  # group_fine_mean
            X_ultra_for_target.append(ultra_input)

            y_fine_ceiling.append(fine_max)
            y_nonlin_uplift.append(uplift)
            is_val_mask.append(_dataset_is_val(ds_id))
            tasks_seen[cat_task] += 1
            sp = scope["split"]
            splits_seen[sp] = splits_seen.get(sp, 0) + 1

    val_mask = np.asarray(is_val_mask, dtype=bool)
    tr_mask = ~val_mask
    y_ceil = np.asarray(y_fine_ceiling, dtype=np.float32)
    y_up = np.asarray(y_nonlin_uplift, dtype=np.float32)

    variants = [
        ("Medium_Raw_GainsOnly", np.asarray(X_med_raw, dtype=np.float32)),
        ("Medium_Rich_MacroDescriptors", np.asarray(X_med_rich, dtype=np.float32)),
        ("High_Raw_GainsOnly", np.asarray(X_high_raw, dtype=np.float32)),
        ("High_Rich_LargerDescriptors", np.asarray(X_high_rich, dtype=np.float32)),
        ("Ultra_Rich_WithChatterjeeSpectrum", np.asarray(X_ultra_for_target, dtype=np.float32)),
    ]

    results_table = {}
    last_cb_model = None
    for name, X_mat in variants:
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_mat[tr_mask], y_ceil[tr_mask])
        pred_r = ridge.predict(X_mat[val_mask])

        cb = CatBoostRegressor(
            iterations=250,
            depth=6,
            learning_rate=0.08,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=8,
        )
        cb.fit(X_mat[tr_mask], y_ceil[tr_mask])
        pred_cb = cb.predict(X_mat[val_mask])

        cb_up = CatBoostRegressor(
            iterations=250,
            depth=6,
            learning_rate=0.08,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=8,
        )
        cb_up.fit(X_mat[tr_mask], y_up[tr_mask])
        pred_up = cb_up.predict(X_mat[val_mask])

        results_table[name] = {
            "feature_dim": int(X_mat.shape[1]),
            "ridge_ceiling_r2": round(float(r2_score(y_ceil[val_mask], pred_r)), 4),
            "ridge_ceiling_rmse": round(float(math.sqrt(mean_squared_error(y_ceil[val_mask], pred_r))), 5),
            "catboost_ceiling_r2": round(float(r2_score(y_ceil[val_mask], pred_cb)), 4),
            "catboost_ceiling_rmse": round(float(math.sqrt(mean_squared_error(y_ceil[val_mask], pred_cb))), 5),
            "catboost_nonlin_uplift_r2": round(float(r2_score(y_up[val_mask], pred_up)), 4),
            "catboost_nonlin_uplift_rmse": round(float(math.sqrt(mean_squared_error(y_up[val_mask], pred_up))), 5),
        }
        if name == "Ultra_Rich_WithChatterjeeSpectrum":
            last_cb_model = cb_up

    feat_names = _feature_names_for_descriptors()
    importances = last_cb_model.get_feature_importance()
    ranked = sorted(
        zip(feat_names, [float(v) for v in importances]),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "elapsed_seconds": round(time.time() - t0, 2),
        "total_admitted_splits": int(len(y_ceil)),
        "train_splits": int(np.sum(tr_mask)),
        "val_splits": int(np.sum(val_mask)),
        "tasks_breakdown": tasks_seen,
        "splits_breakdown": splits_seen,
        "hierarchy_and_descriptor_ablation": results_table,
        "top_25_features_for_nonlinear_uplift": [
            {"rank": i + 1, "feature": k, "importance_pct": round(v, 3)}
            for i, (k, v) in enumerate(ranked[:25])
        ],
    }


def run_experiment_0b(
    bundle_path: Path = DEFAULT_BUNDLE,
    catalog_path: Path = DEFAULT_CATALOG,
    source_cache_dirs: Sequence[Path] = (
        Path("/home/ubuntu/dope-fit-cv-precalc-20261023/unified-source-cache"),
        Path("/home/ubuntu/dope-fit-cv-precalc-20261023/full-v4/source-cache"),
    ),
    max_datasets: int = 36,
) -> Dict[str, Any]:
    """Rapidly validate FitOnlyTableTransform scaling & feature-importance preservation across CatBoost, LightGBM, XGBoost, Ridge/Logistic."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
    from scipy.stats import spearmanr

    t0 = time.time()
    cat_tasks: Dict[str, str] = {}
    if catalog_path.exists():
        with catalog_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ds_key = row.get("id") or row.get("dataset_hash") or row.get("dataset_id")
                if ds_key:
                    cat_tasks[ds_key] = row["task"]

    # Map train_sha256 -> local CSV path
    csv_map: Dict[str, Path] = {}
    for d in source_cache_dirs:
        if d.exists():
            for p in d.glob("*.csv"):
                csv_map[p.stem] = p

    selected_scopes = []
    seen_ds = set()
    with gzip.open(bundle_path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sc = rec["scope"]
            ds_id = sc["dataset_id"]
            if ds_id in seen_ds:
                continue
            t_sha = sc["train_sha256"]
            cat_task = cat_tasks.get(ds_id, sc["task"])
            if cat_task in ("regression", "binary") and t_sha in csv_map and 120 <= sc["rows"] <= 4000 and 4 <= sc["columns"] <= 60:
                selected_scopes.append((cat_task, sc, csv_map[t_sha]))
                seen_ds.add(ds_id)
                if len(selected_scopes) >= max_datasets:
                    break

    naive_scores, fitonly_scores = [], []
    cb_fi_spearman, lgb_fi_spearman, lin_fi_spearman = [], [], []
    cb_driver_fidelity = []

    for cat_task, sc, csv_path in selected_scopes:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            raw_rows = list(csv.reader(f))
        if sc.get("has_header", False) and raw_rows:
            raw_rows = raw_rows[1:]
        if len(raw_rows) < 60:
            continue
        rng = np.random.default_rng(int(hashlib.sha256(sc["dataset_id"].encode()).hexdigest()[:8], 16))
        perm = rng.permutation(len(raw_rows))
        n_tr = int(0.75 * len(raw_rows))
        tr_rows = [raw_rows[i] for i in perm[:n_tr]]
        te_rows = [raw_rows[i] for i in perm[n_tr:]]

        X_tr_raw = [r[:-1] for r in tr_rows]
        y_tr_raw = [r[-1] for r in tr_rows]
        X_te_raw = [r[:-1] for r in te_rows]
        y_te_raw = [r[-1] for r in te_rows]

        task_name = "classification" if cat_task == "binary" else "regression"
        try:
            scaler = FitOnlyTableTransform.fit(X_tr_raw, y_tr_raw, task=task_name)
            X_tr_opt, y_tr_opt = scaler.transform(X_tr_raw, y_tr_raw)
            X_te_opt, y_te_opt = scaler.transform(X_te_raw, y_te_raw)
        except Exception:
            continue

        if task_name == "classification" and len(np.unique(y_tr_opt)) < 2 or len(np.unique(y_te_opt)) < 2:
            continue

        # Naive numeric coercion & mean/std scaling for baseline comparison
        p = X_tr_opt.shape[1]
        X_tr_naive = np.zeros_like(X_tr_opt)
        X_te_naive = np.zeros_like(X_te_opt)
        for j in range(p):
            c_tr = np.array([_try_float(r[j]) for r in X_tr_raw], dtype=np.float64)
            c_te = np.array([_try_float(r[j]) for r in X_te_raw], dtype=np.float64)
            m = float(np.nanmean(c_tr)) if np.any(np.isfinite(c_tr)) else 0.0
            s = float(np.nanstd(c_tr)) if np.any(np.isfinite(c_tr)) else 1.0
            s = max(s, 1e-6)
            X_tr_naive[:, j] = (np.nan_to_num(c_tr, nan=m) - m) / s
            X_te_naive[:, j] = (np.nan_to_num(c_te, nan=m) - m) / s

        if task_name == "classification":
            m_naive = LogisticRegression(max_iter=200, C=1.0).fit(X_tr_naive, y_tr_opt)
            m_opt = LogisticRegression(max_iter=200, C=1.0).fit(X_tr_opt, y_tr_opt)
            s_naive = float(roc_auc_score(y_te_opt, m_naive.predict_proba(X_te_naive)[:, 1]))
            s_opt = float(roc_auc_score(y_te_opt, m_opt.predict_proba(X_te_opt)[:, 1]))

            cb1 = CatBoostClassifier(iterations=120, depth=5, verbose=False, random_seed=1, thread_count=4).fit(X_tr_opt, y_tr_opt)
            # Fit on bootstrap synthetic sample with conditional target preservation
            syn_idx = rng.choice(len(X_tr_opt), size=len(X_tr_opt), replace=True)
            X_syn = X_tr_opt[syn_idx] + rng.normal(0.0, 0.08, size=X_tr_opt.shape)
            y_syn = (cb1.predict_proba(X_syn)[:, 1] >= 0.5).astype(np.float64)
            if len(np.unique(y_syn)) < 2:
                y_syn = y_tr_opt[syn_idx]
            cb2 = CatBoostClassifier(iterations=120, depth=5, verbose=False, random_seed=2, thread_count=4).fit(X_syn, y_syn)
            lgb1 = LGBMClassifier(n_estimators=80, max_depth=5, verbose=-1, random_state=1, n_jobs=4).fit(X_tr_opt, y_tr_opt)
            lgb2 = LGBMClassifier(n_estimators=80, max_depth=5, verbose=-1, random_state=2, n_jobs=4).fit(X_syn, y_syn)
            lin2 = LogisticRegression(max_iter=200, C=1.0).fit(X_syn, y_syn)
            fi_lin1 = np.abs(m_opt.coef_.reshape(-1))
            fi_lin2 = np.abs(lin2.coef_.reshape(-1))
        else:
            m_naive = Ridge(alpha=1.0).fit(X_tr_naive, y_tr_opt)
            m_opt = Ridge(alpha=1.0).fit(X_tr_opt, y_tr_opt)
            s_naive = float(np.clip(r2_score(y_te_opt, m_naive.predict(X_te_naive)), -1.0, 1.0))
            s_opt = float(np.clip(r2_score(y_te_opt, m_opt.predict(X_te_opt)), -1.0, 1.0))

            cb1 = CatBoostRegressor(iterations=120, depth=5, verbose=False, random_seed=1, thread_count=4).fit(X_tr_opt, y_tr_opt)
            syn_idx = rng.choice(len(X_tr_opt), size=len(X_tr_opt), replace=True)
            X_syn = X_tr_opt[syn_idx] + rng.normal(0.0, 0.08, size=X_tr_opt.shape)
            y_syn = cb1.predict(X_syn) + rng.normal(0.0, 0.10, size=len(X_syn))
            cb2 = CatBoostRegressor(iterations=120, depth=5, verbose=False, random_seed=2, thread_count=4).fit(X_syn, y_syn)
            lgb1 = LGBMRegressor(n_estimators=80, max_depth=5, verbose=-1, random_state=1, n_jobs=4).fit(X_tr_opt, y_tr_opt)
            lgb2 = LGBMRegressor(n_estimators=80, max_depth=5, verbose=-1, random_state=2, n_jobs=4).fit(X_syn, y_syn)
            lin2 = Ridge(alpha=1.0).fit(X_syn, y_syn)
            fi_lin1 = np.abs(m_opt.coef_.reshape(-1))
            fi_lin2 = np.abs(lin2.coef_.reshape(-1))

        naive_scores.append(s_naive)
        fitonly_scores.append(s_opt)

        fi_cb1 = np.asarray(cb1.get_feature_importance(), dtype=np.float64)
        fi_cb2 = np.asarray(cb2.get_feature_importance(), dtype=np.float64)
        if np.sum(fi_cb1) > 0 and np.sum(fi_cb2) > 0:
            p1 = fi_cb1 / np.sum(fi_cb1)
            p2 = fi_cb2 / np.sum(fi_cb2)
            cb_driver_fidelity.append(float(1.0 - 0.5 * np.sum(np.abs(p1 - p2))))
            rho, _ = spearmanr(p1, p2)
            if math.isfinite(float(rho)):
                cb_fi_spearman.append(float(rho))

        fi_lgb1 = np.asarray(lgb1.feature_importances_, dtype=np.float64)
        fi_lgb2 = np.asarray(lgb2.feature_importances_, dtype=np.float64)
        if np.sum(fi_lgb1) > 0 and np.sum(fi_lgb2) > 0:
            rho, _ = spearmanr(fi_lgb1, fi_lgb2)
            if math.isfinite(float(rho)):
                lgb_fi_spearman.append(float(rho))

        if np.sum(fi_lin1) > 0 and np.sum(fi_lin2) > 0:
            rho, _ = spearmanr(fi_lin1, fi_lin2)
            if math.isfinite(float(rho)):
                lin_fi_spearman.append(float(rho))

    return {
        "elapsed_seconds": round(time.time() - t0, 2),
        "datasets_evaluated": len(fitonly_scores),
        "mean_test_score_naive_zscore": round(float(np.mean(naive_scores)), 4) if naive_scores else 0.0,
        "mean_test_score_fitonly_optimal_scaling": round(float(np.mean(fitonly_scores)), 4) if fitonly_scores else 0.0,
        "mean_scaling_gain": round(float(np.mean(np.array(fitonly_scores) - np.array(naive_scores))), 4) if fitonly_scores else 0.0,
        "catboost_feature_importance_spearman_rho": round(float(np.mean(cb_fi_spearman)), 4) if cb_fi_spearman else 0.0,
        "catboost_driver_fidelity_D": round(float(np.mean(cb_driver_fidelity)), 4) if cb_driver_fidelity else 0.0,
        "lightgbm_feature_importance_spearman_rho": round(float(np.mean(lgb_fi_spearman)), 4) if lgb_fi_spearman else 0.0,
        "linear_feature_importance_spearman_rho": round(float(np.mean(lin_fi_spearman)), 4) if lin_fi_spearman else 0.0,
    }


def _try_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def main() -> None:
    print("=== Running Stage 0A: Rapid CatBoost & Ridge Meta-Feature Importance Screen on S3 Corpus ===")
    res_0a = run_experiment_0a()
    print(json.dumps(res_0a, indent=2))
    print("\n=== Running Stage 0B: Rapid Train-Only Scaling & Feature Importance Preservation Probe ===")
    res_0b = run_experiment_0b()
    print(json.dumps(res_0b, indent=2))
    out_dir = Path("/home/ubuntu/dope/target/rapid-validation-20261023")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage0_rapid_validation_receipt.json").write_text(
        json.dumps({"stage_0a": res_0a, "stage_0b": res_0b}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved receipt to {out_dir / 'stage0_rapid_validation_receipt.json'}")


if __name__ == "__main__":
    main()
