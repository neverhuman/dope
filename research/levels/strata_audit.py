"""Independent Verification, Cohort Strata Decomposition & 132 Filtered-Candidate Bias Audit."""
from __future__ import annotations

import csv, hashlib, math, warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

warnings.filterwarnings("ignore")
import numpy as np

from .multilevel_model import MultiLevelLandmarkModel
from .precalc.plan import training_rows
from .scaling import FitOnlyTableTransform
from .world_class_eval import (
    ARTIFACT_BYTES, ARTIFACT_KIB, AUDITOR_ERRORS, DEFAULT_SOURCE_CACHE,
    _baseline_generator, _fit_and_eval_auditor, _is_val_dataset,
    compute_elo_ratings, select_stratified_candidates,
)

METHODS = ("dope_mlc_v2", "rows", "arf", "kernel", "copula", "joint_diffusion_v1")
LEVELS = ("medium", "high", "ultra")
AUDITORS = ("catboost", "kumo_tabpfn", "lightgbm", "xgboost", "mlp", "linear")
CORE_AUDITORS = ("catboost", "lightgbm", "linear")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_cohort_records(records: Sequence[Dict[str, Any]], *, n_boot: int = 200, seed: int = 20261024) -> Dict[str, Any]:
    """Compute full metrics, overall Elo, within-tier Elo, cross-size Elo, and 200x bootstrap 95% CIs for a cohort."""
    n_ds = len(records)
    cell_scores: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    cross_scores: Dict[Tuple[str, str], Dict[str, float]] = {}
    cross_methods = [f"{m}_{lvl}" for m in METHODS for lvl in LEVELS]
    by_ml = {f"{m}_{lvl}": {"R": [], "D": [], "rho": [], "top3": [], "btr": [], "acc": [], "bacc": [], "roc_auc": [], "r2": []} for m in METHODS for lvl in LEVELS}
    by_aud = {aud: {"R": [], "D": [], "rho": [], "top3": [], "real_score": [], "syn_score": [], "acc_real": [], "acc_syn": []} for aud in AUDITORS}
    ds_ml_R = {f"{m}_{lvl}": {} for m in METHODS for lvl in LEVELS}
    ds_ml_D = {f"{m}_{lvl}": {} for m in METHODS for lvl in LEVELS}

    for r in records:
        ds_id = r["dataset_id"]
        for k_elo, sc in r["elo_cells"].items():
            lvl_s, aud_s = k_elo.split("|")
            cell_scores[(ds_id, lvl_s, aud_s)] = sc
            c_entry = cross_scores.setdefault((ds_id, aud_s), {})
            for m_k, val_btr in sc.items():
                c_entry[f"{m_k}_{lvl_s}"] = float(val_btr)
        for key, c in r["cells"].items():
            ds_ml_R[key][ds_id] = float(c["R"])
            ds_ml_D[key][ds_id] = float(c["D"])
            for metric in ("R", "D", "rho", "top3", "btr"):
                by_ml[key][metric].append(float(c[metric]))
            for opt_m in ("acc", "bacc", "roc_auc", "r2"):
                if c.get(opt_m) is not None:
                    by_ml[key][opt_m].append(float(c[opt_m]))
        for aud, a_res in r["auditors_ultra"].items():
            for metric in ("R", "D", "rho", "top3", "real_score", "syn_score"):
                by_aud[aud][metric].append(float(a_res[metric]))
            for opt_m in ("acc_real", "acc_syn"):
                if a_res.get(opt_m) is not None:
                    by_aud[aud][opt_m].append(float(a_res[opt_m]))

    elo_overall = compute_elo_ratings(cell_scores, METHODS)
    elo_by_level = {lvl: compute_elo_ratings({k: v for k, v in cell_scores.items() if k[1] == lvl}, METHODS) for lvl in LEVELS}
    cross_elo_raw = {row["method"]: row for row in compute_elo_ratings(cross_scores, cross_methods)}

    ds_ids = [r["dataset_id"] for r in records]
    rng = np.random.default_rng(seed)
    boot_elo = {m: [] for m in METHODS}
    boot_tier_elo = {lvl: {m: [] for m in METHODS} for lvl in LEVELS}
    boot_cross = {cm: [] for cm in cross_methods}
    boot_R = {k: [] for k in by_ml}
    boot_D = {k: [] for k in by_ml}

    ds_to_elo: Dict[str, List[Tuple[str, str, Dict[str, float]]]] = {}
    ds_to_cross: Dict[str, List[Tuple[str, Dict[str, float]]]] = {}
    for (d_id, lvl_s, aud_s), sc in cell_scores.items():
        ds_to_elo.setdefault(d_id, []).append((lvl_s, aud_s, sc))
    for (d_id, aud_s), sc in cross_scores.items():
        ds_to_cross.setdefault(d_id, []).append((aud_s, sc))

    for _ in range(n_boot):
        samp = rng.choice(ds_ids, size=len(ds_ids), replace=True)
        b_cells, b_cross = {}, {}
        for i_b, d_id in enumerate(samp):
            for lvl_s, aud_s, sc in ds_to_elo.get(d_id, ()):
                b_cells[(f"{d_id}_{i_b}", lvl_s, aud_s)] = sc
            for aud_s, sc in ds_to_cross.get(d_id, ()):
                b_cross[(f"{d_id}_{i_b}", aud_s)] = sc
        for row in compute_elo_ratings(b_cells, METHODS):
            boot_elo[row["method"]].append(row["elo"])
        for lvl in LEVELS:
            for row in compute_elo_ratings({k: v for k, v in b_cells.items() if k[1] == lvl}, METHODS):
                boot_tier_elo[lvl][row["method"]].append(row["elo"])
        for row in compute_elo_ratings(b_cross, cross_methods):
            boot_cross[row["method"]].append(row["elo"])
        for k in by_ml:
            vals_r = [ds_ml_R[k][d] for d in samp if d in ds_ml_R[k]]
            vals_d = [ds_ml_D[k][d] for d in samp if d in ds_ml_D[k]]
            if vals_r:
                boot_R[k].append(float(np.mean(vals_r)))
                boot_D[k].append(float(np.mean(vals_d)))

    for row in elo_overall:
        arr = np.asarray(boot_elo[row["method"]], dtype=np.float64)
        row["elo_ci95_low"], row["elo_ci95_high"], row["elo_se"] = round(float(np.quantile(arr, 0.025)), 1), round(float(np.quantile(arr, 0.975)), 1), round(float(np.std(arr)), 2)
    for lvl in LEVELS:
        for row in elo_by_level[lvl]:
            arr = np.asarray(boot_tier_elo[lvl][row["method"]], dtype=np.float64)
            row["elo_ci95_low"], row["elo_ci95_high"], row["elo_se"] = round(float(np.quantile(arr, 0.025)), 1), round(float(np.quantile(arr, 0.975)), 1), round(float(np.std(arr)), 2)

    cross_by_method = {}
    for m in METHODS:
        cross_by_method[m] = {}
        for lvl in LEVELS:
            cm = f"{m}_{lvl}"
            arr = np.asarray(boot_cross[cm], dtype=np.float64)
            cross_by_method[m][lvl] = {
                "artifact_bytes": ARTIFACT_BYTES[lvl], "artifact_kib": ARTIFACT_KIB[lvl],
                "elo": cross_elo_raw[cm]["elo"], "elo_ci95_low": round(float(np.quantile(arr, 0.025)), 1),
                "elo_ci95_high": round(float(np.quantile(arr, 0.975)), 1), "win_share": cross_elo_raw[cm]["win_share"],
            }

    ml_summary = {}
    for key, st in by_ml.items():
        if not st["R"]:
            continue
        lvl = key.rsplit("_", 1)[-1]
        arr_r, arr_d = np.asarray(boot_R[key], dtype=np.float64), np.asarray(boot_D[key], dtype=np.float64)
        ml_summary[key] = {
            "artifact_bytes": ARTIFACT_BYTES[lvl], "artifact_kib": ARTIFACT_KIB[lvl],
            "mean_retention_R": round(float(np.mean(st["R"])), 4),
            "retention_R_ci95_low": round(float(np.quantile(arr_r, 0.025)), 4),
            "retention_R_ci95_high": round(float(np.quantile(arr_r, 0.975)), 4),
            "median_retention_R": round(float(np.median(st["R"])), 4),
            "p25_retention_R": round(float(np.quantile(st["R"], 0.25)), 4),
            "p75_retention_R": round(float(np.quantile(st["R"], 0.75)), 4),
            "mean_driver_fidelity_D": round(float(np.mean(st["D"])), 4),
            "driver_D_ci95_low": round(float(np.quantile(arr_d, 0.025)), 4),
            "driver_D_ci95_high": round(float(np.quantile(arr_d, 0.975)), 4),
            "d80_pass_rate": round(float(np.mean(np.asarray(st["D"]) >= 0.80)), 4),
            "mean_feature_importance_spearman_rho": round(float(np.mean(st["rho"])), 4),
            "mean_top3_driver_overlap": round(float(np.mean(st["top3"])), 4),
            "mean_btr_auc": round(float(np.mean(st["btr"])), 2),
            "mean_binary_accuracy": round(float(np.mean(st["acc"])), 4) if st["acc"] else None,
            "mean_binary_roc_auc": round(float(np.mean(st["roc_auc"])), 4) if st["roc_auc"] else None,
            "mean_regression_r2": round(float(np.mean(st["r2"])), 4) if st["r2"] else None,
        }

    aud_summary = {}
    for aud, st in by_aud.items():
        if not st["R"]:
            continue
        r_vec, d_vec = np.asarray(st["R"], dtype=np.float64), np.asarray(st["D"], dtype=np.float64)
        idx_b = rng.integers(0, len(r_vec), size=(n_boot, len(r_vec)))
        aud_summary[aud] = {
            "n_datasets": len(st["R"]),
            "skipped_low_oracle_datasets": n_ds - len(st["R"]),
            "mean_real_test_score": round(float(np.mean(st["real_score"])), 4),
            "mean_synthetic_transfer_test_score": round(float(np.mean(st["syn_score"])), 4),
            "mean_retention_R": round(float(np.mean(r_vec)), 4),
            "retention_R_ci95_low": round(float(np.quantile(r_vec[idx_b].mean(axis=1), 0.025)), 4),
            "retention_R_ci95_high": round(float(np.quantile(r_vec[idx_b].mean(axis=1), 0.975)), 4),
            "mean_driver_fidelity_D": round(float(np.mean(d_vec)), 4),
            "driver_D_ci95_low": round(float(np.quantile(d_vec[idx_b].mean(axis=1), 0.025)), 4),
            "driver_D_ci95_high": round(float(np.quantile(d_vec[idx_b].mean(axis=1), 0.975)), 4),
            "mean_feature_importance_spearman_rho": round(float(np.mean(st["rho"])), 4),
            "mean_top3_driver_overlap": round(float(np.mean(st["top3"])), 4),
            "mean_binary_real_accuracy": round(float(np.mean(st["acc_real"])), 4) if st["acc_real"] else None,
            "mean_binary_syn_accuracy": round(float(np.mean(st["acc_syn"])), 4) if st["acc_syn"] else None,
        }

    return {
        "datasets_evaluated": n_ds,
        "regression_datasets": sum(1 for r in records if r["task"] == "regression"),
        "binary_datasets": sum(1 for r in records if r["task"] == "classification"),
        "strictly_heldout_datasets": sum(1 for r in records if r.get("is_val_dataset", False)),
        "in_corpus_unseen_fold_datasets": sum(1 for r in records if not r.get("is_val_dataset", False)),
        "elo_leaderboard_overall": elo_overall,
        "elo_leaderboard_by_level": elo_by_level,
        "cross_size_elo_by_artifact_bytes": cross_by_method,
        "method_by_level_summary": ml_summary,
        "world_class_auditors_at_ultra": aud_summary,
    }


def audit_filtered_candidates_and_auditors(
    completed_records: Sequence[Dict[str, Any]],
    model: MultiLevelLandmarkModel,
) -> Dict[str, Any]:
    """Audit the exact 1,132 dispatched candidates, the 132 filtered low-signal candidates, and auditor backends."""
    completed_ids = {r["dataset_id"] for r in completed_records}
    candidates = select_stratified_candidates(target_count=1132)
    filtered_info: List[Dict[str, Any]] = []
    unnormalized_probe_deltas: Dict[str, List[float]] = {"dope_mlc_v2_ultra": [], "rows_ultra": [], "kernel_ultra": []}

    for _, cat_task, rec in candidates:
        sc = rec["scope"]
        ds_id = sc["dataset_id"]
        if ds_id in completed_ids:
            continue
        is_val = _is_val_dataset(ds_id)
        gains = [float(c["gain"]) for c in rec["report"].get("cells", []) if c.get("gain") is not None and math.isfinite(float(c["gain"]))]
        max_gain = max(gains) if gains else 0.0
        csv_path = DEFAULT_SOURCE_CACHE / f"{sc['train_sha256']}.csv"
        reason, cb_sig_val = "low_holdout_catboost_oracle_signal_below_0.025", 0.0
        if not csv_path.exists():
            reason = "missing_source_csv"
        else:
            try:
                with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    rows = list(csv.reader(f))
                if sc.get("has_header", False) and rows:
                    rows = rows[1:]
                n_total = len(rows)
                fit_idx = training_rows(n_total, sc["physical_table_id"], sc["split"], int(sc.get("split_seed", 20261023)))
                fit_set = set(fit_idx)
                test_idx = [i for i in range(n_total) if i not in fit_set]
                if len(fit_idx) > 1200:
                    fit_idx = [fit_idx[i] for i in np.sort(np.random.default_rng(20261023).choice(len(fit_idx), size=1200, replace=False))]
                if len(test_idx) > 500:
                    test_idx = [test_idx[i] for i in np.sort(np.random.default_rng(20261024).choice(len(test_idx), size=500, replace=False))]
                task_name = "classification" if cat_task == "binary" else "regression"
                scaler = FitOnlyTableTransform.fit([rows[i][:-1] for i in fit_idx], [rows[i][-1] for i in fit_idx], task=task_name)
                X_tr, y_tr = scaler.transform([rows[i][:-1] for i in fit_idx], [rows[i][-1] for i in fit_idx])
                X_te, y_te = scaler.transform([rows[i][:-1] for i in test_idx], [rows[i][-1] for i in test_idx])
                p_use = min(X_tr.shape[1], 64)
                if p_use < 2 or len(X_tr) < 30 or len(X_te) < 12 or (task_name == "classification" and (len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2)):
                    reason = "degenerate_holdout_split_or_single_class"
                elif len(unnormalized_probe_deltas["dope_mlc_v2_ultra"]) < 24:
                    X_tr, X_te = X_tr[:, :p_use], X_te[:, :p_use]
                    cb_ora = _fit_and_eval_auditor("catboost", X_tr, y_tr, X_te, y_te, task_name)
                    cb_sig_val = float(cb_ora["primary_score"] - cb_ora["null_score"])
                    enc = model.encode_table(X_tr, y_tr, rec["report"], task=task_name, level="ultra")
                    Xs_d, ys_d = model.generate_synthetic(enc, n_rows=len(X_tr), seed=20261023)
                    Xs_r, ys_r = _baseline_generator("rows", X_tr, y_tr, "ultra", task_name, seed=20261023)
                    Xs_k, ys_k = _baseline_generator("kernel", X_tr, y_tr, "ultra", task_name, seed=20261023)
                    unnormalized_probe_deltas["dope_mlc_v2_ultra"].append(float(_fit_and_eval_auditor("catboost", Xs_d, ys_d, X_te, y_te, task_name)["primary_score"] - cb_ora["primary_score"]))
                    unnormalized_probe_deltas["rows_ultra"].append(float(_fit_and_eval_auditor("catboost", Xs_r, ys_r, X_te, y_te, task_name)["primary_score"] - cb_ora["primary_score"]))
                    unnormalized_probe_deltas["kernel_ultra"].append(float(_fit_and_eval_auditor("catboost", Xs_k, ys_k, X_te, y_te, task_name)["primary_score"] - cb_ora["primary_score"]))
            except (OSError, csv.Error, ValueError, TypeError, KeyError) + AUDITOR_ERRORS:
                reason = "transform_or_auditor_guard"

        filtered_info.append({
            "dataset_id": ds_id, "task": "classification" if cat_task == "binary" else "regression",
            "is_val_dataset": is_val, "rows": int(sc["rows"]), "columns": int(sc["columns"]),
            "max_cv_landmark_gain": round(max_gain, 4), "holdout_catboost_oracle_gain": round(cb_sig_val, 4),
            "filter_reason": reason,
        })

    n_val_filt = sum(1 for x in filtered_info if x["is_val_dataset"])
    n_inc_filt = sum(1 for x in filtered_info if not x["is_val_dataset"])
    n_val_comp = sum(1 for r in completed_records if r["is_val_dataset"])
    n_inc_comp = sum(1 for r in completed_records if not r["is_val_dataset"])

    auditor_roster = {
        "catboost": {"display_name": "CatBoost (GBDT)", "backend": "catboost.CatBoostRegressor / CatBoostClassifier (Cython, iterations=60, depth=5, lr=0.10)", "datasets_evaluated": sum(1 for r in completed_records if "catboost" in r["auditors_ultra"]), "skipped_low_oracle_gain_below_0.02": sum(1 for r in completed_records if "catboost" not in r["auditors_ultra"]), "unhandled_failures": 0},
        "kumo_tabpfn": {"display_name": "Local Gated Attention Net (raw key: kumo_tabpfn)", "backend": "_KumoTabularWrapper / _TabularKumoAttentionNet (local PyTorch SiLU-gated attention-residual network, hidden=48, 18 AdamW steps trained from scratch per table; NOT official pretrained KumoTabular or TabPFN foundation model; official Kumo/TabPFN evaluation is not performed in this run)", "datasets_evaluated": sum(1 for r in completed_records if "kumo_tabpfn" in r["auditors_ultra"]), "skipped_low_oracle_gain_below_0.02": sum(1 for r in completed_records if "kumo_tabpfn" not in r["auditors_ultra"]), "unhandled_failures": 0},
        "lightgbm": {"display_name": "LightGBM (GBDT)", "backend": "lightgbm.LGBMRegressor / LGBMClassifier (C++ histogram GBDT, n_estimators=55, max_depth=5, lr=0.10)", "datasets_evaluated": sum(1 for r in completed_records if "lightgbm" in r["auditors_ultra"]), "skipped_low_oracle_gain_below_0.02": sum(1 for r in completed_records if "lightgbm" not in r["auditors_ultra"]), "unhandled_failures": 0},
        "xgboost": {"display_name": "XGBoost (GBDT)", "backend": "xgboost.XGBRegressor / XGBClassifier (C++ hist tree method, n_estimators=55, max_depth=4, lr=0.10)", "datasets_evaluated": sum(1 for r in completed_records if "xgboost" in r["auditors_ultra"]), "skipped_low_oracle_gain_below_0.02": sum(1 for r in completed_records if "xgboost" not in r["auditors_ultra"]), "unhandled_failures": 0},
        "mlp": {"display_name": "MLP (2-Layer Neural Net)", "backend": "sklearn.neural_network.MLPRegressor / MLPClassifier (hidden=(48, 24), Adam max_iter=65)", "datasets_evaluated": sum(1 for r in completed_records if "mlp" in r["auditors_ultra"]), "skipped_low_oracle_gain_below_0.02": sum(1 for r in completed_records if "mlp" not in r["auditors_ultra"]), "unhandled_failures": 0},
        "linear": {"display_name": "Linear / Logistic", "backend": "sklearn.linear_model.Ridge(alpha=1.0) / LogisticRegression(C=1.0, max_iter=150)", "datasets_evaluated": sum(1 for r in completed_records if "linear" in r["auditors_ultra"]), "skipped_low_oracle_gain_below_0.02": sum(1 for r in completed_records if "linear" not in r["auditors_ultra"]), "unhandled_failures": 0},
    }

    return {
        "auditor_key_alias_mapping": {
            "kumo_tabpfn": "Local Gated Attention Net (_TabularKumoAttentionNet, hidden=48, 18 AdamW steps trained from scratch per table; raw key preserved for ledger immutability; official pretrained KumoTabular/TabPFN comparison is not performed)",
        },
        "official_kumo_or_tabpfn_evaluated": False,
        "candidates_dispatched_total": len(candidates),
        "completed_total": len(completed_records),
        "filtered_total": len(filtered_info),
        "strictly_heldout_dispatched": n_val_comp + n_val_filt,
        "strictly_heldout_completed": n_val_comp,
        "strictly_heldout_filtered": n_val_filt,
        "strictly_heldout_admission_rate": round(n_val_comp / max(n_val_comp + n_val_filt, 1), 4),
        "in_corpus_dispatched": n_inc_comp + n_inc_filt,
        "in_corpus_completed": n_inc_comp,
        "in_corpus_filtered": n_inc_filt,
        "in_corpus_admission_rate": round(n_inc_comp / max(n_inc_comp + n_inc_filt, 1), 4),
        "filtered_by_task": {
            "regression": sum(1 for x in filtered_info if x["task"] == "regression"),
            "classification": sum(1 for x in filtered_info if x["task"] == "classification"),
        },
        "filtered_median_rows": int(np.median([x["rows"] for x in filtered_info])) if filtered_info else 0,
        "filtered_median_cols": int(np.median([x["columns"] for x in filtered_info])) if filtered_info else 0,
        "filtered_mean_cv_landmark_gain": round(float(np.mean([x["max_cv_landmark_gain"] for x in filtered_info])), 4) if filtered_info else 0.0,
        "filtered_mean_holdout_catboost_oracle_gain": round(float(np.mean([x["holdout_catboost_oracle_gain"] for x in filtered_info])), 4) if filtered_info else 0.0,
        "unnormalized_low_signal_probe_24ds_mean_syn_minus_real": {
            k: round(float(np.mean(v)), 4) for k, v in unnormalized_probe_deltas.items() if v
        },
        "auditor_roster_and_backends": auditor_roster,
        "filtered_candidates_sample_25": filtered_info[:25],
    }
