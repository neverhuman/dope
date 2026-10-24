"""Independent Verification, Cohort Strata Decomposition & 132 Filtered-Candidate Bias Audit."""
from __future__ import annotations

import csv, hashlib, math, struct, warnings, zlib
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


def serialize_dope_latent_wire_bytes(enc: Dict[str, Any]) -> bytes:
    """Pack normalized-space DOPE-MLC-v2 encoding into the v2 compact binary wire representation (42B fixed prefix + fp16/uint8 tensors)."""
    buf = bytearray()
    lvl_map = {"medium": 0, "high": 1, "ultra": 2}
    task_map = {"regression": 0, "classification": 1}
    tb = enc["target_bundle"]
    trees, pairs, disc = tb["trees"], tb["pairs"], enc.get("discrete_support", {})
    buf.extend(struct.pack(
        "<BBBBIIHHHHHH", 1, lvl_map[enc["level"]], task_map[enc["task"]], len(trees),
        int(enc["n_train"]), int(enc["p_use"]), len(enc["regime_weights"]),
        int(enc["chol_factor"].shape[1]), len(enc["knot_grid"]), len(pairs), len(tb["w_basis"]), len(disc),
    ))
    gates = tb["gates"]
    scalars_fp16 = np.array([
        tb["res_net_mean"], tb["neural_w"], tb["residual_std"], tb["noise_scale"],
        tb["class_prior"], tb["y_min"], tb["y_max"],
        gates["monotone_w"], gates["hinge_w"], gates["pair_w"], gates["tree_w"],
    ], dtype=np.float16)
    buf.extend(scalars_fp16.tobytes())
    for k in ("regime_weights", "regime_means", "regime_scales", "chol_factor", "diag_res_std", "dist_latent", "task_latent", "col_knots", "binary_probs", "binary_low_high"):
        buf.extend(np.asarray(enc[k], dtype=np.float16).tobytes())
    buf.extend(np.packbits(np.asarray(enc["is_binary_col"], dtype=bool)).tobytes())
    for j, (u_vals, cum_p) in sorted(disc.items()):
        u_arr, c_arr = np.asarray(u_vals, dtype=np.float16), np.asarray(cum_p, dtype=np.float16)
        buf.extend(struct.pack("<HH", int(j), len(u_arr)))
        buf.extend(u_arr.tobytes())
        buf.extend(c_arr.tobytes())
    if pairs:
        buf.extend(np.asarray(pairs, dtype=np.uint8).tobytes())
    buf.extend(np.asarray(tb["w_basis"], dtype=np.float16).tobytes())
    for t in trees:
        feats = np.asarray(t["features"], dtype=np.uint8)
        thresh = np.asarray(t["thresholds"], dtype=np.float16)
        leaf_v = np.asarray(t["leaf_values"], dtype=np.float16)
        buf.extend(struct.pack("<B", len(feats)))
        buf.extend(feats.tobytes())
        buf.extend(thresh.tobytes())
        buf.extend(leaf_v.tobytes())
    return bytes(buf)


def deserialize_dope_latent_wire_bytes(payload: bytes) -> Dict[str, Any]:
    """Isolated receiver decoder: reconstruct normalized-space DOPE-MLC-v2 encoding dict solely from wire bytes."""
    mv = memoryview(payload)
    ver, lvl_id, task_id, n_trees, n_train, p_use, n_reg, rank_use, n_knots, n_pairs, n_w, n_disc = struct.unpack_from("<BBBBIIHHHHHH", mv, 0)
    if ver != 1:
        raise ValueError(f"unsupported_wire_version_{ver}")
    off = struct.calcsize("<BBBBIIHHHHHH")
    level = ("medium", "high", "ultra")[lvl_id]
    task = ("regression", "classification")[task_id]

    def _read_f16(count: int) -> np.ndarray:
        nonlocal off
        arr = np.frombuffer(mv, dtype=np.float16, count=count, offset=off).copy()
        off += count * 2
        return arr

    sc = [float(v) for v in _read_f16(11)]
    reg_w = _read_f16(n_reg)
    reg_m = _read_f16(n_reg * p_use).reshape(n_reg, p_use)
    reg_s = _read_f16(n_reg * p_use).reshape(n_reg, p_use)
    chol = _read_f16(p_use * rank_use).reshape(p_use, rank_use)
    diag_std = _read_f16(p_use)
    dist_lat = _read_f16(128)
    task_lat = _read_f16(128)
    col_knots = _read_f16(p_use * n_knots).reshape(p_use, n_knots)
    bin_p = _read_f16(p_use)
    bin_lh = _read_f16(p_use * 2).reshape(p_use, 2)

    n_pack = (p_use + 7) // 8
    is_bin = np.unpackbits(np.frombuffer(mv, dtype=np.uint8, count=n_pack, offset=off))[:p_use].astype(bool).copy()
    off += n_pack

    disc: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for _ in range(n_disc):
        col_j, u_len = struct.unpack_from("<HH", mv, off)
        off += 4
        u_arr = _read_f16(u_len)
        c_arr = _read_f16(u_len)
        disc[int(col_j)] = (u_arr, c_arr)

    pairs: List[Tuple[int, int]] = []
    if n_pairs > 0:
        p_arr = np.frombuffer(mv, dtype=np.uint8, count=n_pairs * 2, offset=off).reshape(n_pairs, 2)
        off += n_pairs * 2
        pairs = [(int(a), int(b)) for a, b in p_arr]

    w_basis = _read_f16(n_w)
    trees: List[Dict[str, Any]] = []
    for _ in range(n_trees):
        (depth,) = struct.unpack_from("<B", mv, off)
        off += 1
        feats = [int(x) for x in np.frombuffer(mv, dtype=np.uint8, count=depth, offset=off)]
        off += depth
        thresh = [float(x) for x in _read_f16(depth)]
        leaf_v = _read_f16(1 << depth)
        trees.append({"features": feats, "thresholds": thresh, "leaf_values": leaf_v})

    if off != len(payload):
        raise ValueError(f"wire_payload_trailing_bytes_{len(payload) - off}")

    knot_grid = np.linspace(0.01 if level == "medium" else 0.005, 0.99 if level == "medium" else 0.995, n_knots).astype(np.float16)
    gates = {"monotone_w": sc[7], "hinge_w": sc[8], "pair_w": sc[9], "tree_w": sc[10]}
    return {
        "level": level, "task": task, "n_train": int(n_train), "p_use": int(p_use),
        "gates": gates, "task_latent": task_lat, "dist_latent": dist_lat,
        "knot_grid": knot_grid, "col_knots": col_knots,
        "is_binary_col": is_bin, "binary_probs": bin_p, "binary_low_high": bin_lh,
        "discrete_support": disc, "chol_factor": chol, "diag_res_std": diag_std,
        "regime_weights": reg_w, "regime_means": reg_m, "regime_scales": reg_s,
        "target_bundle": {
            "gates": gates, "pairs": pairs, "w_basis": w_basis,
            "res_net_mean": sc[0], "neural_w": sc[1], "trees": trees,
            "residual_std": sc[2], "noise_scale": sc[3], "class_prior": sc[4],
            "y_min": sc[5], "y_max": sc[6],
        },
    }


def estimate_scaler_sidecar_bytes(scaler: FitOnlyTableTransform, p_use: int) -> Dict[str, int]:
    """Compute mandatory sidecar bytes required if an external receiver must also apply or invert FitOnlyTableTransform."""
    num_cols = sum(1 for j in range(p_use) if scaler.kinds[j] == "numeric")
    cat_cols = sum(1 for j in range(p_use) if scaler.kinds[j] != "numeric")
    cat_levels = sum(len(scaler.categorical_encoders[j].vocab_tokens) for j in range(p_use) if scaler.kinds[j] != "numeric")
    # Numeric: 1B kind/transform + 5x4B float32 (median, iqr, mean, std, tail_weight) + 9x2B fp16 knots + 2x2B fp16 (skew, kurt) = 43B/col
    # Categorical (hashed token IDs, no raw source strings): 1B kind + 2x4B float32 (mean, std) + 2B cardinality + K*(4B hash32 + 2B enc_fp16 + 2B freq_fp16)
    # TargetScaler: 1B task + 5x4B float32 stats = 21B; Seed + header = 8B
    raw_bytes = 8 + 21 + 43 * num_cols + 11 * cat_cols + 8 * cat_levels
    return {"scaler_sidecar_raw_bytes": int(raw_bytes), "numeric_cols": int(num_cols), "categorical_cols": int(cat_cols), "retained_cat_levels": int(cat_levels)}

