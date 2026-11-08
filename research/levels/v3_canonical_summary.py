"""Pure canonical ledger and Elo summary reducers for the v3 runner."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .v3_ablation_and_eval import ALL_TIERS as LEVEL_NAMES
from .v3_canonical_roster_eval import CANONICAL_6_AUDITORS, CORE_COMPARATOR_AUDITORS
from .world_class_eval import compute_elo_ratings


def _arr_mean(vals: Sequence[float]) -> Optional[float]:
    return round(float(np.mean(vals)), 4) if vals else None


def _compute_elo_and_paired_block(subset_rows: Sequence[Dict[str, Any]], n_boot: int = 200) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    methods = ("dope_mlc_v3_wire", "rows_wire", "copula_wire", "dope_mlc_v2_wire")
    cell_scores, ds_to_cells = {}, {}
    paired_diffs: Dict[str, List[float]] = {m: [] for m in methods[1:]}
    wtl_counts: Dict[str, Dict[str, int]] = {m: {"wins": 0, "ties_001": 0, "ties_010": 0, "losses": 0, "failed_or_missing": 0} for m in methods[1:]}
    for r in subset_rows:
        ds_id, lvl = r["dataset_id"], r["level"]
        v3_auds = r["domains"]["sender_scaled_test_domain"]["per_auditor"]
        for aud in CORE_COMPARATOR_AUDITORS:
            v3_a = v3_auds.get(aud, {})
            if "clipped_R" not in v3_a:
                continue
            sc_map: Dict[str, float] = {"dope_mlc_v3_wire": 100.0 * float(v3_a["clipped_R"])}
            for comp_name in methods[1:]:
                if comp_name not in r.get("comparators", {}):
                    if comp_name != "dope_mlc_v2_wire" or (r["task"] != "multiclass" and lvl != "small"):
                        wtl_counts[comp_name]["failed_or_missing"] += 1
                    continue
                c_aud = r["comparators"][comp_name]["sender_scaled"]["per_auditor"].get(aud, {})
                if "clipped_R" not in c_aud:
                    wtl_counts[comp_name]["failed_or_missing"] += 1
                    continue
                sc_map[comp_name] = 100.0 * float(c_aud["clipped_R"])
                diff = float(v3_a["clipped_R"]) - float(c_aud["clipped_R"])
                paired_diffs[comp_name].append(diff)
                if abs(diff) <= 0.01:
                    wtl_counts[comp_name]["ties_001"] += 1
                if abs(diff) <= 0.10:
                    wtl_counts[comp_name]["ties_010"] += 1
                elif diff > 0.10:
                    wtl_counts[comp_name]["wins"] += 1
                else:
                    wtl_counts[comp_name]["losses"] += 1
            cell_scores[(ds_id, lvl, aud)] = sc_map
            ds_to_cells.setdefault(ds_id, []).append((lvl, aud, sc_map))

    elo_base = compute_elo_ratings(cell_scores, methods)
    ds_keys, rng = sorted(ds_to_cells.keys()), np.random.default_rng(20261024)
    boot_elos: Dict[str, List[float]] = {m: [] for m in methods}
    if ds_keys:
        for _ in range(n_boot):
            samp = rng.choice(ds_keys, size=len(ds_keys), replace=True)
            b_cells = {(f"{d_k}_{i}", l_k, a_k): sm for i, d_k in enumerate(samp) for l_k, a_k, sm in ds_to_cells[d_k]}
            for er in compute_elo_ratings(b_cells, methods):
                boot_elos[er["method"]].append(float(er["elo"]))
    for er in elo_base:
        arr_b = np.asarray(boot_elos[er["method"]], dtype=np.float64)
        if arr_b.size > 0:
            er["ci95_low"], er["ci95_high"], er["bootstrap_se"] = round(float(np.percentile(arr_b, 2.5)), 1), round(float(np.percentile(arr_b, 97.5)), 1), round(float(np.std(arr_b)), 2)

    paired_summary: Dict[str, Any] = {}
    for comp_name, d_list in paired_diffs.items():
        if not d_list:
            continue
        arr_d = np.asarray(d_list, dtype=np.float64)
        b_means = [float(np.mean(rng.choice(arr_d, size=arr_d.size, replace=True))) for _ in range(n_boot)]
        paired_summary[f"dope_mlc_v3_minus_{comp_name}"] = {"paired_complete_case_cell_count": int(arr_d.size), "mean_delta_R": round(float(np.mean(arr_d)), 4), "median_delta_R": round(float(np.median(arr_d)), 4), "ci95_mean_delta_R": [round(float(np.percentile(b_means, 2.5)), 4), round(float(np.percentile(b_means, 97.5)), 4)], **wtl_counts[comp_name]}
    return elo_base, paired_summary


def summarize_canonical_ledger_and_elo(rows: Sequence[Dict[str, Any]], total_planned_mc: int, total_planned_v2_slice: int, n_boot: int = 200) -> Dict[str, Any]:
    completed = [r for r in rows if r.get("cell_status") == "completed"]
    by_role_lvl, elo_by_role = {}, {}
    planned_by_role = {"eval_gradient_disjoint_multiclass_panel": total_planned_mc, "eval_gradient_disjoint_v2_panel": total_planned_v2_slice}
    for role in ("eval_gradient_disjoint_multiclass_panel", "eval_gradient_disjoint_v2_panel"):
        r_rows = [r for r in completed if r.get("role") == role]
        if not r_rows:
            continue
        r_elo, r_paired = _compute_elo_and_paired_block(r_rows, n_boot=n_boot)
        elo_by_role[role] = {"paired_elo_with_200x_bootstrap_ci": r_elo, "paired_retention_differences_and_wtl": r_paired}
        by_lvl: Dict[str, Any] = {}
        for lvl in LEVEL_NAMES:
            sub = [r for r in r_rows if r.get("level") == lvl]
            if not sub:
                continue
            sc_c = [float(r["domains"]["sender_scaled_test_domain"]["complete_case_mean_clipped_R"]) for r in sub if r["domains"]["sender_scaled_test_domain"]["complete_case_mean_clipped_R"] is not None]
            sc_u = [float(r["domains"]["sender_scaled_test_domain"]["complete_case_mean_unclipped_R"]) for r in sub if r["domains"]["sender_scaled_test_domain"]["complete_case_mean_unclipped_R"] is not None]
            sc_d = [float(r["domains"]["sender_scaled_test_domain"]["mean_driver_fidelity_D"]) for r in sub if r["domains"]["sender_scaled_test_domain"]["mean_driver_fidelity_D"] is not None]
            sc_z = [float(r["domains"]["sender_scaled_test_domain"]["sensitivity_zero_imputed_all_denom_mean_clipped_R"]) for r in sub]
            raw_c = [float(r["domains"]["isolated_receiver_csc1_raw_domain"]["complete_case_mean_clipped_R"]) for r in sub if r["domains"]["isolated_receiver_csc1_raw_domain"]["complete_case_mean_clipped_R"] is not None]
            raw_u = [float(r["domains"]["isolated_receiver_csc1_raw_domain"]["complete_case_mean_unclipped_R"]) for r in sub if r["domains"]["isolated_receiver_csc1_raw_domain"]["complete_case_mean_unclipped_R"] is not None]
            raw_d = [float(r["domains"]["isolated_receiver_csc1_raw_domain"]["mean_driver_fidelity_D"]) for r in sub if r["domains"]["isolated_receiver_csc1_raw_domain"]["mean_driver_fidelity_D"] is not None]
            raw_z = [float(r["domains"]["isolated_receiver_csc1_raw_domain"]["sensitivity_zero_imputed_all_denom_mean_clipped_R"]) for r in sub]
            diag_c = [float(r["domains"]["diagnostic_exact_fp64_inverse_raw_domain"]["complete_case_mean_clipped_R"]) for r in sub if r["domains"]["diagnostic_exact_fp64_inverse_raw_domain"]["complete_case_mean_clipped_R"] is not None]
            wb, w_cap = [int(r["wire_accounting"]["dml3_complete_wire_bytes"]) for r in sub], [1.0 if r["wire_accounting"]["within_cap"] else 0.0 for r in sub]
            per_aud_summary: Dict[str, Any] = {}
            for aud in CANONICAL_6_AUDITORS:
                a_sc = [float(r["domains"]["sender_scaled_test_domain"]["per_auditor"][aud]["clipped_R"]) for r in sub if "clipped_R" in r["domains"]["sender_scaled_test_domain"]["per_auditor"].get(aud, {})]
                a_raw = [float(r["domains"]["isolated_receiver_csc1_raw_domain"]["per_auditor"][aud]["clipped_R"]) for r in sub if "clipped_R" in r["domains"]["isolated_receiver_csc1_raw_domain"]["per_auditor"].get(aud, {})]
                a_drv = [float(r["domains"]["sender_scaled_test_domain"]["per_auditor"][aud]["driver_fidelity_D"]) for r in sub if "driver_fidelity_D" in r["domains"]["sender_scaled_test_domain"]["per_auditor"].get(aud, {})]
                a_fail = sum(1 for r in sub if r["domains"]["sender_scaled_test_domain"]["per_auditor"].get(aud, {}).get("status", "").startswith("failed"))
                a_flr = sum(1 for r in sub if r["domains"]["sender_scaled_test_domain"]["per_auditor"].get(aud, {}).get("denom_floor_applied", False))
                a_part = sum(1 for r in sub if r["domains"]["sender_scaled_test_domain"]["per_auditor"].get(aud, {}).get("train_label_status", "") not in ("authentic_continuous_target", "complete_binary_support", "complete_multiclass_support", ""))
                per_aud_summary[aud] = {
                    "attempted_cells_denom": len(sub), "completed_cells": len(a_sc), "failed_cells": a_fail, "missing_or_error_rate": round(float(a_fail) / float(max(len(sub), 1)), 5),
                    "denom_floor_cells": a_flr, "partial_or_degenerate_label_cells": a_part, "complete_case_sender_scaled_mean_clipped_R": _arr_mean(a_sc),
                    "sensitivity_zero_imputed_sender_scaled_mean_clipped_R": round(float(np.sum(a_sc)) / float(max(len(sub), 1)), 4),
                    "complete_case_receiver_csc1_raw_mean_clipped_R": _arr_mean(a_raw),
                    "sensitivity_zero_imputed_receiver_csc1_raw_mean_clipped_R": round(float(np.sum(a_raw)) / float(max(len(sub), 1)), 4),
                    "mean_driver_fidelity_D": _arr_mean(a_drv), "is_official_kumo_or_tabpfn": False,
                }
            by_lvl[lvl] = {
                "planned_datasets_denom": planned_by_role[role], "completed_datasets": len(sub),
                "dataset_missing_or_unexecuted_rate": round(float(max(planned_by_role[role] - len(sub), 0)) / float(max(planned_by_role[role], 1)), 5),
                "mean_dml3_complete_wire_bytes": round(float(np.mean(wb)), 1), "max_dml3_complete_wire_bytes": int(np.max(wb)), "within_cap_pct": round(100.0 * float(np.mean(w_cap)), 2),
                "sender_scaled_6aud": {"complete_case_mean_clipped_R": _arr_mean(sc_c), "complete_case_mean_unclipped_R": _arr_mean(sc_u), "sensitivity_zero_imputed_all_6aud_mean_clipped_R": _arr_mean(sc_z), "mean_driver_fidelity_D": _arr_mean(sc_d)},
                "isolated_receiver_csc1_raw_6aud": {"is_deployed_compact_wire": True, "complete_case_mean_clipped_R": _arr_mean(raw_c), "complete_case_mean_unclipped_R": _arr_mean(raw_u), "sensitivity_zero_imputed_all_6aud_mean_clipped_R": _arr_mean(raw_z), "mean_driver_fidelity_D": _arr_mean(raw_d)},
                "diagnostic_exact_fp64_inverse_control_6aud": {"is_deployed_compact_wire": False, "complete_case_mean_clipped_R": _arr_mean(diag_c), "mean_abs_delta_vs_csc1": round(float(np.mean(np.abs(np.array(raw_c) - np.array(diag_c)))), 5) if (len(raw_c) == len(diag_c) and raw_c) else None},
                "per_auditor": per_aud_summary,
            }
        by_role_lvl[role] = by_lvl

    elo_base, paired_summary = _compute_elo_and_paired_block(completed, n_boot=n_boot)
    done_mc = len({r["dataset_id"] for r in completed if r.get("role") == "eval_gradient_disjoint_multiclass_panel"})
    done_v2 = len({r["dataset_id"] for r in completed if r.get("role") == "eval_gradient_disjoint_v2_panel"})
    failed_or_skip = len([r for r in rows if r.get("cell_status", "").startswith(("failed", "skipped"))])
    unexec = [r for r in rows if r.get("cell_status", "").startswith(("unexecuted", "interrupted"))]
    return {
        "coverage": {"planned_multiclass_datasets": total_planned_mc, "completed_multiclass_datasets": done_mc, "remaining_unexecuted_multiclass_datasets": max(total_planned_mc - done_mc, 0), "planned_v2_panel_slice_datasets": total_planned_v2_slice, "completed_v2_panel_slice_datasets": done_v2, "remaining_unexecuted_v2_panel_datasets_in_1000_panel": 1000 - done_v2, "completed_dataset_tier_cells": len(completed), "failed_or_skipped_dataset_records": failed_or_skip, "unexecuted_or_interrupted_records": len(unexec)},
        "elo_and_comparator_scope_disclosure": {"elo_and_paired_auditor_subset": list(CORE_COMPARATOR_AUDITORS), "elo_and_paired_domain": "sender_scaled_test_domain", "dope_mlc_v3_retention_and_driver_auditor_roster": list(CANONICAL_6_AUDITORS), "dope_mlc_v2_wire_evaluated_scope": "regression_and_binary_classification_at_medium_high_ultra_tiers_only_no_small_tier_or_multiclass_head"},
        "by_role_and_level": by_role_lvl, "paired_elo_with_200x_bootstrap_ci": elo_base,
        "paired_retention_differences_and_wtl": paired_summary, "elo_and_paired_by_role": elo_by_role,
    }
