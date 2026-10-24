"""Process-Group Bounded Runner & Paired Elo/Uncertainty Reducer for DOPE-MLC-v3 Canonical Roster."""
from __future__ import annotations

import argparse, csv, ctypes, gzip, hashlib, json, multiprocessing as mp, os, queue, signal, sys, time, warnings, zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import numpy as np
import torch

from .budget_planner import TIER_BYTE_CAPS, deserialize_dope_v3_envelope, pack_compact_scaler_sidecar, serialize_dope_v3_envelope
from .multilevel_model import MultiLevelLandmarkModel
from .precalc.plan import training_rows
from .scaling import FitOnlyTableTransform
from .strata_audit import deserialize_dope_latent_wire_bytes, estimate_scaler_sidecar_bytes, serialize_dope_latent_wire_bytes
from .v3_ablation_and_eval import ALL_TIERS as LEVEL_NAMES
from .v3_canonical_roster_eval import (
    AUDITOR_ERRORS, CANONICAL_6_AUDITORS, CANONICAL_V3_AUDITOR_PROTOCOL_SPEC, CORE_COMPARATOR_AUDITORS,
    DEFAULT_V2_CKPT, V3_MANIFEST_PATH, V3_OUT_DIR, evaluate_canonical_oracle, measure_host_cpu_percent,
    pack_copula_complete_wire, pack_rows_complete_wire, score_synthetic_against_oracle,
    unpack_copula_complete_wire, unpack_rows_complete_wire,
)
from .v3_ledger_replay import _load_single_mc_cell
from .world_class_eval import DEFAULT_BUNDLE, DEFAULT_SOURCE_CACHE, compute_elo_ratings


def _load_v2_panel_scopes(ds_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    wanted, found = set(ds_ids), {}
    if not DEFAULT_BUNDLE.exists():
        return found
    with gzip.open(DEFAULT_BUNDLE, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sc = rec.get("scope", {})
            d_id = sc.get("dataset_id")
            if d_id in wanted and sc.get("split") == "outer5-iid-fold0-v1" and d_id not in found:
                found[d_id] = rec
                if len(found) == len(wanted):
                    break
    return found


def _prepare_dataset_split(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task, prep_cfg = spec["task"], CANONICAL_V3_AUDITOR_PROTOCOL_SPEC["preprocessing"]
    max_p = int(prep_cfg["max_p_features"])
    if task == "multiclass":
        cell, _, _ = _load_single_mc_cell(spec["entry"], targeted_recovery=True)
        if cell is None:
            return None
        X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = cell["X_fit_raw"], cell["y_fit_raw"], cell["X_test_raw"], cell["y_test_raw"]
        k_cls, rep = int(cell["k_classes"]), cell["report"]
    else:
        sc = spec["scope"]
        csv_path = DEFAULT_SOURCE_CACHE / f"{sc['train_sha256']}.csv"
        if not csv_path.exists():
            return None
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
        if sc.get("has_header", False) and rows:
            rows = rows[1:]
        if len(rows) < 50:
            return None
        fit_idx = training_rows(len(rows), sc["physical_table_id"], sc["split"], int(sc.get("split_seed", 20261023)))
        fit_set, max_tr, max_te = set(fit_idx), int(prep_cfg["max_train_rows"]), int(prep_cfg["max_test_rows"])
        test_idx = [i for i in range(len(rows)) if i not in fit_set]
        if len(fit_idx) > max_tr:
            fit_idx = [fit_idx[i] for i in np.sort(np.random.default_rng(int(prep_cfg["split_seed_train_subsample"])).choice(len(fit_idx), size=max_tr, replace=False))]
        if len(test_idx) > max_te:
            test_idx = [test_idx[i] for i in np.sort(np.random.default_rng(int(prep_cfg["split_seed_test_subsample"])).choice(len(test_idx), size=max_te, replace=False))]
        X_tr_raw, y_tr_raw = [rows[i][:-1] for i in fit_idx], [rows[i][-1] for i in fit_idx]
        X_te_raw, y_te_raw = [rows[i][:-1] for i in test_idx], [rows[i][-1] for i in test_idx]
        k_cls, rep = 2, spec["report"]
    scaler = FitOnlyTableTransform.fit(X_tr_raw, y_tr_raw, task=task)
    X_tr_s, y_tr_s = scaler.transform(X_tr_raw, y_tr_raw)
    X_te_s, y_te_s = scaler.transform(X_te_raw, y_te_raw)
    p_use = min(X_tr_s.shape[1], max_p)
    X_tr_s, X_te_s = X_tr_s[:, :p_use], X_te_s[:, :p_use]
    X_tr_num, y_tr_inv = scaler.inverse_transform(X_tr_s, y_tr_s)
    X_te_num, y_te_inv = scaler.inverse_transform(X_te_s, y_te_s)
    sc_b = int(estimate_scaler_sidecar_bytes(scaler, p_use)["scaler_sidecar_raw_bytes"])
    return {"X_tr_s": X_tr_s, "y_tr_s": y_tr_s, "X_te_s": X_te_s, "y_te_s": y_te_s, "X_tr_num": X_tr_num, "y_tr_raw": y_tr_raw if task == "multiclass" else y_tr_inv, "X_te_num": X_te_num, "y_te_raw": y_te_raw if task == "multiclass" else y_te_inv, "scaler": scaler, "p_use": p_use, "k_classes": k_cls, "report": rep, "diagnostic_fp64_scaler_sidecar_bytes": sc_b}


def evaluate_dataset_across_tiers(spec: Dict[str, Any], model_v3: MultiLevelLandmarkModel, model_v2: MultiLevelLandmarkModel) -> List[Dict[str, Any]]:
    t0_ds, prep = time.perf_counter(), _prepare_dataset_split(spec)
    if prep is None:
        return [{"dataset_id": spec["dataset_id"], "role": spec["role"], "task": spec["task"], "level": "all", "cell_status": "skipped_invalid_split", "measurement_provenance": "frozen_checkpoint_canonical_roster_eval"}]
    X_tr_s, y_tr_s, X_te_s, y_te_s = prep["X_tr_s"], prep["y_tr_s"], prep["X_te_s"], prep["y_te_s"]
    X_tr_num, y_tr_raw, X_te_num, y_te_raw = prep["X_tr_num"], prep["y_tr_raw"], prep["X_te_num"], prep["y_te_raw"]
    scaler, p_use, k_cls, rep = prep["scaler"], prep["p_use"], prep["k_classes"], prep["report"]
    task, n_tr, n_te = spec["task"], int(X_tr_s.shape[0]), int(X_te_s.shape[0])
    oracle_scaled = evaluate_canonical_oracle(X_tr_s, y_tr_s, X_te_s, y_te_s, task=task, n_classes=k_cls)
    rec_real_sc = FitOnlyTableTransform.fit(X_tr_num, y_tr_raw, task=task)
    X_tr_rr, y_tr_rr = rec_real_sc.transform(X_tr_num, y_tr_raw)
    X_te_rr, y_te_rr = rec_real_sc.transform(X_te_num, y_te_raw)
    oracle_raw = evaluate_canonical_oracle(X_tr_rr[:, :p_use], y_tr_rr, X_te_rr[:, :p_use], y_te_rr, task=task, n_classes=k_cls)
    prep_ms, rows_out = (time.perf_counter() - t0_ds) * 1000.0, []
    for lvl in LEVEL_NAMES:
        t0_l, cap = time.perf_counter(), TIER_BYTE_CAPS[lvl]
        enc_v3 = model_v3.encode_table(X_tr_s, y_tr_s, rep, task=task, level=lvl, adaptive_budget=True, include_v3_primitives=True)
        csc1_sc_bytes = pack_compact_scaler_sidecar(scaler, p_use, level=lvl)
        wire_v3 = serialize_dope_v3_envelope(enc_v3, scaler=scaler, compress=True)
        dec_enc, dec_csc1 = deserialize_dope_v3_envelope(wire_v3)
        X_syn_s, y_syn_s = model_v3.generate_synthetic(dec_enc, n_rows=n_tr, seed=20261024)
        v3_scaled = score_synthetic_against_oracle(oracle_scaled, X_syn_s, y_syn_s, X_te_s, y_te_s, task=task, n_classes=k_cls, auditors=CANONICAL_6_AUDITORS, compute_drivers=True)
        assert dec_csc1 is not None and rec_real_sc.target_scaler is not None
        X_syn_csc1_raw, y_syn_csc1_raw = dec_csc1.inverse_transform(X_syn_s, y_syn_s)
        rec_csc1_fit = FitOnlyTableTransform.fit(X_syn_csc1_raw, y_syn_csc1_raw, task=task)
        X_syn_c_ev, y_syn_c_ev = rec_csc1_fit.transform_features(X_syn_csc1_raw)[:, :p_use], rec_real_sc.target_scaler.transform(y_syn_csc1_raw)
        X_te_c_ev = rec_csc1_fit.transform_features(X_te_num)[:, :p_use]
        X_te_csc1_inv, _ = dec_csc1.inverse_transform(X_te_s, y_te_s)
        csc1_max_te_diff = float(np.max(np.abs(np.nan_to_num(X_te_csc1_inv - X_te_num, nan=0.0))))
        v3_csc1_raw = score_synthetic_against_oracle(oracle_raw, X_syn_c_ev, y_syn_c_ev, X_te_c_ev, y_te_rr, task=task, n_classes=k_cls, auditors=CANONICAL_6_AUDITORS, compute_drivers=True)
        X_syn_fp64_raw, y_syn_fp64_raw = scaler.inverse_transform(X_syn_s, y_syn_s)
        rec_fp64_fit = FitOnlyTableTransform.fit(X_syn_fp64_raw, y_syn_fp64_raw, task=task)
        X_syn_f_ev, y_syn_f_ev = rec_fp64_fit.transform_features(X_syn_fp64_raw)[:, :p_use], rec_real_sc.target_scaler.transform(y_syn_fp64_raw)
        X_te_f_ev = rec_fp64_fit.transform_features(X_te_num)[:, :p_use]
        v3_fp64_diag = score_synthetic_against_oracle(oracle_raw, X_syn_f_ev, y_syn_f_ev, X_te_f_ev, y_te_rr, task=task, n_classes=k_cls, auditors=CANONICAL_6_AUDITORS, compute_drivers=False)
        fp64_sc_b, payload_b = prep["diagnostic_fp64_scaler_sidecar_bytes"], max(len(wire_v3) - len(csc1_sc_bytes) - 10, 1)
        fp64_total_wire_b = 10 + fp64_sc_b + payload_b
        comparators: Dict[str, Any] = {}
        w_rows = pack_rows_complete_wire(X_tr_s, y_tr_s, scaler, lvl, seed=20261024)
        Xr_s, yr_s, _ = unpack_rows_complete_wire(w_rows, n_tr, seed=20261024)
        comparators["rows_wire"] = {"complete_wire_bytes": len(w_rows), "within_cap": len(w_rows) <= cap, "sender_scaled": score_synthetic_against_oracle(oracle_scaled, Xr_s, yr_s, X_te_s, y_te_s, task=task, n_classes=k_cls, auditors=CORE_COMPARATOR_AUDITORS, compute_drivers=False)}
        w_cop = pack_copula_complete_wire(X_tr_s, y_tr_s, scaler, lvl, task=task, n_classes=k_cls)
        Xc_s, yc_s, _ = unpack_copula_complete_wire(w_cop, n_tr, seed=20261024)
        comparators["copula_wire"] = {"complete_wire_bytes": len(w_cop), "within_cap": len(w_cop) <= cap, "sender_scaled": score_synthetic_against_oracle(oracle_scaled, Xc_s, yc_s, X_te_s, y_te_s, task=task, n_classes=k_cls, auditors=CORE_COMPARATOR_AUDITORS, compute_drivers=False)}
        if task in ("regression", "classification") and lvl in ("medium", "high", "ultra"):
            enc_v2 = model_v2.encode_table(X_tr_s, y_tr_s, rep, task=task, level=lvl, adaptive_budget=False, include_v3_primitives=False)
            w_v2_lat = serialize_dope_latent_wire_bytes(enc_v2, version=2)
            w_v2_full = zlib.compress(w_v2_lat + csc1_sc_bytes, level=9)
            Xv2_s, yv2_s = model_v2.generate_synthetic(deserialize_dope_latent_wire_bytes(w_v2_lat), n_rows=n_tr, seed=20261024)
            v2_auds = CANONICAL_6_AUDITORS if lvl == "ultra" else CORE_COMPARATOR_AUDITORS
            comparators["dope_mlc_v2_wire"] = {"latent_wire_v2_bytes": len(w_v2_lat), "complete_wire_zlib_with_csc1_bytes": len(w_v2_full), "within_cap_zlib": len(w_v2_full) <= cap, "sender_scaled": score_synthetic_against_oracle(oracle_scaled, Xv2_s, yv2_s, X_te_s, y_te_s, task=task, n_classes=k_cls, auditors=v2_auds, compute_drivers=(lvl == "ultra"))}
        c_raw, f_raw = v3_csc1_raw["complete_case_mean_clipped_R"], v3_fp64_diag["complete_case_mean_clipped_R"]
        abs_del = round(abs(float(c_raw) - float(f_raw)), 5) if (c_raw is not None and f_raw is not None) else None
        rows_out.append({
            "dataset_id": spec["dataset_id"], "role": spec["role"], "prior_v2_exposure": spec["prior_v2_exposure"], "task": task, "level": lvl,
            "cell_status": "completed", "measurement_provenance": "frozen_checkpoint_canonical_roster_eval", "n_train": n_tr, "n_test": n_te, "p_features": p_use, "k_classes": k_cls,
            "wire_accounting": {"nominal_cap_bytes": cap, "dml3_complete_wire_bytes": len(wire_v3), "csc1_sidecar_raw_bytes": len(csc1_sc_bytes), "payload_bytes_approx": payload_b, "within_cap": len(wire_v3) <= cap, "csc1_is_bit_exact_float64_scaler_parity": False, "csc1_max_abs_diff_test_X": round(csc1_max_te_diff, 5), "diagnostic_fp64_scaler_sidecar_bytes": fp64_sc_b, "diagnostic_fp64_total_wire_bytes": fp64_total_wire_b, "diagnostic_fp64_within_cap": fp64_total_wire_b <= cap},
            "domains": {"sender_scaled_test_domain": v3_scaled, "isolated_receiver_csc1_raw_domain": {"is_deployed_compact_wire": True, "charged_wire_bytes": len(wire_v3), **v3_csc1_raw}, "diagnostic_exact_fp64_inverse_raw_domain": {"is_deployed_compact_wire": False, "role": "diagnostic_control_requiring_full_fp64_scaler_sidecar_bytes", "uncharged_extra_fp64_sidecar_bytes_over_csc1": fp64_sc_b - len(csc1_sc_bytes), "hypothetical_total_wire_bytes_if_transmitted": fp64_total_wire_b, "csc1_vs_fp64_abs_delta_mean_clipped_R": abs_del, **v3_fp64_diag}},
            "comparators": comparators, "costs_ms": {"dataset_prep_and_oracle_ms": round(prep_ms, 2), "tier_total_eval_ms": round((time.perf_counter() - t0_l) * 1000.0, 2)},
        })
    return rows_out


def _worker_loop(task_q: "mp.Queue[Any]", res_q: "mp.Queue[Any]", v3_ckpt_str: str, v2_ckpt_str: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.setsid()
    for fn in (lambda: ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM) if sys.platform.startswith("linux") else 0, lambda: os.nice(10), lambda: torch.set_num_threads(1)):
        try:
            fn()
        except (OSError, RuntimeError):
            pass
    if os.getppid() == 1:
        return
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[k] = "1"
    if v3_ckpt_str and v2_ckpt_str:
        model_v3, model_v2 = MultiLevelLandmarkModel().to(torch.device("cpu")), MultiLevelLandmarkModel().to(torch.device("cpu"))
        model_v3.load_state_dict(torch.load(Path(v3_ckpt_str), map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        model_v2.load_state_dict(torch.load(Path(v2_ckpt_str), map_location="cpu", weights_only=False)["model_state_dict"], strict=False)
        model_v3.eval()
        model_v2.eval()
    else:
        model_v3 = model_v2 = None
        res_q.put(("ready", os.getpid(), os.environ.get("CUDA_VISIBLE_DEVICES", "unset"), None))
    while os.getppid() != 1:
        try:
            item = task_q.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None or model_v3 is None or model_v2 is None:
            break
        w_idx, spec = item
        res_q.put(("started", w_idx, spec["dataset_id"], None))
        try:
            res_q.put(("completed", w_idx, spec["dataset_id"], evaluate_dataset_across_tiers(spec, model_v3, model_v2)))
        except AUDITOR_ERRORS as exc:
            res_q.put(("failed", w_idx, spec["dataset_id"], [{"dataset_id": spec["dataset_id"], "role": spec["role"], "task": spec["task"], "level": "all", "cell_status": "failed_worker_exception", "error": type(exc).__name__, "measurement_provenance": "frozen_checkpoint_canonical_roster_eval"}]))


def _reap_worker_pgroups(workers: Sequence[mp.Process]) -> int:
    reaped = 0
    for w in workers:
        if w.pid is not None and w.is_alive():
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(w.pid, sig)
                except (ProcessLookupError, OSError):
                    break
                time.sleep(0.15)
            reaped += 1
        w.join(timeout=1.0)
    return reaped


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


def _collect_source_sha256s() -> Dict[str, str]:
    base = Path(__file__).resolve().parent
    files = ("v3_canonical_roster_eval.py", "v3_canonical_runner.py", "budget_planner.py", "multilevel_model.py", "scaling.py", "building_block_landmarks.py", "multiclass_head.py")
    return {fn: hashlib.sha256((base / fn).read_bytes()).hexdigest() for fn in files if (base / fn).exists()}


def _build_receipt_common(eff_run_id: str, run_dir: Path, ledger_path: Path, ledger_gz_path: Path, summary: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    v3_ckpt, v2_ckpt = V3_OUT_DIR / "dope_mlc_v3_multitask_best.pt", DEFAULT_V2_CKPT
    lga_cfg = CANONICAL_V3_AUDITOR_PROTOCOL_SPEC["auditors"]["local_gated_attention"]
    rem_v2 = 1000 - int(summary["coverage"]["completed_v2_panel_slice_datasets"])
    return {
        "schema": "dope.v3.canonical_roster_and_v2panel_slice_receipt.v2",
        "run_id": eff_run_id, "run_dir": str(run_dir), "command_line": " ".join(sys.argv),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra,
        "ledger_jsonl_path": str(ledger_path), "ledger_jsonl_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "ledger_jsonl_gz_path": str(ledger_gz_path), "ledger_jsonl_gz_sha256": hashlib.sha256(ledger_gz_path.read_bytes()).hexdigest(),
        "frozen_v3_checkpoint_sha256": hashlib.sha256(v3_ckpt.read_bytes()).hexdigest(),
        "frozen_v2_checkpoint_sha256": hashlib.sha256(v2_ckpt.read_bytes()).hexdigest(),
        "source_file_sha256s": _collect_source_sha256s(), "auditor_protocol_spec": CANONICAL_V3_AUDITOR_PROTOCOL_SPEC,
        "protocol_comparison_disclosure": "This bounded CPU evaluation uses CANONICAL_V3_AUDITOR_PROTOCOL_SPEC (24-25 tree estimators, 32x16 MLP, 12-step hidden_dim=32 LocalGatedAttentionAuditor, max 340 train / 160 test rows / 36 features), which differs from the historical v2 world_class_eval.py wrapper. Historical v2 receipt scores are NOT compared as matched against this run; all DOPE-MLC-v3 vs. DOPE-MLC-v2 / rows_wire / copula_wire comparisons reported here are re-evaluated side-by-side on identical splits, auditors, and byte caps inside this run.",
        "official_kumo_or_tabpfn_evaluated": False,
        "local_attention_auditor_identity": f"LocalGatedAttentionAuditor (_LocalGatedAttentionNet, {lga_cfg['adamw_steps']} AdamW steps, hidden_dim={lga_cfg['hidden_dim']} per table on CPU; NOT official KumoTabular or TabPFN)",
        **summary,
        "remaining_open_scientific_obligations": {
            "full_1000_dataset_v2_panel_under_v3": "CLOSED (1000/1000 completed in unified ledger)" if rem_v2 == 0 else f"OPEN ({rem_v2}/1000 datasets remain unmeasured under v3)",
            "unexecuted_multiclass_datasets_in_canonical_run": "CLOSED (69/69 completed)" if summary["coverage"]["remaining_unexecuted_multiclass_datasets"] == 0 else f"OPEN ({summary['coverage']['remaining_unexecuted_multiclass_datasets']}/69 unexecuted)",
            "source_family_disjointness_beyond_exact_sha256": "OPEN (0 exact train_sha256 overlaps verified; upstream benchmark-family lineage disjointness unverified)",
            "untouched_sealed_confirmatory_generalization": "OPEN (all panels are development/reconciliation validation cohorts)",
        },
    }


def run_bounded_canonical_eval(max_v2_slice: int = 80, workers_count: int = 4, wall_budget_s: float = 840.0, run_id: Optional[str] = None, shard_id: Optional[str] = None, explicit_v2_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    eff_run_id = run_id or shard_id or f"canonical_v3_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest = json.loads(V3_MANIFEST_PATH.read_text(encoding="utf-8"))
    v2_entries = [e for e in manifest["entries"] if e["v3_role"] == "eval_gradient_disjoint_v2_panel"]
    v2_by_id = {e["dataset_id"]: e for e in v2_entries}
    if shard_id or explicit_v2_ids:
        mc_entries = []
        t_ids = list(explicit_v2_ids) if explicit_v2_ids else list(json.loads((V3_OUT_DIR / "v3_remaining_920_v2panel_inventory.json").read_text(encoding="utf-8"))["shards"][str(shard_id)]["dataset_ids"])
        v2_slice = [v2_by_id[did] for did in t_ids if did in v2_by_id]
    else:
        mc_entries = [e for e in manifest["entries"] if e["v3_role"] == "eval_gradient_disjoint_multiclass_panel"]
        reg_e, bin_e = [e for e in v2_entries if e["catalog_task"] == "regression"], [e for e in v2_entries if e["catalog_task"] != "regression"]
        n_reg = min(len(reg_e), int(round(max_v2_slice * 0.62)))
        v2_slice = reg_e[:n_reg] + bin_e[: min(len(bin_e), max_v2_slice - n_reg)]
    run_dir = V3_OUT_DIR / "runs" / eff_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger_path, ledger_gz_path, receipt_path = run_dir / "ledger.jsonl", run_dir / "ledger.jsonl.gz", run_dir / "receipt.json"
    v2_scopes = _load_v2_panel_scopes([e["dataset_id"] for e in v2_slice])
    mc_tasks = [{"dataset_id": e["dataset_id"], "train_sha256": e["train_sha256"], "entry": e, "role": "eval_gradient_disjoint_multiclass_panel", "prior_v2_exposure": e["prior_exposure"], "task": "multiclass"} for e in mc_entries]
    v2_tasks = [{"dataset_id": e["dataset_id"], "train_sha256": e["train_sha256"], "entry": e, "role": "eval_gradient_disjoint_v2_panel", "prior_v2_exposure": e["prior_exposure"], "task": "classification" if e["catalog_task"] == "binary" else "regression", "scope": v2_scopes[e["dataset_id"]]["scope"], "report": v2_scopes[e["dataset_id"]]["report"]} for e in v2_slice if e["dataset_id"] in v2_scopes]
    all_tasks = [t for pair in zip(mc_tasks + [None] * max(0, len(v2_tasks) - len(mc_tasks)), v2_tasks + [None] * max(0, len(mc_tasks) - len(v2_tasks))) for t in pair if t is not None]

    ctx = mp.get_context("spawn")
    task_q, res_q = ctx.Queue(), ctx.Queue()
    v3_ckpt, v2_ckpt = str(V3_OUT_DIR / "dope_mlc_v3_multitask_best.pt"), str(DEFAULT_V2_CKPT)
    workers = [ctx.Process(target=_worker_loop, args=(task_q, res_q, v3_ckpt, v2_ckpt), daemon=True) for _ in range(workers_count)]
    prev_term = signal.signal(signal.SIGTERM, lambda s, _f: (_reap_worker_pgroups(workers), sys.exit(128 + s))[1])
    prev_int = signal.signal(signal.SIGINT, lambda s, _f: (_reap_worker_pgroups(workers), sys.exit(128 + s))[1])
    all_rows, in_flight, completed_ds = [], {}, set()
    next_idx, cpu_pause_events, stopped_by_wall, reaped_cnt = 0, 0, False, 0
    try:
        for w in workers:
            w.start()
        with ledger_path.open("x", encoding="utf-8") as lf:
            while len(completed_ds) < len(all_tasks):
                if (time.perf_counter() - t0) >= wall_budget_s:
                    stopped_by_wall = True
                    break
                if measure_host_cpu_percent(0.08) > 82.0:
                    cpu_pause_events += 1
                    time.sleep(0.5)
                else:
                    while len(in_flight) < workers_count and next_idx < len(all_tasks):
                        in_flight[next_idx] = all_tasks[next_idx]
                        task_q.put((next_idx, all_tasks[next_idx]))
                        next_idx += 1
                try:
                    msg_kind, w_idx, ds_id, payload = res_q.get(timeout=0.4)
                    if msg_kind in ("completed", "failed") and payload is not None:
                        in_flight.pop(w_idx, None)
                        completed_ds.add(ds_id)
                        for r in payload:
                            lf.write(json.dumps(r, sort_keys=True) + "\n")
                            all_rows.append(r)
                        lf.flush()
                except queue.Empty:
                    pass
            if not stopped_by_wall:
                for _ in workers:
                    task_q.put(None)
            reaped_cnt = _reap_worker_pgroups(workers)
            for t_spec in all_tasks:
                if t_spec["dataset_id"] not in completed_ds:
                    in_fl = any(v["dataset_id"] == t_spec["dataset_id"] for v in in_flight.values())
                    rec_u = {"dataset_id": t_spec["dataset_id"], "role": t_spec["role"], "task": t_spec["task"], "level": "all", "cell_status": "interrupted_wall_budget_reached" if in_fl else "unexecuted_wall_budget_reached", "measurement_provenance": "frozen_checkpoint_canonical_roster_eval"}
                    lf.write(json.dumps(rec_u, sort_keys=True) + "\n")
                    all_rows.append(rec_u)
            lf.flush()
    finally:
        reaped_cnt += _reap_worker_pgroups(workers)
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)
    with ledger_gz_path.open("xb") as raw_gf:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_gf, compresslevel=9, mtime=0) as gf:
            gf.write(ledger_path.read_bytes())
    summary = summarize_canonical_ledger_and_elo(all_rows, len(mc_tasks), len(v2_tasks))
    extra = {"shard_id": shard_id, "elapsed_wall_seconds": round(time.perf_counter() - t0, 2), "wall_budget_seconds": wall_budget_s, "stopped_by_wall_budget": stopped_by_wall, "workers_count": workers_count, "cpu_guard_82pct_pause_events": cpu_pause_events, "cpu_guard_semantics": "dispatch_admission_gate_only_82pct_proc_stat_plus_nice10_single_thread_workers_does_not_throttle_already_active_in_flight_workers", "reaped_worker_process_groups": reaped_cnt}
    receipt = _build_receipt_common(eff_run_id, run_dir, ledger_path, ledger_gz_path, summary, extra)
    with receipt_path.open("x", encoding="utf-8") as rf:
        rf.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def aggregate_canonical_runs(source_run_ids: Sequence[str], out_run_id: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    run_dir = V3_OUT_DIR / "runs" / out_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger_path, ledger_gz_path, receipt_path = run_dir / "ledger.jsonl", run_dir / "ledger.jsonl.gz", run_dir / "receipt.json"
    all_rows, non_comp_rows, seen_cells, comp_ds, src_manifests = [], [], set(), set(), []
    for s_id in source_run_ids:
        s_dir = V3_OUT_DIR / "runs" / s_id
        s_rec = json.loads((s_dir / "receipt.json").read_text(encoding="utf-8"))
        s_rows = [json.loads(line) for line in (s_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for r in s_rows:
            if r.get("cell_status") == "completed":
                key = (r["dataset_id"], r["level"])
                if key in seen_cells:
                    raise ValueError(f"Duplicate logical (dataset_id, level) cell {key} in {s_id}")
                seen_cells.add(key)
                comp_ds.add(r["dataset_id"])
                all_rows.append(r)
            else:
                non_comp_rows.append(r)
        src_manifests.append({"run_id": s_id, "ledger_jsonl_sha256": s_rec["ledger_jsonl_sha256"], "ledger_jsonl_gz_sha256": s_rec["ledger_jsonl_gz_sha256"], "elapsed_wall_seconds": s_rec["elapsed_wall_seconds"], "completed_dataset_tier_cells": s_rec["coverage"]["completed_dataset_tier_cells"], "source_file_sha256s": s_rec.get("source_file_sha256s", {})})
    for r in non_comp_rows:
        if r.get("dataset_id") not in comp_ds:
            all_rows.append(r)
    with ledger_path.open("x", encoding="utf-8") as lf:
        for r in all_rows:
            lf.write(json.dumps(r, sort_keys=True) + "\n")
    with ledger_gz_path.open("xb") as raw_gf:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_gf, compresslevel=9, mtime=0) as gf:
            gf.write(ledger_path.read_bytes())
    summary = summarize_canonical_ledger_and_elo(all_rows, 69, 1000)
    extra = {"aggregation_mode": "disjoint_multi_run_union", "source_runs": src_manifests, "duplicate_logical_dataset_level_cells": 0, "total_source_wall_seconds": round(sum(float(x["elapsed_wall_seconds"]) for x in src_manifests), 2), "aggregation_elapsed_wall_seconds": round(time.perf_counter() - t0, 2)}
    receipt = _build_receipt_common(out_run_id, run_dir, ledger_path, ledger_gz_path, summary, extra)
    with receipt_path.open("x", encoding="utf-8") as rf:
        rf.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser()
    for flag, tp, dflt in (("--run-id", str, None), ("--shard-id", str, None), ("--dataset-ids", str, None), ("--aggregate-runs", str, None), ("--max-v2-slice", int, 80), ("--workers", int, 4), ("--wall-budget-s", float, 840.0)):
        p.add_argument(flag, type=tp, default=dflt)
    args = p.parse_args()
    if args.aggregate_runs:
        res = aggregate_canonical_runs([x.strip() for x in args.aggregate_runs.split(",") if x.strip()], args.run_id or "canonical_v3_full_1069_unified_v1")
    else:
        ds_ids = [x.strip() for x in args.dataset_ids.split(",") if x.strip()] if args.dataset_ids else None
        res = run_bounded_canonical_eval(args.max_v2_slice, args.workers, args.wall_budget_s, args.run_id, args.shard_id, ds_ids)
    print(json.dumps(res["coverage"], indent=2))


if __name__ == "__main__":
    main()
