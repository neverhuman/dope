"""Bounded Frozen-Checkpoint Per-Dataset Ledger & Targeted Recovery Pass for DOPE-MLC-v3.

Evaluates the frozen A3 checkpoint (dope_mlc_v3_multitask_best.pt, SHA-256 a36510b9...)
without retraining or modifying any initial task-2354 or v2 receipts. Writes durable
per-dataset/per-tier/per-auditor rows immediately to disk with explicit provenance,
unclipped and clipped retention R, denominator floor/clipping flags, wire SHA-256s,
sender-scaled vs. isolated-receiver CSC1 raw-domain inverse qualification, and timing costs.
"""
from __future__ import annotations

import csv, gzip, hashlib, json, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_env_k, "1")

import numpy as np
import torch
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .budget_planner import (
    TIER_BYTE_CAPS, deserialize_dope_v3_envelope,
    pack_compact_scaler_sidecar, serialize_dope_v3_envelope,
)
from .multilevel_model import MAX_COLS, MultiLevelLandmarkModel
from .scaling import FitOnlyTableTransform
from .v3_ablation_and_eval import (
    ALL_TIERS, _enrich_reg_bin_cell, _synthesize_report_from_train, find_local_blob,
)

EXPECTED_TASK2354_RECEIPT_SHA = "ce58d6b8e1bc5f837236cf30bfc33cd889e2d8177ae07924d2e986e0772204b8"
EXPECTED_FROZEN_A3_CKPT_SHA = "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952"
EXPECTED_TASK2354_SOURCE_SHA = "cc28c6a9f6ba8075e7c837449be8e1bb88e60a75372cc5998c689ee4ec3cdef5"
EXPECTED_MANIFEST_SHA = "66d2a63c3e4b84e69a554af212a2f9f63081d59f366cd1f4b796bfc7eea39bdd"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_arrays(*arrs: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrs:
        c = np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(c.shape).encode("utf-8"))
        h.update(c.tobytes())
    return h.hexdigest()


def _compute_retention_stats(real_score: float, syn_score: float, chance: float, floor: float = 0.03) -> Dict[str, Any]:
    raw_denom = float(real_score - chance)
    floor_applied = bool(raw_denom < floor)
    denom = max(raw_denom, floor)
    unclipped = float((syn_score - chance) / denom)
    return {
        "real_score": round(float(real_score), 6), "syn_score": round(float(syn_score), 6),
        "chance_baseline": round(float(chance), 6), "raw_denom": round(raw_denom, 6),
        "denom_floor": floor, "denom_floor_applied": floor_applied,
        "unclipped_R": round(unclipped, 6), "clipped_R": round(float(np.clip(unclipped, 0.0, 1.25)), 6),
        "clip_low_applied": bool(unclipped < 0.0), "clip_high_applied": bool(unclipped > 1.25),
    }


def _mc_roster(n_rows: int) -> List[Tuple[str, Any]]:
    n_n = min(5, max(1, n_rows - 1))
    return [
        ("RandomForest40_d8", RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42, n_jobs=1)),
        ("LogisticL2_iter200", LogisticRegression(max_iter=200)),
        ("HistGBR_MC", HistGradientBoostingClassifier(max_iter=40, max_depth=5, random_state=42)),
        ("ExtraTrees32_d8", ExtraTreesClassifier(n_estimators=32, max_depth=8, random_state=42, n_jobs=1)),
        ("KNN5_dist", KNeighborsClassifier(n_neighbors=n_n, weights="distance", n_jobs=1)),
        ("RidgeClassifier_L2", RidgeClassifier(alpha=1.0)),
    ]


def evaluate_multiclass_auditors(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, k_classes: int) -> Dict[str, Any]:
    y_tr_i = np.clip(np.rint(y_tr).astype(int), 0, max(k_classes - 1, 1))
    y_te_i = np.clip(np.rint(y_te).astype(int), 0, max(k_classes - 1, 1))
    chance = 1.0 / float(max(k_classes, 2))
    per_auditor, failures = {}, {}
    for name, clf in _mc_roster(X_tr.shape[0]):
        try:
            clf.fit(X_tr, y_tr_i)
            r_acc = float(np.mean(clf.predict(X_te) == y_te_i))
        except Exception as ex:
            r_acc, failures[f"{name}_real"] = chance, type(ex).__name__
        per_auditor[name] = {"real_acc": round(r_acc, 6)}
    return {"per_auditor": per_auditor, "failures": failures, "chance": chance, "u_tr_count": int(np.unique(y_tr_i).size)}


def score_multiclass_syn_against_real(
    real_info: Dict[str, Any], X_syn: np.ndarray, y_syn: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, k_classes: int,
) -> Dict[str, Any]:
    y_syn_i = np.clip(np.rint(y_syn).astype(int), 0, max(k_classes - 1, 1))
    y_te_i = np.clip(np.rint(y_te).astype(int), 0, max(k_classes - 1, 1))
    chance = float(real_info["chance"])
    per_aud: Dict[str, Any] = {}
    failures = dict(real_info["failures"])
    u_syn = np.unique(y_syn_i)
    for name, clf in _mc_roster(X_syn.shape[0]):
        r_acc = float(real_info["per_auditor"][name]["real_acc"])
        if u_syn.size < 2:
            s_acc = float(np.mean(y_te_i == (int(u_syn[0]) if u_syn.size else 0)))
            failures[f"{name}_syn"] = "single_class_synthetic"
        else:
            try:
                clf.fit(X_syn, y_syn_i)
                s_acc = float(np.mean(clf.predict(X_te) == y_te_i))
            except Exception as ex:
                s_acc, failures[f"{name}_syn"] = chance, type(ex).__name__
        per_aud[name] = _compute_retention_stats(r_acc, s_acc, chance, floor=0.03)

    r_2 = 0.5 * (per_aud["RandomForest40_d8"]["real_score"] + per_aud["LogisticL2_iter200"]["real_score"])
    s_2 = 0.5 * (per_aud["RandomForest40_d8"]["syn_score"] + per_aud["LogisticL2_iter200"]["syn_score"])
    r_6 = float(np.mean([v["real_score"] for v in per_aud.values()]))
    s_6 = float(np.mean([v["syn_score"] for v in per_aud.values()]))
    return {
        "per_auditor": per_aud,
        "task2354_2aud_rf_lr": _compute_retention_stats(r_2, s_2, chance, floor=0.03),
        "sklearn_6aud_probe": _compute_retention_stats(r_6, s_6, chance, floor=0.03),
        "failures": failures,
        "synthetic_unique_classes": int(u_syn.size),
        "class_support_coverage": round(float(u_syn.size) / float(max(k_classes, 1)), 6),
    }


def evaluate_reg_bin_dev_cell(
    X_tr: np.ndarray, y_tr: np.ndarray, X_syn: np.ndarray, y_syn: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray, task: str,
) -> Dict[str, Any]:
    n_n = min(5, max(1, X_tr.shape[0] - 1))
    per_aud, failures = {}, {}
    if task == "regression":
        chance, fl = 0.0, 0.05
        roster = [
            ("HistGBR_Reg", HistGradientBoostingRegressor(max_iter=40, max_depth=5, random_state=42)),
            ("RandomForest32_Reg", RandomForestRegressor(n_estimators=32, max_depth=8, random_state=42, n_jobs=1)),
            ("ExtraTrees32_Reg", ExtraTreesRegressor(n_estimators=32, max_depth=8, random_state=42, n_jobs=1)),
            ("Ridge_L2", Ridge(alpha=1.0)),
            ("KNN5_Reg", KNeighborsRegressor(n_neighbors=n_n, weights="distance", n_jobs=1)),
            ("Ridge_Strong_L2", Ridge(alpha=10.0)),
        ]
        for name, reg in roster:
            try:
                r_sc = float(np.clip(r2_score(y_te, type(reg)(**reg.get_params()).fit(X_tr, y_tr).predict(X_te)), -1.0, 1.0))
            except Exception as ex:
                r_sc, failures[f"{name}_real"] = 0.0, type(ex).__name__
            try:
                s_sc = float(np.clip(r2_score(y_te, type(reg)(**reg.get_params()).fit(X_syn, y_syn).predict(X_te)), -1.0, 1.0))
            except Exception as ex:
                s_sc, failures[f"{name}_syn"] = 0.0, type(ex).__name__
            per_aud[name] = _compute_retention_stats(max(r_sc, 0.0), max(s_sc, 0.0), chance, floor=fl)
    else:
        chance, fl = 0.5, 0.03
        y_tr_b, y_te_b, y_syn_b = (y_tr >= 0.5).astype(int), (y_te >= 0.5).astype(int), (y_syn >= 0.5).astype(int)
        for name, clf in _mc_roster(X_tr.shape[0]):
            try:
                c_r = type(clf)(**clf.get_params()).fit(X_tr, y_tr_b)
                sc_r = c_r.decision_function(X_te) if hasattr(c_r, "decision_function") else c_r.predict_proba(X_te)[:, 1]
                r_sc = float(roc_auc_score(y_te_b, sc_r)) if len(np.unique(y_te_b)) > 1 else 0.5
            except Exception as ex:
                r_sc, failures[f"{name}_real"] = 0.5, type(ex).__name__
            try:
                if len(np.unique(y_syn_b)) < 2:
                    s_sc, failures[f"{name}_syn"] = 0.5, "single_class_synthetic"
                else:
                    c_s = type(clf)(**clf.get_params()).fit(X_syn, y_syn_b)
                    sc_s = c_s.decision_function(X_te) if hasattr(c_s, "decision_function") else c_s.predict_proba(X_te)[:, 1]
                    s_sc = float(roc_auc_score(y_te_b, sc_s)) if len(np.unique(y_te_b)) > 1 else 0.5
            except Exception as ex:
                s_sc, failures[f"{name}_syn"] = 0.5, type(ex).__name__
            per_aud[name] = _compute_retention_stats(r_sc, s_sc, chance, floor=fl)

    r_6 = float(np.mean([v["real_score"] for v in per_aud.values()]))
    s_6 = float(np.mean([v["syn_score"] for v in per_aud.values()]))
    return {"per_auditor": per_aud, "sklearn_6aud_probe": _compute_retention_stats(r_6, s_6, chance, floor=fl), "failures": failures}


def _load_single_mc_cell(e: Dict[str, Any], *, targeted_recovery: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    tr_p, te_p = find_local_blob(e["train_sha256"]), find_local_blob(e["test_sha256"])
    if not tr_p or not te_p:
        return None, "missing_local_blob", ""
    with tr_p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        tr_rows = list(csv.reader(f))
    with te_p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        te_rows = list(csv.reader(f))
    if e["has_header"] and tr_rows:
        tr_rows = tr_rows[1:]
    if e["has_header"] and te_rows:
        te_rows = te_rows[1:]
    arr_tr, arr_te = np.asarray(tr_rows, dtype=np.float64), np.asarray(te_rows, dtype=np.float64)
    orig_sha = _sha256_arrays(arr_tr, arr_te)

    min_rows = 16 if targeted_recovery else 20
    if arr_tr.ndim != 2 or arr_tr.shape[0] < min_rows or arr_tr.shape[1] < 2:
        return None, f"train_rows_{arr_tr.shape[0]}_lt_{min_rows}", orig_sha
    if arr_tr.shape[1] > MAX_COLS + 1:
        arr_tr = np.column_stack([arr_tr[:, :MAX_COLS], arr_tr[:, -1]])
        arr_te = np.column_stack([arr_te[:, :MAX_COLS], arr_te[:, -1]])

    rng = np.random.default_rng(int(hashlib.sha256(e["dataset_id"].encode()).hexdigest()[:8], 16))
    n_tr = arr_tr.shape[0]
    perm = rng.permutation(n_tr)
    if targeted_recovery:
        _, first_pos = np.unique(arr_tr[perm, -1], return_index=True)
        seed_idx = perm[np.sort(first_pos)]
        perm_use = np.concatenate([seed_idx, np.setdiff1d(perm, seed_idx, assume_unique=True)])
        n_fit = min(max(int(0.75 * n_tr), max(12, len(seed_idx) + 4)), 384)
        fit_idx, qry_idx = perm_use[:n_fit], perm_use[n_fit : n_fit + 160]
        if qry_idx.size < 4:
            qry_idx = perm_use[-min(12, max(4, n_tr // 3)) :]
    else:
        n_fit = max(int(0.75 * n_tr), 16)
        fit_idx, qry_idx = perm[: min(n_fit, 384)], perm[n_fit : n_fit + 160]
        if qry_idx.size < 6:
            qry_idx = perm[-min(16, n_tr // 3) :]

    X_fit_raw, y_fit_raw = arr_tr[fit_idx, :-1], arr_tr[fit_idx, -1]
    X_qry_raw, y_qry_raw = arr_tr[qry_idx, :-1], arr_tr[qry_idx, -1]
    scaler = FitOnlyTableTransform.fit(X_fit_raw, y_fit_raw, task="multiclass")
    k_cls = len(scaler.target_scaler.class_labels) if scaler.target_scaler else 0
    if k_cls < 3 or X_fit_raw.shape[1] == 0:
        return None, f"sampled_fit_classes_{k_cls}_lt_3_unstratified", orig_sha

    p_use = min(X_fit_raw.shape[1], MAX_COLS)
    X_fit, y_fit = scaler.transform(X_fit_raw[:, :p_use], y_fit_raw)
    X_qry, y_qry = scaler.transform(X_qry_raw[:, :p_use], y_qry_raw)
    X_fit = np.clip(np.nan_to_num(X_fit, nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0).astype(np.float32)
    X_qry = np.clip(np.nan_to_num(X_qry, nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0).astype(np.float32)
    rep = _synthesize_report_from_train(X_fit, y_fit, "multiclass")
    fit_sha = _sha256_arrays(X_fit_raw[:, :p_use], y_fit_raw, arr_te[:256, :p_use], arr_te[:256, -1])
    return {
        "dataset_id": e["dataset_id"], "v3_role": e["v3_role"], "prior_exposure": e["prior_exposure"],
        "task": "multiclass", "k_classes": k_cls, "X_fit": X_fit, "y_fit": y_fit.astype(np.float32),
        "X_qry": X_qry, "y_qry": y_qry.astype(np.float32),
        "X_fit_raw": X_fit_raw[:, :p_use], "y_fit_raw": y_fit_raw,
        "X_test_raw": arr_te[:256, :p_use], "y_test_raw": arr_te[:256, -1],
        "report": rep, "orig_table_sha256": orig_sha, "fit_eval_split_sha256": fit_sha,
    }, None, orig_sha


def run_frozen_checkpoint_ledger_replay(
    output_dir: Path, *, max_wall_seconds: float = 1140.0, seed: int = 20261024,
) -> Dict[str, Any]:
    torch.set_num_threads(1)
    t_start = time.time()
    base_dir = Path("/home/ubuntu/dope/target/s3-multilevel-training-20261023")
    rec_2354_path = output_dir / "v3_dev_val_ablation_and_multitask_receipt.json"
    ckpt_a3_path = output_dir / "dope_mlc_v3_multitask_best.pt"
    manifest_path = output_dir / "v3_corpus_exposure_and_split_manifest.json"
    rec_sha, ckpt_sha, man_sha = _sha256_file(rec_2354_path), _sha256_file(ckpt_a3_path), _sha256_file(manifest_path)
    src_2354_sha = _sha256_file(Path(__file__).parent / "v3_ablation_and_eval.py")
    src_self_sha = _sha256_file(Path(__file__))
    if rec_sha != EXPECTED_TASK2354_RECEIPT_SHA or ckpt_sha != EXPECTED_FROZEN_A3_CKPT_SHA:
        raise RuntimeError(f"Immutable task-2354 artifact hash mismatch: rec={rec_sha}, ckpt={ckpt_sha}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ckpt_data = torch.load(ckpt_a3_path, map_location="cpu", weights_only=False)
    sender = MultiLevelLandmarkModel().eval()
    sender.load_state_dict(ckpt_data["model_state_dict"], strict=True)
    receiver = MultiLevelLandmarkModel().eval()
    receiver.load_state_dict(ckpt_data["model_state_dict"], strict=True)

    ledger_jsonl_path = output_dir / "v3_per_dataset_multitask_ledger.jsonl"
    existing_keys: set = set()
    rows_written: List[Dict[str, Any]] = []
    if ledger_jsonl_path.exists():
        with ledger_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows_written.append(r)
                    existing_keys.add((r["dataset_id"], r.get("tier"), r.get("record_type")))

    def _append_row(row: Dict[str, Any]) -> None:
        key = (row["dataset_id"], row.get("tier"), row.get("record_type"))
        if key in existing_keys:
            return
        with ledger_jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
            fp.flush()
        existing_keys.add(key)
        rows_written.append(row)

    mc_entries = [
        e for e in manifest["entries"]
        if e["catalog_task"] == "multi_label" and e["v3_role"] in ("eval_gradient_disjoint_multiclass_panel", "dev_val_ablation", "train_gradient")
    ]
    stopped_by_wall_budget = False
    for e in mc_entries:
        if time.time() - t_start > max_wall_seconds:
            stopped_by_wall_budget = True
            break
        cell_orig, skip_reason, orig_sha = _load_single_mc_cell(e, targeted_recovery=False)
        if cell_orig is None:
            _append_row({
                "record_type": "task2354_skip_record", "measurement_provenance": "frozen_checkpoint_replay",
                "measured_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_id": e["dataset_id"],
                "v3_role": e["v3_role"], "prior_exposure": e["prior_exposure"], "task": "multiclass",
                "tier": "all", "included_in_initial_task2354": False, "skip_reason_in_task2354": skip_reason,
                "original_table_sha256": orig_sha, "frozen_a3_checkpoint_sha256": ckpt_sha,
                "task2354_receipt_sha256": rec_sha, "replay_source_sha256": src_self_sha,
            })
            if e["v3_role"] == "train_gradient":
                continue
            cell_use, _, _ = _load_single_mc_cell(e, targeted_recovery=True)
            if cell_use is None:
                continue
            inc_2354, rec_mode, rec_type = False, f"targeted_recovery_after_{skip_reason}", "targeted_recovery_replay"
        else:
            if e["v3_role"] == "train_gradient":
                continue
            cell_use = cell_orig
            inc_2354, rec_mode, rec_type = True, "initial_task2354_exact_cell_and_seed", "frozen_checkpoint_replay"

        t_prep0 = time.perf_counter()
        X_f_raw, y_f_raw, X_te_raw, y_te_raw = cell_use["X_fit_raw"], cell_use["y_fit_raw"], cell_use["X_test_raw"], cell_use["y_test_raw"]
        scaler = FitOnlyTableTransform.fit(X_f_raw, y_f_raw, task="multiclass")
        X_f_s, y_f_s = scaler.transform(X_f_raw, y_f_raw)
        X_te_s, y_te_s = scaler.transform(X_te_raw, y_te_raw)
        prep_ms = (time.perf_counter() - t_prep0) * 1000.0
        real_scaled_info = evaluate_multiclass_auditors(X_f_s, y_f_s, X_te_s, y_te_s, cell_use["k_classes"])
        real_raw_info = evaluate_multiclass_auditors(X_f_raw, y_f_s, X_te_raw, y_te_s, cell_use["k_classes"])

        for lvl in ALL_TIERS:
            if (e["dataset_id"], lvl, rec_type) in existing_keys:
                continue
            t_enc0 = time.perf_counter()
            enc = sender.encode_table(X_f_s, y_f_s, cell_use["report"], task="multiclass", level=lvl, adaptive_budget=True, include_v3_primitives=True)
            enc_ms = (time.perf_counter() - t_enc0) * 1000.0
            t_ser0 = time.perf_counter()
            sc_bytes = pack_compact_scaler_sidecar(scaler, X_f_s.shape[1])
            wire = serialize_dope_v3_envelope(enc, scaler=scaler, compress=True)
            ser_ms = (time.perf_counter() - t_ser0) * 1000.0
            wire_sha = hashlib.sha256(wire).hexdigest()
            t_des0 = time.perf_counter()
            dec_enc, dec_scaler = deserialize_dope_v3_envelope(wire)
            des_ms = (time.perf_counter() - t_des0) * 1000.0
            t_gen0 = time.perf_counter()
            X_syn_s, y_syn_idx = receiver.generate_synthetic(dec_enc, n_rows=X_f_s.shape[0], seed=seed)
            gen_ms = (time.perf_counter() - t_gen0) * 1000.0
            t_aud0 = time.perf_counter()
            scaled_eval = score_multiclass_syn_against_real(real_scaled_info, X_syn_s, y_syn_idx, X_te_s, y_te_s, cell_use["k_classes"])
            assert dec_scaler is not None
            X_syn_raw, y_syn_raw = dec_scaler.inverse_transform(X_syn_s, y_syn_idx)
            X_te_rec_s, y_te_rec_idx = dec_scaler.transform(X_te_raw, y_te_raw)
            csc1_max_diff = float(np.max(np.abs(X_te_s - X_te_rec_s)))
            y_syn_rec_idx = dec_scaler.transform_target(y_syn_raw)
            raw_eval = score_multiclass_syn_against_real(real_raw_info, np.asarray(X_syn_raw, dtype=np.float64), y_syn_rec_idx, X_te_raw, y_te_rec_idx, cell_use["k_classes"])
            aud_ms = (time.perf_counter() - t_aud0) * 1000.0
            _append_row({
                "record_type": rec_type, "measurement_provenance": "frozen_checkpoint_replay", "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_id": e["dataset_id"], "v3_role": e["v3_role"], "prior_exposure": e["prior_exposure"], "task": "multiclass", "k_classes": cell_use["k_classes"],
                "n_fit": int(X_f_s.shape[0]), "n_test": int(X_te_s.shape[0]), "p_cols": int(X_f_s.shape[1]), "included_in_initial_task2354": inc_2354,
                "included_in_initial_task2354_meaning": "same_logical_cell_and_seed_evaluated_against_frozen_a3_checkpoint_not_stored_per_dataset_in_task2354",
                "skip_reason_in_task2354": skip_reason, "recovery_mode": rec_mode, "original_table_sha256": orig_sha, "fit_eval_split_sha256": cell_use["fit_eval_split_sha256"],
                "tier": lvl, "cap_bytes": TIER_BYTE_CAPS[lvl], "complete_wire_bytes": len(wire), "scaler_sidecar_bytes": len(sc_bytes), "inner_payload_bytes": len(wire) - len(sc_bytes) - 10,
                "wire_sha256": wire_sha, "within_cap": bool(len(wire) <= TIER_BYTE_CAPS[lvl]), "decoded_envelope_magic": wire[:4].decode("ascii", errors="replace"),
                "shared_decoder": "MultiLevelLandmarkModel", "shared_scaler": "FitOnlyTableTransform/CSC1", "receiver_csc1_max_test_transform_abs_diff": round(csc1_max_diff, 6),
                "costs_ms": {"preprocess_fit_ms": round(prep_ms, 3), "encode_ms": round(enc_ms, 3), "serialize_ms": round(ser_ms, 3), "deserialize_ms": round(des_ms, 3), "generate_ms": round(gen_ms, 3), "auditor_eval_ms": round(aud_ms, 3)},
                "sender_scaled_test_domain": scaled_eval, "isolated_receiver_csc1_raw_domain": raw_eval, "frozen_a3_checkpoint_sha256": ckpt_sha,
                "task2354_receipt_sha256": rec_sha, "task2354_source_sha256": src_2354_sha, "replay_source_sha256": src_self_sha, "manifest_sha256": man_sha,
            })

    raw_cells = torch.load(base_dir / "s3_prepared_splits_cache.pt", map_location="cpu", weights_only=False)
    role_by_id = {e["dataset_id"]: e for e in manifest["entries"] if e["v3_role"] == "dev_val_ablation" and e["catalog_task"] in ("regression", "binary")}
    by_did: Dict[str, List[Dict[str, Any]]] = {}
    for c in raw_cells:
        if c["dataset_id"] in role_by_id:
            by_did.setdefault(c["dataset_id"], []).append(c)

    for did, e in sorted(role_by_id.items()):
        if time.time() - t_start > max_wall_seconds:
            stopped_by_wall_budget = True
            break
        cands = by_did.get(did, [])
        fit_iid = [c for c in cands if c["split"] == "fit-iid-65-v1"]
        if not fit_iid:
            _append_row({
                "record_type": "task2354_skip_record", "measurement_provenance": "frozen_checkpoint_replay", "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_id": did, "v3_role": "dev_val_ablation", "prior_exposure": e["prior_exposure"], "task": e["model_task"], "tier": "all",
                "included_in_initial_task2354": False, "skip_reason_in_task2354": "no_fit_iid_65_v1_cell_in_s3_cache_only_outer5_folds",
                "available_splits": [c["split"] for c in cands], "frozen_a3_checkpoint_sha256": ckpt_sha, "task2354_receipt_sha256": rec_sha, "replay_source_sha256": src_self_sha,
            })
            if not cands:
                continue
            c_use, inc_2354, rec_mode, rec_type, skip_r = cands[0], False, f"targeted_recovery_split_{cands[0]['split']}", "targeted_recovery_replay", "no_fit_iid_65_v1_cell_in_s3_cache_only_outer5_folds"
        else:
            c_use, inc_2354, rec_mode, rec_type, skip_r = fit_iid[0], True, "initial_task2354_exact_cell_and_seed", "frozen_checkpoint_replay", None

        c_enr = _enrich_reg_bin_cell(c_use)
        X_f, y_f = np.asarray(c_enr["X_fit"], dtype=np.float64), np.asarray(c_enr["y_fit"], dtype=np.float64)
        X_q, y_q = np.asarray(c_enr["X_qry"], dtype=np.float64), np.asarray(c_enr["y_qry"], dtype=np.float64)
        split_sha = _sha256_arrays(X_f, y_f, X_q, y_q)
        scaler = FitOnlyTableTransform.fit(X_f, y_f, task=c_enr["task"])

        for lvl in ALL_TIERS:
            if (did, lvl, rec_type) in existing_keys:
                continue
            t_enc0 = time.perf_counter()
            enc = sender.encode_table(X_f, y_f, c_enr["report"], task=c_enr["task"], level=lvl, adaptive_budget=True, include_v3_primitives=True)
            enc_ms = (time.perf_counter() - t_enc0) * 1000.0
            sc_bytes = pack_compact_scaler_sidecar(scaler, X_f.shape[1])
            wire = serialize_dope_v3_envelope(enc, scaler=scaler, compress=True)
            wire_sha = hashlib.sha256(wire).hexdigest()
            dec_enc, _ = deserialize_dope_v3_envelope(wire)
            X_syn, y_syn = receiver.generate_synthetic(dec_enc, n_rows=X_f.shape[0], seed=seed)
            aud_eval = evaluate_reg_bin_dev_cell(X_f, y_f, X_syn, y_syn, X_q, y_q, c_enr["task"])
            _append_row({
                "record_type": rec_type, "measurement_provenance": "frozen_checkpoint_replay", "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_id": did, "v3_role": "dev_val_ablation", "prior_exposure": e["prior_exposure"], "task": c_enr["task"], "split_used": c_use["split"],
                "n_fit": int(X_f.shape[0]), "n_test": int(X_q.shape[0]), "p_cols": int(X_f.shape[1]), "included_in_initial_task2354": inc_2354,
                "included_in_initial_task2354_meaning": "same_logical_cell_and_seed_evaluated_against_frozen_a3_checkpoint_not_stored_per_dataset_in_task2354",
                "skip_reason_in_task2354": skip_r, "recovery_mode": rec_mode, "fit_eval_split_sha256": split_sha, "tier": lvl, "cap_bytes": TIER_BYTE_CAPS[lvl],
                "complete_wire_bytes": len(wire), "scaler_sidecar_bytes": len(sc_bytes), "inner_payload_bytes": len(wire) - len(sc_bytes) - 10,
                "wire_sha256": wire_sha, "within_cap": bool(len(wire) <= TIER_BYTE_CAPS[lvl]), "decoded_envelope_magic": "DML3", "shared_decoder": "MultiLevelLandmarkModel",
                "shared_scaler": "FitOnlyTableTransform/CSC1", "encode_ms": round(enc_ms, 3), "sender_scaled_test_domain": aud_eval,
                "frozen_a3_checkpoint_sha256": ckpt_sha, "task2354_receipt_sha256": rec_sha, "task2354_source_sha256": src_2354_sha, "replay_source_sha256": src_self_sha, "manifest_sha256": man_sha,
            })

    gz_path = output_dir / "v3_per_dataset_multitask_ledger.jsonl.gz"
    with ledger_jsonl_path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        f_out.writelines(f_in)
    gz_sha = _sha256_file(gz_path)

    mc_68_rows = [r for r in rows_written if r["v3_role"] == "eval_gradient_disjoint_multiclass_panel" and r["record_type"] == "frozen_checkpoint_replay"]
    mc_rec_rows = [r for r in rows_written if r["v3_role"] == "eval_gradient_disjoint_multiclass_panel" and r["record_type"] == "targeted_recovery_replay"]
    mc_tier_summary = {}
    for lvl in ALL_TIERS:
        sub = [r for r in mc_68_rows if r["tier"] == lvl]
        sub_all69 = sub + [r for r in mc_rec_rows if r["tier"] == lvl]
        if not sub:
            continue
        mc_tier_summary[lvl] = {
            "initial_68_datasets_count": len(sub), "with_targeted_recovery_69_count": len(sub_all69),
            "cap_bytes": TIER_BYTE_CAPS[lvl],
            "mean_complete_wire_bytes_68": round(float(np.mean([r["complete_wire_bytes"] for r in sub])), 1),
            "max_complete_wire_bytes_68": int(np.max([r["complete_wire_bytes"] for r in sub])),
            "pct_within_cap_69": round(100.0 * float(np.mean([r["within_cap"] for r in sub_all69])), 2),
            "task2354_2aud_clipped_mean_R_68": round(float(np.mean([r["sender_scaled_test_domain"]["task2354_2aud_rf_lr"]["clipped_R"] for r in sub])), 4),
            "task2354_2aud_unclipped_mean_R_68": round(float(np.mean([r["sender_scaled_test_domain"]["task2354_2aud_rf_lr"]["unclipped_R"] for r in sub])), 4),
            "task2354_2aud_denom_floor_count_68": sum(1 for r in sub if r["sender_scaled_test_domain"]["task2354_2aud_rf_lr"]["denom_floor_applied"]),
            "task2354_2aud_clip_high_count_68": sum(1 for r in sub if r["sender_scaled_test_domain"]["task2354_2aud_rf_lr"]["clip_high_applied"]),
            "sklearn_6aud_clipped_mean_R_68": round(float(np.mean([r["sender_scaled_test_domain"]["sklearn_6aud_probe"]["clipped_R"] for r in sub])), 4),
            "sklearn_6aud_unclipped_mean_R_68": round(float(np.mean([r["sender_scaled_test_domain"]["sklearn_6aud_probe"]["unclipped_R"] for r in sub])), 4),
            "sklearn_6aud_clipped_mean_R_all69": round(float(np.mean([r["sender_scaled_test_domain"]["sklearn_6aud_probe"]["clipped_R"] for r in sub_all69])), 4),
            "isolated_receiver_raw_domain_6aud_clipped_mean_R_68": round(float(np.mean([r["isolated_receiver_csc1_raw_domain"]["sklearn_6aud_probe"]["clipped_R"] for r in sub])), 4),
            "isolated_receiver_raw_domain_6aud_unclipped_mean_R_68": round(float(np.mean([r["isolated_receiver_csc1_raw_domain"]["sklearn_6aud_probe"]["unclipped_R"] for r in sub])), 4),
            "max_csc1_test_transform_abs_diff_69": round(float(np.max([r["receiver_csc1_max_test_transform_abs_diff"] for r in sub_all69])), 6),
        }

    summary_receipt = {
        "schema": "dope.v3.frozen_checkpoint_ledger_receipt/v1",
        "study_id": "dope-mlc-v3-frozen-checkpoint-ledger-replay-20261024",
        "measurement_provenance": "frozen_checkpoint_replay",
        "stopped_by_wall_budget": stopped_by_wall_budget,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "cryptographic_bindings": {
            "initial_task2354_receipt_sha256": rec_sha, "frozen_a3_checkpoint_sha256": ckpt_sha,
            "task2354_source_sha256": src_2354_sha, "replay_source_sha256": src_self_sha,
            "manifest_sha256": man_sha, "ledger_jsonl_gz_sha256": gz_sha,
        },
        "total_ledger_rows": len(rows_written),
        "skip_records": [r for r in rows_written if r["record_type"] == "task2354_skip_record"],
        "heldout_multiclass_ledger_summary": mc_tier_summary,
        "open_unmeasured_obligations": {
            "v2_canonical_gbdt_mlp_attention_6aud_roster_on_v3": "OPEN_unmeasured (this bounded ledger uses a 6-auditor scikit-learn roster + task-2354 2-auditor pair, not v2's CatBoost/LightGBM/XGBoost/MLP/Linear/LocalAttention roster)",
            "v3_full_1000_panel_transfer_elo_and_driver_fidelity": "OPEN_unmeasured (the 1,000-dataset v2 panel was evaluated under v2 in strata_1000_audit_receipt_v4_exact_wire_replay.json, not re-evaluated under v3)",
            "source_family_disjointness_audit": "OPEN_unverified_beyond_exact_sha256 (0 cross-role train_sha256/canonical_sha256 overlaps verified; upstream benchmark-family provenance is not encoded in catalog.jsonl)",
        },
    }
    (output_dir / "v3_multitask_ledger_receipt.json").write_text(json.dumps(summary_receipt, indent=2), encoding="utf-8")
    return summary_receipt


if __name__ == "__main__":
    out_dir = Path("/home/ubuntu/dope/target/s3-multilevel-training-20261023/v3")
    print(json.dumps(run_frozen_checkpoint_ledger_replay(out_dir), indent=2))
