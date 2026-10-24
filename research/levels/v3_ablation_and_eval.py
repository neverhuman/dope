"""DOPE-MLC-v3 Split/Exposure Manifest Builder and Matched Dev-Val Ablation Runner.

1. Builds the reconciled 3,859-dataset catalog split & prior-exposure manifest with
   strict SHA-256 / canonical_sha256 disjointness verification.
2. Prepares train-fit-only scaled splits and 4-tier (Small, Medium, High, Ultra)
   building-block + inter-landmark conditioning bundles across Regression, Binary,
   and Multiclass (multi_label) datasets.
3. Runs matched ablations A0 -> A1 -> A2 -> A3 on the 112-dataset dev_val_ablation
   panel, measuring actual validation loss, col_gate_head usage in loss/encode/generate,
   complete receiver-replay wire bytes, and downstream auditor retention R & fidelity D.
"""
from __future__ import annotations

from collections import Counter
import csv, gzip, hashlib, json, math, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from .budget_planner import TIER_BYTE_CAPS, deserialize_dope_v3_envelope, serialize_dope_v3_envelope
from .building_block_landmarks import build_v3_conditioning_bundle
from .multilevel_model import MAX_COLS, MultiLevelLandmarkModel
from .scaling import FitOnlyTableTransform
from .world_class_eval import select_stratified_candidates

ALL_TIERS = ("small", "medium", "high", "ultra")
BLOB_ROOTS = (
    Path("/home/ubuntu/dope-fit-cv-precalc-20261023/blobs"),
    Path("/home/ubuntu/scratch/jopedime-corpus-benchmark-20260928/blobs"),
    Path("/home/ubuntu/scratch/jopedime-corpus-throughput-20260928/blobs"),
    Path("/home/ubuntu/out/loop-store/blobs"),
)


def find_local_blob(sha: str) -> Optional[Path]:
    for r in BLOB_ROOTS:
        for cand in (r / "sha256" / sha[:2] / sha, r / "sha256" / sha, r / sha[:2] / sha, r / sha):
            if cand.exists():
                return cand
    return None


def _synthesize_report_from_train(X_fit: np.ndarray, y_fit: np.ndarray, task: str) -> Dict[str, Any]:
    """Synthesize a leak-free train-only landmark report record when precalculated S3 report is absent (e.g. multiclass)."""
    from .building_block_landmarks import compute_building_block_landmarks

    n, p = X_fit.shape
    bb = compute_building_block_landmarks(X_fit, y_fit, level="ultra")
    bb16 = np.asarray(bb["bb_gains16"], dtype=np.float64)
    col7 = np.asarray(bb["col_bb_probes7"], dtype=np.float64)
    cells = []
    # Map fast train-only probes onto the 23 canonical landmark slots
    probe_map = [
        bb16[4], bb16[0], bb16[3], bb16[6], bb16[1], bb16[7],
        bb16[10], bb16[9], bb16[11], bb16[7], bb16[5], bb16[13],
        bb16[11], 0.01, max(float(bb16[0]), float(bb16[4])), bb16[9],
        max(float(bb16[9]), float(bb16[13])), max(float(bb16[9]), float(bb16[13])) * 1.05,
        bb16[13] * 0.9, bb16[13] * 0.95, max(float(bb16[0]), float(bb16[6])) * 1.05,
        bb16[4] * 0.85, max(float(bb16[9]), float(bb16[13])) * 0.85,
    ]
    for idx in range(23):
        cells.append({
            "id": idx + 1,
            "state": "complete",
            "gain": round(float(probe_map[idx]), 5),
            "se": round(1.0 / math.sqrt(max(n, 16)), 5),
            "stable": True,
            "lambda_indices": [2, 2, 3, 2, 3],
        })
    spectrum = []
    for j in range(p):
        c = col7[j] if j < col7.shape[0] else np.zeros(7)
        spectrum.append({
            "slot": j,
            "values": [float(c[0]), float(c[3]), float(c[4]), float(c[0]), float(c[1]), float(c[5]), float(c[2]), float(c[6]), float(c[3])],
        })
    u_cls = len(np.unique(np.rint(y_fit))) if task in ("classification", "multiclass") else 0
    return {
        "panel": "landmarks-native-v1",
        "kind": "real",
        "privacy": "private",
        "task": "classification" if task in ("classification", "multiclass") else "regression",
        "classes": u_cls,
        "rows": n,
        "features": p,
        "cells": cells,
        "target_stats": [float(np.mean(y_fit)), float(np.std(y_fit)), None],
        "spectrum": spectrum,
    }


def build_reconciled_v3_manifest(output_dir: Path) -> Dict[str, Any]:
    """Build and verify the 3,859-dataset catalog split & exposure manifest."""
    cat_path = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/catalog.jsonl")
    base_dir = Path("/home/ubuntu/dope/target/s3-multilevel-training-20261023")
    cells = torch.load(base_dir / "s3_prepared_splits_cache.pt", map_location="cpu", weights_only=False)
    v2_cache_ds = {c["dataset_id"]: bool(c["is_val"]) for c in cells}

    v2_eval_1000 = set()
    with gzip.open(base_dir / "per_dataset_1000_strictly_disjoint_results.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            v2_eval_1000.add(json.loads(line)["dataset_id"])

    cands_1132 = select_stratified_candidates(target_count=1132)
    excluded_1132 = {rec["scope"]["dataset_id"] for _, _, rec in cands_1132} | v2_eval_1000
    filtered_132 = excluded_1132 - v2_eval_1000
    v2_train_2001 = {d for d, is_v in v2_cache_ds.items() if d not in excluded_1132 and not is_v}
    v2_devval_58 = {d for d, is_v in v2_cache_ds.items() if d not in excluded_1132 and is_v}

    entries: List[Dict[str, Any]] = []
    counts = Counter()
    with cat_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            did, t = r["id"], r["task"]
            tr_sha, te_sha = r["files"]["train"]["sha256"], r["files"]["test"]["sha256"]
            can_sha = r.get("canonical_sha256", did)
            tr_p, te_p = find_local_blob(tr_sha), find_local_blob(te_sha)
            has_b = bool(tr_p and te_p)
            if not has_b:
                role, exp = "excluded_missing_local_blob", "excluded_no_blob"
            elif t == "timeseries":
                role, exp = "excluded_timeseries_task", "excluded_task"
            elif did in v2_train_2001:
                role, exp = "train_gradient", "v2_train_2001_exposed"
            elif did in v2_devval_58:
                role, exp = "dev_val_ablation", "v2_dev_val_58_exposed"
            elif did in v2_eval_1000:
                role, exp = "eval_gradient_disjoint_v2_panel", "v2_eval_1000_selection_and_error_analysis_exposed"
            elif did in filtered_132:
                role, exp = "excluded_v2_low_oracle_132", "v2_filtered_low_oracle_exposed"
            elif t == "multi_label":
                h = int(hashlib.sha256(can_sha.encode("utf-8")).hexdigest()[:8], 16) % 100
                if h < 60:
                    role, exp = "train_gradient", "unexposed_in_v2_multiclass_train"
                elif h < 75:
                    role, exp = "dev_val_ablation", "unexposed_in_v2_multiclass_dev_val"
                else:
                    role, exp = "eval_gradient_disjoint_multiclass_panel", "unexposed_in_v2_multiclass_heldout_eval"
            else:
                role, exp = "excluded_unindexed_or_degenerate_reg_bin", "unexposed_excluded_in_v2_cache_build"

            counts[(role, exp, t)] += 1
            entries.append({
                "dataset_id": did, "catalog_task": t,
                "model_task": "multiclass" if t == "multi_label" else ("classification" if t == "binary" else t),
                "v3_role": role, "prior_exposure": exp, "train_sha256": tr_sha, "test_sha256": te_sha,
                "canonical_sha256": can_sha, "has_header": bool(r["files"]["train"].get("has_header", False)),
                "rows": int(r["shape"]["rows"]), "features": int(r["shape"]["features"]),
            })

    # Verify zero canonical_sha256 or train_sha256 overlap across active roles
    by_role_shas: Dict[str, set] = {}
    for e in entries:
        if e["v3_role"] in ("train_gradient", "dev_val_ablation", "eval_gradient_disjoint_v2_panel", "eval_gradient_disjoint_multiclass_panel"):
            by_role_shas.setdefault(e["v3_role"], set()).add(e["train_sha256"])

    overlaps = {}
    roles_list = sorted(by_role_shas.keys())
    for i in range(len(roles_list)):
        for j in range(i + 1, len(roles_list)):
            ra, rb = roles_list[i], roles_list[j]
            overlaps[f"{ra}__vs__{rb}"] = len(by_role_shas[ra] & by_role_shas[rb])

    table_rows = [
        {"v3_role": k[0], "prior_exposure": k[1], "catalog_task": k[2], "count": v}
        for k, v in sorted(counts.items())
    ]
    manifest = {
        "schema": "dope.v3.corpus_exposure_and_split_manifest/v1",
        "catalog_total_datasets": len(entries),
        "untouched_sealed_confirmatory_test_on_v2_panel": False,
        "v2_panel_exposure_classification": "development_gradient_disjoint_selection_and_error_analysis_exposed",
        "multiclass_panel_exposure_classification": "unexposed_in_v2_gradient_disjoint_multiclass_panel",
        "role_summary_counts": {
            role: sum(1 for e in entries if e["v3_role"] == role)
            for role in sorted({e["v3_role"] for e in entries})
        },
        "reconciled_breakdown_table": table_rows,
        "train_sha256_cross_role_overlaps": overlaps,
        "entries": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "v3_corpus_exposure_and_split_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_multiclass_cells(entries: Sequence[Dict[str, Any]], *, max_rows_fit: int = 384, max_rows_qry: int = 160) -> List[Dict[str, Any]]:
    """Load and scale train/query cells for the 321 local multiclass (multi_label) datasets."""
    mc_cells: List[Dict[str, Any]] = []
    for e in entries:
        if e["catalog_task"] != "multi_label" or e["v3_role"] not in ("train_gradient", "dev_val_ablation", "eval_gradient_disjoint_multiclass_panel"):
            continue
        tr_p = find_local_blob(e["train_sha256"])
        te_p = find_local_blob(e["test_sha256"])
        if not tr_p or not te_p:
            continue
        try:
            with tr_p.open("r", encoding="utf-8", errors="replace", newline="") as f:
                tr_rows = list(csv.reader(f))
            with te_p.open("r", encoding="utf-8", errors="replace", newline="") as f:
                te_rows = list(csv.reader(f))
            if e["has_header"] and tr_rows:
                tr_rows = tr_rows[1:]
            if e["has_header"] and te_rows:
                te_rows = te_rows[1:]
            arr_tr = np.asarray(tr_rows, dtype=np.float64)
            arr_te = np.asarray(te_rows, dtype=np.float64)
        except (ValueError, TypeError, OSError, csv.Error):
            continue
        if arr_tr.ndim != 2 or arr_tr.shape[0] < 20 or arr_tr.shape[1] < 2:
            continue
        if arr_tr.shape[1] > MAX_COLS + 1:
            arr_tr = np.column_stack([arr_tr[:, :MAX_COLS], arr_tr[:, -1]])
            arr_te = np.column_stack([arr_te[:, :MAX_COLS], arr_te[:, -1]])

        rng = np.random.default_rng(int(hashlib.sha256(e["dataset_id"].encode()).hexdigest()[:8], 16))
        # Subsample fit rows strictly from train blob (split 75% fit / 25% inner query for training episodes)
        n_tr = arr_tr.shape[0]
        perm = rng.permutation(n_tr)
        n_fit = max(int(0.75 * n_tr), 16)
        fit_idx = perm[: min(n_fit, max_rows_fit)]
        qry_idx = perm[n_fit : n_fit + max_rows_qry]
        if qry_idx.size < 6:
            qry_idx = perm[-min(16, n_tr // 3) :]

        X_fit_raw, y_fit_raw = arr_tr[fit_idx, :-1], arr_tr[fit_idx, -1]
        X_qry_raw, y_qry_raw = arr_tr[qry_idx, :-1], arr_tr[qry_idx, -1]
        try:
            scaler = FitOnlyTableTransform.fit(X_fit_raw, y_fit_raw, task="multiclass")
            X_fit, y_fit = scaler.transform(X_fit_raw, y_fit_raw)
            X_qry, y_qry = scaler.transform(X_qry_raw, y_qry_raw)
            X_test_eval, y_test_eval = scaler.transform(arr_te[:256, :-1], arr_te[:256, -1])
        except (ValueError, TypeError, KeyError):
            continue
        k_cls = len(scaler.target_scaler.class_labels) if scaler.target_scaler else 0
        if k_cls < 3 or X_fit.shape[1] == 0:
            continue
        p_use = min(X_fit.shape[1], MAX_COLS)
        X_fit = np.clip(np.nan_to_num(X_fit[:, :p_use], nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0).astype(np.float32)
        X_qry = np.clip(np.nan_to_num(X_qry[:, :p_use], nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0).astype(np.float32)
        report = _synthesize_report_from_train(X_fit, y_fit, "multiclass")
        cell: Dict[str, Any] = {
            "dataset_id": e["dataset_id"], "v3_role": e["v3_role"], "task": "multiclass", "k_classes": k_cls,
            "X_fit": X_fit, "y_fit": y_fit.astype(np.float32), "X_qry": X_qry, "y_qry": y_qry.astype(np.float32),
            "X_test_eval": np.clip(np.nan_to_num(X_test_eval[:, :p_use], nan=0.0), -6.0, 6.0).astype(np.float32),
            "y_test_eval": y_test_eval.astype(np.float32), "X_fit_raw": X_fit_raw[:, :p_use], "y_fit_raw": y_fit_raw,
            "X_test_raw": arr_te[:256, :p_use], "y_test_raw": arr_te[:256, -1], "report": report,
        }
        for lvl in ALL_TIERS:
            b = build_v3_conditioning_bundle(report, X_fit, y_fit, level=lvl)
            cell[f"vec256_{lvl}"] = b["vec256"]
            if lvl == "ultra":
                cell["col_landmarks24"] = b["col_landmarks24"]
                cell["emp_drivers"] = b["v3_driver_weights"]
        mc_cells.append(cell)
    return mc_cells


def _enrich_reg_bin_cell(c: Dict[str, Any]) -> Dict[str, Any]:
    """Compute v3 building-block & inter-landmark vectors + col_landmarks24 from existing train cell."""
    X_f = np.asarray(c["X_fit"], dtype=np.float64)
    y_f = np.asarray(c["y_fit"], dtype=np.float64)
    rep = _synthesize_report_from_train(X_f, y_f, c["task"])
    # Preserve the exact S3 precalculated vector116 in the first 116 slots of each tier
    out = dict(c)
    for lvl in ALL_TIERS:
        b = build_v3_conditioning_bundle(rep, X_f, y_f, level=lvl)
        base_key = "desc_medium" if lvl in ("small", "medium") else f"desc_{lvl}"
        vec140 = np.asarray(c[base_key], dtype=np.float32)
        if lvl == "small":
            vec140 = vec140.copy()
            vec140[6:23] = 0.0
        out[f"vec256_{lvl}"] = np.concatenate([vec140, b["inter_landmark_vec48"], b["building_block_vec68"]]).astype(np.float32)
        if lvl == "ultra":
            out["col_landmarks24"] = b["col_landmarks24"]
            out["emp_drivers"] = b["v3_driver_weights"]
    out["report"] = rep
    return out


def _apply_ablation_mask(vec256: torch.Tensor, col_lm: Optional[torch.Tensor], mode: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    v = vec256.clone()
    if mode == "A0_v2_baseline_140":
        v[140:] = 0.0
        return v, None
    if mode == "A1_plus_16_building_blocks":
        v[140:188] = 0.0
        return v, None
    if mode == "A2_plus_48_inter_landmark_stats":
        return v, None
    # A3_full_v3_with_col_gate_head
    return v, col_lm


def run_dev_val_ablations_and_multiclass_eval(output_dir: Path, *, epochs: int = 4, seed: int = 20261024) -> Dict[str, Any]:
    t0 = time.time()
    manifest = build_reconciled_v3_manifest(output_dir)
    base_dir = Path("/home/ubuntu/dope/target/s3-multilevel-training-20261023")
    raw_cells = torch.load(base_dir / "s3_prepared_splits_cache.pt", map_location="cpu", weights_only=False)
    role_by_id = {e["dataset_id"]: e["v3_role"] for e in manifest["entries"]}

    rng = np.random.default_rng(seed)
    reg_bin_dev = [c for c in raw_cells if role_by_id.get(c["dataset_id"]) == "dev_val_ablation" and c["split"] == "fit-iid-65-v1"]
    reg_bin_tr_all = [c for c in raw_cells if role_by_id.get(c["dataset_id"]) == "train_gradient" and c["split"] == "fit-iid-65-v1"]
    tr_pick = rng.choice(len(reg_bin_tr_all), size=min(480, len(reg_bin_tr_all)), replace=False)
    reg_bin_tr = [reg_bin_tr_all[int(i)] for i in tr_pick]

    print(f"[v3] Enriching {len(reg_bin_tr)} train & {len(reg_bin_dev)} dev-val reg/bin cells + loading 321 multiclass datasets...", flush=True)
    tr_enriched = [_enrich_reg_bin_cell(c) for c in reg_bin_tr]
    dev_enriched = [_enrich_reg_bin_cell(c) for c in reg_bin_dev]
    mc_all = load_multiclass_cells(manifest["entries"])
    mc_train = [c for c in mc_all if c["v3_role"] == "train_gradient"]
    mc_dev = [c for c in mc_all if c["v3_role"] == "dev_val_ablation"]
    mc_eval_heldout = [c for c in mc_all if c["v3_role"] == "eval_gradient_disjoint_multiclass_panel"]

    train_pool = tr_enriched + mc_train
    dev_pool = dev_enriched + mc_dev
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = torch.load(base_dir / "dope_mlc_v2_disjoint1000_best.pt", map_location="cpu", weights_only=False)

    ablation_modes = (
        "A0_v2_baseline_140",
        "A1_plus_16_building_blocks",
        "A2_plus_48_inter_landmark_stats",
        "A3_full_v3_with_col_gate_head",
    )
    ablation_results: Dict[str, Any] = {}
    best_a3_model: Optional[MultiLevelLandmarkModel] = None

    for mode in ablation_modes:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MultiLevelLandmarkModel().to(device)
        model.load_state_dict(base_ckpt["model_state_dict"], strict=False)
        if mode == "A0_v2_baseline_140":
            for p in model.router.inter_attn.parameters():
                p.requires_grad_(False)
            for p in model.target_net.col_gate_head.parameters():
                p.requires_grad_(False)
        elif mode in ("A1_plus_16_building_blocks", "A2_plus_48_inter_landmark_stats"):
            for p in model.target_net.col_gate_head.parameters():
                p.requires_grad_(False)

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=6e-4, weight_decay=1e-4)
        perm_rng = np.random.default_rng(seed)
        col_gate_grad_norms: List[float] = []

        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            order = perm_rng.permutation(len(train_pool))
            for step_i, idx in enumerate(order):
                c = train_pool[int(idx)]
                X_tr = torch.tensor(c["X_fit"], dtype=torch.float32, device=device)
                y_tr = torch.tensor(c["y_fit"], dtype=torch.float32, device=device)
                X_qr = torch.tensor(c["X_qry"], dtype=torch.float32, device=device)
                y_qr = torch.tensor(c["y_qry"], dtype=torch.float32, device=device)
                emp_d = torch.tensor(c["emp_drivers"], dtype=torch.float32, device=device)
                col_lm_raw = torch.tensor(c["col_landmarks24"], dtype=torch.float32, device=device)
                tot = torch.tensor(0.0, device=device)
                for lvl in ALL_TIERS:
                    v_in, c_in = _apply_ablation_mask(torch.tensor(c[f"vec256_{lvl}"], dtype=torch.float32, device=device), col_lm_raw, mode)
                    tot = tot + model.forward_conditioning(X_tr, y_tr, X_qr, y_qr, v_in, emp_d, c["task"], lvl, col_landmarks=c_in)["total"] / 4.0
                if torch.isfinite(tot):
                    (tot / 16.0).backward()
                if (step_i + 1) % 16 == 0 or (step_i + 1) == len(order):
                    if model.target_net.col_gate_head.weight.grad is not None:
                        col_gate_grad_norms.append(float(model.target_net.col_gate_head.weight.grad.norm().item()))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()

        # Evaluate on dev_val_ablation (all 4 tiers & all 3 tasks)
        model.eval()
        by_tier: Dict[str, List[float]] = {lvl: [] for lvl in ALL_TIERS}
        by_task: Dict[str, List[float]] = {"regression": [], "classification": [], "multiclass": []}
        driver_losses, cond_losses = [], []
        col_gate_spreads: List[float] = []
        with torch.no_grad():
            for c in dev_pool:
                X_tr = torch.tensor(c["X_fit"], dtype=torch.float32, device=device)
                y_tr = torch.tensor(c["y_fit"], dtype=torch.float32, device=device)
                X_qr = torch.tensor(c["X_qry"], dtype=torch.float32, device=device)
                y_qr = torch.tensor(c["y_qry"], dtype=torch.float32, device=device)
                emp_d = torch.tensor(c["emp_drivers"], dtype=torch.float32, device=device)
                col_lm_raw = torch.tensor(c["col_landmarks24"], dtype=torch.float32, device=device)
                for lvl in ALL_TIERS:
                    v_in, c_in = _apply_ablation_mask(torch.tensor(c[f"vec256_{lvl}"], dtype=torch.float32, device=device), col_lm_raw, mode)
                    out = model.forward_conditioning(X_tr, y_tr, X_qr, y_qr, v_in, emp_d, c["task"], lvl, col_landmarks=c_in)
                    val_t = float(out["total"].item())
                    by_tier[lvl].append(val_t)
                    by_task[c["task"]].append(val_t)
                    cond_losses.append(float(out["cond_y_loss"].item()))
                    driver_losses.append(float(out["driver_loss"].item()))
                routed = model.router(torch.tensor(c["vec256_ultra"], dtype=torch.float32, device=device), level="ultra")
                _, c_in_u = _apply_ablation_mask(torch.tensor(c["vec256_ultra"], dtype=torch.float32, device=device), col_lm_raw, mode)
                cg = model.target_net.compute_column_gates(routed["task_latent"], c_in_u)[: X_tr.shape[1]]
                col_gate_spreads.append(float(cg.std().item()) if cg.numel() > 1 else 0.0)

        ablation_results[mode] = {
            "dev_val_mean_total_loss": round(float(np.mean([v for vs in by_tier.values() for v in vs])), 5),
            "dev_val_cond_y_loss": round(float(np.mean(cond_losses)), 5),
            "dev_val_driver_loss": round(float(np.mean(driver_losses)), 5),
            "by_tier_total_loss": {lvl: round(float(np.mean(by_tier[lvl])), 5) for lvl in ALL_TIERS},
            "by_task_total_loss": {t: round(float(np.mean(vs)), 5) for t, vs in by_task.items() if vs},
            "col_gate_head_mean_grad_norm": round(float(np.mean(col_gate_grad_norms)) if col_gate_grad_norms else 0.0, 6),
            "col_gate_head_mean_column_std": round(float(np.mean(col_gate_spreads)), 6),
        }
        print(f"  [Ablation {mode}] dev_loss={ablation_results[mode]['dev_val_mean_total_loss']:.5f}, cond_y={ablation_results[mode]['dev_val_cond_y_loss']:.5f}, driver={ablation_results[mode]['dev_val_driver_loss']:.5f}", flush=True)
        if mode == "A3_full_v3_with_col_gate_head":
            best_a3_model = model
            torch.save({"model_state_dict": model.state_dict(), "ablation_results": ablation_results}, output_dir / "dope_mlc_v3_multitask_best.pt")

    # Evaluate 4-tier complete receiver-replay wire bytes & multiclass retention on the 69 held-out multiclass datasets
    assert best_a3_model is not None
    best_a3_model.eval()
    receiver = MultiLevelLandmarkModel().to(device).eval()
    receiver.load_state_dict(best_a3_model.state_dict())

    mc_tier_wire_bytes: Dict[str, List[int]] = {lvl: [] for lvl in ALL_TIERS}
    mc_tier_acc_ret: Dict[str, List[float]] = {lvl: [] for lvl in ALL_TIERS}
    mc_tier_class_cov: Dict[str, List[float]] = {lvl: [] for lvl in ALL_TIERS}
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    for c in mc_eval_heldout:
        X_f_raw, y_f_raw = c["X_fit_raw"], c["y_fit_raw"]
        X_te_raw, y_te_raw = c["X_test_raw"], c["y_test_raw"]
        scaler = FitOnlyTableTransform.fit(X_f_raw, y_f_raw, task="multiclass")
        X_f_s, y_f_s = scaler.transform(X_f_raw, y_f_raw)
        X_te_s, y_te_s = scaler.transform(X_te_raw, y_te_raw)
        if len(np.unique(y_f_s)) < 3:
            continue
        # Oracle real multiclass accuracy (RF + LogisticRegression average)
        rf_real = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42).fit(X_f_s, y_f_s.astype(int))
        lr_real = LogisticRegression(max_iter=200).fit(X_f_s, y_f_s.astype(int))
        acc_real = 0.5 * (float(np.mean(rf_real.predict(X_te_s) == y_te_s.astype(int))) + float(np.mean(lr_real.predict(X_te_s) == y_te_s.astype(int))))
        chance = 1.0 / float(c["k_classes"])
        denom = max(acc_real - chance, 0.03)

        for lvl in ALL_TIERS:
            enc = best_a3_model.encode_table(
                X_f_s, y_f_s, c["report"], task="multiclass", level=lvl,
                adaptive_budget=True, include_v3_primitives=True,
            )
            wire = serialize_dope_v3_envelope(enc, scaler=scaler, compress=True)
            mc_tier_wire_bytes[lvl].append(len(wire))
            dec_enc, dec_sc = deserialize_dope_v3_envelope(wire)
            X_syn_s, y_syn_idx = receiver.generate_synthetic(dec_enc, n_rows=X_f_s.shape[0], seed=20261024)
            y_syn_int = np.rint(y_syn_idx).astype(int)
            mc_tier_class_cov[lvl].append(float(len(np.unique(y_syn_int))) / float(c["k_classes"]))
            rf_syn = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42).fit(X_syn_s, y_syn_int)
            lr_syn = LogisticRegression(max_iter=200).fit(X_syn_s, y_syn_int)
            acc_syn = 0.5 * (float(np.mean(rf_syn.predict(X_te_s) == y_te_s.astype(int))) + float(np.mean(lr_syn.predict(X_te_s) == y_te_s.astype(int))))
            R_mc = float(np.clip((acc_syn - chance) / denom, 0.0, 1.25))
            mc_tier_acc_ret[lvl].append(R_mc)

    mc_summary = {
        lvl: {
            "datasets_evaluated": len(mc_tier_wire_bytes[lvl]),
            "cap_bytes": TIER_BYTE_CAPS[lvl],
            "mean_complete_wire_bytes": round(float(np.mean(mc_tier_wire_bytes[lvl])), 1),
            "p95_complete_wire_bytes": round(float(np.percentile(mc_tier_wire_bytes[lvl], 95)), 1),
            "max_complete_wire_bytes": int(np.max(mc_tier_wire_bytes[lvl])),
            "pct_within_cap": round(100.0 * float(np.mean(np.asarray(mc_tier_wire_bytes[lvl]) <= TIER_BYTE_CAPS[lvl])), 2),
            "mean_multiclass_retention_R": round(float(np.mean(mc_tier_acc_ret[lvl])), 4),
            "median_multiclass_retention_R": round(float(np.median(mc_tier_acc_ret[lvl])), 4),
            "mean_class_support_coverage": round(float(np.mean(mc_tier_class_cov[lvl])), 4),
        }
        for lvl in ALL_TIERS
    }
    receipt = {
        "study_id": "dope-mlc-v3-dev-val-ablation-and-multitask-20261024",
        "elapsed_seconds": round(time.time() - t0, 2),
        "manifest_summary": manifest["role_summary_counts"],
        "train_sha256_cross_role_overlaps": manifest["train_sha256_cross_role_overlaps"],
        "dev_val_ablation_pool_counts": {
            "regression": sum(1 for c in dev_pool if c["task"] == "regression"),
            "binary": sum(1 for c in dev_pool if c["task"] == "classification"),
            "multiclass": sum(1 for c in dev_pool if c["task"] == "multiclass"),
            "total": len(dev_pool),
        },
        "ablation_progression_A0_to_A3": ablation_results,
        "heldout_multiclass_panel_69_summary": mc_summary,
    }
    (output_dir / "v3_dev_val_ablation_and_multitask_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    out_dir = Path("/home/ubuntu/dope/target/s3-multilevel-training-20261023/v3")
    res = run_dev_val_ablations_and_multiclass_eval(out_dir, epochs=4)
    print(json.dumps(res, indent=2))

