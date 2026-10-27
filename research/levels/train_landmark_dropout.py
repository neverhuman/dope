"""Landmark-dropout trainer for DOPE-MLC-v3 on the frozen train_gradient roster."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from .building_block_landmarks import build_v3_conditioning_bundle
from .encoder import DOPE_ACTIVE_ENCODER_TENSORS, DopeEncoder629, PINNED_MULTITASK_SAFETENSORS_SHA256
from .multilevel_model import MAX_COLS, MultiLevelLandmarkModel
from .optional_landmarks import (
    LARGER_INDICES, MACRO_INDICES, LandmarkBundle, build_conditioning,
    compute_observed_inter_landmark_stats, observed_report_descriptors,
    observed_target_stats, stratified_subsample, table_classes,
)
from .scaling import FitOnlyTableTransform
MANIFEST_SHA256 = "66d2a63c3e4b84e69a554af212a2f9f63081d59f366cd1f4b796bfc7eea39bdd"
MC_FIT_MANIFEST_SHA256 = "2f7c5ccfc666d305a4423458c42983f70d76776c6f88a0bdf467bcf52cc304c9"
REG_BIN_FIT_MANIFEST_SHA256 = "c8d8e5ab6b3822254dd3df4e1ace3d8f23d7663dc909e4510dd41fcaac3bd68e"
MC_EVIDENCE_MANIFEST_SHA256 = "aaec7871cc544affdb472ac1324fd80b70701f0eaa14f81d6e3ad6f701cbfa12"
RB_EVIDENCE_MANIFEST_SHA256 = "6814da066cb058576d88167695f2a7ada573ca3ae7679afeed6a9b2f79ed3cc9"
ALL_TIERS = ("small", "medium", "high", "ultra")
VIEW_SCHEDULE = ("none", "full", "none", "full", "macro", "macro+larger", "random_sparse", "random_dense")
MACRO_PLUS_LARGER = tuple(sorted(set(MACRO_INDICES) | set(LARGER_INDICES)))
BLOB_ROOTS = tuple(Path(x) for x in (
    "/home/ubuntu/dope/data/real/unique_10378/v3_splits/train", "/home/ubuntu/dope-fit-cv-precalc-20261023/blobs",
    "/home/ubuntu/scratch/jopedime-corpus-benchmark-20260928/blobs", "/home/ubuntu/scratch/jopedime-corpus-throughput-20260928/blobs",
    "/home/ubuntu/out/loop-store/blobs",
))
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def _resolve_train_blob(did: str, train_sha256: str) -> Path:
    for r in BLOB_ROOTS:
        for cand in (r / f"{did}.csv", r / "sha256" / train_sha256[:2] / train_sha256, r / "sha256" / train_sha256, r / train_sha256[:2] / train_sha256, r / train_sha256):
            if cand.exists():
                if _sha256_file(cand) != train_sha256:
                    raise ValueError(f"train_blob_sha_mismatch:{did}")
                return cand
    raise FileNotFoundError(f"missing_authenticated_train_blob:{did}:{train_sha256}")
def _view_seed(run_seed: int, dataset_id: str, split: str, epoch: int, level: str, view_idx: int) -> int:
    msg = f"{run_seed}:{dataset_id}:{split}:{epoch}:{level}:{view_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "little")
def _mask_bundle(
    full_bundle: LandmarkBundle, view: str, *, rng_seed: int, sub300_bundle: Optional[LandmarkBundle] = None
) -> LandmarkBundle:
    if view == "none":
        return LandmarkBundle(values=[0.0] * 23, mask=[False] * 23, subsample_n=None, source_digest=full_bundle.source_digest)
    if view == "full":
        return full_bundle
    if view == "sub300" and sub300_bundle is not None:
        return sub300_bundle
    if view in ("macro", "macro+larger"):
        keep = set(MACRO_INDICES if view == "macro" else MACRO_PLUS_LARGER)
    elif view in ("random_sparse", "random_dense"):
        p_keep = 0.25 if view == "random_sparse" else 0.75
        rng = np.random.default_rng(rng_seed)
        keep = {i for i in range(23) if full_bundle.mask[i] and float(rng.random()) < p_keep}
    else:
        raise ValueError(f"unsupported_view_{view}")
    vals = [float(full_bundle.values[i]) if (i in keep and full_bundle.mask[i]) else 0.0 for i in range(23)]
    msk = [bool(i in keep and full_bundle.mask[i]) for i in range(23)]
    return LandmarkBundle(values=vals, mask=msk, subsample_n=full_bundle.subsample_n, source_digest=full_bundle.source_digest)
def _fast_observed_conditioning(
    bundle: LandmarkBundle, *, full_vec256: np.ndarray, obs_col17: np.ndarray,
    rows: int, features: int, classes: int, task: str, target_stats: Sequence[Optional[float]], level: str,
) -> Tuple[np.ndarray, np.ndarray]:
    rep_desc = observed_report_descriptors(
        bundle, rows=rows, features=features, classes=classes, task=task, target_stats=target_stats, level=level
    )
    inter48 = compute_observed_inter_landmark_stats(rep_desc, level=level)
    vec256 = np.concatenate([rep_desc["vector116"], full_vec256[116:140], inter48, np.zeros(68, dtype=np.float32)]).astype(np.float32)
    return np.nan_to_num(vec256, nan=0.0, posinf=0.0, neginf=0.0), obs_col17
def _load_evidence_index(native_root: Path, expected_sha: str, expected_src_sha: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    ev_path = native_root / "evidence-manifest.jsonl"
    if _sha256_file(ev_path) != expected_sha:
        raise ValueError(f"evidence_manifest_sha_mismatch:{ev_path}")
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ln in ev_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if r.get("record") == "header":
            if r.get("source_manifest_sha256") != expected_src_sha:
                raise ValueError("evidence_header_source_manifest_mismatch")
        else:
            idx[(str(r["dataset_id"]), str(r["arm"]))] = r
    return idx
def _load_verified_native_pair(
    native_root: Path, did: str, tag: str, *, expected_rows: int, expected_features: int,
    ev_index: Dict[Tuple[str, str], Dict[str, Any]], expected_fit_csv_sha: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[LandmarkBundle]]:
    ev = ev_index.get((did, tag))
    if ev is None or int(ev.get("native_exit", 1)) != 0:
        return None, None
    if ev.get("fit_csv_sha256") != expected_fit_csv_sha or not ev.get("fit_csv_verified"):
        raise ValueError(f"evidence_fit_csv_sha_mismatch:{did}:{tag}")
    rep_path = native_root / "reports" / f"{did}.{tag}.json"
    bun_path = native_root / "bundles" / f"{did}.{tag}.json"
    if _sha256_file(rep_path) != ev["report_sha256"] or _sha256_file(bun_path) != ev["bundle_sha256"]:
        raise ValueError(f"native_artifact_sha_mismatch:{did}:{tag}")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    if int(rep.get("features", -1)) != expected_features or (tag == "full" and int(rep.get("rows", -1)) != expected_rows):
        raise ValueError(f"native_report_shape_mismatch:{did}:{tag}")
    bun = LandmarkBundle.from_dict(json.loads(bun_path.read_text(encoding="utf-8")))
    from_rep = LandmarkBundle.from_report(rep, subsample_n=bun.subsample_n, include_uninformative=True, source_digest=bun.source_digest)
    if not bun.same_landmarks(from_rep) or bun.source_digest != ev["bundle_source_digest"]:
        raise ValueError(f"native_bundle_report_mismatch:{did}:{tag}")
    return rep, bun
def prepare_full_partition_cell(
    *, did: str, split: str, task: str, X_part_sc: np.ndarray, y_part_sc: np.ndarray, y_part_raw: np.ndarray,
    X_fit_ep: np.ndarray, y_fit_ep: np.ndarray, X_qry_ep: np.ndarray, y_qry_ep: np.ndarray,
    k_classes: int, rep_full: Dict[str, Any], bun_full: LandmarkBundle,
    rep_n300: Optional[Dict[str, Any]] = None, bun_n300: Optional[LandmarkBundle] = None,
    train_sha256: str = "", unknown_qry_masked: int = 0,
) -> Dict[str, Any]:
    n_part, p = int(X_part_sc.shape[0]), int(X_part_sc.shape[1])
    classes = k_classes if task == "multiclass" else table_classes(y_part_sc, task)
    t_stats = observed_target_stats(y_part_sc, task, y_raw=(y_part_raw if task == "regression" else None))
    out: Dict[str, Any] = {
        "dataset_id": did, "split": split, "task": task, "k_classes": k_classes,
        "partition_rows": n_part, "features": p, "classes": classes, "target_stats": t_stats,
        "X_fit": X_fit_ep, "y_fit": y_fit_ep, "X_qry": X_qry_ep, "y_qry": y_qry_ep,
        "full_bundle": bun_full, "sub300_bundle": bun_n300,
        "train_sha256": train_sha256, "source_digest": bun_full.source_digest,
        "unknown_qry_masked": unknown_qry_masked,
    }
    sub_X, sub_y = None, None
    if rep_n300 is not None and bun_n300 is not None:
        idx300 = stratified_subsample(y_part_raw, task, 300, seed=0)
        sub_X, sub_y = X_part_sc[idx300], y_part_sc[idx300]
    for lvl in ALL_TIERS:
        c_full = build_conditioning(
            X_part_sc, y_part_sc, task=task, bundle=bun_full, mode="full",
            native_report=rep_full, level=lvl, y_train_raw=y_part_raw, target_stats=t_stats,
        )
        vec_f = np.asarray(c_full["vec256"], dtype=np.float32)
        col17_f = np.asarray(c_full["col_landmarks"], dtype=np.float32)
        obs_col17 = np.zeros_like(col17_f)
        obs_col17[:, 9:17] = col17_f[:, 9:17]
        out[f"vec256_{lvl}"] = vec_f
        out[f"col17_{lvl}"] = col17_f
        out[f"obs_col17_{lvl}"] = obs_col17
        if lvl == "ultra":
            out["emp_drivers"] = np.asarray(c_full["cond"]["v3_driver_weights"], dtype=np.float32)
        if rep_n300 is not None and bun_n300 is not None and sub_X is not None and sub_y is not None:
            rt = "classification" if task in ("classification", "multiclass") else "regression"
            overrides = {"rows": n_part, "features": p, "classes": classes, "task": rt, "target_stats": t_stats}
            b_sub = build_v3_conditioning_bundle(rep_n300, sub_X, sub_y, level=lvl, report_overrides=overrides)
            vec_sub = np.asarray(b_sub["vec256"], dtype=np.float32).copy()
            vec_sub[116:140] = vec_f[116:140]
            col24_sub = np.asarray(b_sub["col_landmarks24"], dtype=np.float32)
            col17_sub = np.zeros((MAX_COLS, 17), dtype=np.float32)
            take = min(MAX_COLS, col24_sub.shape[0])
            col17_sub[:take, :9] = col24_sub[:take, :9]
            col17_sub[:, 9:17] = col17_f[:, 9:17]
            out[f"vec256_sub300_{lvl}"] = vec_sub
            out[f"col17_sub300_{lvl}"] = col17_sub
    return out
def _load_manifest_cells(
    *, fit_manifest_path: Path, expected_manifest_sha: str, expected_evidence_sha: str, native_root: Path,
    tg_by_id: Dict[str, Dict[str, Any]], max_per_task: Dict[str, int], seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    if _sha256_file(fit_manifest_path) != expected_manifest_sha:
        raise ValueError(f"fit_manifest_sha_mismatch:{fit_manifest_path}")
    ev_index = _load_evidence_index(native_root, expected_evidence_sha, expected_manifest_sha)
    records_by_task: Dict[str, List[Dict[str, Any]]] = {"regression": [], "classification": [], "multiclass": []}
    with fit_manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            did = str(m["dataset_id"])
            if did not in tg_by_id or m.get("v3_role") != "train_gradient":
                raise ValueError(f"non_train_gradient_in_manifest:{did}")
            records_by_task[str(m["model_task"])].append(m)
    rng = np.random.default_rng(seed)
    selected: List[Dict[str, Any]] = []
    for t_name in ("regression", "classification", "multiclass"):
        cands = sorted(records_by_task[t_name], key=lambda x: str(x["dataset_id"]))
        if not cands:
            continue
        cap = max_per_task.get(t_name, len(cands))
        if len(cands) > cap:
            idxs = sorted(int(i) for i in rng.choice(len(cands), size=cap, replace=False))
            cands = [cands[i] for i in idxs]
        selected.extend(cands)
    cells: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    scope_digests: List[str] = []
    for m in selected:
        did = str(m["dataset_id"])
        tg_entry = tg_by_id[did]
        tr_sha = str(tg_entry["train_sha256"])
        if str(m["train_sha256"]) != tr_sha:
            raise ValueError(f"manifest_train_sha_mismatch:{did}")
        has_header = bool(tg_entry.get("has_header", False))
        fit_ids = [int(i) for i in m["fit_partition_row_ids"]]
        ep_ids = [int(i) for i in m["fit_episode_row_ids"]]
        qry_ids = [int(i) for i in m["qry_episode_row_ids"]]
        if not set(fit_ids).isdisjoint(set(qry_ids)) or not set(ep_ids).issubset(set(fit_ids)):
            raise ValueError(f"invalid_partition_split:{did}")
        p_feat = int(m["features"])
        f_sha = str(m["fit_csv_sha256"])
        rep_full, bun_full = _load_verified_native_pair(
            native_root, did, "full", expected_rows=len(fit_ids), expected_features=p_feat, ev_index=ev_index, expected_fit_csv_sha=f_sha
        )
        if rep_full is None or bun_full is None:
            exclusions.append({"dataset_id": did, "task": m["model_task"], "reason": "native_full_unavailable"})
            continue
        rep_n300, bun_n300 = _load_verified_native_pair(
            native_root, did, "n300", expected_rows=len(fit_ids), expected_features=p_feat, ev_index=ev_index, expected_fit_csv_sha=f_sha
        )
        tr_blob = _resolve_train_blob(did, tr_sha)
        with tr_blob.open("r", encoding="utf-8", errors="replace", newline="") as cfh:
            rows = list(csv.reader(cfh))
        if has_header and rows:
            rows = rows[1:]
        ncol = len(rows[0])
        if ncol > MAX_COLS + 1:
            rows = [r[:MAX_COLS] + [r[-1]] for r in rows]
            ncol = MAX_COLS + 1
        p_use = ncol - 1
        part_rows = [rows[i] for i in fit_ids]
        ep_rows = [rows[i] for i in ep_ids]
        qr_rows = [rows[i] for i in qry_ids]
        task = str(m["model_task"])
        try:
            arr = np.asarray(rows, dtype=np.float64)
            if np.isnan(arr).any():
                raise ValueError("contains_nan")
            X_part_raw, y_part_raw = arr[fit_ids, :p_use], arr[fit_ids, -1]
            X_ep_raw, y_ep_raw = arr[ep_ids, :p_use], arr[ep_ids, -1]
            X_qr_raw, y_qr_raw = arr[qry_ids, :p_use], arr[qry_ids, -1]
        except ValueError:
            X_part_raw, y_part_raw = [r[:p_use] for r in part_rows], np.array([float(r[-1]) for r in part_rows], dtype=np.float64)
            X_ep_raw, y_ep_raw = [r[:p_use] for r in ep_rows], np.array([float(r[-1]) for r in ep_rows], dtype=np.float64)
            X_qr_raw, y_qr_raw = [r[:p_use] for r in qr_rows], np.array([float(r[-1]) for r in qr_rows], dtype=np.float64)
        scaler = FitOnlyTableTransform.fit(X_part_raw, y_part_raw, task=task, max_multiclass_classes=None)
        X_part_sc, y_part_sc = scaler.transform(X_part_raw, y_part_raw, strict_unknown=(task == "multiclass"))
        X_fit_ep, y_fit_ep = scaler.transform(X_ep_raw, y_ep_raw, strict_unknown=(task == "multiclass"))
        X_qry_ep = scaler.transform_features(X_qr_raw)
        unknown_qry_masked = 0
        if task == "multiclass":
            assert scaler.target_scaler is not None
            labels = np.asarray(scaler.target_scaler.class_labels, dtype=np.float64)
            k_cls = int(len(labels))
            if k_cls < 3:
                exclusions.append({"dataset_id": did, "task": task, "reason": "fewer_than_3_classes"})
                continue
            qr_arr = np.asarray(y_qr_raw, dtype=np.float64)
            dists = np.abs(np.round(qr_arr, 6)[:, None] - labels[None, :])
            unseen = (~np.isfinite(qr_arr)) | (np.min(dists, axis=1) > 1e-6)
            unknown_qry_masked = int(np.sum(unseen))
            y_qry_ep = np.where(unseen, -1.0, np.argmin(dists, axis=1).astype(np.float64))
        else:
            k_cls = 2 if task == "classification" else 0
            y_qry_ep = scaler.transform_target(y_qr_raw)
        X_part_sc = np.clip(np.nan_to_num(X_part_sc, nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0)
        X_fit_ep = np.clip(np.nan_to_num(X_fit_ep, nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0).astype(np.float32)
        X_qry_ep = np.clip(np.nan_to_num(X_qry_ep, nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0).astype(np.float32)
        split_name = f"{task}-fit-75-v1"
        cell = prepare_full_partition_cell(
            did=did, split=split_name, task=task, X_part_sc=X_part_sc, y_part_sc=np.asarray(y_part_sc, dtype=np.float64),
            y_part_raw=y_part_raw, X_fit_ep=X_fit_ep, y_fit_ep=np.asarray(y_fit_ep, dtype=np.float32),
            X_qry_ep=X_qry_ep, y_qry_ep=np.asarray(y_qry_ep, dtype=np.float32), k_classes=k_cls,
            rep_full=rep_full, bun_full=bun_full, rep_n300=rep_n300, bun_n300=bun_n300,
            train_sha256=tr_sha, unknown_qry_masked=unknown_qry_masked,
        )
        cells.append(cell)
        scope_digests.append(f"{did}:{split_name}:{tr_sha}:{bun_full.source_digest}")
    return cells, exclusions, scope_digests
def load_strict_train_gradient_pool(
    *, manifest_path: Path, reg_bin_fit_manifest_path: Path, reg_bin_native_dir: Path,
    mc_fit_manifest_path: Path, mc_native_dir: Path,
    max_reg: int = 320, max_bin: int = 200, max_mc: int = 200, seed: int = 20261027,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if _sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("v3_corpus_exposure_and_split_manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tg_by_id = {str(e["dataset_id"]): e for e in manifest["entries"] if e["v3_role"] == "train_gradient"}
    rb_cells, rb_excl, rb_scopes = _load_manifest_cells(
        fit_manifest_path=reg_bin_fit_manifest_path, expected_manifest_sha=REG_BIN_FIT_MANIFEST_SHA256,
        expected_evidence_sha=RB_EVIDENCE_MANIFEST_SHA256, native_root=reg_bin_native_dir, tg_by_id=tg_by_id,
        max_per_task={"regression": max_reg, "classification": max_bin}, seed=seed,
    )
    mc_cells, mc_excl, mc_scopes = _load_manifest_cells(
        fit_manifest_path=mc_fit_manifest_path, expected_manifest_sha=MC_FIT_MANIFEST_SHA256,
        expected_evidence_sha=MC_EVIDENCE_MANIFEST_SHA256, native_root=mc_native_dir, tg_by_id=tg_by_id,
        max_per_task={"multiclass": max_mc}, seed=seed,
    )
    pool = rb_cells + mc_cells
    all_scopes = rb_scopes + mc_scopes
    stats = {
        "train_gradient_manifest_total": len(tg_by_id),
        "mixed_role_caches_opened": 0, "outer_canonical_blobs_opened": 0, "synthesized_reports_used": 0,
        "regression_cells": sum(1 for c in pool if c["task"] == "regression"),
        "binary_cells": sum(1 for c in pool if c["task"] == "classification"),
        "multiclass_cells": sum(1 for c in pool if c["task"] == "multiclass"),
        "cells_with_n300": sum(1 for c in pool if c.get("sub300_bundle") is not None),
        "multiclass_k_ge_33": sum(1 for c in pool if c["task"] == "multiclass" and int(c["k_classes"]) >= 33),
        "unknown_query_labels_masked": sum(int(c.get("unknown_qry_masked", 0)) for c in pool),
        "exclusions": rb_excl + mc_excl,
        "total_cells": len(pool),
        "pool_scope_sha256": hashlib.sha256("\n".join(sorted(all_scopes)).encode("utf-8")).hexdigest(),
    }
    return pool, stats
def train_landmark_dropout_from_pool(
    pool: Sequence[Dict[str, Any]], pool_stats: Dict[str, Any], *,
    output_dir: Path, base_safetensors_path: Path, epochs: int = 8, lr: float = 1.5e-4, seed: int = 20261027,
) -> Dict[str, Any]:
    t0 = time.time()
    if _sha256_file(base_safetensors_path) != PINNED_MULTITASK_SAFETENSORS_SHA256:
        raise ValueError("base checkpoint SHA-256 mismatch vs d0b69603")
    if not pool:
        raise ValueError("empty_training_pool")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_state = load_file(str(base_safetensors_path))
    model = MultiLevelLandmarkModel().to(device)
    model.load_state_dict(base_state, strict=True)
    ref_model = MultiLevelLandmarkModel().to(device).eval()
    ref_model.load_state_dict(base_state, strict=True)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    perm_rng = np.random.default_rng(seed)
    all_views = VIEW_SCHEDULE + ("sub300",)
    view_counts: Dict[str, int] = {v: 0 for v in all_views}
    task_counts: Dict[str, int] = {"regression": 0, "classification": 0, "multiclass": 0}
    task_view_counts: Dict[str, Dict[str, int]] = {t: {v: 0 for v in all_views} for t in task_counts}
    view_losses: Dict[str, List[float]] = {v: [] for v in all_views}
    epoch_records: List[Dict[str, Any]] = []
    opt_steps = 0
    for ep in range(epochs):
        ep_t0 = time.time()
        model.train()
        opt.zero_grad()
        order = perm_rng.permutation(len(pool))
        ep_losses: List[float] = []
        accum_valid = 0
        for step_i, idx in enumerate(order):
            c = pool[int(idx)]
            did, split, task = str(c["dataset_id"]), str(c["split"]), str(c["task"])
            k_cat = int(c["k_classes"]) if task == "multiclass" else None
            X_tr = torch.tensor(c["X_fit"], dtype=torch.float32, device=device)
            y_tr = torch.tensor(c["y_fit"], dtype=torch.float32, device=device)
            X_qr = torch.tensor(c["X_qry"], dtype=torch.float32, device=device)
            y_qr = torch.tensor(c["y_qry"], dtype=torch.float32, device=device)
            emp_d = torch.tensor(c["emp_drivers"], dtype=torch.float32, device=device)
            p_cols = int(X_tr.shape[1])
            tot = torch.tensor(0.0, device=device)
            step_views: List[Tuple[str, float]] = []
            for lvl_i, lvl in enumerate(ALL_TIERS):
                view_idx = (ep + step_i + lvl_i) % len(VIEW_SCHEDULE)
                view = VIEW_SCHEDULE[view_idx]
                if view == "full" and c.get("sub300_bundle") is not None and ((ep + step_i) % 3 == 0):
                    view = "sub300"
                v_full_t = torch.tensor(c[f"vec256_{lvl}"], dtype=torch.float32, device=device)
                c_full_t = torch.tensor(c[f"col17_{lvl}"], dtype=torch.float32, device=device)
                if view == "full":
                    v_in_t, c_in_t = v_full_t, c_full_t
                elif view == "sub300":
                    v_in_t = torch.tensor(c[f"vec256_sub300_{lvl}"], dtype=torch.float32, device=device)
                    c_in_t = torch.tensor(c[f"col17_sub300_{lvl}"], dtype=torch.float32, device=device)
                else:
                    v_seed = _view_seed(seed, did, split, ep, lvl, view_idx)
                    b_masked = _mask_bundle(c["full_bundle"], view, rng_seed=v_seed)
                    v_np, c_np = _fast_observed_conditioning(
                        b_masked, full_vec256=c[f"vec256_{lvl}"], obs_col17=c[f"obs_col17_{lvl}"],
                        rows=c["partition_rows"], features=c["features"], classes=c["classes"],
                        task=task, target_stats=c["target_stats"], level=lvl,
                    )
                    v_in_t = torch.tensor(v_np, dtype=torch.float32, device=device)
                    c_in_t = torch.tensor(c_np, dtype=torch.float32, device=device)
                out = model.forward_conditioning(
                    X_tr, y_tr, X_qr, y_qr, v_in_t, emp_d, task, lvl,
                    col_landmarks=c_in_t, supervision_desc_vec=v_full_t, k_categories=k_cat,
                )
                with torch.no_grad():
                    ref_routed = ref_model.router(v_full_t, level=lvl)
                    ref_cg = ref_model.target_net.compute_column_gates(ref_routed["task_latent"], c_full_t)[:p_cols]
                cur_routed = model.router(v_in_t, level=lvl)
                cur_cg = model.target_net.compute_column_gates(cur_routed["task_latent"], c_in_t)[:p_cols]
                cons_loss = (
                    F.mse_loss(cur_routed["task_latent"], ref_routed["task_latent"])
                    + F.mse_loss(cur_routed["dist_latent"], ref_routed["dist_latent"])
                    + F.mse_loss(cur_cg, ref_cg)
                )
                step_loss = out["total"] + 0.35 * cons_loss
                tot = tot + step_loss / 4.0
                step_views.append((view, float(step_loss.detach().item())))
            if torch.isfinite(tot):
                (tot / 16.0).backward()
                ep_losses.append(float(tot.detach().item()))
                accum_valid += 1
                for v_name, v_val in step_views:
                    view_counts[v_name] += 1
                    task_counts[task] += 1
                    task_view_counts[task][v_name] += 1
                    view_losses[v_name].append(v_val)
            if (step_i + 1) % 16 == 0 or (step_i + 1) == len(order):
                if accum_valid > 0:
                    grads_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters())
                    if grads_finite:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        opt_steps += 1
                opt.zero_grad()
                accum_valid = 0
        rec_ep = {"epoch": ep + 1, "mean_total_loss": round(float(np.mean(ep_losses)), 5), "elapsed_s": round(time.time() - ep_t0, 2)}
        epoch_records.append(rec_ep)
        print(f"[train_landmark_dropout] epoch {ep + 1}/{epochs} loss={rec_ep['mean_total_loss']:.5f} steps={opt_steps} ({rec_ep['elapsed_s']:.1f}s)", flush=True)
    if opt_steps <= 0:
        raise RuntimeError("zero_valid_optimizer_steps")
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "dope_mlc_v3_landmark_dropout.safetensors"
    final_state = {k: v.detach().cpu().contiguous().to(torch.float32) for k, v in model.state_dict().items()}
    save_file(final_state, str(ckpt_path))
    weights_sha256 = _sha256_file(ckpt_path)
    _ = DopeEncoder629.from_pretrained(ckpt_path, expected_sha256=weights_sha256, device="cpu")
    changed_total, changed_active_encoder = 0, 0
    l2_delta_sq, active_l2_delta_sq = 0.0, 0.0
    for k, v_new in final_state.items():
        v_old = base_state[k].cpu().to(torch.float32)
        diff = float(torch.norm(v_new - v_old).item())
        if diff > 0.0:
            changed_total += 1
            l2_delta_sq += diff * diff
            if k.startswith("router.") or k.startswith("target_net.col_gate_head."):
                changed_active_encoder += 1
                active_l2_delta_sq += diff * diff
    if changed_active_encoder <= 0:
        raise RuntimeError("no_active_encoder_parameters_updated")
    receipt = {
        "schema": "dope.v3.landmark_dropout_training_receipt/v1",
        "weights_path": str(ckpt_path), "weights_sha256": weights_sha256,
        "parent_weights_sha256": PINNED_MULTITASK_SAFETENSORS_SHA256,
        "manifest_sha256": MANIFEST_SHA256, "reg_bin_fit_manifest_sha256": REG_BIN_FIT_MANIFEST_SHA256,
        "mc_fit_manifest_sha256": MC_FIT_MANIFEST_SHA256, "reg_bin_evidence_manifest_sha256": RB_EVIDENCE_MANIFEST_SHA256, "mc_evidence_manifest_sha256": MC_EVIDENCE_MANIFEST_SHA256,
        "tensor_count": len(final_state), "parameter_count": int(sum(v.numel() for v in final_state.values())),
        "active_encoder_tensors": DOPE_ACTIVE_ENCODER_TENSORS,
        "changed_tensors_total": changed_total, "changed_active_encoder_tensors": changed_active_encoder,
        "weights_l2_delta": round(float(np.sqrt(l2_delta_sq)), 6),
        "active_encoder_l2_delta": round(float(np.sqrt(active_l2_delta_sq)), 6),
        "optimizer_steps": opt_steps, "epochs": epochs, "learning_rate": lr, "seed": seed,
        "pool_stats": pool_stats, "view_episode_counts": view_counts, "task_episode_counts": task_counts,
        "task_view_episode_counts": task_view_counts,
        "view_mean_loss": {k: round(float(np.mean(vs)), 5) for k, vs in view_losses.items() if vs},
        "epoch_history": epoch_records, "elapsed_seconds": round(time.time() - t0, 2),
    }
    receipt_path = output_dir / "training_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_sha256 = _sha256_file(receipt_path)
    manifest_out = {
        "schema": "dope.v3.landmark_dropout_checkpoint_manifest/v1",
        "checkpoint_identity": "MaskDropoutV1", "weights_path": str(ckpt_path), "weights_sha256": weights_sha256,
        "training_receipt_path": str(receipt_path), "training_receipt_sha256": receipt_sha256,
        "parent_weights_sha256": PINNED_MULTITASK_SAFETENSORS_SHA256,
        "corpus_manifest_sha256": MANIFEST_SHA256, "reg_bin_fit_manifest_sha256": REG_BIN_FIT_MANIFEST_SHA256,
        "mc_fit_manifest_sha256": MC_FIT_MANIFEST_SHA256, "reg_bin_evidence_manifest_sha256": RB_EVIDENCE_MANIFEST_SHA256, "mc_evidence_manifest_sha256": MC_EVIDENCE_MANIFEST_SHA256,
        "pool_scope_sha256": pool_stats["pool_scope_sha256"], "optimizer_steps": opt_steps,
        "changed_active_encoder_tensors": changed_active_encoder,
        "active_encoder_l2_delta": receipt["active_encoder_l2_delta"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_out
def main() -> None:
    ap = argparse.ArgumentParser()
    v3_root = "/home/ubuntu/dope/target/s3-multilevel-training-20261023/v3"
    qa_root = "/home/ubuntu/dope/.agent/worktrees/cc-qa/target/qual"
    ap.add_argument("--out-dir", default=f"{v3_root}/landmark_dropout_v1")
    ap.add_argument("--base-ckpt", default=f"{v3_root}/dope_mlc_v3_multitask_best.safetensors")
    ap.add_argument("--manifest", default=f"{v3_root}/v3_corpus_exposure_and_split_manifest.json")
    ap.add_argument("--reg-bin-fit-manifest", default=f"{v3_root}/reg_bin_train_gradient_fit/manifest.jsonl")
    ap.add_argument("--reg-bin-native", default=f"{qa_root}/rb-train-native-2e12")
    ap.add_argument("--mc-fit-manifest", default=f"{v3_root}/mc_train_gradient_fit/manifest.jsonl")
    ap.add_argument("--mc-native", default=f"{qa_root}/mc-train-native-2e12")
    for flg, tp, df in (("--epochs", int, 8), ("--lr", float, 1.5e-4), ("--seed", int, 20261027), ("--max-reg", int, 320), ("--max-bin", int, 200), ("--max-mc", int, 200)):
        ap.add_argument(flg, type=tp, default=df)
    args = ap.parse_args()
    pool, stats = load_strict_train_gradient_pool(
        manifest_path=Path(args.manifest), reg_bin_fit_manifest_path=Path(args.reg_bin_fit_manifest),
        reg_bin_native_dir=Path(args.reg_bin_native), mc_fit_manifest_path=Path(args.mc_fit_manifest),
        mc_native_dir=Path(args.mc_native), max_reg=args.max_reg, max_bin=args.max_bin,
        max_mc=args.max_mc, seed=args.seed,
    )
    res = train_landmark_dropout_from_pool(
        pool, stats, output_dir=Path(args.out_dir), base_safetensors_path=Path(args.base_ckpt),
        epochs=args.epochs, lr=args.lr, seed=args.seed,
    )
    print(json.dumps(res, indent=2))
if __name__ == "__main__":
    main()
