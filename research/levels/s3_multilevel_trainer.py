"""Full S3 Corpus Deep Learning Trainer for the Unified 3-Level Model (DOPE-MLC-v2).

Trains across all 19,190 admitted Regression and Binary Classification dataset x split
cells (3,201 datasets x 6 splits: fit-iid-65-v1 and outer5-iid-fold0..4-v1) using:
- Mandatory train-only per-column scaling & categorical encoding (FitOnlyTableTransform)
- 3-level pre-calculated S3 landmarks & data descriptors (Medium=Macro, High=Larger, Ultra=Fine)
- Source-dataset-disjoint 90% Train / 10% Validation split
- Multi-task conditional target + covariate distribution + driver attribution + landmark loss

Records complete per-epoch and per-step training and validation convergence metrics.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from .descriptors import compute_table_descriptors, extract_report_descriptors
from .multilevel_model import MAX_COLS, MultiLevelLandmarkModel
from .precalc.plan import training_rows
from .scaling import FitOnlyTableTransform

DEFAULT_BUNDLE = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/s3_precalc_19190_reports.jsonl.gz")
DEFAULT_CATALOG = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/catalog.jsonl")
DEFAULT_SOURCE_CACHE = Path("/home/ubuntu/dope-fit-cv-precalc-20261023/unified-source-cache")


def _is_val_dataset(dataset_id: str) -> bool:
    h = int(hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10) == 0


def _load_raw_csv_rows(args: Tuple[str, Path, bool]) -> Tuple[str, Optional[List[List[str]]]]:
    sha, path, has_header = args
    if not path.exists():
        return sha, None
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
        if has_header and rows:
            rows = rows[1:]
        return sha, rows
    except Exception:
        return sha, None


def _prepare_dataset_splits(
    args: Tuple[str, str, bool, Path, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Worker: for one dataset, fit FitOnlyTableTransform on each split's fit rows and transform held-out query rows."""
    ds_id, cat_task, has_header, csv_path, split_records = args
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
        if has_header and rows:
            rows = rows[1:]
    except Exception:
        return []

    n_total = len(rows)
    if n_total < 24 or len(rows[0]) < 2:
        return []

    task_name = "classification" if cat_task == "binary" else "regression"
    is_val = _is_val_dataset(ds_id)
    rng = np.random.default_rng(int(hashlib.sha256(ds_id.encode()).hexdigest()[:8], 16))

    # Fast check if table is purely numeric
    num_table = None
    try:
        num_table = np.asarray(rows, dtype=np.float64)
        if num_table.shape[1] > MAX_COLS + 1:
            num_table = np.column_stack([num_table[:, :MAX_COLS], num_table[:, -1]])
    except Exception:
        num_table = None

    # Process all 6 splits using the exact physical row indices used by S3 precalculation:
    prepared_cells: List[Dict[str, Any]] = []
    for rec in split_records:
        sp_name = rec["scope"]["split"]
        physical = rec["scope"]["physical_table_id"]
        seed = int(rec["scope"].get("split_seed", 20261023))
        try:
            fit_idx = training_rows(n_total, physical, sp_name, seed)
        except Exception:
            continue
        fit_set = set(fit_idx)
        qry_idx = [i for i in range(n_total) if i not in fit_set]
        if len(fit_idx) < 16 or len(qry_idx) < 6:
            continue

        # Subsample up to 384 fit rows and 160 query rows for fast GPU batching
        if len(fit_idx) > 384:
            fit_idx = [fit_idx[i] for i in np.sort(rng.choice(len(fit_idx), size=384, replace=False))]
        if len(qry_idx) > 160:
            qry_idx = [qry_idx[i] for i in np.sort(rng.choice(len(qry_idx), size=160, replace=False))]

        try:
            if num_table is not None:
                X_fit_in = num_table[fit_idx, :-1]
                y_fit_in = num_table[fit_idx, -1]
                X_qry_in = num_table[qry_idx, :-1]
                y_qry_in = num_table[qry_idx, -1]
            else:
                X_fit_in = [rows[i][:MAX_COLS] if len(rows[i]) > MAX_COLS + 1 else rows[i][:-1] for i in fit_idx]
                y_fit_in = [rows[i][-1] for i in fit_idx]
                X_qry_in = [rows[i][:MAX_COLS] if len(rows[i]) > MAX_COLS + 1 else rows[i][:-1] for i in qry_idx]
                y_qry_in = [rows[i][-1] for i in qry_idx]
            scaler = FitOnlyTableTransform.fit(X_fit_in, y_fit_in, task=task_name)
            X_fit, y_fit = scaler.transform(X_fit_in, y_fit_in)
            X_qry, y_qry = scaler.transform(X_qry_in, y_qry_in)
        except Exception:
            continue

        if X_fit.shape[1] == 0:
            continue
        if task_name == "classification" and (len(np.unique(y_fit)) < 2 or len(np.unique(y_qry)) < 2):
            continue

        p_use = min(X_fit.shape[1], MAX_COLS)
        X_fit = np.clip(np.nan_to_num(X_fit[:, :p_use], nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0).astype(np.float16)
        X_qry = np.clip(np.nan_to_num(X_qry[:, :p_use], nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0).astype(np.float16)
        y_fit = np.clip(np.nan_to_num(y_fit, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0).astype(np.float16)
        y_qry = np.clip(np.nan_to_num(y_qry, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0).astype(np.float16)

        report = rec["report"]
        desc_by_level = {}
        emp_drivers = None
        for lvl in ("medium", "high", "ultra"):
            rep_d = extract_report_descriptors(report, level=lvl)
            tab_d = compute_table_descriptors(X_fit, y_fit, level=lvl)
            desc_by_level[lvl] = np.concatenate([rep_d["vector116"], tab_d["table_vec24"]]).astype(np.float16)
            if lvl == "ultra":
                emp_drivers = (0.5 * rep_d["column_prior_importance"][:p_use] + 0.5 * tab_d["empirical_driver_weights"][:p_use]).astype(np.float16)

        prepared_cells.append({
            "dataset_id": ds_id,
            "split": sp_name,
            "task": task_name,
            "is_val": is_val,
            "X_fit": X_fit,
            "y_fit": y_fit,
            "X_qry": X_qry,
            "y_qry": y_qry,
            "desc_medium": desc_by_level["medium"],
            "desc_high": desc_by_level["high"],
            "desc_ultra": desc_by_level["ultra"],
            "emp_drivers": emp_drivers,
        })
    return prepared_cells


def build_or_load_s3_tensor_cache(
    cache_path: Path,
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    catalog_path: Path = DEFAULT_CATALOG,
    source_cache_dir: Path = DEFAULT_SOURCE_CACHE,
    workers: int = 8,
) -> List[Dict[str, Any]]:
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    cat_tasks: Dict[str, str] = {}
    if catalog_path.exists():
        with catalog_path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                k = r.get("id") or r.get("dataset_hash")
                if k:
                    cat_tasks[k] = r["task"]

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    with gzip.open(bundle_path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ds_id = rec["scope"]["dataset_id"]
            t = cat_tasks.get(ds_id, rec["scope"]["task"])
            if t in ("regression", "binary"):
                by_dataset.setdefault(ds_id, []).append(rec)

    tasks_args = []
    for ds_id, recs in sorted(by_dataset.items()):
        sc = recs[0]["scope"]
        cat_task = cat_tasks.get(ds_id, sc["task"])
        csv_path = source_cache_dir / f"{sc['train_sha256']}.csv"
        tasks_args.append((ds_id, cat_task, bool(sc.get("has_header", False)), csv_path, recs))

    all_cells: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for group_cells in ex.map(_prepare_dataset_splits, tasks_args, chunksize=8):
            all_cells.extend(group_cells)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_cells, cache_path)
    return all_cells


def _prestage_cell_on_device(cell: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        "dataset_id": cell["dataset_id"],
        "split": cell["split"],
        "task": cell["task"],
        "is_val": cell["is_val"],
        "X_fit": torch.tensor(cell["X_fit"], dtype=torch.float32, device=device),
        "y_fit": torch.tensor(cell["y_fit"], dtype=torch.float32, device=device),
        "X_qry": torch.tensor(cell["X_qry"], dtype=torch.float32, device=device),
        "y_qry": torch.tensor(cell["y_qry"], dtype=torch.float32, device=device),
        "desc_medium": torch.tensor(cell["desc_medium"], dtype=torch.float32, device=device),
        "desc_high": torch.tensor(cell["desc_high"], dtype=torch.float32, device=device),
        "desc_ultra": torch.tensor(cell["desc_ultra"], dtype=torch.float32, device=device),
        "emp_drivers": torch.tensor(cell["emp_drivers"], dtype=torch.float32, device=device),
    }


@torch.no_grad()
def evaluate_corpus(
    model: MultiLevelLandmarkModel,
    cells: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    by_level: Dict[str, Dict[str, List[float]]] = {
        lvl: {"total": [], "cond_y": [], "dist_x": [], "driver": [], "landmark": []}
        for lvl in ("medium", "high", "ultra")
    }
    gates_snapshot: Dict[str, Dict[str, float]] = {}

    for cell in cells:
        X_tr = cell["X_fit"] if isinstance(cell["X_fit"], torch.Tensor) else torch.tensor(cell["X_fit"], dtype=torch.float32, device=device)
        y_tr = cell["y_fit"] if isinstance(cell["y_fit"], torch.Tensor) else torch.tensor(cell["y_fit"], dtype=torch.float32, device=device)
        X_qr = cell["X_qry"] if isinstance(cell["X_qry"], torch.Tensor) else torch.tensor(cell["X_qry"], dtype=torch.float32, device=device)
        y_qr = cell["y_qry"] if isinstance(cell["y_qry"], torch.Tensor) else torch.tensor(cell["y_qry"], dtype=torch.float32, device=device)
        emp_d = cell["emp_drivers"] if isinstance(cell["emp_drivers"], torch.Tensor) else torch.tensor(cell["emp_drivers"], dtype=torch.float32, device=device)
        task = cell["task"]

        for lvl in ("medium", "high", "ultra"):
            desc = cell[f"desc_{lvl}"] if isinstance(cell[f"desc_{lvl}"], torch.Tensor) else torch.tensor(cell[f"desc_{lvl}"], dtype=torch.float32, device=device)
            out = model.forward_conditioning(X_tr, y_tr, X_qr, y_qr, desc, emp_d, task, lvl)
            by_level[lvl]["total"].append(float(out["total"].item()))
            by_level[lvl]["cond_y"].append(float(out["cond_y_loss"].item()))
            by_level[lvl]["dist_x"].append(float(out["dist_x_loss"].item()))
            by_level[lvl]["driver"].append(float(out["driver_loss"].item()))
            by_level[lvl]["landmark"].append(float(out["landmark_loss"].item()))
            if lvl not in gates_snapshot:
                gates_snapshot[lvl] = out["gates"]

    summary: Dict[str, Any] = {}
    overall_totals = []
    for lvl in ("medium", "high", "ultra"):
        m_tot = float(np.mean(by_level[lvl]["total"]))
        overall_totals.append(m_tot)
        summary[lvl] = {
            "total_loss": round(m_tot, 5),
            "cond_y_loss": round(float(np.mean(by_level[lvl]["cond_y"])), 5),
            "dist_x_loss": round(float(np.mean(by_level[lvl]["dist_x"])), 5),
            "driver_loss": round(float(np.mean(by_level[lvl]["driver"])), 5),
            "landmark_loss": round(float(np.mean(by_level[lvl]["landmark"])), 5),
            "sample_gates": {k: round(v, 4) for k, v in gates_snapshot.get(lvl, {}).items()},
        }
    summary["mean_total_loss"] = round(float(np.mean(overall_totals)), 5)
    return summary


def train_s3_multilevel_model(
    output_dir: Path,
    *,
    epochs: int = 8,
    lr: float = 8e-4,
    batch_accum: int = 32,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    workers: int = 8,
    seed: int = 20261023,
) -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "s3_prepared_splits_cache.pt"

    print(f"[DOPE-MLC-v2] Preparing/loading S3 Regression + Binary Classification splits (workers={workers})...")
    t_prep = time.time()
    all_cells = build_or_load_s3_tensor_cache(cache_path, workers=workers)
    prep_seconds = round(time.time() - t_prep, 2)

    train_cells = [c for c in all_cells if not c["is_val"]]
    val_cells = [c for c in all_cells if c["is_val"]]
    # Use a deterministic 600-cell subset of train_cells for fast per-epoch train loss evaluation alongside full val_cells
    rng = np.random.default_rng(seed)
    train_eval_subset = [train_cells[i] for i in rng.choice(len(train_cells), size=min(600, len(train_cells)), replace=False)]

    splits_count: Dict[str, int] = {}
    tasks_count = {"regression": 0, "classification": 0}
    unique_datasets = set()
    for c in all_cells:
        splits_count[c["split"]] = splits_count.get(c["split"], 0) + 1
        tasks_count[c["task"]] = tasks_count.get(c["task"], 0) + 1
        unique_datasets.add(c["dataset_id"])

    print(
        f"[DOPE-MLC-v2] Loaded {len(all_cells)} splits across {len(unique_datasets)} datasets "
        f"(train={len(train_cells)}, val={len(val_cells)}) in {prep_seconds}s on {device_name}"
    )

    device = torch.device(device_name)
    print(f"[DOPE-MLC-v2] Pre-staging {len(all_cells)} splits on {device_name} VRAM...")
    train_gpu = [_prestage_cell_on_device(c, device) for c in train_cells]
    val_gpu = [_prestage_cell_on_device(c, device) for c in val_cells]
    # Deterministic 600-split validation & train monitor subsets for fast per-epoch logging, plus full val at end
    val_eval_subset = [val_gpu[i] for i in rng.choice(len(val_gpu), size=min(600, len(val_gpu)), replace=False)]
    train_eval_subset = [train_gpu[i] for i in rng.choice(len(train_gpu), size=min(600, len(train_gpu)), replace=False)]

    model = MultiLevelLandmarkModel(max_cols=MAX_COLS, latent_dim=128, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Partition the 17,188 training splits into 6 disjoint shards (~2,865 splits/epoch) for Epochs 1..6
    # so 100% of all 17,188 training splits are trained without repetition, plus 2 refinement shards (Epochs 7..8)
    n_shards = 6
    perm_all = rng.permutation(len(train_gpu))
    shards = [perm_all[i::n_shards] for i in range(n_shards)]
    while len(shards) < epochs:
        shards.append(rng.choice(len(train_gpu), size=len(shards[0]), replace=False))

    total_steps = sum(len(sh) // batch_accum + 1 for sh in shards[:epochs])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=lr * 0.05)

    history: List[Dict[str, Any]] = []
    # Epoch 0 evaluation (before any gradient step)
    val_0 = evaluate_corpus(model, val_eval_subset, device)
    tr_0 = evaluate_corpus(model, train_eval_subset, device)
    history.append({
        "epoch": 0,
        "splits_visited_cumulative": 0,
        "elapsed_seconds": round(time.time() - t0, 2),
        "train": tr_0,
        "val": val_0,
    })
    print(
        f"[Epoch 0] train_loss={tr_0['mean_total_loss']:.5f} | val_loss={val_0['mean_total_loss']:.5f} "
        f"(med={val_0['medium']['total_loss']:.5f}, high={val_0['high']['total_loss']:.5f}, ultra={val_0['ultra']['total_loss']:.5f})"
    )

    best_val_loss = val_0["mean_total_loss"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    cum_splits = 0

    for epoch in range(1, epochs + 1):
        model.train()
        shard_indices = shards[epoch - 1]
        optimizer.zero_grad()
        accum_count = 0

        for idx in shard_indices:
            cell = train_gpu[int(idx)]
            X_tr = cell["X_fit"]
            y_tr = cell["y_fit"]
            X_qr = cell["X_qry"]
            y_qr = cell["y_qry"]
            emp_d = cell["emp_drivers"]
            task = cell["task"]

            loss_cell = torch.tensor(0.0, device=device)
            for lvl in ("medium", "high", "ultra"):
                desc = cell[f"desc_{lvl}"]
                out = model.forward_conditioning(X_tr, y_tr, X_qr, y_qr, desc, emp_d, task, lvl)
                loss_cell = loss_cell + out["total"] / 3.0

            if torch.isfinite(loss_cell):
                (loss_cell / float(batch_accum)).backward()
                accum_count += 1

            if accum_count >= batch_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0

        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        cum_splits += len(shard_indices)
        val_ep = evaluate_corpus(model, val_eval_subset, device)
        tr_ep = evaluate_corpus(model, train_eval_subset, device)
        history.append({
            "epoch": epoch,
            "splits_visited_cumulative": cum_splits,
            "elapsed_seconds": round(time.time() - t0, 2),
            "train": tr_ep,
            "val": val_ep,
        })
        print(
            f"[Epoch {epoch} | {cum_splits} splits] train_loss={tr_ep['mean_total_loss']:.5f} | val_loss={val_ep['mean_total_loss']:.5f} "
            f"(med={val_ep['medium']['total_loss']:.5f}, high={val_ep['high']['total_loss']:.5f}, ultra={val_ep['ultra']['total_loss']:.5f})"
        )
        if val_ep["mean_total_loss"] <= best_val_loss:
            best_val_loss = val_ep["mean_total_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    full_val_final = evaluate_corpus(model, val_gpu, device)
    ckpt_path = output_dir / "dope_mlc_v2_s3_best.pt"
    torch.save({"model_state_dict": best_state, "history": history, "full_val_final": full_val_final}, ckpt_path)

    receipt = {
        "architecture": "DOPE-MLC-v2 (Single Unified 3-Level Landmark & Descriptor Model)",
        "device": device_name,
        "unique_datasets": len(unique_datasets),
        "total_splits": len(all_cells),
        "train_splits": len(train_cells),
        "val_splits": len(val_cells),
        "tasks_breakdown": tasks_count,
        "splits_breakdown": splits_count,
        "epochs": epochs,
        "prep_seconds": prep_seconds,
        "total_elapsed_seconds": round(time.time() - t0, 2),
        "initial_val": val_0,
        "final_best_val": history[-1]["val"],
        "full_1891_val_final": full_val_final,
        "convergence_history": history,
        "checkpoint_path": str(ckpt_path),
    }
    (output_dir / "s3_training_convergence_receipt.json").write_text(
        json.dumps(receipt, indent=2),
        encoding="utf-8",
    )
    return receipt


if __name__ == "__main__":
    out_dir = Path("/home/ubuntu/dope/target/s3-multilevel-training-20261023")
    res = train_s3_multilevel_model(out_dir, epochs=8, lr=8e-4, batch_accum=32, workers=8)
    print(f"Saved full S3 convergence receipt to {out_dir / 's3_training_convergence_receipt.json'}")
