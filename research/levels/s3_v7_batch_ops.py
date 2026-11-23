"""Maximum-entropy flat-arena episode builder & trust-region differentiable loss for SingleEncV7 (V7.2).

Combines ALL 4,145 S3 train tables (including all 51 >8-class multiclass tables recovered via
class-subset/tail cropping) + 20% infinite synthetic nonlinear SCM/tree/interaction priors +
exact per-step shuffled tri-task quotas (6 reg / 5 bin / 5 mc) + fresh duplicate-grouped splits,
log-uniform row/col cropping, multi-task target swaps, MCAR/MAR missingness, monotonic warps,
and class-ID permutations on EVERY batch, plus trust-region L2 anchoring on neural residuals.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .s3_v7_entropy import (
    append_nonlinear_and_distractor_cols,
    apply_manifold_augmentations,
    apply_real_table_target_swap,
    crop_multiclass_classes,
    sample_synthetic_prior_table,
    synthesize_target_from_regression,
    task_quota_for_step,
)
from .v7_single_encoding import (
    MAX_CLASSES,
    MAX_COLS,
    PACKET_BUDGET_BYTES,
    extract_dse_packet_numpy,
)

TASK_TO_ID = {"regression": 0, "binary": 1, "multiclass": 2}


def _hash_u64(*parts: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(int(p).to_bytes(8, "little", signed=True))
    return int.from_bytes(h.digest(), "little")


class FlatArenaView:
    """Zero-disk memmap reader for `dope.flat-corpus/v1` (`X.f32`, `y.f32`, `dup.i32`, `index.jsonl`)."""

    def __init__(self, arena_dir: str | Path, split: str = "train") -> None:
        root = Path(arena_dir)
        self.root = root
        self.X = np.memmap(root / "X.f32", dtype=np.float32, mode="r")
        self.y = np.memmap(root / "y.f32", dtype=np.float32, mode="r")
        self.dup = np.memmap(root / "dup.i32", dtype=np.int32, mode="r")
        self.entries: list[dict[str, Any]] = []
        self.by_task_fams: dict[str, list[list[int]]] = {"regression": [], "binary": [], "multiclass": []}
        self.by_hash: dict[str, int] = {}
        fam_map: dict[str, dict[str, list[int]]] = {"regression": {}, "binary": {}, "multiclass": {}}

        with (root / "index.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("split") != split:
                    continue
                if int(rec.get("n_rows", 0)) < 64 or int(rec.get("n_cols", 0)) < 2:
                    continue
                task = str(rec.get("task", "regression"))
                if task not in fam_map:
                    continue
                idx = len(self.entries)
                self.entries.append(rec)
                self.by_hash[str(rec.get("dataset_hash", ""))] = idx
                fam = str(rec.get("family_stem") or f"tbl_{rec['table_id']}")
                fam_map[task].setdefault(fam, []).append(idx)

        for task, groups in fam_map.items():
            self.by_task_fams[task] = [groups[k] for k in sorted(groups)]

    def load_slice(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        rec = self.entries[idx]
        nr, nc = int(rec["n_rows"]), int(rec["n_cols"])
        x0, r0 = int(rec["x_off"]), int(rec["row_off"])
        X = np.array(self.X[x0 : x0 + nr * nc].reshape(nr, nc), dtype=np.float32, copy=True)
        y = np.array(self.y[r0 : r0 + nr], dtype=np.float32, copy=True)
        dg = np.array(self.dup[r0 : r0 + nr], dtype=np.int32, copy=True)
        return X, y, dg, rec


def _fit_only_scale(Xa: np.ndarray, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact FIT-only NC2-compatible scaling: median/max(IQR/1.349, sd) + asinh tails -> [-6, 6]."""
    Xa_c = np.where(np.isfinite(Xa), Xa, np.nan)
    med = np.nanmedian(Xa_c, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    q25 = np.nanpercentile(Xa_c, 25, axis=0)
    q75 = np.nanpercentile(Xa_c, 75, axis=0)
    iqr = np.where(np.isfinite(q75 - q25), (q75 - q25) / 1.349, 0.0)
    sd = np.nanstd(Xa_c, axis=0)
    sd = np.where(np.isfinite(sd), sd, 0.0)
    scale = np.maximum(np.maximum(iqr, sd), 1e-6).astype(np.float32)

    def _xform(M: np.ndarray) -> np.ndarray:
        z = (np.where(np.isfinite(M), M, med[None, :]) - med[None, :]) / scale[None, :]
        az = np.abs(z)
        u = np.where(az <= 3.0, z, np.sign(z) * (3.0 + np.arcsinh(np.maximum(az - 3.0, 0.0))))
        return np.clip(u, -6.0, 6.0).astype(np.float32)

    return _xform(Xa), _xform(Xq)


def _pick_family_table(
    arena: FlatArenaView, task: str, seed: int, step: int, slot: int, slots_per_step: int = 16
) -> int | None:
    fams = arena.by_task_fams[task]
    if not fams:
        return None
    n_fams = len(fams)
    seq = step * max(slots_per_step, 1) + slot
    epoch, pos = seq // max(n_fams, 1), seq % n_fams
    cand = int(_hash_u64(seed, epoch, TASK_TO_ID[task]) % max(n_fams - 1, 1)) + 1
    while math.gcd(cand, n_fams) != 1 and cand < n_fams:
        cand += 1
    stride = cand if math.gcd(cand, n_fams) == 1 else 1
    offset = int(_hash_u64(seed, epoch, 777 + TASK_TO_ID[task]) % n_fams)
    members = fams[(offset + pos * stride) % n_fams]
    return members[(epoch + offset + slot) % len(members)]


def _detect_discrete_or_lattice_mask(X: np.ndarray, probe_rows: int = 256) -> np.ndarray:
    """Identify discrete (<= 16 unique values) or regular-lattice columns so jitter never corrupts them (FP-0 / F3)."""
    nr, nc = X.shape
    probe = X[: min(nr, probe_rows)]
    mask = np.zeros((nc,), dtype=bool)
    for j in range(nc):
        col = probe[:, j]
        fin = col[np.isfinite(col)]
        if fin.size < 2:
            mask[j] = True
            continue
        uniq = np.unique(fin.astype(np.float64))
        if uniq.size <= 16:
            mask[j] = True
            continue
        if uniq.size <= 128:
            diffs = np.diff(uniq)
            pos = diffs[diffs > 1e-5]
            if pos.size >= 2:
                d_min = float(np.min(pos))
                step = float(np.median(pos[pos <= 1.25 * d_min]))
                if step > 1e-5:
                    # Subtract origin uniq[0] so shifted regular lattices (e.g., 48/100 levels) are detected!
                    shifted = (uniq - uniq[0]) / step
                    rem = np.abs(shifted - np.rint(shifted))
                    if float(np.mean(rem <= 3e-2)) >= 0.98:
                        mask[j] = True
    return mask


def apply_type_safe_fit_jitter(
    rng: np.random.Generator,
    Xa: np.ndarray,
    jitter_prob: float = 0.30,
    jitter_sd: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FIT-only Gaussian jitter strictly to continuous (non-discrete, non-lattice) columns (FP-0 / F3)."""
    disc_mask = _detect_discrete_or_lattice_mask(Xa)
    cont_idx = np.flatnonzero(~disc_mask)
    if cont_idx.size > 0 and rng.random() < jitter_prob:
        Xa = Xa.copy()
        noise = rng.normal(0.0, jitter_sd, size=(Xa.shape[0], cont_idx.size)).astype(np.float32)
        Xa[:, cont_idx] = np.clip(Xa[:, cont_idx] + noise, -6.0, 6.0)
    return Xa, disc_mask


def build_unique_v7_episode(
    arena: FlatArenaView,
    seed: int,
    step: int,
    global_slot: int,
    n_fit_cap: int = 384,
    n_qry_cap: int = 256,
    max_bytes: int = PACKET_BUDGET_BYTES,
    desired_task: str | None = None,
    slots_per_step: int = 16,
    real_queue: Any | None = None,
    allow_replay_on_exhaust: bool = False,
) -> dict[str, Any] | None:
    """Build one maximum-entropy counter-seeded episode (S3 real + synthetic prior + full cropping)."""
    sh_id = getattr(real_queue, "shard_id", 0) if real_queue is not None else 0
    ep_seed = _hash_u64(seed, step, global_slot, sh_id + 17)
    rng = np.random.default_rng(ep_seed)
    if desired_task is None:
        sched = task_quota_for_step(16, _hash_u64(seed, step, 31337))
        task = sched[global_slot % len(sched)]
    else:
        task = desired_task

    is_synth = False
    is_swapped = False
    is_cropped = False
    crop_ratio = 1.0
    dataset_hash = f"synth:s{sh_id}:{task}:{ep_seed:016x}"
    split_h = hashlib.blake2b(digest_size=16)

    # Eval-matched log-uniform sample-size cropping covering tiny (32) through headline (n_fit_cap up to 1024).
    n_fit_target = int(round(math.exp(rng.uniform(math.log(32.0), math.log(max(64.0, float(n_fit_cap)))))))
    n_qry_lo = min(64.0, max(24.0, float(n_qry_cap) * 0.25))
    n_qry_target = int(round(math.exp(rng.uniform(math.log(n_qry_lo), math.log(max(n_qry_lo + 8.0, float(n_qry_cap)))))))

    use_synth = rng.random() < (0.75 if real_queue is not None else 0.20)
    tbl_idx: int | None = None
    from_reg_synth = False
    if not use_synth:
        from_reg_synth = (task in ("binary", "multiclass")) and (rng.random() < 0.15)
        src_task = "regression" if from_reg_synth else task
        if real_queue is not None:
            tbl_idx = real_queue.pop_unseen(src_task)
            if tbl_idx is None:
                if allow_replay_on_exhaust:
                    tbl_idx = _pick_family_table(
                        arena, src_task, seed, step, global_slot, slots_per_step=slots_per_step
                    )
                if tbl_idx is None:
                    use_synth = True
                    from_reg_synth = False
        else:
            tbl_idx = _pick_family_table(arena, src_task, seed, step, global_slot, slots_per_step=slots_per_step)
            if tbl_idx is None:
                return None

    # Infinite synthetic nonlinear SCM / depth-3 tree / interaction priors (20% mix OR seamless Tier-2 fallback)!
    if use_synth:
        is_synth = True
        Xa_raw, ya, Xq_raw, yq, n_cls = sample_synthetic_prior_table(
            rng, task, n_fit_target, n_qry_target, max_cols=MAX_COLS - 4
        )
        Xa_raw, Xq_raw = apply_manifold_augmentations(rng, Xa_raw, Xq_raw)
        Xa, Xq = _fit_only_scale(Xa_raw, Xq_raw)
        split_h.update(dataset_hash.encode("utf-8"))
    else:
        assert tbl_idx is not None
        X_raw, y_raw, dg, rec = arena.load_slice(tbl_idx)
        dataset_hash = str(rec.get("dataset_hash", ""))
        if from_reg_synth:
            y_syn, n_k = synthesize_target_from_regression(rng, y_raw, task)
            if (task == "binary" and n_k == 2) or (task == "multiclass" and 3 <= n_k <= MAX_CLASSES):
                y_raw = y_syn
                is_swapped = True
            else:
                task = "regression"
        elif rng.random() < 0.15:
            X_raw, y_raw, task, is_swapped = apply_real_table_target_swap(rng, X_raw, y_raw, task)

        # Recover ALL 51 >8-class S3 multiclass datasets via random K-class subset / tail cropping!
        if task == "multiclass":
            X_raw, y_raw, dg, did_cls_crop = crop_multiclass_classes(rng, X_raw, y_raw, dg, MAX_CLASSES)
            is_cropped = is_cropped or did_cls_crop

        nr, nc = X_raw.shape
        if nr < 48 or nc < 2:
            return None
        # Strict duplicate-grouped split (FIX-08): NEVER fall back to row-random splitting across groups!
        uniq_g = np.unique(dg)
        if len(uniq_g) < 4:
            return None
        ratio = float(rng.uniform(0.35, 0.75))
        g_perm = rng.permutation(uniq_g)
        min_fit_g = 2 if len(uniq_g) < 16 else 8
        min_qry_g = 2 if len(uniq_g) < 16 else 6
        g_cut = max(min_fit_g, min(len(g_perm) - min_qry_g, int(round(len(g_perm) * ratio))))
        is_fit = np.isin(dg, g_perm[:g_cut])
        fit_idx, qry_idx = np.flatnonzero(is_fit), np.flatnonzero(~is_fit)
        if len(fit_idx) < 24 or len(qry_idx) < 16:
            return None
        if len(fit_idx) > n_fit_target:
            fit_idx = rng.choice(fit_idx, size=n_fit_target, replace=False)
            is_cropped = True
        if len(qry_idx) > n_qry_target:
            qry_idx = rng.choice(qry_idx, size=n_qry_target, replace=False)

        # Feature column cropping (65% of episodes even when nc <= 44!) + random column permutation!
        max_real_p = min(nc, MAX_COLS - 4)
        if nc >= 4 and (nc > max_real_p or rng.random() < 0.65):
            lo_p = max(3, min(max_real_p, int(math.ceil(0.45 * max_real_p))))
            p_pick = int(rng.integers(lo_p, max_real_p + 1))
            is_cropped = is_cropped or (p_pick < nc)
        else:
            p_pick = max_real_p
        crop_ratio = float(p_pick) / float(max(nc, 1))
        col_perm = rng.choice(nc, size=p_pick, replace=False)
        split_h.update(dataset_hash.encode("utf-8"))
        split_h.update(np.sort(fit_idx.astype(np.int32)).tobytes())
        split_h.update(np.sort(qry_idx.astype(np.int32)).tobytes())
        split_h.update(col_perm.astype(np.int32).tobytes())
        Xa_raw, Xq_raw = apply_manifold_augmentations(rng, X_raw[fit_idx][:, col_perm], X_raw[qry_idx][:, col_perm])
        Xa, Xq = _fit_only_scale(Xa_raw, Xq_raw)
        ya, yq = y_raw[fit_idx].copy(), y_raw[qry_idx].copy()

    # Append nonlinear interactions + distractors, then type/lattice-safe FIT-only jitter (FP-0 / F3).
    Xa, Xq, n_noise = append_nonlinear_and_distractor_cols(rng, Xa, Xq, max_cols=MAX_COLS)
    Xa, _ = apply_type_safe_fit_jitter(rng, Xa, jitter_prob=0.30, jitter_sd=0.015)

    # Target canonicalization + class-ID permutation / binary polarity flip EVERY episode!
    task_id = TASK_TO_ID[task]
    split_h.update(f"|t={task_id}|sw={int(is_swapped)}".encode("utf-8"))
    if task_id == 0:
        if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(yq))):
            return None
        mu, sd = float(np.mean(ya)), max(float(np.std(ya)), 1e-5)
        ya = np.clip((ya - mu) / sd, -6.0, 6.0).astype(np.float32)
        yq = np.clip((yq - mu) / sd, -6.0, 6.0).astype(np.float32)
        if rng.random() < 0.5:
            ya, yq = -ya, -yq
            split_h.update(b"|neg")
        n_cls = 1
    elif task_id == 1:
        u_a = np.unique(ya[np.isfinite(ya)])
        if len(u_a) < 2:
            return None
        hi = float(u_a[-1])
        ya, yq = (ya == hi).astype(np.float32), (yq == hi).astype(np.float32)
        if rng.random() < 0.5:
            ya, yq = 1.0 - ya, 1.0 - yq
            split_h.update(b"|flip")
        if len(np.unique(ya)) < 2 or len(np.unique(yq)) < 2:
            return None
        n_cls = 2
    else:
        ya_int = np.rint(ya).astype(np.int64)
        yq_int = np.rint(yq).astype(np.int64)
        uniq_c = np.unique(ya_int)
        if len(uniq_c) < 3 or len(uniq_c) > MAX_CLASSES:
            return None
        # FIX-07: Never silently remap unseen query classes or mutate the query denominator; refuse split cleanly.
        if not bool(np.all(np.isin(yq_int, uniq_c))):
            return None
        perm_ids = rng.permutation(len(uniq_c))
        mp = {int(c): int(perm_ids[i]) for i, c in enumerate(uniq_c)}
        split_h.update(json.dumps(mp, sort_keys=True).encode("utf-8"))
        ya = np.array([mp[int(v)] for v in ya_int], dtype=np.float32)
        yq = np.array([mp[int(v)] for v in yq_int], dtype=np.float32)
        n_cls = len(uniq_c)

    pkt = extract_dse_packet_numpy(Xa, ya, task_id, n_cls, max_bytes=max_bytes)
    p_eff = pkt["p_eff"]
    disc_col_share = float(np.mean(pkt["col_feats"][:p_eff, 38])) if p_eff > 0 else 0.0
    Xq_pad = np.zeros((n_qry_cap, MAX_COLS), dtype=np.float32)
    yq_pad = np.zeros((n_qry_cap,), dtype=np.float32)
    q_mask = np.zeros((n_qry_cap,), dtype=np.float32)
    nq = min(len(Xq), n_qry_cap)
    Xq_pad[:nq, :p_eff] = Xq[:nq, :p_eff]
    yq_pad[:nq] = yq[:nq]
    q_mask[:nq] = 1.0
    pkt.update(
        X_qry=Xq_pad, y_qry=yq_pad, q_mask=q_mask, n_fit=len(Xa), n_noise=n_noise,
        disc_col_share=disc_col_share,
        is_synth_prior=int(is_synth), is_swapped=int(is_swapped), is_cropped=int(is_cropped),
        col_crop_ratio=float(crop_ratio), dataset_hash=dataset_hash, split_hash=split_h.hexdigest(),
    )
    return pkt


def collate_v7_batch(episodes: list[dict[str, Any]], as_numpy: bool = False) -> dict[str, Any]:
    keys = [
        "col_feats", "glob_feats", "knots", "L", "class_means",
        "W", "b", "prior", "resid_sd", "col_mask", "is_const",
        "X_qry", "y_qry", "q_mask",
    ]
    out: dict[str, Any] = {k: np.stack([e[k] for e in episodes], axis=0) for k in keys}
    for k, dt in (("task_id", np.int64), ("k_eff", np.int64), ("packet_bytes", np.float32), ("n_fit", np.float32), ("n_noise", np.float32)):
        out[k] = np.array([e[k] for e in episodes], dtype=dt)
    for k, dv in (
        ("is_synth_prior", 0), ("is_swapped", 0), ("is_cropped", 0),
        ("col_crop_ratio", 1.0), ("disc_col_share", 0.0),
    ):
        out[k] = np.array([e.get(k, dv) for e in episodes], dtype=np.float32)
    if not as_numpy:
        out = {k: torch.from_numpy(v) for k, v in out.items()}
    out["dataset_hashes"] = [str(e.get("dataset_hash", "")) for e in episodes]
    out["split_hashes"] = [str(e.get("split_hash", "")) for e in episodes]
    return out


from .s3_v7_losses import compute_v7_batch_loss  # noqa: E402,F401
