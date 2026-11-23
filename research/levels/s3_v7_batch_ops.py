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


def build_unique_v7_episode(
    arena: FlatArenaView,
    seed: int,
    step: int,
    global_slot: int,
    n_fit_cap: int = 384,
    n_qry_cap: int = 128,
    max_bytes: int = PACKET_BUDGET_BYTES,
    desired_task: str | None = None,
    slots_per_step: int = 16,
    real_queue: Any | None = None,
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

    # Log-uniform sample-size cropping EVERY episode so model masters tiny (32) AND large (n_fit_cap) tables!
    n_fit_target = int(round(math.exp(rng.uniform(math.log(32.0), math.log(max(64.0, float(n_fit_cap)))))))
    n_qry_target = int(round(math.exp(rng.uniform(math.log(24.0), math.log(max(32.0, float(n_qry_cap)))))))

    use_synth = rng.random() < (0.75 if real_queue is not None else 0.20)
    tbl_idx: int | None = None
    from_reg_synth = False
    if not use_synth:
        from_reg_synth = (task in ("binary", "multiclass")) and (rng.random() < 0.15)
        src_task = "regression" if from_reg_synth else task
        if real_queue is not None:
            tbl_idx = real_queue.pop_unseen(src_task)
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
        # Fresh duplicate-grouped random train/test split EVERY batch (wide ratio [0.35, 0.75])!
        uniq_g = np.unique(dg)
        ratio = float(rng.uniform(0.35, 0.75))
        if len(uniq_g) < 16:
            perm = rng.permutation(nr)
            cut = max(24, min(nr - 16, int(nr * ratio)))
            fit_idx, qry_idx = perm[:cut], perm[cut:]
        else:
            g_perm = rng.permutation(uniq_g)
            g_cut = max(8, min(len(g_perm) - 6, int(len(g_perm) * ratio)))
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

    # Append nonlinear interactions + distractors, then mild FIT-only Gaussian jitter.
    Xa, Xq, n_noise = append_nonlinear_and_distractor_cols(rng, Xa, Xq, max_cols=MAX_COLS)
    Xa = np.clip(Xa + rng.normal(0.0, 0.015, size=Xa.shape).astype(np.float32), -6.0, 6.0)

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
        uniq_c = np.unique(np.rint(ya).astype(np.int64))
        if len(uniq_c) < 3 or len(uniq_c) > MAX_CLASSES:
            return None
        perm_ids = rng.permutation(len(uniq_c))
        mp = {int(c): int(perm_ids[i]) for i, c in enumerate(uniq_c)}
        split_h.update(json.dumps(mp, sort_keys=True).encode("utf-8"))
        ya = np.array([mp.get(int(round(float(v))), 0) for v in ya], dtype=np.float32)
        yq = np.array([mp.get(int(round(float(v))), 0) for v in yq], dtype=np.float32)
        n_cls = len(uniq_c)

    pkt = extract_dse_packet_numpy(Xa, ya, task_id, n_cls, max_bytes=max_bytes)
    p_eff = pkt["p_eff"]
    Xq_pad = np.zeros((n_qry_cap, MAX_COLS), dtype=np.float32)
    yq_pad = np.zeros((n_qry_cap,), dtype=np.float32)
    q_mask = np.zeros((n_qry_cap,), dtype=np.float32)
    nq = min(len(Xq), n_qry_cap)
    Xq_pad[:nq, :p_eff] = Xq[:nq, :p_eff]
    yq_pad[:nq] = yq[:nq]
    q_mask[:nq] = 1.0
    pkt.update(
        X_qry=Xq_pad, y_qry=yq_pad, q_mask=q_mask, n_fit=len(Xa), n_noise=n_noise,
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
    for k, dv in (("is_synth_prior", 0), ("is_swapped", 0), ("is_cropped", 0), ("col_crop_ratio", 1.0)):
        out[k] = np.array([e.get(k, dv) for e in episodes], dtype=np.float32)
    if not as_numpy:
        out = {k: torch.from_numpy(v) for k, v in out.items()}
    out["dataset_hashes"] = [str(e.get("dataset_hash", "")) for e in episodes]
    out["split_hashes"] = [str(e.get("split_hash", "")) for e in episodes]
    return out


def compute_v7_batch_loss(
    x_syn: torch.Tensor,
    logits_syn: torch.Tensor,
    y_reg_syn: torch.Tensor,
    batch: dict[str, torch.Tensor],
    logits_qry: torch.Tensor | None = None,
    sd_qry: torch.Tensor | None = None,
    ref: dict[str, torch.Tensor] | None = None,
    hard_tstr_v73: bool = False,
    rff_dim: int = 16,
    swd_projs: int = 24,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiable multi-auditor loss + proper-scoring-rule NLL + trust-region residual anchor."""
    bsz, n_syn, p_dim = x_syn.shape
    device = x_syn.device
    col_mask, X_qry, y_qry, q_mask, task_id, k_eff = (
        batch["col_mask"], batch["X_qry"] * batch["col_mask"].unsqueeze(1),
        batch["y_qry"], batch["q_mask"], batch["task_id"], batch["k_eff"],
    )

    # 1. Differentiable RFF + Linear TSTR surrogate on (X_syn -> QRY)
    w_rff = torch.randn((p_dim, rff_dim), device=device, dtype=x_syn.dtype) * 0.45
    b_rff = torch.rand((1, 1, rff_dim), device=device, dtype=x_syn.dtype) * (2.0 * math.pi)
    phi_syn = torch.cat(
        [x_syn, torch.cos(torch.matmul(x_syn, w_rff) + b_rff) * (2.0 / rff_dim) ** 0.5, torch.ones((bsz, n_syn, 1), device=device)],
        dim=-1,
    )
    phi_qry = torch.cat(
        [X_qry, torch.cos(torch.matmul(X_qry, w_rff) + b_rff) * (2.0 / rff_dim) ** 0.5, torch.ones((bsz, X_qry.shape[1], 1), device=device)],
        dim=-1,
    )
    f_dim = phi_syn.shape[-1]
    reg_diag = torch.full((f_dim,), 2.0, device=device, dtype=torch.float32)
    if hard_tstr_v73:
        reg_diag[-1] = 1e-4
    gram_s = torch.matmul(phi_syn.transpose(1, 2), phi_syn) + torch.diag(reg_diag).unsqueeze(0)

    y_soft = torch.zeros((bsz, n_syn, MAX_CLASSES), device=device, dtype=torch.float32)
    reg_m, bin_m, mc_m = task_id == 0, task_id == 1, task_id == 2
    if reg_m.any():
        y_soft[reg_m, :, 0] = y_reg_syn[reg_m]
    if bin_m.any():
        p_b = torch.sigmoid(logits_syn[bin_m, :, 0].clamp(-8.0, 8.0))
        if hard_tstr_v73:
            y_h = (torch.rand_like(p_b) < p_b).float()
            y_soft[bin_m, :, 0] = y_h - p_b.detach() + p_b
        else:
            y_soft[bin_m, :, 0] = p_b
    if mc_m.any():
        c_idx = torch.arange(MAX_CLASSES, device=device).unsqueeze(0)
        valid_c = c_idx < k_eff[mc_m].unsqueeze(1)
        lg_mc = logits_syn[mc_m].clamp(-8.0, 8.0).masked_fill(~valid_c.unsqueeze(1), -1e4)
        p_mc = torch.softmax(lg_mc, dim=-1)
        if hard_tstr_v73:
            u_g = torch.rand_like(lg_mc).clamp(1e-5, 1.0 - 1e-5)
            g_idx = (lg_mc - torch.log(-torch.log(u_g))).argmax(dim=-1)
            y_oh = torch.nn.functional.one_hot(g_idx, num_classes=MAX_CLASSES).float()
            y_soft[mc_m] = y_oh - p_mc.detach() + p_mc
        else:
            y_soft[mc_m] = p_mc

    beta_s = torch.linalg.solve(gram_s, torch.matmul(phi_syn.transpose(1, 2), y_soft))
    pred_q = torch.matmul(phi_qry, beta_s)

    tstr_losses, ret_vals = [], []
    q_denom = q_mask.sum(dim=1).clamp_min(1.0)
    if reg_m.any():
        err_s = ((pred_q[reg_m, :, 0] - y_qry[reg_m]) ** 2 * q_mask[reg_m]).sum(dim=1) / q_denom[reg_m]
        tstr_losses.append(err_s.clamp(max=25.0).mean())
        ym = (y_qry[reg_m] * q_mask[reg_m]).sum(dim=1, keepdim=True) / q_denom[reg_m].unsqueeze(1)
        var_y = (((y_qry[reg_m] - ym) ** 2) * q_mask[reg_m]).sum(dim=1) / q_denom[reg_m]
        ok_v = var_y > 0.05
        if ok_v.any():
            ret_vals.extend((1.0 - err_s[ok_v] / var_y[ok_v]).detach().cpu().tolist())
    if bin_m.any():
        p_q = pred_q[bin_m, :, 0].clamp(1e-4, 1.0 - 1e-4)
        yb = y_qry[bin_m].clamp(0.0, 1.0)
        bce = -(yb * torch.log(p_q) + (1.0 - yb) * torch.log(1.0 - p_q))
        tstr_losses.append(((bce * q_mask[bin_m]).sum(dim=1) / q_denom[bin_m]).clamp(max=10.0).mean())
    if mc_m.any():
        c_idx = torch.arange(MAX_CLASSES, device=device).unsqueeze(0)
        valid_c = c_idx < k_eff[mc_m].unsqueeze(1)
        probs_q = pred_q[mc_m].clamp(min=1e-4) * valid_c.unsqueeze(1).float()
        probs_q = probs_q / probs_q.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        yi = y_qry[mc_m].long().clamp(0, MAX_CLASSES - 1)
        p_true = torch.gather(probs_q, 2, yi.unsqueeze(-1)).squeeze(-1).clamp(min=1e-4)
        tstr_losses.append(((-torch.log(p_true) * q_mask[mc_m]).sum(dim=1) / q_denom[mc_m]).clamp(max=10.0).mean())
    tstr_loss = sum(tstr_losses) / max(len(tstr_losses), 1)

    # 2. Marginal 1D-W1 + Correlation Frobenius + 24-projection Sliced Wasserstein
    q_grid = torch.linspace(0.05, 0.95, 17, device=device)
    qs_syn = torch.quantile(x_syn, q_grid, dim=1)
    n_q_use = max(16, min(int(q_mask.sum(dim=1).min().item()), X_qry.shape[1]))
    qs_qry = torch.quantile(X_qry[:, :n_q_use, :], q_grid, dim=1)
    marg_loss = ((qs_syn - qs_qry).abs().mean(dim=0) * col_mask).sum() / col_mask.sum().clamp_min(1.0)

    xs_c = x_syn - x_syn.mean(dim=1, keepdim=True)
    xq_c = X_qry[:, :n_q_use, :] - X_qry[:, :n_q_use, :].mean(dim=1, keepdim=True)
    xs_n = xs_c / xs_c.std(dim=1, keepdim=True).clamp_min(0.05)
    xq_n = xq_c / xq_c.std(dim=1, keepdim=True).clamp_min(0.05)
    c_syn = torch.matmul(xs_n.transpose(1, 2), xs_n) / float(n_syn)
    c_qry = torch.matmul(xq_n.transpose(1, 2), xq_n) / float(n_q_use)
    pair_mask = col_mask.unsqueeze(2) * col_mask.unsqueeze(1)
    corr_loss = (((c_syn - c_qry) ** 2) * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    proj = torch.randn((p_dim, swd_projs), device=device, dtype=x_syn.dtype)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-6)
    swd_loss = (
        torch.quantile(torch.matmul(x_syn, proj), q_grid, dim=1)
        - torch.quantile(torch.matmul(X_qry[:, :n_q_use, :], proj), q_grid, dim=1)
    ).abs().mean()

    # 3. Direct proper-scoring-rule conditional NLL on real held-out (X_qry, y_qry)
    cond_losses = []
    if logits_qry is not None and sd_qry is not None:
        if reg_m.any():
            mu_q = logits_qry[reg_m, :, 0]
            sd_v = sd_qry[reg_m].clamp(0.05, 4.0)
            mse_q = ((mu_q - y_qry[reg_m]) ** 2 * q_mask[reg_m]).sum(dim=1) / q_denom[reg_m]
            nll_q = (0.5 * (((mu_q - y_qry[reg_m]) / sd_v) ** 2 + 2.0 * torch.log(sd_v)) * q_mask[reg_m]).sum(dim=1) / q_denom[reg_m]
            cond_losses.append((mse_q.clamp(max=25.0) + 0.25 * nll_q.clamp(min=-2.0, max=10.0)).mean())
        if bin_m.any():
            bce_q = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_qry[bin_m, :, 0].clamp(-10.0, 10.0), y_qry[bin_m].clamp(0.0, 1.0), reduction="none"
            )
            cond_losses.append(((bce_q * q_mask[bin_m]).sum(dim=1) / q_denom[bin_m]).clamp(max=10.0).mean())
        if mc_m.any():
            c_idx = torch.arange(MAX_CLASSES, device=device).unsqueeze(0)
            valid_c = c_idx < k_eff[mc_m].unsqueeze(1)
            lg_mc = logits_qry[mc_m].clamp(-10.0, 10.0).masked_fill(~valid_c.unsqueeze(1), -1e4)
            logp = torch.log_softmax(lg_mc, dim=-1)
            yi = y_qry[mc_m].long().clamp(0, MAX_CLASSES - 1)
            nll_mc = -torch.gather(logp, 2, yi.unsqueeze(-1)).squeeze(-1)
            cond_losses.append(((nll_mc * q_mask[mc_m]).sum(dim=1) / q_denom[mc_m]).clamp(max=10.0).mean())
    cond_loss = sum(cond_losses) / max(len(cond_losses), 1) if cond_losses else tstr_loss

    # 4. Trust-region L2 anchor on neural packet residuals (eliminates warmup dip below analytic step-0!)
    anchor_loss = torch.zeros((), device=device, dtype=torch.float32)
    if ref is not None:
        anchor_loss = (
            0.04 * (ref["d_W"] ** 2).mean()
            + 0.05 * (ref["d_L"] ** 2).mean()
            + 0.05 * (ref["d_knots"] ** 2).mean()
            + 0.02 * (ref["temp_sd"] ** 2).mean()
        )
        if hard_tstr_v73:
            anchor_loss = (
                anchor_loss
                + 1.50 * (ref["d_W"] ** 2).mean() + 2.50 * (ref["w2"] ** 2).mean()
                + 0.50 * (ref["w1"] ** 2).mean() + 1.00 * (ref["temp_sd"] ** 2).mean()
                + 0.25 * (ref["d_knots"] ** 2).mean() + 0.25 * (ref["d_L"] ** 2).mean()
            )
            if bin_m.any():
                tau_scale = torch.exp(ref["temp_sd"][bin_m, 0])
                anchor_loss = (
                    anchor_loss
                    + 6.0 * (ref["d_W"][bin_m, 0] ** 2).mean() + 12.0 * (ref["w2"][bin_m, :, 0] ** 2).mean()
                    + 4.0 * (ref["temp_sd"][bin_m, 0] ** 2).mean() + 0.50 * (torch.relu(0.90 - tau_scale) ** 2).mean()
                )

    mmd_proxy = marg_loss + 0.5 * corr_loss + 0.5 * swd_loss
    total_loss = 0.70 * cond_loss + 0.30 * tstr_loss + 1.0 * marg_loss + 1.2 * corr_loss + 0.9 * swd_loss + anchor_loss
    metrics = {
        "loss": float(total_loss.detach().item()),
        "cond_loss": float(cond_loss.detach().item()),
        "tstr_loss": float(tstr_loss.detach().item()),
        "mmd_loss": float(mmd_proxy.detach().item()),
        "marg_loss": float(marg_loss.detach().item()),
        "corr_loss": float(corr_loss.detach().item()),
        "swd_loss": float(swd_loss.detach().item()),
        "anchor_loss": float(anchor_loss.detach().item()),
        "train_unclipped_retention": float(np.median(ret_vals)) if ret_vals else 0.0,
    }
    return total_loss, metrics
