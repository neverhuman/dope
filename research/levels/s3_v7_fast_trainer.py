"""Screaming multi-node CPU-DDP trainer for SingleEncV7 (V7.2: Max-Entropy S3 + Q_nl + Full Disk Telemetry).

Reads the memory-mapped flat corpus (`/dev/shm/dope-flat/s3-v250`), synthesizes exact per-step
shuffled tri-task batches (6 reg / 5 bin / 5 mc) mixing ALL 4,145 S3 tables (including >8-class
cropped multiclass) + 20% infinite synthetic nonlinear SCM/tree priors + full row/col/class
cropping + target swaps + missingness/warps, logs EVERY step loss-decomposition & entropy telemetry
to disk (`train.jsonl`), saves canonical F32 rank-<=2 SafeTensors checkpoints, and evaluates strictly
on the isolated 30-table DEV list (never touching CONFIRM).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import datetime
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .s3_v7_batch_ops import (
    FlatArenaView, _fit_only_scale, _hash_u64,
    build_unique_v7_episode, collate_v7_batch, compute_v7_batch_loss,
)
from .s3_v7_entropy import RealDatasetOnePassQueue, diloco_sync_step, task_quota_for_step
from .s3_v7_eval import (
    append_jsonl as _append_jsonl, atomic_json as _atomic_json,
    compute_lr_factor, eval_binary, eval_multiclass, eval_realism_and_privacy, eval_regression,
    freeze_target_heads, load_v7_resume_state, save_v7_resume_state, save_v7_safetensors,
    save_v7_valley_soup, split_dev_table_abt,
)
from .v7_single_encoding import (
    MAX_CLASSES, MAX_COLS, MAX_RANK, PACKET_BUDGET_BYTES,
    SingleEncV7, extract_dse_packet_numpy,
)


@torch.no_grad()
def evaluate_v7_dev_only(
    model: SingleEncV7,
    val_arena: FlatArenaView,
    dev_hashes: list[str],
    step: int,
    seed: int = 20261122,
) -> dict[str, Any]:
    """Held-out evaluation strictly restricted to `dev_hashes` (NEVER touches CONFIRM!)."""
    model.eval()
    t0 = time.time()
    rng = np.random.default_rng(seed)
    rets, by_t = [], {"regression": [], "binary": [], "multiclass": []}
    parities, realisms, privs, drvs, c2sts, margs, dcrs, mias, copies, b_sizes, within_caps = (
        [], [], [], [], [], [], [], [], [], [], []
    )
    for dh in dev_hashes:
        idx = val_arena.by_hash.get(dh)
        if idx is None:
            continue
        X_raw, y_raw, dg, rec = val_arena.load_slice(idx)
        task = str(rec["task"])
        spl = split_dev_table_abt(X_raw, y_raw, dg, task, rng, _fit_only_scale, MAX_COLS, MAX_CLASSES)
        if spl is None:
            continue
        Xa, ya, Xb, yb, Xt, yt, task_id, k_cls = spl

        pkt = extract_dse_packet_numpy(Xa, ya, task_id, k_cls, max_bytes=PACKET_BUDGET_BYTES)
        p_eff = pkt["p_eff"]
        n_syn = max(len(Xa) * 2, 256)
        pkt["X_qry"] = np.zeros((16, MAX_COLS), dtype=np.float32)
        pkt["y_qry"] = np.zeros((16,), dtype=np.float32)
        pkt["q_mask"] = np.ones((16,), dtype=np.float32)
        pkt["n_fit"] = len(Xa)
        pkt["n_noise"] = 0
        batch = collate_v7_batch([pkt])
        gen = torch.Generator().manual_seed(seed ^ (idx * 7919))
        zf = torch.randn((1, n_syn, MAX_RANK), generator=gen)
        zn = torch.randn((1, n_syn, MAX_COLS), generator=gen)
        yn = torch.randn((1, n_syn), generator=gen)
        x_s, lg_s, y_r = model.sample_batch(batch, n_syn, zf, zn, yn)
        Xs = x_s[0, :, :p_eff].numpy()
        if task_id == 0:
            res = eval_regression(Xa[:, :p_eff], ya, Xb[:, :p_eff], yb, Xs, y_r[0].numpy(), Xt[:, :p_eff], yt)
        elif task_id == 1:
            p1 = torch.sigmoid(lg_s[0, :, 0]).numpy()
            ys = (torch.rand((n_syn,), generator=gen).numpy() < p1).astype(np.float32)
            res = eval_binary(Xa[:, :p_eff], ya, Xb[:, :p_eff], yb, Xs, ys, Xt[:, :p_eff], yt)
        else:
            probs = torch.softmax(lg_s[0, :, :k_cls], dim=-1).numpy()
            ys = (np.cumsum(probs, axis=-1) < torch.rand((n_syn, 1), generator=gen).numpy()).sum(axis=-1).clip(0, k_cls - 1).astype(np.float32)
            res = eval_multiclass(Xa[:, :p_eff], ya, Xb[:, :p_eff], yb, Xs, ys, Xt[:, :p_eff], yt, k_cls)
        rp = eval_realism_and_privacy(Xa[:, :p_eff], Xb[:, :p_eff], Xs, Xt[:, :p_eff])
        b_sizes.append(pkt["packet_bytes"])
        within_caps.append(1.0 if pkt["packet_bytes"] <= PACKET_BUDGET_BYTES else 0.0)
        if math.isfinite(res["retention"]):
            rets.append(res["retention"])
            by_t[task].append(res["retention"])
        if math.isfinite(res["parity"]):
            parities.append(res["parity"])
        if math.isfinite(res["driver_spearman"]):
            drvs.append(res["driver_spearman"])
        realisms.append(rp["realism_pass"])
        privs.append(rp["privacy_pass"])
        c2sts.append(rp["c2st_auc"])
        margs.append(rp["marg_auc"])
        dcrs.append(rp["dcr_share"])
        mias.append(rp["mia_auc"])
        copies.append(rp["exact_copy_excess"])

    model.train()
    med = lambda xs: round(float(np.median(xs)), 4) if xs else None
    mean = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    return {
        "time": round(time.time(), 3), "step": int(step), "arch": "single_enc_v7",
        "recipe": "v7.2_max_entropy_qnl", "artifact_class": "source_free",
        "eval_role": "dev_only_ema_unconfirmed", "eval_wall_s": round(time.time() - t0, 2),
        "tables_evaluated": len(rets), "tables_total": len(b_sizes),
        "within_cap_rate": mean(within_caps), "encoding_bytes_median": med(b_sizes),
        "encoding_bytes_max": max(b_sizes, default=0), "main_kpi_retention_median": med(rets),
        "main_kpi_retention_mean": mean(rets), "regression_retention": med(by_t["regression"]),
        "binary_retention": med(by_t["binary"]), "multiclass_retention": med(by_t["multiclass"]),
        "multilabel_retention": None, "parity_coverage": mean(parities),
        "realism_coverage": mean(realisms), "privacy_pass_coverage": mean(privs),
        "driver_spearman_median": med(drvs), "c2st_auc_median": med(c2sts),
        "marg_auc_median": med(margs), "dcr_share_median": med(dcrs),
        "mia_auc_median": med(mias), "exact_copy_excess_median": med(copies),
        "formal_dp_state": "not_claimed",
    }


def _fetch_microbatch(
    arena: FlatArenaView, seed: int, step: int, rank: int, world_size: int,
    micro_bsz: int, n_fit_cap: int, seen_splits: set[str] | None = None,
    real_queue: RealDatasetOnePassQueue | None = None,
) -> dict[str, Any] | None:
    """Enforce exact tri-task quota, zero split repeats, and optional zero dataset repeats."""
    quota = task_quota_for_step(micro_bsz, _hash_u64(seed, step, rank))
    eps: list[dict[str, Any]] = []
    slots_per_step = world_size * micro_bsz
    base_slot = rank * micro_bsz
    for qi, want_task in enumerate(quota):
        for retry in range(12):
            ep = build_unique_v7_episode(
                arena, seed + retry * 1_000_003, step, base_slot + qi, n_fit_cap=n_fit_cap,
                n_qry_cap=128, desired_task=want_task, slots_per_step=slots_per_step, real_queue=real_queue,
            )
            if ep is None:
                continue
            sh = str(ep["split_hash"])
            if seen_splits is not None and sh in seen_splits:
                continue
            if seen_splits is not None:
                seen_splits.add(sh)
            eps.append(ep)
            break
    return collate_v7_batch(eps) if len(eps) >= 2 else None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CPU-DDP/DiLoCo SingleEncV7 Trainer (V7.2/V7.3 Zero-Repeat)")
    for flag, kw in (
        ("--arena-dir", dict(type=str, default="/dev/shm/dope-flat/s3-v250")),
        ("--run-dir", dict(type=str, required=True)), ("--run-name", dict(type=str, default="v72_run")),
        ("--dev-hashes-file", dict(type=str, default="")), ("--steps", dict(type=int, default=12000)),
        ("--micro-batch", dict(type=int, default=16)), ("--grad-accum", dict(type=int, default=1)),
        ("--n-syn", dict(type=int, default=192)), ("--d-model", dict(type=int, default=128)),
        ("--n-layers", dict(type=int, default=2)), ("--h-target", dict(type=int, default=24)),
        ("--n-heads", dict(type=int, default=4)), ("--device", dict(type=str, default="cpu")),
        ("--lr", dict(type=float, default=2e-4)), ("--weight-decay", dict(type=float, default=1e-2)),
        ("--grad-clip", dict(type=float, default=1.0)), ("--warmup-steps", dict(type=int, default=100)),
        ("--ema-decay", dict(type=float, default=0.992)), ("--eval-every", dict(type=int, default=150)),
        ("--ckpt-every", dict(type=int, default=150)), ("--log-every", dict(type=int, default=1)),
        ("--torch-threads", dict(type=int, default=1)), ("--prefetch-workers", dict(type=int, default=0)),
        ("--seed", dict(type=int, default=2026)), ("--resume", dict(type=str, default="")),
        ("--lr-schedule", dict(type=str, default="cosine", choices=["cosine", "wsd"])),
        ("--soup-k", dict(type=int, default=3)), ("--hard-tstr-v73", dict(action="store_true")),
        ("--freeze-target-heads", dict(action="store_true")),
        ("--never-repeat-datasets", dict(action="store_true")),
        ("--shard-id", dict(type=int, default=0)), ("--num-shards", dict(type=int, default=1)),
        ("--sync-dir", dict(type=str, default="")), ("--sync-every", dict(type=int, default=50)),
        ("--sync-outer-lr", dict(type=float, default=0.7)), ("--sync-outer-momentum", dict(type=float, default=0.9)),
    ):
        ap.add_argument(flag, **kw)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=15))

    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    sync_mode = "diloco_outer_momentum" if args.sync_dir else ("ddp_gloo" if world_size > 1 else "independent_ws1")
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / "config.json", {
            "arch": "single_enc_v7", "recipe": "v7.3_zero_repeat_qnl" if args.hard_tstr_v73 else "v7.2_max_entropy_qnl",
            "col_in_dim": 46, "artifact_class": "source_free", "run_name": args.run_name,
            "world_size": world_size, "sync_mode": sync_mode, "never_repeat_datasets": bool(args.never_repeat_datasets),
            "shard_id": args.shard_id, "num_shards": args.num_shards,
            "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
            "global_batch": world_size * args.micro_batch * args.grad_accum,
            "d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
            "h_target": args.h_target, "lr": args.lr, "seed": args.seed,
            "budget_bytes": PACKET_BUDGET_BYTES, "scheduler": "SCHED_IDLE_CPU_FLEET",
        })

    train_arena = FlatArenaView(args.arena_dir, split="train")
    val_arena = FlatArenaView(args.arena_dir, split="val") if rank == 0 else None
    dev_hashes: list[str] = []
    if rank == 0 and args.dev_hashes_file and Path(args.dev_hashes_file).is_file():
        dev_hashes = [l.strip() for l in Path(args.dev_hashes_file).read_text().splitlines() if l.strip()]

    base_model = SingleEncV7(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, h_target=args.h_target)
    if args.freeze_target_heads:
        freeze_target_heads(base_model)
    if world_size > 1:
        for p in base_model.parameters():
            dist.broadcast(p.data, src=0)
    ema_model = copy.deepcopy(base_model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(
        [p for p in base_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )
    top_dev_ckpts: list[tuple[float, str]] = []
    arch_sig = (args.d_model, args.n_layers, args.n_heads, args.h_target)
    resume_meta = {
        "arch": "single_enc_v7", "recipe": "v7.2_max_entropy_qnl", "d_model": args.d_model,
        "n_layers": args.n_layers, "n_heads": args.n_heads, "h_target": args.h_target, "seed": args.seed,
    }
    start_step, opt_steps, episodes_seen, best_dev_kpi = 0, 0, 0, -999.0
    seen_splits: set[str] = set()
    seen_datasets: set[str] = set()
    resume_p: Path | None = None
    if args.resume == "auto":
        cand = run_dir / "checkpoints" / "latest_resume.pt"
        if cand.is_file():
            resume_p = cand
    elif args.resume:
        resume_p = Path(args.resume)
    if resume_p is not None and rank == 0:
        start_step, opt_steps, episodes_seen, best_dev_kpi, seen_splits, seen_datasets = load_v7_resume_state(
            resume_p, base_model, ema_model, opt, arch_sig
        )
        print(f"[V7.2 RESUME] loaded {resume_p} at step={start_step} opt_steps={opt_steps} best_dev={best_dev_kpi:.4f}", flush=True)
    if world_size > 1:
        st_t = torch.tensor([start_step, opt_steps, episodes_seen], dtype=torch.long)
        dist.broadcast(st_t, src=0)
        start_step, opt_steps, episodes_seen = int(st_t[0].item()), int(st_t[1].item()), int(st_t[2].item())
        for p in base_model.parameters():
            dist.broadcast(p.data, src=0)
        for p in ema_model.parameters():
            dist.broadcast(p.data, src=0)
    real_queue = (
        RealDatasetOnePassQueue(train_arena, args.shard_id, args.num_shards, args.seed, seen_datasets)
        if args.never_repeat_datasets else None
    )
    anchor_state: dict[str, np.ndarray] = (
        {k: v.detach().cpu().to(torch.float32).numpy().copy() for k, v in base_model.state_dict().items()}
        if args.sync_dir else {}
    )
    sync_round, global_sha_short = 0, ""
    if args.sync_dir and rank == 0:
        sync_round, global_sha_short = diloco_sync_step(
            args.sync_dir, base_model, ema_model, anchor_state, args.sync_outer_lr, args.sync_outer_momentum
        )
    if rank == 0:
        print(
            f"[V7.2 Max-Entropy] world_size={world_size} global_batch={world_size * args.micro_batch * args.grad_accum} "
            f"params={base_model.parameter_count():,} ({base_model.parameter_count() * 4 / 1e6:.2f} MB F32) start_step={start_step}",
            flush=True,
        )

    if start_step == 0 and rank == 0 and val_arena is not None and dev_hashes:
        ev0 = evaluate_v7_dev_only(ema_model, val_arena, dev_hashes, step=0)
        _append_jsonl(run_dir / "eval.jsonl", ev0)
        best_dev_kpi = float(ev0.get("main_kpi_retention_median") or -999.0)
        print(
            f"[V7.2 DEV step=0] kpi={ev0['main_kpi_retention_median']} reg={ev0['regression_retention']} "
            f"bin={ev0['binary_retention']} mc={ev0['multiclass_retention']} c2st={ev0['c2st_auc_median']} "
            f"bytes={ev0['encoding_bytes_median']}B",
            flush=True,
        )

    pool = ThreadPoolExecutor(max_workers=args.prefetch_workers) if args.prefetch_workers > 0 else None
    n_fit_cap = int(256 + 256 * min(1.0, (start_step + 1) / max(1.0, args.steps * 0.5)))
    fut_next = (
        pool.submit(_fetch_microbatch, train_arena, args.seed, start_step + 1, rank, world_size, args.micro_batch, n_fit_cap, seen_splits, real_queue)
        if (pool is not None and start_step < args.steps) else None
    )

    zero_valid_streak, win_tables = 0, 0
    run_t0 = time.time()
    win_t0 = run_t0
    m_keys = ("loss", "cond_loss", "tstr_loss", "mmd_loss", "marg_loss", "corr_loss", "swd_loss", "anchor_loss", "train_unclipped_retention")

    try:
        for step in range(start_step + 1, args.steps + 1):
            prog = min(1.0, step / max(1.0, args.steps * 0.5))
            n_fit_cap = int(256 + 256 * prog)
            opt.zero_grad(set_to_none=True)
            acc = {k: 0.0 for k in m_keys}
            acc_bytes, acc_nfit, acc_noise, acc_crop = [], [], [], []
            step_datasets: list[str] = []
            step_splits: list[str] = []
            c_reg = c_bin = c_mc = c_synth = c_swap = c_crop = 0
            valid_micros = 0

            for ga in range(args.grad_accum):
                if pool is not None and fut_next is not None:
                    batch = fut_next.result()
                    next_s = step + (1 if ga + 1 == args.grad_accum else 0)
                    fut_next = pool.submit(
                        _fetch_microbatch, train_arena, args.seed + (ga + 1) * 10007,
                        next_s, rank, world_size, args.micro_batch, n_fit_cap, seen_splits, real_queue,
                    )
                else:
                    batch = _fetch_microbatch(
                        train_arena, args.seed + ga * 10007, step, rank, world_size, args.micro_batch, n_fit_cap, seen_splits, real_queue
                    )
                if batch is None:
                    continue
                bsz = batch["col_mask"].shape[0]
                zf = torch.randn((bsz, args.n_syn, MAX_RANK))
                zn = torch.randn((bsz, args.n_syn, MAX_COLS))
                yn = torch.randn((bsz, args.n_syn))
                ref = base_model.refine_packet(batch["col_feats"], batch["glob_feats"], batch["col_mask"])
                x_s, lg_s, y_r = base_model.sample_batch(batch, args.n_syn, zf, zn, yn, ref=ref)
                lg_q, sd_q = base_model.predict_target_on_x(batch, ref, batch["X_qry"] * batch["col_mask"].unsqueeze(1))
                loss, m = compute_v7_batch_loss(
                    x_s, lg_s, y_r, batch, logits_qry=lg_q, sd_qry=sd_q, ref=ref, hard_tstr_v73=args.hard_tstr_v73
                )
                if not torch.isfinite(loss):
                    continue
                (loss / float(args.grad_accum)).backward()
                for k in m_keys:
                    acc[k] += m[k]
                acc_bytes.append(float(batch["packet_bytes"].mean().item()))
                acc_nfit.append(float(batch["n_fit"].mean().item()))
                acc_noise.append(float(batch["n_noise"].mean().item()))
                acc_crop.append(float(batch["col_crop_ratio"].mean().item()))
                step_datasets.extend(batch["dataset_hashes"])
                step_splits.extend(batch["split_hashes"])
                tid = batch["task_id"]
                c_reg += int((tid == 0).sum().item())
                c_bin += int((tid == 1).sum().item())
                c_mc += int((tid == 2).sum().item())
                c_synth += int(batch["is_synth_prior"].sum().item())
                c_swap += int(batch["is_swapped"].sum().item())
                c_crop += int(batch["is_cropped"].sum().item())
                valid_micros += 1

            if valid_micros == 0:
                zero_valid_streak += 1
                if zero_valid_streak >= 3:
                    raise RuntimeError("abort_3_consecutive_zero_valid_steps")
                continue
            zero_valid_streak = 0

            if world_size > 1:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(float(world_size))

            lr_now = args.lr * compute_lr_factor(step, args.steps, args.warmup_steps, args.lr_schedule)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            gnorm = float(torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip).item())
            opt.step()
            opt_steps += 1

            with torch.no_grad():
                d_ema = min(args.ema_decay, (1.0 + opt_steps) / (10.0 + opt_steps))
                for pe, pb in zip(ema_model.parameters(), base_model.parameters()):
                    pe.data.mul_(d_ema).add_(pb.data, alpha=1.0 - d_ema)
            if args.sync_dir and rank == 0 and args.sync_every > 0 and (step % args.sync_every == 0 or step == args.steps):
                sync_round, global_sha_short = diloco_sync_step(
                    args.sync_dir, base_model, ema_model, anchor_state, args.sync_outer_lr, args.sync_outer_momentum
                )

            step_tables = world_size * args.micro_batch * valid_micros
            episodes_seen += step_tables
            win_tables += step_tables
            if rank == 0:
                _append_jsonl(run_dir / "exposure.jsonl", {
                    "step": step, "time": round(time.time(), 3),
                    "datasets": step_datasets, "splits": step_splits,
                })

            if rank == 0 and (step % args.log_every == 0 or step == 1):
                now = time.time()
                tbl_s = win_tables / max(now - win_t0, 1e-3)
                win_tables, win_t0 = 0, now
                row = {
                    "time": round(now, 3), "wall_s": round(now - run_t0, 2),
                    "step": step, "opt_steps": opt_steps, "sync_mode": sync_mode,
                    "sync_round": sync_round, "global_sha": global_sha_short,
                    "arch": "single_enc_v7", "recipe": "v7.2_max_entropy_qnl", "artifact_class": "source_free",
                    **{k: round(acc[k] / valid_micros, 4) for k in m_keys},
                    "grad_norm": round(gnorm, 4), "lr": round(lr_now, 6),
                    "tables_per_s": round(tbl_s, 2), "episodes_seen": episodes_seen,
                    "real_queue_remaining": real_queue.remaining() if real_queue is not None else -1,
                    "encoding_bytes_avg": round(float(np.mean(acc_bytes)), 1),
                    "avg_n_fit": round(float(np.mean(acc_nfit)), 1),
                    "avg_noise_cols": round(float(np.mean(acc_noise)), 2),
                    "avg_col_crop_ratio": round(float(np.mean(acc_crop)), 3),
                    "n_reg": c_reg, "n_bin": c_bin, "n_mc": c_mc,
                    "n_synth_prior": c_synth, "n_swapped": c_swap, "n_cropped": c_crop,
                }
                _append_jsonl(run_dir / "train.jsonl", row)
                if step % max(args.log_every * 10, 10) == 0 or step == 1:
                    print(
                        f"step={step:4d} | {tbl_s:6.1f} tbl/s | loss={row['loss']:.4f} cond={row['cond_loss']:.4f} "
                        f"tstr={row['tstr_loss']:.4f} mmd={row['mmd_loss']:.4f} anc={row['anchor_loss']:.4f} "
                        f"mix={c_reg}/{c_bin}/{c_mc} syn={c_synth} swp={c_swap} crp={c_crop} bytes={row['encoding_bytes_avg']:.0f}B",
                        flush=True,
                    )

            if rank == 0 and opt_steps > 0 and (step % args.ckpt_every == 0 or step == args.steps):
                ck_name = f"step_{step:06d}_ema.safetensors"
                ck_bytes, ck_sha = save_v7_safetensors(ema_model, run_dir / "checkpoints" / ck_name)
                ev = evaluate_v7_dev_only(ema_model, val_arena, dev_hashes, step=step) if (val_arena and dev_hashes) else None
                role = "step_ema"
                if ev is not None:
                    ev["checkpoint"] = ck_name
                    _append_jsonl(run_dir / "eval.jsonl", ev)
                    kpi_now = float(ev.get("main_kpi_retention_median") or -999.0)
                    print(
                        f"[V7.2 DEV step={step}] kpi={ev['main_kpi_retention_median']} reg={ev['regression_retention']} "
                        f"bin={ev['binary_retention']} mc={ev['multiclass_retention']} c2st={ev['c2st_auc_median']} "
                        f"priv={ev['privacy_pass_coverage']} bytes={ev['encoding_bytes_median']}B",
                        flush=True,
                    )
                    if kpi_now > best_dev_kpi:
                        best_dev_kpi, role = kpi_now, "best_main_kpi"
                        save_v7_safetensors(ema_model, run_dir / "checkpoints" / "dev_best_ema.safetensors")
                    if args.soup_k >= 2 and math.isfinite(kpi_now):
                        top_dev_ckpts.append((kpi_now, ck_name))
                        top_dev_ckpts.sort(key=lambda x: x[0], reverse=True)
                        del top_dev_ckpts[args.soup_k :]
                        if len(top_dev_ckpts) >= 2:
                            save_v7_valley_soup(
                                [run_dir / "checkpoints" / nm for _, nm in top_dev_ckpts],
                                run_dir / "checkpoints" / f"dev_soup_k{args.soup_k}_ema.safetensors",
                            )
                save_v7_resume_state(
                    run_dir / "checkpoints" / "latest_resume.pt", base_model, ema_model, opt,
                    step, opt_steps, episodes_seen, best_dev_kpi, resume_meta,
                    seen_splits=seen_splits, seen_datasets=seen_datasets,
                )
                _append_jsonl(run_dir / "checkpoints" / "ledger.jsonl", {
                    **(ev or {}), "checkpoint": ck_name, "role": role, "arch": "single_enc_v7",
                    "recipe": "v7.2_max_entropy_qnl", "step": step, "time": round(time.time(), 3),
                    "bytes": ck_bytes, "sha256": ck_sha[:16],
                })
                _atomic_json(run_dir / "latest.json", {
                    "step": step, "opt_steps": opt_steps, "checkpoint": ck_name, "bytes": ck_bytes,
                    "sha256": ck_sha, "resume_checkpoint": "latest_resume.pt", "arch": "single_enc_v7",
                    "recipe": "v7.2_max_entropy_qnl", "artifact_class": "source_free",
                    "dev_best_unconfirmed_kpi": best_dev_kpi,
                })
                win_tables, win_t0 = 0, time.time()
    finally:
        if pool is not None:
            pool.shutdown(wait=False)
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
