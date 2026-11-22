"""Multi-node PyTorch DDP GPU trainer for DopeFidelityV5 over ALL S3 catalog/v250 datasets.

Supports 1..N GPU ranks across xbabe5/xbabe7/xbabe8/xbabe9 (or single CPU/GPU for tests) with:
- Strict per-process VRAM ceiling so co-hosted jsr jobs are never disrupted
- Fresh Rule A-1 duplicate-grouped random FIT/QUERY split on EVERY batch
- Dynamic synthetic column insertion (noise distractors + nonlinear derived cols) & random row lengths
- Unified Regression + Binary + Multiclass (S3 multi_label) + Multi-Label training
- Direct Differentiable Coupled Multi-Learner TSTR Retention (MAIN KPI) + Driver + Realism + InfoNCE
- Atomic safetensors checkpoints + 10-step train.jsonl telemetry for the live DOPE Dashboard
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from safetensors.torch import save_file
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .s3_dynamic_augment import AugmentConfig, S3CorpusStore, sample_dynamic_episode
from .transfer_jx_heads import split_invariance_infonce
from .v5_fidelity_net import DopeFidelityV5, compute_main_kpi_loss


def setup_distributed(vram_cap_gib: float | None, requested_device: str) -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if (requested_device == "cuda" and torch.cuda.is_available()) else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        rank, world_size, local_rank = 0, 1, 0

    if requested_device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if vram_cap_gib is not None and vram_cap_gib > 0:
            total_gib = torch.cuda.get_device_properties(local_rank).total_memory / (1024.0 ** 3)
            frac = min(max(vram_cap_gib / max(total_gib, 1.0), 0.08), 0.85)
            torch.cuda.set_per_process_memory_fraction(frac, device=local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def save_atomic_checkpoint(
    run_dir: Path,
    model_raw: DopeFidelityV5,
    ema_state: dict[str, torch.Tensor],
    step: int,
    meta: dict[str, Any],
) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step_name = f"step_{step:07d}.safetensors"
    step_tmp = ckpt_dir / f".{step_name}.tmp"
    ema_tensors = {k: v.detach().cpu().contiguous() for k, v in ema_state.items()}
    save_file(ema_tensors, str(step_tmp))
    os.replace(step_tmp, ckpt_dir / step_name)

    latest_tmp = ckpt_dir / ".latest.safetensors.tmp"
    save_file(ema_tensors, str(latest_tmp))
    os.replace(latest_tmp, ckpt_dir / "latest.safetensors")

    ptr = {
        "step": step,
        "checkpoint": step_name,
        "latest": "latest.safetensors",
        "d_model": model_raw.d_model,
        "n_layers": len(model_raw.blocks),
        "latent_dim": model_raw.latent_dim,
        "param_count": sum(p.numel() for p in model_raw.parameters()),
        "saved_at": time.time(),
        **meta,
    }
    ptr_tmp = ckpt_dir / ".latest.json.tmp"
    ptr_tmp.write_text(json.dumps(ptr, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(ptr_tmp, ckpt_dir / "latest.json")
    return ckpt_dir / step_name


def update_ema(ema_state: dict[str, torch.Tensor], model_raw: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for k, v in model_raw.state_dict().items():
            if k in ema_state and v.is_floating_point():
                ema_state[k].mul_(decay).add_(v.detach().float(), alpha=1.0 - decay)
            else:
                ema_state[k] = v.detach().clone()


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    rank, world_size, local_rank, device = setup_distributed(args.vram_cap_gib, args.device)
    torch.manual_seed(args.seed + rank * 1009)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank * 1009)
        torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.jsonl"

    store = S3CorpusStore(Path(args.corpus_dir), split="train", cache_cap=args.cache_cap)
    aug_cfg = AugmentConfig(
        min_fit_rows=args.min_fit_rows,
        max_fit_rows=args.max_fit_rows,
        min_query_rows=args.min_query_rows,
        max_query_rows=args.max_query_rows,
        aug_prob=args.aug_prob,
        max_noise_cols=args.max_noise_cols,
        max_derived_cols=args.max_derived_cols,
        multilabel_view_prob=args.multilabel_prob,
        enable_augmentation=not args.no_augment,
    )

    model_raw = DopeFidelityV5(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        latent_dim=128,
    ).to(device)
    param_count = sum(p.numel() for p in model_raw.parameters())
    ema_state = {k: v.detach().float().clone() for k, v in model_raw.state_dict().items()}

    if world_size > 1:
        model: torch.nn.Module = DDP(
            model_raw,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    else:
        model = model_raw

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))

    def lr_lambda(step_idx: int) -> float:
        if step_idx < args.warmup_steps:
            return float(step_idx + 1) / float(max(args.warmup_steps, 1))
        prog = (step_idx - args.warmup_steps) / float(max(args.total_steps - args.warmup_steps, 1))
        return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if rank == 0:
        cfg_dump = {
            "run_name": args.run_name,
            "model": "DopeFidelityV5",
            "param_count": param_count,
            "world_size": world_size,
            "device": str(device),
            "vram_cap_gib": args.vram_cap_gib,
            "corpus_train_tables": len(store),
            "corpus_by_task": {k: len(v) for k, v in store.by_task.items()},
            "total_steps": args.total_steps,
            "episodes_per_step_per_rank": args.episodes_per_step,
            "lr": args.lr,
            "aug_enabled": aug_cfg.enable_augmentation,
            "started_at": time.time(),
        }
        (run_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")
        print(
            f"[DOPE-V5 DDP] Starting {args.run_name}: params={param_count:,}, world_size={world_size}, "
            f"tables={len(store)} ({cfg_dump['corpus_by_task']}), vram_cap={args.vram_cap_gib} GiB",
            flush=True,
        )

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    t0 = time.time()
    last_log_t = t0
    seen_splits: set[str] = set()
    task_totals = {"regression": 0, "binary": 0, "multiclass": 0, "multilabel": 0}
    acc_metrics = {
        "loss": [],
        "tstr": [],
        "driver": [],
        "realism": [],
        "infonce": [],
        "ret": [],
        "n_fit": [],
        "n_cols": [],
        "n_noise": [],
        "n_derived": [],
    }

    import numpy as np

    rng_idx = np.random.default_rng(args.seed + rank * 7919)

    for step in range(1, args.total_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        step_loss_sum = 0.0
        step_tstr = 0.0
        step_drv = 0.0
        step_real = 0.0
        step_ret = 0.0
        valid_eps = 0
        z_views_a: list[torch.Tensor] = []
        z_views_b: list[torch.Tensor] = []
        fam_views: list[str] = []

        for ep_i in range(args.episodes_per_step):
            tbl_idx = store.sample_balanced_index(rng_idx)
            table = store.load_table(tbl_idx)
            ep = sample_dynamic_episode(
                table,
                global_seed=args.seed,
                step=step * 32 + ep_i,
                rank=rank,
                cfg=aug_cfg,
                view_tag="a",
            )
            seen_splits.add(ep["split_sig"])
            task_totals[ep["task"]] = task_totals.get(ep["task"], 0) + 1
            a_st = ep["aug_stats"]
            acc_metrics["n_fit"].append(a_st["n_fit"])
            acc_metrics["n_cols"].append(a_st["n_cols"])
            acc_metrics["n_noise"].append(a_st["n_noise_cols"])
            acc_metrics["n_derived"].append(a_st["n_derived_cols"])

            try:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    out = compute_main_kpi_loss(
                        model.module if isinstance(model, DDP) else model,
                        ep,
                        device,
                        step_seed=args.seed + step * 97 + rank * 13 + ep_i,
                    )
                loss_ep = out["total"] / float(args.episodes_per_step)
                if not bool(torch.isfinite(loss_ep).all()):
                    continue
                # Collect paired fresh random split view for InfoNCE split invariance on first 2 episodes
                if ep_i < 2:
                    ep_b = sample_dynamic_episode(
                        table,
                        global_seed=args.seed,
                        step=step * 32 + ep_i,
                        rank=rank,
                        cfg=aug_cfg,
                        view_tag="b",
                    )
                    seen_splits.add(ep_b["split_sig"])
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                        enc_b = (model.module if isinstance(model, DDP) else model).encode_table(
                            torch.as_tensor(ep_b["X_fit"], dtype=torch.float32, device=device),
                            torch.as_tensor(ep_b["y_fit"], dtype=torch.float32, device=device),
                            ep_b["task"],
                            int(ep_b["n_classes"]),
                        )
                    z_views_a.append(out["task_latent"])
                    z_views_b.append(enc_b["task_latent"])
                    fam_views.append(ep["family_stem"] or ep["dataset_hash"])

                loss_ep.backward()
                step_loss_sum += float(out["total"].detach().item())
                step_tstr += out["tstr_loss"]
                step_drv += out["driver_loss"]
                step_real += out["realism_loss"]
                step_ret += max(-1.0, min(2.0, out["unclipped_retention"]))
                valid_eps += 1
            except RuntimeError:
                continue

        infonce_val = 0.0
        if len(z_views_a) >= 2 and len(set(fam_views)) >= 2:
            try:
                za = torch.stack(z_views_a, dim=0).float()
                zb = torch.stack(z_views_b, dim=0).float()
                inv_loss, _ = split_invariance_infonce(za, zb, fam_views, temperature=0.15)
                if bool(torch.isfinite(inv_loss).all()):
                    (0.10 * inv_loss).backward()
                    infonce_val = float(inv_loss.detach().item())
            except Exception:
                pass

        # Ensure DDP gradient synchronization across ranks when world_size > 1
        if world_size > 1:
            for p in model_raw.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(float(world_size))

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model_raw.parameters(), 1.0).item())
        if math.isfinite(grad_norm) and valid_eps > 0:
            optimizer.step()
            scheduler.step()
            update_ema(ema_state, model_raw, decay=0.995 if step > 50 else 0.90)

        if valid_eps > 0:
            acc_metrics["loss"].append(step_loss_sum / valid_eps)
            acc_metrics["tstr"].append(step_tstr / valid_eps)
            acc_metrics["driver"].append(step_drv / valid_eps)
            acc_metrics["realism"].append(step_real / valid_eps)
            acc_metrics["infonce"].append(infonce_val)
            acc_metrics["ret"].append(step_ret / valid_eps)

        if step % args.log_every == 0 or step == 1 or step == args.total_steps:
            now = time.time()
            dt = max(now - last_log_t, 1e-4)
            n_logged = max(len(acc_metrics["loss"]), 1)
            tables_step = (args.log_every if step > 1 else 1) * args.episodes_per_step * world_size
            rec = {
                "step": step,
                "time": round(now, 3),
                "wall_s": round(now - t0, 2),
                "rank": rank,
                "world_size": world_size,
                "lr": round(float(scheduler.get_last_lr()[0]), 7),
                "grad_norm": round(grad_norm, 4) if math.isfinite(grad_norm) else 0.0,
                "loss": round(sum(acc_metrics["loss"]) / n_logged, 5),
                "tstr_loss": round(sum(acc_metrics["tstr"]) / n_logged, 5),
                "driver_loss": round(sum(acc_metrics["driver"]) / n_logged, 5),
                "realism_loss": round(sum(acc_metrics["realism"]) / n_logged, 5),
                "infonce_loss": round(sum(acc_metrics["infonce"]) / n_logged, 5),
                "train_unclipped_retention": round(sum(acc_metrics["ret"]) / n_logged, 4),
                "tables_per_s": round(tables_step / dt, 2),
                "unique_splits_seen": len(seen_splits) * world_size,
                "avg_n_fit": round(sum(acc_metrics["n_fit"]) / max(len(acc_metrics["n_fit"]), 1), 1),
                "avg_n_cols": round(sum(acc_metrics["n_cols"]) / max(len(acc_metrics["n_cols"]), 1), 1),
                "avg_noise_cols": round(sum(acc_metrics["n_noise"]) / max(len(acc_metrics["n_noise"]), 1), 2),
                "avg_derived_cols": round(sum(acc_metrics["n_derived"]) / max(len(acc_metrics["n_derived"]), 1), 2),
                "task_counts": copy.deepcopy(task_totals),
                "gpu_vram_mib": round(torch.cuda.memory_allocated(device) / (1024 ** 2), 1) if device.type == "cuda" else 0.0,
            }
            if rank == 0:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
                print(
                    f"[step {step:05d}/{args.total_steps}] loss={rec['loss']:.4f} "
                    f"tstr={rec['tstr_loss']:.4f} ret={rec['train_unclipped_retention']:.3f} "
                    f"drv={rec['driver_loss']:.4f} real={rec['realism_loss']:.4f} "
                    f"splits={rec['unique_splits_seen']} rate={rec['tables_per_s']:.1f} tbl/s "
                    f"vram={rec['gpu_vram_mib']:.0f} MiB",
                    flush=True,
                )
            for buf in acc_metrics.values():
                buf.clear()
            last_log_t = now

        if rank == 0 and (step % args.ckpt_every == 0 or step == args.total_steps):
            save_atomic_checkpoint(
                run_dir,
                model_raw,
                ema_state,
                step,
                {
                    "run_name": args.run_name,
                    "unique_splits_seen": len(seen_splits) * world_size,
                    "world_size": world_size,
                },
            )

    if world_size > 1:
        dist.destroy_process_group()
    return {"step": args.total_steps, "unique_splits_seen": len(seen_splits) * world_size}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multi-node DDP trainer for DopeFidelityV5 on S3 v250")
    ap.add_argument("--corpus-dir", default="/home/ubuntu/dope-data/s3-v250-corpus")
    ap.add_argument("--run-dir", default="/home/ubuntu/dope-data/runs/dope_v5_ddp")
    ap.add_argument("--run-name", default="dope_v5_s3_ddp3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vram-cap-gib", type=float, default=6.5)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--total-steps", type=int, default=20000)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--episodes-per-step", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--cache-cap", type=int, default=512)
    ap.add_argument("--min-fit-rows", type=int, default=32)
    ap.add_argument("--max-fit-rows", type=int, default=512)
    ap.add_argument("--min-query-rows", type=int, default=32)
    ap.add_argument("--max-query-rows", type=int, default=256)
    ap.add_argument("--aug-prob", type=float, default=0.85)
    ap.add_argument("--max-noise-cols", type=int, default=6)
    ap.add_argument("--max-derived-cols", type=int, default=4)
    ap.add_argument("--multilabel-prob", type=float, default=0.12)
    ap.add_argument("--no-augment", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
