"""Multi-node DDP & dedicated-GPU high-speed trainer for DOPE v2.0.0 (SingleEncV6 & FlowV6).

Features:
- 100% unique-per-batch RAM-resident async prefetch (`s3_fast_pipeline.py`) with progressive
  3-stage data-size ramp and JopeDime regret-weighted hard-table curriculum.
- 6-layer gradient smoothing: large unique-dataset effective batch, decoupled AdamW (0.9, 0.98),
  linear warmup + cosine LR, TF32 + BF16 autocast, adaptive grad-norm clipping + spike guard,
  dual-timescale Polyak EMA (0.995 / 0.9995), and immutable SHA-256 `checkpoints/ledger.jsonl`.
- Dynamic co-host VRAM/CPU watchdog protecting co-hosted `jsr` ASR jobs on shared GPUs.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from safetensors.torch import save_file
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .s3_fast_pipeline import AsyncBatchPrefetcher, RamCorpusStore, curriculum_stage_for_step
from .v5_fidelity_net import DopeFidelityV5, compute_main_kpi_loss
from .v6_single_encoding import DopeSingleEncV6, compute_single_enc_loss


def _configure_vram_cap(device: torch.device, vram_cap_gib: float) -> None:
    if device.type != "cuda" or vram_cap_gib <= 0.0:
        return
    total_gib = torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 3)
    frac = min(max(vram_cap_gib / max(total_gib, 1.0), 0.05), 0.85)
    torch.cuda.set_per_process_memory_fraction(frac, device)


def _cohost_yield_if_needed(device: torch.device, min_free_gib: float, max_cpu_pct: float) -> bool:
    """Protect co-hosted `jsr` / cluster jobs: yield cleanly if free VRAM or CPU headroom dips."""
    yielded = False
    for _ in range(12):
        need_wait = False
        if device.type == "cuda" and min_free_gib > 0.0:
            free_b, _ = torch.cuda.mem_get_info(device)
            if (free_b / (1024.0 ** 3)) < min_free_gib:
                torch.cuda.empty_cache()
                need_wait = True
        if max_cpu_pct > 0.0:
            try:
                load1 = os.getloadavg()[0]
                cores = max(os.cpu_count() or 1, 1)
                if (load1 / cores) * 100.0 > max_cpu_pct:
                    need_wait = True
            except OSError:
                pass
        if not need_wait:
            break
        yielded = True
        time.sleep(3.0)
    return yielded


def _split_invariance_infonce(
    z_a: list[torch.Tensor], z_b: list[torch.Tensor], families: list[str], tau: float = 0.12
) -> torch.Tensor:
    if len(z_a) < 2:
        return torch.tensor(0.0, device=z_a[0].device if z_a else "cpu")
    za = F.normalize(torch.stack(z_a, dim=0).float(), dim=-1)
    zb = F.normalize(torch.stack(z_b, dim=0).float(), dim=-1)
    sim = (za @ zb.T) / tau
    n = sim.shape[0]
    mask = torch.eye(n, device=sim.device, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if i != j and families[i] == families[j]:
                mask[i, j] = True
    pos = sim.diagonal()
    neg = sim.masked_fill(mask, -1e9)
    all_logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
    return F.cross_entropy(all_logits, torch.zeros(n, device=sim.device, dtype=torch.long))


def _param_groups(model: nn.Module, wd: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name.lower() or "bias" in name.lower() or "emb" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def save_atomic_checkpoint(
    run_dir: Path,
    model_raw: nn.Module,
    ema_state: dict[str, torch.Tensor],
    step: int,
    meta: dict[str, Any],
    keep_last: int = 12,
) -> dict[str, Any]:
    """Save EMA weights atomically, record immutable SHA-256 in `checkpoints/ledger.jsonl`."""
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = f"step_{step:07d}.safetensors"
    target = ckpt_dir / name
    tmp = ckpt_dir / f".{name}.tmp"
    tensors = {k: v.detach().cpu().contiguous() for k, v in ema_state.items()}
    save_file(tensors, str(tmp))
    os.replace(tmp, target)
    raw_bytes = target.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()

    latest = ckpt_dir / "latest.safetensors"
    latest_tmp = ckpt_dir / ".latest.safetensors.tmp"
    save_file(tensors, str(latest_tmp))
    os.replace(latest_tmp, latest)

    info = {
        "step": step,
        "checkpoint": name,
        "sha256": sha,
        "bytes": len(raw_bytes),
        "role": "step_ema",
        "arch": meta.get("arch", "single_enc_v6"),
        "d_model": int(getattr(model_raw, "d_model", 256)),
        "n_layers": int(getattr(model_raw, "n_layers", 6)),
        "n_heads": int(getattr(model_raw, "n_heads", 8)),
        "global_dim": int(getattr(model_raw, "global_dim", getattr(model_raw, "latent_dim", 128))),
        "col_dim": int(getattr(model_raw, "col_dim", 24)),
        "latent_dim": int(getattr(model_raw, "latent_dim", getattr(model_raw, "global_dim", 128))),
        "time": round(time.time(), 3),
        **meta,
    }
    ptr_tmp = ckpt_dir / ".latest.json.tmp"
    ptr_tmp.write_text(json.dumps(info, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(ptr_tmp, ckpt_dir / "latest.json")
    with (ckpt_dir / "ledger.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(info, sort_keys=True) + "\n")

    # Prune old step checkpoints while NEVER deleting protected best/valley/soup checkpoints
    protected: set[str] = {"latest.safetensors", "best_main_kpi.safetensors", "best_soup_k3.safetensors"}
    vp = ckpt_dir / "valleys.json"
    if vp.is_file():
        try:
            for item in json.loads(vp.read_text(encoding="utf-8")).get("valleys", []):
                protected.add(str(item.get("checkpoint", "")))
        except (OSError, ValueError):
            pass
    step_files = sorted(ckpt_dir.glob("step_*.safetensors"))
    for old in step_files[:-keep_last]:
        if old.name not in protected:
            old.unlink(missing_ok=True)
    return info


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.arch == "single_enc_v6":
        return DopeSingleEncV6(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            global_dim=args.global_dim,
            col_dim=args.col_dim,
        )
    return DopeFidelityV5(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        latent_dim=args.global_dim,
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if world_size > 1:
        backend = "nccl" if (torch.cuda.is_available() and args.device.startswith("cuda")) else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev_idx = local_rank if ":" not in args.device else int(args.device.split(":")[1])
        device = torch.device(f"cuda:{dev_idx}")
        torch.cuda.set_device(device)
        _configure_vram_cap(device, args.vram_cap_gib)
        use_bf16 = torch.cuda.is_bf16_supported()
    else:
        device = torch.device("cpu")
        use_bf16 = False

    torch.manual_seed(args.seed + rank * 101)
    run_dir = Path(args.run_dir)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train.jsonl"
    regret_path = run_dir / "curriculum_regret.json"

    t_load = time.time()
    store = RamCorpusStore(Path(args.corpus_dir), split="train", preload=not args.no_preload)
    if rank == 0:
        print(f"[DOPE v2.0.0] Preloaded {len(store)} S3 tables into RAM in {time.time() - t_load:.2f}s", flush=True)

    model_raw = build_model(args).to(device)
    n_params = sum(p.numel() for p in model_raw.parameters())
    model: nn.Module = (
        DDP(model_raw, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)
        if world_size > 1
        else model_raw
    )
    optimizer = torch.optim.AdamW(
        _param_groups(model_raw, wd=args.weight_decay), lr=args.lr, betas=(0.9, 0.98), eps=1e-6
    )

    def lr_lambda(step_idx: int) -> float:
        if step_idx < args.warmup_steps:
            return float(step_idx + 1) / float(max(args.warmup_steps, 1))
        prog = float(step_idx - args.warmup_steps) / float(max(args.total_steps - args.warmup_steps, 1))
        return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema_slow = {k: v.detach().clone().float() for k, v in model_raw.state_dict().items()}
    ema_fast = {k: v.detach().clone().float() for k, v in model_raw.state_dict().items()}

    prefetch = AsyncBatchPrefetcher(
        store,
        batch_size=args.episodes_per_step,
        global_seed=args.seed,
        rank=rank,
        num_workers=args.prep_workers,
        regret_path=regret_path,
    )

    if rank == 0:
        print(
            f"[DOPE v2.0.0] Starting {args.run_name} arch={args.arch} params={n_params:,} "
            f"world_size={world_size} eps/step={args.episodes_per_step}x{args.grad_accum} "
            f"vram_cap={args.vram_cap_gib}G cohost_guard={args.cohost_min_free_gib}G",
            flush=True,
        )

    seen_splits: set[str] = set()
    task_totals: dict[str, int] = {"regression": 0, "binary": 0, "multiclass": 0, "multilabel": 0}
    grad_hist: deque[float] = deque(maxlen=64)
    skipped_spikes = 0
    cohost_yields = 0
    acc: dict[str, list[float]] = {
        k: [] for k in ("loss", "tstr", "distill", "drv", "real", "mmd", "infonce", "ret", "gn", "n_fit", "n_cols", "n_noise", "n_derived", "enc_bytes")
    }
    last_log_t = time.time()

    try:
        for step in range(1, args.total_steps + 1):
            if step % 10 == 0 and _cohost_yield_if_needed(device, args.cohost_min_free_gib, args.cohost_max_cpu_pct):
                cohost_yields += 1

            model.train()
            optimizer.zero_grad(set_to_none=True)
            s_loss = s_tstr = s_dist = s_drv = s_real = s_mmd = s_ret = s_encb = 0.0
            valid_eps = 0
            z_a_list: list[torch.Tensor] = []
            z_b_list: list[torch.Tensor] = []
            fam_list: list[str] = []

            for accum_i in range(args.grad_accum):
                _, eps_a, eps_b = prefetch.next_batch()
                denom = float(max(len(eps_a) * args.grad_accum, 1))
                for ep_i, ep in enumerate(eps_a):
                    seen_splits.add(ep["split_sig"])
                    task_totals[ep["task"]] = task_totals.get(ep["task"], 0) + 1
                    st = ep["aug_stats"]
                    acc["n_fit"].append(st["n_fit"])
                    acc["n_cols"].append(st["n_cols"])
                    acc["n_noise"].append(st["n_noise_cols"])
                    acc["n_derived"].append(st["n_derived_cols"])
                    try:
                        raw_m = model.module if isinstance(model, DDP) else model
                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                            if args.arch == "single_enc_v6":
                                out = compute_single_enc_loss(raw_m, ep, device, args.seed + step * 97 + rank * 13 + ep_i + accum_i * 31)
                            else:
                                out = compute_main_kpi_loss(raw_m, ep, device, args.seed + step * 97 + rank * 13 + ep_i + accum_i * 31)
                        loss_ep = out["total"] / denom
                        if not bool(torch.isfinite(loss_ep).all()):
                            continue
                        if accum_i == 0 and ep_i < len(eps_b):
                            eb = eps_b[ep_i]
                            seen_splits.add(eb["split_sig"])
                            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                                xf_b = torch.as_tensor(eb["X_fit"], dtype=torch.float32, device=device)
                                yf_b = torch.as_tensor(eb["y_fit"], dtype=torch.float32, device=device)
                                mf_b = torch.as_tensor(eb["miss_fit"], dtype=torch.float32, device=device)
                                if args.arch == "single_enc_v6":
                                    eb_enc = raw_m.encode_to_single_latent(xf_b, yf_b, mf_b, eb["task"], int(eb["n_classes"]))
                                else:
                                    eb_enc = raw_m.encode_table(xf_b, yf_b, eb["task"], int(eb["n_classes"]), mf_b)
                            z_a_list.append(out["task_latent"])
                            z_b_list.append(eb_enc["task_latent"])
                            fam_list.append(ep["family_stem"] or ep["dataset_hash"])
                        loss_ep.backward()
                        s_loss += float(out["total"].item())
                        s_tstr += float(out["tstr_loss"])
                        s_dist += float(out.get("distill_loss", 0.0))
                        s_drv += float(out["driver_loss"])
                        s_real += float(out["realism_loss"])
                        s_mmd += float(out.get("mmd_loss", 0.0))
                        s_encb += float(out.get("encoding_bytes", 0.0))
                        if math.isfinite(out["unclipped_retention"]):
                            s_ret += float(out["unclipped_retention"])
                        valid_eps += 1
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower() and device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue

            infonce_val = 0.0
            if len(z_a_list) >= 2:
                info_l = 0.05 * _split_invariance_infonce(z_a_list, z_b_list, fam_list)
                if bool(torch.isfinite(info_l).all()):
                    info_l.backward()
                    infonce_val = float(info_l.item())

            # Adaptive grad-norm clipping + spike guard (Layer 3 of gradient smoothing!)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
            med_gn = float(np.median(grad_hist)) if len(grad_hist) >= 16 else 10.0
            if valid_eps > 0 and math.isfinite(gn) and (gn <= max(med_gn * 10.0, 20.0) or step <= args.warmup_steps):
                optimizer.step()
                scheduler.step()
                grad_hist.append(gn)
                # Dual-timescale Polyak EMA (0.995 fast, 0.9995 slow)
                d_slow = min(0.9995, 1.0 - 1.0 / float(step + 25))
                d_fast = min(0.995, 1.0 - 1.0 / float(step + 10))
                with torch.no_grad():
                    for k, v in model_raw.state_dict().items():
                        vf = v.detach().float()
                        ema_slow[k].mul_(d_slow).add_(vf, alpha=1.0 - d_slow)
                        ema_fast[k].mul_(d_fast).add_(vf, alpha=1.0 - d_fast)
            else:
                skipped_spikes += 1
                optimizer.zero_grad(set_to_none=True)

            if valid_eps > 0:
                acc["loss"].append(s_loss / valid_eps)
                acc["tstr"].append(s_tstr / valid_eps)
                acc["distill"].append(s_dist / valid_eps)
                acc["drv"].append(s_drv / valid_eps)
                acc["real"].append(s_real / valid_eps)
                acc["mmd"].append(s_mmd / valid_eps)
                acc["infonce"].append(infonce_val)
                acc["ret"].append(s_ret / valid_eps)
                acc["gn"].append(gn if math.isfinite(gn) else 0.0)
                acc["enc_bytes"].append(s_encb / valid_eps)

            if step % args.log_every == 0 or step == 1:
                now = time.time()
                wall = max(now - last_log_t, 1e-3)
                eps_window = (1 if step == 1 else args.log_every) * args.episodes_per_step * args.grad_accum * world_size
                vram_mib = float(torch.cuda.memory_allocated(device) / (1024.0 ** 2)) if device.type == "cuda" else 0.0
                if rank == 0 and acc["loss"]:
                    st_cfg = curriculum_stage_for_step(step)
                    rec = {
                        "step": step,
                        "total_steps": args.total_steps,
                        "time": round(now, 3),
                        "wall_s": round(wall, 3),
                        "arch": args.arch,
                        "stage": st_cfg.stage_id,
                        "world_size": world_size,
                        "loss": round(float(np.mean(acc["loss"])), 5),
                        "tstr_loss": round(float(np.mean(acc["tstr"])), 5),
                        "distill_loss": round(float(np.mean(acc["distill"])), 5),
                        "driver_loss": round(float(np.mean(acc["drv"])), 5),
                        "realism_loss": round(float(np.mean(acc["real"])), 5),
                        "mmd_loss": round(float(np.mean(acc["mmd"])), 5),
                        "infonce_loss": round(float(np.mean(acc["infonce"])), 5),
                        "train_retention": round(float(np.median(acc["ret"])), 4),
                        "grad_norm": round(float(np.mean(acc["gn"])), 4),
                        "skipped_spikes": skipped_spikes,
                        "cohost_yields": cohost_yields,
                        "lr": float(scheduler.get_last_lr()[0]),
                        "tables_per_s": round(eps_window / wall, 2),
                        "unique_splits_seen": len(seen_splits) * world_size,
                        "encoding_bytes_avg": round(float(np.mean(acc["enc_bytes"])), 1),
                        "avg_n_fit": round(float(np.mean(acc["n_fit"])), 1),
                        "avg_n_cols": round(float(np.mean(acc["n_cols"])), 1),
                        "avg_noise_cols": round(float(np.mean(acc["n_noise"])), 2),
                        "avg_derived_cols": round(float(np.mean(acc["n_derived"])), 2),
                        "regret_boost_mean": round(float(np.mean(store.regret_weights)), 3),
                        "gpu_vram_mib": round(vram_mib, 1),
                        "task_totals": dict(task_totals),
                    }
                    with train_log.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    print(
                        f"[step {step:05d}/{args.total_steps} S{st_cfg.stage_id}] loss={rec['loss']:.4f} "
                        f"tstr={rec['tstr_loss']:.4f} mmd={rec['mmd_loss']:.4f} drv={rec['driver_loss']:.4f} "
                        f"gn={rec['grad_norm']:.2f} splits={rec['unique_splits_seen']} "
                        f"rate={rec['tables_per_s']:.1f} tbl/s enc={rec['encoding_bytes_avg']:.0f}B vram={vram_mib:.0f}M",
                        flush=True,
                    )
                for buf in acc.values():
                    buf.clear()
                last_log_t = now

            if rank == 0 and (step % args.ckpt_every == 0 or step == args.total_steps):
                save_atomic_checkpoint(
                    run_dir,
                    model_raw,
                    ema_slow,
                    step,
                    {
                        "run_name": args.run_name,
                        "arch": args.arch,
                        "stage": curriculum_stage_for_step(step).stage_id,
                        "unique_splits_seen": len(seen_splits) * world_size,
                        "world_size": world_size,
                        "train_loss": round(s_loss / max(valid_eps, 1), 5),
                        "train_tstr": round(s_tstr / max(valid_eps, 1), 5),
                        "train_mmd": round(s_mmd / max(valid_eps, 1), 5),
                    },
                )
    finally:
        prefetch.close()
        if world_size > 1:
            dist.destroy_process_group()
    return {"step": args.total_steps, "unique_splits_seen": len(seen_splits) * world_size}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DOPE v2.0.0 high-speed multi-node trainer")
    ap.add_argument("--corpus-dir", default="/home/ubuntu/dope-data/s3-v250-corpus")
    ap.add_argument("--run-dir", default="/home/ubuntu/dope-data/runs/dope_v2")
    ap.add_argument("--run-name", default="dope_single_enc_v2")
    ap.add_argument("--arch", choices=("single_enc_v6", "flow_v6"), default="single_enc_v6")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vram-cap-gib", type=float, default=6.0)
    ap.add_argument("--cohost-min-free-gib", type=float, default=0.0)
    ap.add_argument("--cohost-max-cpu-pct", type=float, default=0.0)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--global-dim", type=int, default=128)
    ap.add_argument("--col-dim", type=int, default=24)
    ap.add_argument("--total-steps", type=int, default=30000)
    ap.add_argument("--warmup-steps", type=int, default=400)
    ap.add_argument("--episodes-per-step", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--prep-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--no-preload", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
