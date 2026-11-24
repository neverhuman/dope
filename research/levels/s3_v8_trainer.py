"""Resumable 3-phase GPU/CPU/DiLoCo trainer for `SingleEncV8` (DOPE v3.0).

Key invariants:
- Fresh FIT/QRY split every episode with zero per-table feature/evidence cache (09:50Z binding rule).
- Consumes `s3_v8_feed.FactoryFeed` (`episode_provider(feed_step, b)`) when supplied, supporting
  gradient accumulation (`--grad-accum-steps`) and width curriculum (`--max-p-curriculum`).
- Strict `CUDA` guard + `--max-wall-s` cap, 3 EMAs, Top-K Valley Soup, optional PCGrad-Lite /
  Periodic-ASAM, exact full-state save/resume, and canonical `F32` `rank <= 2` SafeTensors export.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from research.levels.s3_v8_losses import (
    compute_v8_episode_loss,
    loss_direct_conditional_nll,
    pcgrad_lite_combine,
)
from research.levels.s3_v8_priors import sample_contrastive_xor_pair, sample_v8_prior_episode
from research.levels.s3_v8_trainer_io import (
    _episode_to_tensors,
    _export_state_dict_safetensors,
    _update_ema,
    build_valley_soup,
    check_cuda_exclusive_guard,
    consume_episode_tensors,
    save_full_checkpoint,
)
from research.levels.v8_packet import DEC_CODE_DIM
from research.levels.v8_single_encoding import SingleEncV8

__all__ = [
    "_episode_to_tensors",
    "_export_state_dict_safetensors",
    "_update_ema",
    "build_valley_soup",
    "check_cuda_exclusive_guard",
    "consume_episode_tensors",
    "parse_args",
    "save_full_checkpoint",
    "train_v8",
]


def train_v8(args: argparse.Namespace, episode_provider: Any | None = None) -> dict[str, Any]:
    """Run resumable SingleEncV8 training loop (CPU or CUDA GPU, optional DiLoCo + C6 episode provider)."""
    torch_threads = int(getattr(args, "torch_threads", 0) or 0)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    shared_vram_gb = float(getattr(args, "allow_shared_gpu_vram_gb", 0.0) or 0.0)
    check_cuda_exclusive_guard(args.device, shared_vram_gb=shared_vram_gb)
    device = torch.device(args.device)
    deterministic = bool(getattr(args, "deterministic", False))
    no_tf32 = bool(getattr(args, "no_tf32", False))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    if no_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    cfg = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_enc_layers": args.n_enc_layers,
        "n_dec_layers": args.n_dec_layers,
        "code_dim": int(getattr(args, "code_dim", DEC_CODE_DIM) or DEC_CODE_DIM),
        "with_owner_encoder": not args.decoder_only,
        "ff_mult": int(getattr(args, "ff_mult", 2) or 2),
    }
    export_dtype = str(getattr(args, "export_dtype", "f32") or "f32")
    no_delta_w = bool(getattr(args, "no_delta_w", False))
    historical_forward = bool(getattr(args, "historical_forward", False))
    all_rank_norm = bool(getattr(args, "all_rank_norm", False)) or historical_forward
    active_rank_v1 = not all_rank_norm
    dec_cap = None if all_rank_norm else "active-rank-v1"
    run_flags = {
        "no_delta_w": no_delta_w,
        "historical_forward": historical_forward,
        "all_rank_norm": all_rank_norm,
        "deterministic": deterministic,
        "no_tf32": no_tf32,
    }
    model = SingleEncV8(**cfg).to(device)
    init_from = str(getattr(args, "init_from", "") or "")
    if init_from:
        from safetensors.torch import load_file as _st_load
        model.load_state_dict(_st_load(init_from, device=str(device)), strict=False)
    if no_delta_w:
        with torch.no_grad():
            model.delta_W.weight.zero_()
            model.delta_W.bias.zero_()
        for name, p in model.named_parameters():
            if name.startswith("delta_W."):
                p.requires_grad_(False)
    if bool(getattr(args, "freeze_target_heads", False)):
        for name, p in model.named_parameters():
            if any(name.startswith(pfx) for pfx in ("delta_W.", "node_", "bilin_", "gate_head.")):
                p.requires_grad_(False)
    if bool(getattr(args, "freeze_x_heads", False)):
        for name, p in model.named_parameters():
            if any(name.startswith(pfx) for pfx in ("delta_knots.", "delta_L.", "flow_")):
                p.requires_grad_(False)

    bb_mult = float(getattr(args, "backbone_lr_mult", 1.0) or 1.0)
    bb_warm = int(getattr(args, "backbone_warmup_steps", 500) or 500)
    bb_prefixes = ("dec_blocks.", "dec_col_in.", "dec_glob_in.", "dec_final_norm.")
    if bb_mult < 0.999:
        bb_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(bb_prefixes)]
        hd_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(bb_prefixes)]
        param_groups = [{"params": hd_params}, {"params": bb_params}]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
    lr_sched_kind = str(getattr(args, "lr_schedule", "cosine") or "cosine")

    def _lr_lambda(st: int) -> float:
        if warmup_steps > 0 and st < warmup_steps:
            return float(st + 1) / float(warmup_steps)
        rem = max(args.steps - warmup_steps, 1)
        prog = min(1.0, max(0.0, float(st - warmup_steps) / float(rem)))
        if lr_sched_kind == "wsd":
            if prog < 0.80:
                return 1.0
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * (prog - 0.80) / 0.20))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    def _bb_lr_lambda(st: int) -> float:
        base = _lr_lambda(st)
        if bb_warm <= 0 or st >= bb_warm:
            return base
        ramp = bb_mult + (1.0 - bb_mult) * (float(st) / float(bb_warm))
        return base * ramp

    lr_lambdas: Any = [_lr_lambda, _bb_lr_lambda] if len(param_groups) == 2 else _lr_lambda
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
    emas: dict[str, dict[str, torch.Tensor]] = {
        "0.99": copy.deepcopy(model.state_dict()),
        "0.999": copy.deepcopy(model.state_dict()),
        "0.9998": copy.deepcopy(model.state_dict()),
    }
    start_step = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "latest.pt"
    if args.resume and ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_flags = ck.get("run_flags")
        if isinstance(saved_flags, dict):
            for fk, fv in run_flags.items():
                if fk in saved_flags and bool(saved_flags[fk]) != bool(fv):
                    raise ValueError(
                        f"resume flag mismatch on {fk}: checkpoint={saved_flags[fk]} vs current={fv}"
                    )
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler") is not None:
            scheduler.load_state_dict(ck["scheduler"])
        emas = ck["emas"]
        if no_delta_w:
            for dname in ("delta_W.weight", "delta_W.bias"):
                if float(model.state_dict()[dname].detach().abs().max().item()) != 0.0:
                    raise ValueError(f"resumed model has non-zero {dname} under --no-delta-w")
                for decay_s, ema_st in emas.items():
                    if float(ema_st[dname].detach().abs().max().item()) != 0.0:
                        raise ValueError(f"resumed EMA {decay_s} has non-zero {dname} under --no-delta-w")
        start_step = int(ck["step"])
        torch.set_rng_state(ck["rng_state"]["torch"].cpu())
        if device.type == "cuda" and ck["rng_state"].get("cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_state"]["cuda"].cpu(), device=device)
        np_rng.bit_generator.state = ck["rng_state"]["numpy"]

    sync_dir = str(getattr(args, "sync_dir", "") or "")
    sync_every = int(getattr(args, "sync_every", 25) or 25)
    sync_round = 0
    anchor_state: dict[str, np.ndarray] = {}
    ema_proxy = SingleEncV8(**cfg).to(device) if sync_dir else None
    if sync_dir:
        from research.levels.s3_v7_entropy import diloco_sync_step
        assert ema_proxy is not None
        ema_proxy.load_state_dict(emas["0.999"])
        anchor_state = {k: v.detach().cpu().to(torch.float32).numpy().copy() for k, v in model.state_dict().items()}
        sync_round, _ = diloco_sync_step(
            sync_dir, model, ema_proxy, anchor_state,
            float(getattr(args, "sync_outer_lr", 0.25)),
            float(getattr(args, "sync_outer_momentum", 0.70)),
            base_round=0, worker_id=str(getattr(args, "run_name", "v8_w0")),
        )
        if no_delta_w:
            with torch.no_grad():
                model.delta_W.weight.zero_()
                model.delta_W.bias.zero_()
                ema_proxy.delta_W.weight.zero_()
                ema_proxy.delta_W.bias.zero_()
        emas["0.999"] = copy.deepcopy(ema_proxy.state_dict())

    t0 = time.monotonic()
    log_path = out_dir / "train.jsonl"
    metrics_last: dict[str, Any] = {}
    steps_done, non_finite_streak = 0, 0
    ckpt_every = int(getattr(args, "ckpt_every", 0) or 0)
    grad_accum = max(1, int(getattr(args, "grad_accum_steps", 1) or 1))
    prior_mix_prob = float(getattr(args, "prior_mix_prob", 0.0) or 0.0)
    max_p_cur = int(getattr(args, "max_p_curriculum", 0) or 0)
    cur_steps = int(getattr(args, "curriculum_steps", 1000) or 1000)
    alpha_floor = float(getattr(args, "alpha_floor", 0.0) or 0.0)
    alpha_floor_steps = int(getattr(args, "alpha_floor_steps", 1000) or 1000)
    alpha_floor_w = float(getattr(args, "alpha_floor_weight", 0.05) or 0.05)
    resid_cond_w = float(getattr(args, "resid_cond_weight", 0.0) or 0.0)

    stop_at = int(getattr(args, "stop_at_step", 0) or 0)
    end_step = min(args.steps, stop_at) if stop_at > 0 else args.steps
    for step in range(start_step + 1, end_step + 1):
        if args.max_wall_s > 0 and (time.monotonic() - t0) >= args.max_wall_s:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)

        phase = 1 if step <= max(args.steps // 5, 1) else (2 if step <= (4 * args.steps) // 5 else 3)
        loss_terms_accum: dict[str, float] = {}
        pcgrad_info: dict[str, float] = {}
        step_loss_sum = 0.0
        total_eps = 0

        for ga in range(grad_accum):
            feed_step = (step - 1) * grad_accum + ga + 1
            ep_losses: list[torch.Tensor] = []
            for b_i in range(args.episodes_per_step):
                use_prior = (episode_provider is None) or (
                    prior_mix_prob > 0.0 and np_rng.random() < prior_mix_prob
                )
                if not use_prior:
                    ep = episode_provider(feed_step, b_i)
                elif args.contrastive_prob > 0.0 and b_i == 0 and np_rng.random() < args.contrastive_prob:
                    ep, _ = sample_contrastive_xor_pair(np_rng, n_fit=args.n_fit, n_qry=args.n_qry)
                else:
                    p_lim = min(max_p_cur, 24) if (max_p_cur > 0 and step <= cur_steps) else None
                    p_arg = int(np_rng.integers(4, max(p_lim, 5) + 1)) if p_lim is not None else None
                    ep = sample_v8_prior_episode(np_rng, n_fit=args.n_fit, n_qry=args.n_qry, p=p_arg)

                td = consume_episode_tensors(
                    ep, device, use_rust_packet=args.use_rust_packet,
                    budget_bytes=args.budget_bytes, seed=args.seed + feed_step * 31 + b_i,
                    drop_tier_prob=0.30 if phase >= 2 else 0.0,
                )
                p_ep = td["ev_col"].shape[0]
                col_mask = torch.ones(1, p_ep, device=device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    z_fsq, _ = model.encode_evidence(
                        td["ev_col"][None], td["ev_edge"][None], td["ev_glob"][None],
                        col_mask=col_mask, tier_dropout=(phase >= 2),
                    )
                    plan = model.decode_plan(
                        td["dec_col"][None], td["dec_edge"][None], td["dec_glob"][None],
                        z_fsq, col_mask=col_mask,
                    )
                raw_alpha = plan["alpha_table"]
                cur_a_floor = 0.0
                if alpha_floor > 0.0:
                    half = max(alpha_floor_steps // 2, 1)
                    if step <= half:
                        cur_a_floor = alpha_floor
                    elif step <= alpha_floor_steps:
                        t_lin = float(step - half) / float(max(alpha_floor_steps - half, 1))
                        cur_a_floor = alpha_floor * (1.0 - 0.8 * t_lin)
                    else:
                        cur_a_floor = 0.2 * alpha_floor
                    plan = dict(plan)
                    plan["alpha_table"] = torch.clamp(raw_alpha, min=cur_a_floor)
                pkt_an = None if historical_forward else td.get("pkt_analytic")
                dec = model.decode_batch_differentiable(
                    plan,
                    td["knots"][None], td["loadings"][None], td["W_base"][None], td["b_base"][None],
                    td["resid_sd"][None], td["z_max"][None], td["is_cont"][None], col_mask,
                    n_rows=args.n_syn, antithetic=True, pkt_analytic=pkt_an,
                    active_rank_v1=active_rank_v1,
                )
                xq_clean = torch.nan_to_num(td["X_qry"], nan=0.0)
                zq_in = None if historical_forward else (td["z_qry"][None] if "z_qry" in td else None)
                knots_q = None if historical_forward else td["knots"][None]
                zmax_q = None if historical_forward else td["z_max"][None]
                logits_qry = model.predict_target_on_x(
                    plan, td["W_base"][None], td["b_base"][None], xq_clean[None],
                    z_in=zq_in, base_knots=knots_q, z_max=zmax_q, pkt_analytic=pkt_an,
                )[0]
                loss_w = {
                    "direct_cond": float(getattr(args, "direct_cond_weight", 0.60)),
                    "anchor": float(getattr(args, "anchor_weight", 0.15)),
                    "anchor_nl": float(getattr(args, "anchor_nonlinear_weight", 0.40)),
                    "joint_swd": float(getattr(args, "joint_swd_weight", 0.0)),
                }
                ld = compute_v8_episode_loss(
                    dec["x_syn"][0], dec["logits"][0], dec["y_reg"][0], dec["resid_sd_eff"][0],
                    td["X_fit"], td["y_fit"], td["X_qry"], td["y_qry"],
                    td["task_id"], td["n_classes"],
                    weights=loss_w, logits_qry=logits_qry, plan=plan,
                )
                l_aux = 0.05 * torch.nn.functional.mse_loss(model.aux_ev_head(plan["h_dec"][0]), td["ev_col"])
                l_a_pen = (
                    alpha_floor_w * (torch.relu(cur_a_floor - raw_alpha) ** 2).mean()
                    if cur_a_floor > 0.0
                    else l_aux.new_tensor(0.0)
                )
                if resid_cond_w > 0.0:
                    plan_nl = dict(plan)
                    plan_nl["delta_W"] = torch.zeros_like(plan["delta_W"])
                    logits_nl = model.predict_target_on_x(
                        plan_nl, td["W_base"][None].detach(), td["b_base"][None].detach(), xq_clean[None],
                        z_in=zq_in, base_knots=knots_q, z_max=zmax_q, pkt_analytic=pkt_an,
                    )[0]
                    yq_raw = td["y_qry"]
                    if td["task_id"] == 0:
                        vq = torch.isfinite(yq_raw)
                    else:
                        k_lim = 2 if td["task_id"] == 1 else max(2, min(int(td["n_classes"]), logits_nl.shape[-1]))
                        vq = torch.isfinite(yq_raw) & (yq_raw >= 0.0) & (yq_raw < float(k_lim))
                    if int(vq.sum().item()) >= 4:
                        l_resid_nl = loss_direct_conditional_nll(
                            logits_nl[vq], dec["resid_sd_eff"][0].detach(), yq_raw[vq],
                            td["task_id"], td["n_classes"],
                        )
                    else:
                        l_resid_nl = l_aux.new_tensor(0.0)
                else:
                    l_resid_nl = l_aux.new_tensor(0.0)
                ep_tot = ld["total"] + l_aux + l_a_pen + resid_cond_w * l_resid_nl
                if max_p_cur > 0 and step <= cur_steps and p_ep > max_p_cur:
                    ep_tot = ep_tot * 0.25
                ep_losses.append(ep_tot)
                total_eps += 1
                for k, v in ld.items():
                    loss_terms_accum[k] = loss_terms_accum.get(k, 0.0) + float(v.detach().item())
                loss_terms_accum["l_aux"] = loss_terms_accum.get("l_aux", 0.0) + float(l_aux.detach().item())
                loss_terms_accum["l_resid_nl"] = loss_terms_accum.get("l_resid_nl", 0.0) + float(l_resid_nl.detach().item())
                loss_terms_accum["alpha_gate"] = loss_terms_accum.get("alpha_gate", 0.0) + float(raw_alpha[0].detach().item())

            micro_loss = torch.stack(ep_losses).mean()
            step_loss_sum += float(micro_loss.detach().item()) / float(grad_accum)
            if args.use_pcgrad and len(ep_losses) >= 2:
                params = [p for p in model.parameters() if p.requires_grad]
                comb_g, pcgrad_info = pcgrad_lite_combine(ep_losses, params)
                scale = 1.0 / float(len(ep_losses) * grad_accum)
                for p, g in zip(params, comb_g):
                    p.grad = (g * scale) if p.grad is None else (p.grad + g * scale)
            else:
                (micro_loss / float(grad_accum)).backward()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
        if not math.isfinite(grad_norm) or not math.isfinite(step_loss_sum):
            optimizer.zero_grad(set_to_none=True)
            non_finite_streak += 1
            if non_finite_streak >= 5:
                raise FloatingPointError(f"non-finite loss/grad at step {step}: loss={step_loss_sum} grad={grad_norm}")
            continue
        non_finite_streak = 0

        optimizer.step()
        scheduler.step()
        for decay_s, ema_st in emas.items():
            _update_ema(ema_st, model, float(decay_s))

        if sync_dir and sync_every > 0 and (step % sync_every == 0):
            from research.levels.s3_v7_entropy import diloco_sync_step
            assert ema_proxy is not None
            ema_proxy.load_state_dict(emas["0.999"])
            sync_round, _ = diloco_sync_step(
                sync_dir, model, ema_proxy, anchor_state,
                float(getattr(args, "sync_outer_lr", 0.25)),
                float(getattr(args, "sync_outer_momentum", 0.70)),
                base_round=sync_round, worker_id=str(getattr(args, "run_name", "v8_w0")),
            )
            if no_delta_w:
                with torch.no_grad():
                    model.delta_W.weight.zero_()
                    model.delta_W.bias.zero_()
                    ema_proxy.delta_W.weight.zero_()
                    ema_proxy.delta_W.bias.zero_()
            emas["0.999"] = copy.deepcopy(ema_proxy.state_dict())

        bilin_norm = float(
            (model.bilin_u.weight.norm() + model.bilin_v.weight.norm() + model.bilin_w.weight.norm()).detach().item()
        )
        inv_ep = 1.0 / float(max(total_eps, 1))
        rec = {
            "step": step,
            "phase": phase,
            "lr": float(scheduler.get_last_lr()[0]),
            "grad_norm": grad_norm,
            "wall_s": round(time.monotonic() - t0, 3),
            "sync_round": sync_round,
            "eff_batch": total_eps,
            "bilin_norm": round(bilin_norm, 5),
            **{k: round(v * inv_ep, 5) for k, v in loss_terms_accum.items()},
            **{k: round(v, 5) for k, v in pcgrad_info.items()},
        }
        metrics_last = rec
        steps_done += 1
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        if args.save_every > 0 and (step % args.save_every == 0 or step == end_step):
            rng_payload: dict[str, Any] = {
                "torch": torch.get_rng_state(),
                "numpy": np_rng.bit_generator.state,
            }
            if device.type == "cuda":
                rng_payload["cuda"] = torch.cuda.get_rng_state(device=device)
            save_full_checkpoint(
                ckpt_path, model, optimizer, scheduler, emas, step, rng_payload, cfg,
                run_flags=run_flags,
            )
        if ckpt_every > 0 and (step % ckpt_every == 0):
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _export_state_dict_safetensors(
                cfg, emas["0.999"], ckpt_dir / f"step_{step:06d}_ema.safetensors",
                dtype=export_dtype, decoder_capability=dec_cap, assert_zero_delta_w=no_delta_w,
            )

    st_path = model.export_v8_safetensors(
        out_dir / "model_v8.safetensors",
        dtype=export_dtype,
        decoder_capability=dec_cap,
        assert_zero_delta_w=no_delta_w,
    )
    _export_state_dict_safetensors(
        cfg, emas["0.999"], out_dir / "model_v8_ema.safetensors",
        dtype=export_dtype, decoder_capability=dec_cap, assert_zero_delta_w=no_delta_w,
    )
    return {
        "steps_done": steps_done,
        "last_step": metrics_last.get("step", start_step),
        "last_metrics": metrics_last,
        "checkpoint_pt": str(ckpt_path),
        "safetensors": str(st_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SingleEncV8 (DOPE v3.0) resumable trainer")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--run-name", type=str, default="v8_run")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--allow-shared-gpu-vram-gb", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--episodes-per-step", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--max-p-curriculum", type=int, default=0)
    ap.add_argument("--curriculum-steps", type=int, default=1000)
    ap.add_argument("--n-fit", type=int, default=256)
    ap.add_argument("--n-qry", type=int, default=192)
    ap.add_argument("--n-syn", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-enc-layers", type=int, default=4)
    ap.add_argument("--n-dec-layers", type=int, default=2)
    ap.add_argument("--decoder-only", action="store_true")
    ap.add_argument(
        "--code-dim",
        type=int,
        default=DEC_CODE_DIM,
        help="learned FSQ code width; native wire Tiers 1/2/3 use 64/128/256 (32/64/128 B), "
        "other widths are research-only capacity/channel probes",
    )
    ap.add_argument("--ff-mult", type=int, default=2)
    ap.add_argument("--export-dtype", type=str, default="f32", choices=["f32", "bf16"])
    ap.add_argument(
        "--all-rank-norm",
        action="store_true",
        help="use historical 12-rank loading normalization and omit dope_dse8_decoder=active-rank-v1 metadata",
    )
    ap.add_argument(
        "--historical-forward",
        action="store_true",
        help="reproduce pre-parity training forward (no packet analytic/typed-marginal/FIT-midrank terms, "
        "all-12-rank loading norm, omits active-rank-v1 metadata and marker tensor)",
    )
    ap.add_argument(
        "--no-delta-w",
        action="store_true",
        help="zero and freeze delta_W linear correction head while keeping feature, nonlinear, and residual-scale "
        "heads matched; asserted exact zero on every SafeTensors export",
    )
    ap.add_argument("--use-rust-packet", action="store_true")
    ap.add_argument("--budget-bytes", type=int, default=2048)
    ap.add_argument("--prior-mix-prob", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-schedule", type=str, default="wsd", choices=["cosine", "wsd"])
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--direct-cond-weight", type=float, default=0.60)
    ap.add_argument("--anchor-weight", type=float, default=0.15)
    ap.add_argument("--anchor-nonlinear-weight", type=float, default=0.40)
    ap.add_argument("--joint-swd-weight", type=float, default=0.0)
    ap.add_argument("--alpha-floor", type=float, default=0.0)
    ap.add_argument("--alpha-floor-steps", type=int, default=1000)
    ap.add_argument("--alpha-floor-weight", type=float, default=0.05)
    ap.add_argument("--resid-cond-weight", type=float, default=0.0)
    ap.add_argument("--backbone-lr-mult", type=float, default=1.0)
    ap.add_argument("--backbone-warmup-steps", type=int, default=500)
    ap.add_argument("--init-from", type=str, default="")
    ap.add_argument("--freeze-target-heads", action="store_true")
    ap.add_argument("--freeze-x-heads", action="store_true")
    ap.add_argument("--use-pcgrad", action="store_true")
    ap.add_argument("--use-asam", action="store_true")
    ap.add_argument("--asam-every", type=int, default=10)
    ap.add_argument("--contrastive-prob", type=float, default=0.15)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--sync-dir", type=str, default="")
    ap.add_argument("--sync-every", type=int, default=25)
    ap.add_argument("--sync-outer-lr", type=float, default=0.25)
    ap.add_argument("--sync-outer-momentum", type=float, default=0.70)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--stop-at-step",
        type=int,
        default=0,
        help="optional early stop step (0=disabled) preserving --steps schedule/phase boundaries for exact resume",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="disable cuDNN autotune and enable deterministic PyTorch kernels for exact GPU resume parity",
    )
    ap.add_argument(
        "--no-tf32",
        action="store_true",
        help="disable CUDA TF32 matmul/cuDNN precision (kept enabled by default for throughput)",
    )
    ap.add_argument("--max-wall-s", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train_v8(parse_args()), indent=2))
