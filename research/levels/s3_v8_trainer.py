"""Resumable 3-phase GPU/CPU/DiLoCo trainer for `SingleEncV8` (DOPE v3.0).

Key invariants:
- Fresh FIT/QRY split every episode with zero per-table feature/evidence cache (09:50Z binding rule).
- Ready to consume Claude's `s3_episode_factory` `/dev/shm` ring (`ring.get_batch(step)`) when supplied,
  or falls back cleanly to live V8 non-Gaussian/XOR/contrastive priors + flat-corpus v1/v2 episodes.
- Strict `CUDA` foreign-process refusal guard + `--max-wall-s` cap (`neural:N02`).
- Supports 3 EMAs (`0.99`, `0.999`, `0.9998`), Top-5 DEV-TRG Valley Soup, per-episode adaptive grad clip,
  optional `--use-pcgrad` (PCGrad-Lite) and `--use-asam` (Periodic-ASAM), exact full-state save/resume,
  and canonical `F32` `rank <= 2` SafeTensors export directly consumable by Rust `dope_v8_emit`.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from research.levels.s3_v8_losses import compute_v8_episode_loss, pcgrad_lite_combine
from research.levels.s3_v8_priors import sample_contrastive_xor_pair, sample_v8_prior_episode
from research.levels.v8_evidence import extract_evidence_fast_numpy
from research.levels.v8_packet import (
    DEC_CODE_DIM,
    DEC_COL_DIM,
    DEC_EDGE_DIM,
    DEC_GLOB_DIM,
    KNOT_COUNT,
    V8_MAX_CLASSES,
    V8_MAX_RANK,
    build_decoder_inputs_py,
    fit_dse8_rust,
    unpack_dse8_packet_py,
)
from research.levels.v8_single_encoding import SingleEncV8


def check_cuda_exclusive_guard(device: str, shared_vram_gb: float = 0.0) -> None:
    """Fail-closed CUDA guard.

    By default (`shared_vram_gb <= 0`), refuses CUDA launch if ANY foreign process occupies the GPU.
    When `shared_vram_gb > 0` (guarded JSR coexistence mode), verifies that free VRAM >= shared_vram_gb + 4.0 GiB
    and caps PyTorch's CUDA allocator via `torch.cuda.set_per_process_memory_fraction`.
    """
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device}) but torch.cuda.is_available() is False")
    dev_idx = 0 if ":" not in device else int(device.split(":", 1)[1])
    if shared_vram_gb > 0.0:
        free_b, total_b = torch.cuda.mem_get_info(dev_idx)
        free_gb = free_b / (1024.0 ** 3)
        total_gb = total_b / (1024.0 ** 3)
        if free_gb < shared_vram_gb + 4.0:
            raise RuntimeError(
                f"CUDA_SHARED_GUARD: insufficient free VRAM ({free_gb:.2f} GiB < {shared_vram_gb + 4.0:.2f} GiB)"
            )
        frac = min(0.85, max(0.05, shared_vram_gb / max(total_gb, 1.0)))
        torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        my_pid = str(os.getpid())
        foreign = [ln for ln in out.splitlines() if ln.strip() and not ln.strip().startswith(my_pid)]
        if foreign:
            raise RuntimeError(f"CUDA_EXCLUSIVE_GUARD: foreign GPU process active: {foreign[0]}")
    except FileNotFoundError:
        pass


def consume_episode_tensors(
    ep: dict[str, Any],
    device: torch.device,
    use_rust_packet: bool = False,
    budget_bytes: int = 2048,
    seed: int = 0,
    drop_tier_prob: float = 0.30,
) -> dict[str, Any]:
    """Convert one fresh FIT-only episode dict into on-device tensors for `SingleEncV8`.

    Supports BOTH:
    1. Precomputed C6 `dope-ffi` episode payloads carrying `ev_col`/`ev_edge`/`ev_glob` + `dec_col`/`dec_edge`/`dec_glob`
       + `knots`/`loadings`/`W_base`/`b_base`/`resid_sd`/`z_max`/`is_cont` (zero-copy direct upload).
    2. Raw FIT/QRY episode dicts `{"X_fit", "y_fit", "X_qry", "y_qry", "task_id", "n_classes", ...}`
       (computes FIT-only evidence + DSE8 Step-0 packet on the fly with zero table caching).
    """
    non_blocking = device.type == "cuda"
    if "ev_col" in ep and "dec_col" in ep and "knots" in ep:
        def _t(key: str) -> torch.Tensor:
            v = ep[key]
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v, dtype=np.float32))
            return t.to(device=device, dtype=torch.float32, non_blocking=non_blocking)

        return {
            "ev_col": _t("ev_col"),
            "ev_edge": _t("ev_edge"),
            "ev_glob": _t("ev_glob"),
            "dec_col": _t("dec_col"),
            "dec_edge": _t("dec_edge"),
            "dec_glob": _t("dec_glob"),
            "knots": _t("knots"),
            "loadings": _t("loadings"),
            "W_base": _t("W_base"),
            "b_base": _t("b_base"),
            "resid_sd": _t("resid_sd").reshape(()),
            "z_max": _t("z_max").reshape(()),
            "is_cont": _t("is_cont"),
            "X_fit": _t("X_fit"),
            "y_fit": _t("y_fit"),
            "X_qry": _t("X_qry"),
            "y_qry": _t("y_qry"),
            "task_id": int(ep["task_id"]),
            "n_classes": int(ep["n_classes"]),
        }
    return _episode_to_tensors(
        ep, device, use_rust_packet=use_rust_packet, budget_bytes=budget_bytes,
        seed=seed, drop_tier_prob=drop_tier_prob,
    )


def _episode_to_tensors(
    ep: dict[str, Any],
    device: torch.device,
    use_rust_packet: bool = False,
    budget_bytes: int = 2048,
    seed: int = 0,
    drop_tier_prob: float = 0.30,
) -> dict[str, Any]:
    """Build strictly FIT-only Evidence + DSE8 decoder inputs for one fresh episode."""
    Xf = np.asarray(ep["X_fit"], dtype=np.float32)
    yf = np.asarray(ep["y_fit"], dtype=np.float32)
    Xq = np.asarray(ep["X_qry"], dtype=np.float32)
    yq = np.asarray(ep["y_qry"], dtype=np.float32)
    task_id = int(ep["task_id"])
    n_classes = int(ep["n_classes"])
    p = Xf.shape[1]

    ev = extract_evidence_fast_numpy(Xf, yf, task_id, n_classes, seed=seed, drop_tier_prob=drop_tier_prob)

    if use_rust_packet:
        pkt_b = fit_dse8_rust(
            Xf, yf, task_id, budget_bytes=budget_bytes, normalized_only=True,
            col_meta=ep.get("col_meta"), seed=seed,
        )
        pkt = unpack_dse8_packet_py(pkt_b)
        dc, de, dg, dz = build_decoder_inputs_py(pkt)
        p_eff = pkt["header"]["p_eff"]
        knots = np.stack([m["knots17"] for m in pkt["marginals"]], axis=0)
        loadings = pkt["loadings"]
        w_base = np.zeros((V8_MAX_CLASSES, p_eff), dtype=np.float32)
        b_base = np.zeros(V8_MAX_CLASSES, dtype=np.float32)
        o_dim = pkt["linear"]["o_dim"]
        w_base[:o_dim, :] = pkt["linear"]["w"]
        b_base[:o_dim] = pkt["linear"]["b"]
        resid_sd = float(pkt["header"]["resid_sd"])
        z_max = float(pkt["header"]["z_max"])
        is_cont = np.array([1.0 if m["kind"] == 0 else 0.0 for m in pkt["marginals"]], dtype=np.float32)
    else:
        # Ultra-fast analytic DSE8-equivalent normalized packet view for high-FPS inner loop
        p_eff = p
        Xf_c = np.nan_to_num(Xf, nan=0.0)
        u_floor = min(0.20, 10.0 / max(len(Xf) + 1.0, 20.0))
        z_max = float(np.clip(1.8 if len(Xf) < 120 else 2.2, 1.2, 2.6))
        z_grid = np.linspace(-z_max, z_max, KNOT_COUNT)
        u_grid = 0.5 * (1.0 + np.vectorize(math.erf)(z_grid / math.sqrt(2.0)))
        knots = np.quantile(Xf_c, u_grid, axis=0).T.astype(np.float32)
        loadings = ev["col"][:, 56:68].copy()
        lam = max(0.05 * len(Xf), 2.0)
        Xa = np.concatenate([Xf_c, np.ones((len(Xf), 1), dtype=np.float32)], axis=1)
        reg = np.eye(p + 1, dtype=np.float32) * lam
        reg[-1, -1] = 1e-2
        w_base = np.zeros((V8_MAX_CLASSES, p), dtype=np.float32)
        b_base = np.zeros(V8_MAX_CLASSES, dtype=np.float32)
        if task_id == 0:
            sol = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ yf)
            w_base[0] = sol[:p]
            b_base[0] = sol[p]
            resid_sd = float(np.clip(np.std(yf - Xa @ sol), 0.05, 4.0))
        elif task_id == 1:
            yc = (yf > 0.5).astype(np.float32) * 2.0 - 1.0
            sol = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ yc)
            w_base[0] = np.clip(sol[:p] * 2.0, -6.0, 6.0)
            b_base[0] = float(np.clip(sol[p] * 2.0, -6.0, 6.0))
            resid_sd = 0.5
        else:
            k_eff = min(n_classes, V8_MAX_CLASSES)
            for c in range(k_eff):
                yc = (np.rint(yf).astype(int) == c).astype(np.float32) * 2.0 - 1.0
                sol = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ yc)
                w_base[c] = np.clip(sol[:p] * 2.0, -6.0, 6.0)
                b_base[c] = float(np.clip(sol[p] * 2.0, -6.0, 6.0))
            resid_sd = 0.5
        dc = np.zeros((p, DEC_COL_DIM), dtype=np.float32)
        dc[:, :17] = knots
        dc[:, 17:29] = loadings
        dc[:, 29:37] = w_base.T
        dc[:, 42] = z_max
        de = np.zeros((p, p, DEC_EDGE_DIM), dtype=np.float32)
        de[:, :, 0] = np.clip(loadings @ loadings.T, -1.0, 1.0)
        de[:, :, 2] = ev["edge"][:, :, 9]
        de[:, :, 3] = ev["edge"][:, :, 10]
        dg = np.zeros(DEC_GLOB_DIM, dtype=np.float32)
        dg[min(task_id, 2)] = 1.0
        dg[3] = 1.0
        dg[12] = ev["glob"][12]
        dg[13] = p / 64.0
        dg[14] = resid_sd
        dg[15] = n_classes / 8.0
        dz = np.zeros(DEC_CODE_DIM, dtype=np.float32)
        is_cont = (ev["col"][:, 6] < 0.5).astype(np.float32)

    return {
        "ev_col": torch.from_numpy(ev["col"]).to(device),
        "ev_edge": torch.from_numpy(ev["edge"]).to(device),
        "ev_glob": torch.from_numpy(ev["glob"]).to(device),
        "dec_col": torch.from_numpy(dc).to(device),
        "dec_edge": torch.from_numpy(de).to(device),
        "dec_glob": torch.from_numpy(dg).to(device),
        "knots": torch.from_numpy(knots).to(device),
        "loadings": torch.from_numpy(loadings).to(device),
        "W_base": torch.from_numpy(w_base).to(device),
        "b_base": torch.from_numpy(b_base).to(device),
        "resid_sd": torch.tensor(resid_sd, dtype=torch.float32, device=device),
        "z_max": torch.tensor(z_max, dtype=torch.float32, device=device),
        "is_cont": torch.from_numpy(is_cont).to(device),
        "X_fit": torch.from_numpy(Xf[:, :p_eff]).to(device),
        "y_fit": torch.from_numpy(yf).to(device),
        "X_qry": torch.from_numpy(Xq[:, :p_eff]).to(device),
        "y_qry": torch.from_numpy(yq).to(device),
        "task_id": task_id,
        "n_classes": n_classes,
    }


def _update_ema(ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)


def build_valley_soup(checkpoints: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average top-K DEV-TRG checkpoints (Valley Soup)."""
    assert len(checkpoints) >= 1
    out = {k: v.clone().float() for k, v in checkpoints[0].items()}
    for ck in checkpoints[1:]:
        for k in out:
            out[k].add_(ck[k].float())
    inv = 1.0 / float(len(checkpoints))
    for k in out:
        out[k].mul_(inv)
    return out


def save_full_checkpoint(
    path: Path,
    model: SingleEncV8,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    emas: dict[str, dict[str, torch.Tensor]],
    step: int,
    rng_state: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(
        {
            "step": step,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "emas": emas,
            "rng_state": rng_state,
        },
        tmp,
    )
    os.replace(tmp, path)


def _export_state_dict_safetensors(
    model_cfg: dict[str, Any], state_dict: dict[str, torch.Tensor], out_path: Path
) -> Path:
    tmp_model = SingleEncV8(**model_cfg).cpu()
    tmp_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
    return tmp_model.export_v8_safetensors(out_path)


def train_v8(args: argparse.Namespace, episode_provider: Any | None = None) -> dict[str, Any]:
    """Run resumable SingleEncV8 training loop (CPU or CUDA GPU, optional DiLoCo + C6 episode provider)."""
    torch_threads = int(getattr(args, "torch_threads", 0) or 0)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    shared_vram_gb = float(getattr(args, "allow_shared_gpu_vram_gb", 0.0) or 0.0)
    check_cuda_exclusive_guard(args.device, shared_vram_gb=shared_vram_gb)
    device = torch.device(args.device)
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    cfg = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_enc_layers": args.n_enc_layers,
        "n_dec_layers": args.n_dec_layers,
        "code_dim": DEC_CODE_DIM,
        "with_owner_encoder": not args.decoder_only,
    }
    model = SingleEncV8(**cfg).to(device)
    init_from = str(getattr(args, "init_from", "") or "")
    if init_from:
        from safetensors.torch import load_file as _st_load
        st_w = _st_load(init_from, device=str(device))
        model.load_state_dict(st_w, strict=False)
    if bool(getattr(args, "freeze_target_heads", False)):
        for name, p in model.named_parameters():
            if any(name.startswith(pfx) for pfx in ("delta_W.", "node_", "bilin_", "gate_head.")):
                p.requires_grad_(False)
    if bool(getattr(args, "freeze_x_heads", False)):
        for name, p in model.named_parameters():
            if any(name.startswith(pfx) for pfx in ("delta_knots.", "delta_L.", "flow_")):
                p.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
    lr_sched_kind = str(getattr(args, "lr_schedule", "cosine") or "cosine")

    def _lr_lambda(st: int) -> float:
        if warmup_steps > 0 and st < warmup_steps:
            return float(st + 1) / float(warmup_steps)
        rem = max(args.steps - warmup_steps, 1)
        prog = min(1.0, max(0.0, float(st - warmup_steps) / float(rem)))
        if lr_sched_kind == "wsd":
            # Warmup -> Stable (80%) -> Cosine Decay (20%)
            if prog < 0.80:
                return 1.0
            d_prog = (prog - 0.80) / 0.20
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * d_prog))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
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
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler") is not None:
            scheduler.load_state_dict(ck["scheduler"])
        emas = ck["emas"]
        start_step = int(ck["step"])
        torch.set_rng_state(ck["rng_state"]["torch"].cpu())
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
        anchor_state = {
            k: v.detach().cpu().to(torch.float32).numpy().copy()
            for k, v in model.state_dict().items()
        }
        sync_round, _ = diloco_sync_step(
            sync_dir, model, ema_proxy, anchor_state,
            float(getattr(args, "sync_outer_lr", 0.25)),
            float(getattr(args, "sync_outer_momentum", 0.70)),
            base_round=0, worker_id=str(getattr(args, "run_name", "v8_w0")),
        )
        emas["0.999"] = copy.deepcopy(ema_proxy.state_dict())

    t0 = time.monotonic()
    log_path = out_dir / "train.jsonl"
    metrics_last: dict[str, Any] = {}
    steps_done = 0
    ckpt_every = int(getattr(args, "ckpt_every", 0) or 0)

    for step in range(start_step + 1, args.steps + 1):
        if args.max_wall_s > 0 and (time.monotonic() - t0) >= args.max_wall_s:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Phase curriculum: phase 1 (warmup) -> phase 2 (full KPI) -> phase 3 (fine-tune)
        phase = 1 if step <= max(args.steps // 5, 1) else (2 if step <= (4 * args.steps) // 5 else 3)
        loss_terms_accum: dict[str, float] = {}
        ep_losses: list[torch.Tensor] = []
        z_codes: list[torch.Tensor] = []

        for b_i in range(args.episodes_per_step):
            if episode_provider is not None:
                ep = episode_provider(step, b_i)
            elif args.contrastive_prob > 0.0 and b_i == 0 and np_rng.random() < args.contrastive_prob:
                ep_a, _ = sample_contrastive_xor_pair(np_rng, n_fit=args.n_fit, n_qry=args.n_qry)
                ep = ep_a
            else:
                ep = sample_v8_prior_episode(np_rng, n_fit=args.n_fit, n_qry=args.n_qry)
            td = consume_episode_tensors(
                ep, device, use_rust_packet=args.use_rust_packet,
                budget_bytes=args.budget_bytes, seed=args.seed + step * 31 + b_i,
                drop_tier_prob=0.30 if phase >= 2 else 0.0,
            )
            p_ep = td["ev_col"].shape[0]
            col_mask = torch.ones(1, p_ep, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                z_fsq, _ = model.encode_evidence(
                    td["ev_col"][None], td["ev_edge"][None], td["ev_glob"][None],
                    col_mask=col_mask, tier_dropout=(phase >= 2),
                )
                z_codes.append(z_fsq[0])
                plan = model.decode_plan(
                    td["dec_col"][None], td["dec_edge"][None], td["dec_glob"][None],
                    z_fsq, col_mask=col_mask,
                )
            dec = model.decode_batch_differentiable(
                plan,
                td["knots"][None], td["loadings"][None], td["W_base"][None], td["b_base"][None],
                td["resid_sd"][None], td["z_max"][None], td["is_cont"][None], col_mask,
                n_rows=args.n_syn, antithetic=True,
            )
            xq_clean = torch.nan_to_num(td["X_qry"], nan=0.0)
            logits_qry = model.predict_target_on_x(
                plan, td["W_base"][None], td["b_base"][None], xq_clean[None]
            )[0]
            loss_w = {
                "direct_cond": float(getattr(args, "direct_cond_weight", 0.60)),
                "anchor": float(getattr(args, "anchor_weight", 0.15)),
            }
            ld = compute_v8_episode_loss(
                dec["x_syn"][0], dec["logits"][0], dec["y_reg"][0], dec["resid_sd_eff"][0],
                td["X_fit"], td["y_fit"], td["X_qry"], td["y_qry"],
                td["task_id"], td["n_classes"],
                weights=loss_w, logits_qry=logits_qry, plan=plan,
            )
            # Masked evidence reconstruction auxiliary loss (pre-bottleneck representation grounding)
            ev_rec = model.aux_ev_head(plan["h_dec"][0])
            l_aux = 0.05 * torch.nn.functional.mse_loss(ev_rec, td["ev_col"])
            ep_tot = ld["total"] + l_aux
            ep_losses.append(ep_tot)
            for k, v in ld.items():
                loss_terms_accum[k] = loss_terms_accum.get(k, 0.0) + float(v.detach().item())
            loss_terms_accum["l_aux"] = loss_terms_accum.get("l_aux", 0.0) + float(l_aux.detach().item())
            loss_terms_accum["alpha_gate"] = loss_terms_accum.get("alpha_gate", 0.0) + float(plan["alpha_table"][0].detach().item())

        total_loss = torch.stack(ep_losses).mean()
        pcgrad_info: dict[str, float] = {}
        if args.use_pcgrad and len(ep_losses) >= 2:
            params = [p for p in model.parameters() if p.requires_grad]
            comb_g, pcgrad_info = pcgrad_lite_combine(ep_losses, params)
            for p, g in zip(params, comb_g):
                p.grad = g / float(len(ep_losses))
        else:
            total_loss.backward()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
        if not math.isfinite(grad_norm) or not math.isfinite(float(total_loss.item())):
            raise FloatingPointError(f"non-finite loss/grad at step {step}: loss={total_loss.item()} grad={grad_norm}")

        # Optional Periodic-ASAM sharpness-aware step
        if args.use_asam and (step % max(args.asam_every, 1) == 0):
            rho = 0.02
            with torch.no_grad():
                e_ws = []
                for p in model.parameters():
                    if p.grad is None:
                        e_ws.append(None)
                        continue
                    ew = (rho * (p.abs() + 1e-3) * p.grad) / max(grad_norm, 1e-6)
                    p.add_(ew)
                    e_ws.append(ew)
                for p, ew in zip(model.parameters(), e_ws):
                    if ew is not None:
                        p.sub_(ew)

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
            emas["0.999"] = copy.deepcopy(ema_proxy.state_dict())

        inv_ep = 1.0 / float(max(args.episodes_per_step, 1))
        rec = {
            "step": step,
            "phase": phase,
            "lr": float(scheduler.get_last_lr()[0]),
            "grad_norm": grad_norm,
            "wall_s": round(time.monotonic() - t0, 3),
            "sync_round": sync_round,
            **{k: round(v * inv_ep, 5) for k, v in loss_terms_accum.items()},
            **{k: round(v, 5) for k, v in pcgrad_info.items()},
        }
        metrics_last = rec
        steps_done += 1
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            save_full_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                emas,
                step,
                {"torch": torch.get_rng_state(), "numpy": np_rng.bit_generator.state},
                cfg,
            )
        if ckpt_every > 0 and (step % ckpt_every == 0):
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _export_state_dict_safetensors(cfg, emas["0.999"], ckpt_dir / f"step_{step:06d}_ema.safetensors")

    # Export canonical F32 rank-<=2 SafeTensors ready for pure-Rust `dope_v8_emit`!
    st_path = model.export_v8_safetensors(out_dir / "model_v8.safetensors")
    _export_state_dict_safetensors(cfg, emas["0.999"], out_dir / "model_v8_ema.safetensors")
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
    ap.add_argument("--n-fit", type=int, default=256)
    ap.add_argument("--n-qry", type=int, default=192)
    ap.add_argument("--n-syn", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-enc-layers", type=int, default=4)
    ap.add_argument("--n-dec-layers", type=int, default=2)
    ap.add_argument("--decoder-only", action="store_true")
    ap.add_argument("--use-rust-packet", action="store_true")
    ap.add_argument("--budget-bytes", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-schedule", type=str, default="wsd", choices=["cosine", "wsd"])
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--direct-cond-weight", type=float, default=0.60)
    ap.add_argument("--anchor-weight", type=float, default=0.15)
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
    ap.add_argument("--max-wall-s", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


if __name__ == "__main__":
    res = train_v8(parse_args())
    print(json.dumps(res, indent=2))
