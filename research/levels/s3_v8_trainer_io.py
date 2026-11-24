"""I/O, episode tensorization, CUDA guard, EMA, and Valley Soup helpers for `s3_v8_trainer.py`."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from research.levels.v8_evidence import extract_evidence_fast_numpy
from research.levels.v8_packet import (
    DEC_CODE_DIM,
    DEC_COL_DIM,
    DEC_EDGE_DIM,
    DEC_GLOB_DIM,
    KNOT_COUNT,
    V8_MAX_CLASSES,
    build_decoder_inputs_py,
    fit_dse8_rust,
    unpack_dse8_packet_py,
)
from research.levels.v8_parity_decode import compute_fit_midrank_z_np, extract_packet_analytic_tensors
from research.levels.v8_single_encoding import SingleEncV8


def _resolve_target_gpu_uuid(dev_idx: int) -> str:
    """Resolve the verified physical NVIDIA GPU UUID for logical CUDA device `dev_idx` (fail-closed)."""
    props = torch.cuda.get_device_properties(dev_idx)
    raw_uuid = getattr(props, "uuid", None)
    if raw_uuid is not None:
        s = str(raw_uuid).strip()
        if s:
            return s if s.startswith("GPU-") else f"GPU-{s}"
    # Fallback via CUDA_VISIBLE_DEVICES -> nvidia-smi physical index/UUID table.
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    idx_to_uuid: dict[str, str] = {}
    ordered_uuids: list[str] = []
    for ln in out.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 2 and parts[1]:
            idx_to_uuid[parts[0]] = parts[1]
            ordered_uuids.append(parts[1])
    if vis:
        tokens = [t.strip() for t in vis.split(",") if t.strip()]
        if dev_idx >= len(tokens):
            raise RuntimeError(f"CUDA_EXCLUSIVE_GUARD: dev_idx {dev_idx} out of CUDA_VISIBLE_DEVICES={vis!r}")
        tok = tokens[dev_idx]
        if tok.startswith("GPU-"):
            if tok in ordered_uuids:
                return tok
            raise RuntimeError(f"CUDA_EXCLUSIVE_GUARD: unknown GPU UUID {tok!r} in CUDA_VISIBLE_DEVICES")
        if tok in idx_to_uuid:
            return idx_to_uuid[tok]
        raise RuntimeError(f"CUDA_EXCLUSIVE_GUARD: cannot map CUDA_VISIBLE_DEVICES token {tok!r} to GPU UUID")
    if dev_idx < len(ordered_uuids):
        return ordered_uuids[dev_idx]
    raise RuntimeError(f"CUDA_EXCLUSIVE_GUARD: indeterminate GPU UUID mapping for dev_idx={dev_idx}")


def check_cuda_exclusive_guard(device: str, shared_vram_gb: float = 0.0) -> None:
    """Fail-closed per-GPU CUDA guard scoped to the verified physical GPU UUID.

    By default (`shared_vram_gb <= 0`), refuses CUDA launch if ANY foreign numeric PID occupies the
    target GPU UUID (allowing independent jobs on separate GPUs of a multi-GPU host like xbabe7!).
    Fails closed if `nvidia-smi` is missing or GPU UUID mapping is indeterminate.
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
        target_uuid = _resolve_target_gpu_uuid(dev_idx)
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        raise RuntimeError(f"CUDA_EXCLUSIVE_GUARD: fail-closed GPU inspection error: {exc}") from exc
    my_pid = os.getpid()
    for ln in out.splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 2:
            raise RuntimeError(f"CUDA_EXCLUSIVE_GUARD: malformed nvidia-smi compute row: {s!r}")
        gpu_uuid = parts[0]
        if gpu_uuid != target_uuid:
            continue
        try:
            pid = int(parts[1])
        except ValueError as exc:
            raise RuntimeError(f"CUDA_EXCLUSIVE_GUARD: non-numeric GPU PID in row: {s!r}") from exc
        if pid != my_pid:
            raise RuntimeError(f"CUDA_EXCLUSIVE_GUARD: foreign GPU process on {target_uuid}: {s}")


def consume_episode_tensors(
    ep: dict[str, Any],
    device: torch.device,
    use_rust_packet: bool = False,
    budget_bytes: int = 2048,
    seed: int = 0,
    drop_tier_prob: float = 0.30,
) -> dict[str, Any]:
    """Convert one fresh FIT-only episode dict into on-device tensors for `SingleEncV8`.

    Supports BOTH precomputed C6 `dope-ffi` episode payloads (`ev_col`/`dec_col`/`knots`)
    and raw FIT/QRY episode dicts (computes FIT-only evidence + DSE8 Step-0 packet on the fly).
    """
    non_blocking = device.type == "cuda"
    if "ev_col" in ep and "dec_col" in ep and "knots" in ep:
        def _t(key: str) -> torch.Tensor:
            v = ep[key]
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v, dtype=np.float32))
            return t.to(device=device, dtype=torch.float32, non_blocking=non_blocking)

        pkt_analytic = None
        if ep.get("packet"):
            try:
                pkt_analytic = extract_packet_analytic_tensors(
                    unpack_dse8_packet_py(bytes(ep["packet"])), device
                )
            except Exception:
                pkt_analytic = None
        if "Z_qry" in ep:
            z_qry_t = _t("Z_qry")
        else:
            xf_np = np.asarray(ep["X_fit"], dtype=np.float32)
            xq_np = np.asarray(ep["X_qry"], dtype=np.float32)
            z_qry_t = torch.from_numpy(compute_fit_midrank_z_np(xf_np, xq_np)).to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            )
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
            "z_qry": z_qry_t,
            "task_id": int(ep["task_id"]),
            "n_classes": int(ep["n_classes"]),
            "pkt_analytic": pkt_analytic,
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

    pkt_analytic = None
    if use_rust_packet:
        pkt_b = fit_dse8_rust(
            Xf, yf, task_id, budget_bytes=budget_bytes, normalized_only=True,
            col_meta=ep.get("col_meta"), seed=seed,
        )
        pkt = unpack_dse8_packet_py(pkt_b)
        pkt_analytic = extract_packet_analytic_tensors(pkt, device)
        dc, de, dg, _dz = build_decoder_inputs_py(pkt)
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
        p_eff = p
        Xf_c = np.nan_to_num(Xf, nan=0.0)
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
            w_base[0], b_base[0] = sol[:p], sol[p]
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
        is_cont = (ev["col"][:, 6] < 0.5).astype(np.float32)

    zq_np = compute_fit_midrank_z_np(Xf[:, :p_eff], Xq[:, :p_eff])
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
        "z_qry": torch.from_numpy(zq_np).to(device),
        "task_id": task_id,
        "n_classes": n_classes,
        "pkt_analytic": pkt_analytic,
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
    run_flags: dict[str, Any] | None = None,
    counters: dict[str, int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    payload: dict[str, Any] = {
        "step": step,
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "emas": emas,
        "rng_state": rng_state,
    }
    if run_flags is not None:
        payload["run_flags"] = dict(run_flags)
    if counters is not None:
        payload["counters"] = {k: int(v) for k, v in counters.items()}
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _export_state_dict_safetensors(
    model_cfg: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    out_path: Path,
    dtype: str = "f32",
    decoder_capability: str | None = "active-rank-v1",
    assert_zero_delta_w: bool = False,
) -> Path:
    tmp_model = SingleEncV8(**model_cfg).cpu()
    tmp_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
    return tmp_model.export_v8_safetensors(
        out_path,
        dtype=dtype,
        decoder_capability=decoder_capability,
        assert_zero_delta_w=assert_zero_delta_w,
    )


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
