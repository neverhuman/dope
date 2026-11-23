"""I/O, episode tensorization, CUDA guard, EMA, and Valley Soup helpers for `s3_v8_trainer.py`."""
from __future__ import annotations

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

    Supports BOTH precomputed C6 `dope-ffi` episode payloads (`ev_col`/`dec_col`/`knots`)
    and raw FIT/QRY episode dicts (computes FIT-only evidence + DSE8 Step-0 packet on the fly).
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
