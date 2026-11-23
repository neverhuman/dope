"""Function-preserving model surgery for `SingleEncV8` (DOPE v3.0).

Supports two strictly offline, deterministic SafeTensors (`F32`, `rank <= 2`) surgery modes that
produce checkpoints natively loadable by frozen `dope 3.0.0` (`SingleEncV8Weights`) with zero
`engine/` changes:

1. `v7-to-v8`: Grafts a pre-trained `SingleEncV7` checkpoint (e.g. shipped `dope 2.0.0`
   `models/single-enc-v7/model.safetensors`, `d_model=160, n_heads=4, n_layers=3`) into
   `SingleEncV8(d_model=160, n_heads=4, n_enc_layers=3, n_dec_layers=3)` via exact 1:1 input
   coordinate remapping (`col_in` [0..36] -> `dec_col_in` [0..36], 38 (is_disc) -> 39 (is_int),
   [39..41] -> [41..43]; `glob_in` [0..15] minus slot 3 -> the same `dec_glob_in` slots), exact Pre-LN Split-QKV block transfer with zero-init
   `edge_proj`, pre-trained column-attention cloning into `enc_blocks`, shipped `delta_W` transfer,
   bilinear cubic-saddle-point breaking (`bilin_u/v ~ N(0, 0.05), bilin_w = 0`), and open initial
   trust gate (`alpha_table = 0.25`).
2. `repair-v8`: Performs function-preserving in-place surgery on an existing `SingleEncV8`
   checkpoint (e.g. `v8-r2` EMA `d_model=128, enc=4, dec=2`):
   - Unsticks the dead cubic saddle point (`bilin_u/v` initialized non-zero while `bilin_w = 0`,
     preserving exact Step-0 output).
   - Reopens a collapsed trust gate (`alpha_old -> alpha_new = 0.25`) by damping
     `gate_head.weight[0]` and setting `gate_head.bias[0] = logit(alpha_new) + 2.0` while rescaling
     `node_leaf.{weight,bias}` by `alpha_old / alpha_new`, preserving Step-0 `alpha * Soft-NODE`
     magnitude while amplifying nonlinear & encoder gradients ~10x.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
import torch

from research.levels.v8_packet import DEC_CODE_DIM
from research.levels.v8_single_encoding import SingleEncV8

# Exact 1:1 feature-index correspondence from V7 `col_in` (42 base dims) to V8 `dec_col_in` (48 dims).
V7_TO_V8_COL_MAP: list[tuple[int, int]] = (
    [(i, i) for i in range(17)]        # knots17
    + [(17 + i, 17 + i) for i in range(12)]  # loadings (rank 12)
    + [(29 + i, 29 + i) for i in range(8)]   # linear.w (up to 8 classes)
    + [(38, 39)]                             # is_disc -> is_int (V8's discrete flag)
    + [(39, 41), (40, 42), (41, 43)]         # log1p(span), z_max, knot_level/17
)
# V7 never sets global slot 3, while V8 slot 3 is `normalized_only`: leave it untrained (zero weight).
V7_TO_V8_GLOB_MAP: list[tuple[int, int]] = [(i, i) for i in range(16) if i != 3]
# V7 encodes width as p/48 while DSE8 encodes p/64 (`V8_MAX_COLS = 64`), so rescale slot 13 by 64/48.
V7_TO_V8_GLOB_SCALE: dict[int, float] = {13: 64.0 / 48.0}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logit(p: float) -> float:
    pc = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(pc / (1.0 - pc))


def _infer_v7_shape(sd: dict[str, torch.Tensor]) -> tuple[int, int]:
    d_model = int(sd["col_in.weight"].shape[0])
    n_layers = 0
    while f"blocks.{n_layers}.q_proj.weight" in sd:
        n_layers += 1
    if n_layers <= 0:
        raise ValueError("donor V7 checkpoint contains no `blocks.{i}.q_proj.weight` tensors")
    return d_model, n_layers


def _infer_v8_shape(sd: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    d_model = int(sd["dec_col_in.weight"].shape[0])
    n_enc = 0
    while f"enc_blocks.{n_enc}.q_proj.weight" in sd:
        n_enc += 1
    n_dec = 0
    while f"dec_blocks.{n_dec}.q_proj.weight" in sd:
        n_dec += 1
    if n_dec <= 0:
        raise ValueError("donor V8 checkpoint contains no `dec_blocks.{i}.q_proj.weight` tensors")
    return d_model, n_enc, n_dec


def _copy_block_from_v7(
    dst_sd: dict[str, torch.Tensor],
    src_sd: dict[str, torch.Tensor],
    src_idx: int,
    dst_prefix: str,
) -> None:
    for sub in ("norm1", "q_proj", "k_proj", "v_proj", "o_proj", "norm2", "ff1", "ff2"):
        for kind in ("weight", "bias"):
            dst_sd[f"{dst_prefix}.{sub}.{kind}"] = src_sd[f"blocks.{src_idx}.{sub}.{kind}"].clone().float()
    dst_sd[f"{dst_prefix}.edge_proj.weight"].zero_()
    dst_sd[f"{dst_prefix}.edge_proj.bias"].zero_()


def _apply_bilin_and_node_init(
    sd: dict[str, torch.Tensor],
    gen: torch.Generator,
    bilin_std: float = 0.05,
    node_std: float = 0.02,
    zero_bilin_w: bool = True,
) -> None:
    for k in ("bilin_u.weight", "bilin_v.weight"):
        sd[k] = torch.empty_like(sd[k]).normal_(0.0, bilin_std, generator=gen)
    sd["bilin_u.bias"].zero_()
    sd["bilin_v.bias"].zero_()
    if zero_bilin_w:
        sd["bilin_w.weight"].zero_()
        sd["bilin_w.bias"].zero_()
    if node_std > 0.0:
        for k in ("node_sel.weight", "node_thr.weight"):
            sd[k] = torch.empty_like(sd[k]).normal_(0.0, node_std, generator=gen)
        sd["node_sel.bias"].zero_()
        sd["node_thr.bias"].zero_()


def surgery_v7_to_v8(
    donor_path: Path,
    out_path: Path,
    *,
    n_heads: int = 4,
    n_enc_layers: int | None = None,
    target_alpha: float = 0.25,
    zero_delta_w: bool = False,
    seed: int = 2026,
) -> dict[str, Any]:
    """Graft a pre-trained `SingleEncV7` checkpoint into `SingleEncV8`."""
    src_sd = st_load(str(donor_path), device="cpu")
    d_model, n_v7_layers = _infer_v7_shape(src_sd)
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
    enc_layers = n_v7_layers if n_enc_layers is None else int(n_enc_layers)

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 17)
    model = SingleEncV8(
        d_model=d_model,
        n_heads=n_heads,
        n_enc_layers=enc_layers,
        n_dec_layers=n_v7_layers,
        code_dim=DEC_CODE_DIM,
        with_owner_encoder=True,
    )
    dst_sd = {k: v.clone().float() for k, v in model.state_dict().items() if not k.startswith("aux_ev_head.")}

    # 1. Exact coordinate-remapped decoder input projections (zero-init new V8 slots)
    dst_sd["dec_col_in.weight"].zero_()
    for s_i, d_i in V7_TO_V8_COL_MAP:
        dst_sd["dec_col_in.weight"][:, d_i] = src_sd["col_in.weight"][:, s_i].float()
    dst_sd["dec_col_in.bias"] = src_sd["col_in.bias"].clone().float()

    dst_sd["dec_glob_in.weight"].zero_()
    for s_i, d_i in V7_TO_V8_GLOB_MAP:
        scale = float(V7_TO_V8_GLOB_SCALE.get(d_i, 1.0))
        dst_sd["dec_glob_in.weight"][:, d_i] = src_sd["glob_in.weight"][:, s_i].float() * scale
    dst_sd["dec_glob_in.bias"] = src_sd["glob_in.bias"].clone().float()

    # Tiny code injection preserves Step-0 V7 decoder representations while allowing encoder grads
    dst_sd["dec_code_in.weight"] = torch.empty_like(dst_sd["dec_code_in.weight"]).normal_(0.0, 1e-3, generator=gen)
    dst_sd["dec_code_in.bias"].zero_()

    # 2. Exact 1:1 Pre-LN Split-QKV Transformer block transfer (decoder + encoder backbone clone)
    for i in range(n_v7_layers):
        _copy_block_from_v7(dst_sd, src_sd, i, f"dec_blocks.{i}")
    dst_sd["dec_final_norm.weight"] = src_sd["final_norm.weight"].clone().float()
    dst_sd["dec_final_norm.bias"] = src_sd["final_norm.bias"].clone().float()

    for i in range(min(enc_layers, n_v7_layers)):
        _copy_block_from_v7(dst_sd, src_sd, i, f"enc_blocks.{i}")
    dst_sd["enc_final_norm.weight"] = src_sd["final_norm.weight"].clone().float()
    dst_sd["enc_final_norm.bias"] = src_sd["final_norm.bias"].clone().float()

    # 3. Transfer shipped V7 delta_W unless explicitly zeroed
    if not zero_delta_w and "delta_W.weight" in src_sd:
        dst_sd["delta_W.weight"] = src_sd["delta_W.weight"].clone().float()
        dst_sd["delta_W.bias"] = src_sd["delta_W.bias"].clone().float()

    # 4. Break dead bilinear cubic saddle point & open trust gate (Step-0 nonlinear delta = 0)
    _apply_bilin_and_node_init(dst_sd, gen, bilin_std=0.05, node_std=0.02, zero_bilin_w=True)
    dst_sd["gate_head.weight"].zero_()
    dst_sd["gate_head.bias"].zero_()
    dst_sd["gate_head.bias"][0] = _logit(target_alpha) + 2.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_sd = {k: v.contiguous().to(torch.float32) for k, v in sorted(dst_sd.items())}
    st_save(export_sd, str(out_path))
    return {
        "mode": "v7-to-v8",
        "donor": str(donor_path),
        "donor_sha256": _sha256_file(donor_path),
        "output": str(out_path),
        "output_sha256": _sha256_file(out_path),
        "d_model": d_model,
        "n_heads": n_heads,
        "n_enc_layers": enc_layers,
        "n_dec_layers": n_v7_layers,
        "zero_delta_w": bool(zero_delta_w),
        "target_alpha": float(target_alpha),
        "num_tensors": len(export_sd),
        "total_params": int(sum(v.numel() for v in export_sd.values())),
        "norms": {
            k: round(float(export_sd[k].norm().item()), 5)
            for k in ("dec_col_in.weight", "delta_W.weight", "bilin_u.weight", "bilin_v.weight", "bilin_w.weight", "gate_head.bias")
        },
    }


def surgery_repair_v8(
    donor_path: Path,
    out_path: Path,
    *,
    observed_alpha: float = 0.025,
    target_alpha: float = 0.25,
    gate_weight_damp: float = 0.10,
    force_bilin_reinit: bool = False,
    seed: int = 2026,
) -> dict[str, Any]:
    """Function-preserving repair on a trained V8 checkpoint: unstick `bilin` + reopen `alpha_table`."""
    src_sd = st_load(str(donor_path), device="cpu")
    d_model, n_enc, n_dec = _infer_v8_shape(src_sd)
    dst_sd = {k: v.clone().float() for k, v in src_sd.items() if not k.startswith("aux_ev_head.")}
    gen = torch.Generator().manual_seed(seed + 31)

    bilin_u_norm_before = float(dst_sd["bilin_u.weight"].norm().item())
    bilin_reinit = force_bilin_reinit or bilin_u_norm_before < 1e-6
    if bilin_reinit:
        _apply_bilin_and_node_init(dst_sd, gen, bilin_std=0.05, node_std=0.0, zero_bilin_w=True)

    # Function-preserving trust-gate reopening: damp negative gate_head.weight[0] and rescale node_leaf
    ratio = min(max(float(observed_alpha) / max(float(target_alpha), 1e-4), 0.01), 1.0)
    dst_sd["gate_head.weight"][0].mul_(float(gate_weight_damp))
    dst_sd["gate_head.bias"][0] = _logit(target_alpha) + 2.0
    dst_sd["node_leaf.weight"].mul_(ratio)
    dst_sd["node_leaf.bias"].mul_(ratio)
    if not bilin_reinit:
        dst_sd["bilin_w.weight"].mul_(ratio)
        dst_sd["bilin_w.bias"].mul_(ratio)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_sd = {k: v.contiguous().to(torch.float32) for k, v in sorted(dst_sd.items())}
    st_save(export_sd, str(out_path))
    return {
        "mode": "repair-v8",
        "donor": str(donor_path),
        "donor_sha256": _sha256_file(donor_path),
        "output": str(out_path),
        "output_sha256": _sha256_file(out_path),
        "d_model": d_model,
        "n_enc_layers": n_enc,
        "n_dec_layers": n_dec,
        "bilin_u_norm_before": round(bilin_u_norm_before, 6),
        "bilin_reinitialized": bool(bilin_reinit),
        "observed_alpha": float(observed_alpha),
        "target_alpha": float(target_alpha),
        "node_leaf_scale": round(ratio, 5),
        "num_tensors": len(export_sd),
        "total_params": int(sum(v.numel() for v in export_sd.values())),
        "norms": {
            k: round(float(export_sd[k].norm().item()), 5)
            for k in ("delta_W.weight", "node_leaf.weight", "bilin_u.weight", "bilin_v.weight", "bilin_w.weight", "gate_head.weight", "gate_head.bias")
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Function-preserving SingleEncV8 model surgery")
    ap.add_argument("--mode", choices=["v7-to-v8", "repair-v8"], required=True)
    ap.add_argument("--donor", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-enc-layers", type=int, default=None)
    ap.add_argument("--observed-alpha", type=float, default=0.025)
    ap.add_argument("--target-alpha", type=float, default=0.25)
    ap.add_argument("--gate-weight-damp", type=float, default=0.10)
    ap.add_argument("--zero-delta-w", action="store_true")
    ap.add_argument("--force-bilin-reinit", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    if a.mode == "v7-to-v8":
        info = surgery_v7_to_v8(
            a.donor, a.out, n_heads=a.n_heads, n_enc_layers=a.n_enc_layers,
            target_alpha=a.target_alpha, zero_delta_w=a.zero_delta_w, seed=a.seed,
        )
    else:
        info = surgery_repair_v8(
            a.donor, a.out, observed_alpha=a.observed_alpha, target_alpha=a.target_alpha,
            gate_weight_damp=a.gate_weight_damp, force_bilin_reinit=a.force_bilin_reinit, seed=a.seed,
        )
    if a.manifest is not None:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
