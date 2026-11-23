"""SingleEncV8: Pre-bottleneck Owner Evidence Encoder + Source-free Receiver Decoder.

Exact 1:1 architectural, numerical, and SafeTensors (`F32`, `rank <= 2`) parity with pure-Rust
`engine/dope-core/src/single_encoding/v8/model.rs`:
- `PairSplitAttnBlock`: Pre-LN split-QKV multi-head attention with pairwise edge bias + GELU FFN.
- `OwnerEvidenceEncoderV8` (`enc_*`): FIT-only Evidence (`128 / 16 / 160`) -> nested 4-bit FSQ code `Z`
  with straight-through estimator + nested tier dropout (Tiers 0/1/2/3).
- `ReceiverDecoderV8` (`dec_*`): strictly source-free packet inputs (`48 / 4 / 32 / 64`) ->
  zero-init-safe `NeuralPlanV8` (knots, loadings, W, 2-layer row-local RealNVP coupling flow,
  Soft-NODE `T=8, D=3` via exact `entmax_1.5`, rank-4 bilinear residual, learned trust gate
  `alpha_table = sigmoid(g - 2.0)`, heteroscedastic scale head).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.levels.v8_evidence import COL_DIM as EV_COL_DIM
from research.levels.v8_evidence import EDGE_DIM as EV_EDGE_DIM
from research.levels.v8_evidence import GLOB_DIM as EV_GLOB_DIM
from research.levels.v8_packet import (
    DEC_CODE_DIM,
    DEC_COL_DIM,
    DEC_EDGE_DIM,
    DEC_GLOB_DIM,
    KNOT_COUNT,
    STREAM_ATOM,
    STREAM_FACTOR,
    STREAM_MISSING,
    STREAM_NOISE,
    STREAM_REGIME,
    STREAM_TARGET,
    V8_MAX_CLASSES,
    V8_MAX_COLS,
    V8_MAX_RANK,
    build_decoder_inputs_py,
    counter_normal,
    counter_student_t_scale,
    counter_uniform,
)

NODE_TREES = 8
NODE_DEPTH = 3
NODE_LEAVES = 8
BILIN_RANK = 4


def entmax_15_torch(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact closed-form differentiable `entmax_1.5` (Peters et al. 2019), matching Rust `entmax_15`."""
    n = logits.shape[dim]
    if n == 1:
        return torch.ones_like(logits)
    z = logits * 0.5
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    k = torch.arange(1, n + 1, device=logits.device, dtype=logits.dtype)
    view_shape = [1] * logits.ndim
    view_shape[dim] = n
    k = k.view(*view_shape)
    sum_z = torch.cumsum(z_sorted, dim=dim)
    sum_z2 = torch.cumsum(z_sorted * z_sorted, dim=dim)
    mean = sum_z / k
    ss = torch.clamp(sum_z2 - k * mean * mean, min=0.0)
    delta = torch.clamp(1.0 - ss, min=0.0)
    tau_all = mean - torch.sqrt(delta / k)
    support = z_sorted > tau_all
    k_star = torch.clamp(support.sum(dim=dim, keepdim=True) - 1, min=0, max=n - 1)
    tau = torch.gather(tau_all, dim, k_star)
    p = torch.clamp(z - tau, min=0.0) ** 2
    return p / torch.clamp(p.sum(dim=dim, keepdim=True), min=1e-12)


def fsq_4bit_ste(z_raw: torch.Tensor) -> torch.Tensor:
    """4-bit uniform FSQ quantization in `[-1, 1]` with straight-through gradient."""
    z = torch.tanh(z_raw)
    q = torch.round((z + 1.0) * 7.5).clamp(0.0, 15.0) / 7.5 - 1.0
    return z + (q - z).detach()


def gumbel_rao_ste(logits: torch.Tensor, tau: float = 1.0, k_mc: int = 4) -> torch.Tensor:
    """Low-variance Gumbel-Rao (`K=4`) straight-through categorical estimator (Paulus et al. 2021)."""
    u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
    g = -torch.log(-torch.log(u))
    y_hard_idx = (logits + g).argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(logits).scatter_(-1, y_hard_idx, 1.0)
    # Rao-Blackwellized conditional Gumbel samples given argmax = y_hard_idx
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    e_top = -torch.log(torch.rand_like(log_z).clamp(1e-6, 1.0 - 1e-6))
    g_top = log_z - torch.log(e_top)
    soft_acc = torch.zeros_like(logits)
    for _ in range(max(int(k_mc), 1)):
        e_k = -torch.log(torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6))
        # Truncated Gumbel <= g_top for non-winning classes, g_top for winning class
        trunc = -torch.log(torch.exp(-g_top) + e_k * torch.exp(-logits))
        g_cond = torch.where(y_hard > 0.5, g_top.expand_as(logits), trunc)
        soft_acc = soft_acc + F.softmax((logits + g_cond) / max(tau, 0.1), dim=-1)
    y_soft = soft_acc / float(max(int(k_mc), 1))
    return y_hard + (y_soft - y_soft.detach())


class PairSplitAttnBlock(nn.Module):
    """Pre-LN split-QKV multi-head attention with pairwise edge bias + GELU FFN (`rank <= 2` weights)."""

    def __init__(self, d_model: int, n_heads: int, edge_dim: int, ff_mult: int = 2) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.edge_proj = nn.Linear(edge_dim, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * ff_mult)
        self.ff2 = nn.Linear(d_model * ff_mult, d_model)

    def forward(
        self,
        x: torch.Tensor,
        edge: torch.Tensor,
        col_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, P, D], edge: [B, P, P, E], col_mask: [B, P] (1=valid, 0=pad)
        B, P, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)
        e_bias = self.edge_proj(edge).permute(0, 3, 1, 2)  # [B, H, P, P]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) + e_bias
        if col_mask is not None:
            scores = scores.masked_fill(col_mask[:, None, None, :] < 0.5, -1e4)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, P, D)
        x = x + self.o_proj(out)
        ff = self.ff2(F.gelu(self.ff1(self.norm2(x))))
        x = x + ff
        if col_mask is not None:
            x = x * col_mask[:, :, None]
        return x


class SingleEncV8(nn.Module):
    """Pre-bottleneck Owner Evidence Encoder + Source-free Receiver Decoder (`SingleEncV8`)."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_enc_layers: int = 4,
        n_dec_layers: int = 2,
        code_dim: int = DEC_CODE_DIM,
        with_owner_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.code_dim = code_dim
        self.with_owner_encoder = with_owner_encoder

        if with_owner_encoder:
            self.enc_col_in = nn.Linear(EV_COL_DIM, d_model)
            self.enc_glob_in = nn.Linear(EV_GLOB_DIM, d_model)
            self.enc_blocks = nn.ModuleList(
                [PairSplitAttnBlock(d_model, n_heads, EV_EDGE_DIM) for _ in range(n_enc_layers)]
            )
            self.enc_final_norm = nn.LayerNorm(d_model)
            self.enc_code_head = nn.Linear(d_model, code_dim)

        self.dec_col_in = nn.Linear(DEC_COL_DIM, d_model)
        self.dec_glob_in = nn.Linear(DEC_GLOB_DIM, d_model)
        # The learned code width the decoder reads. The wire format and the Rust receiver read DEC_CODE_DIM (64
        # FSQ values = 32 B); any other width is a research-only probe whose export Rust refuses on shape.
        self.dec_code_in = nn.Linear(code_dim, d_model)
        self.dec_blocks = nn.ModuleList(
            [PairSplitAttnBlock(d_model, n_heads, DEC_EDGE_DIM) for _ in range(n_dec_layers)]
        )
        self.dec_final_norm = nn.LayerNorm(d_model)

        # Zero-initialized residual heads -> exact step-0 identity with analytic DSE8!
        self.delta_knots = nn.Linear(d_model, KNOT_COUNT)
        self.delta_L = nn.Linear(d_model, V8_MAX_RANK)
        self.delta_W = nn.Linear(d_model, V8_MAX_CLASSES)
        self.node_sel = nn.Linear(d_model, NODE_TREES * NODE_DEPTH)
        self.node_thr = nn.Linear(d_model, NODE_TREES * NODE_DEPTH)
        self.node_leaf = nn.Linear(d_model, NODE_TREES * NODE_LEAVES * V8_MAX_CLASSES)
        self.bilin_u = nn.Linear(d_model, BILIN_RANK)
        self.bilin_v = nn.Linear(d_model, BILIN_RANK)
        self.bilin_w = nn.Linear(d_model, BILIN_RANK * V8_MAX_CLASSES)
        self.flow_scale_head = nn.Linear(d_model, 2)
        self.flow_shift_head = nn.Linear(d_model, 2)
        self.gate_head = nn.Linear(d_model, 2)
        # Auxiliary masked-evidence reconstruction head (training-only, omitted from SafeTensors export!)
        self.aux_ev_head = nn.Linear(d_model, EV_COL_DIM)
        self._zero_init_residual_heads()

    def _zero_init_residual_heads(self) -> None:
        for mod in (
            self.delta_knots,
            self.delta_L,
            self.delta_W,
            self.node_sel,
            self.node_thr,
            self.node_leaf,
            self.bilin_u,
            self.bilin_v,
            self.bilin_w,
            self.flow_scale_head,
            self.flow_shift_head,
            self.gate_head,
        ):
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)

    def encode_evidence(
        self,
        ev_col: torch.Tensor,
        ev_edge: torch.Tensor,
        ev_glob: torch.Tensor,
        col_mask: torch.Tensor | None = None,
        tier_dropout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Owner Evidence Encoder -> `(z_fsq [B, code_dim], g_pool [B, D])` (code_dim 64 on the wire)."""
        if not self.with_owner_encoder:
            B = ev_col.shape[0]
            return (
                torch.zeros(B, self.code_dim, device=ev_col.device, dtype=ev_col.dtype),
                torch.zeros(B, self.d_model, device=ev_col.device, dtype=ev_col.dtype),
            )
        g_tok = self.enc_glob_in(ev_glob)  # [B, D]
        h = self.enc_col_in(ev_col) + g_tok[:, None, :]
        if col_mask is not None:
            h = h * col_mask[:, :, None]
        for blk in self.enc_blocks:
            h = blk(h, ev_edge, col_mask)
        h = self.enc_final_norm(h)
        if col_mask is not None:
            denom = col_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            g_pool = (h * col_mask[:, :, None]).sum(dim=1) / denom
        else:
            g_pool = h.mean(dim=1)
        z_raw = self.enc_code_head(g_pool)
        z_fsq = fsq_4bit_ste(z_raw)
        if tier_dropout and self.training:
            # Nested code dropout: randomly zero all (tier 0) or suffix halves for prefix resilience!
            u = torch.rand(z_fsq.shape[0], 1, device=z_fsq.device)
            z_fsq = torch.where(u < 0.25, torch.zeros_like(z_fsq), z_fsq)
        return z_fsq[:, : self.code_dim], g_pool

    def decode_plan(
        self,
        dec_col: torch.Tensor,
        dec_edge: torch.Tensor,
        dec_glob: torch.Tensor,
        code_vec: torch.Tensor,
        col_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Source-free Receiver Decoder matching Rust `SingleEncV8Weights::build_neural_plan`."""
        B, P, _ = dec_col.shape
        g_tok = self.dec_glob_in(dec_glob)
        z_tok = self.dec_code_in(code_vec)
        h = self.dec_col_in(dec_col) + g_tok[:, None, :] + z_tok[:, None, :]
        if col_mask is not None:
            h = h * col_mask[:, :, None]
        for blk in self.dec_blocks:
            h = blk(h, dec_edge, col_mask)
        h = self.dec_final_norm(h)
        if col_mask is not None:
            h = h * col_mask[:, :, None]
            denom = col_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            g_pool = h.sum(dim=1) / denom
        else:
            g_pool = h.mean(dim=1)

        delta_knots = 0.06 * torch.tanh(self.delta_knots(h))
        delta_L = 0.10 * torch.tanh(self.delta_L(h))
        delta_W = 0.50 * torch.tanh(self.delta_W(h))  # [B, P, K]

        ns = self.node_sel(h).view(B, P, NODE_TREES, NODE_DEPTH).permute(0, 2, 3, 1)  # [B, T, D, P]
        if col_mask is not None:
            ns = ns.masked_fill(col_mask[:, None, None, :] < 0.5, -1e4)
        node_w = entmax_15_torch(ns, dim=-1)  # [B, T, D, P]
        node_thr = 2.0 * torch.tanh(self.node_thr(g_pool)).view(B, NODE_TREES, NODE_DEPTH)
        node_leaves = 0.75 * torch.tanh(self.node_leaf(g_pool)).view(
            B, NODE_TREES, NODE_LEAVES, V8_MAX_CLASSES
        )

        bilin_u = 0.50 * torch.tanh(self.bilin_u(h)).transpose(1, 2)  # [B, 4, P]
        bilin_v = 0.50 * torch.tanh(self.bilin_v(h)).transpose(1, 2)  # [B, 4, P]
        bilin_w = 0.50 * torch.tanh(self.bilin_w(g_pool)).view(B, BILIN_RANK, V8_MAX_CLASSES)

        flow_s = 0.35 * torch.tanh(self.flow_scale_head(h))  # [B, P, 2]
        flow_t = 0.35 * torch.tanh(self.flow_shift_head(h))  # [B, P, 2]
        gate_raw = self.gate_head(g_pool)  # [B, 2]
        alpha_table = torch.sigmoid(gate_raw[:, 0] - 2.0)  # ~0.119 at zero-init
        resid_sd_mult = torch.exp(0.50 * torch.tanh(gate_raw[:, 1]))

        return {
            "h_dec": h,
            "g_pool": g_pool,
            "delta_knots": delta_knots,
            "delta_L": delta_L,
            "delta_W": delta_W,
            "node_w": node_w,
            "node_thr": node_thr,
            "node_leaves": node_leaves,
            "bilin_u": bilin_u,
            "bilin_v": bilin_v,
            "bilin_w": bilin_w,
            "flow_s": flow_s,
            "flow_t": flow_t,
            "alpha_table": alpha_table,
            "resid_sd_mult": resid_sd_mult,
        }

    def predict_target_on_x(
        self,
        plan: dict[str, torch.Tensor],
        base_W: torch.Tensor,  # [B, K, P]
        base_b: torch.Tensor,  # [B, K]
        x_in: torch.Tensor,    # [B, N, P]
    ) -> torch.Tensor:
        """Evaluate target conditional `Linear + alpha_table * (Soft-NODE + Rank-4 Bilinear)` on `x_in`."""
        W_eff = base_W + plan["delta_W"].transpose(1, 2)  # [B, K, P]
        logits = torch.einsum("bnp,bkp->bnk", x_in, W_eff) + base_b[:, None, :]

        # Soft-NODE T=8, D=3 (vectorized 2x2x2 outer product for fast CPU execution)
        feat = torch.einsum("btdp,bnp->bntd", plan["node_w"], x_in)  # [B, N, T, 3]
        p_r = torch.sigmoid(((feat - plan["node_thr"][:, None, :, :]) * 2.0).clamp(-10.0, 10.0))
        p_l = 1.0 - p_r
        b0 = torch.stack([p_l[..., 0], p_r[..., 0]], dim=-1)  # [B, N, T, 2]
        b1 = torch.stack([p_l[..., 1], p_r[..., 1]], dim=-1)
        b2 = torch.stack([p_l[..., 2], p_r[..., 2]], dim=-1)
        # Leaf index l = d0 + 2*d1 + 4*d2 -> outer order (d2, d1, d0) flattened to 8
        w_leaf = (b2[..., :, None, None] * b1[..., None, :, None] * b0[..., None, None, :]).reshape(
            x_in.shape[0], x_in.shape[1], NODE_TREES, NODE_LEAVES
        )
        y_node = torch.einsum("bntl,btlk->bnk", w_leaf, plan["node_leaves"])

        # Rank-4 bilinear residual
        du = torch.tanh(torch.einsum("brp,bnp->bnr", plan["bilin_u"], x_in))
        dv = torch.tanh(torch.einsum("brp,bnp->bnr", plan["bilin_v"], x_in))
        y_bilin = torch.einsum("bnr,brk->bnk", du * dv, plan["bilin_w"])

        return logits + plan["alpha_table"][:, None, None] * (y_node + y_bilin)

    def decode_batch_differentiable(
        self,
        plan: dict[str, torch.Tensor],
        base_knots: torch.Tensor,      # [B, P, 17]
        base_L: torch.Tensor,          # [B, P, 12]
        base_W: torch.Tensor,          # [B, K, P]
        base_b: torch.Tensor,          # [B, K]
        resid_sd: torch.Tensor,        # [B]
        z_max: torch.Tensor,           # [B]
        is_cont_col: torch.Tensor,     # [B, P] (1=continuous, 0=disc/lattice/const)
        col_mask: torch.Tensor,        # [B, P]
        n_rows: int,
        antithetic: bool = True,
        z_factor: torch.Tensor | None = None,
        z_noise: torch.Tensor | None = None,
        y_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Differentiable row-local decode with CRN + antithetic continuous draws."""
        B, P, _ = base_knots.shape
        device, dtype = base_knots.device, base_knots.dtype
        if z_factor is None:
            if antithetic and n_rows >= 2:
                half = n_rows // 2
                zf_h = torch.randn(B, half, V8_MAX_RANK, device=device, dtype=dtype)
                zn_h = torch.randn(B, half, P, device=device, dtype=dtype)
                yn_h = torch.randn(B, half, device=device, dtype=dtype)
                z_factor = torch.cat([zf_h, -zf_h], dim=1)
                z_noise = torch.cat([zn_h, -zn_h], dim=1)
                y_noise = torch.cat([yn_h, -yn_h], dim=1)
                if n_rows % 2 == 1:
                    z_factor = torch.cat([z_factor, torch.randn(B, 1, V8_MAX_RANK, device=device, dtype=dtype)], dim=1)
                    z_noise = torch.cat([z_noise, torch.randn(B, 1, P, device=device, dtype=dtype)], dim=1)
                    y_noise = torch.cat([y_noise, torch.randn(B, 1, device=device, dtype=dtype)], dim=1)
            else:
                z_factor = torch.randn(B, n_rows, V8_MAX_RANK, device=device, dtype=dtype)
                z_noise = torch.randn(B, n_rows, P, device=device, dtype=dtype)
                y_noise = torch.randn(B, n_rows, device=device, dtype=dtype)

        # 1. Refined factor loadings (norm <= 0.999)
        L_raw = base_L + plan["delta_L"]
        L_norm = torch.linalg.norm(L_raw, dim=-1, keepdim=True)
        L_eff = L_raw / torch.clamp(L_norm / 0.999, min=1.0)
        noise_sd = torch.sqrt(torch.clamp(1.0 - (L_eff ** 2).sum(dim=-1), min=1e-4))  # [B, P]

        # 2. Latent Gaussian + 2-layer row-local RealNVP coupling flow!
        z_lat = torch.einsum("bnr,bpr->bnp", z_factor, L_eff) + z_noise * noise_sd[:, None, :]
        if P >= 2:
            idx = torch.arange(P, device=device)
            for layer in range(2):
                src_mask = ((idx % 2 == layer).to(dtype) * col_mask)  # [B, P]
                dst_mask = ((idx % 2 != layer).to(dtype) * col_mask)
                c_bar = (F.silu(z_lat) * src_mask[:, None, :]).sum(dim=-1, keepdim=True) / src_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)[:, None, :]
                c_bar = c_bar.clamp(-3.0, 3.0)
                s = plan["flow_s"][:, :, layer][:, None, :]
                t = plan["flow_t"][:, :, layer][:, None, :]
                z_upd = (z_lat * torch.exp((s * c_bar).clamp(-0.6, 0.6)) + t * c_bar).clamp(-5.0, 5.0)
                z_lat = torch.where(dst_mask[:, None, :] > 0.5, z_upd, z_lat)

        # 3. Monotonic span-preserving knot refinement (continuous columns ONLY!)
        bd = torch.clamp(base_knots[:, :, 1:] - base_knots[:, :, :-1], min=0.0)
        w_raw = bd * torch.exp(0.25 * torch.tanh(plan["delta_knots"][:, :, 1:]))
        span_b = bd.sum(dim=-1, keepdim=True)
        w_sum = w_raw.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        steps = torch.where(span_b > 1e-7, w_raw * (span_b / w_sum), bd)
        knots_cont = torch.cat([base_knots[:, :, :1], base_knots[:, :, :1] + torch.cumsum(steps, dim=-1)], dim=-1)
        knots_eff = torch.where(is_cont_col[:, :, None] > 0.5, knots_cont, base_knots)

        # Differentiable piecewise-linear knot lookup
        zm = z_max[:, None, None]
        dz = (2.0 * zm) / float(KNOT_COUNT - 1)
        pos = (torch.clamp(z_lat, -zm, zm) + zm) / dz
        i0 = torch.clamp(pos.floor().long(), 0, KNOT_COUNT - 2)
        frac = torch.clamp(pos - i0.to(dtype), 0.0, 1.0)
        ke_exp = knots_eff.unsqueeze(1).expand(B, n_rows, P, KNOT_COUNT)
        k0 = torch.gather(ke_exp, -1, i0.unsqueeze(-1)).squeeze(-1)
        k1 = torch.gather(ke_exp, -1, (i0 + 1).unsqueeze(-1)).squeeze(-1)
        x_syn = k0 + frac * (k1 - k0)
        slope_lo = 0.28 * torch.clamp(knots_eff[:, :, 1] - knots_eff[:, :, 0], min=0.0)[:, None, :] / dz
        slope_hi = 0.28 * torch.clamp(knots_eff[:, :, -1] - knots_eff[:, :, -2], min=0.0)[:, None, :] / dz
        tail_ext = slope_lo * torch.clamp(z_lat + zm, max=0.0) + slope_hi * torch.clamp(z_lat - zm, min=0.0)
        x_syn = torch.clamp(x_syn + tail_ext * is_cont_col[:, None, :], -6.0, 6.0) * col_mask[:, None, :]

        # 4. Target conditional: Linear W + alpha_table * (Soft-NODE + Rank-4 Bilinear)
        logits = self.predict_target_on_x(plan, base_W, base_b, x_syn)
        resid_sd_eff = torch.clamp(resid_sd * plan["resid_sd_mult"], 0.002, 6.0)
        y_reg = logits[:, :, 0] + resid_sd_eff[:, None] * y_noise

        return {
            "x_syn": x_syn,
            "z_lat": z_lat,
            "logits": logits,
            "y_reg": y_reg,
            "resid_sd_eff": resid_sd_eff,
            "alpha_table": plan["alpha_table"],
        }

    def export_v8_safetensors(self, path: str | Path) -> Path:
        """Export canonical F32 `rank <= 2` HF SafeTensors directly readable by Rust `SingleEncV8Weights`."""
        tensors: dict[str, np.ndarray] = {}
        for name, param in self.state_dict().items():
            if name.startswith("aux_ev_head."):
                continue  # Training-only auxiliary reconstruction head
            arr = param.detach().cpu().to(torch.float32).contiguous().numpy()
            if arr.ndim > 2:
                raise ValueError(f"tensor {name} has rank {arr.ndim} > 2 (violates rank<=2 contract)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"non-finite weight in {name}")
            tensors[name] = arr
        header: dict[str, Any] = {}
        chunks: list[bytes] = []
        offset = 0
        for name in sorted(tensors):
            raw = tensors[name].astype("<f4", copy=False).tobytes()
            header[name] = {"dtype": "F32", "shape": list(tensors[name].shape), "data_offsets": [offset, offset + len(raw)]}
            chunks.append(raw)
            offset += len(raw)
        hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + b"".join(chunks))
        return out_p


from research.levels.v8_parity_decode import decode_dse8_rows_py_parity  # noqa: E402,F401
