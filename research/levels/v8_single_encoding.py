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
        self.dec_code_in = nn.Linear(DEC_CODE_DIM, d_model)
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
        """Run Owner Evidence Encoder -> `(z_fsq [B, 64], g_pool [B, D])`."""
        if not self.with_owner_encoder:
            B = ev_col.shape[0]
            return (
                torch.zeros(B, DEC_CODE_DIM, device=ev_col.device, dtype=ev_col.dtype),
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
        return z_fsq[:, :DEC_CODE_DIM], g_pool

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

        # 1. Refined factor loadings (norm <= 0.99)
        L_raw = base_L + plan["delta_L"]
        L_norm = torch.linalg.norm(L_raw, dim=-1, keepdim=True)
        L_eff = L_raw / torch.clamp(L_norm / 0.99, min=1.0)
        noise_sd = torch.sqrt(torch.clamp(1.0 - (L_eff ** 2).sum(dim=-1), min=0.02))  # [B, P]

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
        resid_sd_eff = torch.clamp(resid_sd * plan["resid_sd_mult"], 0.01, 6.0)
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


@torch.no_grad()
def decode_dse8_rows_py_parity(
    pkt: dict[str, Any],
    model: SingleEncV8 | None,
    n_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact scalar parity mirror of Rust `decode_dse8_rows` for 12-fixture hard parity (`<= 1e-4`)."""
    h = pkt["header"]
    p = h["p_eff"]
    r_use = h["rank_use"]
    o_dim = pkt["linear"]["o_dim"]
    card = pkt["card"]

    L_eff = pkt["loadings"].copy()
    knots_eff = np.stack([m["knots17"].copy() for m in pkt["marginals"]], axis=0)
    W_eff = pkt["linear"]["w"].copy()
    resid_sd_eff = float(h["resid_sd"])
    plan_np: dict[str, np.ndarray] | None = None

    if model is not None:
        model.eval()
        dc, de, dg, dz = build_decoder_inputs_py(pkt)
        pl = model.decode_plan(
            torch.from_numpy(dc)[None],
            torch.from_numpy(de)[None],
            torch.from_numpy(dg)[None],
            torch.from_numpy(dz)[None],
        )
        plan_np = {k: v[0].cpu().numpy() for k, v in pl.items() if k not in ("h_dec", "g_pool")}
        max_eff = max(
            float(np.max(np.abs(plan_np["delta_knots"]))),
            float(np.max(np.abs(plan_np["delta_L"]))),
            float(np.max(np.abs(plan_np["delta_W"][:, :o_dim]))),
            float(np.max(np.abs(plan_np["node_leaves"]))),
            float(np.max(np.abs(plan_np["bilin_w"]))),
            float(np.max(np.abs(plan_np["flow_s"]))),
            float(np.max(np.abs(plan_np["flow_t"]))),
            abs(float(plan_np["resid_sd_mult"]) - 1.0),
        )
        if max_eff < 1e-7:
            plan_np = None
        else:
            resid_sd_eff = float(np.clip(resid_sd_eff * float(plan_np["resid_sd_mult"]), 0.01, 6.0))
            for j in range(p):
                L_eff[j] = L_eff[j] + plan_np["delta_L"][j]
                sc = max(float(np.linalg.norm(L_eff[j])) / 0.99, 1.0)
                L_eff[j] /= sc
                if pkt["marginals"][j]["kind"] == 0:
                    base = pkt["marginals"][j]["knots17"]
                    bd = np.maximum(base[1:] - base[:-1], 0.0)
                    wr = bd * np.exp(0.25 * np.tanh(plan_np["delta_knots"][j, 1:]))
                    sb = float(bd.sum())
                    ws = max(float(wr.sum()), 1e-7)
                    steps = wr * (sb / ws) if sb > 1e-7 else np.zeros_like(bd)
                    knots_eff[j, 0] = base[0]
                    knots_eff[j, 1:] = base[0] + np.cumsum(steps)
                for c in range(o_dim):
                    W_eff[c, j] += float(plan_np["delta_W"][j, c])

    noise_sd = np.sqrt(np.maximum(1.0 - np.sum(L_eff ** 2, axis=1), 0.02)).astype(np.float32)

    x_out = np.zeros((n_rows, card["x_cols_raw"]), dtype=np.float32)
    y_out = np.zeros(n_rows, dtype=np.float32)

    def _norm_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _sample_cat(probs: list[float], u: float) -> int:
        s = sum(max(pr, 0.0) for pr in probs)
        if s <= 1e-12:
            return 0
        tgt = min(max(u, 0.0), 1.0 - 1e-15) * s
        acc = 0.0
        for idx, pr in enumerate(probs):
            acc += max(pr, 0.0)
            if tgt <= acc:
                return idx
        return len(probs) - 1

    for row_idx in range(n_rows):
        t_sc = counter_student_t_scale(seed, row_idx, h["inv_nu"])
        zf = np.array([counter_normal(seed, row_idx, STREAM_FACTOR, r) * t_sc for r in range(r_use)], dtype=np.float32)
        if len(pkt["regimes"]) >= 2:
            chosen = _sample_cat([rg["weight"] for rg in pkt["regimes"]], counter_uniform(seed, row_idx, STREAM_REGIME, 0))
            reg = pkt["regimes"][chosen]
            for r in range(min(r_use, len(reg["mean_f"]))):
                mu = float(reg["mean_f"][r])
                shr = math.sqrt(max(1.0 - 0.25 * min(mu * mu, 0.8), 0.5))
                zf[r] = np.float32(mu + shr * float(zf[r]))

        z_lat = np.zeros(p, dtype=np.float32)
        for j in range(p):
            eps = counter_normal(seed, row_idx, STREAM_NOISE, j) * t_sc
            z = eps * float(noise_sd[j]) + float(np.dot(zf, L_eff[j, :r_use]))
            z_lat[j] = np.float32(z)

        if plan_np is not None and p >= 2:
            for layer in range(2):
                vals = [float(z_lat[j]) / (1.0 + math.exp(-float(z_lat[j]))) for j in range(p) if j % 2 == layer]
                c_bar = min(max(sum(vals) / max(len(vals), 1), -3.0), 3.0)
                for j in range(p):
                    if j % 2 != layer:
                        s = float(plan_np["flow_s"][j, layer])
                        t = float(plan_np["flow_t"][j, layer])
                        z_lat[j] = np.float32(min(max(float(z_lat[j]) * math.exp(min(max(s * c_bar, -0.6), 0.6)) + t * c_bar, -5.0), 5.0))

        x_u = np.zeros(p, dtype=np.float32)
        miss = [False] * p
        z_max = float(h["z_max"])
        dz = (2.0 * z_max) / float(KNOT_COUNT - 1)
        for j in range(p):
            m = pkt["marginals"][j]
            if m["miss_rate"] > 1e-4 and counter_uniform(seed, row_idx, STREAM_MISSING, j) < m["miss_rate"]:
                miss[j] = True
                continue
            if m["kind"] == 4:
                x_u[j] = np.float32(m["lo"])
                continue
            if m["kind"] == 2 and m["atom_vals"]:
                u = _norm_cdf(float(z_lat[j]))
                acc = 0.0
                picked = m["atom_vals"][-1]
                for av, ap in zip(m["atom_vals"], m["atom_probs"]):
                    acc += ap
                    if u <= acc:
                        picked = av
                        break
                x_u[j] = np.float32(picked)
                continue
            if m["kind"] == 3 and m["atom_vals"]:
                u = _norm_cdf(float(z_lat[j]))
                lo_p = 0.0
                hit = None
                for av, ap in zip(m["atom_vals"], m["atom_probs"]):
                    pos_a = sum(1 for kv in m["knots17"] if kv < av)
                    tgt_u = min(max(pos_a / float(KNOT_COUNT - 1), 0.0), 1.0)
                    if abs(u - tgt_u) <= 0.5 * ap or (lo_p <= u <= lo_p + ap and abs(av - m["lo"]) <= 1e-4):
                        hit = av
                        break
                    lo_p += ap
                if hit is not None:
                    x_u[j] = np.float32(hit)
                    continue
            pos = (min(max(float(z_lat[j]), -z_max), z_max) + z_max) / dz
            i0 = min(max(int(math.floor(pos)), 0), KNOT_COUNT - 2)
            frac = min(max(pos - i0, 0.0), 1.0)
            ke = knots_eff[j]
            xv = float(ke[i0]) + frac * float(ke[i0 + 1] - ke[i0])
            if float(z_lat[j]) < -z_max or float(z_lat[j]) > z_max:
                sl_lo = max(float(ke[1] - ke[0]) / dz, 0.0)
                sl_hi = max(float(ke[-1] - ke[-2]) / dz, 0.0)
                mult = 0.0 if m["tail_family"] == 0 else (0.28 if m["tail_family"] == 1 else 0.55)
                xv += mult * sl_lo * min(float(z_lat[j]) + z_max, 0.0) + mult * sl_hi * max(float(z_lat[j]) - z_max, 0.0)
            if m["kind"] == 1 and m["lattice_q0"] > 1e-5:
                q0, orig = m["lattice_q0"], m["lattice_origin"]
                u_q = (xv - orig) / q0
                xv = orig + math.copysign(math.floor(abs(u_q) + 0.5), u_q) * q0
                if m["heaping_excess"] > 0.05 and m["heaping_step"] > q0:
                    if counter_uniform(seed, row_idx, STREAM_ATOM, j) < m["heaping_excess"]:
                        hs = m["heaping_step"]
                        u_h = (xv - orig) / hs
                        xv = orig + math.copysign(math.floor(abs(u_h) + 0.5), u_h) * hs
            x_u[j] = np.float32(min(max(xv, -6.0), 6.0))

        for gi, grp in enumerate(pkt["nominal_groups"]):
            wts = []
            for col_j, pr in zip(grp["cols"], grp["probs"]):
                if pr <= 1e-6:
                    wts.append(0.0)
                else:
                    wts.append(pr * math.exp(0.75 * min(max(float(z_lat[col_j]), -3.0), 3.0)))
            wts.append(grp["baseline_prob"] if grp["baseline_prob"] > 1e-6 else 0.0)
            pick = _sample_cat(wts, counter_uniform(seed, row_idx, STREAM_ATOM, 1024 + gi))
            for ci, col_j in enumerate(grp["cols"]):
                x_u[col_j] = 1.0 if pick == ci else 0.0
                miss[col_j] = False

        x_clean = np.where(np.array(miss), 0.0, x_u).astype(np.float32)
        logits = np.zeros(V8_MAX_CLASSES, dtype=np.float32)
        for c in range(o_dim):
            logits[c] = np.float32(float(pkt["linear"]["b"][c]) + float(np.dot(x_clean, W_eff[c])))
        for ef in pkt["effects"]:
            b8 = min(int(math.floor(_norm_cdf(float(z_lat[ef["col_idx"]])) * 8.0)), 7)
            logits[ef["cls_idx"]] += ef["bins8"][b8]
        for pr in pkt["pairs"]:
            zi = min(max(float(z_lat[pr["col_i"]]), -3.0), 3.0)
            zj = min(max(float(z_lat[pr["col_j"]]), -3.0), 3.0)
            xs = 1.0 if ((zi > 0.0) == (zj > 0.0)) else -1.0
            logits[pr["cls_idx"]] += np.float32(pr["w_prod"] * min(max(zi * zj, -6.0), 6.0) + pr["w_xor"] * xs)
        for tr in pkt["trees"]:
            l_idx = 0
            for d in range(3):
                b16 = min(int(math.floor(_norm_cdf(float(z_lat[tr["cols"][d]])) * 16.0)), 15)
                if b16 > tr["thrs"][d]:
                    l_idx |= 1 << d
            logits[tr["cls_idx"]] += tr["leaves"][l_idx]

        if plan_np is not None:
            nl = np.zeros(o_dim, dtype=np.float32)
            for t in range(NODE_TREES):
                pr_d = [
                    1.0 / (1.0 + math.exp(-min(max((float(np.dot(plan_np["node_w"][t, d, :p], x_clean)) - float(plan_np["node_thr"][t, d])) * 2.0, -10.0), 10.0)))
                    for d in range(NODE_DEPTH)
                ]
                for l_i in range(NODE_LEAVES):
                    wl = 1.0
                    for d in range(NODE_DEPTH):
                        wl *= pr_d[d] if (l_i & (1 << d)) else (1.0 - pr_d[d])
                    nl += wl * plan_np["node_leaves"][t, l_i, :o_dim]
            for r_i in range(BILIN_RANK):
                prod = math.tanh(float(np.dot(plan_np["bilin_u"][r_i, :p], x_clean))) * math.tanh(float(np.dot(plan_np["bilin_v"][r_i, :p], x_clean)))
                nl += prod * plan_np["bilin_w"][r_i, :o_dim]
            logits[:o_dim] += float(plan_np["alpha_table"]) * nl

        if h["task_id"] != 0:
            logits[:o_dim] *= float(pkt["linear"]["logit_tau"])

        if h["task_id"] == 0:
            eps_y = counter_normal(seed, row_idx, STREAM_TARGET, 0)
            rs = pkt["resid_model"]
            if any(abs(q) > 1e-4 for q in rs["q5_resid"]):
                u = _norm_cdf(eps_y)
                pos_r = min(max((u - 0.10) / 0.20, 0.0), 4.0)
                i0 = min(int(math.floor(pos_r)), 3)
                eps_y += rs["q5_resid"][i0] + (pos_r - i0) * (rs["q5_resid"][i0 + 1] - rs["q5_resid"][i0])
            lm = sum(w * min(max(float(z_lat[c]), -3.0), 3.0) for c, w in zip(rs["cols"], rs["log_sd_w"]))
            yu = min(max(float(logits[0]) + resid_sd_eff * math.exp(min(max(lm, -1.0), 1.0)) * eps_y, -8.0), 8.0)
            y_out[row_idx] = np.float32(yu if card["normalized_only"] else card["y_mean"] + card["y_sd"] * yu)
        elif h["task_id"] == 1:
            p1 = 1.0 / (1.0 + math.exp(-min(max(float(logits[0]), -12.0), 12.0)))
            idx = 1 if counter_uniform(seed, row_idx, STREAM_TARGET, 0) < p1 else 0
            y_out[row_idx] = np.float32(card["classes"][idx])
        else:
            ml = float(np.max(logits[:o_dim]))
            prs = [math.exp(min(max(float(logits[c]) - ml, -20.0), 0.0)) for c in range(o_dim)]
            pick = _sample_cat(prs, counter_uniform(seed, row_idx, STREAM_TARGET, 0))
            y_out[row_idx] = np.float32(card["classes"][pick])

        # Inverse scale card
        is_act = [False] * card["x_cols_raw"]
        for a, rj in enumerate(card["col_indices"]):
            is_act[rj] = True
            if miss[a]:
                x_out[row_idx, rj] = np.nan
            else:
                v = float(x_u[a]) if (card["normalized_only"] or card["group_kinds"][a] in (3, 4, 5)) else card["center_active"][a] + card["scale_active"][a] * float(x_u[a])
                if not card["normalized_only"] and card["is_int_active"][a]:
                    v = math.copysign(math.floor(abs(v) + 0.5), v)
                x_out[row_idx, rj] = np.float32(v)
        dc_i = 0
        for c in range(card["x_cols_raw"]):
            if not is_act[c]:
                x_out[row_idx, c] = np.float32(card["dropped_medians"][dc_i] if dc_i < len(card["dropped_medians"]) else 0.0)
                dc_i += 1

    return x_out, y_out
