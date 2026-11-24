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

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.levels.v8_evidence import COL_DIM as EV_COL_DIM
from research.levels.v8_evidence import EDGE_DIM as EV_EDGE_DIM
from research.levels.v8_evidence import GLOB_DIM as EV_GLOB_DIM
from research.levels.v8_model_blocks import (
    BILIN_RANK,
    NODE_DEPTH,
    NODE_LEAVES,
    NODE_TREES,
    PairSplitAttnBlock,
    entmax_15_torch,
    export_v8_safetensors_dict,
    fsq_4bit_ste,
    gumbel_rao_ste,
)
from research.levels.v8_packet import (
    DEC_CODE_DIM,
    DEC_COL_DIM,
    DEC_EDGE_DIM,
    DEC_GLOB_DIM,
    KNOT_COUNT,
    V8_MAX_CLASSES,
    V8_MAX_COLS,
    V8_MAX_RANK,
)


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
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        if code_dim < 64 or code_dim % 2 != 0:
            raise ValueError(
                f"code_dim must be an even integer >= 64 (native wire Tiers 1/2/3: 64/128/256), got {code_dim}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.code_dim = code_dim
        self.with_owner_encoder = with_owner_encoder
        self.ff_mult = ff_mult

        if with_owner_encoder:
            self.enc_col_in = nn.Linear(EV_COL_DIM, d_model)
            self.enc_glob_in = nn.Linear(EV_GLOB_DIM, d_model)
            self.enc_blocks = nn.ModuleList(
                [PairSplitAttnBlock(d_model, n_heads, EV_EDGE_DIM, ff_mult=ff_mult) for _ in range(n_enc_layers)]
            )
            self.enc_final_norm = nn.LayerNorm(d_model)
            self.enc_code_head = nn.Linear(d_model, code_dim)

        self.dec_col_in = nn.Linear(DEC_COL_DIM, d_model)
        self.dec_glob_in = nn.Linear(DEC_GLOB_DIM, d_model)
        # The learned code width the decoder reads. Native wire supports Tiers 1/2/3 (64/128/256 FSQ values =
        # 32/64/128 B); wider widths (e.g. 1024 = 512 B) are research-only probes refused by Rust on shape.
        self.dec_code_in = nn.Linear(code_dim, d_model)
        self.dec_blocks = nn.ModuleList(
            [PairSplitAttnBlock(d_model, n_heads, DEC_EDGE_DIM, ff_mult=ff_mult) for _ in range(n_dec_layers)]
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
            # Nested code dropout: randomly zero all (tier 0) or suffix tiers for prefix resilience!
            u = torch.rand(z_fsq.shape[0], 1, device=z_fsq.device)
            z_fsq = torch.where(u < 0.25, torch.zeros_like(z_fsq), z_fsq)
            if self.code_dim > 64:
                mask_t1 = torch.zeros(1, self.code_dim, device=z_fsq.device, dtype=z_fsq.dtype)
                mask_t1[:, :64] = 1.0
                z_fsq = torch.where((u >= 0.25) & (u < 0.50), z_fsq * mask_t1, z_fsq)
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
        *,
        z_in: torch.Tensor | None = None,
        base_knots: torch.Tensor | None = None,
        z_max: torch.Tensor | None = None,
        pkt_analytic: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Evaluate exact DSE8 target conditional matching Rust `decode_dse8_row_at`:
        `(Linear W + Analytic Effects/Pairs/Trees + alpha_table * (Soft-NODE + Rank-4 Bilinear)) * logit_tau`."""
        W_eff = base_W + plan["delta_W"].transpose(1, 2)  # [B, K, P]
        logits = torch.einsum("bnp,bkp->bnk", x_in, W_eff) + base_b[:, None, :]

        if pkt_analytic is not None:
            z_eval = z_in
            if z_eval is None and base_knots is not None and z_max is not None:
                z_eval = invert_knots_to_z_torch(x_in, base_knots, z_max)
            if z_eval is not None:
                logits = logits + eval_analytic_packet_logits_torch(z_eval, pkt_analytic)

        # Soft-NODE T=8, D=3 (vectorized 2x2x2 outer product for fast CPU/GPU execution)
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

        out = logits + plan["alpha_table"][:, None, None] * (y_node + y_bilin)
        if pkt_analytic is not None and int(pkt_analytic.get("task_id", 0)) != 0:
            tau = float(pkt_analytic.get("logit_tau", 1.0))
            if abs(tau - 1.0) > 1e-6:
                out = out * tau
        return out

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
        pkt_analytic: dict[str, Any] | None = None,
        active_rank_v1: bool = True,
        t_scale_override: torch.Tensor | None = None,
        regime_idx_override: torch.Tensor | None = None,
        u_missing_override: torch.Tensor | None = None,
        u_atom_override: torch.Tensor | None = None,
        u_nominal_override: torch.Tensor | None = None,
        u_target_override: torch.Tensor | None = None,
        u_regime_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Differentiable row-local decode with CRN + antithetic continuous draws & exact DSE8 parity."""
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

        # Match exact Rust Student-t scaling & multi-regime mixture when packet metadata is supplied.
        z_f_use = z_factor
        z_n_use = z_noise
        r_use = V8_MAX_RANK
        if pkt_analytic is not None:
            r_use = max(0, min(int(pkt_analytic.get("rank_use", V8_MAX_RANK)), V8_MAX_RANK))
            inv_nu = float(pkt_analytic.get("inv_nu", 0.0))
            if t_scale_override is not None:
                t_scale = t_scale_override.to(device=device, dtype=dtype)
                z_f_use = z_f_use * t_scale
                z_n_use = z_n_use * t_scale
            elif inv_nu > 1e-4:
                nu = min(max(1.0 / inv_nu, 2.2), 500.0)
                g_t = torch.randn(B, n_rows, 1, device=device, dtype=dtype)
                term = (1.0 - 2.0 / (9.0 * nu) + g_t * math.sqrt(2.0 / (9.0 * nu))).clamp(min=0.20)
                t_scale = (term.pow(3).rsqrt()).clamp(0.4, 3.5)
                z_f_use = z_f_use * t_scale
                z_n_use = z_n_use * t_scale
            reg_w = pkt_analytic.get("reg_weights")
            reg_m = pkt_analytic.get("reg_means")
            if reg_w is not None and reg_m is not None and r_use > 0:
                if regime_idx_override is not None:
                    chosen = regime_idx_override.to(device=device, dtype=torch.long)
                elif u_regime_override is not None:
                    chosen = categorical_f64_torch(
                        reg_w[None, None, :].expand(B, n_rows, -1), u_regime_override.to(device=device)
                    )
                else:
                    chosen = torch.multinomial(reg_w, B * n_rows, replacement=True).view(B, n_rows)
                mu_r = reg_m.to(dtype=dtype)[chosen]  # [B, N, 12]
                shr_r = (1.0 - (mu_r * mu_r).clamp(max=0.75)).clamp(min=0.25).sqrt()
                z_f_use = mu_r + shr_r * z_f_use
            if r_use < V8_MAX_RANK:
                rmask = (torch.arange(V8_MAX_RANK, device=device) < r_use).to(dtype)
                z_f_use = z_f_use * rmask[None, None, :]

        # 1. Refined factor loadings (norm <= 0.999); active-rank-v1 masks inactive ranks r >= r_use.
        L_raw = base_L + plan["delta_L"]
        if active_rank_v1 and r_use < V8_MAX_RANK:
            rmask = (torch.arange(V8_MAX_RANK, device=device) < r_use).to(dtype)
            L_raw = L_raw * rmask[None, None, :]
        L_norm = torch.linalg.norm(L_raw, dim=-1, keepdim=True)
        L_eff = L_raw / torch.clamp(L_norm / 0.999, min=1.0)
        noise_sd = torch.sqrt(torch.clamp(1.0 - (L_eff ** 2).sum(dim=-1), min=1e-4))  # [B, P]

        # 2. Latent Gaussian + 2-layer row-local RealNVP coupling flow!
        z_lat = torch.einsum("bnr,bpr->bnp", z_f_use, L_eff) + z_n_use * noise_sd[:, None, :]
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
        if pkt_analytic is not None and pkt_analytic.get("tail_mult") is not None:
            tm = pkt_analytic["tail_mult"].to(device=device, dtype=dtype)[None, None, :P]
        else:
            tm = 0.28 * is_cont_col[:, None, :]
        slope_lo = tm * torch.clamp(knots_eff[:, :, 1] - knots_eff[:, :, 0], min=0.0)[:, None, :] / dz
        slope_hi = tm * torch.clamp(knots_eff[:, :, -1] - knots_eff[:, :, -2], min=0.0)[:, None, :] / dz
        tail_ext = slope_lo * torch.clamp(z_lat + zm, max=0.0) + slope_hi * torch.clamp(z_lat - zm, min=0.0)
        x_knot = torch.clamp(x_syn + tail_ext, -6.0, 6.0) * col_mask[:, None, :]
        x_syn, miss_mask = apply_typed_marginals_ste_torch(
            x_knot, z_lat, pkt_analytic,
            u_missing=u_missing_override, u_atom=u_atom_override, u_nominal=u_nominal_override,
        )
        x_syn = x_syn * col_mask[:, None, :]

        # 4. Target conditional: exact DSE8 analytic (W + effects + pairs + trees) + neural residual!
        logits = self.predict_target_on_x(
            plan, base_W, base_b, x_syn, z_in=z_lat, pkt_analytic=pkt_analytic
        )
        resid_sd_eff = torch.clamp(resid_sd * plan["resid_sd_mult"], 0.002, 6.0)
        eps_y = y_noise
        sd_row = resid_sd_eff[:, None]
        if pkt_analytic is not None and int(pkt_analytic.get("task_id", 0)) == 0:
            q5 = pkt_analytic.get("rs_q5")
            if q5 is not None:
                u_y = _rust_norm_cdf_torch(eps_y).to(dtype=dtype)
                pos_r = ((u_y - 0.10) / 0.20).clamp(0.0, 4.0)
                i0_r = pos_r.floor().long().clamp(0, 3)
                f_r = pos_r - i0_r.to(dtype)
                q5d = q5.to(dtype=dtype)
                eps_y = eps_y + q5d[i0_r] + f_r * (q5d[i0_r + 1] - q5d[i0_r])
            rs_cols = pkt_analytic.get("rs_cols")
            rs_w = pkt_analytic.get("rs_log_sd_w")
            if rs_cols is not None and rs_w is not None:
                rc = rs_cols.clamp(0, max(P - 1, 0))
                lm = (z_lat[:, :, rc].clamp(-3.0, 3.0) * rs_w.to(dtype=dtype)[None, None, :]).sum(dim=-1)
                sd_row = sd_row * lm.clamp(-1.0, 1.0).exp()
        y_reg = logits[:, :, 0] + sd_row * eps_y
        if pkt_analytic is not None:
            y_reg = y_reg.clamp(-8.0, 8.0)
        task_id_an = int(pkt_analytic.get("task_id", 0)) if pkt_analytic is not None else 0
        o_dim_an = int(pkt_analytic.get("o_dim", 1)) if pkt_analytic is not None else 1
        if task_id_an != 0:
            ut = (
                u_target_override.to(device=device, dtype=torch.float64)
                if u_target_override is not None
                else torch.rand((B, n_rows), device=device, dtype=torch.float64)
            )
            y_sampled = sample_classes_torch(logits[:, :, :o_dim_an], task_id_an, ut).to(dtype=dtype)
        else:
            y_sampled = y_reg

        return {
            "x_syn": x_syn,
            "miss_mask": miss_mask,
            "z_lat": z_lat,
            "logits": logits,
            "y_reg": y_reg,
            "y_sampled": y_sampled,
            "resid_sd_eff": resid_sd_eff,
            "alpha_table": plan["alpha_table"],
        }

    def export_v8_safetensors(
        self,
        path: str | Path,
        dtype: str = "f32",
        decoder_capability: str | None = "active-rank-v1",
        assert_zero_delta_w: bool = False,
    ) -> Path:
        """Export canonical F32 or BF16 `rank <= 2` HF SafeTensors readable by Rust `SingleEncV8Weights`."""
        return export_v8_safetensors_dict(
            self.state_dict(),
            path,
            dtype=dtype,
            decoder_capability=decoder_capability,
            assert_zero_delta_w=assert_zero_delta_w,
        )


from research.levels.v8_parity_decode import (  # noqa: E402,F401
    _rust_norm_cdf_torch,
    apply_typed_marginals_ste_torch,
    categorical_f64_torch,
    decode_dse8_rows_py_parity,
    eval_analytic_packet_logits_torch,
    invert_knots_to_z_torch,
    sample_classes_torch,
)
