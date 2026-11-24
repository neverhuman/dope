"""Core building blocks and SafeTensors export helpers for `SingleEncV8`."""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Low-variance log-domain Gumbel-Rao (`K=4`) straight-through categorical estimator."""
    # Shift by max logit and compute truncated Gumbel in log domain via logaddexp so
    # large-magnitude analytic logits (|logit| > 88.7 in FP32) never overflow exp(-logits).
    l_s = logits - logits.amax(dim=-1, keepdim=True).detach()
    u = torch.rand_like(l_s).clamp(1e-6, 1.0 - 1e-6)
    g = -torch.log(-torch.log(u))
    y_hard_idx = (l_s + g).argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(l_s).scatter_(-1, y_hard_idx, 1.0)
    log_z = torch.logsumexp(l_s, dim=-1, keepdim=True)
    e_top = -torch.log(torch.rand_like(log_z).clamp(1e-6, 1.0 - 1e-6))
    g_top = log_z - torch.log(e_top)
    soft_acc = torch.zeros_like(l_s)
    for _ in range(max(int(k_mc), 1)):
        e_k = -torch.log(torch.rand_like(l_s).clamp(1e-6, 1.0 - 1e-6))
        trunc = -torch.logaddexp(-g_top, torch.log(e_k) - l_s)
        g_cond = torch.where(y_hard > 0.5, g_top.expand_as(l_s), trunc)
        soft_acc = soft_acc + F.softmax((l_s + g_cond) / max(tau, 0.1), dim=-1)
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


def export_v8_safetensors_dict(
    state_dict: dict[str, torch.Tensor],
    path: str | Path,
    dtype: str = "f32",
    decoder_capability: str | None = "active-rank-v1",
    assert_zero_delta_w: bool = False,
) -> Path:
    """Export canonical F32 or BF16 `rank <= 2` HF SafeTensors readable by Rust `SingleEncV8Weights`."""
    dt = dtype.lower()
    if dt not in ("f32", "bf16"):
        raise ValueError(f"unsupported export dtype {dtype!r}; expected 'f32' or 'bf16'")
    st_dtype = "F32" if dt == "f32" else "BF16"
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for name, param in state_dict.items():
        if name.startswith("aux_ev_head."):
            continue  # Training-only auxiliary reconstruction head
        t_cpu = param.detach().cpu().to(torch.float32).contiguous()
        if t_cpu.ndim > 2:
            raise ValueError(f"tensor {name} has rank {t_cpu.ndim} > 2 (violates rank<=2 contract)")
        if not torch.isfinite(t_cpu).all():
            raise ValueError(f"non-finite weight in {name}")
        if assert_zero_delta_w and name.startswith("delta_W."):
            max_abs = float(t_cpu.abs().max().item()) if t_cpu.numel() > 0 else 0.0
            if max_abs != 0.0:
                raise ValueError(f"non-zero {name} (max_abs={max_abs}) under assert_zero_delta_w")
        shape = list(t_cpu.shape)
        if dt == "f32":
            raw = t_cpu.numpy().astype("<f4", copy=False).tobytes()
        else:
            bf = t_cpu.to(torch.bfloat16).contiguous()
            raw = bf.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
        tensors[name] = (st_dtype, shape, raw)
    header: dict[str, Any] = {}
    if decoder_capability is not None:
        cap = str(decoder_capability)
        header["__metadata__"] = {"dope_dse8_decoder": cap}
        if cap == "active-rank-v1":
            # 1-byte U8 guard tensor refused by v3.0.0 F32-only SafeTensors loaders.
            tensors["__dope_dse8_decoder_active_rank_v1__"] = ("U8", [1], b"\x01")
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        t_dtype, shape, raw = tensors[name]
        header[name] = {"dtype": t_dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_bytes(struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + b"".join(chunks))
    return out_p
