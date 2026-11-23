"""SingleEncV7 (V7.2): compact (< 2M params, < 8 MB F32) zero-raw-row <= 2 KiB packet residual model.

Consumes ONLY dequantized `dope.single-encoding/v2`-aligned sufficient statistics:
- 17-knot normal-score marginals,
- rank-12 Gaussian copula factor loadings,
- analytic linear/main target base W_base/b_base,
- V7.2 4-quartile Empirical-Bayes-shrunken nonlinear target residuals Q_nl[p, 4] (4 B/col int8),
runs a rank-<=2 split-QKV column Transformer with zero-initialized residual heads, and decodes
synthetic rows strictly row-locally via counter-seeded noise with zero batch statistics.
"""
from __future__ import annotations

import math
import struct
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_COLS = 48
KNOT_COUNT = 17
MAX_RANK = 12
MAX_CLASSES = 8
PACKET_BUDGET_BYTES = 2048
Z_GRID_NP = np.linspace(-2.5, 2.5, KNOT_COUNT, dtype=np.float32)
COL_FEAT_DIM_V1 = KNOT_COUNT + MAX_RANK + MAX_CLASSES + 5  # 42 base floats
Q_NL_BINS = 4
COL_FEAT_DIM = COL_FEAT_DIM_V1 + Q_NL_BINS  # 46 floats (42 base + 4 quartile nonlinear residuals)
GLOB_FEAT_DIM = 4 + MAX_CLASSES + 4  # 16 global floats


def _norm_ppf_grid(probs: np.ndarray) -> np.ndarray:
    """Fast Acklam rational approximation of standard normal inverse CDF in NumPy."""
    p = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    q = p - 0.5
    out = np.empty_like(p)
    mid = np.abs(q) <= 0.425
    if np.any(mid):
        qm = q[mid]
        r = 0.180625 - qm * qm
        num = ((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
                   + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
               + 133.14166789178437745) * r + 3.387132872796366608
        den = ((((((5226.4952788528545610 * r + 28729.085735721942674) * r + 39307.895800092710610) * r
                   + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.18700749205790830) * r
               + 42.313330701600911252) * r + 1.0
        out[mid] = qm * num / den
    tail = ~mid
    if np.any(tail):
        pt = p[tail]
        rt = np.where(pt < 0.5, pt, 1.0 - pt)
        s = np.sqrt(-np.log(rt))
        num = ((((((7.7454501427834140764e-4 * s + 0.022723844989269184583) * s + 0.24178072517745061177) * s
                   + 1.2704582524523683826) * s + 3.6478483247632046050) * s + 5.7694972214606914055) * s
               + 4.6303378461565452959) * s + 1.4234371107496835773
        den = ((((((1.0507500716444168432e-9 * s + 5.4759380849953449460e-4) * s + 0.015198666563616457197) * s
                   + 0.14810397642748007459) * s + 0.68976733498510000455) * s + 1.6763848301838038494) * s
               + 2.0531916266377588219) * s + 1.0
        val = num / den
        out[tail] = np.where(pt < 0.5, -val, val)
    return out.astype(np.float32)


def _quant_i8(arr: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.clip(np.rint(arr / max(scale, 1e-6)), -127, 127).astype(np.int8)
    return q, q.astype(np.float32) * np.float32(scale)


def _compute_q_nl_dq(
    X: np.ndarray,
    y_fit: np.ndarray,
    z_mat: np.ndarray,
    W_dq: np.ndarray,
    b_dq: np.ndarray,
    resid_sd: float,
    task_id: int,
    k_eff: int,
) -> np.ndarray:
    """Bit-exact NumPy mirror of Rust `fit_q_nl_i8`: 4-quartile EB-shrunken nonlinear target residuals."""
    n_fit, p_eff = X.shape
    out = np.zeros((p_eff, Q_NL_BINS), dtype=np.float32)
    if n_fit < 12 or p_eff == 0:
        return out
    if task_id == 0:
        pred = X @ W_dq[0] + b_dq[0]
        resid = np.clip((y_fit.astype(np.float64) - pred.astype(np.float64)) / max(float(resid_sd), 0.10), -4.0, 4.0)
    elif task_id == 1:
        z = np.clip((X @ W_dq[0] + b_dq[0]).astype(np.float64), -8.0, 8.0)
        pr = 1.0 / (1.0 + np.exp(-z))
        resid = ((y_fit > 0.5).astype(np.float64) - pr) * 2.0
    else:
        lg = (X @ W_dq[:k_eff].T + b_dq[:k_eff][None, :]).astype(np.float64)
        lg = np.clip(lg - lg.max(axis=1, keepdims=True), -20.0, 0.0)
        ex = np.exp(lg)
        pr = ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)
        yi = np.clip(np.rint(y_fit).astype(np.int64), 0, k_eff - 1)
        raw_s = 1.0 - pr[np.arange(n_fit), yi]
        resid = (raw_s - float(raw_s.mean())) * 2.5

    # Bin by normal-score quartiles: (-inf, -0.6745), [-0.6745, 0), [0, 0.6745), [0.6745, inf)
    b0 = z_mat < -0.6745
    b1 = (~b0) & (z_mat < 0.0)
    b2 = (z_mat >= 0.0) & (z_mat < 0.6745)
    b3 = z_mat >= 0.6745
    for bi, mask in enumerate((b0, b1, b2, b3)):
        sums = (resid[:, None] * mask).sum(axis=0)
        cnts = mask.sum(axis=0).astype(np.float64)
        eb = sums / (cnts + 8.0)
        q = np.clip(np.rint(eb / 0.015), -127, 127).astype(np.int8)
        out[:, bi] = q.astype(np.float32) * np.float32(0.015)
    return out


def extract_dse_packet_numpy(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    task_id: int,
    n_classes: int,
    max_bytes: int = PACKET_BUDGET_BYTES,
) -> dict[str, Any]:
    """Fit and quantize zero-raw-row sufficient statistics + Q_nl[p,4] strictly within `max_bytes`."""
    n_fit, p = X_fit.shape
    p_eff = min(p, MAX_COLS)
    X = np.nan_to_num(X_fit[:, :p_eff].astype(np.float32, copy=False), nan=0.0, posinf=6.0, neginf=-6.0)
    k_eff = 1 if task_id == 0 else (1 if task_id == 1 else max(3, min(int(n_classes), MAX_CLASSES)))
    o_dim = 1 if task_id in (0, 1) else k_eff

    # 1. Normal-score marginal quantiles (17 knots equally spaced in [-z_max, +z_max]).
    u_floor = min(0.20, 10.0 / max(n_fit + 1.0, 20.0))
    z_max = float(np.clip(_norm_ppf_grid(np.array([1.0 - u_floor]))[0], 1.2, 2.6))
    z_grid_col = np.linspace(-z_max, z_max, KNOT_COUNT, dtype=np.float64)
    u_clamped = 0.5 * (1.0 + np.array([math.erf(float(z) / math.sqrt(2.0)) for z in z_grid_col], dtype=np.float64))
    knots_raw = np.quantile(X, u_clamped, axis=0).T.astype(np.float32)
    col_span = np.maximum(knots_raw[:, -1] - knots_raw[:, 0], 1e-5)
    is_const = (col_span < 1e-4).astype(np.float32)
    sample_r = X[: min(n_fit, 256)]
    is_disc = np.array(
        [1.0 if len(np.unique(np.round(sample_r[:, j], 4))) <= 16 else 0.0 for j in range(p_eff)],
        dtype=np.float32,
    )

    # 2. Normal scores (continuous ranks + discrete SP1 conditional-mean disattenuation).
    z_mat = np.zeros((n_fit, p_eff), dtype=np.float32)
    for j in range(p_eff):
        if is_const[j] > 0.5:
            continue
        col = X[:, j]
        if is_disc[j] > 0.5:
            order_j = np.argsort(col, kind="mergesort")
            s_col = col[order_j]
            _, first_idx, cnts = np.unique(s_col, return_index=True, return_counts=True)
            mid_u = (first_idx + 0.5 * cnts) / float(n_fit)
            atom_z = _norm_ppf_grid(mid_u)
            z_col = np.empty(n_fit, dtype=np.float32)
            for fi, cn, az in zip(first_idx, cnts, atom_z):
                z_col[order_j[fi : fi + cn]] = az
            z_mat[:, j] = (z_col - float(np.mean(z_col))) / max(float(np.std(z_col)), 0.15)
        else:
            ranks_j = np.argsort(np.argsort(col, kind="mergesort")).astype(np.float32)
            z_mat[:, j] = _norm_ppf_grid((ranks_j + 0.5) / float(n_fit))

    # Budget-compliant layout selection (accounts for embedded scale card + Q_nl[p, 4]!).
    header_b = 24 + (8 if task_id == 0 else 0)
    target_b = o_dim * (8 + p_eff)
    rank_use, knot_level, has_q_nl = 0, 5, False
    for cand_k in (17, 9, 5):
        found = False
        for cand_q in (True, False):
            for cand_r in (12, 8, 6, 4, 2, 1, 0):
                est = header_b + target_b + p_eff * (11 + cand_k + cand_r + (4 if cand_q else 0))
                if est <= max_bytes:
                    rank_use, knot_level, has_q_nl = cand_r, cand_k, cand_q
                    found = True
                    break
            if found:
                break
        if found:
            break

    # Quantize knots to wire precision (f16 endpoints + 8-bit relative codes) and dequantize.
    lo_f16 = knots_raw[:, 0].astype(np.float16).astype(np.float32)
    hi_f16 = knots_raw[:, -1].astype(np.float16).astype(np.float32)
    span_f16 = np.maximum(hi_f16 - lo_f16, 1e-5)
    rel = np.clip((knots_raw - lo_f16[:, None]) / span_f16[:, None], 0.0, 1.0)
    if knot_level == 5:
        idx_k = np.array([0, 4, 8, 12, 16])
        q8 = np.rint(rel[:, idx_k] * 255.0) / 255.0
        rel_dq = np.stack([np.interp(np.arange(KNOT_COUNT), idx_k, q8[j]) for j in range(p_eff)], axis=0)
    elif knot_level == 9:
        idx_k = np.arange(0, 17, 2)
        q8 = np.rint(rel[:, idx_k] * 255.0) / 255.0
        rel_dq = np.stack([np.interp(np.arange(KNOT_COUNT), idx_k, q8[j]) for j in range(p_eff)], axis=0)
    else:
        rel_dq = np.rint(rel * 255.0) / 255.0
    knots_dq = (lo_f16[:, None] + span_f16[:, None] * rel_dq.astype(np.float32)).astype(np.float32)
    for j in range(p_eff):
        if is_disc[j] > 0.5 and is_const[j] < 0.5:
            vals, cnts = np.unique(X[:, j], return_counts=True)
            kept = vals[cnts >= 10]
            if len(kept) >= 2:
                nearest = np.argmin(np.abs(knots_dq[j, :, None] - kept[None, :]), axis=1)
                knots_dq[j] = kept[nearest].astype(np.float32)
    knots_dq = np.maximum.accumulate(knots_dq, axis=1)

    # Factor loadings L: [p_eff, MAX_RANK] (int8 / 127.0)
    L_dq = np.zeros((p_eff, MAX_RANK), dtype=np.float32)
    class_means_dq = np.zeros((MAX_CLASSES, MAX_RANK), dtype=np.float32)
    if rank_use > 0 and p_eff >= 2 and n_fit >= 8:
        zc = z_mat - z_mat.mean(axis=0, keepdims=True)
        cov = (zc.T @ zc) / max(float(n_fit - 1), 1.0)
        evals, evecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(evals)[::-1][: min(rank_use, p_eff)]
        vals = np.maximum(evals[order], 0.0)
        raw_L = (evecs[:, order] * np.sqrt(vals)[None, :]).astype(np.float32)
        row_norm = np.linalg.norm(raw_L, axis=1, keepdims=True)
        raw_L = raw_L / np.maximum(row_norm / 0.99, 1.0)
        _, raw_L_dq = _quant_i8(raw_L, 1.0 / 127.0)
        L_dq[:, : raw_L_dq.shape[1]] = raw_L_dq

    # 3. Analytic target model base (S3: ridge / Newton-IRLS logistic / multiclass log-odds ridge).
    W_dq = np.zeros((MAX_CLASSES, p_eff), dtype=np.float32)
    b_dq = np.zeros((MAX_CLASSES,), dtype=np.float32)
    prior = np.zeros((MAX_CLASSES,), dtype=np.float32)
    resid_sd = np.float32(0.5)
    Xb = np.concatenate([X, np.ones((n_fit, 1), dtype=np.float32)], axis=1)
    gram = (Xb.T @ Xb).astype(np.float64)
    lam = max(4.0, 0.05 * float(n_fit))
    reg_mat = np.eye(p_eff + 1, dtype=np.float64) * lam
    reg_mat[-1, -1] = 1e-2

    if task_id == 0:
        yf = y_fit.astype(np.float64)
        sol = np.linalg.solve(gram + reg_mat, Xb.T @ yf)
        w0, b0 = sol[:p_eff].astype(np.float32), float(sol[-1])
        pred = X @ w0 + b0
        resid_sd = np.float32(np.clip(np.std(yf - pred), 0.05, 4.0))
        w_scale = float(np.max(np.abs(w0))) / 127.0 + 1e-6
        _, W_dq[0] = _quant_i8(w0, w_scale)
        b_dq[0] = np.float32(np.clip(b0, -20.0, 20.0))
        prior[0] = np.float32(np.clip(np.mean(yf), -6.0, 6.0))
    elif task_id == 1:
        yb = (y_fit > 0.5).astype(np.float64)
        p1 = float(np.clip(yb.mean(), 0.02, 0.98))
        prior[0], prior[1] = 1.0 - p1, p1
        wb = np.zeros(p_eff + 1, dtype=np.float64)
        wb[-1] = math.log(p1 / (1.0 - p1))
        reg_bin = np.eye(p_eff + 1, dtype=np.float64) * max(1.0, 0.005 * float(n_fit))
        reg_bin[-1, -1] = 1e-2
        for _ in range(5):
            logits = np.clip(Xb @ wb, -8.0, 8.0)
            pr = 1.0 / (1.0 + np.exp(-logits))
            wt = np.maximum(pr * (1.0 - pr), 0.02)
            H = (Xb.T * wt[None, :]) @ Xb + reg_bin
            g = Xb.T @ (pr - yb) + reg_bin @ wb
            wb -= np.linalg.solve(H, g)
        w0, b0 = np.clip(wb[:p_eff], -8.0, 8.0).astype(np.float32), float(np.clip(wb[-1], -8.0, 8.0))
        w_scale = float(np.max(np.abs(w0))) / 127.0 + 1e-6
        _, W_dq[0] = _quant_i8(w0, w_scale)
        b_dq[0] = np.float32(b0)
    else:
        yi = np.clip(np.rint(y_fit).astype(np.int64), 0, k_eff - 1)
        Y_oh = np.eye(k_eff, dtype=np.float64)[yi]
        cnts = np.maximum(Y_oh.sum(axis=0), 1.0)
        pr_c = cnts / cnts.sum()
        prior[:k_eff] = pr_c.astype(np.float32)
        Y_tgt = (Y_oh - pr_c[None, :]) * 3.0 + np.log(np.clip(pr_c, 1e-3, 1.0))[None, :]
        sol = np.linalg.solve(gram + reg_mat, Xb.T @ Y_tgt)
        W_mc = np.clip(sol[:p_eff].T, -6.0, 6.0).astype(np.float32)
        b_mc = np.clip(sol[-1], -6.0, 6.0).astype(np.float32)
        for c in range(k_eff):
            sc = float(np.max(np.abs(W_mc[c]))) / 127.0 + 1e-6
            _, W_dq[c] = _quant_i8(W_mc[c], sc)
            b_dq[c] = np.float32(b_mc[c])

    # 4. V7.2 4-quartile Empirical-Bayes nonlinear target residuals Q_nl[p_eff, 4] (4 B/col int8).
    q_nl_dq = (
        _compute_q_nl_dq(X, y_fit, z_mat, W_dq, b_dq, float(resid_sd), task_id, k_eff)
        if has_q_nl
        else np.zeros((p_eff, Q_NL_BINS), dtype=np.float32)
    )
    packet_bytes = header_b + target_b + p_eff * (11 + knot_level + rank_use + (4 if has_q_nl else 0))

    col_feats = np.zeros((MAX_COLS, COL_FEAT_DIM), dtype=np.float32)
    knots_pad = np.zeros((MAX_COLS, KNOT_COUNT), dtype=np.float32)
    L_pad = np.zeros((MAX_COLS, MAX_RANK), dtype=np.float32)
    W_pad = np.zeros((MAX_CLASSES, MAX_COLS), dtype=np.float32)
    col_mask = np.zeros((MAX_COLS,), dtype=np.float32)

    knots_pad[:p_eff] = knots_dq
    L_pad[:p_eff] = L_dq
    W_pad[:, :p_eff] = W_dq
    col_mask[:p_eff] = 1.0
    col_feats[:p_eff, 0:KNOT_COUNT] = knots_dq
    col_feats[:p_eff, KNOT_COUNT : KNOT_COUNT + MAX_RANK] = L_dq
    col_feats[:p_eff, KNOT_COUNT + MAX_RANK : KNOT_COUNT + MAX_RANK + MAX_CLASSES] = W_dq.T
    col_feats[:p_eff, COL_FEAT_DIM_V1 - 5] = is_const
    col_feats[:p_eff, COL_FEAT_DIM_V1 - 4] = is_disc
    col_feats[:p_eff, COL_FEAT_DIM_V1 - 3] = np.log1p(col_span)
    col_feats[:p_eff, COL_FEAT_DIM_V1 - 2] = np.float32(z_max)
    col_feats[:p_eff, COL_FEAT_DIM_V1 - 1] = float(knot_level) / float(KNOT_COUNT)
    col_feats[:p_eff, COL_FEAT_DIM_V1:COL_FEAT_DIM] = q_nl_dq

    glob_feats = np.zeros((GLOB_FEAT_DIM,), dtype=np.float32)
    glob_feats[min(task_id, 3)] = 1.0
    glob_feats[4 : 4 + MAX_CLASSES] = prior
    glob_feats[-4] = math.log2(max(n_fit, 2)) / 12.0
    glob_feats[-3] = float(p_eff) / float(MAX_COLS)
    glob_feats[-2] = float(resid_sd)
    glob_feats[-1] = float(k_eff) / float(MAX_CLASSES)

    return {
        "col_feats": col_feats,
        "glob_feats": glob_feats,
        "knots": knots_pad,
        "L": L_pad,
        "class_means": class_means_dq,
        "W": W_pad,
        "b": b_dq,
        "prior": prior,
        "resid_sd": np.float32(resid_sd),
        "col_mask": col_mask,
        "is_const": np.pad(is_const, (0, MAX_COLS - p_eff)),
        "p_eff": p_eff,
        "k_eff": k_eff,
        "task_id": task_id,
        "packet_bytes": int(packet_bytes),
        "rank_use": int(rank_use),
    }


def pack_v7_wire_header(p_eff: int, task_id: int, k_eff: int, rank_use: int, packet_bytes: int) -> bytes:
    """Compact 16-byte canonical header matching DSE7 container conventions."""
    return struct.pack("<4sBBBBII", b"DSE7", 2, task_id & 0xFF, k_eff & 0xFF, rank_use & 0xFF, p_eff, packet_bytes)


class SplitAttnBlock(nn.Module):
    """Pre-LN Transformer block with strictly split rank-<=2 q/k/v/o projections (Rust-ready)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, p, d = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(b, p, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, p, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, p, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        key_mask = (mask < 0.5).view(b, 1, 1, p)
        scores = scores.masked_fill(key_mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, p, d)
        x = x + self.o_proj(out)
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x * mask.unsqueeze(-1)


class SingleEncV7(nn.Module):
    """Packet-in, packet-residual-out model (< 2M params, < 8 MB F32, rank-<=2, row-local decode)."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, h_target: int = 24) -> None:
        super().__init__()
        self.d_model = d_model
        self.h_target = h_target
        self.col_in = nn.Linear(COL_FEAT_DIM, d_model)
        self.glob_in = nn.Linear(GLOB_FEAT_DIM, d_model)
        self.blocks = nn.ModuleList([SplitAttnBlock(d_model, n_heads, d_model * 2) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.delta_knots = nn.Linear(d_model, KNOT_COUNT)
        self.delta_L = nn.Linear(d_model, MAX_RANK)
        self.delta_W = nn.Linear(d_model, MAX_CLASSES)
        self.hyper_w1 = nn.Linear(d_model, h_target)
        self.hyper_b1 = nn.Linear(d_model, h_target)
        self.hyper_w2 = nn.Linear(d_model, h_target * MAX_CLASSES)
        self.hyper_temp = nn.Linear(d_model, 2)
        self._zero_init_residuals()

    def _zero_init_residuals(self) -> None:
        for mod in (self.delta_knots, self.delta_L, self.delta_W, self.hyper_w2, self.hyper_temp):
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)
        nn.init.normal_(self.hyper_w1.weight, std=0.05)
        nn.init.zeros_(self.hyper_w1.bias)
        nn.init.zeros_(self.hyper_b1.weight)
        nn.init.zeros_(self.hyper_b1.bias)

    def refine_packet(
        self, col_feats: torch.Tensor, glob_feats: torch.Tensor, col_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        g_tok = self.glob_in(glob_feats).unsqueeze(1)
        h = (self.col_in(col_feats) + g_tok) * col_mask.unsqueeze(-1)
        for blk in self.blocks:
            h = blk(h, col_mask)
        h = self.final_norm(h) * col_mask.unsqueeze(-1)
        denom = col_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        g_pool = h.sum(dim=1) / denom
        b = h.shape[0]
        return {
            "d_knots": 0.06 * torch.tanh(self.delta_knots(h)),
            "d_L": 0.10 * torch.tanh(self.delta_L(h)),
            "d_W": 0.50 * torch.tanh(self.delta_W(h)).transpose(1, 2),
            "w1": 0.50 * torch.tanh(self.hyper_w1(h)),
            "b1": 0.50 * torch.tanh(self.hyper_b1(g_pool)),
            "w2": 0.50 * torch.tanh(self.hyper_w2(g_pool)).view(b, self.h_target, MAX_CLASSES),
            "temp_sd": 0.50 * torch.tanh(self.hyper_temp(g_pool)),
        }

    def sample_batch(
        self,
        batch: dict[str, torch.Tensor],
        n_syn: int,
        z_factor: torch.Tensor,
        z_noise: torch.Tensor,
        y_noise: torch.Tensor,
        ref: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strictly row-local vectorized decode: `(packet, counter_noise) -> (X_syn, y_logits, y_syn)`."""
        col_mask = batch["col_mask"]
        is_const = batch["is_const"]
        ref = ref if ref is not None else self.refine_packet(batch["col_feats"], batch["glob_feats"], col_mask)

        L_raw = batch["L"] + ref["d_L"]
        L_norm = torch.linalg.norm(L_raw, dim=-1, keepdim=True)
        L_eff = L_raw / torch.clamp(L_norm / 0.99, min=1.0)
        psi = torch.clamp(1.0 - (L_eff * L_eff).sum(dim=-1), min=0.02)
        z_lat = torch.matmul(z_factor, L_eff.transpose(1, 2)) + z_noise * torch.sqrt(psi).unsqueeze(1)

        knots_base = batch["knots"]
        d_k = ref["d_knots"]
        base_diffs = torch.clamp(knots_base[:, :, 1:] - knots_base[:, :, :-1], min=0.0)
        w_raw = base_diffs * torch.exp(0.25 * torch.tanh(d_k[:, :, 1:]))
        span_b = base_diffs.sum(dim=-1, keepdim=True)
        w_sum = w_raw.sum(dim=-1, keepdim=True).clamp_min(1e-7)
        w_norm = torch.where(span_b > 1e-7, w_raw * (span_b / w_sum), base_diffs)
        k0 = knots_base[:, :, 0:1]
        knots_eff = torch.cat([k0, k0 + torch.cumsum(w_norm, dim=-1)], dim=-1)

        # Notice COL_FEAT_DIM_V1 - 2 is z_max and COL_FEAT_DIM_V1 - 4 is is_disc (matches Rust exactly!).
        z_max_t = batch["col_feats"][:, 0:1, COL_FEAT_DIM_V1 - 2].clamp(1.2, 2.6).unsqueeze(-1)
        dz = (2.0 * z_max_t) / float(KNOT_COUNT - 1)
        pos = (z_lat.clamp(-z_max_t, z_max_t) + z_max_t) / dz
        i0 = pos.floor().long().clamp(0, KNOT_COUNT - 2)
        frac = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)
        k_t = knots_eff.transpose(1, 2)
        v0 = torch.gather(k_t, 1, i0)
        v1 = torch.gather(k_t, 1, i0 + 1)
        x_cont = v0 + frac * (v1 - v0)
        is_disc = (batch["col_feats"][:, :, COL_FEAT_DIM_V1 - 4] > 0.5).to(x_cont.dtype).unsqueeze(1)
        slope_lo = 0.25 * ((knots_eff[:, :, 1] - knots_eff[:, :, 0]).unsqueeze(1)) / dz
        slope_hi = 0.25 * ((knots_eff[:, :, -1] - knots_eff[:, :, -2]).unsqueeze(1)) / dz
        x_cont = x_cont + (1.0 - is_disc) * (
            slope_lo * torch.clamp(z_lat + z_max_t, max=0.0)
            + slope_hi * torch.clamp(z_lat - z_max_t, min=0.0)
        )
        k_disc_t = knots_base.transpose(1, 2)
        dv0 = torch.gather(k_disc_t, 1, i0)
        dv1 = torch.gather(k_disc_t, 1, i0 + 1)
        x_disc_hard = torch.where(frac < 0.5, dv0, dv1)
        x_disc = x_cont + (x_disc_hard - x_cont).detach()
        x_syn = torch.where(is_disc > 0.5, x_disc, x_cont)
        x_syn = x_syn.clamp(-6.0, 6.0) * (1.0 - is_const).unsqueeze(1) * col_mask.unsqueeze(1)

        logits, resid_sd_eff = self.predict_target_on_x(batch, ref, x_syn)
        y_reg = logits[:, :, 0] + resid_sd_eff * y_noise
        return x_syn, logits, y_reg

    def predict_target_on_x(
        self, batch: dict[str, torch.Tensor], ref: dict[str, torch.Tensor], x_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        W_eff = (batch["W"] + ref["d_W"]) * batch["col_mask"].unsqueeze(1)
        lin_logits = torch.matmul(x_in, W_eff.transpose(1, 2)) + batch["b"].unsqueeze(1)
        hid = F.silu(torch.matmul(x_in, ref["w1"]) + ref["b1"].unsqueeze(1))
        nl_res = torch.matmul(hid, ref["w2"])
        temp_scale = torch.exp(ref["temp_sd"][:, 0]).view(-1, 1, 1)
        sd_scale = torch.exp(ref["temp_sd"][:, 1]).view(-1, 1)
        return (lin_logits + nl_res) * temp_scale, batch["resid_sd"].unsqueeze(1) * sd_scale

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
