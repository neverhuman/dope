"""DOPE v2.0.0 True Zero-Raw-Row Single-Encoding Neural Anonymizer (DopeSingleEncV6).

Compresses real table + FIT-only 48-dim landmarks into a compact anonymized single encoding:
  Z = (z_global in R^{global_dim}, col_codes in R^{P x col_dim}) (~1.2-4 KiB in FP16)
and decodes synthetic (X_syn, y_syn) STRICTLY from Z (ZERO raw rows passed to decoder!).
"""
from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v6_landmarks import (
    LANDMARK_DIM,
    MAX_CLASSES_SUPPORT,
    N_KNOTS,
    TASK_IDS,
    _irls_downstream_eval,
    _normal_cdf,
    _normal_ppf,
    _rbf_mmd_loss,
    extract_rich_landmarks,
)

__all__ = [
    "DopeSingleEncV6",
    "LANDMARK_DIM",
    "MAX_CLASSES_SUPPORT",
    "N_KNOTS",
    "PreNormSetBlock",
    "TASK_IDS",
    "_irls_downstream_eval",
    "_normal_cdf",
    "_normal_ppf",
    "_rbf_mmd_loss",
    "compute_single_enc_loss",
    "extract_rich_landmarks",
]


class PreNormSetBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class DopeSingleEncV6(nn.Module):
    """Strict Zero-Raw-Row Single-Encoding Tabular Anonymizer & Multi-Task Generator."""

    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 8,
        global_dim: int = 128,
        col_dim: int = 24,
        n_regimes: int = 4,
        copula_rank: int = 12,
        target_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.global_dim = global_dim
        self.col_dim = col_dim
        self.n_regimes = n_regimes
        self.copula_rank = copula_rank
        self.target_hidden = target_hidden

        self.col_in = nn.Sequential(
            nn.Linear(LANDMARK_DIM, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.task_emb = nn.Embedding(4, d_model)
        self.blocks = nn.ModuleList([PreNormSetBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool_q = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.to_z_global = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, global_dim))
        self.to_col_code = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, col_dim))

        # === ZERO-RAW-ROW DECODER G(Z) ===
        z_in = col_dim + global_dim
        self.dec_regime_pi = nn.Linear(global_dim, n_regimes)
        self.dec_copula = nn.Sequential(
            nn.Linear(z_in, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, n_regimes * (copula_rank + 2)),
        )
        self.flow_ctx_proj = nn.Sequential(nn.Linear(z_in, 64), nn.GELU())
        self.flow_s1 = nn.Linear(65, 2)
        self.flow_s2 = nn.Linear(65, 2)
        self.dec_knots = nn.Sequential(nn.Linear(z_in, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, N_KNOTS))
        self.dec_miss = nn.Linear(z_in, 1)
        self.dec_driver = nn.Sequential(nn.Linear(z_in, 64), nn.GELU(), nn.Linear(64, 1))
        self.hyper_w1 = nn.Sequential(nn.Linear(z_in, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, target_hidden * 2))
        self.hyper_b1 = nn.Linear(global_dim, target_hidden)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_dim, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, target_hidden * MAX_CLASSES_SUPPORT + MAX_CLASSES_SUPPORT + 2),
        )
        self.hyper_noise = nn.Linear(global_dim, 2)

    def encoding_bytes_fp16(self, p: int) -> int:
        return (self.global_dim + p * self.col_dim + MAX_CLASSES_SUPPORT) * 2

    def encode_to_single_latent(
        self,
        X_fit: torch.Tensor,
        y_fit: torch.Tensor,
        miss_fit: torch.Tensor | None,
        task: str,
        n_classes: int,
    ) -> dict[str, torch.Tensor]:
        landmarks, priors, emp_qs = extract_rich_landmarks(X_fit, y_fit, miss_fit, task, n_classes)
        tid = torch.tensor(TASK_IDS.get(task, 0), device=X_fit.device, dtype=torch.long)
        h = self.col_in(landmarks).unsqueeze(0) + self.task_emb(tid).view(1, 1, -1)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)
        pooled, _ = self.pool_attn(self.pool_q, h, h, need_weights=False)
        z_global = self.to_z_global(pooled.reshape(1, -1)).squeeze(0)
        col_codes = self.to_col_code(h.squeeze(0))
        return {
            "z_global": z_global,
            "col_codes": col_codes,
            "priors": priors,
            "emp_qs": emp_qs,
            "task_latent": z_global[: self.global_dim // 2],
            "dist_latent": z_global[self.global_dim // 2 :],
        }

    def decode_from_single_latent(
        self,
        enc: dict[str, torch.Tensor],
        task: str,
        n_classes: int,
        n_syn: int,
        gen: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Generate (X_syn, y_syn) STRICTLY from single encoding Z (ZERO raw rows!)."""
        with torch.autocast(device_type=enc["z_global"].device.type, enabled=False):
            zg = enc["z_global"].float()
            cc = enc["col_codes"].float()
            priors = enc["priors"].float()
            p = cc.shape[0]
            dev = zg.device
            ctx = torch.cat([cc, zg.unsqueeze(0).expand(p, -1)], dim=-1)

            pi = F.softmax(self.dec_regime_pi(zg), dim=-1)
            cop_raw = self.dec_copula(ctx).view(p, self.n_regimes, self.copula_rank + 2)
            reg_mu = 1.5 * torch.tanh(cop_raw[:, :, 0])
            reg_sd = F.softplus(cop_raw[:, :, 1]).clamp(0.15, 2.0)
            reg_L = 0.7 * torch.tanh(cop_raw[:, :, 2:])

            g_u = torch.rand((n_syn, self.n_regimes), generator=gen, device=dev).clamp(1e-5, 1.0 - 1e-5)
            reg_w = F.softmax((torch.log(pi.clamp_min(1e-5)).unsqueeze(0) - torch.log(-torch.log(g_u))) / 0.5, dim=-1)
            eps_r = torch.randn((n_syn, self.copula_rank), generator=gen, device=dev)
            eps_p = torch.randn((n_syn, p), generator=gen, device=dev)
            z_lat = (reg_w @ reg_mu.T) + torch.einsum("nm,nr,pmr->np", reg_w, eps_r, reg_L) + (reg_w @ reg_sd.T) * eps_p

            f_ctx = self.flow_ctx_proj(ctx)
            mask_even = (torch.arange(p, device=dev) % 2 == 0).float().unsqueeze(0)
            for fhead, m in [(self.flow_s1, mask_even), (self.flow_s2, 1.0 - mask_even)]:
                agg = (z_lat * m).mean(dim=1, keepdim=True).expand(-1, p)
                inp = torch.cat([f_ctx.unsqueeze(0).expand(n_syn, -1, -1), agg.unsqueeze(-1)], dim=-1)
                st = fhead(inp)
                z_lat = z_lat * m + (z_lat * torch.exp(0.35 * torch.tanh(st[:, :, 0])) + 0.6 * torch.tanh(st[:, :, 1])) * (1.0 - m)

            z_norm = (z_lat - z_lat.mean(dim=0, keepdim=True)) / z_lat.std(dim=0, unbiased=False).clamp_min(1e-4)
            u_syn = _normal_cdf(z_norm).clamp(1e-5, 1.0 - 1e-5)

            k_raw = self.dec_knots(ctx)
            k0 = 6.0 * torch.tanh(k_raw[:, :1] * 0.5)
            knots = torch.cat([k0, k0 + torch.cumsum(F.softplus(k_raw[:, 1:]) * 0.5, dim=1)], dim=1).clamp(-6.0, 6.0)
            pos = u_syn * (N_KNOTS - 1)
            lo = pos.floor().long().clamp(0, N_KNOTS - 2)
            frac = pos - lo.float()
            cids = torch.arange(p, device=dev).unsqueeze(0).expand(n_syn, -1)
            X_syn = (knots[cids, lo] + frac * (knots[cids, lo + 1] - knots[cids, lo])).clamp(-6.0, 6.0)
            miss_prob = torch.sigmoid(self.dec_miss(ctx).squeeze(-1) - 2.0)
            driver_weights = torch.sigmoid(self.dec_driver(ctx).squeeze(-1))

            y_logits = self.predict_target_from_latent(X_syn, enc, task, n_classes)
            n_params = self.hyper_noise(zg)
            noise_sd = F.softplus(n_params[0]).clamp(0.03, 1.5)
            temp = (F.softplus(n_params[1]) + 0.5).clamp(0.5, 15.0)
            if task == "regression":
                y_syn = (y_logits.squeeze(-1) + torch.randn((n_syn,), generator=gen, device=dev) * noise_sd).clamp(-6.0, 6.0)
            elif task == "binary":
                y_syn = torch.sigmoid(y_logits.squeeze(-1) * temp).clamp(1e-4, 1.0 - 1e-4)
            elif task == "multilabel":
                y_syn = torch.sigmoid(y_logits * temp).clamp(1e-4, 1.0 - 1e-4)
            else:
                k = min(max(n_classes, 2), MAX_CLASSES_SUPPORT)
                p_log = torch.log(priors[:k].clamp_min(1e-4)).unsqueeze(0)
                y_syn = F.softmax(p_log + y_logits[:, :k] * temp, dim=-1).clamp(1e-4, 1.0)
            return {
                "X_syn": X_syn, "y_syn": y_syn, "y_logits": y_logits,
                "knots": knots, "miss_prob": miss_prob, "driver_weights": driver_weights,
            }

    def predict_target_from_latent(
        self, X: torch.Tensor, enc: dict[str, torch.Tensor], task: str, n_classes: int
    ) -> torch.Tensor:
        zg, cc = enc["z_global"].float(), enc["col_codes"].float()
        p = cc.shape[0]
        ctx = torch.cat([cc, zg.unsqueeze(0).expand(p, -1)], dim=-1)
        w1_all = self.hyper_w1(ctx)
        w1_lin = w1_all[:, : self.target_hidden] / math.sqrt(max(p, 1))
        w1_quad = w1_all[:, self.target_hidden :] / math.sqrt(max(p, 1))
        xf = X.float().clamp(-6.0, 6.0)
        h1 = F.gelu(xf @ w1_lin + 0.35 * (xf.pow(2).clamp_max(16.0) @ w1_quad) + self.hyper_b1(zg))
        w2_flat = self.hyper_w2(zg)
        k_out = 1 if task in ("regression", "binary") else min(max(n_classes, 2), MAX_CLASSES_SUPPORT)
        w2 = w2_flat[: self.target_hidden * MAX_CLASSES_SUPPORT].view(self.target_hidden, MAX_CLASSES_SUPPORT)[:, :k_out]
        b2 = w2_flat[self.target_hidden * MAX_CLASSES_SUPPORT : self.target_hidden * MAX_CLASSES_SUPPORT + MAX_CLASSES_SUPPORT][:k_out]
        return (h1 @ w2) / math.sqrt(self.target_hidden) + b2


def compute_single_enc_loss(
    model: DopeSingleEncV6, episode: dict[str, Any], device: torch.device, step_seed: int
) -> dict[str, Any]:
    """Compute smoothed Main-KPI + Direct Distillation + MMD/Quantile Realism + Driver loss."""
    X_fit = torch.as_tensor(episode["X_fit"], dtype=torch.float32, device=device)
    y_fit = torch.as_tensor(episode["y_fit"], dtype=torch.float32, device=device)
    miss_fit = torch.as_tensor(episode["miss_fit"], dtype=torch.float32, device=device)
    X_qry = torch.as_tensor(episode["X_qry"], dtype=torch.float32, device=device)
    y_qry = torch.as_tensor(episode["y_qry"], dtype=torch.float32, device=device)
    dmask = torch.as_tensor(episode["distractor_mask"], dtype=torch.bool, device=device)
    task, n_classes = str(episode["task"]), int(episode["n_classes"])

    enc = model.encode_to_single_latent(X_fit, y_fit, miss_fit, task, n_classes)
    gen = torch.Generator(device=device).manual_seed(step_seed & 0x7FFF_FFFF)
    syn = model.decode_from_single_latent(enc, task, n_classes, max(int(X_fit.shape[0]) * 2, 128), gen)
    X_syn, y_syn = syn["X_syn"], syn["y_syn"]

    l_syn1, l_null1, imp_syn1 = _irls_downstream_eval(X_syn, y_syn, X_qry, y_qry, task, n_classes)
    with torch.no_grad():
        l_real1, _, imp_real1 = _irls_downstream_eval(X_fit, y_fit, X_qry, y_qry, task, n_classes)

    p = X_fit.shape[1]
    rff_d = min(48, max(16, p * 2))
    rgen = torch.Generator(device=device).manual_seed((step_seed ^ 0x517C) & 0x7FFF_FFFF)
    W_rff = torch.randn((p, rff_d), generator=rgen, device=device) * (0.7 / math.sqrt(max(p, 1)))
    b_rff = torch.rand((1, rff_d), generator=rgen, device=device) * (2.0 * math.pi)
    l_syn2, l_null2, _ = _irls_downstream_eval(
        torch.cat([X_syn, torch.cos(X_syn @ W_rff + b_rff)], dim=1), y_syn,
        torch.cat([X_qry, torch.cos(X_qry @ W_rff + b_rff)], dim=1), y_qry, task, n_classes, alpha=0.4,
    )
    with torch.no_grad():
        l_real2, _, _ = _irls_downstream_eval(
            torch.cat([X_fit, torch.cos(X_fit @ W_rff + b_rff)], dim=1), y_fit,
            torch.cat([X_qry, torch.cos(X_qry @ W_rff + b_rff)], dim=1), y_qry, task, n_classes, alpha=0.4,
        )

    g1 = F.smooth_l1_loss(l_syn1, l_real1.detach(), beta=0.5) + 0.5 * F.relu(l_syn1 - l_real1.detach())
    g2 = F.smooth_l1_loss(l_syn2, l_real2.detach(), beta=0.5) + 0.5 * F.relu(l_syn2 - l_real2.detach())
    tstr_loss = (0.55 * g1 + 0.45 * g2).clamp_max(5.0)

    q_logits = model.predict_target_from_latent(X_qry, enc, task, n_classes)
    if task == "regression":
        distill_loss = F.smooth_l1_loss(q_logits.squeeze(-1), y_qry.float(), beta=1.0)
    elif task == "binary":
        distill_loss = F.binary_cross_entropy_with_logits(q_logits.squeeze(-1), y_qry.float().clamp(0.0, 1.0))
    elif task == "multilabel":
        ym = y_qry.float() if y_qry.ndim == 2 else y_qry.float().unsqueeze(1)
        distill_loss = F.binary_cross_entropy_with_logits(q_logits, ym.clamp(0.0, 1.0))
    else:
        k = min(max(n_classes, 2), MAX_CLASSES_SUPPORT)
        distill_loss = F.cross_entropy(torch.log(enc["priors"][:k].clamp_min(1e-4)).unsqueeze(0) + q_logits[:, :k], y_qry.long().clamp(0, k - 1))
    distill_loss = distill_loss.clamp_max(5.0)

    knot_loss = F.smooth_l1_loss(syn["knots"], enc["emp_qs"].detach(), beta=0.25)
    xc_f, xc_s = X_fit - X_fit.mean(dim=0, keepdim=True), X_syn - X_syn.mean(dim=0, keepdim=True)
    cov_loss = F.smooth_l1_loss((xc_s.T @ xc_s) / float(max(X_syn.shape[0], 1)), ((xc_f.T @ xc_f) / float(max(X_fit.shape[0], 1))).detach(), beta=0.25)
    mmd_loss = _rbf_mmd_loss(X_fit, X_syn)
    miss_loss = F.binary_cross_entropy(syn["miss_prob"].clamp(1e-4, 1.0 - 1e-4), miss_fit.mean(dim=0).clamp(0.0, 1.0))
    realism_loss = (knot_loss + 0.8 * cov_loss + 1.5 * mmd_loss + 0.2 * miss_loss).clamp_max(5.0)

    ir = imp_real1.detach() / imp_real1.detach().sum().clamp_min(1e-5)
    is_ = imp_syn1 / imp_syn1.sum().clamp_min(1e-5)
    dw = syn["driver_weights"]
    dist_pen = dw[dmask].pow(2).mean() if bool(dmask.any()) else torch.tensor(0.0, device=device)
    driver_loss = (0.5 * (1.0 - F.cosine_similarity(ir.unsqueeze(0), is_.unsqueeze(0)).squeeze(0)) + 0.5 * F.mse_loss(dw / dw.sum().clamp_min(1e-5), ir) * float(p) + 0.5 * dist_pen).clamp_max(3.0)

    gain1, gain2 = float((l_null1 - l_real1).item()), float((l_null2 - l_real2).item())
    rets = [r for r, g in [(float(((l_null1 - l_syn1) / gain1).item()) if gain1 > 1e-3 else float("nan"), gain1), (float(((l_null2 - l_syn2) / gain2).item()) if gain2 > 1e-3 else float("nan"), gain2)] if math.isfinite(r)]
    unclipped_ret = float(sum(rets) / len(rets)) if rets else float("nan")
    total = 1.0 * tstr_loss + 0.45 * distill_loss + 0.70 * realism_loss + 0.25 * driver_loss
    return {
        "total": total, "tstr_loss": float(tstr_loss.item()), "distill_loss": float(distill_loss.item()),
        "driver_loss": float(driver_loss.item()), "realism_loss": float(realism_loss.item()),
        "mmd_loss": float(mmd_loss.item()), "unclipped_retention": unclipped_ret,
        "task_latent": enc["task_latent"], "encoding_bytes": model.encoding_bytes_fp16(p),
    }
