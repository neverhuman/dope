"""DOPE Fidelity V5 Neural Generator & Direct Main-KPI TSTR Retention Objective.

Cross-column Set-Attention Transformer + Differentiable Cholesky Gaussian Copula +
Monotonic Marginal Spline Decoder + Unified Regression / Binary / Multiclass / Multi-Label
Target Generator trained directly with Multi-Learner Differentiable Coupled TSTR Retention
(Ridge + Nonlinear RFF Kernel), Driver/Distractor Attribution Alignment, Joint/Marginal
Realism, and Split-Invariance InfoNCE.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

TASK_IDS = {"regression": 0, "binary": 1, "multiclass": 2, "multilabel": 3}
N_QUANTILE_KNOTS = 17
COPULA_RANK = 16


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _normal_ppf(u: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv((2.0 * u.clamp(1e-5, 1.0 - 1e-5) - 1.0).clamp(-0.99999, 0.99999))


def extract_column_stats(
    X_fit: torch.Tensor,
    y_fit: torch.Tensor,
    task: str,
    n_classes: int,
) -> torch.Tensor:
    """Permutation-equivariant 24-dim per-column statistical & target-probe summary (FIT-only)."""
    with torch.autocast(device_type=X_fit.device.type, enabled=False):
        x = X_fit.float()
        y = y_fit.float()
        n, p = x.shape
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False).clamp_min(1e-5)
        z = ((x - mean) / std).clamp(-6.0, 6.0)
        qs = torch.quantile(
            x,
            torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=x.device, dtype=torch.float32),
            dim=0,
        )  # (5, p)
        skew = (z ** 3).mean(dim=0).clamp(-6.0, 6.0) / 6.0
        kurt = ((z ** 4).mean(dim=0) - 3.0).clamp(-3.0, 18.0) / 10.0
        abs_m = z.abs().mean(dim=0)
        frac_pos = (x > 0.0).float().mean(dim=0)
        frac_zero = (x.abs() < 1e-4).float().mean(dim=0)
        # Approximate discreteness / atom share via fraction of repeated neighbors in sorted column
        xs, _ = torch.sort(x, dim=0)
        tie_frac = (xs[1:] - xs[:-1]).abs().lt(1e-4).float().mean(dim=0) if n > 1 else torch.zeros_like(mean)
        asinh_m = torch.asinh(z).mean(dim=0)
        sin_m = torch.sin(1.5 * z).mean(dim=0)
        cos_m = torch.cos(1.5 * z).mean(dim=0)

        # Target scalar projection for fast driver probing (works across reg/bin/mc/ml)
        if y.ndim == 2:
            y_sc = y.mean(dim=1)
        elif task == "multiclass" and n_classes >= 3:
            y_sc = y / max(float(n_classes - 1), 1.0)
        else:
            y_sc = y
        yc = y_sc - y_sc.mean()
        ys = yc.std(unbiased=False).clamp_min(1e-5)
        yz = (yc / ys).unsqueeze(1)  # (n, 1)

        corr1 = (z * yz).mean(dim=0).clamp(-1.0, 1.0)
        corr_abs = corr1.abs()
        def _std_corr(feat: torch.Tensor) -> torch.Tensor:
            fn = (feat - feat.mean(dim=0)) / feat.std(dim=0, unbiased=False).clamp_min(1e-5)
            return (fn * yz).mean(dim=0).clamp(-1.0, 1.0)

        corr2, corr_t, corr_r = _std_corr(z * z), _std_corr(torch.tanh(z)), _std_corr(F.relu(z))
        ctx_n = torch.full((p,), math.log1p(n) / 8.0, device=x.device, dtype=torch.float32)
        ctx_p = torch.full((p,), float(p) / 64.0, device=x.device, dtype=torch.float32)
        ctx_t = torch.full((p,), float(TASK_IDS.get(task, 0)) / 3.0, device=x.device, dtype=torch.float32)

        feats = torch.stack(
            [
                mean.clamp(-4.0, 4.0), std.clamp(0.0, 4.0),
                qs[0].clamp(-5.0, 5.0), qs[1].clamp(-5.0, 5.0), qs[2].clamp(-5.0, 5.0),
                qs[3].clamp(-5.0, 5.0), qs[4].clamp(-5.0, 5.0),
                skew, kurt, abs_m, frac_pos, frac_zero, tie_frac, asinh_m, sin_m, cos_m,
                corr1, corr_abs, corr2, corr_t, corr_r, ctx_n, ctx_p, ctx_t,
            ],
            dim=-1,
        )
        return torch.nan_to_num(feats, nan=0.0, posinf=4.0, neginf=-4.0)


class SetTransformerBlock(nn.Module):
    """Pre-norm multi-head self-attention block over feature columns (permutation-equivariant)."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, ff_mult: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.norm1(h)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        h = h + attn_out
        return h + self.ff(self.norm2(h))


class DopeFidelityV5(nn.Module):
    """Cross-table neural synthetic program generator directly optimizing Main-KPI TSTR retention."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        latent_dim: int = 128,
        copula_rank: int = COPULA_RANK,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.copula_rank = copula_rank

        self.col_in = nn.Sequential(nn.Linear(24, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.task_emb = nn.Embedding(4, d_model)
        self.blocks = nn.ModuleList([SetTransformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool_q = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.task_head = nn.Sequential(nn.Linear(d_model, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())
        self.dist_head = nn.Sequential(nn.Linear(d_model, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())
        self.driver_head = nn.Sequential(nn.Linear(d_model + latent_dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.marginal_warp_head = nn.Sequential(nn.Linear(d_model + latent_dim, 128), nn.GELU(), nn.Linear(128, N_QUANTILE_KNOTS))
        self.copula_factor_head = nn.Sequential(nn.Linear(d_model + latent_dim, 128), nn.GELU(), nn.Linear(128, copula_rank + 1))
        self.gate_head = nn.Sequential(nn.Linear(latent_dim * 2, 128), nn.GELU(), nn.Linear(128, 9))
        self.col_basis_scale = nn.Sequential(nn.Linear(d_model + latent_dim, 128), nn.GELU(), nn.Linear(128, 6))
        for head in (self.marginal_warp_head, self.copula_factor_head, self.gate_head, self.col_basis_scale):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            self.gate_head[-1].bias.copy_(torch.tensor([1.0, 0.2, 0.4, 0.0, 0.2, 0.2, -1.5, -1.0, 0.0]))

    def encode_table(
        self,
        X_fit: torch.Tensor,
        y_fit: torch.Tensor,
        task: str,
        n_classes: int,
    ) -> dict[str, torch.Tensor]:
        stats = extract_column_stats(X_fit, y_fit, task, n_classes)
        tid = torch.tensor(TASK_IDS.get(task, 0), device=X_fit.device, dtype=torch.long)
        h = self.col_in(stats).unsqueeze(0) + self.task_emb(tid).view(1, 1, -1)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)  # (1, P, D)
        pooled, _ = self.pool_attn(self.pool_q, h, h, need_weights=False)  # (1, 2, D)
        task_lat = self.task_head(pooled[0, 0])  # (latent_dim,)
        dist_lat = self.dist_head(pooled[0, 1])  # (latent_dim,)
        h_col = h.squeeze(0)  # (P, D)
        p = h_col.shape[0]
        t_ctx = torch.cat([h_col, task_lat.unsqueeze(0).expand(p, -1)], dim=-1)
        d_ctx = torch.cat([h_col, dist_lat.unsqueeze(0).expand(p, -1)], dim=-1)

        driver_logits = self.driver_head(t_ctx).squeeze(-1)
        driver_weights = torch.sigmoid(driver_logits)  # (P,) in (0, 1)
        knot_warp = 0.15 * torch.tanh(self.marginal_warp_head(d_ctx))
        cop_out = self.copula_factor_head(d_ctx)
        cop_res = 0.25 * torch.tanh(cop_out[:, : self.copula_rank])
        cop_blend = torch.sigmoid(cop_out[:, self.copula_rank] + 1.5)  # starts near ~0.82 empirical trust
        col_scales = 1.0 + 0.5 * torch.tanh(self.col_basis_scale(t_ctx))  # (P, 6)

        raw_g = self.gate_head(torch.cat([task_lat, dist_lat], dim=-1))
        gates = {
            "linear": torch.sigmoid(raw_g[0]) * 1.5 + 0.25,
            "sq": torch.sigmoid(raw_g[1]) * 0.8,
            "asinh": torch.sigmoid(raw_g[2]) * 1.0,
            "sin": torch.sigmoid(raw_g[3]) * 0.6,
            "relu_pos": torch.sigmoid(raw_g[4]) * 0.8,
            "relu_neg": torch.sigmoid(raw_g[5]) * 0.8,
            "ridge_alpha": F.softplus(raw_g[6]) * 0.25 + 1e-2,
            "noise_scale": torch.sigmoid(raw_g[7]) * 0.8 + 0.05,
            "logit_temp": torch.sigmoid(raw_g[8]) * 1.5 + 0.75,
        }
        return {
            "h_col": h_col,
            "task_latent": task_lat,
            "dist_latent": dist_lat,
            "driver_weights": driver_weights,
            "knot_warp": knot_warp,
            "cop_res": cop_res,
            "cop_blend": cop_blend,
            "col_scales": col_scales,
            "gates": gates,
        }

    def _build_basis(
        self,
        X: torch.Tensor,
        enc: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        g = enc["gates"]
        dw = (0.15 + 0.85 * enc["driver_weights"]).unsqueeze(0)
        cs = enc["col_scales"]
        xw = X * dw
        return torch.cat(
            [
                g["linear"] * xw * cs[:, 0],
                g["sq"] * (xw * xw).clamp_max(16.0) * cs[:, 1],
                g["asinh"] * torch.asinh(xw) * cs[:, 2],
                g["sin"] * torch.sin(1.5 * xw) * cs[:, 3],
                g["relu_pos"] * F.relu(xw - 0.25) * cs[:, 4],
                g["relu_neg"] * F.relu(-xw - 0.25) * cs[:, 5],
            ],
            dim=1,
        )

    def synthesize_differentiable(
        self,
        X_fit: torch.Tensor,
        y_fit: torch.Tensor,
        enc: dict[str, torch.Tensor],
        task: str,
        n_classes: int,
        n_syn: int,
        gen: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Differentiable joint copula + monotonic spline + multi-task target emission (FP32 stable)."""
        with torch.autocast(device_type=X_fit.device.type, enabled=False):
            xf = X_fit.float()
            yf = y_fit.float()
            n_fit, p = xf.shape
            dev = xf.device

            # 1. Empirical normal-score copula factor from FIT + learned differentiable refinement
            ranks = torch.argsort(torch.argsort(xf, dim=0), dim=0).float()
            u_emp = (ranks + 0.5) / float(n_fit)
            z_emp = _normal_ppf(u_emp)
            corr = (z_emp.T @ z_emp) / float(max(n_fit, 1))
            corr = 0.5 * (corr + corr.T)
            diag_idx = torch.arange(p, device=dev)
            corr[diag_idx, diag_idx] = 1.0
            corr_reg = 0.96 * corr + 0.04 * torch.eye(p, device=dev)
            chol_emp = torch.linalg.cholesky(corr_reg)

            eps_emp = torch.randn((n_syn, p), generator=gen, device=dev, dtype=torch.float32)
            eps_r = torch.randn((n_syn, self.copula_rank), generator=gen, device=dev, dtype=torch.float32)
            z_cop_emp = eps_emp @ chol_emp.T
            z_cop_res = eps_r @ enc["cop_res"].float().T
            blend = enc["cop_blend"].float().unsqueeze(0)
            z_syn = blend * z_cop_emp + (1.0 - blend) * z_cop_res
            z_syn = z_syn / z_syn.std(dim=0, unbiased=False).clamp_min(1e-4)
            u_syn = _normal_cdf(z_syn).clamp(1e-5, 1.0 - 1e-5)

            # 2. Monotonic 17-knot empirical + exact identity-initialized multiplicative warp
            q_grid = torch.linspace(0.01, 0.99, N_QUANTILE_KNOTS, device=dev, dtype=torch.float32)
            base_knots = torch.quantile(xf, q_grid, dim=0).T  # (p, K)
            warp = enc["knot_warp"].float()
            base_diffs = (base_knots[:, 1:] - base_knots[:, :-1]).clamp_min(0.0)
            warp_mult = torch.exp(0.25 * torch.tanh(warp[:, 1:]))
            k0 = base_knots[:, :1] + 0.15 * torch.tanh(warp[:, :1])
            knots = torch.cat([k0, k0 + torch.cumsum(base_diffs * warp_mult, dim=1)], dim=1)
            # Linear interpolation per column
            pos = u_syn * (N_QUANTILE_KNOTS - 1)
            lo = pos.floor().long().clamp(0, N_QUANTILE_KNOTS - 2)
            frac = pos - lo.float()
            col_ids = torch.arange(p, device=dev).unsqueeze(0).expand(n_syn, -1)
            v_lo = knots[col_ids, lo]
            v_hi = knots[col_ids, lo + 1]
            X_syn = (v_lo + frac * (v_hi - v_lo)).clamp(-6.0, 6.0)

            # 3. Differentiable multi-task conditional target emission via FP32 closed-form Ridge
            phi_f = self._build_basis(xf, enc)
            phi_s = self._build_basis(X_syn, enc)
            ones_f = torch.ones((n_fit, 1), device=dev, dtype=torch.float32)
            ones_s = torch.ones((n_syn, 1), device=dev, dtype=torch.float32)
            Df = torch.cat([ones_f, phi_f], dim=1)
            Ds = torch.cat([ones_s, phi_s], dim=1)
            alpha = enc["gates"]["ridge_alpha"].float()
            reg = torch.full((Df.shape[1],), alpha.item(), device=dev, dtype=torch.float32) * (alpha / alpha.detach())
            reg[0] = 1e-5
            gram = (Df.T @ Df) / float(n_fit) + torch.diag(reg)

            if task == "regression":
                w_t = torch.linalg.solve(gram, (Df.T @ yf.unsqueeze(1)) / float(n_fit))
                y_hat_f = (Df @ w_t).squeeze(1)
                res_sd = (yf - y_hat_f).std(unbiased=False).clamp(0.02, 2.5)
                eps_y = torch.randn((n_syn,), generator=gen, device=dev, dtype=torch.float32)
                y_syn = (Ds @ w_t).squeeze(1) + enc["gates"]["noise_scale"].float() * res_sd * eps_y
                y_soft = y_syn
            elif task == "binary":
                prior = yf.mean().clamp(0.02, 0.98)
                w_t = torch.linalg.solve(gram, (Df.T @ (yf - prior).unsqueeze(1)) / float(n_fit))
                temp = enc["gates"]["logit_temp"].float()
                logits = torch.logit(prior) + (3.2 * temp) * (Ds @ w_t).squeeze(1)
                probs = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
                u_y = torch.rand((n_syn,), generator=gen, device=dev, dtype=torch.float32)
                hard = (u_y < probs.detach()).float()
                y_syn = hard + (probs - probs.detach())
                y_soft = probs
            elif task == "multiclass":
                k = max(n_classes, 3)
                y_oh = F.one_hot(yf.round().long().clamp(0, k - 1), num_classes=k).float()
                prior = y_oh.mean(dim=0).clamp(1e-3, 1.0)
                w_t = torch.linalg.solve(gram, (Df.T @ (y_oh - prior)) / float(n_fit))
                temp = enc["gates"]["logit_temp"].float()
                logits = torch.log(prior) + (3.2 * temp) * (Ds @ w_t)
                probs = torch.softmax(logits, dim=1).clamp(1e-4, 1.0)
                cdf = probs.cumsum(dim=1)
                u_y = torch.rand((n_syn, 1), generator=gen, device=dev, dtype=torch.float32)
                idx = (u_y >= cdf.detach()).sum(dim=1).clamp(0, k - 1)
                hard = F.one_hot(idx, num_classes=k).float()
                y_syn = hard + (probs - probs.detach())  # soft-gradient one-hot matrix (n_syn, K)
                y_soft = probs
            else:  # multilabel: y_fit has shape (n_fit, L) in {0, 1}
                prior = yf.mean(dim=0).clamp(0.02, 0.98)
                w_t = torch.linalg.solve(gram, (Df.T @ (yf - prior)) / float(n_fit))
                temp = enc["gates"]["logit_temp"].float()
                logits = torch.logit(prior) + (3.2 * temp) * (Ds @ w_t)
                probs = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
                u_y = torch.rand(probs.shape, generator=gen, device=dev, dtype=torch.float32)
                hard = (u_y < probs.detach()).float()
                y_syn = hard + (probs - probs.detach())
                y_soft = probs

            return {"X_syn": X_syn, "y_syn": y_syn, "y_soft": y_soft}


def _fit_eval_downstream_learner(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_qr: torch.Tensor,
    y_qr: torch.Tensor,
    task: str,
    n_classes: int,
    rff_w: torch.Tensor | None,
    rff_b: torch.Tensor | None,
    ridge_lam: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 differentiable downstream learner (Ridge or RFF-Kernel) returning (qry_loss, null_loss, w_imp)."""
    with torch.autocast(device_type=X_tr.device.type, enabled=False):
        xtr, xqr = X_tr.float(), X_qr.float()
        ytr, yqr = y_tr.float(), y_qr.float()
        n_tr, p = xtr.shape
        dev = xtr.device

        if rff_w is None:
            f_tr = torch.cat([xtr, (xtr * xtr).clamp_max(16.0), torch.sin(xtr)], dim=1)
            f_qr = torch.cat([xqr, (xqr * xqr).clamp_max(16.0), torch.sin(xqr)], dim=1)
        else:
            proj_tr = xtr @ rff_w + rff_b
            proj_qr = xqr @ rff_w + rff_b
            f_tr = torch.cat([xtr, torch.cos(proj_tr), torch.sin(proj_tr)], dim=1)
            f_qr = torch.cat([xqr, torch.cos(proj_qr), torch.sin(proj_qr)], dim=1)

        Dtr = torch.cat([torch.ones((n_tr, 1), device=dev), f_tr], dim=1)
        Dqr = torch.cat([torch.ones((xqr.shape[0], 1), device=dev), f_qr], dim=1)
        eye = torch.eye(Dtr.shape[1], device=dev) * ridge_lam
        eye[0, 0] = 1e-5
        gram = (Dtr.T @ Dtr) / float(max(n_tr, 1)) + eye

        if task == "regression":
            w = torch.linalg.solve(gram, (Dtr.T @ ytr.unsqueeze(1)) / float(max(n_tr, 1)))
            pred = (Dqr @ w).squeeze(1)
            loss = F.mse_loss(pred, yqr)
            null_loss = F.mse_loss(torch.full_like(yqr, ytr.mean().detach()), yqr)
            w_imp = w[1 : 1 + p, 0].abs()
        elif task == "binary":
            p0 = ytr.mean().clamp(0.02, 0.98)
            w = torch.linalg.solve(gram, (Dtr.T @ (ytr - p0).unsqueeze(1)) / float(max(n_tr, 1)))
            logits = torch.logit(p0) + 3.0 * (Dqr @ w).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, yqr)
            null_loss = F.binary_cross_entropy_with_logits(torch.full_like(yqr, torch.logit(p0).detach()), yqr)
            w_imp = w[1 : 1 + p, 0].abs()
        elif task == "multiclass":
            k = max(n_classes, 3)
            y_oh = ytr if (ytr.ndim == 2 and ytr.shape[1] == k) else F.one_hot(ytr.round().long().clamp(0, k - 1), k).float()
            p0 = y_oh.mean(dim=0).clamp(1e-3, 1.0)
            w = torch.linalg.solve(gram, (Dtr.T @ (y_oh - p0)) / float(max(n_tr, 1)))
            logits = torch.log(p0) + 3.0 * (Dqr @ w)
            y_idx = yqr.round().long().clamp(0, k - 1)
            loss = F.cross_entropy(logits, y_idx)
            null_logits = torch.log(p0).detach().unsqueeze(0).expand(y_idx.shape[0], -1)
            null_loss = F.cross_entropy(null_logits, y_idx)
            w_imp = w[1 : 1 + p].abs().mean(dim=1)
        else:  # multilabel
            p0 = ytr.mean(dim=0).clamp(0.02, 0.98)
            w = torch.linalg.solve(gram, (Dtr.T @ (ytr - p0)) / float(max(n_tr, 1)))
            logits = torch.logit(p0) + 3.0 * (Dqr @ w)
            loss = F.binary_cross_entropy_with_logits(logits, yqr)
            null_logits = torch.logit(p0).detach().unsqueeze(0).expand_as(yqr)
            null_loss = F.binary_cross_entropy_with_logits(null_logits, yqr)
            w_imp = w[1 : 1 + p].abs().mean(dim=1)

        return loss, null_loss, w_imp


def compute_main_kpi_loss(
    model: DopeFidelityV5,
    episode: dict[str, Any],
    device: torch.device,
    step_seed: int,
) -> dict[str, Any]:
    """Compute end-to-end differentiable Main-KPI TSTR Retention + Driver + Realism loss."""
    X_fit = torch.as_tensor(episode["X_fit"], dtype=torch.float32, device=device)
    y_fit = torch.as_tensor(episode["y_fit"], dtype=torch.float32, device=device)
    X_qry = torch.as_tensor(episode["X_qry"], dtype=torch.float32, device=device)
    y_qry = torch.as_tensor(episode["y_qry"], dtype=torch.float32, device=device)
    dmask = torch.as_tensor(episode["distractor_mask"], dtype=torch.bool, device=device)
    task: str = episode["task"]
    n_classes: int = int(episode["n_classes"])

    enc = model.encode_table(X_fit, y_fit, task, n_classes)
    gen = torch.Generator(device=device).manual_seed(step_seed & 0x7FFF_FFFF)
    n_syn = max(int(X_fit.shape[0]) * 2, 128)
    syn = model.synthesize_differentiable(X_fit, y_fit, enc, task, n_classes, n_syn, gen)
    X_syn, y_syn = syn["X_syn"], syn["y_syn"]

    # Shared RFF projection for nonlinear downstream learner 2
    p = X_fit.shape[1]
    rff_dim = min(48, max(16, p * 2))
    rff_w = torch.randn((p, rff_dim), generator=gen, device=device) / math.sqrt(max(p, 1))
    rff_b = torch.rand((1, rff_dim), generator=gen, device=device) * (2.0 * math.pi)

    # Learner 1 (Polynomial/Periodic Ridge) + Learner 2 (Nonlinear RFF Kernel)
    l_syn1, l_null1, imp_s1 = _fit_eval_downstream_learner(X_syn, y_syn, X_qry, y_qry, task, n_classes, None, None)
    l_real1, _, imp_r1 = _fit_eval_downstream_learner(X_fit, y_fit, X_qry, y_qry, task, n_classes, None, None)
    l_syn2, l_null2, imp_s2 = _fit_eval_downstream_learner(X_syn, y_syn, X_qry, y_qry, task, n_classes, rff_w, rff_b)
    l_real2, _, imp_r2 = _fit_eval_downstream_learner(X_fit, y_fit, X_qry, y_qry, task, n_classes, rff_w, rff_b)

    gain1 = (l_null1.detach() - l_real1.detach()).clamp_min(0.05)
    gain2 = (l_null2.detach() - l_real2.detach()).clamp_min(0.05)
    ret1 = (l_null1.detach() - l_syn1.detach()) / gain1
    ret2 = (l_null2.detach() - l_syn2.detach()) / gain2
    unclipped_ret = float(0.5 * (ret1.item() + ret2.item()))

    # Direct normalized retention gap minimization (pushes unclipped retention -> 1.00+)
    tstr_loss = (
        0.5 * (F.smooth_l1_loss(l_syn1, l_real1.detach()) / gain1 + F.smooth_l1_loss(l_syn2, l_real2.detach()) / gain2)
        + 0.20 * (l_syn1 + l_syn2)
    )

    # Driver attribution alignment + explicit distractor-zeroing loss
    imp_real = (imp_r1.detach() + imp_r2.detach())
    imp_syn = (imp_s1 + imp_s2)
    imp_real_n = imp_real / (imp_real.sum() + 1e-6)
    imp_syn_n = imp_syn / (imp_syn.sum() + 1e-6)
    driver_loss = F.l1_loss(imp_syn_n, imp_real_n)
    pred_dw = enc["driver_weights"]
    driver_loss = driver_loss + 0.5 * F.l1_loss(pred_dw / (pred_dw.sum() + 1e-6), imp_real_n)
    if bool(dmask.any()):
        driver_loss = driver_loss + 0.5 * pred_dw[dmask].mean()

    # Joint covariance + 1D quantile W1 + moment realism loss (suppresses C2ST & MargAUC)
    cov_f = (X_fit.T @ X_fit) / float(max(X_fit.shape[0], 1))
    cov_s = (X_syn.T @ X_syn) / float(max(X_syn.shape[0], 1))
    q_eval = torch.linspace(0.05, 0.95, 9, device=device)
    w1_loss = F.l1_loss(torch.quantile(X_syn, q_eval, dim=0), torch.quantile(X_fit, q_eval, dim=0))
    realism_loss = F.mse_loss(cov_s, cov_f) + 0.5 * w1_loss + 0.25 * F.mse_loss(X_syn.mean(0), X_fit.mean(0))

    total = tstr_loss + 0.25 * driver_loss + 0.25 * realism_loss
    return {
        "total": total,
        "tstr_loss": float(tstr_loss.detach().item()),
        "driver_loss": float(driver_loss.detach().item()),
        "realism_loss": float(realism_loss.detach().item()),
        "unclipped_retention": unclipped_ret,
        "task_latent": enc["task_latent"],
        "dist_latent": enc["dist_latent"],
    }
