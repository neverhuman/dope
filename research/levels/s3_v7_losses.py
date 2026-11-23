"""Differentiable multi-auditor batch loss for `SingleEncV7` (split out of `s3_v7_batch_ops.py`)."""
from __future__ import annotations

import math

import numpy as np
import torch

from .v7_single_encoding import MAX_CLASSES


def compute_v7_batch_loss(
    x_syn: torch.Tensor,
    logits_syn: torch.Tensor,
    y_reg_syn: torch.Tensor,
    batch: dict[str, torch.Tensor],
    logits_qry: torch.Tensor | None = None,
    sd_qry: torch.Tensor | None = None,
    ref: dict[str, torch.Tensor] | None = None,
    hard_tstr_v73: bool = False,
    rff_dim: int = 16,
    swd_projs: int = 24,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiable multi-auditor loss + proper-scoring-rule NLL + trust-region residual anchor."""
    bsz, n_syn, p_dim = x_syn.shape
    device = x_syn.device
    col_mask, X_qry, y_qry, q_mask, task_id, k_eff = (
        batch["col_mask"], batch["X_qry"] * batch["col_mask"].unsqueeze(1),
        batch["y_qry"], batch["q_mask"], batch["task_id"], batch["k_eff"],
    )

    # 1. Differentiable RFF + Linear TSTR surrogate on (X_syn -> QRY)
    w_rff = torch.randn((p_dim, rff_dim), device=device, dtype=x_syn.dtype) * 0.45
    b_rff = torch.rand((1, 1, rff_dim), device=device, dtype=x_syn.dtype) * (2.0 * math.pi)
    phi_syn = torch.cat(
        [x_syn, torch.cos(torch.matmul(x_syn, w_rff) + b_rff) * (2.0 / rff_dim) ** 0.5, torch.ones((bsz, n_syn, 1), device=device)],
        dim=-1,
    )
    phi_qry = torch.cat(
        [X_qry, torch.cos(torch.matmul(X_qry, w_rff) + b_rff) * (2.0 / rff_dim) ** 0.5, torch.ones((bsz, X_qry.shape[1], 1), device=device)],
        dim=-1,
    )
    f_dim = phi_syn.shape[-1]
    reg_diag = torch.full((f_dim,), 2.0, device=device, dtype=torch.float32)
    if hard_tstr_v73:
        reg_diag[-1] = 1e-4
    gram_s = torch.matmul(phi_syn.transpose(1, 2), phi_syn) + torch.diag(reg_diag).unsqueeze(0)

    y_soft = torch.zeros((bsz, n_syn, MAX_CLASSES), device=device, dtype=torch.float32)
    reg_m, bin_m, mc_m = task_id == 0, task_id == 1, task_id == 2
    if reg_m.any():
        y_soft[reg_m, :, 0] = y_reg_syn[reg_m]
    if bin_m.any():
        p_b = torch.sigmoid(logits_syn[bin_m, :, 0].clamp(-8.0, 8.0))
        if hard_tstr_v73:
            y_h = (torch.rand_like(p_b) < p_b).float()
            y_soft[bin_m, :, 0] = y_h - p_b.detach() + p_b
        else:
            y_soft[bin_m, :, 0] = p_b
    if mc_m.any():
        c_idx = torch.arange(MAX_CLASSES, device=device).unsqueeze(0)
        valid_c = c_idx < k_eff[mc_m].unsqueeze(1)
        lg_mc = logits_syn[mc_m].clamp(-8.0, 8.0).masked_fill(~valid_c.unsqueeze(1), -1e4)
        p_mc = torch.softmax(lg_mc, dim=-1)
        if hard_tstr_v73:
            u_g = torch.rand_like(lg_mc).clamp(1e-5, 1.0 - 1e-5)
            g_idx = (lg_mc - torch.log(-torch.log(u_g))).argmax(dim=-1)
            y_oh = torch.nn.functional.one_hot(g_idx, num_classes=MAX_CLASSES).float()
            y_soft[mc_m] = y_oh - p_mc.detach() + p_mc
        else:
            y_soft[mc_m] = p_mc

    beta_s = torch.linalg.solve(gram_s, torch.matmul(phi_syn.transpose(1, 2), y_soft))
    pred_q = torch.matmul(phi_qry, beta_s)

    tstr_losses, ret_vals = [], []
    q_denom = q_mask.sum(dim=1).clamp_min(1.0)
    if reg_m.any():
        err_s = ((pred_q[reg_m, :, 0] - y_qry[reg_m]) ** 2 * q_mask[reg_m]).sum(dim=1) / q_denom[reg_m]
        tstr_losses.append(err_s.clamp(max=25.0).mean())
        ym = (y_qry[reg_m] * q_mask[reg_m]).sum(dim=1, keepdim=True) / q_denom[reg_m].unsqueeze(1)
        var_y = (((y_qry[reg_m] - ym) ** 2) * q_mask[reg_m]).sum(dim=1) / q_denom[reg_m]
        ok_v = var_y > 0.05
        if ok_v.any():
            ret_vals.extend((1.0 - err_s[ok_v] / var_y[ok_v]).detach().cpu().tolist())
    if bin_m.any():
        p_q = pred_q[bin_m, :, 0].clamp(1e-4, 1.0 - 1e-4)
        yb = y_qry[bin_m].clamp(0.0, 1.0)
        bce = -(yb * torch.log(p_q) + (1.0 - yb) * torch.log(1.0 - p_q))
        tstr_losses.append(((bce * q_mask[bin_m]).sum(dim=1) / q_denom[bin_m]).clamp(max=10.0).mean())
    if mc_m.any():
        c_idx = torch.arange(MAX_CLASSES, device=device).unsqueeze(0)
        valid_c = c_idx < k_eff[mc_m].unsqueeze(1)
        probs_q = pred_q[mc_m].clamp(min=1e-4) * valid_c.unsqueeze(1).float()
        probs_q = probs_q / probs_q.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        yi = y_qry[mc_m].long().clamp(0, MAX_CLASSES - 1)
        p_true = torch.gather(probs_q, 2, yi.unsqueeze(-1)).squeeze(-1).clamp(min=1e-4)
        tstr_losses.append(((-torch.log(p_true) * q_mask[mc_m]).sum(dim=1) / q_denom[mc_m]).clamp(max=10.0).mean())
    tstr_loss = sum(tstr_losses) / max(len(tstr_losses), 1)

    # 2. Per-episode Marginal 1D-W1 + Correlation Frobenius + 24-projection Sliced Wasserstein (FP-0 / F4)
    # Strictly per-episode masked query rows: zero cross-episode min-truncation!
    q_grid = torch.linspace(0.05, 0.95, 17, device=device)
    proj = torch.randn((p_dim, swd_projs), device=device, dtype=x_syn.dtype)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-6)
    marg_terms: list[torch.Tensor] = []
    corr_terms: list[torch.Tensor] = []
    swd_terms: list[torch.Tensor] = []
    for bi in range(bsz):
        valid_rows = q_mask[bi] > 0.5
        nq_i = int(valid_rows.sum().item())
        cm_i = col_mask[bi]
        p_sum = cm_i.sum().clamp_min(1.0)
        if nq_i < 4:
            continue
        xq_i = X_qry[bi, valid_rows]  # [nq_i, p_dim]
        xs_i = x_syn[bi]              # [n_syn, p_dim]

        qs_syn_i = torch.quantile(xs_i, q_grid, dim=0)
        qs_qry_i = torch.quantile(xq_i, q_grid, dim=0)
        marg_terms.append(((qs_syn_i - qs_qry_i).abs().mean(dim=0) * cm_i).sum() / p_sum)

        xs_c_i = xs_i - xs_i.mean(dim=0, keepdim=True)
        xq_c_i = xq_i - xq_i.mean(dim=0, keepdim=True)
        xs_n_i = xs_c_i / xs_c_i.std(dim=0, keepdim=True).clamp_min(0.05)
        xq_n_i = xq_c_i / xq_c_i.std(dim=0, keepdim=True).clamp_min(0.05)
        c_syn_i = torch.matmul(xs_n_i.transpose(0, 1), xs_n_i) / float(n_syn)
        c_qry_i = torch.matmul(xq_n_i.transpose(0, 1), xq_n_i) / float(nq_i)
        pm_i = cm_i.unsqueeze(1) * cm_i.unsqueeze(0)
        corr_terms.append((((c_syn_i - c_qry_i) ** 2) * pm_i).sum() / pm_i.sum().clamp_min(1.0))

        proj_i = proj * cm_i.unsqueeze(1)
        proj_i = proj_i / proj_i.norm(dim=0, keepdim=True).clamp_min(1e-6)
        swd_terms.append(
            (
                torch.quantile(torch.matmul(xs_i, proj_i), q_grid, dim=0)
                - torch.quantile(torch.matmul(xq_i, proj_i), q_grid, dim=0)
            ).abs().mean()
        )
    zero_t = torch.zeros((), device=device, dtype=torch.float32)
    marg_loss = torch.stack(marg_terms).mean() if marg_terms else zero_t
    corr_loss = torch.stack(corr_terms).mean() if corr_terms else zero_t
    swd_loss = torch.stack(swd_terms).mean() if swd_terms else zero_t

    # 3. Direct proper-scoring-rule conditional NLL on real held-out (X_qry, y_qry)
    cond_losses = []
    if logits_qry is not None and sd_qry is not None:
        if reg_m.any():
            mu_q = logits_qry[reg_m, :, 0]
            sd_v = sd_qry[reg_m].clamp(0.05, 4.0)
            mse_q = ((mu_q - y_qry[reg_m]) ** 2 * q_mask[reg_m]).sum(dim=1) / q_denom[reg_m]
            nll_q = (0.5 * (((mu_q - y_qry[reg_m]) / sd_v) ** 2 + 2.0 * torch.log(sd_v)) * q_mask[reg_m]).sum(dim=1) / q_denom[reg_m]
            cond_losses.append((mse_q.clamp(max=25.0) + 0.25 * nll_q.clamp(min=-2.0, max=10.0)).mean())
        if bin_m.any():
            bce_q = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_qry[bin_m, :, 0].clamp(-10.0, 10.0), y_qry[bin_m].clamp(0.0, 1.0), reduction="none"
            )
            cond_losses.append(((bce_q * q_mask[bin_m]).sum(dim=1) / q_denom[bin_m]).clamp(max=10.0).mean())
        if mc_m.any():
            c_idx = torch.arange(MAX_CLASSES, device=device).unsqueeze(0)
            valid_c = c_idx < k_eff[mc_m].unsqueeze(1)
            lg_mc = logits_qry[mc_m].clamp(-10.0, 10.0).masked_fill(~valid_c.unsqueeze(1), -1e4)
            logp = torch.log_softmax(lg_mc, dim=-1)
            yi = y_qry[mc_m].long().clamp(0, MAX_CLASSES - 1)
            nll_mc = -torch.gather(logp, 2, yi.unsqueeze(-1)).squeeze(-1)
            cond_losses.append(((nll_mc * q_mask[mc_m]).sum(dim=1) / q_denom[mc_m]).clamp(max=10.0).mean())
    cond_loss = sum(cond_losses) / max(len(cond_losses), 1) if cond_losses else tstr_loss

    # 4. Trust-region L2 anchor on neural packet residuals (eliminates warmup dip below analytic step-0!)
    anchor_loss = torch.zeros((), device=device, dtype=torch.float32)
    if ref is not None:
        anchor_loss = (
            0.04 * (ref["d_W"] ** 2).mean()
            + 0.05 * (ref["d_L"] ** 2).mean()
            + 0.05 * (ref["d_knots"] ** 2).mean()
            + 0.02 * (ref["temp_sd"] ** 2).mean()
        )
        if hard_tstr_v73:
            anchor_loss = (
                anchor_loss
                + 1.50 * (ref["d_W"] ** 2).mean() + 2.50 * (ref["w2"] ** 2).mean()
                + 0.50 * (ref["w1"] ** 2).mean() + 1.00 * (ref["temp_sd"] ** 2).mean()
                + 0.25 * (ref["d_knots"] ** 2).mean() + 0.25 * (ref["d_L"] ** 2).mean()
            )
            if bin_m.any():
                tau_scale = torch.exp(ref["temp_sd"][bin_m, 0])
                anchor_loss = (
                    anchor_loss
                    + 6.0 * (ref["d_W"][bin_m, 0] ** 2).mean() + 12.0 * (ref["w2"][bin_m, :, 0] ** 2).mean()
                    + 4.0 * (ref["temp_sd"][bin_m, 0] ** 2).mean() + 0.50 * (torch.relu(0.90 - tau_scale) ** 2).mean()
                )

    mmd_proxy = marg_loss + 0.5 * corr_loss + 0.5 * swd_loss
    total_loss = 0.70 * cond_loss + 0.30 * tstr_loss + 1.0 * marg_loss + 1.2 * corr_loss + 0.9 * swd_loss + anchor_loss
    metrics = {
        "loss": float(total_loss.detach().item()),
        "cond_loss": float(cond_loss.detach().item()),
        "tstr_loss": float(tstr_loss.detach().item()),
        "mmd_loss": float(mmd_proxy.detach().item()),
        "marg_loss": float(marg_loss.detach().item()),
        "corr_loss": float(corr_loss.detach().item()),
        "swd_loss": float(swd_loss.detach().item()),
        "anchor_loss": float(anchor_loss.detach().item()),
        "train_unclipped_retention": float(np.median(ret_vals)) if ret_vals else 0.0,
    }
    return total_loss, metrics
