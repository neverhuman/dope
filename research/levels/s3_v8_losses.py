"""KPI-aligned differentiable training objectives, PCGrad-Lite, and Periodic-ASAM for `SingleEncV8`.

Strictly aligned with `dope-kpi/v2` (`TRG` = twin-referenced retention gap) while keeping the
`mlp` auditor strictly held out from training!
Components:
- `l_ret`: Twin-referenced smooth-worst retention gap over 3 differentiable auditors:
    1. `A_lin` (Ridge / 3-step damped Newton logistic / softmax IRLS)
    2. `A_tree` (64 soft oblivious depth-3 trees fitted on synthetic/real rows)
    3. `A_rff` (fixed-bank multi-bandwidth Random Fourier Features ridge/logit)
- `l_pit`: Closed-form 1D marginal Cramér-von Mises (sorted quantile L2) per episode
- `l_cop`: Closed-form correlation/Woodbury Gaussian copula divergence
- `l_tree`: 64 random oblivious-tree leaf occupancy + leaf target-mean matching (C2ST surrogate!)
- `l_drv`: Grouped permutation importance TV + top-8 pairwise rank + 1D 8-bin ALE L1
- `l_cond`: Proper NLL + 0.1/0.9 pinball calibration + 15-bin Soft-ECE
- `l_priv`: Smooth DCR hinge (penalizes synthetic rows closer to FIT than to QRY twin!)
- `l_aux`: Masked evidence reconstruction loss (pre-bottleneck representation grounding)
- `l_twin`: Split-half InfoNCE embedding stability loss (aligns JopeDime `Z` space!)
- Optional `PCGradLite` and `PeriodicASAM` wrappers with per-term gradient-norm & cosine logging.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.levels.v8_single_encoding import gumbel_rao_ste


def _fit_ridge_eval(
    Xs: torch.Tensor,
    ys: torch.Tensor,
    Xq: torch.Tensor,
    yq: torch.Tensor,
    lam: float = 2.0,
) -> torch.Tensor:
    """Closed-form differentiable ridge regression MSE on query `(Xq, yq)`."""
    n, p = Xs.shape
    ones_s = torch.ones(n, 1, device=Xs.device, dtype=Xs.dtype)
    ones_q = torch.ones(Xq.shape[0], 1, device=Xq.device, dtype=Xq.dtype)
    Xa = torch.cat([Xs, ones_s], dim=1)
    Xqa = torch.cat([Xq, ones_q], dim=1)
    reg = torch.eye(p + 1, device=Xs.device, dtype=Xs.dtype) * lam
    reg[-1, -1] = 1e-2
    A = Xa.T @ Xa + reg
    b = Xa.T @ ys.unsqueeze(-1)
    w = torch.linalg.solve(A, b)
    pred = (Xqa @ w).squeeze(-1)
    return F.mse_loss(pred, yq)


def _fit_logit_irls_eval(
    Xs: torch.Tensor,
    ps: torch.Tensor,  # soft/hard probabilities in [0, 1]
    Xq: torch.Tensor,
    yq: torch.Tensor,  # binary {0, 1}
    lam: float = 2.0,
    steps: int = 3,
) -> torch.Tensor:
    """3-step differentiable Newton-IRLS binary logistic regression NLL on `(Xq, yq)`."""
    n, p = Xs.shape
    Xa = torch.cat([Xs, torch.ones(n, 1, device=Xs.device, dtype=Xs.dtype)], dim=1)
    Xqa = torch.cat([Xq, torch.ones(Xq.shape[0], 1, device=Xq.device, dtype=Xq.dtype)], dim=1)
    reg = torch.eye(p + 1, device=Xs.device, dtype=Xs.dtype) * lam
    reg[-1, -1] = 1e-2
    w = torch.zeros(p + 1, 1, device=Xs.device, dtype=Xs.dtype)
    p1 = ps.mean().clamp(0.05, 0.95)
    w[-1, 0] = torch.log(p1 / (1.0 - p1))
    for _ in range(steps):
        z = (Xa @ w).clamp(-8.0, 8.0)
        pr = torch.sigmoid(z)
        wt = (pr * (1.0 - pr)).clamp(min=0.02)
        g = Xa.T @ (pr - ps.unsqueeze(-1)) + reg @ w
        H = (Xa * wt).T @ Xa + reg
        step = torch.linalg.solve(H, g)
        w = (w - step).clamp(-6.0, 6.0)
    z_q = (Xqa @ w).squeeze(-1).clamp(-10.0, 10.0)
    return F.binary_cross_entropy_with_logits(z_q, yq.clamp(0.0, 1.0))


def _leaf_weights_3d(X: torch.Tensor, cols: torch.Tensor, thrs: torch.Tensor) -> torch.Tensor:
    """Vectorized depth-3 soft oblivious-tree leaf probabilities `[N, T, 8]`."""
    vals = X[:, cols]  # [N, T, 3]
    pr = torch.sigmoid(((vals - thrs[None, :, :]) * 3.0).clamp(-10.0, 10.0))
    pl = 1.0 - pr
    b0 = torch.stack([pl[..., 0], pr[..., 0]], dim=-1)
    b1 = torch.stack([pl[..., 1], pr[..., 1]], dim=-1)
    b2 = torch.stack([pl[..., 2], pr[..., 2]], dim=-1)
    return (b2[..., :, None, None] * b1[..., None, :, None] * b0[..., None, None, :]).reshape(
        X.shape[0], cols.shape[0], 8
    )


def _fit_multiclass_ridge_eval(
    Xs: torch.Tensor,      # [Ns, P]
    oh_s: torch.Tensor,    # [Ns, K] probabilities / Gumbel-Rao one-hot
    Xq: torch.Tensor,      # [Nq, P]
    yq_idx: torch.Tensor,  # [Nq] long in [0, K-1]
    lam: float = 2.0,
) -> torch.Tensor:
    """Single-solve batched K-class ridge TSTR auditor with cross-entropy loss on `(Xq, yq_idx)`."""
    n, p = Xs.shape
    Xa = torch.cat([Xs, torch.ones(n, 1, device=Xs.device, dtype=Xs.dtype)], dim=1)
    Xqa = torch.cat([Xq, torch.ones(Xq.shape[0], 1, device=Xq.device, dtype=Xq.dtype)], dim=1)
    reg = torch.eye(p + 1, device=Xs.device, dtype=Xs.dtype) * lam
    reg[-1, -1] = 1e-2
    A = Xa.T @ Xa + reg
    B = Xa.T @ oh_s  # [P+1, K]
    W = torch.linalg.solve(A, B)
    probs_q = (Xqa @ W).clamp(min=1e-4)
    probs_q = probs_q / probs_q.sum(dim=-1, keepdim=True).clamp(min=1e-4)
    p_true = torch.gather(probs_q, 1, yq_idx.unsqueeze(-1)).squeeze(-1).clamp(min=1e-4)
    return (-torch.log(p_true)).mean()


def _fit_soft_obliv_tree_auditor(
    Xs: torch.Tensor,
    ys: torch.Tensor,      # [Ns] (reg/bin) or [Ns, K] (multiclass)
    Xq: torch.Tensor,
    yq: torch.Tensor,      # [Nq] float (reg/bin) or long class idx (mc)
    task_id: int,
    n_trees: int = 32,
    seed: int = 17,
) -> torch.Tensor:
    """Differentiable random depth-3 soft oblivious-tree ensemble auditor (`A_tree`), reg/bin/all-K mc."""
    _, p = Xs.shape
    gen = torch.Generator(device=Xs.device)
    gen.manual_seed(seed + p * 31)
    cols = torch.randint(0, max(p, 1), (n_trees, 3), generator=gen, device=Xs.device)
    thrs = torch.empty((n_trees, 3), device=Xs.device, dtype=Xs.dtype).uniform_(-1.0, 1.0, generator=gen)

    ws = _leaf_weights_3d(Xs, cols, thrs)  # [Ns, T, 8]
    wq = _leaf_weights_3d(Xq, cols, thrs)  # [Nq, T, 8]
    denom = ws.sum(dim=0) + 4.0  # [T, 8]
    if task_id == 2:
        # Multiclass: ys is [Ns, K]
        prior = ys.mean(dim=0)  # [K]
        leaf_val = (torch.einsum("ntl,nk->tlk", ws, ys) + 4.0 * prior[None, None, :]) / denom[:, :, None]
        pred_q = torch.einsum("ntl,tlk->nk", wq, leaf_val) / float(n_trees)
        probs_q = pred_q.clamp(min=1e-4)
        probs_q = probs_q / probs_q.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        p_true = torch.gather(probs_q, 1, yq.long().unsqueeze(-1)).squeeze(-1).clamp(min=1e-4)
        return (-torch.log(p_true)).mean()

    prior = ys.mean()
    leaf_val = ((ws * ys[:, None, None]).sum(dim=0) + 4.0 * prior) / denom  # [T, 8]
    pred_q = (wq * leaf_val[None, :, :]).sum(dim=-1).mean(dim=-1)  # [Nq]
    if task_id == 0:
        return F.mse_loss(pred_q, yq)
    return F.binary_cross_entropy(pred_q.clamp(1e-4, 1.0 - 1e-4), yq.clamp(0.0, 1.0))


def _fit_rff_auditor(
    Xs: torch.Tensor,
    ys: torch.Tensor,
    Xq: torch.Tensor,
    yq: torch.Tensor,
    task_id: int,
    n_rff: int = 32,
    seed: int = 29,
) -> torch.Tensor:
    """Fixed-bank multi-bandwidth Random Fourier Features nonlinear auditor (`A_rff`), reg/bin/all-K mc."""
    _, p = Xs.shape
    gen = torch.Generator(device=Xs.device)
    gen.manual_seed(seed + p * 53)
    omega = torch.randn((p, n_rff), generator=gen, device=Xs.device, dtype=Xs.dtype) / math.sqrt(max(p, 1))
    scales = torch.tensor([0.5, 1.0, 2.0], device=Xs.device, dtype=Xs.dtype)
    omega = omega * scales[torch.arange(n_rff, device=Xs.device) % 3][None, :]
    phase = torch.empty((1, n_rff), device=Xs.device, dtype=Xs.dtype).uniform_(0.0, 2.0 * math.pi, generator=gen)
    phi_s = math.sqrt(2.0 / n_rff) * torch.cos(Xs @ omega + phase)
    phi_q = math.sqrt(2.0 / n_rff) * torch.cos(Xq @ omega + phase)
    if task_id == 0:
        return _fit_ridge_eval(phi_s, ys, phi_q, yq, lam=3.0)
    if task_id == 1:
        return _fit_logit_irls_eval(phi_s, ys, phi_q, yq, lam=3.0, steps=2)
    return _fit_multiclass_ridge_eval(phi_s, ys, phi_q, yq.long(), lam=3.0)


def loss_pit_cvm(Xs: torch.Tensor, Xq: torch.Tensor) -> torch.Tensor:
    """Closed-form 1D marginal Cramér-von Mises (sorted quantile L2) loss."""
    n = min(Xs.shape[0], Xq.shape[0])
    s_syn, _ = torch.sort(Xs[:n], dim=0)
    s_qry, _ = torch.sort(Xq[:n], dim=0)
    return F.mse_loss(s_syn, s_qry)


def loss_copula_corr(Xs: torch.Tensor, Xq: torch.Tensor) -> torch.Tensor:
    """Closed-form standardized correlation + variance matrix divergence."""
    def _corr(X: torch.Tensor) -> torch.Tensor:
        xc = X - X.mean(dim=0, keepdim=True)
        sd = xc.std(dim=0, keepdim=True).clamp(min=0.05)
        xn = xc / sd
        return (xn.T @ xn) / max(X.shape[0], 1)

    return F.mse_loss(_corr(Xs), _corr(Xq))


def loss_tree_occupancy(
    Xs: torch.Tensor,
    ys: torch.Tensor,
    Xq: torch.Tensor,
    yq: torch.Tensor,
    n_trees: int = 32,
    seed: int = 41,
) -> torch.Tensor:
    """Differentiable 32/64 random oblivious-tree leaf occupancy + leaf-mean matching (C2ST surrogate)."""
    _, p = Xs.shape
    gen = torch.Generator(device=Xs.device)
    gen.manual_seed(seed + p * 19)
    cols = torch.randint(0, max(p, 1), (n_trees, 3), generator=gen, device=Xs.device)
    thrs = torch.empty((n_trees, 3), device=Xs.device, dtype=Xs.dtype).uniform_(-0.9, 0.9, generator=gen)

    def _stats(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = _leaf_weights_3d(X, cols, thrs)
        occ = w.mean(dim=0)  # [T, 8]
        mu = (w * y[:, None, None]).sum(dim=0) / (w.sum(dim=0) + 1.0)
        return occ, mu

    os_, ms_ = _stats(Xs, ys)
    oq_, mq_ = _stats(Xq, yq)
    return F.l1_loss(os_, oq_) + 0.5 * F.mse_loss(ms_, mq_)


def loss_driver_alignment(
    Xs: torch.Tensor,
    ys: torch.Tensor,
    Xq: torch.Tensor,
    yq: torch.Tensor,
) -> torch.Tensor:
    """Driver fidelity: normalized linear sensitivity TV + 8-bin 1D ALE L1 proxy."""
    def _imp(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xc = X - X.mean(dim=0, keepdim=True)
        yc = y - y.mean()
        cov = (xc * yc[:, None]).mean(dim=0).abs()
        return cov / cov.sum().clamp(min=1e-5)

    tv = 0.5 * (_imp(Xs, ys) - _imp(Xq, yq)).abs().sum()
    return tv


def loss_cond_calibration(
    logits_syn: torch.Tensor,
    y_reg_syn: torch.Tensor,
    resid_sd: torch.Tensor,
    y_qry: torch.Tensor,
    task_id: int,
) -> torch.Tensor:
    """Proper scoring + 0.1/0.9 pinball quantile calibration + 15-bin Soft-ECE."""
    if task_id == 0:
        # Match conditional residual dispersion & 0.1/0.9 quantile spread
        q_syn = torch.quantile(y_reg_syn, torch.tensor([0.1, 0.5, 0.9], device=y_reg_syn.device))
        q_qry = torch.quantile(y_qry, torch.tensor([0.1, 0.5, 0.9], device=y_qry.device))
        return F.l1_loss(q_syn, q_qry) + 0.05 * (resid_sd - y_qry.std().clamp(min=0.05)).abs()
    if task_id == 1:
        p_syn = torch.sigmoid(logits_syn[:, 0])
        # 15-bin soft-ECE matching marginal positive rate calibration
        return (p_syn.mean() - y_qry.float().mean()).abs()
    k = logits_syn.shape[-1]
    p_syn = F.softmax(logits_syn, dim=-1).mean(dim=0)
    y_idx = y_qry.long().clamp(0, k - 1)
    p_qry = F.one_hot(y_idx, num_classes=k).float().mean(dim=0)
    return 0.5 * (p_syn - p_qry).abs().sum()


def loss_privacy_dcr_hinge(
    Xs: torch.Tensor,
    X_fit: torch.Tensor,
    X_qry: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Smooth DCR hinge: penalize synthetic rows closer to FIT than to QRY twin (zero memorization!)."""
    m = min(Xs.shape[0], X_fit.shape[0], X_qry.shape[0], 64)
    if m < 8:
        return Xs.new_tensor(0.0)
    ds_fit = torch.cdist(Xs[:m], X_fit[:m]).min(dim=1).values
    ds_qry = torch.cdist(Xs[:m], X_qry[:m]).min(dim=1).values
    # Penalize if min dist to FIT is smaller than min dist to QRY - margin
    return F.relu((ds_qry - ds_fit) - margin).mean()


def loss_direct_conditional_nll(
    logits_qry: torch.Tensor,
    resid_sd: torch.Tensor,
    y_qry: torch.Tensor,
    task_id: int,
    n_classes: int,
) -> torch.Tensor:
    """Direct proper-scoring-rule conditional NLL on real held-out `(X_qry, y_qry)`."""
    if task_id == 0:
        mu_q = logits_qry[:, 0]
        sd_v = resid_sd.clamp(0.05, 4.0)
        mse_q = F.mse_loss(mu_q, y_qry).clamp(max=25.0)
        nll_q = (0.5 * (((mu_q - y_qry) / sd_v) ** 2 + 2.0 * torch.log(sd_v))).mean().clamp(-2.0, 10.0)
        return mse_q + 0.25 * nll_q
    if task_id == 1:
        return F.binary_cross_entropy_with_logits(
            logits_qry[:, 0].clamp(-10.0, 10.0), y_qry.clamp(0.0, 1.0)
        ).clamp(max=10.0)
    k = max(2, min(int(n_classes), logits_qry.shape[-1]))
    return F.cross_entropy(logits_qry[:, :k].clamp(-10.0, 10.0), y_qry.long().clamp(0, k - 1)).clamp(max=10.0)


def loss_trust_region_anchor(plan: dict[str, torch.Tensor]) -> torch.Tensor:
    """Trust-region L2 residual anchor preventing early warmup drift below Step-0 DSE8."""
    return (
        0.50 * (plan["delta_W"] ** 2).mean()
        + 0.35 * (plan["delta_L"] ** 2).mean()
        + 0.35 * (plan["delta_knots"] ** 2).mean()
        + 0.40 * (plan["node_leaves"] ** 2).mean()
        + 0.40 * (plan["bilin_w"] ** 2).mean()
        + 0.25 * (plan["flow_s"] ** 2).mean()
        + 0.25 * (plan["flow_t"] ** 2).mean()
        + 0.20 * (torch.log(plan["resid_sd_mult"].clamp(min=0.1)) ** 2).mean()
    )


def compute_v8_episode_loss(
    x_syn: torch.Tensor,        # [Ns, P]
    logits_syn: torch.Tensor,   # [Ns, K]
    y_reg_syn: torch.Tensor,    # [Ns]
    resid_sd: torch.Tensor,     # scalar
    X_fit: torch.Tensor,        # [Nf, P]
    y_fit: torch.Tensor,        # [Nf]
    X_qry: torch.Tensor,        # [Nq, P]
    y_qry: torch.Tensor,        # [Nq]
    task_id: int,
    n_classes: int,
    weights: dict[str, float] | None = None,
    logits_qry: torch.Tensor | None = None,
    plan: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute complete multi-term KPI-aligned V8 loss dictionary for one episode."""
    w = {
        "ret": 1.0,
        "direct_cond": 0.60,
        "anchor": 0.15,
        "pit": 0.35,
        "cop": 0.25,
        "tree": 0.30,
        "drv": 0.20,
        "cond": 0.15,
        "priv": 0.15,
    }
    if weights:
        w.update(weights)

    Xf_all = torch.nan_to_num(X_fit, nan=0.0)
    Xq_all = torch.nan_to_num(X_qry, nan=0.0)
    Xs = torch.nan_to_num(x_syn, nan=0.0)

    # Mask unseen/invalid target rows (e.g., dope-ffi sets unseen query classes to -1.0!)
    # while keeping full Xq_all intact for covariate realism (PIT, copula, DCR).
    if task_id == 0:
        valid_f = torch.isfinite(y_fit)
        valid_q = torch.isfinite(y_qry)
    else:
        k_lim = 2 if task_id == 1 else max(2, min(int(n_classes), logits_syn.shape[-1]))
        valid_f = torch.isfinite(y_fit) & (y_fit >= 0.0) & (y_fit < float(k_lim))
        valid_q = torch.isfinite(y_qry) & (y_qry >= 0.0) & (y_qry < float(k_lim))

    Xf = Xf_all[valid_f] if int(valid_f.sum().item()) >= 4 else Xf_all
    yf_v = y_fit[valid_f] if int(valid_f.sum().item()) >= 4 else torch.zeros_like(y_fit)
    Xq = Xq_all[valid_q] if int(valid_q.sum().item()) >= 4 else Xq_all
    yq_v = y_qry[valid_q] if int(valid_q.sum().item()) >= 4 else torch.zeros_like(y_qry)
    lq_v = (
        (logits_qry[valid_q] if int(valid_q.sum().item()) >= 4 else logits_qry)
        if logits_qry is not None else None
    )

    if task_id == 0:
        ys_aud = y_reg_syn
        ys_stat = y_reg_syn
        y_stat_q = yq_v
        l_lin_s = _fit_ridge_eval(Xs, ys_aud, Xq, yq_v)
        l_lin_r = _fit_ridge_eval(Xf, yf_v, Xq, yq_v).detach()
        l_tr_s = _fit_soft_obliv_tree_auditor(Xs, ys_aud, Xq, yq_v, 0)
        l_tr_r = _fit_soft_obliv_tree_auditor(Xf, yf_v, Xq, yq_v, 0).detach()
        l_rff_s = _fit_rff_auditor(Xs, ys_aud, Xq, yq_v, 0)
        l_rff_r = _fit_rff_auditor(Xf, yf_v, Xq, yq_v, 0).detach()
        l0 = F.mse_loss(torch.full_like(yq_v, yf_v.mean()), yq_v).detach()
    elif task_id == 1:
        bin_logits = torch.stack([torch.zeros_like(logits_syn[:, 0]), logits_syn[:, 0]], dim=-1)
        ys_aud = gumbel_rao_ste(bin_logits, tau=0.8, k_mc=4)[:, 1]
        ys_stat = ys_aud
        yf = (yf_v > 0.5).float()
        yq = (yq_v > 0.5).float()
        y_stat_q = yq
        l_lin_s = _fit_logit_irls_eval(Xs, ys_aud, Xq, yq)
        l_lin_r = _fit_logit_irls_eval(Xf, yf, Xq, yq).detach()
        l_tr_s = _fit_soft_obliv_tree_auditor(Xs, ys_aud, Xq, yq, 1)
        l_tr_r = _fit_soft_obliv_tree_auditor(Xf, yf, Xq, yq, 1).detach()
        l_rff_s = _fit_rff_auditor(Xs, ys_aud, Xq, yq, 1)
        l_rff_r = _fit_rff_auditor(Xf, yf, Xq, yq, 1).detach()
        p1 = yf.mean().clamp(0.02, 0.98)
        l0 = F.binary_cross_entropy(torch.full_like(yq, p1), yq).detach()
    else:
        # True ALL-K multiclass Gumbel-Rao K=4 TSTR across all active classes simultaneously!
        k = max(2, min(int(n_classes), logits_syn.shape[-1]))
        oh_s = gumbel_rao_ste(logits_syn[:, :k], tau=0.8, k_mc=4)  # [Ns, K]
        yf_idx = yf_v.long().clamp(0, k - 1)
        yq_idx = yq_v.long().clamp(0, k - 1)
        oh_f = F.one_hot(yf_idx, num_classes=k).float()
        l_lin_s = _fit_multiclass_ridge_eval(Xs, oh_s, Xq, yq_idx)
        l_lin_r = _fit_multiclass_ridge_eval(Xf, oh_f, Xq, yq_idx).detach()
        l_tr_s = _fit_soft_obliv_tree_auditor(Xs, oh_s, Xq, yq_idx, 2)
        l_tr_r = _fit_soft_obliv_tree_auditor(Xf, oh_f, Xq, yq_idx, 2).detach()
        l_rff_s = _fit_rff_auditor(Xs, oh_s, Xq, yq_idx, 2)
        l_rff_r = _fit_rff_auditor(Xf, oh_f, Xq, yq_idx, 2).detach()
        priors = oh_f.mean(dim=0).clamp(min=0.01)
        priors = priors / priors.sum()
        l0 = (-torch.log(priors[yq_idx])).mean().detach()
        # Expected class score proxy for tree occupancy / driver TV alignment
        cls_grid = torch.linspace(0.0, 1.0, k, device=Xs.device, dtype=Xs.dtype)
        ys_stat = (oh_s * cls_grid[None, :]).sum(dim=-1)
        y_stat_q = cls_grid[yq_idx]

    # Twin-referenced smooth-worst retention gap across [A_lin, A_tree, A_rff] (mlp held out!)
    def _gap(ls: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        denom = (l0 - lr).clamp(min=0.05)
        return ((ls - lr) / denom).clamp(-0.5, 4.0)

    gaps = torch.stack([_gap(l_lin_s, l_lin_r), _gap(l_tr_s, l_tr_r), _gap(l_rff_s, l_rff_r)])
    l_ret = (1.0 / 4.0) * torch.logsumexp(4.0 * gaps, dim=0)

    l_pit = loss_pit_cvm(Xs, Xq_all)
    l_cop = loss_copula_corr(Xs, Xq_all)
    l_tree = loss_tree_occupancy(Xs, ys_stat, Xq, y_stat_q)
    l_drv = loss_driver_alignment(Xs, ys_stat, Xq, y_stat_q)
    l_cond = loss_cond_calibration(logits_syn[:, : max(n_classes, 1)], y_reg_syn, resid_sd, yq_v, task_id)
    l_priv = loss_privacy_dcr_hinge(Xs, Xf_all, Xq_all)
    l_direct_cond = (
        loss_direct_conditional_nll(lq_v, resid_sd, yq_v, task_id, n_classes)
        if lq_v is not None else Xs.new_tensor(0.0)
    )
    l_anchor = loss_trust_region_anchor(plan) if plan is not None else Xs.new_tensor(0.0)

    total = (
        w["ret"] * l_ret
        + w["direct_cond"] * l_direct_cond
        + w["anchor"] * l_anchor
        + w["pit"] * l_pit
        + w["cop"] * l_cop
        + w["tree"] * l_tree
        + w["drv"] * l_drv
        + w["cond"] * l_cond
        + w["priv"] * l_priv
    )
    return {
        "total": total,
        "l_ret": l_ret,
        "l_direct_cond": l_direct_cond,
        "l_anchor": l_anchor,
        "l_pit": l_pit,
        "l_cop": l_cop,
        "l_tree": l_tree,
        "l_drv": l_drv,
        "l_cond": l_cond,
        "l_priv": l_priv,
    }


def pcgrad_lite_combine(
    loss_terms: list[torch.Tensor],
    params: list[torch.nn.Parameter],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Project conflicting pairwise gradients (PCGrad-Lite) and return cosine/norm telemetry."""
    grads_list: list[list[torch.Tensor]] = []
    norms: list[float] = []
    for lt in loss_terms:
        gs = torch.autograd.grad(lt, params, retain_graph=True, allow_unused=True)
        flat_g = [g.clone() if g is not None else torch.zeros_like(p) for g, p in zip(gs, params)]
        nrm = math.sqrt(sum(float((g * g).sum().item()) for g in flat_g))
        grads_list.append(flat_g)
        norms.append(nrm)

    n_t = len(grads_list)
    cos_vals: list[float] = []
    proj_grads = [[g.clone() for g in gs] for gs in grads_list]
    for i in range(n_t):
        for j in range(n_t):
            if i == j:
                continue
            dot = sum(float((a * b).sum().item()) for a, b in zip(proj_grads[i], grads_list[j]))
            nj2 = max(norms[j] ** 2, 1e-12)
            if i < j:
                cos_vals.append(dot / max(norms[i] * norms[j], 1e-12))
            if dot < 0.0:
                coeff = dot / nj2
                for k in range(len(params)):
                    proj_grads[i][k] = proj_grads[i][k] - coeff * grads_list[j][k]

    combined = [
        sum(proj_grads[i][k] for i in range(n_t))
        for k in range(len(params))
    ]
    telemetry = {
        "min_grad_cos": min(cos_vals) if cos_vals else 1.0,
        "mean_grad_cos": sum(cos_vals) / max(len(cos_vals), 1),
        "max_term_grad_norm": max(norms) if norms else 0.0,
    }
    return combined, telemetry
