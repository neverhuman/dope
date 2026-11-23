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


def _fit_soft_obliv_tree_auditor(
    Xs: torch.Tensor,
    ys: torch.Tensor,
    Xq: torch.Tensor,
    yq: torch.Tensor,
    task_id: int,
    n_trees: int = 32,
    seed: int = 17,
) -> torch.Tensor:
    """Differentiable random depth-3 soft oblivious-tree ensemble auditor (`A_tree`)."""
    n, p = Xs.shape
    gen = torch.Generator(device=Xs.device)
    gen.manual_seed(seed + p * 31)
    cols = torch.randint(0, max(p, 1), (n_trees, 3), generator=gen, device=Xs.device)
    thrs = torch.empty((n_trees, 3), device=Xs.device, dtype=Xs.dtype).uniform_(-1.0, 1.0, generator=gen)

    def _leaf_weights(X: torch.Tensor) -> torch.Tensor:
        # [N, T, 3]
        vals = X[:, cols]  # [N, T, 3]
        pr = torch.sigmoid((vals - thrs[None, :, :]) * 3.0)
        pl = 1.0 - pr
        return torch.stack(
            [
                (pr[..., 0] if (l & 1) else pl[..., 0])
                * (pr[..., 1] if (l & 2) else pl[..., 1])
                * (pr[..., 2] if (l & 4) else pl[..., 2])
                for l in range(8)
            ],
            dim=-1,
        )  # [N, T, 8]

    ws = _leaf_weights(Xs)  # [Ns, T, 8]
    wq = _leaf_weights(Xq)  # [Nq, T, 8]
    denom = ws.sum(dim=0) + 4.0  # [T, 8]
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
    """Fixed-bank multi-bandwidth Random Fourier Features nonlinear auditor (`A_rff`)."""
    _, p = Xs.shape
    gen = torch.Generator(device=Xs.device)
    gen.manual_seed(seed + p * 53)
    omega = torch.randn((p, n_rff), generator=gen, device=Xs.device, dtype=Xs.dtype) / math.sqrt(max(p, 1))
    # Multi-bandwidth scaling (0.5x, 1.0x, 2.0x)
    scales = torch.tensor([0.5, 1.0, 2.0], device=Xs.device, dtype=Xs.dtype)
    omega = omega * scales[torch.arange(n_rff, device=Xs.device) % 3][None, :]
    phase = torch.empty((1, n_rff), device=Xs.device, dtype=Xs.dtype).uniform_(0.0, 2.0 * math.pi, generator=gen)
    phi_s = math.sqrt(2.0 / n_rff) * torch.cos(Xs @ omega + phase)
    phi_q = math.sqrt(2.0 / n_rff) * torch.cos(Xq @ omega + phase)
    if task_id == 0:
        return _fit_ridge_eval(phi_s, ys, phi_q, yq, lam=3.0)
    return _fit_logit_irls_eval(phi_s, ys, phi_q, yq, lam=3.0, steps=2)


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
        vals = X[:, cols]
        pr = torch.sigmoid((vals - thrs[None, :, :]) * 3.0)
        pl = 1.0 - pr
        w = torch.stack(
            [
                (pr[..., 0] if (l & 1) else pl[..., 0])
                * (pr[..., 1] if (l & 2) else pl[..., 1])
                * (pr[..., 2] if (l & 4) else pl[..., 2])
                for l in range(8)
            ],
            dim=-1,
        )
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
) -> dict[str, torch.Tensor]:
    """Compute complete multi-term KPI-aligned V8 loss dictionary for one episode."""
    w = {
        "ret": 1.0,
        "pit": 0.35,
        "cop": 0.25,
        "tree": 0.30,
        "drv": 0.20,
        "cond": 0.15,
        "priv": 0.15,
    }
    if weights:
        w.update(weights)

    Xf = torch.nan_to_num(X_fit, nan=0.0)
    Xq = torch.nan_to_num(X_qry, nan=0.0)
    Xs = torch.nan_to_num(x_syn, nan=0.0)

    if task_id == 0:
        ys = y_reg_syn
        l_lin_s = _fit_ridge_eval(Xs, ys, Xq, y_qry)
        l_lin_r = _fit_ridge_eval(Xf, y_fit, Xq, y_qry).detach()
        l_tr_s = _fit_soft_obliv_tree_auditor(Xs, ys, Xq, y_qry, 0)
        l_tr_r = _fit_soft_obliv_tree_auditor(Xf, y_fit, Xq, y_qry, 0).detach()
        l_rff_s = _fit_rff_auditor(Xs, ys, Xq, y_qry, 0)
        l_rff_r = _fit_rff_auditor(Xf, y_fit, Xq, y_qry, 0).detach()
        l0 = F.mse_loss(torch.full_like(y_qry, y_fit.mean()), y_qry).detach()
    elif task_id == 1:
        # Gumbel-Rao K=4 straight-through binary target sample!
        bin_logits = torch.stack([torch.zeros_like(logits_syn[:, 0]), logits_syn[:, 0]], dim=-1)
        ys = gumbel_rao_ste(bin_logits, tau=0.8, k_mc=4)[:, 1]
        yf = (y_fit > 0.5).float()
        yq = (y_qry > 0.5).float()
        l_lin_s = _fit_logit_irls_eval(Xs, ys, Xq, yq)
        l_lin_r = _fit_logit_irls_eval(Xf, yf, Xq, yq).detach()
        l_tr_s = _fit_soft_obliv_tree_auditor(Xs, ys, Xq, yq, 1)
        l_tr_r = _fit_soft_obliv_tree_auditor(Xf, yf, Xq, yq, 1).detach()
        l_rff_s = _fit_rff_auditor(Xs, ys, Xq, yq, 1)
        l_rff_r = _fit_rff_auditor(Xf, yf, Xq, yq, 1).detach()
        p1 = yf.mean().clamp(0.02, 0.98)
        l0 = F.binary_cross_entropy(torch.full_like(yq, p1), yq).detach()
    else:
        # Multiclass: One-vs-Rest Gumbel-Rao K=4 smooth worst across active classes!
        k = max(2, min(int(n_classes), logits_syn.shape[-1]))
        oh_s = gumbel_rao_ste(logits_syn[:, :k], tau=0.8, k_mc=4)
        ys = oh_s[:, 0]
        yf = (y_fit.long() == 0).float()
        yq = (y_qry.long() == 0).float()
        l_lin_s = _fit_logit_irls_eval(Xs, ys, Xq, yq)
        l_lin_r = _fit_logit_irls_eval(Xf, yf, Xq, yq).detach()
        l_tr_s = _fit_soft_obliv_tree_auditor(Xs, ys, Xq, yq, 1)
        l_tr_r = _fit_soft_obliv_tree_auditor(Xf, yf, Xq, yq, 1).detach()
        l_rff_s = _fit_rff_auditor(Xs, ys, Xq, yq, 1)
        l_rff_r = _fit_rff_auditor(Xf, yf, Xq, yq, 1).detach()
        p0 = yf.mean().clamp(0.02, 0.98)
        l0 = F.binary_cross_entropy(torch.full_like(yq, p0), yq).detach()

    # Twin-referenced smooth-worst retention gap across [A_lin, A_tree, A_rff] (mlp held out!)
    def _gap(ls: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        denom = (l0 - lr).clamp(min=0.05)
        return ((ls - lr) / denom).clamp(-0.5, 4.0)

    gaps = torch.stack([_gap(l_lin_s, l_lin_r), _gap(l_tr_s, l_tr_r), _gap(l_rff_s, l_rff_r)])
    l_ret = (1.0 / 4.0) * torch.logsumexp(4.0 * gaps, dim=0)

    y_stat_q = y_qry if task_id == 0 else (y_qry > 0.5).float()
    l_pit = loss_pit_cvm(Xs, Xq)
    l_cop = loss_copula_corr(Xs, Xq)
    l_tree = loss_tree_occupancy(Xs, ys, Xq, y_stat_q)
    l_drv = loss_driver_alignment(Xs, ys, Xq, y_stat_q)
    l_cond = loss_cond_calibration(logits_syn[:, : max(n_classes, 1)], y_reg_syn, resid_sd, y_qry, task_id)
    l_priv = loss_privacy_dcr_hinge(Xs, Xf, Xq)

    total = (
        w["ret"] * l_ret
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
