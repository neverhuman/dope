"""Differentiable TSTR surrogate auditors (`A_lin`, `A_tree`, `A_rff`) for `s3_v8_losses.py`."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _fit_ridge_eval(
    Xs: torch.Tensor, ys: torch.Tensor, Xq: torch.Tensor, yq: torch.Tensor, lam: float = 2.0,
) -> torch.Tensor:
    """Closed-form differentiable ridge regression MSE on query `(Xq, yq)`."""
    n, p = Xs.shape
    Xa = torch.cat([Xs, torch.ones(n, 1, device=Xs.device, dtype=Xs.dtype)], dim=1)
    Xqa = torch.cat([Xq, torch.ones(Xq.shape[0], 1, device=Xq.device, dtype=Xq.dtype)], dim=1)
    reg = torch.eye(p + 1, device=Xs.device, dtype=Xs.dtype) * lam
    reg[-1, -1] = 1e-2
    w = torch.linalg.solve(Xa.T @ Xa + reg, Xa.T @ ys.unsqueeze(-1))
    return F.mse_loss((Xqa @ w).squeeze(-1), yq)


def _fit_logit_irls_eval(
    Xs: torch.Tensor, ps: torch.Tensor, Xq: torch.Tensor, yq: torch.Tensor, lam: float = 2.0, steps: int = 3,
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
    Xf_guide: torch.Tensor | None = None,
    yf_guide: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable depth-3 soft oblivious-tree ensemble auditor (`A_tree`), reg/bin/all-K mc.

    When FIT-only `(Xf_guide, yf_guide)` are provided on wide tables (`p > 6`), half the trees
    use FIT-only top-|corr| signal columns & FIT empirical quantiles (zero query leakage!) so
    `A_tree` measures real nonlinear transfer instead of pure noise splits.
    """
    _, p = Xs.shape
    gen = torch.Generator(device=Xs.device)
    gen.manual_seed(seed + p * 31)
    cols = torch.randint(0, max(p, 1), (n_trees, 3), generator=gen, device=Xs.device)
    thrs = torch.empty((n_trees, 3), device=Xs.device, dtype=Xs.dtype).uniform_(-1.0, 1.0, generator=gen)

    if Xf_guide is not None and yf_guide is not None and p > 6 and Xf_guide.shape[0] >= 12 and n_trees >= 4:
        with torch.no_grad():
            xf_c = Xf_guide - Xf_guide.mean(dim=0, keepdim=True)
            xf_sd = xf_c.std(dim=0, keepdim=True).clamp(min=1e-4)
            y_ref = yf_guide.float()
            if y_ref.ndim == 2:
                yc = y_ref - y_ref.mean(dim=0, keepdim=True)
                ysd = yc.std(dim=0, keepdim=True).clamp(min=1e-4)
                corr = ((xf_c / xf_sd).T @ (yc / ysd)).abs().max(dim=1).values / float(Xf_guide.shape[0])
            else:
                yc = y_ref - y_ref.mean()
                ysd = yc.std().clamp(min=1e-4)
                corr = (((xf_c / xf_sd) * (yc / ysd)[:, None]).mean(dim=0)).abs()
            k_top = min(p, 8)
            top_cols = torch.topk(corr, k_top).indices  # [k_top]
            n_guided = n_trees // 2
            pick = torch.randint(0, k_top, (n_guided, 3), generator=gen, device=Xs.device)
            g_cols = top_cols[pick]  # [n_guided, 3]
            q_levels = torch.tensor([0.25, 0.50, 0.75], device=Xs.device, dtype=Xs.dtype)
            q_vals = torch.quantile(Xf_guide[:, top_cols], q_levels, dim=0)  # [3, k_top]
            q_pick = torch.randint(0, 3, (n_guided, 3), generator=gen, device=Xs.device)
            g_thrs = q_vals[q_pick, pick]
            cols = torch.cat([g_cols, cols[n_guided:]], dim=0)
            thrs = torch.cat([g_thrs, thrs[n_guided:]], dim=0)

    ws = _leaf_weights_3d(Xs, cols, thrs)  # [Ns, T, 8]
    wq = _leaf_weights_3d(Xq, cols, thrs)  # [Nq, T, 8]
    denom = ws.sum(dim=0) + 4.0  # [T, 8]
    if task_id == 2:
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
