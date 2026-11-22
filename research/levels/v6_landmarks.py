"""FIT-only permutation-invariant landmark extractor & differentiable IRLS/MMD losses for DOPE v2.0.0."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

TASK_IDS = {"regression": 0, "binary": 1, "multiclass": 2, "multilabel": 3}
N_KNOTS = 17
LANDMARK_DIM = 48
MAX_CLASSES_SUPPORT = 32


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z * 0.7071067811865475))


def _normal_ppf(u: torch.Tensor) -> torch.Tensor:
    return 1.4142135623730951 * torch.erfinv((2.0 * u - 1.0).clamp(-0.99995, 0.99995))


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = a - a.mean(dim=0, keepdim=True)
    bc = b - b.mean(dim=0, keepdim=True)
    num = (ac * bc).sum(dim=0)
    den = (ac.pow(2).sum(dim=0).sqrt() * bc.pow(2).sum(dim=0).sqrt()).clamp_min(1e-6)
    return (num / den).clamp(-1.0, 1.0)


@torch.no_grad()
def extract_rich_landmarks(
    X_fit: torch.Tensor,
    y_fit: torch.Tensor,
    miss_fit: torch.Tensor | None,
    task: str,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 48-dim FIT-only permutation-invariant landmarks per column + global class priors."""
    xf, yf = X_fit.float(), y_fit.float()
    n, p = xf.shape
    dev = xf.device
    mf = miss_fit.float() if miss_fit is not None else torch.zeros_like(xf)

    q_grid = torch.linspace(0.0, 1.0, N_KNOTS, device=dev)
    qs = torch.quantile(xf, q_grid, dim=0).T
    mean = xf.mean(dim=0)
    std = xf.std(dim=0, unbiased=False).clamp_min(1e-5)
    z = ((xf - mean) / std).clamp(-6.0, 6.0)
    skew = z.pow(3).mean(dim=0).clamp(-6.0, 6.0) / 6.0
    kurt = (z.pow(4).mean(dim=0) - 3.0).clamp(-4.0, 12.0) / 12.0
    miss_rate = mf.mean(dim=0).clamp(0.0, 1.0)

    u_counts = torch.tensor([float(torch.unique(xf[:, j]).numel()) for j in range(p)], device=dev)
    u_ratio = (u_counts / float(max(n, 1))).clamp(0.0, 1.0)
    is_disc16 = (u_counts <= 16.0).float()
    is_disc64 = (u_counts <= 64.0).float()
    mode_mass = torch.zeros(p, device=dev)
    for j in range(p):
        _, cnts = torch.unique(xf[:, j], return_counts=True)
        mode_mass[j] = cnts.max().float() / float(max(n, 1))

    views = [z, (z * z).clamp_max(16.0), torch.tanh(z), F.relu(z), F.relu(-z)]
    priors = torch.zeros(MAX_CLASSES_SUPPORT, device=dev)
    if task == "multiclass":
        k = min(max(n_classes, 2), MAX_CLASSES_SUPPORT)
        y_oh = F.one_hot(yf.long().clamp(0, k - 1), num_classes=k).float()
        pr = y_oh.mean(dim=0).clamp_min(1e-4)
        priors[:k] = pr
        sup_feats = []
        for v in views:
            vc = v - v.mean(dim=0, keepdim=True)
            vvar = vc.pow(2).mean(dim=0).clamp_min(1e-6)
            cm = (y_oh.T @ vc) / y_oh.sum(dim=0, keepdim=True).T.clamp_min(1.0)
            eta2 = ((pr.unsqueeze(1) * cm.pow(2)).sum(dim=0) / vvar).clamp(0.0, 1.0).sqrt()
            max_c = torch.zeros(p, device=dev)
            for c in range(k):
                max_c = torch.maximum(max_c, _safe_corr(v, y_oh[:, c:c + 1]).abs())
            sup_feats.extend([eta2, max_c])
        A = z.T @ z + (0.25 * n + 2.0) * torch.eye(p, device=dev)
        w_lin = torch.linalg.solve(A, z.T @ (y_oh - pr.unsqueeze(0)))
        lin_imp = w_lin.abs().mean(dim=1).clamp_max(4.0) / 4.0
        y_sig = (y_oh - pr.unsqueeze(0))[:, 0:1]
        y_g = torch.tensor([float(k) / 32.0, float(-(pr * pr.log()).sum() / math.log(k + 1e-6))], device=dev)
    elif task == "multilabel":
        ym = yf if yf.ndim == 2 else yf.unsqueeze(1)
        nl = min(ym.shape[1], MAX_CLASSES_SUPPORT)
        ym = ym[:, :nl]
        pr = ym.mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
        priors[:nl] = pr
        sup_feats = []
        for v in views:
            cs = torch.stack([_safe_corr(v, ym[:, l:l + 1]).abs() for l in range(nl)], dim=0)
            sup_feats.extend([cs.mean(dim=0), cs.max(dim=0).values])
        A = z.T @ z + (0.25 * n + 2.0) * torch.eye(p, device=dev)
        w_lin = torch.linalg.solve(A, z.T @ (ym - pr.unsqueeze(0)))
        lin_imp = w_lin.abs().mean(dim=1).clamp_max(4.0) / 4.0
        y_sig = (ym - pr.unsqueeze(0)).mean(dim=1, keepdim=True)
        y_g = torch.tensor([float(nl) / 32.0, float(pr.mean())], device=dev)
    else:
        ys = yf.reshape(-1, 1)
        if task == "binary":
            p1 = float(ys.mean().clamp(1e-4, 1.0 - 1e-4))
            priors[0], priors[1] = 1.0 - p1, p1
            y_g = torch.tensor([p1, 4.0 * p1 * (1.0 - p1)], device=dev)
        else:
            priors[0] = float(ys.mean().clamp(-6.0, 6.0))
            priors[1] = float(ys.std(unbiased=False).clamp(1e-4, 6.0))
            y_g = torch.tensor([priors[0].item() / 6.0, priors[1].item() / 6.0], device=dev)
        sup_feats = []
        for v in views:
            c_v = _safe_corr(v, ys)
            sup_feats.extend([c_v, c_v.abs()])
        A = z.T @ z + (0.2 * n + 2.0) * torch.eye(p, device=dev)
        w_lin = torch.linalg.solve(A, z.T @ (ys - ys.mean()))
        lin_imp = w_lin.squeeze(1).abs().clamp_max(4.0) / 4.0
        y_sig = ys - ys.mean()

    corr_m = (z.T @ z) / float(max(n, 1))
    corr_m.fill_diagonal_(0.0)
    abs_c = corr_m.abs()
    mean_c = abs_c.mean(dim=1)
    max_c_feat = abs_c.max(dim=1).values if p > 1 else torch.zeros(p, device=dev)
    pair_max = torch.zeros(p, device=dev)
    if p > 1:
        top_idx = torch.topk(lin_imp + sup_feats[1], k=min(p, 12)).indices
        for j in range(p):
            prods = (z[:, j:j + 1] * z[:, top_idx]).clamp(-16.0, 16.0)
            pc = _safe_corr(prods, y_sig).abs()
            pc[top_idx == j] = 0.0
            pair_max[j] = pc.max() if pc.numel() > 0 else 0.0

    log_n = torch.full((p,), math.log(max(n, 2)) / 10.0, device=dev)
    log_p = torch.full((p,), math.log(max(p, 2)) / 5.0, device=dev)
    g0 = torch.full((p,), float(y_g[0]), device=dev)
    g1 = torch.full((p,), float(y_g[1]), device=dev)
    iqr = (qs[:, 12] - qs[:, 4]).clamp(0.0, 12.0) / 6.0
    span90 = (qs[:, 15] - qs[:, 1]).clamp(0.0, 12.0) / 6.0
    tail_hi = (qs[:, 16] - qs[:, 14]).clamp(0.0, 6.0) / 6.0
    tail_lo = (qs[:, 2] - qs[:, 0]).clamp(0.0, 6.0) / 6.0
    feats = torch.cat(
        [
            qs / 6.0,
            torch.stack([
                mean.clamp(-6.0, 6.0) / 6.0, std.clamp(0.0, 6.0) / 6.0,
                skew, kurt, miss_rate, u_ratio, is_disc16, is_disc64, mode_mass,
                *sup_feats, lin_imp, mean_c, max_c_feat, pair_max,
                log_n, log_p, g0, g1, iqr, span90, tail_hi, tail_lo,
            ], dim=1),
        ],
        dim=1,
    )
    return torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0), priors, qs


def _rbf_mmd_loss(X_real: torch.Tensor, X_syn: torch.Tensor) -> torch.Tensor:
    """Multi-bandwidth RBF-MMD realism loss (drives C2ST AUC down directly)."""
    nr, ns = min(X_real.shape[0], 128), min(X_syn.shape[0], 128)
    xr, xs = X_real[:nr].float(), X_syn[:ns].float()
    ps = float(max(xr.shape[1], 1))
    d_rr = torch.cdist(xr, xr).pow(2) / ps
    d_ss = torch.cdist(xs, xs).pow(2) / ps
    d_rs = torch.cdist(xr, xs).pow(2) / ps
    mmd = torch.tensor(0.0, device=xr.device)
    for bw in (0.25, 1.0, 4.0):
        mmd = mmd + (torch.exp(-d_rr / bw).mean() + torch.exp(-d_ss / bw).mean() - 2.0 * torch.exp(-d_rs / bw).mean()).clamp_min(0.0)
    return mmd / 3.0


def _irls_downstream_eval(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_ev: torch.Tensor,
    y_ev: torch.Tensor,
    task: str,
    n_classes: int,
    alpha: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable FP32 Ridge (reg) or 3-step IRLS Logistic/Softmax (bin/mc/ml) downstream learner."""
    nt, p = X_tr.shape
    ne = X_ev.shape[0]
    dev = X_tr.device
    Dt = torch.cat([X_tr.float(), torch.ones((nt, 1), device=dev)], dim=1)
    De = torch.cat([X_ev.float(), torch.ones((ne, 1), device=dev)], dim=1)
    eye = torch.eye(p + 1, device=dev)
    eye[-1, -1] = 1e-3
    lam = alpha * nt + 1.0

    if task == "regression":
        yt, ye = y_tr.float().reshape(-1, 1), y_ev.float().reshape(-1, 1)
        w = torch.linalg.solve(Dt.T @ Dt + lam * eye, Dt.T @ yt)
        loss = F.smooth_l1_loss((De @ w).clamp(-8.0, 8.0), ye, beta=1.0)
        null_loss = F.smooth_l1_loss(torch.full_like(ye, float(yt.mean())), ye, beta=1.0)
        return loss, null_loss, w[:p, 0].abs()

    if task == "binary":
        yt = y_tr.float().reshape(-1, 1).clamp(1e-4, 1.0 - 1e-4)
        ye = y_ev.float().reshape(-1, 1).clamp(0.0, 1.0)
        p1 = yt.mean().clamp(1e-3, 1.0 - 1e-3)
        w = torch.zeros((p + 1, 1), device=dev)
        w[-1, 0] = torch.log(p1 / (1.0 - p1))
        for _ in range(3):
            pr = torch.sigmoid((Dt @ w).clamp(-12.0, 12.0))
            h = (pr * (1.0 - pr)).clamp_min(0.02)
            w = w - 0.85 * torch.linalg.solve((Dt * h).T @ Dt + lam * eye, Dt.T @ (pr - yt) + lam * (eye @ w)).clamp(-3.0, 3.0)
        loss = F.binary_cross_entropy_with_logits((De @ w).clamp(-12.0, 12.0), ye)
        null_loss = F.binary_cross_entropy_with_logits(torch.full_like(ye, float(torch.log(p1 / (1.0 - p1)))), ye)
        return loss, null_loss, w[:p, 0].abs()

    if task == "multilabel":
        yt = y_tr.float().clamp(1e-4, 1.0 - 1e-4)
        ye = y_ev.float().clamp(0.0, 1.0)
        if yt.ndim == 1:
            yt, ye = yt.unsqueeze(1), ye.unsqueeze(1)
        pr = yt.mean(dim=0, keepdim=True).clamp(1e-3, 1.0 - 1e-3)
        w = torch.zeros((p + 1, yt.shape[1]), device=dev)
        w[-1, :] = torch.log(pr / (1.0 - pr)).squeeze(0)
        for _ in range(2):
            prob = torch.sigmoid((Dt @ w).clamp(-12.0, 12.0))
            h = (prob * (1.0 - prob)).mean(dim=1, keepdim=True).clamp_min(0.04)
            w = w - 0.85 * torch.linalg.solve((Dt * h).T @ Dt + lam * eye, Dt.T @ (prob - yt) + lam * (eye @ w)).clamp(-3.0, 3.0)
        loss = F.binary_cross_entropy_with_logits((De @ w).clamp(-12.0, 12.0), ye)
        null_loss = F.binary_cross_entropy_with_logits(torch.log(pr / (1.0 - pr)).expand_as(ye), ye)
        return loss, null_loss, w[:p].abs().mean(dim=1)

    k = min(max(n_classes, 2), MAX_CLASSES_SUPPORT)
    yt_p = y_tr.float() if y_tr.ndim == 2 else F.one_hot(y_tr.long().clamp(0, k - 1), num_classes=k).float()
    ye_i = y_ev.long().clamp(0, k - 1) if y_ev.ndim == 1 else y_ev.argmax(dim=-1)
    pr = yt_p.mean(dim=0).clamp_min(1e-3)
    w = torch.zeros((p + 1, k), device=dev)
    w[-1, :] = torch.log(pr)
    for _ in range(3):
        prob = F.softmax((Dt @ w).clamp(-12.0, 12.0), dim=-1)
        H = 0.25 * (Dt.T @ Dt) + lam * eye
        w = w - 0.9 * torch.linalg.solve(H, Dt.T @ (prob - yt_p) + lam * (eye @ w)).clamp(-3.0, 3.0)
    loss = F.cross_entropy((De @ w).clamp(-12.0, 12.0), ye_i)
    null_loss = F.cross_entropy(torch.log(pr).unsqueeze(0).expand(ne, -1), ye_i)
    return loss, null_loss, w[:p].abs().mean(dim=1)
