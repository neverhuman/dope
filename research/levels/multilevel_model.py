"""Unified 3-Level Landmark- and Descriptor-Conditioned Tabular Model (DOPE-MLC-v2).

Single unified model supporting three nested encoding tiers:
- Medium (4,096 coords / Macro tier): conditioned on 9 Macro landmarks + Macro table descriptors
- High   (8,192 coords / Larger tier): adds 9 Larger interaction/basis/tree landmarks (18 total) + spectral descriptors
- Ultra  (10,240 coords / Fine tier): adds 5 Fine GBDT/local/neural landmarks (23 total) + Chatterjee xi spectrum
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .descriptors import compute_table_descriptors, extract_report_descriptors

LEVEL_WIDTHS: Dict[str, int] = {"medium": 4096, "high": 8192, "ultra": 10240}
MAX_COLS = 64
QUANTILE_KNOTS = np.linspace(0.01, 0.99, 15)


class LandmarkHyperRouter(nn.Module):
    """Maps level-masked landmark + data descriptive vectors (140 dims) to structural gates and latents."""

    def __init__(self, in_dim: int = 140, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.level_embed = nn.Embedding(3, hidden_dim)
        self.input_norm = nn.LayerNorm(in_dim)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
        )
        self.gate_head = nn.Linear(hidden_dim, 10)
        self.task_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.dist_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.landmark_recon_head = nn.Linear(hidden_dim, 23)
        nn.init.zeros_(self.gate_head.weight)
        with torch.no_grad():
            self.gate_head.bias.copy_(torch.tensor([0.0, 0.4, 0.3, 0.3, 0.2, 0.8, 0.5, 1.2, 0.5, -0.2], dtype=torch.float32))

    def forward(self, desc_vec: torch.Tensor, level: str) -> Dict[str, torch.Tensor]:
        lvl_idx = {"medium": 0, "high": 1, "ultra": 2}[level]
        squeeze = desc_vec.ndim == 1
        if squeeze:
            desc_vec = desc_vec.unsqueeze(0)
        lvl_t = torch.full((desc_vec.shape[0],), lvl_idx, dtype=torch.long, device=desc_vec.device)
        h = self.backbone(self.input_norm(desc_vec)) + self.level_embed(lvl_t)
        raw_gates = self.gate_head(h)
        is_high = 1.0 if level in ("high", "ultra") else 0.0
        gates = {
            "ridge_alpha": F.softplus(raw_gates[:, 0]) + (0.35 if level == "medium" else 0.15 if level == "high" else 0.08),
            "monotone_w": torch.sigmoid(raw_gates[:, 1]) * (0.55 if level == "medium" else 0.65),
            "hinge_w": torch.sigmoid(raw_gates[:, 2]) * 0.55 * is_high,
            "pair_w": torch.sigmoid(raw_gates[:, 3]) * 0.60 * is_high,
            "rff_w": torch.sigmoid(raw_gates[:, 4]) * 0.45 * is_high,
            "tree_w": torch.sigmoid(raw_gates[:, 5]) * (0.20 if level == "medium" else 0.70 if level == "high" else 1.00),
            "neural_w": torch.sigmoid(raw_gates[:, 6]) * (0.35 if level == "medium" else 0.65 if level == "high" else 1.00),
            "copula_retention": torch.sigmoid(raw_gates[:, 7]) * (0.68 if level == "medium" else 0.90 if level == "high" else 0.99),
            "regime_sharpness": torch.sigmoid(raw_gates[:, 8]) * (0.35 if level == "medium" else 0.75 if level == "high" else 1.00),
            "noise_scale": torch.sigmoid(raw_gates[:, 9]) * (0.32 if level == "medium" else 0.20 if level == "high" else 0.12),
        }
        task_lat, dist_lat, lm_pred = torch.tanh(self.task_latent_head(h)), torch.tanh(self.dist_latent_head(h)), torch.tanh(self.landmark_recon_head(h))
        if squeeze:
            gates = {k: v.squeeze(0) for k, v in gates.items()}
            task_lat, dist_lat, lm_pred = task_lat.squeeze(0), dist_lat.squeeze(0), lm_pred.squeeze(0)
        return {"gates": gates, "task_latent": task_lat, "dist_latent": dist_lat, "landmark_pred": lm_pred}


class ConditionalHyperTargetNet(nn.Module):
    """Differentiable conditional target predictor P_phi(Y | X, z_task, z_landmark)."""

    def __init__(self, max_cols: int = MAX_COLS, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.max_cols = max_cols
        self.film_gamma = nn.Linear(latent_dim, max_cols)
        self.film_beta = nn.Linear(latent_dim, max_cols)
        self.fc1 = nn.Linear(max_cols + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, 1)
        self.col_gate_head = nn.Linear(latent_dim + 17, 1)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, X_pad: torch.Tensor, task_latent: torch.Tensor, col_mask: torch.Tensor) -> torch.Tensor:
        t_exp = (task_latent.unsqueeze(0) if task_latent.ndim == 1 else task_latent).expand(X_pad.shape[0], -1)
        X_mod = (X_pad * (1.0 + 0.3 * torch.tanh(self.film_gamma(t_exp))) + 0.1 * torch.tanh(self.film_beta(t_exp))) * col_mask.unsqueeze(0)
        h = F.silu(self.fc1(torch.cat([X_mod, t_exp], dim=-1)))
        return self.out_head(F.silu(self.fc2(h)) + h).squeeze(-1)


class CovariateRefinementNet(nn.Module):
    """Learned residual denoiser that refines copula-sampled covariates in normal-score space."""

    def __init__(self, max_cols: int = MAX_COLS, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_cols + latent_dim, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, max_cols),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, X_pad: torch.Tensor, dist_latent: torch.Tensor, col_mask: torch.Tensor) -> torch.Tensor:
        d_exp = (dist_latent.unsqueeze(0) if dist_latent.ndim == 1 else dist_latent).expand(X_pad.shape[0], -1)
        return X_pad + 0.15 * torch.tanh(self.net(torch.cat([X_pad, d_exp], dim=-1))) * col_mask.unsqueeze(0)


class MultiLevelLandmarkModel(nn.Module):
    """Single Unified Model with 3 Nested Feature & Compression Tiers (Medium, High, Ultra)."""

    def __init__(self, max_cols: int = MAX_COLS, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.max_cols, self.latent_dim = max_cols, latent_dim
        self.router = LandmarkHyperRouter(in_dim=140, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.target_net = ConditionalHyperTargetNet(max_cols=max_cols, latent_dim=latent_dim, hidden_dim=hidden_dim // 2)
        self.cov_net = CovariateRefinementNet(max_cols=max_cols, latent_dim=latent_dim, hidden_dim=hidden_dim // 2)

    @staticmethod
    def build_conditioning_vector(
        report: Dict[str, Any], X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, level: str, device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, np.ndarray]]:
        rep_desc = extract_report_descriptors(report, level=level)
        tab_desc = compute_table_descriptors(X_train_scaled, y_train_scaled, level=level)
        vec140 = np.concatenate([rep_desc["vector116"], tab_desc["table_vec24"]]).astype(np.float32)
        return torch.tensor(vec140, dtype=torch.float32, device=device), rep_desc, tab_desc

    def forward_conditioning(
        self, X_train: torch.Tensor, y_train: torch.Tensor, X_query: torch.Tensor, y_query: torch.Tensor,
        desc_vec140: torch.Tensor, emp_drivers: torch.Tensor, task: str, level: str,
    ) -> Dict[str, torch.Tensor]:
        device = X_train.device
        n_tr, p = X_train.shape
        n_qr, p_use = X_query.shape[0], min(p, self.max_cols)
        X_tr, X_qr = X_train[:, :p_use], X_query[:, :p_use]

        routed = self.router(desc_vec140, level=level)
        gates, task_lat, dist_lat, lm_pred = routed["gates"], routed["task_latent"], routed["dist_latent"], routed["landmark_pred"]

        phi_tr, phi_qr = _build_differentiable_basis(X_tr, gates, level=level), _build_differentiable_basis(X_qr, gates, level=level)
        gram = (phi_tr.T @ phi_tr) / float(max(n_tr, 1))
        eye = torch.eye(phi_tr.shape[1], device=device, dtype=X_tr.dtype)
        eye[0, 0] = 1e-6
        w_basis = torch.linalg.solve(gram + gates["ridge_alpha"].clamp(1e-3, 50.0) * eye, (phi_tr.T @ y_train) / float(max(n_tr, 1)))
        pred_basis_qr = phi_qr @ w_basis

        X_tr_pad = torch.zeros((n_tr, self.max_cols), device=device, dtype=X_tr.dtype)
        X_qr_pad = torch.zeros((n_qr, self.max_cols), device=device, dtype=X_qr.dtype)
        col_mask = torch.zeros(self.max_cols, device=device, dtype=X_tr.dtype)
        X_tr_pad[:, :p_use], X_qr_pad[:, :p_use], col_mask[:p_use] = X_tr, X_qr, 1.0

        res_tr = self.target_net(X_tr_pad, task_lat, col_mask)
        res_qr = self.target_net(X_qr_pad, task_lat, col_mask)
        y_hat_qr = pred_basis_qr + gates["neural_w"] * (res_qr - res_tr.mean())

        if task == "classification":
            p_bar = y_train.mean().clamp(0.02, 0.98)
            logits_qr = torch.log(p_bar / (1.0 - p_bar)) + 2.8 * (y_hat_qr - p_bar)
            cond_loss = 0.65 * F.binary_cross_entropy_with_logits(logits_qr, y_query) + 0.35 * F.mse_loss(torch.sigmoid(logits_qr), y_query)
        else:
            cond_loss = 0.60 * F.huber_loss(y_hat_qr, y_query, delta=1.0) + 0.40 * F.mse_loss(y_hat_qr, y_query)

        X_cov_ref = self.cov_net(X_tr_pad, dist_lat, col_mask)[:, :p_use]
        cov_tr, cov_qr = (X_cov_ref.T @ X_cov_ref) / float(max(n_tr, 1)), (X_qr.T @ X_qr) / float(max(n_qr, 1))
        off_mask = 1.0 - torch.eye(p_use, device=device)
        target_cov = cov_qr * (gates["copula_retention"] * off_mask + torch.eye(p_use, device=device))
        dist_loss = F.mse_loss(cov_tr, target_cov) + 0.25 * F.mse_loss(X_cov_ref.mean(dim=0), X_qr.mean(dim=0))

        lin_w = torch.abs(w_basis[1 : 1 + p_use])
        film_g = torch.abs(1.0 + 0.3 * torch.tanh(self.target_net.film_gamma(task_lat.unsqueeze(0)).squeeze(0)[:p_use]))
        pred_imp = lin_w * film_g
        driver_loss = F.l1_loss(pred_imp / (pred_imp.sum() + 1e-8), emp_drivers[:p_use] / (emp_drivers[:p_use].sum() + 1e-8))

        true_gains, true_mask = desc_vec140[:23], desc_vec140[23:46] > 0.5
        lm_loss = F.huber_loss(lm_pred[true_mask], true_gains[true_mask]) if true_mask.any() else lm_pred.sum() * 0.0
        total_loss = cond_loss + 0.25 * dist_loss + 0.20 * driver_loss + 0.10 * lm_loss
        return {
            "total": total_loss, "cond_y_loss": cond_loss.detach(), "dist_x_loss": dist_loss.detach(),
            "driver_loss": driver_loss.detach(), "landmark_loss": lm_loss.detach(),
            "gates": {k: float(v.detach().item()) for k, v in gates.items()},
        }

    @torch.no_grad()
    def encode_table(
        self, X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, report: Dict[str, Any],
        *, task: str = "regression", level: str = "ultra",
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        vec140, rep_desc, tab_desc = self.build_conditioning_vector(report, X_train_scaled, y_train_scaled, level=level, device=device)
        routed = self.router(vec140, level=level)
        gates = {k: float(v.item()) for k, v in routed["gates"].items()}
        task_lat = routed["task_latent"].detach().cpu().numpy().astype(np.float16)
        dist_lat = routed["dist_latent"].detach().cpu().numpy().astype(np.float16)

        X, y = np.asarray(X_train_scaled, dtype=np.float64), np.asarray(y_train_scaled, dtype=np.float64).reshape(-1)
        n, p = X.shape
        p_use = min(p, self.max_cols)
        X_use = X[:, :p_use]

        n_knots = 17 if level == "medium" else 33 if level == "high" else 65
        knot_grid = np.linspace(0.005 if level != "medium" else 0.01, 0.995 if level != "medium" else 0.99, n_knots)
        col_knots = np.quantile(X_use, knot_grid, axis=0).T.astype(np.float16)
        col_means, col_stds = np.mean(X_use, axis=0).astype(np.float16), np.maximum(np.std(X_use, axis=0), 1e-6).astype(np.float16)
        is_binary_col = np.array([len(np.unique(X_use[:, j])) <= 2 for j in range(p_use)], dtype=bool)
        binary_probs = np.array([float(np.mean(X_use[:, j] > np.mean(X_use[:, j]))) if is_binary_col[j] else 0.0 for j in range(p_use)], dtype=np.float16)
        binary_low_high = np.array([[float(np.min(X_use[:, j])), float(np.max(X_use[:, j]))] if is_binary_col[j] else [0.0, 0.0] for j in range(p_use)], dtype=np.float16)
        discrete_support: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        max_disc = 8 if level == "medium" else 16 if level == "high" else 32
        for j in range(p_use):
            if not is_binary_col[j]:
                u_vals, u_counts = np.unique(X_use[:, j], return_counts=True)
                if 2 < len(u_vals) <= max_disc:
                    discrete_support[j] = (u_vals.astype(np.float16), np.cumsum(u_counts.astype(np.float64) / float(max(n, 1))).astype(np.float16))

        Z_norm = _to_normal_scores(X_use)
        corr = (Z_norm.T @ Z_norm) / float(max(n, 1))
        np.fill_diagonal(corr, 1.0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        ret = max(float(gates["copula_retention"]), 0.72 if level == "medium" else 0.92 if level == "high" else 0.99)
        eigvals, eigvecs = np.linalg.eigh(ret * corr + (1.0 - ret) * np.eye(p_use))
        eigvals, eigvecs = np.maximum(eigvals[::-1], 1e-5), eigvecs[:, ::-1]
        rank_use = min(p_use, 8 if level == "medium" else 20 if level == "high" else p_use)
        chol_factor = (eigvecs[:, :rank_use] * np.sqrt(eigvals[:rank_use])[None, :]).astype(np.float16)
        diag_res_std = np.sqrt(np.maximum(1.0 - np.sum(chol_factor.astype(np.float64) ** 2, axis=1), 1e-4)).astype(np.float16)

        n_regimes = 1 if level == "medium" else 4 if level == "high" else 8
        regime_weights, regime_means, regime_scales = _fit_copula_regimes(Z_norm, n_regimes=n_regimes, sharpness=gates["regime_sharpness"])
        target_bundle = _fit_hierarchical_target_bundle(
            X_use, y, gates=gates, col_prior=rep_desc["column_prior_importance"][:p_use],
            emp_drivers=tab_desc["empirical_driver_weights"][:p_use], task=task, level=level,
            target_net=self.target_net, task_lat_tensor=routed["task_latent"],
        )
        width = LEVEL_WIDTHS[level]
        code_vec = _pack_code_vector(width=width, col_knots=col_knots, chol_factor=chol_factor, target_bundle=target_bundle, task_lat=task_lat, dist_lat=dist_lat)
        return {
            "level": level, "task": task, "code_width": width, "code_fp16": code_vec.astype(np.float16),
            "p_use": p_use, "n_train": n, "gates": gates, "task_latent": task_lat, "dist_latent": dist_lat,
            "knot_grid": knot_grid.astype(np.float16), "col_knots": col_knots, "col_means": col_means,
            "col_stds": col_stds, "is_binary_col": is_binary_col, "binary_probs": binary_probs,
            "binary_low_high": binary_low_high, "discrete_support": discrete_support,
            "chol_factor": chol_factor, "diag_res_std": diag_res_std,
            "regime_weights": regime_weights.astype(np.float16), "regime_means": regime_means.astype(np.float16),
            "regime_scales": regime_scales.astype(np.float16), "target_bundle": target_bundle,
        }

    @torch.no_grad()
    def generate_synthetic(
        self, encoding: Dict[str, Any], n_rows: Optional[int] = None, *, seed: int = 20261023
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, p_use = int(encoding["n_train"] if n_rows is None else n_rows), int(encoding["p_use"])
        level, task = encoding["level"], encoding["task"]
        rng = np.random.default_rng(seed)

        reg_w = np.asarray(encoding["regime_weights"], dtype=np.float64)
        reg_w = reg_w / np.sum(reg_w)
        reg_m, reg_s = np.asarray(encoding["regime_means"], dtype=np.float64), np.asarray(encoding["regime_scales"], dtype=np.float64)
        chol, diag_std = np.asarray(encoding["chol_factor"], dtype=np.float64), np.asarray(encoding["diag_res_std"], dtype=np.float64)

        reg_idx = rng.choice(len(reg_w), size=n, p=reg_w)
        Z_syn = reg_m[reg_idx] + reg_s[reg_idx] * (rng.standard_normal((n, chol.shape[1])) @ chol.T + rng.standard_normal((n, p_use)) * diag_std[None, :])

        device = next(self.parameters()).device
        Z_pad = torch.zeros((n, self.max_cols), dtype=torch.float32, device=device)
        Z_pad[:, :p_use] = torch.tensor(Z_syn, dtype=torch.float32, device=device)
        col_mask = torch.zeros(self.max_cols, dtype=torch.float32, device=device)
        col_mask[:p_use] = 1.0
        dist_lat = torch.tensor(np.asarray(encoding["dist_latent"], dtype=np.float32), device=device)
        Z_ref = self.cov_net(Z_pad, dist_lat, col_mask)[:, :p_use].detach().cpu().numpy().astype(np.float64)
        Z_syn = 0.94 * Z_syn + 0.06 * Z_ref
        Z_syn = (Z_syn - np.mean(Z_syn, axis=0)) / np.maximum(np.std(Z_syn, axis=0), 1e-6)

        U_syn = np.clip(0.5 * (1.0 + _erf_approx(Z_syn / math.sqrt(2.0))), 0.001, 0.999)
        knot_grid, knots = np.asarray(encoding.get("knot_grid", QUANTILE_KNOTS), dtype=np.float64), np.asarray(encoding["col_knots"], dtype=np.float64)
        is_bin, bin_p, bin_lh = np.asarray(encoding["is_binary_col"], dtype=bool), np.asarray(encoding["binary_probs"], dtype=np.float64), np.asarray(encoding["binary_low_high"], dtype=np.float64)
        disc_sup = encoding.get("discrete_support", {})

        X_syn = np.zeros((n, p_use), dtype=np.float64)
        for j in range(p_use):
            if is_bin[j]:
                X_syn[:, j] = np.where(U_syn[:, j] >= (1.0 - bin_p[j]), bin_lh[j, 1], bin_lh[j, 0])
            elif j in disc_sup:
                u_vals, cum_p = disc_sup[j]
                X_syn[:, j] = np.asarray(u_vals, dtype=np.float64)[np.clip(np.searchsorted(np.asarray(cum_p, dtype=np.float64), U_syn[:, j]), 0, len(u_vals) - 1)]
            else:
                X_syn[:, j] = np.interp(U_syn[:, j], knot_grid, knots[j])

        y_syn = _predict_from_target_bundle(
            X_syn, encoding["target_bundle"], task=task, level=level, rng=rng,
            target_net=self.target_net, task_lat=torch.tensor(np.asarray(encoding["task_latent"], dtype=np.float32), device=device), device=device,
        )
        return X_syn, y_syn


def _build_differentiable_basis(X: torch.Tensor, gates: Dict[str, torch.Tensor], *, level: str) -> torch.Tensor:
    n, p = X.shape
    mw = gates["monotone_w"]
    parts = [torch.ones((n, 1), device=X.device, dtype=X.dtype), X, mw * torch.sign(X) * torch.log1p(torch.abs(X)), mw * torch.tanh(X * 0.75), 0.5 * mw * (torch.abs(X) - 0.8)]
    if level in ("high", "ultra"):
        hw, pw = gates["hinge_w"], gates["pair_w"]
        parts.extend([hw * F.relu(X - 0.5), hw * F.relu(-X - 0.5), 0.35 * hw * torch.sin(X * 1.5)])
        pair_cols = [pw * (X[:, i : i + 1] * X[:, j : j + 1]) for i in range(min(p, 8)) for j in range(i + 1, min(p, 8))]
        if pair_cols:
            parts.append(torch.cat(pair_cols, dim=1))
    if level == "ultra":
        tw = gates["tree_w"]
        parts.extend([tw * torch.tanh(3.0 * (X - 1.0)), tw * torch.tanh(3.0 * (X + 1.0)), 0.5 * tw * torch.exp(-0.5 * (X ** 2))])
    return torch.cat(parts, dim=1)


def _to_normal_scores(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    out = np.zeros_like(X, dtype=np.float64)
    for j in range(p):
        u = (np.argsort(np.argsort(X[:, j])).astype(np.float64) + 0.5) / float(max(n, 1))
        out[:, j] = _erfinv_approx(2.0 * u - 1.0) * math.sqrt(2.0)
    return out


def _erf_approx(x: np.ndarray) -> np.ndarray:
    sign, ax = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-(ax ** 2)))


def _erfinv_approx(y: np.ndarray) -> np.ndarray:
    y, a = np.clip(y, -0.999999, 0.999999), 0.147
    ln = np.log(1.0 - y ** 2)
    term1 = 2.0 / (math.pi * a) + 0.5 * ln
    return np.sign(y) * np.sqrt(np.maximum(np.sqrt(term1 ** 2 - ln / a) - term1, 0.0))


def _fit_copula_regimes(Z: np.ndarray, *, n_regimes: int, sharpness: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, p = Z.shape
    if n_regimes <= 1 or n < 16:
        return np.array([1.0], dtype=np.float64), np.zeros((1, p), dtype=np.float64), np.ones((1, p), dtype=np.float64)
    u, s, _ = np.linalg.svd(Z, full_matrices=False)
    score = u[:, 0] * s[0]
    bounds = np.quantile(score, np.linspace(0.0, 1.0, n_regimes + 1))
    weights, means, scales = np.zeros(n_regimes, dtype=np.float64), np.zeros((n_regimes, p), dtype=np.float64), np.ones((n_regimes, p), dtype=np.float64)
    for k in range(n_regimes):
        mask = ((score >= bounds[k]) & (score < bounds[k + 1])) if k < n_regimes - 1 else (score >= bounds[k])
        if np.sum(mask) >= 2:
            weights[k] = float(np.mean(mask))
            means[k] = sharpness * np.mean(Z[mask], axis=0)
            scales[k] = (1.0 - 0.5 * sharpness) + 0.5 * sharpness * np.maximum(np.std(Z[mask], axis=0), 0.25)
        else:
            weights[k] = 1.0 / float(n_regimes)
    return weights / np.sum(weights), means, scales


def _build_numpy_basis(X: np.ndarray, gates: Dict[str, float], top_pair_indices: Sequence[Tuple[int, int]], *, level: str) -> np.ndarray:
    n, mw = X.shape[0], float(gates["monotone_w"])
    parts = [np.ones((n, 1), dtype=np.float64), X, mw * np.sign(X) * np.log1p(np.abs(X)), mw * np.tanh(0.75 * X), 0.5 * mw * (np.abs(X) - 0.8)]
    if level in ("high", "ultra"):
        hw, pw = float(gates["hinge_w"]), float(gates["pair_w"])
        parts.extend([hw * np.maximum(X - 0.5, 0.0), hw * np.maximum(-X - 0.5, 0.0), 0.35 * hw * np.sin(1.5 * X)])
        if top_pair_indices:
            parts.append(np.column_stack([pw * (X[:, i] * X[:, j]) for i, j in top_pair_indices]))
    if level == "ultra":
        tw = float(gates["tree_w"])
        parts.extend([tw * np.tanh(3.0 * (X - 1.0)), tw * np.tanh(3.0 * (X + 1.0)), 0.5 * tw * np.exp(-0.5 * (X ** 2))])
    return np.column_stack(parts)


def _fit_oblivious_trees(
    X: np.ndarray, residual: np.ndarray, driver_prior: np.ndarray, *, n_trees: int, depth: int = 3, learning_rate: float = 0.18
) -> List[Dict[str, Any]]:
    n, p = X.shape
    if n_trees <= 0 or n < 12 or p == 0:
        return []
    r, trees = residual.copy(), []
    col_quantiles = [np.unique(np.quantile(X[:, f], np.linspace(0.15, 0.85, 7))) for f in range(p)]

    for _ in range(n_trees):
        if float(np.std(r)) <= 1e-7:
            break
        r_centered = r - float(np.mean(r))
        corrs = np.abs((X.T @ r_centered) / float(n)) * (0.5 + 0.5 * driver_prior / max(float(np.max(driver_prior)), 1e-6))
        top_feats = np.argsort(corrs)[::-1][: min(p, 8)]
        chosen_feats, chosen_thresh, leaf_idx = [], [], np.zeros(n, dtype=np.int64)
        for d in range(depth):
            best_gain, best_f, best_t = -1.0, int(top_feats[d % len(top_feats)]), 0.0
            for f in top_feats:
                for q in col_quantiles[f]:
                    new_idx = (leaf_idx << 1) | (X[:, f] >= q).astype(np.int64)
                    counts = np.bincount(new_idx, minlength=1 << (d + 1)).astype(np.float64)
                    sums = np.bincount(new_idx, weights=r_centered, minlength=1 << (d + 1))
                    gain = float(np.sum((sums ** 2) / (counts + 3.0)))
                    if gain > best_gain:
                        best_gain, best_f, best_t = gain, int(f), float(q)
            chosen_feats.append(best_f)
            chosen_thresh.append(best_t)
            leaf_idx = (leaf_idx << 1) | (X[:, best_f] >= best_t).astype(np.int64)
        counts = np.bincount(leaf_idx, minlength=1 << depth).astype(np.float64)
        sums = np.bincount(leaf_idx, weights=r, minlength=1 << depth)
        leaf_vals = learning_rate * (sums / (counts + 4.0))
        r = r - leaf_vals[leaf_idx]
        trees.append({"features": chosen_feats, "thresholds": [round(t, 4) for t in chosen_thresh], "leaf_values": leaf_vals.astype(np.float16)})
    return trees


def _eval_oblivious_trees(X: np.ndarray, trees: Sequence[Dict[str, Any]]) -> np.ndarray:
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for t in trees:
        idx = np.zeros(n, dtype=np.int64)
        for f, th in zip(t["features"], t["thresholds"]):
            idx = (idx << 1) | (X[:, f] >= th).astype(np.int64)
        out += np.asarray(t["leaf_values"], dtype=np.float64)[idx]
    return out


def _fit_hierarchical_target_bundle(
    X: np.ndarray, y: np.ndarray, *, gates: Dict[str, float], col_prior: np.ndarray, emp_drivers: np.ndarray,
    task: str, level: str, target_net: ConditionalHyperTargetNet, task_lat_tensor: torch.Tensor,
) -> Dict[str, Any]:
    n, p = X.shape
    combined_prior = 0.5 * col_prior + 0.5 * emp_drivers
    top_cols = np.argsort(combined_prior)[::-1][: min(p, 10)]
    max_pairs = 0 if level == "medium" else 16 if level == "high" else 28
    pairs = [(int(top_cols[a]), int(top_cols[b])) for a in range(len(top_cols)) for b in range(a + 1, len(top_cols))][:max_pairs]

    Phi = _build_numpy_basis(X, gates, pairs, level=level)
    gram = (Phi.T @ Phi) / float(max(n, 1))
    eye = np.eye(Phi.shape[1], dtype=np.float64)
    eye[0, 0] = 1e-6
    w_basis = np.linalg.solve(gram + float(gates["ridge_alpha"]) * eye, (Phi.T @ y) / float(max(n, 1)))
    pred_basis = Phi @ w_basis

    device = task_lat_tensor.device
    X_pad, col_mask = torch.zeros((n, target_net.max_cols), dtype=torch.float32, device=device), torch.zeros(target_net.max_cols, dtype=torch.float32, device=device)
    X_pad[:, :p], col_mask[:p] = torch.tensor(X, dtype=torch.float32, device=device), 1.0
    res_net = target_net(X_pad, task_lat_tensor, col_mask).detach().cpu().numpy().astype(np.float64)
    res_net_mean = float(np.mean(res_net))
    residual = y - (pred_basis + float(gates["neural_w"]) * (res_net - res_net_mean))

    trees = _fit_oblivious_trees(
        X, residual, combined_prior, n_trees=4 if level == "medium" else 24 if level == "high" else 56,
        depth=2 if level == "medium" else 3 if level == "high" else 4,
        learning_rate=0.16 if level == "medium" else 0.19 if level == "high" else 0.22,
    )
    final_res = residual - _eval_oblivious_trees(X, trees)
    return {
        "gates": gates, "pairs": pairs, "w_basis": w_basis.astype(np.float16), "res_net_mean": round(res_net_mean, 6),
        "neural_w": round(float(gates["neural_w"]), 5), "trees": trees, "residual_std": round(float(np.std(final_res)), 6),
        "noise_scale": round(float(gates["noise_scale"]), 5), "class_prior": round(float(np.mean(y)) if task == "classification" else 0.5, 6),
        "y_min": round(float(np.min(y)), 5), "y_max": round(float(np.max(y)), 5),
    }


def _predict_from_target_bundle(
    X_syn: np.ndarray, bundle: Dict[str, Any], *, task: str, level: str,
    rng: np.random.Generator, target_net: ConditionalHyperTargetNet, task_lat: torch.Tensor, device: torch.device,
) -> np.ndarray:
    n, p = X_syn.shape
    gates_for_basis = bundle.get("gates") or {
        "monotone_w": 0.55 if level == "medium" else 0.65, "hinge_w": 0.55 if level in ("high", "ultra") else 0.0,
        "pair_w": 0.60 if level in ("high", "ultra") else 0.0, "tree_w": 1.00 if level == "ultra" else 0.0,
    }
    Phi = _build_numpy_basis(X_syn, gates_for_basis, bundle["pairs"], level=level)
    w_basis = np.asarray(bundle["w_basis"], dtype=np.float64)
    m = min(Phi.shape[1], w_basis.size)
    pred = Phi[:, :m] @ w_basis[:m]

    X_pad, col_mask = torch.zeros((n, target_net.max_cols), dtype=torch.float32, device=device), torch.zeros(target_net.max_cols, dtype=torch.float32, device=device)
    X_pad[:, :p], col_mask[:p] = torch.tensor(X_syn, dtype=torch.float32, device=device), 1.0
    res_net = target_net(X_pad, task_lat, col_mask).detach().cpu().numpy().astype(np.float64)
    pred = pred + float(bundle["neural_w"]) * (res_net - float(bundle["res_net_mean"])) + _eval_oblivious_trees(X_syn, bundle["trees"])

    if task == "classification":
        prior = float(np.clip(bundle["class_prior"], 0.01, 0.99))
        score = pred + rng.normal(0.0, float(bundle["noise_scale"]) * (0.12 if level == "medium" else 0.06 if level == "high" else 0.03), size=n)
        y_out = (score >= float(np.quantile(score, 1.0 - prior))).astype(np.float64)
        if n >= 2 and len(np.unique(y_out)) < 2:
            y_out[int(np.argmax(score))], y_out[int(np.argmin(score))] = 1.0, 0.0
        return y_out

    noise_sigma = float(bundle["residual_std"]) * float(bundle["noise_scale"]) * (0.42 if level == "medium" else 0.22 if level == "high" else 0.10)
    return np.clip(pred + rng.normal(0.0, noise_sigma, size=n), float(bundle["y_min"]) - 0.5, float(bundle["y_max"]) + 0.5)


def _pack_code_vector(
    *, width: int, col_knots: np.ndarray, chol_factor: np.ndarray,
    target_bundle: Dict[str, Any], task_lat: np.ndarray, dist_lat: np.ndarray,
) -> np.ndarray:
    parts = [
        col_knots.reshape(-1).astype(np.float32), chol_factor.reshape(-1).astype(np.float32),
        np.asarray(target_bundle["w_basis"], dtype=np.float32).reshape(-1),
        task_lat.reshape(-1).astype(np.float32), dist_lat.reshape(-1).astype(np.float32),
    ] + [np.asarray(t["leaf_values"], dtype=np.float32).reshape(-1) for t in target_bundle["trees"]]
    flat = np.concatenate(parts)
    out = np.zeros(width, dtype=np.float32)
    take = min(width, flat.size)
    out[:take] = np.tanh(flat[:take] / 4.0) * 4.0
    return out
