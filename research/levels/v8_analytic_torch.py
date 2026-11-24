"""Vectorized PyTorch DSE8 analytic packet helpers matching Rust `decode_dse8_rows`."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from research.levels.v8_packet import KNOT_COUNT, V8_MAX_CLASSES


def extract_packet_analytic_tensors(pkt: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Extract vectorized PyTorch tensors for exact DSE8 analytic decode parity during training."""
    h = pkt["header"]
    margs = pkt.get("marginals") or []
    tail_m = [0.0 if int(m["tail_family"]) == 0 else (0.28 if int(m["tail_family"]) == 1 else 0.55) for m in margs]
    out: dict[str, Any] = {
        "task_id": int(h["task_id"]),
        "o_dim": int(pkt["linear"]["o_dim"]),
        "rank_use": int(h["rank_use"]),
        "inv_nu": float(h["inv_nu"]),
        "logit_tau": float(pkt["linear"].get("logit_tau", 1.0)),
        "tail_mult": torch.tensor(tail_m, device=device, dtype=torch.float32) if tail_m else None,
        "marginals": margs,
        "nominal_groups": pkt.get("nominal_groups") or [],
        "reg_weights": None,
        "reg_means": None,
        "ef_col": None,
        "ef_cls": None,
        "ef_bins8": None,
        "pr_ci": None,
        "pr_cj": None,
        "pr_cls": None,
        "pr_wprod": None,
        "pr_wxor": None,
        "tr_cols": None,
        "tr_thrs": None,
        "tr_cls": None,
        "tr_leaves": None,
        "rs_cols": None,
        "rs_log_sd_w": None,
        "rs_q5": None,
    }
    regs = pkt.get("regimes") or []
    if len(regs) >= 2:
        wts = [max(float(rg["weight"]), 0.0) for rg in regs]
        s_w = sum(wts)
        if s_w > 1e-12:
            out["reg_weights"] = torch.tensor([w / s_w for w in wts], device=device, dtype=torch.float32)
            mf = np.zeros((len(regs), 12), dtype=np.float32)
            for ri, rg in enumerate(regs):
                arr = rg["mean_f"]
                k = min(len(arr), 12)
                mf[ri, :k] = arr[:k]
            out["reg_means"] = torch.from_numpy(mf).to(device=device, dtype=torch.float32)
    efs = pkt.get("effects") or []
    if efs:
        out["ef_col"] = torch.tensor([int(e["col_idx"]) for e in efs], device=device, dtype=torch.long)
        out["ef_cls"] = torch.tensor([int(e["cls_idx"]) for e in efs], device=device, dtype=torch.long)
        out["ef_bins8"] = torch.tensor([list(e["bins8"]) for e in efs], device=device, dtype=torch.float32)
    prs = pkt.get("pairs") or []
    if prs:
        out["pr_ci"] = torch.tensor([int(pr["col_i"]) for pr in prs], device=device, dtype=torch.long)
        out["pr_cj"] = torch.tensor([int(pr["col_j"]) for pr in prs], device=device, dtype=torch.long)
        out["pr_cls"] = torch.tensor([int(pr["cls_idx"]) for pr in prs], device=device, dtype=torch.long)
        out["pr_wprod"] = torch.tensor([float(pr["w_prod"]) for pr in prs], device=device, dtype=torch.float32)
        out["pr_wxor"] = torch.tensor([float(pr["w_xor"]) for pr in prs], device=device, dtype=torch.float32)
    trs = pkt.get("trees") or []
    if trs:
        out["tr_cols"] = torch.tensor([list(t["cols"]) for t in trs], device=device, dtype=torch.long)
        out["tr_thrs"] = torch.tensor([list(t["thrs"]) for t in trs], device=device, dtype=torch.long)
        out["tr_cls"] = torch.tensor([int(t["cls_idx"]) for t in trs], device=device, dtype=torch.long)
        out["tr_leaves"] = torch.tensor([list(t["leaves"]) for t in trs], device=device, dtype=torch.float32)
    rs = pkt.get("resid_model") or {}
    if rs.get("cols"):
        out["rs_cols"] = torch.tensor([int(c) for c in rs["cols"]], device=device, dtype=torch.long)
        out["rs_log_sd_w"] = torch.tensor([float(w) for w in rs["log_sd_w"]], device=device, dtype=torch.float32)
    q5 = rs.get("q5_resid") or []
    if len(q5) == 5 and any(abs(float(v)) > 1e-4 for v in q5):
        out["rs_q5"] = torch.tensor([float(v) for v in q5], device=device, dtype=torch.float32)
    return out


def _rust_norm_cdf_torch(z: torch.Tensor) -> torch.Tensor:
    """Exact float64 mirror of Rust `dope_core::stats::norm_cdf`."""
    x = -z.double() / math.sqrt(2.0)
    mag = x.abs()
    t = (1.0 + 0.5 * mag).reciprocal()
    poly = 0.17087277
    for c in (-0.82215223, 1.48851587, -1.13520398, 0.27886807, -0.18628806, 0.09678418, 0.37409196, 1.00002368):
        poly = c + t * poly
    res = t * (-mag.square() - 1.26551223 + t * poly).exp()
    return 0.5 * torch.where(x >= 0.0, res, 2.0 - res)


def categorical_f64_torch(weights: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Exact float64 mirror of Rust `rng_v8::sample_categorical`."""
    cum = weights.double().cumsum(dim=-1)
    tot = cum[..., -1]
    thr = uniform.double().clamp(0.0, 1.0 - 1e-15) * tot
    idx = (thr.unsqueeze(-1) > cum).sum(dim=-1).clamp(max=weights.shape[-1] - 1)
    return torch.where(tot > 1e-12, idx, torch.zeros_like(idx))


def sample_classes_torch(logits: torch.Tensor, task_id: int, uniform: torch.Tensor) -> torch.Tensor:
    """Exact mirror of Rust binary/multiclass target sampling from scaled logits."""
    if task_id == 1:
        return (uniform.double() < torch.sigmoid(logits[..., 0].double().clamp(-12.0, 12.0))).long()
    shifted = (logits - logits.amax(dim=-1, keepdim=True)).clamp(-20.0, 0.0)
    return categorical_f64_torch(shifted.double().exp(), uniform)


def apply_typed_marginals_ste_torch(
    x_knot: torch.Tensor,
    z_lat: torch.Tensor,
    pkt_analytic: dict[str, Any] | None,
    *,
    u_missing: torch.Tensor | None = None,
    u_atom: torch.Tensor | None = None,
    u_nominal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exact Rust typed marginals (const/lattice/heaping/atoms/missing/nominal) with STE."""
    B, N, P = x_knot.shape
    device, dtype = x_knot.device, x_knot.dtype
    miss_mask = torch.zeros((B, N, P), dtype=torch.bool, device=device)
    if not pkt_analytic:
        return x_knot, miss_mask
    margs = pkt_analytic.get("marginals") or []
    nom_groups = pkt_analytic.get("nominal_groups") or []
    has_non_cont = any(int(m["kind"]) != 0 or float(m["miss_rate"]) > 1e-4 for m in margs)
    if not has_non_cont and not nom_groups:
        return x_knot, miss_mask

    with torch.no_grad():
        x_ex = x_knot.detach().clone()
        u_cdf: torch.Tensor | None = None
        for j in range(min(P, len(margs))):
            m = margs[j]
            kind = int(m["kind"])
            mr = float(m["miss_rate"])
            if mr > 1e-4:
                um = u_missing[:, :, j].to(device=device, dtype=torch.float64) if u_missing is not None else torch.rand((B, N), device=device, dtype=torch.float64)
                miss_mask[:, :, j] = um < mr
            if kind == 4:
                x_ex[:, :, j] = float(m["lo"])
            elif kind == 2 and m.get("atom_vals"):
                if u_cdf is None:
                    u_cdf = _rust_norm_cdf_torch(z_lat.detach()).float()
                uj = u_cdf[:, :, j]
                atoms = x_ex.new_tensor([float(v) for v in m["atom_vals"]])
                probs = x_ex.new_tensor([float(pr) for pr in m["atom_probs"]])
                idx_a = (uj.unsqueeze(-1) > probs.cumsum(0)[None, None, :]).sum(dim=-1).clamp(max=atoms.numel() - 1)
                x_ex[:, :, j] = atoms[idx_a]
            elif kind == 3 and m.get("atom_vals"):
                if u_cdf is None:
                    u_cdf = _rust_norm_cdf_torch(z_lat.detach()).float()
                uj = u_cdf[:, :, j]
                xj = x_ex[:, :, j]
                assigned = torch.zeros((B, N), dtype=torch.bool, device=device)
                lo_p = 0.0
                k17 = m["knots17"]
                m_lo = float(m["lo"])
                for av, ap in zip(m["atom_vals"], m["atom_probs"]):
                    av_f, ap_f = float(av), float(ap)
                    pos_a = sum(1 for kv in k17 if float(kv) < av_f)
                    tgt_u = min(max(pos_a / float(KNOT_COUNT - 1), 0.0), 1.0)
                    cond = (uj - tgt_u).abs() <= 0.5 * ap_f
                    if abs(av_f - m_lo) <= 1e-4:
                        cond = cond | ((uj >= lo_p) & (uj <= lo_p + ap_f))
                    hit = (~assigned) & cond
                    xj = torch.where(hit, xj.new_tensor(av_f), xj)
                    assigned = assigned | hit
                    lo_p += ap_f
                x_ex[:, :, j] = xj
            elif kind == 1 and float(m["lattice_q0"]) > 1e-5:
                q0 = float(m["lattice_q0"])
                orig = float(m["lattice_origin"])
                u_q = (x_ex[:, :, j] - orig) / q0
                sgn = torch.where(u_q < 0.0, u_q.new_tensor(-1.0), u_q.new_tensor(1.0))
                xv = orig + sgn * torch.floor(u_q.abs() + 0.5) * q0
                he = float(m["heaping_excess"])
                hs = float(m["heaping_step"])
                if he > 0.05 and hs > q0:
                    ua = u_atom[:, :, j].to(device=device, dtype=torch.float64) if u_atom is not None else torch.rand((B, N), device=device, dtype=torch.float64)
                    u_h = (xv - orig) / hs
                    sgn_h = torch.where(u_h < 0.0, u_h.new_tensor(-1.0), u_h.new_tensor(1.0))
                    xv_h = orig + sgn_h * torch.floor(u_h.abs() + 0.5) * hs
                    xv = torch.where(ua < he, xv_h, xv)
                x_ex[:, :, j] = xv.clamp(-6.0, 6.0)
            if mr > 1e-4:
                x_ex[:, :, j] = torch.where(miss_mask[:, :, j], x_ex.new_tensor(0.0), x_ex[:, :, j])

        for gi, grp in enumerate(nom_groups):
            cols = [int(c) for c in grp["cols"]]
            probs = [float(pr) for pr in grp["probs"]]
            if not cols or any(c >= P for c in cols):
                continue
            p_vec = torch.tensor(probs, device=device, dtype=torch.float64)
            z_cols = z_lat.detach()[:, :, cols].double().clamp(-3.0, 3.0)
            wts = torch.where(p_vec[None, None, :] > 1e-6, p_vec[None, None, :] * torch.exp(0.75 * z_cols), 0.0)
            bp = float(grp.get("baseline_prob", 0.0))
            base_w = torch.full((B, N, 1), bp if bp > 1e-6 else 0.0, device=device, dtype=torch.float64)
            wts_all = torch.cat([wts, base_w], dim=-1)
            un = u_nominal[:, :, gi].to(device=device, dtype=torch.float64) if u_nominal is not None else torch.rand((B, N), device=device, dtype=torch.float64)
            pick = categorical_f64_torch(wts_all, un)
            for idx_c, cj in enumerate(cols):
                x_ex[:, :, cj] = (pick == idx_c).to(dtype)
                miss_mask[:, :, cj] = False
        x_ex = x_ex.clamp(-6.0, 6.0)

    keep = (~miss_mask).to(dtype)
    x_ste = (x_knot + (x_ex - x_knot).detach()) * keep
    return x_ste, miss_mask


def compute_fit_midrank_z_np(x_fit: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """FIT-only empirical midrank normal scores matching Rust `CdfIntervals::build`.

    Every column's tie-group CDF `[lo, hi]` is defined strictly by the finite FIT values
    in `x_fit[:, j]` (zero query leakage). Non-finite entries map to `0.0`.
    """
    xf = np.asarray(x_fit, dtype=np.float64)
    xe = np.asarray(x_eval, dtype=np.float64)
    n_eval, p = xe.shape
    out = np.zeros((n_eval, p), dtype=np.float32)
    sqrt2 = math.sqrt(2.0)
    for j in range(p):
        col_f = xf[:, j]
        fin_f = col_f[np.isfinite(col_f)]
        n_obs = fin_f.size
        if n_obs < 2:
            continue
        s_fit = np.sort(fin_f)
        col_e = xe[:, j]
        fin_mask = np.isfinite(col_e)
        if not np.any(fin_mask):
            continue
        vals = col_e[fin_mask]
        tol = 1e-12 * np.maximum(np.abs(vals), 1.0)
        lo = np.searchsorted(s_fit, vals - tol, side="left").astype(np.float64) / float(n_obs)
        hi = np.searchsorted(s_fit, vals + tol, side="right").astype(np.float64) / float(n_obs)
        mid = np.clip(0.5 * (lo + hi), 1e-4, 1.0 - 1e-4)
        z = torch.erfinv(torch.from_numpy(2.0 * mid - 1.0)).numpy() * sqrt2
        out[fin_mask, j] = np.clip(z, -3.5, 3.5).astype(np.float32)
    return out


def invert_knots_to_z_torch(
    x_in: torch.Tensor, base_knots: torch.Tensor, z_max: torch.Tensor
) -> torch.Tensor:
    """Monotonic packet-only inverse of the 17-knot quantile map (`x_unit -> z_lat`)."""
    B, N, P = x_in.shape
    k_exp = base_knots.unsqueeze(1)  # [B, 1, P, 17]
    x_exp = x_in.unsqueeze(-1)  # [B, N, P, 1]
    idx_hi = (k_exp <= x_exp).sum(dim=-1).clamp(1, KNOT_COUNT - 1)  # [B, N, P] in 1..16
    idx_lo = idx_hi - 1
    k_lo = torch.gather(k_exp.expand(B, N, P, KNOT_COUNT), -1, idx_lo.unsqueeze(-1)).squeeze(-1)
    k_hi = torch.gather(k_exp.expand(B, N, P, KNOT_COUNT), -1, idx_hi.unsqueeze(-1)).squeeze(-1)
    denom = (k_hi - k_lo).clamp(min=1e-6)
    frac = ((x_in - k_lo) / denom).clamp(0.0, 1.0)
    pos = idx_lo.to(x_in.dtype) + frac
    zm = z_max.reshape(B, 1, 1)
    dz = (2.0 * zm) / float(KNOT_COUNT - 1)
    return (pos * dz - zm).clamp(-4.0, 4.0)


def eval_analytic_packet_logits_torch(
    z_lat: torch.Tensor, pkt_analytic: dict[str, Any] | None
) -> torch.Tensor:
    """Vectorized exact mirror of Rust `eval_effects_and_pairs` + `TreesSectionV8::eval_row`."""
    B, N, P = z_lat.shape
    out = torch.zeros(B, N, V8_MAX_CLASSES, device=z_lat.device, dtype=z_lat.dtype)
    if not pkt_analytic:
        return out
    u_cdf = _rust_norm_cdf_torch(z_lat).to(dtype=z_lat.dtype)
    if pkt_analytic.get("ef_col") is not None:
        cols = pkt_analytic["ef_col"].clamp(0, max(P - 1, 0))
        clss = pkt_analytic["ef_cls"].clamp(0, V8_MAX_CLASSES - 1)
        bins8 = pkt_analytic["ef_bins8"].to(dtype=z_lat.dtype)  # [E, 8]
        b8 = (u_cdf[:, :, cols] * 8.0).floor().long().clamp(0, 7)  # [B, N, E]
        vals = bins8[torch.arange(cols.shape[0], device=z_lat.device)[None, None, :], b8]
        out.scatter_add_(2, clss[None, None, :].expand(B, N, -1), vals)
    if pkt_analytic.get("pr_ci") is not None:
        ci = pkt_analytic["pr_ci"].clamp(0, max(P - 1, 0))
        cj = pkt_analytic["pr_cj"].clamp(0, max(P - 1, 0))
        clss = pkt_analytic["pr_cls"].clamp(0, V8_MAX_CLASSES - 1)
        zi = z_lat[:, :, ci].clamp(-3.0, 3.0)
        zj = z_lat[:, :, cj].clamp(-3.0, 3.0)
        xs = torch.where((zi > 0.0) == (zj > 0.0), z_lat.new_tensor(1.0), z_lat.new_tensor(-1.0))
        wp = pkt_analytic["pr_wprod"].to(dtype=z_lat.dtype)[None, None, :]
        wx = pkt_analytic["pr_wxor"].to(dtype=z_lat.dtype)[None, None, :]
        vals = wp * (zi * zj).clamp(-6.0, 6.0) + wx * xs
        out.scatter_add_(2, clss[None, None, :].expand(B, N, -1), vals)
    if pkt_analytic.get("tr_cols") is not None:
        tcols = pkt_analytic["tr_cols"].clamp(0, max(P - 1, 0))  # [T, 3]
        tthrs = pkt_analytic["tr_thrs"]  # [T, 3]
        tclss = pkt_analytic["tr_cls"].clamp(0, V8_MAX_CLASSES - 1)  # [T]
        tleaves = pkt_analytic["tr_leaves"].to(dtype=z_lat.dtype)  # [T, 8]
        b16 = (u_cdf[:, :, tcols] * 16.0).floor().long().clamp(0, 15)  # [B, N, T, 3]
        gt = (b16 > tthrs[None, None, :, :]).long()
        l_idx = gt[..., 0] + 2 * gt[..., 1] + 4 * gt[..., 2]  # [B, N, T]
        vals = tleaves[torch.arange(tcols.shape[0], device=z_lat.device)[None, None, :], l_idx]
        out.scatter_add_(2, tclss[None, None, :].expand(B, N, -1), vals)
    return out
