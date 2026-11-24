"""Scalar parity mirror of Rust `decode_dse8_rows` for 12-fixture hard parity (`<= 1e-4`)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from research.levels.v8_packet import (
    KNOT_COUNT,
    STREAM_ATOM,
    STREAM_FACTOR,
    STREAM_MISSING,
    STREAM_NOISE,
    STREAM_REGIME,
    STREAM_TARGET,
    V8_MAX_CLASSES,
    build_decoder_inputs_py,
    counter_normal,
    counter_student_t_scale,
    counter_uniform,
)

if TYPE_CHECKING:
    from research.levels.v8_single_encoding import SingleEncV8

NODE_TREES = 8
NODE_DEPTH = 3
NODE_LEAVES = 8
BILIN_RANK = 4


@torch.no_grad()
def decode_dse8_rows_py_parity(
    pkt: dict[str, Any],
    model: SingleEncV8 | None,
    n_rows: int,
    seed: int,
    active_rank_v1: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact scalar parity mirror of Rust `decode_dse8_rows` for 12-fixture hard parity (`<= 1e-4`)."""
    h = pkt["header"]
    p = h["p_eff"]
    r_use = h["rank_use"]
    o_dim = pkt["linear"]["o_dim"]
    card = pkt["card"]

    L_eff = pkt["loadings"].copy()
    knots_eff = np.stack([m["knots17"].copy() for m in pkt["marginals"]], axis=0)
    W_eff = pkt["linear"]["w"].copy()
    resid_sd_eff = float(h["resid_sd"])
    plan_np: dict[str, np.ndarray] | None = None

    if model is not None:
        model.eval()
        dc, de, dg, dz = build_decoder_inputs_py(pkt)
        pl = model.decode_plan(
            torch.from_numpy(dc)[None],
            torch.from_numpy(de)[None],
            torch.from_numpy(dg)[None],
            torch.from_numpy(dz)[None],
        )
        plan_np = {k: v[0].cpu().numpy() for k, v in pl.items() if k not in ("h_dec", "g_pool")}
        r_lim = min(r_use, L_eff.shape[1]) if active_rank_v1 else L_eff.shape[1]
        dl_eff = plan_np["delta_L"][:, :r_lim] if r_lim > 0 else np.zeros((p, 1), dtype=np.float32)
        max_eff = max(
            float(np.max(np.abs(plan_np["delta_knots"]))),
            float(np.max(np.abs(dl_eff))),
            float(np.max(np.abs(plan_np["delta_W"][:, :o_dim]))),
            float(np.max(np.abs(plan_np["node_leaves"]))),
            float(np.max(np.abs(plan_np["bilin_w"]))),
            float(np.max(np.abs(plan_np["flow_s"]))),
            float(np.max(np.abs(plan_np["flow_t"]))),
            abs(float(plan_np["resid_sd_mult"]) - 1.0),
        )
        if max_eff < 1e-7:
            plan_np = None
        else:
            resid_sd_eff = float(np.clip(resid_sd_eff * float(plan_np["resid_sd_mult"]), 0.002, 6.0))
            for j in range(p):
                if active_rank_v1:
                    L_eff[j, :r_lim] = L_eff[j, :r_lim] + plan_np["delta_L"][j, :r_lim]
                    L_eff[j, r_lim:] = 0.0
                    sc = max(float(np.linalg.norm(L_eff[j, :r_lim])) / 0.999, 1.0)
                    L_eff[j, :r_lim] /= sc
                else:
                    L_eff[j] = L_eff[j] + plan_np["delta_L"][j]
                    sc = max(float(np.linalg.norm(L_eff[j])) / 0.999, 1.0)
                    L_eff[j] /= sc
                if pkt["marginals"][j]["kind"] == 0:
                    base = pkt["marginals"][j]["knots17"]
                    bd = np.maximum(base[1:] - base[:-1], 0.0)
                    wr = bd * np.exp(0.25 * np.tanh(plan_np["delta_knots"][j, 1:]))
                    sb = float(bd.sum())
                    ws = max(float(wr.sum()), 1e-7)
                    steps = wr * (sb / ws) if sb > 1e-7 else np.zeros_like(bd)
                    knots_eff[j, 0] = base[0]
                    knots_eff[j, 1:] = base[0] + np.cumsum(steps)
                for c in range(o_dim):
                    W_eff[c, j] += float(plan_np["delta_W"][j, c])

    r_noise = min(r_use, L_eff.shape[1]) if (model is not None and active_rank_v1) else L_eff.shape[1]
    noise_sd = np.sqrt(np.maximum(1.0 - np.sum(L_eff[:, :r_noise] ** 2, axis=1), 1e-4)).astype(np.float32)

    x_out = np.zeros((n_rows, card["x_cols_raw"]), dtype=np.float32)
    y_out = np.zeros(n_rows, dtype=np.float32)

    def _norm_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _sample_cat(probs: list[float], u: float) -> int:
        s = sum(max(pr, 0.0) for pr in probs)
        if s <= 1e-12:
            return 0
        tgt = min(max(u, 0.0), 1.0 - 1e-15) * s
        acc = 0.0
        for idx, pr in enumerate(probs):
            acc += max(pr, 0.0)
            if tgt <= acc:
                return idx
        return len(probs) - 1

    for row_idx in range(n_rows):
        t_sc = counter_student_t_scale(seed, row_idx, h["inv_nu"])
        zf = np.array([counter_normal(seed, row_idx, STREAM_FACTOR, r) * t_sc for r in range(r_use)], dtype=np.float32)
        if len(pkt["regimes"]) >= 2:
            chosen = _sample_cat([rg["weight"] for rg in pkt["regimes"]], counter_uniform(seed, row_idx, STREAM_REGIME, 0))
            reg = pkt["regimes"][chosen]
            for r in range(min(r_use, len(reg["mean_f"]))):
                mu = float(reg["mean_f"][r])
                shr = math.sqrt(max(1.0 - min(mu * mu, 0.75), 0.25))
                zf[r] = np.float32(mu + shr * float(zf[r]))

        z_lat = np.zeros(p, dtype=np.float32)
        for j in range(p):
            eps = counter_normal(seed, row_idx, STREAM_NOISE, j) * t_sc
            z = eps * float(noise_sd[j]) + float(np.dot(zf, L_eff[j, :r_use]))
            z_lat[j] = np.float32(z)

        if plan_np is not None and p >= 2:
            for layer in range(2):
                vals = [float(z_lat[j]) / (1.0 + math.exp(-float(z_lat[j]))) for j in range(p) if j % 2 == layer]
                c_bar = min(max(sum(vals) / max(len(vals), 1), -3.0), 3.0)
                for j in range(p):
                    if j % 2 != layer:
                        s = float(plan_np["flow_s"][j, layer])
                        t = float(plan_np["flow_t"][j, layer])
                        z_lat[j] = np.float32(min(max(float(z_lat[j]) * math.exp(min(max(s * c_bar, -0.6), 0.6)) + t * c_bar, -5.0), 5.0))

        x_u = np.zeros(p, dtype=np.float32)
        miss = [False] * p
        z_max = float(h["z_max"])
        dz = (2.0 * z_max) / float(KNOT_COUNT - 1)
        for j in range(p):
            m = pkt["marginals"][j]
            if m["miss_rate"] > 1e-4 and counter_uniform(seed, row_idx, STREAM_MISSING, j) < m["miss_rate"]:
                miss[j] = True
                continue
            if m["kind"] == 4:
                x_u[j] = np.float32(m["lo"])
                continue
            if m["kind"] == 2 and m["atom_vals"]:
                u = _norm_cdf(float(z_lat[j]))
                acc = 0.0
                picked = m["atom_vals"][-1]
                for av, ap in zip(m["atom_vals"], m["atom_probs"]):
                    acc += ap
                    if u <= acc:
                        picked = av
                        break
                x_u[j] = np.float32(picked)
                continue
            if m["kind"] == 3 and m["atom_vals"]:
                u = _norm_cdf(float(z_lat[j]))
                lo_p = 0.0
                hit = None
                for av, ap in zip(m["atom_vals"], m["atom_probs"]):
                    pos_a = sum(1 for kv in m["knots17"] if kv < av)
                    tgt_u = min(max(pos_a / float(KNOT_COUNT - 1), 0.0), 1.0)
                    if abs(u - tgt_u) <= 0.5 * ap or (lo_p <= u <= lo_p + ap and abs(av - m["lo"]) <= 1e-4):
                        hit = av
                        break
                    lo_p += ap
                if hit is not None:
                    x_u[j] = np.float32(hit)
                    continue
            pos = (min(max(float(z_lat[j]), -z_max), z_max) + z_max) / dz
            i0 = min(max(int(math.floor(pos)), 0), KNOT_COUNT - 2)
            frac = min(max(pos - i0, 0.0), 1.0)
            ke = knots_eff[j]
            xv = float(ke[i0]) + frac * float(ke[i0 + 1] - ke[i0])
            if float(z_lat[j]) < -z_max or float(z_lat[j]) > z_max:
                sl_lo = max(float(ke[1] - ke[0]) / dz, 0.0)
                sl_hi = max(float(ke[-1] - ke[-2]) / dz, 0.0)
                mult = 0.0 if m["tail_family"] == 0 else (0.28 if m["tail_family"] == 1 else 0.55)
                xv += mult * sl_lo * min(float(z_lat[j]) + z_max, 0.0) + mult * sl_hi * max(float(z_lat[j]) - z_max, 0.0)
            if m["kind"] == 1 and m["lattice_q0"] > 1e-5:
                q0, orig = m["lattice_q0"], m["lattice_origin"]
                u_q = (xv - orig) / q0
                xv = orig + math.copysign(math.floor(abs(u_q) + 0.5), u_q) * q0
                if m["heaping_excess"] > 0.05 and m["heaping_step"] > q0:
                    if counter_uniform(seed, row_idx, STREAM_ATOM, j) < m["heaping_excess"]:
                        hs = m["heaping_step"]
                        u_h = (xv - orig) / hs
                        xv = orig + math.copysign(math.floor(abs(u_h) + 0.5), u_h) * hs
            x_u[j] = np.float32(min(max(xv, -6.0), 6.0))

        for gi, grp in enumerate(pkt["nominal_groups"]):
            wts = []
            for col_j, pr in zip(grp["cols"], grp["probs"]):
                if pr <= 1e-6:
                    wts.append(0.0)
                else:
                    wts.append(pr * math.exp(0.75 * min(max(float(z_lat[col_j]), -3.0), 3.0)))
            wts.append(grp["baseline_prob"] if grp["baseline_prob"] > 1e-6 else 0.0)
            pick = _sample_cat(wts, counter_uniform(seed, row_idx, STREAM_ATOM, 1024 + gi))
            for ci, col_j in enumerate(grp["cols"]):
                x_u[col_j] = 1.0 if pick == ci else 0.0
                miss[col_j] = False

        x_clean = np.where(np.array(miss), 0.0, x_u).astype(np.float32)
        logits = np.zeros(V8_MAX_CLASSES, dtype=np.float32)
        for c in range(o_dim):
            logits[c] = np.float32(float(pkt["linear"]["b"][c]) + float(np.dot(x_clean, W_eff[c])))
        for ef in pkt["effects"]:
            b8 = min(int(math.floor(_norm_cdf(float(z_lat[ef["col_idx"]])) * 8.0)), 7)
            logits[ef["cls_idx"]] += ef["bins8"][b8]
        for pr in pkt["pairs"]:
            zi = min(max(float(z_lat[pr["col_i"]]), -3.0), 3.0)
            zj = min(max(float(z_lat[pr["col_j"]]), -3.0), 3.0)
            xs = 1.0 if ((zi > 0.0) == (zj > 0.0)) else -1.0
            logits[pr["cls_idx"]] += np.float32(pr["w_prod"] * min(max(zi * zj, -6.0), 6.0) + pr["w_xor"] * xs)
        for tr in pkt["trees"]:
            l_idx = 0
            for d in range(3):
                b16 = min(int(math.floor(_norm_cdf(float(z_lat[tr["cols"][d]])) * 16.0)), 15)
                if b16 > tr["thrs"][d]:
                    l_idx |= 1 << d
            logits[tr["cls_idx"]] += tr["leaves"][l_idx]

        if plan_np is not None:
            nl = np.zeros(o_dim, dtype=np.float32)
            for t in range(NODE_TREES):
                pr_d = [
                    1.0 / (1.0 + math.exp(-min(max((float(np.dot(plan_np["node_w"][t, d, :p], x_clean)) - float(plan_np["node_thr"][t, d])) * 2.0, -10.0), 10.0)))
                    for d in range(NODE_DEPTH)
                ]
                for l_i in range(NODE_LEAVES):
                    wl = 1.0
                    for d in range(NODE_DEPTH):
                        wl *= pr_d[d] if (l_i & (1 << d)) else (1.0 - pr_d[d])
                    nl += wl * plan_np["node_leaves"][t, l_i, :o_dim]
            for r_i in range(BILIN_RANK):
                prod = math.tanh(float(np.dot(plan_np["bilin_u"][r_i, :p], x_clean))) * math.tanh(float(np.dot(plan_np["bilin_v"][r_i, :p], x_clean)))
                nl += prod * plan_np["bilin_w"][r_i, :o_dim]
            logits[:o_dim] += float(plan_np["alpha_table"]) * nl

        if h["task_id"] != 0:
            logits[:o_dim] *= float(pkt["linear"]["logit_tau"])

        if h["task_id"] == 0:
            eps_y = counter_normal(seed, row_idx, STREAM_TARGET, 0)
            rs = pkt["resid_model"]
            if any(abs(q) > 1e-4 for q in rs["q5_resid"]):
                u = _norm_cdf(eps_y)
                pos_r = min(max((u - 0.10) / 0.20, 0.0), 4.0)
                i0 = min(int(math.floor(pos_r)), 3)
                eps_y += rs["q5_resid"][i0] + (pos_r - i0) * (rs["q5_resid"][i0 + 1] - rs["q5_resid"][i0])
            lm = sum(w * min(max(float(z_lat[c]), -3.0), 3.0) for c, w in zip(rs["cols"], rs["log_sd_w"]))
            yu = min(max(float(logits[0]) + resid_sd_eff * math.exp(min(max(lm, -1.0), 1.0)) * eps_y, -8.0), 8.0)
            y_out[row_idx] = np.float32(yu if card["normalized_only"] else card["y_mean"] + card["y_sd"] * yu)
        elif h["task_id"] == 1:
            p1 = 1.0 / (1.0 + math.exp(-min(max(float(logits[0]), -12.0), 12.0)))
            idx = 1 if counter_uniform(seed, row_idx, STREAM_TARGET, 0) < p1 else 0
            y_out[row_idx] = np.float32(card["classes"][idx])
        else:
            ml = float(np.max(logits[:o_dim]))
            prs = [math.exp(min(max(float(logits[c]) - ml, -20.0), 0.0)) for c in range(o_dim)]
            pick = _sample_cat(prs, counter_uniform(seed, row_idx, STREAM_TARGET, 0))
            y_out[row_idx] = np.float32(card["classes"][pick])

        is_act = [False] * card["x_cols_raw"]
        for a, rj in enumerate(card["col_indices"]):
            is_act[rj] = True
            if miss[a]:
                x_out[row_idx, rj] = np.nan
            else:
                v = float(x_u[a]) if (card["normalized_only"] or card["group_kinds"][a] in (3, 4, 5)) else card["center_active"][a] + card["scale_active"][a] * float(x_u[a])
                if not card["normalized_only"] and card["is_int_active"][a]:
                    v = math.copysign(math.floor(abs(v) + 0.5), v)
                x_out[row_idx, rj] = np.float32(v)
        dc_i = 0
        for c in range(card["x_cols_raw"]):
            if not is_act[c]:
                x_out[row_idx, c] = np.float32(card["dropped_medians"][dc_i] if dc_i < len(card["dropped_medians"]) else 0.0)
                dc_i += 1

    return x_out, y_out


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
    z_grid = torch.linspace(-1.0, 1.0, KNOT_COUNT, device=x_in.device, dtype=x_in.dtype)
    # Count how many interior knot boundaries are strictly <= x_in (handles flat/discrete ties smoothly)
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
