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
        max_eff = max(
            float(np.max(np.abs(plan_np["delta_knots"]))),
            float(np.max(np.abs(plan_np["delta_L"]))),
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

    noise_sd = np.sqrt(np.maximum(1.0 - np.sum(L_eff ** 2, axis=1), 1e-4)).astype(np.float32)

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
