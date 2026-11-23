"""DSE8 v1 wire parser, Rust `dope_v8_emit` bridge, and exact row-local counter PRNG mirror.

Provides:
- Rust CLI wrappers (`fit_dse8_rust`, `sample_dse8_rust`, `truncate_dse8_rust`, `inspect_dse8_rust`)
- Exact Python unpacker (`unpack_dse8_packet_py`) matching `engine/dope-core/src/single_encoding/v8/packet.rs`
- Exact bit-identical counter PRNG (`counter_u64`, `counter_uniform`, `counter_normal`,
  `counter_student_t_scale`) matching `rng_v8.rs`
- Exact Receiver Decoder input builder (`build_decoder_inputs_py`) matching `model.rs`!
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any
import zlib

import numpy as np

from research.levels.v8_evidence import find_v8_emit_binary, read_dopemat, write_dopemat

DSE8_MAGIC = b"DSE8"
DSE8_VERSION = 1
DSE8_HEADER_BYTES = 24
KNOT_COUNT = 17
V8_MAX_RANK = 12
V8_MAX_CLASSES = 8
V8_MAX_COLS = 64
DEC_COL_DIM = 48
DEC_EDGE_DIM = 4
DEC_GLOB_DIM = 32
DEC_CODE_DIM = 64

STREAM_REGIME = 0
STREAM_FACTOR = 1
STREAM_NOISE = 2
STREAM_ATOM = 3
STREAM_MISSING = 4
STREAM_TARGET = 5
STREAM_STUDENT_T = 6

MASK64 = (1 << 64) - 1


def counter_u64(seed: int, row_idx: int, stream_id: int, dim: int) -> int:
    """Bit-exact Python mirror of Rust `rng_v8::counter_u64`."""
    x = (
        (seed & MASK64)
        + ((row_idx * 0x9E3779B97F4A7C15) & MASK64)
        + ((stream_id * 0xBF58476D1CE4E5B9) & MASK64)
        + ((dim * 0x94D049BB133111EB) & MASK64)
    ) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def counter_uniform(seed: int, row_idx: int, stream_id: int, dim: int) -> float:
    bits = counter_u64(seed, row_idx, stream_id, dim) >> 11
    return (bits + 0.5) * (1.0 / float(1 << 53))


def counter_normal(seed: int, row_idx: int, stream_id: int, dim: int) -> float:
    u1 = counter_uniform(seed, row_idx, stream_id, dim * 2)
    u2 = counter_uniform(seed, row_idx, stream_id, dim * 2 + 1)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def counter_student_t_scale(seed: int, row_idx: int, inv_nu: float) -> float:
    if inv_nu <= 1e-4:
        return 1.0
    nu = min(max(1.0 / float(inv_nu), 2.2), 500.0)
    g = counter_normal(seed, row_idx, STREAM_STUDENT_T, 0)
    term = 1.0 - 2.0 / (9.0 * nu) + g * math.sqrt(2.0 / (9.0 * nu))
    chi2_over_nu = max(term, 0.20) ** 3
    return float(min(max(1.0 / math.sqrt(chi2_over_nu), 0.4), 3.5))


def _f16_bits_to_f32(bits: int) -> float:
    return float(np.frombuffer(struct.pack("<H", bits), dtype="<f2")[0])


def _bf16_bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _decode_tree_scale(u: int) -> float:
    lo, hi = math.log(1e-4), math.log(0.10)
    return float(math.exp(lo + (u / 255.0) * (hi - lo)))


def _reconstruct_knots17(lo: float, span: float, rel_q: list[int]) -> np.ndarray:
    kl = max(len(rel_q), 2)
    step = (KNOT_COUNT - 1) // max(kl - 1, 1)
    knots = np.zeros(KNOT_COUNT, dtype=np.float32)
    for ki in range(KNOT_COUNT):
        seg = min(ki // step, kl - 2)
        frac = (ki - seg * step) / float(step)
        r0 = rel_q[seg] / 255.0
        r1 = rel_q[seg + 1] / 255.0
        knots[ki] = np.float32(lo + span * (r0 + frac * (r1 - r0)))
    for ki in range(1, KNOT_COUNT):
        if knots[ki] < knots[ki - 1]:
            knots[ki] = knots[ki - 1]
    return knots


def unpack_dse8_packet_py(raw: bytes) -> dict[str, Any]:
    """Fail-closed Python unpacker matching Rust `dope_core::single_encoding::v8::unpack_dse8_packet`."""
    if len(raw) < DSE8_HEADER_BYTES + 4 or raw[:4] != DSE8_MAGIC or raw[4] != DSE8_VERSION:
        raise ValueError("invalid DSE8 header")
    body = raw[:-4]
    exp_crc = struct.unpack("<I", raw[-4:])[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != exp_crc:
        raise ValueError("DSE8 crc32 mismatch")

    task_id, k_eff, p_eff = raw[5], raw[6], raw[7]
    x_cols_raw = struct.unpack("<H", raw[8:10])[0]
    flags, rank_use, knot_level = raw[10], raw[11], raw[12]
    log2_nfit_norm = raw[13] / 255.0
    z_max = struct.unpack("<H", raw[14:16])[0] / 10000.0
    resid_sd = struct.unpack("<H", raw[16:18])[0] / 10000.0
    p1_prior = raw[18] / 255.0
    inv_nu = (raw[19] / 255.0) * 0.45
    section_count = raw[20]
    code_tier = raw[21]

    normalized_only = bool(flags & 0x01)
    has_col_map = bool(flags & 0x02)
    has_custom_classes = bool(flags & 0x04)
    has_dropped_meds = bool(flags & 0x08)
    o_dim = k_eff if task_id == 2 else 1

    card: dict[str, Any] = {
        "normalized_only": normalized_only,
        "x_cols_raw": x_cols_raw,
        "col_indices": list(range(p_eff)),
        "group_kinds": [0] * p_eff,
        "center_active": [0.0] * p_eff,
        "scale_active": [1.0] * p_eff,
        "is_int_active": [False] * p_eff,
        "dropped_medians": [],
        "y_mean": 0.0,
        "y_sd": 1.0,
        "classes": [] if task_id == 0 else list(range(k_eff)),
    }
    marginals: list[dict[str, Any]] = []
    loadings = np.zeros((p_eff, V8_MAX_RANK), dtype=np.float32)
    linear: dict[str, Any] = {"o_dim": o_dim, "w": np.zeros((o_dim, p_eff), dtype=np.float32), "b": np.zeros(o_dim, dtype=np.float32), "logit_tau": 1.0}
    nominal_groups: list[dict[str, Any]] = []
    regimes: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    trees: list[dict[str, Any]] = []
    resid_model: dict[str, Any] = {"cols": [], "log_sd_w": [], "q5_resid": [0.0] * 5}
    code_packed = b""

    cur = DSE8_HEADER_BYTES
    last_tag = 0
    weights_fp: bytes | None = None
    for _ in range(section_count):
        tag = body[cur]
        if tag <= last_tag:
            raise ValueError("DSE8 duplicate or out-of-order section tag")
        last_tag = tag
        slen = struct.unpack("<H", body[cur + 1 : cur + 3])[0]
        cur += 3
        pl = body[cur : cur + slen]
        cur += slen

        if tag == 0x01:  # TAG_SCALE
            sc = 0
            if task_id == 0:
                card["y_mean"], card["y_sd"] = struct.unpack("<ff", pl[sc : sc + 8])
                sc += 8
            elif has_custom_classes:
                card["classes"] = list(struct.unpack(f"<{k_eff}i", pl[sc : sc + 4 * k_eff]))
                sc += 4 * k_eff
            if has_col_map:
                card["col_indices"] = list(struct.unpack(f"<{p_eff}H", pl[sc : sc + 2 * p_eff]))
                sc += 2 * p_eff
            for a in range(p_eff):
                fb = pl[sc]
                sc += 1
                gk = fb & 0x0F
                card["group_kinds"][a] = gk
                card["is_int_active"][a] = bool(fb & 0x10)
                if gk not in (3, 4, 5):
                    card["center_active"][a] = struct.unpack("<f", pl[sc : sc + 4])[0]
                    card["scale_active"][a] = _bf16_bits_to_f32(struct.unpack("<H", pl[sc + 4 : sc + 6])[0])
                    sc += 6
            if has_dropped_meds:
                nd = x_cols_raw - p_eff
                card["dropped_medians"] = [
                    _bf16_bits_to_f32(struct.unpack("<H", pl[sc + 2 * i : sc + 2 * i + 2])[0])
                    for i in range(nd)
                ]
        elif tag == 0x02:  # TAG_MARG
            mc = 0
            for _ in range(p_eff):
                fl = pl[mc]
                kind = fl & 0x07
                tail_family = (fl >> 3) & 0x03
                is_int = bool(fl & (1 << 5))
                miss_rate = pl[mc + 1] / 255.0
                lo = _f16_bits_to_f32(struct.unpack("<H", pl[mc + 2 : mc + 4])[0])
                span = _f16_bits_to_f32(struct.unpack("<H", pl[mc + 4 : mc + 6])[0])
                mc += 6
                if kind == 4:
                    rel_q = [0] * knot_level
                    knots17 = np.full(KNOT_COUNT, lo, dtype=np.float32)
                else:
                    rel_q = list(pl[mc : mc + knot_level])
                    mc += knot_level
                    knots17 = _reconstruct_knots17(lo, max(span, 1e-5), rel_q)
                lat_q0, lat_orig, h_step, h_ex = 0.0, 0.0, 0.0, 0.0
                atom_vals, atom_probs = [], []
                if kind == 1:
                    lat_q0 = _f16_bits_to_f32(struct.unpack("<H", pl[mc : mc + 2])[0])
                    lat_orig = _f16_bits_to_f32(struct.unpack("<H", pl[mc + 2 : mc + 4])[0])
                    h_step = _f16_bits_to_f32(struct.unpack("<H", pl[mc + 4 : mc + 6])[0])
                    h_ex = pl[mc + 6] / 255.0
                    mc += 7
                elif kind in (2, 3):
                    ka = pl[mc]
                    mc += 1
                    for _ in range(ka):
                        atom_vals.append(_f16_bits_to_f32(struct.unpack("<H", pl[mc : mc + 2])[0]))
                        atom_probs.append(pl[mc + 2] / 255.0)
                        mc += 3
                    if kind == 2 and sum(atom_probs) > 1e-6:
                        s_p = sum(atom_probs)
                        atom_probs = [p / s_p for p in atom_probs]
                marginals.append({
                    "kind": kind, "tail_family": tail_family, "is_int": is_int,
                    "miss_rate": miss_rate, "lo": lo, "span": span, "knots17": knots17,
                    "lattice_q0": lat_q0, "lattice_origin": lat_orig,
                    "heaping_step": h_step, "heaping_excess": h_ex,
                    "atom_vals": atom_vals, "atom_probs": atom_probs,
                })
        elif tag == 0x03:  # TAG_COPULA
            arr = np.frombuffer(pl, dtype=np.int8).reshape(p_eff, rank_use)
            loadings[:, :rank_use] = arr.astype(np.float32) / 127.0
        elif tag == 0x04:  # TAG_LINEAR
            logit_tau = min(max(_f16_bits_to_f32(struct.unpack("<H", pl[:2])[0]), 0.2), 5.0)
            linear["logit_tau"] = logit_tau
            lc = 2
            for c in range(o_dim):
                sc_w, bv = struct.unpack("<ff", pl[lc : lc + 8])
                lc += 8
                linear["b"][c] = bv
                qi = np.frombuffer(pl[lc : lc + p_eff], dtype=np.int8)
                linear["w"][c] = qi.astype(np.float32) * np.float32(sc_w)
                lc += p_eff
        elif tag == 0x05:  # TAG_NOMINAL
            ng = pl[0]
            nc = 1
            for _ in range(ng):
                w_g, bp = pl[nc], pl[nc + 1] / 255.0
                nc += 2
                cols_g, probs_g = [], []
                for _ in range(w_g):
                    cols_g.append(pl[nc])
                    probs_g.append(pl[nc + 1] / 255.0)
                    nc += 2
                nominal_groups.append({"cols": cols_g, "probs": probs_g, "baseline_prob": bp})
        elif tag == 0x06:  # TAG_REGIMES
            m_r, r_reg = pl[0], pl[1]
            rc = 2
            for _ in range(m_r):
                wt = pl[rc] / 255.0
                rc += 1
                mf = [struct.unpack("<b", pl[rc + k : rc + k + 1])[0] / 50.0 for k in range(r_reg)]
                rc += r_reg
                regimes.append({"weight": wt, "mean_f": mf})
            sw = sum(r["weight"] for r in regimes)
            if sw > 1e-6:
                for r in regimes:
                    r["weight"] /= sw
        elif tag == 0x07:  # TAG_EFFECTS
            for off in range(0, len(pl), 12):
                ch = pl[off : off + 12]
                sc_e = _f16_bits_to_f32(struct.unpack("<H", ch[2:4])[0])
                b8 = np.frombuffer(ch[4:12], dtype=np.int8).astype(np.float32) * np.float32(sc_e)
                effects.append({"col_idx": ch[0], "cls_idx": ch[1], "bins8": b8})
        elif tag == 0x08:  # TAG_PAIRS
            for off in range(0, len(pl), 5):
                ci, cj, cl, wp, wx = struct.unpack("<BBBbb", pl[off : off + 5])
                pairs.append({"col_i": ci, "col_j": cj, "cls_idx": cl, "w_prod": wp / 100.0, "w_xor": wx / 100.0})
        elif tag == 0x09:  # TAG_TREES
            for off in range(0, len(pl), 14):
                ch = pl[off : off + 14]
                cols_t = [ch[0], ch[1], ch[2]]
                thrs_t = [ch[3] & 0x0F, (ch[3] >> 4) & 0x0F, ch[4] & 0x0F]
                cls_t = (ch[4] >> 4) & 0x0F
                sc_t = _decode_tree_scale(ch[5])
                lvs = np.frombuffer(ch[6:14], dtype=np.int8).astype(np.float32) * np.float32(sc_t)
                trees.append({"cls_idx": cls_t, "cols": cols_t, "thrs": thrs_t, "leaves": lvs})
        elif tag == 0x0A:  # TAG_RESID
            kr = pl[0]
            cols_r, w_r = [], []
            for i in range(kr):
                cols_r.append(pl[1 + 2 * i])
                w_r.append(struct.unpack("<b", pl[2 + 2 * i : 3 + 2 * i])[0] / 100.0)
            q5 = [struct.unpack("<b", pl[1 + 2 * kr + i : 2 + 2 * kr + i])[0] / 100.0 for i in range(5)]
            resid_model = {"cols": cols_r, "log_sd_w": w_r, "q5_resid": q5}
        elif tag == 0x0B:  # TAG_CODE
            code_packed = pl
        elif tag == 0x0C:  # TAG_WEIGHTS_FP
            if len(pl) != 8:
                raise ValueError("DSE8 invalid weights_fp len")
            weights_fp = pl
        else:
            raise ValueError(f"DSE8 unknown section tag {tag}")

    code_vec = np.zeros(DEC_CODE_DIM, dtype=np.float32)
    for b_idx, bv in enumerate(code_packed):
        if 2 * b_idx < DEC_CODE_DIM:
            code_vec[2 * b_idx] = (bv & 0x0F) / 7.5 - 1.0
        if 2 * b_idx + 1 < DEC_CODE_DIM:
            code_vec[2 * b_idx + 1] = ((bv >> 4) & 0x0F) / 7.5 - 1.0

    return {
        "header": {
            "task_id": task_id, "k_eff": k_eff, "p_eff": p_eff, "x_cols_raw": x_cols_raw,
            "flags": flags, "rank_use": rank_use, "knot_level": knot_level,
            "log2_nfit_norm": log2_nfit_norm, "z_max": z_max, "resid_sd": resid_sd,
            "p1_prior": p1_prior, "inv_nu": inv_nu, "code_tier": code_tier,
        },
        "card": card, "marginals": marginals, "loadings": loadings, "linear": linear,
        "nominal_groups": nominal_groups, "regimes": regimes, "effects": effects,
        "pairs": pairs, "trees": trees, "resid_model": resid_model, "code_vec": code_vec,
        "weights_fp": weights_fp,
    }


def build_decoder_inputs_py(pkt: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact mirror of Rust `model::build_decoder_inputs` -> `(col [P,48], edge [P,P,4], glob [32], code [64])`."""
    h = pkt["header"]
    p = h["p_eff"]
    col = np.zeros((p, DEC_COL_DIM), dtype=np.float32)
    for j in range(p):
        m = pkt["marginals"][j]
        col[j, :KNOT_COUNT] = m["knots17"]
        col[j, 17:29] = pkt["loadings"][j]
        for c in range(min(pkt["linear"]["o_dim"], V8_MAX_CLASSES)):
            col[j, 29 + c] = pkt["linear"]["w"][c, j]
        col[j, 37] = m["kind"] / 4.0
        col[j, 38] = m["tail_family"] / 2.0
        col[j, 39] = 1.0 if m["is_int"] else 0.0
        col[j, 40] = m["miss_rate"]
        col[j, 41] = math.log(1.0 + m["span"])
        col[j, 42] = h["z_max"]
        col[j, 43] = h["knot_level"] / float(KNOT_COUNT)
        col[j, 44] = len(m["atom_vals"]) / 8.0
        col[j, 45] = m["heaping_excess"]

    edge = np.zeros((p, p, DEC_EDGE_DIM), dtype=np.float32)
    L = pkt["loadings"]
    edge[:, :, 0] = np.clip(L @ L.T, -1.0, 1.0)
    for g in pkt["nominal_groups"]:
        for a in g["cols"]:
            for b in g["cols"]:
                if a < p and b < p:
                    edge[a, b, 1] = 1.0
    for pr in pkt["pairs"]:
        i, j = pr["col_i"], pr["col_j"]
        if i < p and j < p:
            edge[i, j, 2] = edge[j, i, 2] = pr["w_prod"]
            edge[i, j, 3] = edge[j, i, 3] = pr["w_xor"]

    glob = np.zeros(DEC_GLOB_DIM, dtype=np.float32)
    glob[min(h["task_id"], 2)] = 1.0
    glob[3] = 1.0 if pkt["card"]["normalized_only"] else 0.0
    if h["task_id"] == 1:
        glob[4] = 1.0 - h["p1_prior"]
        glob[5] = h["p1_prior"]
    elif h["task_id"] == 2:
        k = h["k_eff"]
        bv = pkt["linear"]["b"][:k]
        ex = np.exp(np.clip(bv - bv.max(), -20.0, 0.0))
        glob[4 : 4 + min(k, 8)] = ex / max(float(ex.sum()), 1e-6)
    glob[12] = h["log2_nfit_norm"]
    glob[13] = p / float(V8_MAX_COLS)
    glob[14] = h["resid_sd"]
    glob[15] = h["k_eff"] / float(V8_MAX_CLASSES)
    glob[16] = h["inv_nu"]
    glob[17] = len(pkt["regimes"]) / 8.0
    glob[18] = len(pkt["effects"]) / 24.0
    glob[19] = len(pkt["pairs"]) / 24.0
    glob[20] = len(pkt["trees"]) / 40.0
    glob[21] = len(pkt["nominal_groups"]) / 8.0
    glob[22] = h["code_tier"] / 3.0
    glob[23] = pkt["linear"]["logit_tau"]

    return col, edge, glob, pkt["code_vec"].astype(np.float32)


def fit_dse8_rust(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    task_id: int,
    budget_bytes: int = 2048,
    normalized_only: bool = False,
    checkpoint: Path | None = None,
    code_tier: int = 0,
    col_meta: list[dict[str, Any]] | None = None,
    seed: int = 0,
    bin_path: Path | None = None,
) -> bytes:
    """Fit a DSE8 v1 wire packet via canonical pure-Rust `dope_v8_emit fit`."""
    exe = bin_path or find_v8_emit_binary()
    if exe is None:
        raise FileNotFoundError("dope_v8_emit binary not found")
    task_name = {0: "regression", 1: "binary", 2: "multiclass"}[int(task_id)]
    with tempfile.TemporaryDirectory(prefix="dope_v8_fit_") as td:
        tdp = Path(td)
        in_p, out_p = tdp / "fit.mat", tdp / "pkt.bin"
        write_dopemat(in_p, X_fit, y_fit)
        cmd = [
            str(exe), "fit",
            "--in", str(in_p),
            "--task", task_name,
            "--budget", str(int(budget_bytes)),
            "--code-tier", str(int(code_tier)),
            "--seed", str(int(seed)),
            "--out", str(out_p),
        ]
        if normalized_only:
            cmd.append("--normalized-only")
        if checkpoint is not None:
            cmd.extend(["--checkpoint", str(checkpoint)])
        if col_meta is not None:
            mp = tdp / "meta.json"
            mp.write_text(json.dumps(col_meta))
            cmd.extend(["--col-meta", str(mp)])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_p.read_bytes()


def sample_dse8_rust(
    pkt_bytes: bytes,
    n_rows: int,
    seed: int,
    checkpoint: Path | None = None,
    n_heads: int = 4,
    bin_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode rows strictly via canonical pure-Rust `dope_v8_emit sample`."""
    exe = bin_path or find_v8_emit_binary()
    if exe is None:
        raise FileNotFoundError("dope_v8_emit binary not found")
    with tempfile.TemporaryDirectory(prefix="dope_v8_smp_") as td:
        tdp = Path(td)
        pkt_p, out_p = tdp / "pkt.bin", tdp / "syn.mat"
        pkt_p.write_bytes(pkt_bytes)
        cmd = [
            str(exe), "sample",
            "--model", str(pkt_p),
            "--rows", str(int(n_rows)),
            "--seed", str(int(seed)),
            "--n-heads", str(int(n_heads)),
            "--out", str(out_p),
        ]
        if checkpoint is not None:
            cmd.extend(["--checkpoint", str(checkpoint)])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return read_dopemat(out_p)
