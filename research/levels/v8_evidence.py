"""Pure-function FIT-only Evidence Bank bridge & NumPy mirror for DOPE v3 (`dope.evidence-layout/v1`).

Binding constraints (09:50Z):
- Zero per-table caching: every episode computes evidence strictly from its fresh FIT split.
- Structurally FIT-only: query/test rows never enter any function here.
- Supports native Rust extraction via `dope_v8_emit evidence` (canonical source of truth)
  and a fast vectorised NumPy extractor for high-throughput training episodes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any

import numpy as np

COL_DIM = 128
EDGE_DIM = 16
CLASS_DIM = 16
CLASS_COL_DIM = 4
GROUP_DIM = 8
GLOB_DIM = 160
DOPEMAT_MAGIC = b"DOPEMAT1"

_LAYOUT_PATH = Path(__file__).resolve().parents[2] / "contracts" / "evidence-layout-v1.json"


def evidence_layout_sha256() -> str:
    return hashlib.sha256(_LAYOUT_PATH.read_bytes()).hexdigest()


def find_v8_emit_binary() -> Path | None:
    env_p = os.environ.get("DOPE_V8_EMIT_BIN")
    if env_p and Path(env_p).is_file():
        return Path(env_p)
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "engine" / "target" / "release" / "dope_v8_emit",
        repo_root / "engine" / "target" / "debug" / "dope_v8_emit",
        Path("/home/ubuntu/dope/.agent/target-v3-dse8/release/dope_v8_emit"),
        Path("/home/ubuntu/dope/.agent/target-v3-dse8/debug/dope_v8_emit"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def write_dopemat(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    X32 = np.asarray(X, dtype=np.float32)
    y32 = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    n, p = X32.shape
    body = np.ascontiguousarray(np.concatenate([X32, y32], axis=1), dtype="<f4")
    hdr = struct.pack("<8sIIII", DOPEMAT_MAGIC, n, p, 1, 0)
    path.write_bytes(hdr + body.tobytes())


def read_dopemat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != DOPEMAT_MAGIC:
        raise ValueError("invalid DOPEMAT1 header")
    n, p, yc, _ = struct.unpack("<IIII", raw[8:24])
    if yc != 1:
        raise ValueError(f"expected y_cols=1, got {yc}")
    arr = np.frombuffer(raw[24:], dtype="<f4").reshape(n, p + 1)
    return arr[:, :p].copy(), arr[:, p].copy()


def extract_evidence_rust(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    task_id: int,
    n_classes: int = 0,
    tier: str = "t2",
    target_free: bool = False,
    col_meta: list[dict[str, Any]] | None = None,
    seed: int = 0,
    bin_path: Path | None = None,
) -> dict[str, Any]:
    """Call canonical pure-Rust `dope_v8_emit evidence` strictly on FIT rows."""
    exe = bin_path or find_v8_emit_binary()
    if exe is None:
        raise FileNotFoundError("dope_v8_emit binary not found; build with cargo build -p dope-arena --bin dope_v8_emit")
    task_name = {0: "regression", 1: "binary", 2: "multiclass"}[int(task_id)]
    with tempfile.TemporaryDirectory(prefix="dope_v8_ev_") as td:
        tdp = Path(td)
        in_p = tdp / "fit.mat"
        out_p = tdp / "ev.json"
        write_dopemat(in_p, X_fit, y_fit)
        cmd = [
            str(exe), "evidence",
            "--in", str(in_p),
            "--task", task_name,
            "--classes", str(int(n_classes)),
            "--tier", tier,
            "--seed", str(int(seed)),
            "--out", str(out_p),
        ]
        if target_free:
            cmd.append("--target-free")
        if col_meta is not None:
            meta_p = tdp / "meta.json"
            meta_p.write_text(json.dumps(col_meta))
            cmd.extend(["--col-meta", str(meta_p)])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        raw = json.loads(out_p.read_text())
    return {
        "p": int(raw["p"]),
        "k_eff": int(raw["k_eff"]),
        "layout_sha256": str(raw["layout_sha256"]),
        "col": np.asarray(raw["col"], dtype=np.float32),
        "col_mask": np.asarray(raw["col_mask"], dtype=np.uint8),
        "edge": np.asarray(raw["edge"], dtype=np.float32).reshape(int(raw["p"]), int(raw["p"]), EDGE_DIM),
        "edge_mask": np.asarray(raw["edge_mask"], dtype=np.uint16).reshape(int(raw["p"]), int(raw["p"])),
        "class_tok": np.asarray(raw["class_tok"], dtype=np.float32),
        "class_col": np.asarray(raw["class_col"], dtype=np.float32),
        "group_tok": np.asarray(raw["group_tok"], dtype=np.float32),
        "glob": np.asarray(raw["glob"], dtype=np.float32),
        "glob_mask": np.asarray(raw["glob_mask"], dtype=np.uint8),
        "col_composite_signal": np.asarray(raw["col_composite_signal"], dtype=np.float64),
    }


def _squash(v: np.ndarray | float) -> np.ndarray | float:
    return np.clip(v / (1.0 + np.abs(v)), -1.0, 1.0)


def extract_evidence_fast_numpy(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    task_id: int,
    n_classes: int = 1,
    seed: int = 0,
    drop_tier_prob: float = 0.0,
) -> dict[str, np.ndarray]:
    """Fast vectorised FIT-only Evidence extractor (layout-aligned 128/16/160) for batched training.

    Supports evidence-tier dropout (zeroing T2/T3 slots with probability `drop_tier_prob`)
    so `SingleEncV8` degrades gracefully under variable inference compute tiers.
    """
    X = np.asarray(X_fit, dtype=np.float32)
    y = np.asarray(y_fit, dtype=np.float64).reshape(-1)
    n, p = X.shape
    k_eff = 1 if task_id == 0 else (2 if task_id == 1 else max(2, min(int(n_classes), 8)))
    col = np.zeros((p, COL_DIM), dtype=np.float32)
    edge = np.zeros((p, p, EDGE_DIM), dtype=np.float32)
    glob = np.zeros((GLOB_DIM,), dtype=np.float32)

    fin = np.isfinite(X)
    miss_rate = 1.0 - fin.mean(axis=0)
    Xc = np.where(fin, X, 0.0).astype(np.float64)
    col[:, 0] = 1.0
    col[:, 48] = miss_rate.astype(np.float32)
    col[:, 51] = float(1.0 / math.sqrt(max(n, 1)))
    col[:, 52] = float(n / (n + 20.0))

    # Quantiles & support
    u_floor = min(0.20, 10.0 / max(n + 1.0, 20.0))
    z_max = float(np.clip(_norm_ppf_scalar(1.0 - u_floor), 1.2, 2.6))
    z_grid = np.linspace(-z_max, z_max, 17, dtype=np.float64)
    u_grid = 0.5 * (1.0 + np.vectorize(math.erf)(z_grid / math.sqrt(2.0)))
    qs = np.quantile(Xc, u_grid, axis=0).T  # [p, 17]
    med = np.median(Xc, axis=0)
    iqr = np.quantile(Xc, 0.75, axis=0) - np.quantile(Xc, 0.25, axis=0)
    sd = np.std(Xc, axis=0)
    rsd = np.maximum(np.maximum(iqr / 1.349, sd), 1e-6)
    span = np.maximum(qs[:, -1] - qs[:, 0], 0.0)
    is_const = span < 1e-6
    col[:, 6] = is_const.astype(np.float32)
    col[:, 16:33] = np.clip((qs - med[:, None]) / rsd[:, None] - z_grid[None, :], -4.0, 4.0).astype(np.float32)

    # Normal scores & correlation matrix
    ranks = np.argsort(np.argsort(Xc, axis=0), axis=0).astype(np.float64)
    u_rk = (ranks + 0.5) / max(n, 1)
    Z = np.vectorize(_norm_ppf_scalar)(np.clip(u_rk, 1e-5, 1.0 - 1e-5))
    Z[:, is_const] = 0.0
    Z_std = Z / np.maximum(np.std(Z, axis=0, keepdims=True), 1e-6)
    R = (Z_std.T @ Z_std) / max(n, 1)
    np.fill_diagonal(R, 1.0)
    R = np.clip(R, -0.985, 0.985)
    vals, vecs = np.linalg.eigh(R)
    ord_e = np.argsort(vals)[::-1]
    vals = np.maximum(vals[ord_e], 0.0)
    vecs = vecs[:, ord_e]
    r_use = min(p, 12)
    L12 = np.zeros((p, 12), dtype=np.float64)
    for k in range(r_use):
        sgn = -1.0 if float(np.sum(vecs[:, k] ** 3)) < 0.0 else 1.0
        L12[:, k] = sgn * vecs[:, k] * math.sqrt(float(vals[k]))
    norms = np.linalg.norm(L12, axis=1, keepdims=True)
    L12 = np.where(norms > 0.99, L12 * (0.99 / np.maximum(norms, 1e-6)), L12)
    col[:, 56:68] = L12.astype(np.float32)
    col[:, 68] = np.clip(1.0 - np.sum(L12 ** 2, axis=1), 0.02, 1.0).astype(np.float32)

    edge[:, :, 0] = R.astype(np.float32)
    edge[:, :, 12] = np.clip(R - L12 @ L12.T, -1.0, 1.0).astype(np.float32)
    edge[:, :, 15] = (np.abs(R) > 0.70).astype(np.float32)

    # Target associations & signed XOR quadrant contrast
    y_c = y - float(np.mean(y))
    y_sd = max(float(np.std(y_c)), 1e-6)
    yn = y_c / y_sd
    r_y = (Z_std.T @ yn) / max(n, 1)
    col[:, 78] = np.clip(r_y, -1.0, 1.0).astype(np.float32)
    col[:, 95] = np.clip(np.abs(r_y), 0.0, 1.0).astype(np.float32)

    # Vectorised pair product & XOR quadrant contrast on standardized residual
    lam = max(0.05 * n, 2.0)
    w_lin = np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), Xc.T @ yn)
    resid = yn - Xc @ w_lin
    resid = resid / max(float(np.std(resid)), 1e-4)
    for b in range(8):
        lo_u, hi_u = b / 8.0, (b + 1) / 8.0
        m_b = (u_rk >= lo_u) & (u_rk < hi_u)
        cnt = m_b.sum(axis=0).astype(np.float64)
        eb = (m_b * resid[:, None]).sum(axis=0) / (cnt + 8.0)
        col[:, 96 + b] = np.tanh(eb).astype(np.float32)

    if p <= 48 and n >= 40:
        sgn_z = np.where(Z > 0.0, 1.0, -1.0)
        # Signed XOR contrast: E[resid * sgn(z_i) * sgn(z_j)] * sqrt(n)
        xor_mat = ((sgn_z * resid[:, None]).T @ sgn_z) / max(n, 1) * math.sqrt(n)
        np.fill_diagonal(xor_mat, 0.0)
        edge[:, :, 10] = _squash(xor_mat / 3.0).astype(np.float32)
        prod_mat = ((Z_std * resid[:, None]).T @ Z_std) / max(n, 1) * math.sqrt(n)
        np.fill_diagonal(prod_mat, 0.0)
        edge[:, :, 9] = _squash(prod_mat / 3.0).astype(np.float32)

    # Global metadata & spectrum
    glob[min(task_id, 2)] = 1.0
    glob[12] = float(math.log2(max(min(n, 16384), 16)) / 14.0)
    glob[13] = float(min(p / 48.0, 2.0))
    glob[15] = float(k_eff / 8.0)
    eig_s = vals / max(float(vals.sum()), 1e-9)
    glob[16 : 16 + min(12, p)] = eig_s[: min(12, p)].astype(np.float32)
    glob[54] = float(miss_rate.mean())
    glob[55] = float(is_const.mean())
    glob[64] = float(np.clip(1.0 - float(np.mean((yn - Xc @ w_lin) ** 2)), -1.0, 1.0))

    if drop_tier_prob > 0.0:
        rng = np.random.default_rng(seed ^ 0xD809_1234)
        if rng.random() < drop_tier_prob:
            col[:, 116:] = 0.0
            edge[:, :, 11] = 0.0
            glob[104:] = 0.0

    return {
        "col": np.nan_to_num(np.clip(col, -4.0, 4.0), nan=0.0),
        "edge": np.nan_to_num(np.clip(edge, -4.0, 4.0), nan=0.0),
        "glob": np.nan_to_num(np.clip(glob, -4.0, 4.0), nan=0.0),
    }


def _norm_ppf_scalar(p: float) -> float:
    """Acklam standard normal inverse CDF (matching Rust `dope_core::stats::norm_ppf`)."""
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = ((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
                   + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
               + 133.14166789178437745) * r + 3.387132872796366608
        den = ((((((5226.4952788528545610 * r + 28729.085735721942674) * r + 39307.895800092710610) * r
                   + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.18700749205790830) * r
               + 42.313330701600911252) * r + 1.0
        return float(q * num / den)
    rt = p if p < 0.5 else 1.0 - p
    s = math.sqrt(-math.log(rt))
    num = ((((((7.7454501427834140764e-4 * s + 0.022723844989269184583) * s + 0.24178072517745061177) * s
               + 1.2704582524523683826) * s + 3.6478483247632046050) * s + 5.7694972214606914055) * s
           + 4.6303378461565452959) * s + 1.4234371107496835773
    den = ((((((1.0507500716444168432e-9 * s + 5.4759380849953449460e-4) * s + 0.015198666563616457197) * s
               + 0.14810397642748007459) * s + 0.68976733498510000455) * s + 1.6763848301838038494) * s
           + 2.0531916266377588219) * s + 1.0
    val = num / den
    return float(-val if p < 0.5 else val)
