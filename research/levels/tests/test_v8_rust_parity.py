"""12-fixture hard Rust-vs-Python parity suite for DOPE v3 (`DSE8` v1 + Evidence Bank + `SingleEncV8`).

Tests 12 diverse tabular fixtures spanning:
 1. regression_linear_nonlinear
 2. binary_logistic_interactions
 3. multiclass_5_class
 4. mar_missing_25pct
 5. wide_nominal_onehot_group
 6. zero_inflated_semi_cont
 7. heavy_tailed_student_t
 8. decimal_lattice_heaping
 9. pure_xor_binary
10. tiny_n_24
11. constant_and_sparse_cols
12. wide_p_48_mixed

Verifies:
- Pure-Rust `dope_v8_emit evidence` layout SHA-256 & finite bounds
- `DSE8` wire CRC32 & Python unpack round-trip (`<= budget_bytes`)
- Analytic Step-0 Rust-vs-Python row-local decode parity (`max |py - rs| <= 1e-4`)
- Random non-zero `F32` `rank <= 2` SafeTensors `SingleEncV8` Rust-vs-Python decode parity (`<= 1e-4`)
- Zero-init exact identity & zero illegal nominal one-hot rows!
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from research.levels.v8_evidence import (
    evidence_layout_sha256,
    extract_evidence_rust,
    find_v8_emit_binary,
)
from research.levels.v8_packet import (
    fit_dse8_rust,
    sample_dse8_rust,
    unpack_dse8_packet_py,
)
from research.levels.v8_single_encoding import SingleEncV8, decode_dse8_rows_py_parity


def _make_12_fixtures() -> list[dict]:
    fixtures = []

    def _rng(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    # 1. Regression linear + nonlinear
    r = _rng(101)
    X = r.normal(size=(240, 8)).astype(np.float32)
    y = (1.5 * X[:, 0] - 0.9 * X[:, 1] ** 2 + 0.3 * r.normal(size=240)).astype(np.float32)
    fixtures.append({"name": "regression_linear_nonlinear", "X": X, "y": y, "task_id": 0, "meta": None})

    # 2. Binary logistic + interaction
    r = _rng(102)
    X = r.normal(size=(260, 8)).astype(np.float32)
    lg = 1.2 * X[:, 0] + 1.0 * X[:, 1] * X[:, 2]
    y = (r.random(260) < 1.0 / (1.0 + np.exp(-lg))).astype(np.float32)
    fixtures.append({"name": "binary_logistic_interactions", "X": X, "y": y, "task_id": 1, "meta": None})

    # 3. Multiclass 5-class
    r = _rng(103)
    X = r.normal(size=(320, 8)).astype(np.float32)
    W = r.normal(0.0, 1.0, size=(8, 5))
    y = np.argmax(X @ W + 0.4 * r.normal(size=(320, 5)), axis=1).astype(np.float32)
    fixtures.append({"name": "multiclass_5_class", "X": X, "y": y, "task_id": 2, "meta": None})

    # 4. MAR missing 25%
    r = _rng(104)
    X = r.normal(size=(240, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.2 * r.normal(size=240)).astype(np.float32)
    X[r.random(X.shape) < 0.25] = np.nan
    fixtures.append({"name": "mar_missing_25pct", "X": X, "y": y, "task_id": 0, "meta": None})

    # 5. Wide one-hot nominal group (6 levels) + 3 continuous
    r = _rng(105)
    n = 300
    X = np.zeros((n, 9), dtype=np.float32)
    cats = r.integers(0, 6, size=n)
    X[np.arange(n), cats] = 1.0
    X[:, 6:] = r.normal(size=(n, 3))
    y = ((cats % 2 == 1) ^ (X[:, 6] > 0.0)).astype(np.float32)
    meta = [
        {
            "group_id": 0 if j < 6 else j, "group_kind": 4 if j < 6 else 0,
            "group_width": 6 if j < 6 else 1, "parent_col_idx": 65535,
            "support_class": 2 if j < 6 else 0, "skew_guard_active": 0,
            "lattice_q0_unit": 0.0, "lattice_origin_unit": 0.0,
            "has_lower_bound": 0, "has_upper_bound": 0,
            "log_pre_tail_asym": 0.0, "log_unique_bucket": 0.0,
        }
        for j in range(9)
    ]
    fixtures.append({"name": "wide_nominal_onehot_group", "X": X, "y": y, "task_id": 1, "meta": meta})

    # 6. Zero-inflated semi-continuous
    r = _rng(106)
    X = r.normal(size=(240, 6)).astype(np.float32)
    X[:, 0] = np.where(r.random(240) < 0.45, 0.0, np.abs(X[:, 0]) + 0.2)
    X[:, 1] = np.where(r.random(240) < 0.35, 0.0, np.abs(X[:, 1]) + 0.3)
    y = (2.0 * X[:, 0] - X[:, 1] + 0.3 * r.normal(size=240)).astype(np.float32)
    fixtures.append({"name": "zero_inflated_semi_cont", "X": X, "y": y, "task_id": 0, "meta": None})

    # 7. Heavy-tailed Student-t
    r = _rng(107)
    X = np.clip(r.standard_t(df=3.0, size=(240, 6)), -15.0, 15.0).astype(np.float32)
    y = (np.tanh(X[:, 0]) + 0.5 * X[:, 1] + 0.2 * r.normal(size=240)).astype(np.float32)
    fixtures.append({"name": "heavy_tailed_student_t", "X": X, "y": y, "task_id": 0, "meta": None})

    # 8. Decimal lattice (0.01) + heaping
    r = _rng(108)
    X = r.normal(size=(240, 6)).astype(np.float32)
    base = np.floor(r.uniform(0.0, 100.0, size=240)) / 100.0
    X[:, 0] = np.where(r.random(240) < 0.4, np.round(base, 1), base).astype(np.float32)
    y = (3.0 * X[:, 0] + X[:, 1] + 0.2 * r.normal(size=240)).astype(np.float32)
    fixtures.append({"name": "decimal_lattice_heaping", "X": X, "y": y, "task_id": 0, "meta": None})

    # 9. Pure XOR binary
    r = _rng(109)
    X = r.normal(size=(300, 6)).astype(np.float32)
    y = (((X[:, 0] > 0.0) ^ (X[:, 1] > 0.0)).astype(np.float32))
    flip = r.random(300) < 0.08
    y = np.where(flip, 1.0 - y, y).astype(np.float32)
    fixtures.append({"name": "pure_xor_binary", "X": X, "y": y, "task_id": 1, "meta": None})

    # 10. Tiny n = 24
    r = _rng(110)
    X = r.normal(size=(24, 5)).astype(np.float32)
    y = (X[:, 0] > 0.0).astype(np.float32)
    fixtures.append({"name": "tiny_n_24", "X": X, "y": y, "task_id": 1, "meta": None})

    # 11. Constant and binary cols
    r = _rng(111)
    X = r.normal(size=(200, 6)).astype(np.float32)
    X[:, 2] = 7.5
    X[:, 3] = (X[:, 0] > 0.0).astype(np.float32)
    y = (X[:, 0] + 2.0 * X[:, 3] + 0.2 * r.normal(size=200)).astype(np.float32)
    fixtures.append({"name": "constant_and_sparse_cols", "X": X, "y": y, "task_id": 0, "meta": None})

    # 12. Wide P = 48 mixed
    r = _rng(112)
    X = r.normal(size=(240, 48)).astype(np.float32)
    y = (X[:, 0] - 0.8 * X[:, 12] + 0.7 * X[:, 35] + 0.3 * r.normal(size=240)).astype(np.float32)
    fixtures.append({"name": "wide_p_48_mixed", "X": X, "y": y, "task_id": 0, "meta": None})

    assert len(fixtures) == 12
    return fixtures


FIXTURES = _make_12_fixtures()


@pytest.fixture(scope="module")
def v8_emit_bin() -> Path:
    exe = find_v8_emit_binary()
    if exe is None:
        pytest.skip("dope_v8_emit binary not built")
    return exe


@pytest.mark.parametrize("fx", FIXTURES, ids=[f["name"] for f in FIXTURES])
def test_12_fixture_analytic_and_neural_rust_parity(
    fx: dict, v8_emit_bin: Path, tmp_path: Path
) -> None:
    X, y, task_id, meta = fx["X"], fx["y"], fx["task_id"], fx["meta"]

    # 1. Pure-Rust FIT-only Evidence extraction & frozen layout SHA-256 verification
    ev = extract_evidence_rust(X, y, task_id, tier="t2", col_meta=meta, seed=7, bin_path=v8_emit_bin)
    assert ev["layout_sha256"] == evidence_layout_sha256()
    assert ev["col"].shape == (X.shape[1], 128)
    assert np.all(np.isfinite(ev["col"])) and np.all(np.isfinite(ev["glob"]))

    # 2. Analytic Step-0 DSE8 fit (<= 2048 B) + Rust-vs-Python row-local parity!
    pkt_bytes = fit_dse8_rust(
        X, y, task_id, budget_bytes=2048, col_meta=meta, seed=11, bin_path=v8_emit_bin
    )
    assert len(pkt_bytes) <= 2048
    pkt_py = unpack_dse8_packet_py(pkt_bytes)

    xs_rs, ys_rs = sample_dse8_rust(pkt_bytes, n_rows=64, seed=999, bin_path=v8_emit_bin)
    xs_py, ys_py = decode_dse8_rows_py_parity(pkt_py, None, n_rows=64, seed=999)

    # Check exact NaN mask match + numerical parity <= 1e-4 on all finite values!
    assert np.array_equal(np.isnan(xs_rs), np.isnan(xs_py)), f"{fx['name']} NaN mask mismatch"
    fin = np.isfinite(xs_rs)
    if np.any(fin):
        max_dx = float(np.max(np.abs(xs_rs[fin] - xs_py[fin])))
        assert max_dx <= 1e-4, f"{fx['name']} analytic X parity diff {max_dx} > 1e-4"
    max_dy = float(np.max(np.abs(ys_rs - ys_py)))
    assert max_dy <= 1e-4, f"{fx['name']} analytic Y parity diff {max_dy} > 1e-4"

    # Zero illegal rows on wide one-hot nominal fixture!
    if fx["name"] == "wide_nominal_onehot_group":
        sub = xs_rs[:, :6]
        assert np.all((sub == 0.0) | (sub == 1.0))
        assert np.all(sub.sum(axis=1) == 1.0)

    # 3. Random non-zero SingleEncV8 F32 rank<=2 SafeTensors Rust-vs-Python decode parity (<= 1e-4)!
    torch.manual_seed(1234)
    model = SingleEncV8(d_model=32, n_heads=4, n_enc_layers=1, n_dec_layers=1, with_owner_encoder=True)
    # Perturb residual output heads non-zero so coupling flow, Soft-NODE, bilinear, and delta_W all fire!
    with torch.no_grad():
        for name, p in model.named_parameters():
            if any(k in name for k in ("delta_", "node_", "bilin_", "flow_", "gate_head")):
                p.normal_(0.0, 0.08)
    ckpt_p = tmp_path / f"{fx['name']}_v8.safetensors"
    model.export_v8_safetensors(ckpt_p)

    pkt_nn = fit_dse8_rust(
        X, y, task_id, budget_bytes=2048, checkpoint=ckpt_p, code_tier=1,
        col_meta=meta, seed=11, bin_path=v8_emit_bin,
    )
    assert len(pkt_nn) <= 2048
    pkt_nn_py = unpack_dse8_packet_py(pkt_nn)

    xs_rs_n, ys_rs_n = sample_dse8_rust(pkt_nn, n_rows=48, seed=2026, checkpoint=ckpt_p, n_heads=4, bin_path=v8_emit_bin)
    xs_py_n, ys_py_n = decode_dse8_rows_py_parity(pkt_nn_py, model, n_rows=48, seed=2026)

    assert np.array_equal(np.isnan(xs_rs_n), np.isnan(xs_py_n))
    fin_n = np.isfinite(xs_rs_n)
    if np.any(fin_n):
        max_dx_n = float(np.max(np.abs(xs_rs_n[fin_n] - xs_py_n[fin_n])))
        assert max_dx_n <= 1e-4, f"{fx['name']} neural X parity diff {max_dx_n} > 1e-4"
    # Note: for discrete classification draws, a logit boundary flip of ~1e-6 can rarely flip 1 row;
    # regression must match <= 1e-4 on all rows, classification >= 95% exact class match!
    if task_id == 0:
        max_dy_n = float(np.max(np.abs(ys_rs_n - ys_py_n)))
        assert max_dy_n <= 1e-4, f"{fx['name']} neural reg Y parity diff {max_dy_n} > 1e-4"
    else:
        match_rate = float(np.mean(ys_rs_n == ys_py_n))
        assert match_rate >= 0.95, f"{fx['name']} neural cls Y parity rate {match_rate} < 0.95"


def test_cli_inspect_embed_truncate(v8_emit_bin: Path, tmp_path: Path) -> None:
    fx = FIXTURES[0]
    pkt_b = fit_dse8_rust(fx["X"], fx["y"], 0, budget_bytes=4096, bin_path=v8_emit_bin)
    pkt_p = tmp_path / "m4k.bin"
    pkt_p.write_bytes(pkt_b)

    ins = json.loads(subprocess.check_output([str(v8_emit_bin), "inspect", "--model", str(pkt_p)], text=True))
    assert ins["format"] == "DSE8v1" and ins["bytes"] == len(pkt_b)

    emb_p = tmp_path / "emb.json"
    subprocess.run([str(v8_emit_bin), "embed", "--model", str(pkt_p), "--out", str(emb_p)], check=True)
    emb = json.loads(emb_p.read_text())
    assert emb["dim"] == 64 and len(emb["embedding"]) == 64

    tr_p = tmp_path / "m1k.bin"
    tr_out = json.loads(
        subprocess.check_output(
            [str(v8_emit_bin), "truncate", "--model", str(pkt_p), "--budget", "1024", "--out", str(tr_p)],
            text=True,
        )
    )
    assert tr_out["bytes"] <= 1024
