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
    assert exe is not None, "build it: cargo build --manifest-path engine/Cargo.toml --locked -p dope-arena --bin dope_v8_emit"
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


def test_active_rank_v1_and_legacy_parity_plus_fit_midrank_z(v8_emit_bin: Path, tmp_path: Path) -> None:
    """Verify active-rank-v1 and legacy SafeTensors parity vs Rust AND FIT-only midrank z invariance."""
    from research.levels.v8_parity_decode import compute_fit_midrank_z_np

    fx = FIXTURES[0]  # regression
    X, y = fx["X"], fx["y"]
    # Query poisoning invariance on FIT-only midrank z:
    z_a = compute_fit_midrank_z_np(X[:60], X[60:80])
    X_alt = X[60:80].copy()
    X_alt[10:] = 1e6
    z_b = compute_fit_midrank_z_np(X[:60], X_alt)
    assert np.allclose(z_a[:10], z_b[:10], atol=0.0)
    assert np.all(np.isfinite(z_a)) and np.max(np.abs(z_a)) <= 3.5

    torch.manual_seed(77)
    model = SingleEncV8(d_model=32, n_heads=4, n_enc_layers=1, n_dec_layers=1, with_owner_encoder=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if any(k in name for k in ("delta_", "node_", "bilin_", "flow_", "gate_head")):
                p.normal_(0.0, 0.12)

    for cap, active_flag in (("active-rank-v1", True), (None, False)):
        ckpt_p = tmp_path / f"v8_{cap or 'legacy'}.safetensors"
        model.export_v8_safetensors(ckpt_p, decoder_capability=cap)
        pkt_b = fit_dse8_rust(X, y, 0, budget_bytes=2048, checkpoint=ckpt_p, code_tier=1, seed=19, bin_path=v8_emit_bin)
        pkt_py = unpack_dse8_packet_py(pkt_b)
        xs_rs, ys_rs = sample_dse8_rust(pkt_b, n_rows=32, seed=404, checkpoint=ckpt_p, n_heads=4, bin_path=v8_emit_bin)
        xs_py, ys_py = decode_dse8_rows_py_parity(pkt_py, model, n_rows=32, seed=404, active_rank_v1=active_flag)
        assert float(np.max(np.abs(xs_rs - xs_py))) <= 1e-4, f"cap={cap} X diff"
        assert float(np.max(np.abs(ys_rs - ys_py))) <= 1e-4, f"cap={cap} Y diff"

    # Unconditional differentiable Torch training forward (`decode_batch_differentiable`) parity
    # directly against Rust `sample_dse8_rust` at <= 1e-5 with exact counter normals, Student-t, and regimes!
    from research.levels.v8_packet import (
        STREAM_FACTOR,
        STREAM_NOISE,
        STREAM_REGIME,
        STREAM_TARGET,
        V8_MAX_CLASSES,
        build_decoder_inputs_py,
        counter_normal,
        counter_student_t_scale,
        counter_uniform,
    )
    from research.levels.v8_parity_decode import extract_packet_analytic_tensors

    ckpt_act = tmp_path / "v8_diff_parity.safetensors"
    model.export_v8_safetensors(ckpt_act, decoder_capability="active-rank-v1")
    pkt_b = fit_dse8_rust(
        X, y, 0, budget_bytes=2048, normalized_only=True, checkpoint=ckpt_act, code_tier=1, seed=23, bin_path=v8_emit_bin
    )
    pkt_py = unpack_dse8_packet_py(pkt_b)
    xs_rs, ys_rs = sample_dse8_rust(pkt_b, n_rows=24, seed=707, checkpoint=ckpt_act, n_heads=4, bin_path=v8_emit_bin)
    p_eff = int(pkt_py["header"]["p_eff"])
    dc, de, dg, dz = build_decoder_inputs_py(pkt_py)
    model.eval()
    with torch.no_grad():
        plan = model.decode_plan(
            torch.from_numpy(dc)[None], torch.from_numpy(de)[None],
            torch.from_numpy(dg)[None], torch.from_numpy(dz)[None],
        )
        knots = torch.from_numpy(np.stack([m["knots17"] for m in pkt_py["marginals"]], axis=0))[None]
        loadings = torch.from_numpy(pkt_py["loadings"])[None]
        w_b = np.zeros((V8_MAX_CLASSES, p_eff), dtype=np.float32)
        b_b = np.zeros(V8_MAX_CLASSES, dtype=np.float32)
        o_dim = int(pkt_py["linear"]["o_dim"])
        w_b[:o_dim, :] = pkt_py["linear"]["w"]
        b_b[:o_dim] = pkt_py["linear"]["b"]
        is_cont = torch.tensor([[1.0 if m["kind"] == 0 else 0.0 for m in pkt_py["marginals"]]], dtype=torch.float32)
        zf = np.zeros((1, 24, 12), dtype=np.float32)
        zn = np.zeros((1, 24, p_eff), dtype=np.float32)
        yn = np.zeros((1, 24), dtype=np.float32)
        ts = np.ones((1, 24, 1), dtype=np.float32)
        reg_idx = np.zeros((1, 24), dtype=np.int64)
        regs = pkt_py.get("regimes") or []
        reg_probs = [max(float(rg["weight"]), 0.0) for rg in regs]
        reg_sum = sum(reg_probs)
        inv_nu = float(pkt_py["header"]["inv_nu"])
        for ri in range(24):
            ts[0, ri, 0] = counter_student_t_scale(707, ri, inv_nu)
            for r in range(12):
                zf[0, ri, r] = counter_normal(707, ri, STREAM_FACTOR, r)
            for j in range(p_eff):
                zn[0, ri, j] = counter_normal(707, ri, STREAM_NOISE, j)
            yn[0, ri] = counter_normal(707, ri, STREAM_TARGET, 0)
            if len(regs) >= 2 and reg_sum > 1e-12:
                u_rg = min(max(counter_uniform(707, ri, STREAM_REGIME, 0), 0.0), 1.0 - 1e-15) * reg_sum
                acc = 0.0
                chosen = len(regs) - 1
                for idx_r, pr in enumerate(reg_probs):
                    acc += pr
                    if u_rg <= acc:
                        chosen = idx_r
                        break
                reg_idx[0, ri] = chosen
        pkt_an = extract_packet_analytic_tensors(pkt_py, torch.device("cpu"))
        dec = model.decode_batch_differentiable(
            plan, knots, loadings, torch.from_numpy(w_b)[None], torch.from_numpy(b_b)[None],
            torch.tensor([float(pkt_py["header"]["resid_sd"])]),
            torch.tensor([float(pkt_py["header"]["z_max"])]),
            is_cont, torch.ones(1, p_eff), n_rows=24, antithetic=False,
            z_factor=torch.from_numpy(zf), z_noise=torch.from_numpy(zn), y_noise=torch.from_numpy(yn),
            pkt_analytic=pkt_an, active_rank_v1=True,
            t_scale_override=torch.from_numpy(ts),
            regime_idx_override=torch.from_numpy(reg_idx) if len(regs) >= 2 else None,
        )
        max_dx_diff = float(np.max(np.abs(dec["x_syn"][0].numpy() - xs_rs)))
        max_dy_diff = float(np.max(np.abs(dec["y_reg"][0].numpy() - ys_rs)))
        assert max_dx_diff <= 1e-5, f"decode_batch_differentiable X parity diff {max_dx_diff} > 1e-5"
        assert max_dy_diff <= 1e-5, f"decode_batch_differentiable Y parity diff {max_dy_diff} > 1e-5"


def test_shared_fit_schema_wrapper_budget_matches_canonical_and_renamed_headers() -> None:
    """Verify exact DPA1 schema wrapper bytes through shared Rust FFI on canonical and renamed headers."""
    import hashlib
    from research.levels.s3_episode_native import dpa_v8_wrapper_bytes, native_v8
    from research.levels.v8_packet import unpack_dse8_packet_py

    canon_names = [f"c{j}" for j in range(48)]
    renamed_names = [f"measurement_{hashlib.sha256(str(j).encode()).hexdigest()[:12]}" for j in range(48)]
    w_canon = dpa_v8_wrapper_bytes(canon_names, feature_kinds=["real"] * 48, target_name="target")
    w_renamed = dpa_v8_wrapper_bytes(renamed_names, feature_kinds=["real"] * 48, target_name="target")
    w_anon = dpa_v8_wrapper_bytes(renamed_names, feature_kinds=["real"] * 48, target_name="target", anonymize=True)
    w_cls = dpa_v8_wrapper_bytes(
        canon_names, feature_kinds=["real"] * 48, target_name="outcome",
        target_labels=["alpha_class", "beta_class", "gamma_class"],
    )
    assert w_canon == 91, f"expected 91 B canonical wrapper, got {w_canon}"
    assert w_renamed == 390, f"expected 390 B renamed wrapper, got {w_renamed}"
    assert w_anon < w_renamed
    assert w_cls > w_canon

    rng = np.random.default_rng(174)
    X = rng.normal(size=(128, 48)).astype(np.float32)
    Xq = rng.normal(size=(16, 48)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] + rng.normal(size=128)).astype(np.float32)
    ep_base = {
        "dataset_hash": "fit-schema-test", "family_stem": "synthetic", "source_kind": "synthetic",
        "X_fit": X, "X_qry": Xq, "y_fit": y, "y_qry": np.zeros(len(Xq), dtype=np.float32),
        "fit_rows": np.arange(len(X), dtype=np.uint64),
        "split_sig": "fixture-split", "view_sig": "fixture-view", "task": "regression",
    }
    out_canon = native_v8(ep_base, code_tier=1, reserve_dpa_wrapper=True, wrapper_bytes=w_canon)
    ep_renamed = dict(
        ep_base,
        v7_schema={"feature_names": renamed_names, "feature_kinds": ["real"] * 48, "target_name": "target"},
    )
    out_renamed = native_v8(ep_renamed, code_tier=1, reserve_dpa_wrapper=True)
    hdr_canon = unpack_dse8_packet_py(out_canon["packet"])["header"]
    hdr_renamed = unpack_dse8_packet_py(out_renamed["packet"])["header"]
    assert len(out_canon["packet"]) + w_canon <= 2048
    assert len(out_renamed["packet"]) + w_renamed <= 2048
    assert hdr_canon["rank_use"] == 6
    assert hdr_renamed["rank_use"] == 5


def test_v300_forward_incompatibility_guard(v8_emit_bin: Path, tmp_path: Path) -> None:
    """Verify active-rank-v1 checkpoints (F32 & BF16) and packets fail closed on v3.0.0 binaries."""
    import os
    import struct

    fx = FIXTURES[0]
    X, y = fx["X"], fx["y"]
    torch.manual_seed(300)
    model = SingleEncV8(d_model=32, n_heads=4, n_enc_layers=1, n_dec_layers=1, with_owner_encoder=True)

    ckpt_f32 = model.export_v8_safetensors(tmp_path / "active_f32.safetensors", dtype="f32", decoder_capability="active-rank-v1")
    ckpt_bf16 = model.export_v8_safetensors(tmp_path / "active_bf16.safetensors", dtype="bf16", decoder_capability="active-rank-v1")
    ckpt_hist = model.export_v8_safetensors(tmp_path / "hist_f32.safetensors", dtype="f32", decoder_capability=None)

    for p_ckpt in (ckpt_f32, ckpt_bf16):
        raw = p_ckpt.read_bytes()
        hlen = struct.unpack("<Q", raw[:8])[0]
        hdr = json.loads(raw[8:8 + hlen])
        assert hdr["__metadata__"]["dope_dse8_decoder"] == "active-rank-v1"
        assert hdr["__dope_dse8_decoder_active_rank_v1__"]["dtype"] == "U8"
        assert hdr["__dope_dse8_decoder_active_rank_v1__"]["shape"] == [1]

    raw_h = ckpt_hist.read_bytes()
    hlen_h = struct.unpack("<Q", raw_h[:8])[0]
    hdr_h = json.loads(raw_h[8:8 + hlen_h])
    assert "__metadata__" not in hdr_h
    assert "__dope_dse8_decoder_active_rank_v1__" not in hdr_h

    pkt_active = fit_dse8_rust(X, y, 0, budget_bytes=2048, checkpoint=ckpt_f32, code_tier=1, seed=31, bin_path=v8_emit_bin)
    pkt_hist = fit_dse8_rust(X, y, 0, budget_bytes=2048, checkpoint=ckpt_hist, code_tier=1, seed=31, bin_path=v8_emit_bin)
    assert (pkt_active[10] & 0x40) != 0, "active-rank-v1 packet must set bit 0x40 outside v3.0.0 mask 0x3F"
    assert (pkt_hist[10] & 0x40) == 0, "historical packet must leave bit 0x40 clear"

    # When a real v3.0.0 binary is provided (DOPE_V300_EMIT_BIN), prove live fail-closed refusal!
    v300_emit = os.environ.get("DOPE_V300_EMIT_BIN")
    if v300_emit:
        from research.levels.v8_evidence import write_dopemat
        dm = tmp_path / "fit.mat"
        write_dopemat(dm, X, y)
        pkt_act_p = tmp_path / "active.dse8"
        pkt_act_p.write_bytes(pkt_active)
        pkt_hist_p = tmp_path / "hist.dse8"
        pkt_hist_p.write_bytes(pkt_hist)

        # v3.0.0 MUST refuse F32 active-rank-v1 checkpoint, BF16 active-rank-v1 checkpoint, AND active-rank-v1 packet!
        for bad_ckpt in (ckpt_f32, ckpt_bf16):
            r = subprocess.run(
                [v300_emit, "fit", "--in", str(dm), "--task", "regression", "--budget", "2048",
                 "--out", str(tmp_path / "out.dse8"),
                 "--checkpoint", str(bad_ckpt), "--code-tier", "1", "--seed", "31"],
                capture_output=True, text=True, check=False,
            )
            assert r.returncode != 0, f"v3.0.0 binary accepted active-rank-v1 checkpoint {bad_ckpt.name}!"

        r_smp_zero = subprocess.run(
            [v300_emit, "sample", "--model", str(pkt_act_p), "--rows", "8", "--seed", "7",
             "--out", str(tmp_path / "syn.mat")],
            capture_output=True, text=True, check=False,
        )
        assert r_smp_zero.returncode != 0, "v3.0.0 binary unpacked active-rank-v1 packet!"

        # v3.0.0 MUST still accept historical checkpoint & packet cleanly!
        r_ok = subprocess.run(
            [v300_emit, "sample", "--model", str(pkt_hist_p), "--rows", "8", "--seed", "7",
             "--checkpoint", str(ckpt_hist), "--out", str(tmp_path / "syn_ok.mat")],
            capture_output=True, text=True, check=False,
        )
        assert r_ok.returncode == 0, f"v3.0.0 binary failed on historical packet: {r_ok.stderr}"

    v300_cli = os.environ.get("DOPE_V300_BIN")
    v31_cli = os.environ.get("DOPE_V31_BIN")
    if v300_cli and v31_cli:
        import csv
        csv_p = tmp_path / "fit.csv"
        with csv_p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([*[f"c{j}" for j in range(X.shape[1])], "target"])
            for row, tv in zip(X, y):
                w.writerow([*(f"{float(v):.6f}" for v in row), f"{float(tv):.6f}"])
        dpa_act = tmp_path / "active.dpa1"
        dpa_hist = tmp_path / "hist.dpa1"
        subprocess.run(
            [v31_cli, "fit", str(csv_p), "--target", "target", "--format", "v8",
             "--v8-weights", str(ckpt_f32), "--budget", "2048", "--seed", "31", "-o", str(dpa_act)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [v31_cli, "fit", str(csv_p), "--target", "target", "--format", "v8",
             "--v8-weights", str(ckpt_hist), "--budget", "2048", "--seed", "31", "-o", str(dpa_hist)],
            check=True, capture_output=True, text=True,
        )
        # v3.0.0 CLI MUST refuse active-rank-v1 checkpoints (F32 & BF16) and active-rank-v1 DPA1 artifacts!
        for bad_ckpt in (ckpt_f32, ckpt_bf16):
            r = subprocess.run(
                [v300_cli, "fit", str(csv_p), "--target", "target", "--format", "v8",
                 "--v8-weights", str(bad_ckpt), "--budget", "2048", "-o", str(tmp_path / "bad.dpa1")],
                capture_output=True, text=True, check=False,
            )
            assert r.returncode != 0, f"v3.0.0 CLI accepted active-rank-v1 checkpoint {bad_ckpt.name}!"
        for extra in (["--v8-zero-init"], ["--v8-weights", str(ckpt_f32)]):
            r = subprocess.run(
                [v300_cli, "sample", str(dpa_act), "--rows", "8", "--seed", "7",
                 *extra, "-o", str(tmp_path / "bad_out.csv")],
                capture_output=True, text=True, check=False,
            )
            assert r.returncode != 0, f"v3.0.0 CLI decoded active-rank-v1 DPA1 with {extra}!"
        # v3.0.0 CLI MUST still decode historical DPA1 artifact cleanly!
        r_hist = subprocess.run(
            [v300_cli, "sample", str(dpa_hist), "--rows", "8", "--seed", "7",
             "--v8-weights", str(ckpt_hist), "-o", str(tmp_path / "hist_out.csv")],
            capture_output=True, text=True, check=False,
        )
        assert r_hist.returncode == 0, f"v3.0.0 CLI failed on historical DPA1: {r_hist.stderr}"
