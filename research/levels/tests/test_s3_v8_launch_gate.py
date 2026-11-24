"""20-step single-CPU Launch-Gate Smoke Test for `SingleEncV8` (DOPE v3.0).

Verifies end-to-end training & deployment readiness before any fleet GPU/CPU wave:
1. 10-step initial training -> finite losses/grads, >= 1 optimizer step, full checkpoint save
2. Exact resume for 10 more steps (total = 20 steps) -> contiguous step count, finite telemetry
3. Canonical F32 `rank <= 2` SafeTensors export -> native Rust `dope_v8_emit fit` + `sample` round-trip!
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.levels.s3_v8_eval import evaluate_dse8_cell
from research.levels.s3_v8_priors import sample_v8_prior_episode
from research.levels.s3_v8_trainer import parse_args, train_v8
from research.levels.v8_evidence import find_v8_emit_binary


def test_20_step_launch_gate_resume_and_rust_eval(tmp_path: Path) -> None:
    out_dir = tmp_path / "v8_smoke"

    # Phase A: Steps 1..10
    args_a = parse_args([
        "--out-dir", str(out_dir),
        "--device", "cpu",
        "--steps", "10",
        "--episodes-per-step", "2",
        "--n-fit", "128",
        "--n-qry", "96",
        "--n-syn", "128",
        "--d-model", "32",
        "--n-heads", "4",
        "--n-enc-layers", "1",
        "--n-dec-layers", "1",
        "--save-every", "5",
        "--seed", "42",
    ])
    res_a = train_v8(args_a)
    assert res_a["steps_done"] == 10
    assert res_a["last_step"] == 10
    assert Path(res_a["checkpoint_pt"]).is_file()
    assert Path(res_a["safetensors"]).is_file()

    # Phase B: Exact Resume Steps 11..20 (with PCGrad-Lite enabled!)
    args_b = parse_args([
        "--out-dir", str(out_dir),
        "--device", "cpu",
        "--steps", "20",
        "--episodes-per-step", "2",
        "--n-fit", "128",
        "--n-qry", "96",
        "--n-syn", "128",
        "--d-model", "32",
        "--n-heads", "4",
        "--n-enc-layers", "1",
        "--n-dec-layers", "1",
        "--use-pcgrad",
        "--save-every", "10",
        "--resume",
        "--seed", "42",
    ])
    res_b = train_v8(args_b)
    assert res_b["steps_done"] == 10
    assert res_b["last_step"] == 20

    lines = (out_dir / "train.jsonl").read_text().strip().splitlines()
    assert len(lines) == 20
    last = json.loads(lines[-1])
    assert last["step"] == 20 and last["grad_norm"] > 0.0

    # Phase C: Native Rust `dope_v8_emit` + `s3_v8_eval` end-to-end check with exported SafeTensors!
    exe = find_v8_emit_binary()
    if exe is not None:
        rng = np.random.default_rng(777)
        ep = sample_v8_prior_episode(rng, n_fit=180, n_qry=240, p=8, task_id=1, mechanism="xor")
        Xb, yb = ep["X_qry"][:120], ep["y_qry"][:120]
        Xt, yt = ep["X_qry"][120:], ep["y_qry"][120:]
        ev_res = evaluate_dse8_cell(
            ep["X_fit"], ep["y_fit"], Xb, yb, Xt, yt,
            task_id=1, n_classes=2, budget_bytes=2048,
            checkpoint=Path(res_b["safetensors"]), col_meta=ep["col_meta"], seed=19,
        )
        assert ev_res["packet_bytes"] <= 2048.0
        assert ev_res["illegal_onehot_rows"] == 0.0
        assert 0.0 <= ev_res["c2st_xy"] <= 1.0


def test_cuda_exclusive_guard_uuid_isolation_and_bf16_export(tmp_path: Path) -> None:
    import os
    import types
    import unittest.mock as mock

    from research.levels import s3_v8_trainer_io as io_mod
    from research.levels.v8_single_encoding import SingleEncV8

    fake_props = {
        0: types.SimpleNamespace(uuid="GPU-aaaa-0000"),
        1: types.SimpleNamespace(uuid="GPU-bbbb-1111"),
    }
    with mock.patch.object(io_mod.torch.cuda, "is_available", return_value=True), \
         mock.patch.object(io_mod.torch.cuda, "get_device_properties", side_effect=lambda i: fake_props[i]):
        # Foreign PID on GPU 0 blocks cuda:0 while allowing cuda:1!
        with mock.patch.object(io_mod.subprocess, "check_output", return_value="GPU-aaaa-0000, 999999, python\n"):
            io_mod.check_cuda_exclusive_guard("cuda:1")
            try:
                io_mod.check_cuda_exclusive_guard("cuda:0")
                raise AssertionError("cuda:0 should refuse foreign PID")
            except RuntimeError as exc:
                assert "GPU-aaaa-0000" in str(exc)
        # Prefix-matching PID must NOT match os.getpid()!
        my_pid = os.getpid()
        with mock.patch.object(io_mod.subprocess, "check_output", return_value=f"GPU-aaaa-0000, {my_pid}0, python\n"):
            try:
                io_mod.check_cuda_exclusive_guard("cuda:0")
                raise AssertionError("prefix PID should fail closed")
            except RuntimeError:
                pass
        # Missing nvidia-smi must fail closed!
        with mock.patch.object(io_mod.subprocess, "check_output", side_effect=FileNotFoundError("missing")):
            try:
                io_mod.check_cuda_exclusive_guard("cuda:0")
                raise AssertionError("missing nvidia-smi must fail closed")
            except RuntimeError as exc:
                assert "fail-closed" in str(exc)

    # BF16 + Tier-3 (256-dim) + 12-head export must be ~2x smaller than F32.
    m = SingleEncV8(d_model=48, n_heads=8, n_enc_layers=1, n_dec_layers=1, code_dim=256, ff_mult=4)
    p32 = m.export_v8_safetensors(tmp_path / "m32.safetensors", dtype="f32")
    p16 = m.export_v8_safetensors(tmp_path / "m16.safetensors", dtype="bf16")
    assert p16.stat().st_size * 2 < p32.stat().st_size + 16384


def test_analytic_packet_train_serve_parity_and_loss_fixes(tmp_path: Path) -> None:
    """Verify exact vectorized DSE8 analytic parity (effects/pairs/trees/temp/regimes) & loss fixes."""
    import math
    import numpy as np
    import torch

    from research.levels.s3_v8_losses import compute_v8_episode_loss
    from research.levels.v8_parity_decode import (
        eval_analytic_packet_logits_torch,
        extract_packet_analytic_tensors,
    )
    from research.levels.v8_single_encoding import SingleEncV8

    pkt = {
        "header": {"task_id": 1, "rank_use": 2, "inv_nu": 0.0, "z_max": 2.2, "resid_sd": 0.5, "p_eff": 4},
        "linear": {"o_dim": 1, "logit_tau": 1.65},
        "regimes": [{"weight": 0.6, "mean_f": [0.4, -0.2]}, {"weight": 0.4, "mean_f": [-0.5, 0.3]}],
        "effects": [{"col_idx": 1, "cls_idx": 0, "bins8": [0.1, 0.2, 0.3, -0.1, -0.2, 0.0, 0.4, 0.5]}],
        "pairs": [{"col_i": 0, "col_j": 2, "cls_idx": 0, "w_prod": 0.35, "w_xor": -0.25}],
        "trees": [{
            "cols": [0, 1, 3],
            "thrs": [7, 4, 10],
            "cls_idx": 0,
            "leaves": [-0.4, 0.1, 0.2, -0.1, 0.3, -0.2, 0.5, 0.6],
        }],
        "resid_model": {"cols": [], "log_sd_w": [], "q5_resid": [0.0] * 5},
    }
    pkt_an = extract_packet_analytic_tensors(pkt, torch.device("cpu"))
    torch.manual_seed(42)
    z_lat = torch.randn(1, 16, 4)
    vec_delta = eval_analytic_packet_logits_torch(z_lat, pkt_an)[0, :, 0].numpy()

    # Compare against exact scalar reference matching Rust `decode_dse8_row_at`!
    def _cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    for r in range(16):
        zr = z_lat[0, r].tolist()
        ref = 0.0
        b8 = min(int(math.floor(_cdf(zr[1]) * 8.0)), 7)
        ref += pkt["effects"][0]["bins8"][b8]
        zi, zj = min(max(zr[0], -3.0), 3.0), min(max(zr[2], -3.0), 3.0)
        xs = 1.0 if ((zi > 0.0) == (zj > 0.0)) else -1.0
        ref += 0.35 * min(max(zi * zj, -6.0), 6.0) - 0.25 * xs
        li = 0
        for d, (c, thr) in enumerate(zip([0, 1, 3], [7, 4, 10])):
            if min(int(math.floor(_cdf(zr[c]) * 16.0)), 15) > thr:
                li |= 1 << d
        ref += pkt["trees"][0]["leaves"][li]
        assert abs(vec_delta[r] - ref) < 1e-5, f"row {r}: {vec_delta[r]} vs {ref}"

    # Step-0 zero-init model must reproduce (linear + analytic) * logit_tau exactly!
    m = SingleEncV8(d_model=32, n_heads=4, n_enc_layers=1, n_dec_layers=1).eval()
    plan = m.decode_plan(
        torch.zeros(1, 4, 48), torch.zeros(1, 4, 4, 4), torch.zeros(1, 32), torch.zeros(1, 64)
    )
    W_base = torch.randn(1, 8, 4)
    b_base = torch.randn(1, 8)
    x_in = torch.randn(1, 16, 4)
    lg_full = m.predict_target_on_x(plan, W_base, b_base, x_in, z_in=z_lat, pkt_analytic=pkt_an)
    lg_lin = torch.einsum("bnp,bkp->bnk", x_in, W_base) + b_base[:, None, :]
    exp_0 = (lg_lin[0, :, 0] + torch.from_numpy(vec_delta)) * 1.65
    torch.testing.assert_close(lg_full[0, :, 0], exp_0, rtol=1e-5, atol=1e-5)

    # Verify FIT-only guided trees + joint SWD + unseen multiclass -1 query mask stay finite!
    xs = torch.randn(64, 10, requires_grad=True)
    lgs = torch.randn(64, 8, requires_grad=True)
    xf = torch.randn(64, 10)
    yf = torch.randint(0, 4, (64,)).float()
    xq = torch.randn(32, 10)
    yq = torch.randint(0, 4, (32,)).float()
    yq[:5] = -1.0  # Unseen query classes must be masked cleanly!
    ld = compute_v8_episode_loss(
        xs, lgs, xs[:, 0], torch.tensor(0.5), xf, yf, xq, yq,
        task_id=2, n_classes=4, weights={"joint_swd": 0.25, "anchor_nl": 0.0},
    )
    assert torch.isfinite(ld["total"]) and torch.isfinite(ld["l_swd"])
    ld["total"].backward()
    assert xs.grad is not None and torch.isfinite(xs.grad).all()
    assert np.isfinite(float(ld["l_swd"].item())) and float(ld["l_swd"].item()) > 0.0


def test_no_delta_w_and_historical_forward_ablation_switches(tmp_path: Path) -> None:
    """Verify Addendum G `--no-delta-w` (C-dW) & `--historical-forward` (L) switches + resume/export guards."""
    import os
    import struct
    import subprocess
    import torch

    from research.levels.v8_single_encoding import SingleEncV8

    def _read_st(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
        raw = path.read_bytes()
        hlen = struct.unpack("<Q", raw[:8])[0]
        hdr = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
        body = raw[8 + hlen :]
        arrs: dict[str, np.ndarray] = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s0, s1 = v["data_offsets"]
            dt = np.float32 if v["dtype"] == "F32" else np.uint8
            arrs[k] = np.frombuffer(body[s0:s1], dtype=dt).reshape(v["shape"]).copy()
        return hdr, arrs

    exe = find_v8_emit_binary()
    rust_flags = ["--use-rust-packet"] if exe is not None else []
    base_cfg = [
        "--device", "cpu",
        "--episodes-per-step", "2",
        "--n-fit", "96",
        "--n-qry", "64",
        "--n-syn", "64",
        "--d-model", "32",
        "--n-heads", "4",
        "--n-enc-layers", "1",
        "--n-dec-layers", "1",
        "--warmup-steps", "1",
        "--seed", "11",
        *rust_flags,
    ]

    # 1. --no-delta-w: uninterrupted 8 steps vs 4 + resume 4 (same --steps 8 schedule!) must agree byte-for-byte!
    dir_full = tmp_path / "ndw_full"
    dir_res = tmp_path / "ndw_res"
    train_v8(parse_args(["--out-dir", str(dir_full), "--steps", "8", "--save-every", "4", "--no-delta-w", *base_cfg]))
    train_v8(parse_args([
        "--out-dir", str(dir_res), "--steps", "8", "--stop-at-step", "4", "--save-every", "4", "--no-delta-w", *base_cfg
    ]))
    train_v8(parse_args([
        "--out-dir", str(dir_res), "--steps", "8", "--save-every", "4", "--no-delta-w", "--resume", *base_cfg
    ]))

    for fname in ("model_v8.safetensors", "model_v8_ema.safetensors"):
        b_full = (dir_full / fname).read_bytes()
        b_res = (dir_res / fname).read_bytes()
        assert b_full == b_res, f"{fname} uninterrupted vs resumed mismatch under --no-delta-w"
        hdr, arrs = _read_st(dir_full / fname)
        assert hdr.get("__metadata__") == {"dope_dse8_decoder": "active-rank-v1"}
        assert "__dope_dse8_decoder_active_rank_v1__" in hdr
        assert float(np.max(np.abs(arrs["delta_W.weight"]))) == 0.0
        assert float(np.max(np.abs(arrs["delta_W.bias"]))) == 0.0
        assert float(np.max(np.abs(arrs["delta_L.weight"]))) > 0.0
        assert float(np.max(np.abs(arrs["node_leaf.weight"]))) > 0.0
        assert float(np.max(np.abs(arrs["gate_head.weight"]))) > 0.0

    # Fail-closed resume mismatch (dropping --no-delta-w) & exporter zero-assertion guard.
    try:
        train_v8(parse_args(["--out-dir", str(dir_res), "--steps", "9", "--resume", *base_cfg]))
        raise AssertionError("resume without --no-delta-w must fail closed")
    except ValueError as exc:
        assert "no_delta_w" in str(exc)

    m_chk = SingleEncV8(d_model=32, n_heads=4, n_enc_layers=1, n_dec_layers=1)
    with torch.no_grad():
        m_chk.delta_W.bias.fill_(0.01)
    try:
        m_chk.export_v8_safetensors(tmp_path / "bad_ndw.safetensors", assert_zero_delta_w=True)
        raise AssertionError("export with non-zero delta_W must fail closed under assert_zero_delta_w")
    except ValueError as exc:
        assert "delta_W" in str(exc)

    # 2. --historical-forward (L) vs corrected (C): omits active-rank-v1 markers, diverges from C, resumes byte-exact!
    dir_hist = tmp_path / "hist_full"
    dir_hist_r = tmp_path / "hist_res"
    dir_corr = tmp_path / "corr_full"
    train_v8(parse_args([
        "--out-dir", str(dir_hist), "--steps", "6", "--save-every", "3", "--historical-forward", *base_cfg
    ]))
    train_v8(parse_args([
        "--out-dir", str(dir_hist_r), "--steps", "6", "--stop-at-step", "3", "--save-every", "3",
        "--historical-forward", *base_cfg,
    ]))
    train_v8(parse_args([
        "--out-dir", str(dir_hist_r), "--steps", "6", "--save-every", "3", "--historical-forward", "--resume", *base_cfg
    ]))
    train_v8(parse_args(["--out-dir", str(dir_corr), "--steps", "6", "--save-every", "6", *base_cfg]))

    assert (dir_hist / "model_v8.safetensors").read_bytes() == (dir_hist_r / "model_v8.safetensors").read_bytes()
    assert (dir_hist / "model_v8_ema.safetensors").read_bytes() == (dir_hist_r / "model_v8_ema.safetensors").read_bytes()
    hdr_h, arrs_h = _read_st(dir_hist / "model_v8.safetensors")
    hdr_c, arrs_c = _read_st(dir_corr / "model_v8.safetensors")
    hdr_h_ema, _ = _read_st(dir_hist / "model_v8_ema.safetensors")
    assert "__metadata__" not in hdr_h and "__metadata__" not in hdr_h_ema
    assert "__dope_dse8_decoder_active_rank_v1__" not in hdr_h
    assert "__dope_dse8_decoder_active_rank_v1__" not in hdr_h_ema
    assert hdr_c.get("__metadata__") == {"dope_dse8_decoder": "active-rank-v1"}
    if exe is not None:
        diff_w = float(np.max(np.abs(arrs_h["delta_W.weight"] - arrs_c["delta_W.weight"])))
        assert diff_w > 1e-5, f"expected --historical-forward and C to diverge on Rust packets, diff={diff_w}"

    try:
        train_v8(parse_args(["--out-dir", str(dir_hist_r), "--steps", "7", "--resume", *base_cfg]))
        raise AssertionError("resume without --historical-forward must fail closed")
    except ValueError as exc:
        assert "historical_forward" in str(exc)

    # Native Rust check: Q4 loads C-dW; optional live v3.0.0 binary loads L and refuses C-dW!
    if exe is not None:
        rng = np.random.default_rng(909)
        ep = sample_v8_prior_episode(rng, n_fit=140, n_qry=160, p=6, task_id=1, mechanism="xor")
        ev = evaluate_dse8_cell(
            ep["X_fit"], ep["y_fit"], ep["X_qry"][:80], ep["y_qry"][:80], ep["X_qry"][80:], ep["y_qry"][80:],
            task_id=1, n_classes=2, budget_bytes=2048,
            checkpoint=dir_full / "model_v8_ema.safetensors", col_meta=ep["col_meta"], seed=23,
        )
        assert ev["packet_bytes"] <= 2048.0
        v300_emit = os.environ.get("DOPE_V300_EMIT_BIN", "")
        if v300_emit and Path(v300_emit).is_file():
            from research.levels.v8_evidence import write_dopemat
            dm = tmp_path / "l_chk.dopemat"
            write_dopemat(dm, ep["X_fit"], ep["y_fit"])
            ok_l = subprocess.run(
                [
                    v300_emit, "fit", "--in", str(dm), "--task", "binary", "--out", str(tmp_path / "l.dse8"),
                    "--checkpoint", str(dir_hist / "model_v8_ema.safetensors"),
                ],
                capture_output=True, text=True, check=False,
            )
            assert ok_l.returncode == 0, f"v3.0.0 should load --historical-forward checkpoint: {ok_l.stderr}"
            bad_c = subprocess.run(
                [
                    v300_emit, "fit", "--in", str(dm), "--task", "binary", "--out", str(tmp_path / "c.dse8"),
                    "--checkpoint", str(dir_full / "model_v8_ema.safetensors"),
                ],
                capture_output=True, text=True, check=False,
            )
            assert bad_c.returncode != 0, "v3.0.0 must refuse --no-delta-w active-rank-v1 checkpoint"
            assert "F32" in (bad_c.stderr + bad_c.stdout) or "U8" in (bad_c.stderr + bad_c.stdout)

    # 3. Historical y_reg clamp bypass (pkt_analytic=None) vs native +/-8 clamp (pkt_analytic!=None),
    #    plus --deterministic flag resume enforcement!
    m_y = SingleEncV8(d_model=32, n_heads=4, n_enc_layers=1, n_dec_layers=1).eval()
    pl_y = m_y.decode_plan(
        torch.zeros(1, 2, 48), torch.zeros(1, 2, 2, 4), torch.zeros(1, 32), torch.zeros(1, 64)
    )
    knots_y = torch.linspace(-2.0, 2.0, 17).view(1, 1, 17).expand(1, 2, 17).clone()
    load_y = torch.zeros(1, 2, 12)
    w_y = torch.zeros(1, 8, 2)
    b_y = torch.zeros(1, 8)
    b_y[0, 0] = 10.5
    rsd_y = torch.tensor([0.5])
    zmax_y = torch.tensor([2.0])
    cont_y = torch.ones(1, 2)
    mask_y = torch.ones(1, 2)
    zf_y = torch.zeros(1, 4, 12)
    zn_y = torch.zeros(1, 4, 2)
    yn_y = torch.zeros(1, 4)
    out_hist = m_y.decode_batch_differentiable(
        pl_y, knots_y, load_y, w_y, b_y, rsd_y, zmax_y, cont_y, mask_y,
        n_rows=4, antithetic=False, z_factor=zf_y, z_noise=zn_y, y_noise=yn_y,
        pkt_analytic=None, active_rank_v1=False,
    )
    out_corr = m_y.decode_batch_differentiable(
        pl_y, knots_y, load_y, w_y, b_y, rsd_y, zmax_y, cont_y, mask_y,
        n_rows=4, antithetic=False, z_factor=zf_y, z_noise=zn_y, y_noise=yn_y,
        pkt_analytic={"task_id": 0, "o_dim": 1, "rank_use": 0, "inv_nu": 0.0},
        active_rank_v1=True,
    )
    assert float(out_hist["y_reg"][0, 0].item()) > 10.0, "historical forward must not clamp y_reg at 8.0"
    assert float(out_corr["y_reg"][0, 0].item()) == 8.0, "corrected forward must clamp y_reg to +/-8.0"

    dir_det = tmp_path / "det_chk"
    train_v8(parse_args([
        "--out-dir", str(dir_det), "--steps", "4", "--stop-at-step", "2", "--save-every", "2",
        "--deterministic", *base_cfg,
    ]))
    try:
        train_v8(parse_args(["--out-dir", str(dir_det), "--steps", "4", "--resume", *base_cfg]))
        raise AssertionError("resume without --deterministic must fail closed")
    except ValueError as exc:
        assert "deterministic" in str(exc)
