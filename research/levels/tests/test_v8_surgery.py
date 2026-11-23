"""Unit & native Rust loading tests for `SingleEncV8` model surgery (`v7-to-v8` & `repair-v8`)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
import torch

from research.levels.s3_v8_trainer import parse_args, train_v8
from research.levels.v7_single_encoding import SingleEncV7
from research.levels.v8_model_surgery import surgery_repair_v8, surgery_v7_to_v8
from research.levels.v8_packet import DEC_CODE_DIM, fit_dse8_rust, sample_dse8_rust
from research.levels.v8_single_encoding import SingleEncV8

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_V7 = REPO_ROOT / "models" / "single-enc-v7" / "model.safetensors"


def _find_v8_emit_bin() -> Path | None:
    env = os.environ.get("DOPE_V8_EMIT")
    for cand in (
        *((Path(env),) if env else ()),
        REPO_ROOT / "engine" / "target" / "release" / "dope_v8_emit",
    ):
        if cand.is_file():
            return cand
    return None


def test_v7_to_v8_surgery_preserves_backbone_and_breaks_saddle_point(tmp_path: Path) -> None:
    assert SHIPPED_V7.is_file(), f"missing shipped V7 checkpoint: {SHIPPED_V7}"
    out_st = tmp_path / "v8_from_v7.safetensors"
    info = surgery_v7_to_v8(SHIPPED_V7, out_st, n_heads=4, n_enc_layers=3, target_alpha=0.25, seed=7)
    assert info["d_model"] == 160
    assert info["n_enc_layers"] == 3
    assert info["n_dec_layers"] == 3
    assert info["norms"]["bilin_u.weight"] > 0.0
    assert info["norms"]["bilin_v.weight"] > 0.0
    assert info["norms"]["bilin_w.weight"] == 0.0

    # Load into SingleEncV8 and verify exact V7 block transfer + nonzero step-1 bilin_w grad
    v8 = SingleEncV8(d_model=160, n_heads=4, n_enc_layers=3, n_dec_layers=3, code_dim=DEC_CODE_DIM)
    v8.load_state_dict(st_load(str(out_st)), strict=False)
    v8.eval()

    # Verify exact backbone equivalence with V7 blocks & 64/48 width-feature rescale
    v7_sd = st_load(str(SHIPPED_V7))
    v8_sd = st_load(str(out_st))
    assert float((v8_sd["dec_glob_in.weight"][:, 13] - v7_sd["glob_in.weight"][:, 13] * (64.0 / 48.0)).abs().max().item()) < 1e-5
    v7 = SingleEncV7(d_model=160, n_heads=4, n_layers=3)
    v7_blocks_sd = {k: v for k, v in v7_sd.items() if k.startswith(("blocks.", "final_norm."))}
    v7.load_state_dict(v7_blocks_sd, strict=False)
    v7.eval()
    torch.manual_seed(11)
    h0 = torch.randn(1, 6, 160)
    mask = torch.ones(1, 6)
    edge = torch.randn(1, 6, 6, 4)
    with torch.no_grad():
        hv7 = h0.clone()
        for blk in v7.blocks:
            hv7 = blk(hv7, mask)
        hv7 = v7.final_norm(hv7)
        hv8 = h0.clone()
        for blk in v8.dec_blocks:
            hv8 = blk(hv8, edge, mask)
        hv8 = v8.dec_final_norm(hv8)
    assert float((hv7 - hv8).abs().max().item()) < 1e-5

    # Step-0 bilinear contribution is exact 0, yet bilin_w receives nonzero gradient immediately
    v8.train()
    dec_col = torch.randn(1, 6, 48)
    dec_glob = torch.randn(1, 32)
    code_vec = torch.zeros(1, 64)
    plan = v8.decode_plan(dec_col, edge, dec_glob, code_vec, mask)
    assert abs(float(plan["alpha_table"][0].item()) - 0.25) < 1e-3
    x_in = torch.randn(1, 16, 6)
    base_W = torch.zeros(1, 8, 6)
    base_b = torch.zeros(1, 8)
    logits = v8.predict_target_on_x(plan, base_W, base_b, x_in)
    logits.pow(2).mean().backward()
    assert v8.bilin_w.weight.grad is not None
    assert float(v8.bilin_w.weight.grad.norm().item()) > 1e-5

    # Verify frozen pure-Rust dope_v8_emit loads & samples d_model=160, enc=3, dec=3 natively!
    emit_bin = _find_v8_emit_bin()
    if emit_bin is not None:
        rng = np.random.default_rng(19)
        xs = rng.normal(size=(80, 4)).astype(np.float32)
        ys = (xs[:, 0] * xs[:, 1] + 0.3 * xs[:, 2] > 0.0).astype(np.float32)
        pkt_bytes = fit_dse8_rust(
            xs, ys, task_id=1, budget_bytes=2048, checkpoint=out_st, code_tier=1, seed=5, bin_path=emit_bin
        )
        assert 0 < len(pkt_bytes) <= 2048
        x_syn, y_syn = sample_dse8_rust(
            pkt_bytes, n_rows=32, seed=5, checkpoint=out_st, n_heads=4, bin_path=emit_bin
        )
        assert x_syn.shape == (32, 4)
        assert y_syn.shape == (32,)
        assert np.all(np.isfinite(x_syn)) and np.all(np.isfinite(y_syn))


def test_repair_v8_unsticks_bilin_and_reopens_gate_preserving_scale(tmp_path: Path) -> None:
    torch.manual_seed(3)
    m = SingleEncV8(d_model=64, n_heads=4, n_enc_layers=2, n_dec_layers=2, code_dim=DEC_CODE_DIM)
    with torch.no_grad():
        m.node_sel.weight.normal_(0.0, 0.05)
        m.node_leaf.weight.normal_(0.0, 0.03)
        m.gate_head.bias[0] = -1.663553  # sigmoid(-3.663553) == 0.025
        m.gate_head.weight[0].zero_()
    donor = tmp_path / "collapsed.safetensors"
    st_save({k: v.contiguous().float() for k, v in m.state_dict().items() if not k.startswith("aux_ev_head.")}, str(donor))

    repaired = tmp_path / "repaired.safetensors"
    info = surgery_repair_v8(donor, repaired, observed_alpha=0.025, target_alpha=0.25, gate_weight_damp=0.10, seed=9)
    assert info["bilin_reinitialized"] is True
    assert info["norms"]["bilin_u.weight"] > 0.0
    assert info["norms"]["bilin_w.weight"] == 0.0

    m2 = SingleEncV8(d_model=64, n_heads=4, n_enc_layers=2, n_dec_layers=2, code_dim=DEC_CODE_DIM)
    m2.load_state_dict(st_load(str(repaired)), strict=False)
    m.eval()
    m2.eval()
    dec_col = torch.randn(1, 5, 48)
    dec_edge = torch.randn(1, 5, 5, 4)
    dec_glob = torch.randn(1, 32)
    code_vec = torch.zeros(1, 64)
    x_in = torch.randn(1, 12, 5)
    base_W = torch.randn(1, 8, 5) * 0.2
    base_b = torch.zeros(1, 8)
    with torch.no_grad():
        p1 = m.decode_plan(dec_col, dec_edge, dec_glob, code_vec)
        p2 = m2.decode_plan(dec_col, dec_edge, dec_glob, code_vec)
        y1 = m.predict_target_on_x(p1, base_W, base_b, x_in)
        y2 = m2.predict_target_on_x(p2, base_W, base_b, x_in)
    assert abs(float(p1["alpha_table"][0].item()) - 0.025) < 1e-3
    assert abs(float(p2["alpha_table"][0].item()) - 0.250) < 1e-3
    assert float((y1 - y2).abs().max().item()) < 5e-3

    # Train 3 steps with v8-r3 flags -> bilin_w wakes up and alpha_gate stays open!
    out_dir = tmp_path / "train_smoke"
    args = parse_args([
        "--out-dir", str(out_dir), "--steps", "3", "--episodes-per-step", "2",
        "--d-model", "64", "--n-heads", "4", "--n-enc-layers", "2", "--n-dec-layers", "2",
        "--n-fit", "64", "--n-qry", "48", "--n-syn", "64",
        "--init-from", str(repaired), "--anchor-weight", "0.02",
        "--alpha-floor", "0.25", "--alpha-floor-steps", "10", "--resid-cond-weight", "0.35",
        "--backbone-lr-mult", "0.25", "--backbone-warmup-steps", "5", "--save-every", "3",
    ])
    res = train_v8(args)
    assert res["steps_done"] == 3
    assert res["last_metrics"]["alpha_gate"] >= 0.20
    st_after = st_load(res["safetensors"])
    assert float(st_after["bilin_w.weight"].norm().item()) > 0.0
