"""Regression tests for FP-0 loop repairs in `s3_v7_batch_ops` (F3, F4, FIX-07, FIX-08)."""
from __future__ import annotations

import numpy as np
import torch

from research.levels.s3_v7_batch_ops import (
    _fit_only_scale,
    apply_type_safe_fit_jitter,
    collate_v7_batch,
    compute_v7_batch_loss,
)
from research.levels.v7_single_encoding import (
    MAX_COLS,
    MAX_RANK,
    SingleEncV7,
    extract_dse_packet_numpy,
)


def test_f3_type_safe_jitter_preserves_discrete_and_lattice_columns() -> None:
    rng = np.random.default_rng(42)
    n = 400
    c_bin = rng.integers(0, 2, size=n).astype(np.float32)
    c_ord = rng.integers(0, 5, size=n).astype(np.float32)
    c_lat = (rng.integers(0, 30, size=n) * 0.25).astype(np.float32)
    c_lat48 = (0.37 + rng.integers(0, 48, size=n) * 0.1).astype(np.float32)
    c_lat100 = (-1.13 + rng.integers(0, 100, size=n) * 0.05).astype(np.float32)
    c_cont = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    X_raw = np.stack([c_bin, c_ord, c_lat, c_lat48, c_lat100, c_cont], axis=1)
    Xa, _ = _fit_only_scale(X_raw, X_raw)

    # Force jitter_prob=1.0 to assert discrete/lattice columns stay bit-identical while continuous changes.
    Xa_j, disc_mask = apply_type_safe_fit_jitter(np.random.default_rng(7), Xa, jitter_prob=1.0, jitter_sd=0.015)
    assert disc_mask.tolist() == [True, True, True, True, True, False]
    for ci in range(5):
        np.testing.assert_array_equal(Xa_j[:, ci], Xa[:, ci])
    assert not np.allclose(Xa_j[:, 5], Xa[:, 5])

    y = (c_bin * 0.8 + c_cont * 0.5 + rng.normal(0, 0.2, n)).astype(np.float32)
    pkt = extract_dse_packet_numpy(Xa_j, y, 0, 1)
    # Slot 38 is is_disc: binary and 5-level ordinal MUST remain is_disc == 1.0!
    assert pkt["col_feats"][0, 38] == 1.0
    assert pkt["col_feats"][1, 38] == 1.0
    assert pkt["col_feats"][5, 38] == 0.0


def test_f4_per_episode_realism_loss_has_zero_cross_episode_truncation() -> None:
    rng = np.random.default_rng(123)
    eps = []
    for seed_i, nq in ((1, 200), (2, 24)):
        r = np.random.default_rng(seed_i)
        Xa = r.normal(0, 1, size=(128, 6)).astype(np.float32)
        ya = (Xa[:, 0] * 1.5 - Xa[:, 1] * 0.7 + r.normal(0, 0.3, 128)).astype(np.float32)
        Xq = r.normal(0, 1, size=(nq, 6)).astype(np.float32)
        yq = (Xq[:, 0] * 1.5 - Xq[:, 1] * 0.7 + r.normal(0, 0.3, nq)).astype(np.float32)
        pkt = extract_dse_packet_numpy(Xa, ya, 0, 1)
        Xq_pad = np.zeros((256, MAX_COLS), dtype=np.float32)
        yq_pad = np.zeros((256,), dtype=np.float32)
        q_mask = np.zeros((256,), dtype=np.float32)
        Xq_pad[:nq, :6] = Xq
        yq_pad[:nq] = yq
        q_mask[:nq] = 1.0
        pkt.update(X_qry=Xq_pad, y_qry=yq_pad, q_mask=q_mask, n_fit=128, n_noise=0, disc_col_share=0.0)
        eps.append(pkt)

    # Compare episode 0's realism losses when paired with episode 1 (nq=24) vs episode 0 alone (nq=200).
    # If batch-min truncation were still present, pairing with nq=24 would alter episode 0's loss!
    model = SingleEncV7(d_model=64, n_heads=4, n_layers=1, h_target=16)
    b_solo = collate_v7_batch([eps[0]])
    b_pair = collate_v7_batch([eps[0], eps[1]])

    torch.manual_seed(99)
    zf0 = torch.randn((1, 64, MAX_RANK))
    zn0 = torch.randn((1, 64, MAX_COLS))
    yn0 = torch.randn((1, 64))
    ref0 = model.refine_packet(b_solo["col_feats"], b_solo["glob_feats"], b_solo["col_mask"])
    xs0, lg0, yr0 = model.sample_batch(b_solo, 64, zf0, zn0, yn0, ref=ref0)
    torch.manual_seed(555)
    _, m_solo = compute_v7_batch_loss(xs0, lg0, yr0, b_solo, ref=ref0)

    # Build a 2-episode batch where both episodes are identical to eps[0], EXCEPT ep1 has nq=24!
    ep0_short = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in eps[0].items()}
    ep0_short["q_mask"][24:] = 0.0
    b_mixed = collate_v7_batch([eps[0], ep0_short])
    zf_m = zf0.repeat(2, 1, 1)
    zn_m = zn0.repeat(2, 1, 1)
    yn_m = yn0.repeat(2, 1)
    ref_m = model.refine_packet(b_mixed["col_feats"], b_mixed["glob_feats"], b_mixed["col_mask"])
    xs_m, lg_m, yr_m = model.sample_batch(b_mixed, 64, zf_m, zn_m, yn_m, ref=ref_m)

    # Compute ep0_short solo so we can verify exact average = 0.5 * (solo_long + solo_short).
    b_short = collate_v7_batch([ep0_short])
    ref_s = model.refine_packet(b_short["col_feats"], b_short["glob_feats"], b_short["col_mask"])
    xs_s, lg_s, yr_s = model.sample_batch(b_short, 64, zf0, zn0, yn0, ref=ref_s)
    torch.manual_seed(555)
    _, m_short = compute_v7_batch_loss(xs_s, lg_s, yr_s, b_short, ref=ref_s)
    torch.manual_seed(555)
    _, m_mixed = compute_v7_batch_loss(xs_m, lg_m, yr_m, b_mixed, ref=ref_m)

    # Under per-episode calculation, mixed batch equals exact average of long + short (not short + short!).
    assert abs(m_solo["marg_loss"] - m_short["marg_loss"]) > 1e-4
    assert abs(m_mixed["marg_loss"] - 0.5 * (m_solo["marg_loss"] + m_short["marg_loss"])) < 1e-5
    assert abs(m_mixed["corr_loss"] - 0.5 * (m_solo["corr_loss"] + m_short["corr_loss"])) < 1e-5
    assert abs(m_mixed["swd_loss"] - 0.5 * (m_solo["swd_loss"] + m_short["swd_loss"])) < 1e-5
