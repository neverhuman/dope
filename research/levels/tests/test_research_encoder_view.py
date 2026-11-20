"""Research-view contract (`WP1` Ultra + `WP2` 196D Packet). Offline; no weight download."""
import inspect
import unittest
from pathlib import Path

import numpy as np
import torch

from research.levels.encoder import CANONICAL_GATE_KEYS_14, DopeEncoder629
from research.levels.multilevel_model import MultiLevelLandmarkModel
from research.levels.research_encoder_view import (
    FORBIDDEN_QUERY_KEYS,
    LEVEL_CODE,
    PACKET_FEATURES_SCHEMA,
    PACKET_FEATURES_WIDTH,
    PACKET_GLOBAL_FIELDS,
    PACKET_GLOBAL_SCHEMA,
    PACKET_GLOBAL_WIDTH,
    RESEARCH_ENCODER_VIEW_RECIPE,
    encode_research_view,
    packet_features_v1,
    packet_global_v0,
    packet_global_v1,
)


def _encoder():
    torch.manual_seed(20261118)
    return DopeEncoder629().eval()


def _inputs(batch=2):
    torch.manual_seed(7)
    return torch.randn(batch, 256), torch.randn(batch, 64, 17)


def _real_encoding(task="regression", level="high", seed=11):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((64, 5)).astype(np.float64)
    X[:, 3] = (X[:, 3] > 0.0).astype(np.float64)
    X[:, 4] = rng.integers(0, 4, size=64).astype(np.float64)
    if task == "regression":
        y = (0.7 * X[:, 0] - 0.4 * (X[:, 1] ** 2) + 0.2 * rng.standard_normal(64)).astype(np.float64)
    elif task == "classification":
        y = (X[:, 0] - 0.5 * X[:, 1] > 0.0).astype(np.float64)
    else:
        y = rng.integers(0, 3, size=64).astype(np.float64)
    torch.manual_seed(42)
    model = MultiLevelLandmarkModel().eval()
    return model.encode_table(X, y, task=task, level=level, landmark_mode="none")


class ResearchEncoderViewTests(unittest.TestCase):
    def test_schema_deduplicates_noise_scale_and_excludes_extrema(self):
        self.assertEqual(RESEARCH_ENCODER_VIEW_RECIPE, "research-encoder-view/v1")
        self.assertEqual(PACKET_GLOBAL_SCHEMA, "dope.packet-global/v1")
        self.assertEqual(PACKET_FEATURES_SCHEMA, "dope.packet-features/v1")
        self.assertEqual(len(PACKET_GLOBAL_FIELDS), PACKET_GLOBAL_WIDTH)
        self.assertEqual(len(set(PACKET_GLOBAL_FIELDS)), PACKET_GLOBAL_WIDTH)
        self.assertEqual(PACKET_GLOBAL_FIELDS.count("noise_scale"), 1)
        self.assertNotIn("y_min", PACKET_GLOBAL_FIELDS)
        self.assertNotIn("y_max", PACKET_GLOBAL_FIELDS)
        source = Path(__import__("research.levels.research_encoder_view", fromlist=["x"]).__file__).read_text()
        self.assertNotIn("dope629-ultra/v1", source)
        forward = inspect.signature(DopeEncoder629.forward)
        self.assertEqual(list(forward.parameters), ["self", "dataset_features", "col_landmarks"])

    def test_high_delegates_to_forward_and_refuses_ultra_conditioning(self):
        encoder = _encoder()
        features, columns = _inputs()
        seen = {}
        real = encoder.forward

        def wrapped(dataset_features, col_landmarks):
            seen["called"] = True
            return real(dataset_features, col_landmarks)

        encoder.forward = wrapped
        view = encode_research_view(encoder, features, columns, level="high")
        direct = real(features, columns)
        self.assertTrue(seen.get("called"))
        self.assertTrue(torch.equal(view, direct))
        with self.assertRaisesRegex(ValueError, "conditioning_level"):
            encode_research_view(encoder, features, columns, level="high", conditioning_level="ultra")

    def test_ultra_requires_explicit_ultra_conditioning_eval_and_finite_inputs(self):
        encoder = _encoder()
        features, columns = _inputs()
        for bad_cond in (None, "high", "low"):
            with self.assertRaisesRegex(ValueError, "conditioning_level"):
                encode_research_view(encoder, features, columns, level="ultra", conditioning_level=bad_cond)

        ultra = encode_research_view(
            encoder, features, columns, level="ultra", conditioning_level="ultra",
        )
        high = encoder(features, columns)
        self.assertFalse(ultra.requires_grad)
        self.assertEqual(tuple(ultra.shape), (2, 629))
        self.assertTrue(torch.isfinite(ultra).all())
        self.assertTrue(torch.equal(ultra[:, :256], features))
        self.assertFalse(torch.equal(ultra[:, 512:526], high[:, 512:526]))

        # Non-finite input must fail cleanly on ultra rather than collapsing 373 router coords to 0.
        bad_feat = features.clone()
        bad_feat[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            encode_research_view(encoder, bad_feat, columns, level="ultra", conditioning_level="ultra")

        train_enc = DopeEncoder629().train()
        with self.assertRaisesRegex(ValueError, "eval"):
            encode_research_view(train_enc, features, columns, level="ultra", conditioning_level="ultra")

    def test_real_encode_table_across_regression_binary_and_multiclass(self):
        for task in ("regression", "classification", "multiclass"):
            for level in ("high", "ultra"):
                enc = _real_encoding(task=task, level=level)
                feat = packet_features_v1(enc)
                glob = packet_global_v1(enc)
                self.assertEqual(len(feat), PACKET_FEATURES_WIDTH)
                self.assertEqual(len(glob), PACKET_GLOBAL_WIDTH)
                self.assertEqual(glob, feat[:PACKET_GLOBAL_WIDTH])
                self.assertEqual(packet_global_v0(enc), glob)
                self.assertTrue(all(np.isfinite(v) for v in feat))
                self.assertEqual(glob[-1], LEVEL_CODE[level])

                # Changing y_min / y_max extrema by 6 orders of magnitude leaves packet unchanged.
                enc_ext = dict(enc)
                enc_ext["target_bundle"] = dict(enc["target_bundle"], y_min=-765432.0, y_max=987654.0)
                self.assertEqual(packet_features_v1(enc_ext), feat)

                # P-S2 covariate-only ablation zeros target-bundle slots while keeping covariates.
                no_tgt = packet_features_v1(enc, include_target_bundle=False)
                self.assertEqual(len(no_tgt), PACKET_FEATURES_WIDTH)
                self.assertEqual(no_tgt[14:17], (0.0, 0.0, 0.0))
                self.assertEqual(no_tgt[40:44], (0.0, 0.0, 0.0, 0.0))
                self.assertEqual(no_tgt[172:188], (0.0,) * 16)
                self.assertEqual(no_tgt[188:196], feat[188:196])

    def test_multiclass_label_and_column_permutation_invariance(self):
        enc = _real_encoding(task="multiclass", level="high")
        base = packet_global_v1(enc)
        perm_enc = dict(enc)
        priors = list(enc["target_bundle"]["class_priors"])
        perm_enc["target_bundle"] = dict(enc["target_bundle"], class_priors=priors[::-1])
        perm_glob = packet_global_v1(perm_enc)
        for a, b in zip(base, perm_glob):
            self.assertAlmostEqual(a, b, places=6)

        reg_enc = _real_encoding(task="regression", level="high")
        p = int(reg_enc["p_use"])
        order = [3, 0, 4, 1, 2]
        inv = [order.index(j) for j in range(p)]
        col_perm = dict(
            reg_enc,
            col_knots=[reg_enc["col_knots"][j] for j in order],
            chol_factor=np.asarray(reg_enc["chol_factor"])[order],
            diag_res_std=[reg_enc["diag_res_std"][j] for j in order],
            is_binary_col=[reg_enc["is_binary_col"][j] for j in order],
            binary_probs=[reg_enc["binary_probs"][j] for j in order],
        )
        if isinstance(reg_enc.get("discrete_support"), dict):
            col_perm["discrete_support"] = {
                inv[int(k)]: v for k, v in reg_enc["discrete_support"].items() if int(k) < p
            }
        for key in ("regime_means", "regime_scales"):
            arr = np.asarray(reg_enc.get(key, np.zeros((0, p))))
            if arr.ndim == 2 and arr.shape[1] >= p:
                col_perm[key] = arr[:, order]
        tb = dict(reg_enc["target_bundle"])
        tb["pairs"] = [(inv[int(a)], inv[int(c)]) for a, c in tb.get("pairs", ())]
        tb["trees"] = [
            dict(t, features=[inv[int(f)] if 0 <= int(f) < p else int(f) for f in t["features"]])
            for t in tb.get("trees", ())
        ]
        w = list(np.asarray(tb.get("w_basis", ())).reshape(-1))
        if len(w) >= 1 + 3 * p:
            tb["w_basis"] = (
                [w[0]]
                + [w[1 + j] for j in order]
                + [w[1 + p + j] for j in order]
                + [w[1 + 2 * p + j] for j in order]
                + w[1 + 3 * p :]
            )
        col_perm["target_bundle"] = tb
        for a, b in zip(packet_features_v1(reg_enc), packet_features_v1(col_perm)):
            self.assertAlmostEqual(a, b, places=6)

    def test_packet_refusals_block_query_keys_and_redact_values(self):
        enc = _real_encoding(task="regression", level="high")
        marker = "7.25"
        for key in FORBIDDEN_QUERY_KEYS:
            with self.assertRaises(ValueError) as query_err:
                packet_features_v1(dict(enc, **{key: marker}))
            self.assertIn(key, str(query_err.exception))
            self.assertNotIn(marker, str(query_err.exception))
        with self.assertRaisesRegex(ValueError, "fit role"):
            packet_features_v1(dict(enc, role="query"))
        poisoned = dict(enc)
        poisoned["target_bundle"] = dict(enc["target_bundle"], y_max=float(marker), residual_std=float("nan"))
        with self.assertRaises(ValueError) as nan_err:
            packet_features_v1(poisoned)
        self.assertNotIn(marker, str(nan_err.exception))
        bad_gates = dict(enc, gates={k: 0.1 for k in CANONICAL_GATE_KEYS_14[:-1]})
        with self.assertRaises(ValueError):
            packet_features_v1(bad_gates)


if __name__ == "__main__":
    unittest.main()
