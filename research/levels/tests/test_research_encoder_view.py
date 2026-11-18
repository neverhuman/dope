"""Research-view contract. Does not download or require the released checkpoint."""
import inspect
import math
import unittest
from pathlib import Path

import torch

from research.levels.encoder import CANONICAL_GATE_KEYS_14, DopeEncoder629
from research.levels.research_encoder_view import (
    LEVEL_CODE,
    PACKET_GLOBAL_FIELDS,
    PACKET_GLOBAL_SCHEMA,
    PACKET_GLOBAL_WIDTH,
    RESEARCH_ENCODER_VIEW_RECIPE,
    encode_research_view,
    packet_global_v0,
)


def _encoder():
    torch.manual_seed(20261118)
    return DopeEncoder629().eval()


def _inputs(batch=2):
    torch.manual_seed(7)
    return torch.randn(batch, 256), torch.randn(batch, 64, 17)


def _encoding(**overrides):
    gates = {key: float(index + 1) for index, key in enumerate(CANONICAL_GATE_KEYS_14)}
    encoding = {
        "level": "high",
        "n_train": 8,
        "p_use": 4,
        "gates": gates,
        "target_bundle": {
            "residual_std": 1.5,
            "noise_scale": 0.25,
            "class_prior": 0.5,
            "y_min": -2.0,
            "y_max": 3.0,
            "trees": [{"leaf": 0.0}, {"leaf": 1.0}],
            "pairs": [(0, 1)],
        },
        "col_knots": [[0.1, 0.2], [0.3, 0.4]],
        "col_means": [9.0, 8.0],
    }
    encoding.update(overrides)
    return encoding


class ResearchEncoderViewTests(unittest.TestCase):
    def test_recipe_is_not_a_product_identity(self):
        self.assertEqual(RESEARCH_ENCODER_VIEW_RECIPE, "research-encoder-view/v0")
        self.assertEqual(PACKET_GLOBAL_SCHEMA, "dope.packet-global/v0")
        source = Path(__import__("research.levels.research_encoder_view", fromlist=["x"]).__file__).read_text()
        self.assertNotIn("dope629-ultra/v1", source)
        forward = inspect.signature(DopeEncoder629.forward)
        self.assertEqual(list(forward.parameters), ["self", "dataset_features", "col_landmarks"])

    def test_high_calls_forward_and_matches_it(self):
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
        self.assertEqual(tuple(view.shape), (2, 629))
        self.assertTrue(torch.equal(view[:, :256], features))

    def test_ultra_is_separate_finite_and_not_high(self):
        encoder = _encoder()
        features, columns = _inputs()
        called = {}
        real = encoder.forward

        def wrapped(dataset_features, col_landmarks):
            called["forward"] = True
            return real(dataset_features, col_landmarks)

        encoder.forward = wrapped
        router = encoder.router.forward
        levels = []

        def routed(desc_vec, level):
            levels.append(level)
            return router(desc_vec, level)

        encoder.router.forward = routed
        ultra = encode_research_view(encoder, features, columns, level="ultra")
        self.assertNotIn("forward", called)
        self.assertEqual(levels, ["ultra"])
        high = real(features, columns)
        self.assertEqual(tuple(ultra.shape), (2, 629))
        self.assertTrue(torch.isfinite(ultra).all())
        self.assertTrue(torch.equal(ultra[:, :256], features))
        self.assertFalse(torch.equal(ultra[:, 512:526], high[:, 512:526]))

    def test_bad_level_and_nan_inputs(self):
        encoder = _encoder()
        features, columns = _inputs()
        for level in ("low", "medium", "small", "", None):
            with self.assertRaises(ValueError):
                encode_research_view(encoder, features, columns, level=level)
        features = features.clone()
        features[0, 0] = float("nan")
        columns = columns.clone()
        columns[0, 0, 0] = float("inf")
        high = encode_research_view(encoder, features, columns, level="high")
        ultra = encode_research_view(encoder, features, columns, level="ultra")
        self.assertTrue(torch.isfinite(high).all())
        self.assertTrue(torch.isfinite(ultra).all())

    def test_packet_fixture_ignores_columns_and_refuses_queries(self):
        expected = tuple(float(index + 1) for index in range(14)) + (
            1.5, 0.25, 0.5, -2.0, 3.0, math.log(8), 4.0, 2.0, 1.0, LEVEL_CODE["high"],
        )
        self.assertEqual(len(expected), PACKET_GLOBAL_WIDTH)
        self.assertEqual(len(PACKET_GLOBAL_FIELDS), PACKET_GLOBAL_WIDTH)
        left = packet_global_v0(_encoding())
        right = packet_global_v0(_encoding(col_knots=[[4.0, 5.0]], col_means=[0.0, 1.0]))
        self.assertEqual(left, expected)
        self.assertEqual(left, right)
        ultra = packet_global_v0(_encoding(level="ultra"))
        self.assertEqual(ultra[-1], LEVEL_CODE["ultra"])
        self.assertEqual(ultra[:-1], expected[:-1])

    def test_packet_refusals_do_not_echo_values(self):
        marker = "7.25"
        missing = _encoding()
        del missing["target_bundle"]["residual_std"]
        with self.assertRaises(ValueError) as missing_gate:
            packet_global_v0(missing)
        self.assertNotIn(marker, str(missing_gate.exception))
        self.assertIn("target_bundle.residual_std", str(missing_gate.exception))
        poisoned = _encoding()
        poisoned["target_bundle"]["y_max"] = float(marker)
        poisoned["target_bundle"]["residual_std"] = float("nan")
        with self.assertRaises(ValueError) as nonfinite:
            packet_global_v0(poisoned)
        self.assertNotIn(marker, str(nonfinite.exception))
        for key in ("X_query", "y_query", "query_rows", "query_features"):
            with self.assertRaises(ValueError) as query:
                packet_global_v0(_encoding(**{key: marker}))
            self.assertIn(key, str(query.exception))
            self.assertNotIn(marker, str(query.exception))
        with self.assertRaises(ValueError):
            packet_global_v0(_encoding(level="low"))
        with self.assertRaises(ValueError):
            packet_global_v0(_encoding(n_train=0))
        with self.assertRaises(ValueError):
            packet_global_v0(_encoding(p_use=0))


if __name__ == "__main__":
    unittest.main()
