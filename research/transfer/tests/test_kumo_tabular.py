"""Kumo native-interface fixtures; official checkpoint parity is unmeasured."""
from dataclasses import replace
import json
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research.transfer.adapters.base import AdapterError
from research.transfer.adapters.auditors.kumo_tabular import (
    CODE_REVISION, KumoRecipe, KumoTabularAdapter, QUANTILES,
)
from research.transfer.tests.test_other_auditors import PortFixture, ROOT, batch, query


class KumoNativeFixture:
    def __init__(self, columns=QUANTILES, values=None):
        self.columns = columns
        self.values = values
        self.calls = []

    def fit(self, *, x, y, **kwargs):
        self.calls.append(("fit", np.asarray(x).copy(), np.asarray(y).copy(), kwargs))

    def predict(self, *, x):
        self.calls.append(("predict", np.asarray(x).copy()))
        values = self.values if self.values is not None else [[float(i) ** 2 for i in range(1, 1000)]] * len(x)
        return self.columns, values


class KumoTests(unittest.TestCase):
    def port(self, recipe, native=None):
        native = native or KumoNativeFixture()
        fixture = PortFixture(self, recipe, "kumo_tabular", CODE_REVISION, factory=lambda **_: native)
        return fixture.adapter(KumoTabularAdapter, recipe), fixture, native

    def test_raw_targets_explicit_ensemble_context_seed_and_feature_only_queries(self):
        recipe = KumoRecipe("regression", estimators=4, estimator_batch_size=2)
        port, fixture, native = self.port(recipe)
        port.prepare_context(batch(), 17)
        result = port.predict_original_units(query(), 23)
        np.testing.assert_array_equal(native.calls[0][2], [[10.], [30.], [50.]])
        self.assertEqual(native.calls[0][3], dict(num_estimators=4, estimator_batch_size=2, generator=17))
        self.assertEqual(fixture.kwargs[0], dict(task="regression", size="large", device="cpu"))
        np.testing.assert_array_equal(native.calls[1][1], [[3.], [4.]])
        self.assertEqual(result.consumed_rows, 3)
        self.assertEqual(result.embedding_unavailable, "unsupported")
        expected = math.fsum(float(i) ** 2 for i in range(1, 1000)) / 999
        self.assertEqual(result.values, ((expected,), (expected,)))

    def test_nonlinear_inverse_is_applied_before_point_summary(self):
        # Independent inverse counterexample: z -> z^2. The native SDK has
        # already inverse-transformed each quantile before this port sees it.
        z = tuple(float(i) for i in range(1, 1000))
        raw = tuple(v ** 2 for v in z)
        port, _, _ = self.port(KumoRecipe("regression"), KumoNativeFixture(values=[raw, raw]))
        port.prepare(batch(), 1)
        value = port.predict(query(), 2).values[0][0]
        self.assertEqual(value, math.fsum(raw) / len(raw))
        self.assertNotEqual(value, (math.fsum(z) / len(z)) ** 2)

    def test_native_regression_target_matches_float32_head_and_refuses_overflow(self):
        import torch
        recipe = KumoRecipe("regression")
        fixture = PortFixture(self, recipe, "kumo_tabular", CODE_REVISION)
        port = fixture.adapter(KumoTabularAdapter, recipe, runtime=replace(fixture.runtime, factory=None))
        port._context_seed = 7
        class NativeHead:
            def fit(self, *, x, y, **kwargs):
                self.x, self.y = x, y
                # Independent native operation that rejects double input with
                # the official head's float32 parameter dtype.
                torch.nn.Linear(1, 1)(y)
        head = NativeHead()
        with patch.dict(sys.modules, {"sdm": SimpleNamespace(Stype=object(), TableTensor=object())}):
            port._fit_native(head, np.asarray([[0.], [1.], [2.]]), np.asarray([10., 30., 50.]))
            self.assertEqual(head.x.dtype, torch.float64)
            self.assertEqual(head.y.dtype, torch.float32)
            self.assertEqual(head.y.flatten().tolist(), [10., 30., 50.])
            with self.assertRaisesRegex(AdapterError, "native_target_precision_exceeded"):
                port._fit_native(head, np.asarray([[0.]]), np.asarray([1e100]))

    def test_preserves_named_quantile_grid_and_median_without_flattening(self):
        native = KumoNativeFixture()
        port, _, _ = self.port(KumoRecipe("regression", output="quantiles"), native)
        port.prepare(batch(), 1)
        result = port.predict(query(), 2)
        self.assertEqual(result.kind, "distribution")
        self.assertEqual(len(result.values[0]), 999)
        self.assertEqual(result.values[0][499], 500. ** 2)
        median, _, _ = self.port(KumoRecipe("regression", output="median"))
        median.prepare(batch(), 1)
        self.assertEqual(median.predict(query(), 2).values, ((250000.,), (250000.,)))
        for columns, values in ((tuple(reversed(QUANTILES)), None),
                                (QUANTILES, np.zeros((4, 2, 999))),
                                (QUANTILES, [[1., 0.] + [2.] * 997] * 2)):
            bad, _, _ = self.port(KumoRecipe("regression"), KumoNativeFixture(columns, values))
            bad.prepare(batch(), 1)
            with self.assertRaises(AdapterError):
                bad.predict(query(), 2)

    def test_classification_aligns_named_native_classes_and_refuses_missing_context(self):
        recipe = KumoRecipe("classification", output="probability", class_order=("b", "a"))
        native = KumoNativeFixture(("1", "0"), [[.2, .8], [.9, .1]])
        port, _, _ = self.port(recipe, native)
        port.prepare(batch(targets=("b", "a", "b")), 7)
        result = port.predict(query(), 8)
        self.assertEqual(result.class_order, ("b", "a"))
        self.assertEqual(result.values, ((.8, .2), (.1, .9)))
        np.testing.assert_array_equal(native.calls[0][2], [[0], [1], [0]])
        with self.assertRaises(AdapterError):
            port.prepare(batch(targets=("b", "b", "b")), 7)

    def test_refuses_hidden_native_feature_selection_and_recipe_mismatch(self):
        recipe = KumoRecipe("regression")
        port, fixture, _ = self.port(recipe)
        training = batch()
        wide = replace(training.features, rows=((1.,) * 501,) * 3)
        with self.assertRaises(AdapterError):
            port.prepare(replace(training, features=wide), 7)
        for changed in (replace(recipe, size="small"), replace(recipe, estimators=8),
                        replace(recipe, output="median")):
            with self.assertRaisesRegex(AdapterError, "recipe_pin_mismatch|transform_pin_mismatch"):
                fixture.adapter(KumoTabularAdapter, changed)

    def test_real_synthetic_contexts_share_declared_recipe_resources_and_usage(self):
        recipe = KumoRecipe("regression")
        real, fixture, native = self.port(recipe)
        real.prepare(batch(), 7)
        real_result = real.predict(query(), 8)
        synthetic = fixture.adapter(KumoTabularAdapter, recipe)
        synthetic.prepare(replace(batch(), role="synthetic_train"), 7)
        synth_result = synthetic.predict(query(), 8)
        self.assertEqual(real.identity, synthetic.identity)
        self.assertEqual(real_result, synth_result)
        self.assertEqual(native.calls[0][3], native.calls[2][3])

    def test_lock_is_unqualified_and_historical_lane_is_separate(self):
        lock = json.loads((ROOT / "research/transfer/locks/kumo-tabular.json").read_text())
        self.assertEqual(lock["status"], "unqualified")
        self.assertFalse(lock["checkpoint"]["local_verification"])
        self.assertIn("large/16-estimator", lock["historical_lane"])
        self.assertIn("official_checkpoint_minimal_example_parity", lock["unmeasured_gates"])
        self.assertEqual(lock["source"]["revision"], CODE_REVISION)


if __name__ == "__main__":
    unittest.main()
