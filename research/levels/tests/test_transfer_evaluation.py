import dataclasses
import json
import unittest

import numpy as np

from research.levels.transfer_evaluation import (
    EVALUATION_PROTOCOL, EvaluationInputError, FrozenRealFitReceiver,
    PreparedEvaluationView, evaluate_prepared_transfer,
)
from research.levels.scaling import FitOnlyTableTransform


class TransferEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(20, dtype=float).reshape(-1, 1)
        self.y = np.arange(20, dtype=float)
        self.receiver = FrozenRealFitReceiver.fit(self.x, self.y, task="regression", fit_scope="a" * 64)

    def test_real_fit_map_is_shared_and_synthetic_refit_reproduces_bug(self):
        test = np.array([[30.], [40.]])
        synth = self.x + 100
        a = FitOnlyTableTransform.fit(self.x, self.y, task="regression")
        b = FitOnlyTableTransform.fit(synth, self.y, task="regression")
        self.assertFalse(np.array_equal(a.transform_features(test), b.transform_features(test)))
        real = self.receiver.prepare(self.x, self.y, role="real_fit")
        query = self.receiver.prepare(test, [30, 40], role="real_test")
        seen = []
        def oracle(x, y, xt, yt):
            seen.append((xt.copy(), yt.copy()))
            return object()
        def score(ref, x, y, xt, yt):
            np.testing.assert_array_equal(xt, query.x)
            np.testing.assert_array_equal(yt, query.y)
            return 1
        for s in [synth, synth * 100]:
            arm = self.receiver.prepare(s, self.y, role="synthetic_fit")
            out = evaluate_prepared_transfer(real, arm, query, oracle=oracle, scorer=score)
            self.assertEqual(out["protocol"], EVALUATION_PROTOCOL)
            self.assertEqual(out["real_test_sha256"], query.content_sha256)
        np.testing.assert_array_equal(seen[0][0], seen[1][0])

    def test_test_and_synthetic_poisoning_never_changes_fitted_metadata(self):
        before = self.receiver.metadata()
        self.receiver.prepare([[1e9]], [1e12], role="real_test")
        self.receiver.prepare([[1e6]], [-1e12], role="synthetic_fit")
        self.assertEqual(before, self.receiver.metadata())
        query = self.receiver.prepare([[40.]], [190.], role="real_test")
        self.assertEqual(query.y[0], 10.0)  # no tanh/clip on real query targets

    def test_mixed_categories_unknowns_and_missing_use_frozen_fit(self):
        x = [[float(i), "red" if i % 2 else "blue"] for i in range(20)]
        r = FrozenRealFitReceiver.fit(x, self.y, task="regression", fit_scope="b" * 64)
        before = r.metadata()
        view = r.prepare([[None, "NEW_PRIVATE_CATEGORY"], [1., None]], [20., 21.], role="real_test")
        self.assertEqual(view.x.shape, (2, 2))
        self.assertEqual(before, r.metadata())
        self.assertNotIn("PRIVATE", json.dumps(r.metadata()))

    def test_already_prepared_input_has_explicit_space_and_no_second_apply(self):
        v = self.receiver.prepare(self.x, self.y, role="real_fit")
        with self.assertRaisesRegex(EvaluationInputError, "^evaluation_double_transform$"):
            self.receiver.prepare(v, self.y, role="synthetic_fit")
        with self.assertRaisesRegex(EvaluationInputError, "^evaluation_double_transform$"):
            self.receiver.prepare(self.x, self.y, role="synthetic_fit", input_space="prepared")

    def test_mismatch_mutation_roles_and_public_failures(self):
        a = self.receiver.prepare(self.x, self.y, role="real_fit")
        s = self.receiver.prepare(self.x, self.y, role="synthetic_fit")
        t = self.receiver.prepare(self.x, self.y, role="real_test")
        for changed in [dataclasses.replace(s, space_id="b" * 64), dataclasses.replace(s, role="real_test")]:
            with self.assertRaises(EvaluationInputError):
                evaluate_prepared_transfer(a, changed, t, oracle=lambda *x: None, scorer=lambda *x: None)
        s.x.setflags(write=True)
        s.x[0, 0] += 1
        with self.assertRaisesRegex(EvaluationInputError, "^evaluation_view_mutated$"):
            evaluate_prepared_transfer(a, s, t, oracle=lambda *x: None, scorer=lambda *x: None)
        for x, y in [([], []), ([[float("inf")]], [1.]), ([[1.]], [float("nan")])]:
            with self.assertRaises(EvaluationInputError) as caught:
                PreparedEvaluationView.create(x, y, space_id="a" * 64, fit_scope="b" * 64,
                    feature_groups=(0,), task="regression", role="real_test")
            self.assertNotIn("PRIVATE", str(caught.exception))

    def test_frozen_class_order_rejects_unknown_query_classes(self):
        y = [0., 1., 2., 0., 1.] * 4
        r = FrozenRealFitReceiver.fit(self.x, y, task="multiclass", fit_scope="c" * 64)
        np.testing.assert_array_equal(r.prepare([[2.]], [2.], role="real_test").y, [2.])
        with self.assertRaises(EvaluationInputError):
            r.prepare([[2.]], [999.], role="real_test")


class CanonicalReceiverBridgeTests(unittest.TestCase):
    def test_prospective_canonical_receiver_has_no_synthetic_fit_and_replay_stays_explicit(self):
        # Isolate the real orchestration function from optional third-party
        # learners. Its callbacks use fixed arrays and train no models.
        import ast
        from pathlib import Path
        from typing import Dict, Any, List
        import time
        from unittest.mock import patch
        from research.levels.transfer_evaluation import LEGACY_REPLAY_PROTOCOL
        source = (Path(__file__).resolve().parents[1] / "v3_canonical_runner.py").read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "evaluate_dataset_across_tiers")
        scope = {"Dict": Dict, "Any": Any, "List": List, "MultiLevelLandmarkModel": object,
                 "np": np, "time": time, "hashlib": __import__("hashlib"), "json": json,
                 "FitOnlyTableTransform": FitOnlyTableTransform,
                 "FrozenRealFitReceiver": FrozenRealFitReceiver,
                 "EvaluationInputError": EvaluationInputError,
                 "EVALUATION_PROTOCOL": EVALUATION_PROTOCOL, "LEGACY_REPLAY_PROTOCOL": LEGACY_REPLAY_PROTOCOL,
                 "LEVEL_NAMES": ["small"], "TIER_BYTE_CAPS": {"small": 2048},
                 "CANONICAL_6_AUDITORS": ("fixture",), "CORE_COMPARATOR_AUDITORS": ("fixture",)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "canonical_receiver_unit", "exec"), scope)
        x = np.arange(20., dtype=float).reshape(-1, 1)
        y = np.arange(20., dtype=float)
        class Identity:
            target_scaler = type("Target", (), {"transform": staticmethod(np.asarray)})()
            inverse_transform = staticmethod(lambda a, b: (a, b))
        prep = {"X_tr_s": x, "y_tr_s": y, "X_te_s": x + 30, "y_te_s": y + 30,
                "X_tr_num": x, "y_tr_raw": y, "X_te_num": x + 30, "y_te_raw": y + 30,
                "y_tr_original": y, "y_te_original": y + 30, "scaler": Identity(),
                "p_use": 1, "k_classes": 2, "report": {}, "diagnostic_fp64_scaler_sidecar_bytes": 1}
        class Model:
            encode_table = staticmethod(lambda *a, **k: {})
            generate_synthetic = staticmethod(lambda *a, **k: (x + 100, y + 100))
        scope.update(_prepare_dataset_split=lambda _: prep,
                     pack_compact_scaler_sidecar=lambda *a, **k: b"s",
                     serialize_dope_v3_envelope=lambda *a, **k: b"w" * 20,
                     deserialize_dope_v3_envelope=lambda *a: ({}, Identity()),
                     pack_rows_complete_wire=lambda *a, **k: b"r",
                     unpack_rows_complete_wire=lambda *a, **k: (x, y, None),
                     pack_copula_complete_wire=lambda *a, **k: b"c",
                     unpack_copula_complete_wire=lambda *a, **k: (x, y, None))
        spec = {"dataset_id": "fixture", "role": "dev", "task": "regression",
                "prior_v2_exposure": None, "train_sha256": "a" * 64}
        for protocol in [EVALUATION_PROTOCOL, LEGACY_REPLAY_PROTOCOL]:
            oracles, scores = [], []
            def oracle(a, b, xt, yt, **kwargs):
                oracles.append((xt.copy(), yt.copy()))
                return len(oracles)
            def score(ref, a, b, xt, yt, **kwargs):
                scores.append((ref, xt.copy(), yt.copy()))
                return {"complete_case_mean_clipped_R": 1.}
            scope.update(evaluate_canonical_oracle=oracle, score_synthetic_against_oracle=score)
            with patch.object(FitOnlyTableTransform, "fit", wraps=FitOnlyTableTransform.fit) as fits:
                rows = scope["evaluate_dataset_across_tiers"]({**spec, "evaluation_protocol": protocol}, Model(), Model())
                synthetic_fits = sum(np.asarray(call.args[0]).mean() > 90 for call in fits.call_args_list)
            if protocol == EVALUATION_PROTOCOL:
                self.assertEqual(synthetic_fits, 0)
                np.testing.assert_array_equal(scores[1][1], oracles[1][0])
                np.testing.assert_array_equal(scores[2][1], oracles[1][0])
                self.assertEqual(rows[0]["evaluation_protocol"], EVALUATION_PROTOCOL)
                self.assertEqual(rows[0]["receiver_feature_domain"], "legacy_inverse_proxy_not_original_raw")
                self.assertGreater(oracles[1][1].max(), 1.)
            else:
                self.assertEqual(synthetic_fits, 2)
                self.assertFalse(np.array_equal(scores[1][1], oracles[1][0]))
                self.assertNotIn("evaluation_protocol", rows[0])


if __name__ == "__main__":
    unittest.main()
