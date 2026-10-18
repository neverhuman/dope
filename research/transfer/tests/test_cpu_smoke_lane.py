"""Actual native CPU parity on simulated data; no external model qualification."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from research.transfer.adapters.base import AdapterError, FeatureBatch, TrainingBatch
from research.transfer.adapters.auditors import conventional as C
from research.transfer.evidence import digest

ROOT = Path(__file__).resolve().parents[3]


def pin(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(task, estimator, role="development_train"):
    lock = C.cpu_smoke_lock()
    x = np.linspace(-2., 2., 96)
    q = np.linspace(-1.9, 1.9, 24)
    features = lambda values: tuple((float(v), float(v * v)) for v in values)
    labels = lambda values: tuple(float(100. + 7. * v - 3. * v * v) if task == "regression"
                                  else int(v > 0) if task == "binary"
                                  else int(v > -.65) + int(v > .65) for v in values)
    train = TrainingBatch(FeatureBatch(tuple(pin(f"fit-{i}") for i in range(len(x))),
                                      features(x), pin("prepared-numeric-schema")), labels(x), role)
    query = FeatureBatch(tuple(pin(f"query-{i}") for i in range(len(q))),
                         features(q), train.features.schema_sha256)
    return {"estimator": estimator, "task": task, "seed": 1729,
            "parameters": dict(lock["estimators"][estimator]["task_grids"][task][0]),
            "training": train, "query": query, "counts": dict.fromkeys(lock["count_fields"], len(x)),
            "lock_sha256": digest(lock)}, labels(q)


def reference(request):
    # Instantiate the official native interface independently of the adapter factory.
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    task, name, params = request["task"], request["estimator"], request["parameters"]
    regression = task == "regression"
    if name == "decision_tree":
        cls = DecisionTreeRegressor if regression else DecisionTreeClassifier
        model = cls(**params, random_state=1729)
    else:
        linear = (Ridge(**params, random_state=1729) if regression else
                  LogisticRegression(**params, random_state=1729, n_jobs=1))
        basis = [SplineTransformer(n_knots=3, degree=2, knots="uniform", extrapolation="linear",
                                   include_bias=False, order="C", sparse_output=False)] if name == "additive_spline_surrogate" else []
        model = make_pipeline(*basis, StandardScaler(), linear)
    with threadpool_limits(limits=1):
        model.fit(request["training"].features.rows, request["training"].targets)
        if regression:
            return np.asarray(model.predict(request["query"].rows)).reshape(-1, 1)
        values = model.predict_proba(request["query"].rows)
    values = np.clip(values, 1e-12, 1 - 1e-12)
    return values / values.sum(axis=1, keepdims=True)


class CpuSmokeLaneTests(unittest.TestCase):
    def refused(self, request):
        with self.assertRaises(AdapterError) as captured:
            C.cpu_smoke_predict(request)
        error = captured.exception
        self.assertEqual(str(error), "cpu_smoke_refused")
        diagnostic = "".join(traceback.format_exception(error))
        for secret in ("PRIVATE_HEADER", "PRIVATE_VALUE", "987654321", "/private/source"):
            self.assertNotIn(secret, diagnostic)

    def test_lock_pins_small_native_panel_and_separate_qualification_scope(self):
        lock = C.cpu_smoke_lock()
        self.assertEqual(set(lock["estimators"]), {"ridge", "logistic_l2", "decision_tree", "additive_spline_surrogate"})
        self.assertEqual(lock["qualification_scope"], "cpu_smoke_only")
        self.assertFalse(any(lock[k] for k in ("foundation_model_qualified", "gpu_panel_qualified", "tdl_panel_qualified")))
        self.assertEqual((lock["n_jobs"], lock["threadpool_limit"], lock["resources"]["gpu_bytes"]), (1, 1, 0))
        self.assertEqual(lock["seeds"], [1729])
        for task, classes in lock["class_order"].items():
            self.assertEqual(len(classes), {"binary": 2, "multiclass": 3, "regression": 0}[task])
            self.assertTrue(all(len(c) == 64 and int(c, 16) >= 0 for c in classes))

    def test_native_bitwise_reproducibility_parity_and_useful_null_controls(self):
        lock = C.cpu_smoke_lock()
        for estimator, panel in lock["estimators"].items():
            for task in panel["task_grids"]:
                with self.subTest(estimator=estimator, task=task):
                    request, labels = fixture(task, estimator)
                    with patch("socket.socket.connect", side_effect=AssertionError("network prohibited")):
                        first, second = C.cpu_smoke_predict(request), C.cpu_smoke_predict(request)
                    a, b = np.asarray(first["values"]), np.asarray(second["values"])
                    self.assertEqual(a.tobytes(), b.tobytes())
                    np.testing.assert_array_equal(a, reference(request))
                    self.assertEqual(first["request_sha256"], second["request_sha256"])
                    self.assertEqual(first["counts"], dict.fromkeys(lock["count_fields"], 96))
                    self.assertEqual(first["row_ids"], request["query"].row_ids)
                    self.assertEqual(len(first["values"]), 24)
                    self.assertFalse(first["qualified"])
                    self.assertEqual(first["qualification_scope"], "cpu_smoke_only")
                    null = dict(request, training=replace(request["training"], targets=tuple(
                        np.random.RandomState(80).permutation(request["training"].targets).tolist())))
                    control = np.asarray(C.cpu_smoke_predict(null)["values"])
                    if task == "regression":
                        useful = np.mean((a[:, 0] - labels) ** 2)
                        self.assertLess(useful, np.mean((control[:, 0] - labels) ** 2) * .4)
                        self.assertGreater(a[:, 0].min(), 50.)  # Original target units.
                    else:
                        truth = np.asarray(labels)
                        loss = -np.log(a[np.arange(len(truth)), truth]).mean()
                        null_loss = -np.log(control[np.arange(len(truth)), truth]).mean()
                        self.assertLess(loss, null_loss)
                        self.assertGreaterEqual(np.mean(a.argmax(axis=1) == truth), .8)

    def test_fresh_cpu_processes_repeat_prediction_bytes(self):
        program = """import hashlib, numpy as np
from research.transfer.tests.test_cpu_smoke_lane import fixture
from research.transfer.adapters.auditors.conventional import cpu_smoke_predict
for task,name in [('regression','ridge'),('binary','decision_tree'),('multiclass','additive_spline_surrogate')]:
    request,_ = fixture(task,name)
    print(hashlib.sha256(np.asarray(cpu_smoke_predict(request)['values']).tobytes()).hexdigest())
"""
        runs = [subprocess.run([sys.executable, "-B", "-c", program], cwd=ROOT,
                               capture_output=True, text=True, timeout=30, check=True).stdout for _ in range(2)]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(len(runs[0].splitlines()), 3)

    def test_query_label_leaks_and_every_count_mismatch_refuse_before_fit(self):
        request, labels = fixture("binary", "logistic_l2")
        for key in ("y_query", "query_labels", "PRIVATE_HEADER", "/private/source"):
            self.refused(dict(request, **{key: labels}))
        for field in C.cpu_smoke_lock()["count_fields"]:
            for bad in (95, 97, True, 96., -1):
                self.refused(dict(request, counts=dict(request["counts"], **{field: bad})))
        self.refused(dict(request, counts=dict(request["counts"], y_query=24)))
        labelled_query = TrainingBatch(request["query"], tuple(labels), "development_train")
        self.refused(dict(request, query=labelled_query))
        with patch.object(C, "_cpu_smoke_model") as backend:
            self.refused(dict(request, y_query=labels))
            backend.assert_not_called()

    def test_unpinned_names_recipes_runtime_or_seed_are_refused(self):
        request, _ = fixture("regression", "ridge")
        for name in ("hist_gbdt", "ebm", "PRIVATE_HEADER", "/private/source", None):
            self.refused(dict(request, estimator=name))
        for params in ({"alpha": True}, {"alpha": 2.}, {**request["parameters"], "n_jobs": 2}):
            self.refused(dict(request, parameters=params))
        for seed in (0, 1730, True):
            self.refused(dict(request, seed=seed))
        self.refused(dict(request, lock_sha256="0" * 64))
        with patch.object(C, "version", return_value="0.0.0"):
            self.refused(request)

    def test_nonfinite_invalid_probabilities_and_native_errors_never_escape(self):
        for task, name in (("regression", "ridge"), ("binary", "logistic_l2")):
            request, _ = fixture(task, name)
            model = unittest.mock.Mock()
            model.classes_ = np.arange(2)
            method = model.predict if task == "regression" else model.predict_proba
            shape = (24,) if task == "regression" else (24, 2)
            for value in (float("nan"), float("inf"), -float("inf")):
                method.return_value = np.full(shape, value)
                with patch.object(C, "_cpu_smoke_model", return_value=model):
                    self.refused(request)
            for cls in (ValueError, RuntimeError, SystemExit):
                model.fit.side_effect = cls("PRIVATE_HEADER PRIVATE_VALUE 987654321 /private/source")
                with patch.object(C, "_cpu_smoke_model", return_value=model):
                    self.refused(request)
        request, _ = fixture("binary", "decision_tree")
        for value in (-.1, .4, 1.1):
            model.fit.side_effect = None
            model.predict_proba.return_value = np.full((24, 2), value)
            with patch.object(C, "_cpu_smoke_model", return_value=model):
                self.refused(request)

    def test_fit_receives_all_context_query_only_and_clipping_is_deterministic(self):
        request, _ = fixture("binary", "decision_tree", "synthetic_train")
        native_factory, calls = C._cpu_smoke_model, []
        def recording(*args):
            model = native_factory(*args)
            fit, predict = model.fit, model.predict_proba
            def record_fit(x, y):
                from threadpoolctl import threadpool_info
                self.assertTrue(all(pool["num_threads"] == 1 for pool in threadpool_info()))
                self.assertEqual((x.dtype, y.dtype), (np.dtype("float64"), np.dtype("float64")))
                calls.append(("fit", x.copy(), y.copy()))
                return fit(x, y)
            def record_predict(x):
                calls.append(("query", x.copy()))
                return predict(x)
            model.fit, model.predict_proba = record_fit, record_predict
            return model
        with patch.object(C, "_cpu_smoke_model", side_effect=recording):
            result = C.cpu_smoke_predict(request)
        self.assertEqual([c[0] for c in calls], ["fit", "query"])
        np.testing.assert_array_equal(calls[0][1], request["training"].features.rows)
        np.testing.assert_array_equal(calls[0][2], request["training"].targets)
        np.testing.assert_array_equal(calls[1][1], request["query"].rows)
        values = np.asarray(result["values"])
        self.assertTrue(((values > 0) & (values < 1)).all())
        np.testing.assert_allclose(values.sum(axis=1), 1., rtol=0, atol=1e-15)

    def test_shape_schema_order_and_resource_caps_fail_closed(self):
        request, _ = fixture("regression", "ridge")
        self.refused(dict(request, query=replace(request["query"], schema_sha256=pin("different-schema"))))
        self.refused(dict(request, query=request["training"].features))
        self.refused(dict(request, query=replace(request["query"], rows=tuple((1.,) for _ in range(24)))))
        bad = replace(request["query"], rows=tuple(("PRIVATE_VALUE", 987654321.) for _ in range(24)))
        self.refused(dict(request, query=bad))
        many = FeatureBatch(tuple(pin(f"large-{i}") for i in range(257)), ((1., 2.),) * 257,
                            request["query"].schema_sha256)
        self.refused(dict(request, training=TrainingBatch(many, (1.,) * 257, "synthetic_train"),
                          counts=dict.fromkeys(request["counts"], 257)))
        self.refused(dict(request, query=many))
        wide = replace(request["training"].features, rows=((1.,) * 17,) * 96)
        self.refused(dict(request, training=replace(request["training"], features=wide)))
        with patch.object(C.resource, "getrusage", return_value=unittest.mock.Mock(ru_maxrss=3 * 1024**2)):
            self.refused(request)


if __name__ == "__main__":
    unittest.main()
