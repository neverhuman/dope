"""Numerical research diagnostic controls; generated arrays only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnostics", HERE / "expanded_validation_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def pinned_author_tree(relative_path, expected_sha256):
    payload = (HERE / "fixtures" / "expanded-validation-metrics" / "author-source" / (relative_path + ".txt")).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("author_source_pin_mismatch")
    return ast.parse(payload)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        rng = np.random.default_rng(19)
        cls.fit, cls.val, cls.synth = (rng.normal(size=(120, 3)) for _ in range(3))

    def test_alpha_beta_matches_pinned_author_equations(self):
        from sklearn.neighbors import NearestNeighbors
        tree = pinned_author_tree("synthcity/src/synthcity/metrics/eval_statistical.py",
                                 "623ee83c6f0f1873c94cd1243e26b36e4941850915e5b5c2c943954cd8018e7d")
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "AlphaPrecision")
        fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "metrics")
        fn.decorator_list = []
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        env = {"np": self.np, "NearestNeighbors": NearestNeighbors}
        exec(compile(module, "pinned_author_equations", "exec"), env)
        real, synth = metrics.subset(self.val, 120), metrics.subset(self.synth, 120, metrics.SEED + 1)
        author = env["metrics"](None, real, synth)
        actual = metrics.alpha_beta(self.val, self.synth)
        self.np.testing.assert_array_equal(actual["precision_curve"], author[1])
        self.np.testing.assert_array_equal(actual["recall_curve"], author[2])
        self.assertEqual(actual["value"]["alpha_precision"], author[3])
        self.assertEqual(actual["value"]["beta_recall"], author[4])
        short = self.synth[:-7]
        n = min(len(self.val), len(short), metrics.MAX_ROWS)
        author = env["metrics"](None, metrics.subset(self.val, n),
                                metrics.subset(short, n, metrics.SEED + 1))
        actual = metrics.alpha_beta(self.val, short)
        self.assertEqual(actual["equal_subsample_rows"], n)
        self.assertEqual(actual["value"]["alpha_precision"], author[3])
        self.assertEqual(actual["value"]["beta_recall"], author[4])
        with self.assertRaisesRegex(ValueError, "author_source_pin_mismatch"):
            pinned_author_tree("synthcity/src/synthcity/metrics/eval_statistical.py", "0" * 64)

    def test_marginal_copy_shift_and_constant_pairs(self):
        copy = metrics.fidelity(self.val, self.val.copy(), ["continuous"] * 3)
        self.assertTrue(all(row["value"] == 0 for row in copy["marginals"]))
        shifted = metrics.fidelity(self.val, self.val + 100, ["continuous"] * 3)
        self.assertTrue(all(row["value"] == 1 for row in shifted["marginals"]))
        constant = metrics.fidelity(self.np.ones((20, 2)), self.np.ones((30, 2)), ["continuous"] * 2)
        self.assertEqual(constant["pairwise_pearson_difference"]["status"], "unavailable")

    def test_categorical_total_variation_and_contingency(self):
        a = self.np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        b = self.np.tile([[0, 0]], (4, 1))
        got = metrics.fidelity(a, b, ["categorical"] * 2)
        self.assertEqual(got["marginals"][0]["value"], .5)
        self.assertEqual(got["categorical_contingency_tv"]["value"]["mean"], .75)
        self.assertEqual(got["pairwise_pearson_eligibility"]["eligible_continuous_pairs"], 0)
        mixed = metrics.fidelity(self.np.column_stack((self.val[:, :2], a.repeat(30, axis=0))),
                                 self.np.column_stack((self.synth[:, :2], b.repeat(30, axis=0))),
                                 ["continuous", "continuous", "categorical", "categorical"])
        self.assertEqual(mixed["pairwise_pearson_eligibility"], {
            "total_pairs": 6, "eligible_continuous_pairs": 1, "excluded_constant_pairs": 0,
            "measured_pairs": 1, "mixed_pairs": 4, "categorical_pairs": 1})
        self.assertEqual(mixed["mixed_type_dependence"]["reason"], "mixed_type_dependence_not_measured")

    def test_grouped_detection_seed_replay_and_insufficient_groups(self):
        a = metrics.c2st(self.val, self.synth, "logistic")
        b = metrics.c2st(self.val, self.synth, "logistic")
        self.assertEqual(a, b)
        self.assertFalse(a["duplicate_group_leakage"])
        tiny = metrics.c2st(self.np.ones((20, 2)), self.np.ones((20, 2)), "logistic")
        self.assertEqual(tiny["status"], "unavailable")
        signed = self.np.array([[0., -0.], [-0., 0.]])
        self.assertEqual(*metrics.row_groups(signed))

    def test_duplicate_groups_span_detection_classes_and_real_null_is_disjoint(self):
        duplicated = self.np.repeat(self.val, 2, axis=0)
        copy = metrics.c2st(duplicated, duplicated.copy(), "logistic")
        self.assertEqual(copy["value"], .5)
        self.assertEqual(copy["pooled_oof_auc"], .5)
        self.assertFalse(copy["duplicate_group_leakage"])
        self.assertFalse(set(metrics.row_groups(self.val)) & set(metrics.row_groups(self.synth)))
        null = metrics.c2st(self.val, self.synth, "logistic")
        self.assertEqual(null["status"], "ok")
        self.assertEqual(null["rows_per_class"], 120)

    def test_copied_training_privacy_and_disjoint_domias(self):
        p = metrics.privacy(self.fit, self.val, self.fit.copy(), ["continuous"] * 3)
        self.assertEqual(p["dcr_fit"]["mean"], 0)
        self.assertEqual(p["share_closer_to_train"], 1)
        self.assertEqual(p["distance_mia"]["value"], 1)
        self.assertEqual(p["domias_kde"]["status"], "ok")
        self.assertFalse(p["formal_dp"])
        self.assertFalse(p["hipaa_deidentification"])
        duplicated_validation = self.np.repeat(self.val, 2, axis=0)
        self.assertEqual(metrics.privacy(self.fit, duplicated_validation, self.synth,
                         ["continuous"] * 3)["domias_kde"]["status"], "ok")

    def test_balanced_dcr_references_and_exact_half_ties(self):
        from sklearn.neighbors import NearestNeighbors
        val = self.np.repeat(self.val, 2, axis=0)
        got = metrics.privacy(self.fit, val, self.synth, ["continuous"] * 3)
        nref = min(len(self.fit), len(val), metrics.MAX_ROWS)
        synth = metrics.subset(self.synth, 120, metrics.SEED + 2)
        a = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(metrics.subset(self.fit, nref)).kneighbors(synth)[0][:, 0] / self.np.sqrt(3)
        b = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(metrics.subset(val, nref, metrics.SEED + 1)).kneighbors(synth)[0][:, 0] / self.np.sqrt(3)
        self.assertEqual(got["balanced_reference_rows"], nref)
        self.assertEqual(got["balanced_share_closer_to_train"], float(self.np.mean((a < b) + .5 * (a == b))))
        tied = metrics.privacy(self.np.array([[0.], [2.]]), self.np.array([[1.], [3.]]),
                               self.np.array([[.5], [2.5]]), ["continuous"])
        self.assertEqual(tied["share_closer_to_train"], .5)
        self.assertEqual(tied["balanced_share_closer_to_train"], .5)

    def test_unique_group_weighted_attack_queries(self):
        original = metrics.privacy(self.fit, self.val, self.synth, ["continuous"] * 3)
        repeated = metrics.privacy(self.np.repeat(self.fit, 2, axis=0),
                                   self.np.repeat(self.val, 3, axis=0), self.synth, ["continuous"] * 3)
        self.assertEqual(original["distance_mia"], repeated["distance_mia"])
        self.assertEqual(repeated["unique_query_group_counts"], {"fit": 120, "validation": 120})
        self.assertEqual(repeated["attack_query_weighting"], "uniform_unique_projected_row_groups")
        self.assertEqual(repeated["kde_reference_weighting"], "remaining_disjoint_real_rows")
        for rows in (self.fit, self.val):
            a = metrics.unique_group_rows(rows, metrics.SEED)
            b = metrics.unique_group_rows(self.np.repeat(rows, 3, axis=0), metrics.SEED)
            self.np.testing.assert_array_equal(a, b)

    def test_domias_ratio_matches_pinned_author_with_disjoint_reference_groups(self):
        from scipy.stats import gaussian_kde
        from sklearn.metrics import roc_auc_score
        tree = pinned_author_tree("DOMIAS/src/domias/evaluator.py",
                                 "d455600f448f8cf072def8dd8790b22c95f6316a69626311a35adfbc0d7b5f8c")
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_performance")
        ratio_assignment = next(node for node in ast.walk(fn) if isinstance(node, ast.Assign) and
                                any(isinstance(target, ast.Name) and target.id == "p_rel" for target in node.targets))
        expression = ast.fix_missing_locations(ast.Expression(body=ratio_assignment.value))
        calls = []
        def captured_kde(samples):
            kernel = gaussian_kde(samples)
            call = {"samples": samples.T.copy()}
            calls.append(call)
            def density(queries):
                call["queries"] = queries.T.copy()
                call["density"] = kernel(queries)
                return call["density"]
            return density
        with patch("scipy.stats.gaussian_kde", side_effect=captured_kde):
            got = metrics.privacy(self.fit, self.np.repeat(self.val, 2, axis=0), self.synth, ["continuous"] * 3)
        self.assertEqual(len(calls), 2)
        self.np.testing.assert_array_equal(calls[0]["queries"], calls[1]["queries"])
        nq = got["domias_kde"]["member_queries"]
        query = calls[0]["queries"]
        for rows in (query[:nq], query[nq:]):
            self.assertEqual(len(set(metrics.row_groups(rows))), nq)
        self.assertFalse(set(metrics.row_groups(calls[1]["samples"])) & set(metrics.row_groups(query[nq:])))
        self.assertFalse(set(metrics.row_groups(calls[1]["samples"])) & set(metrics.row_groups(self.fit)))
        self.assertEqual(len(calls[1]["samples"]), 120)
        author_ratio = eval(compile(expression, "pinned_author_domias_ratio", "eval"), {
            "p_G_evaluated": calls[0]["density"], "p_R_evaluated": calls[1]["density"]})
        y = self.np.concatenate((self.np.ones(nq), self.np.zeros(nq)))
        self.assertEqual(got["domias_kde"]["value"], float(roc_auc_score(y, author_ratio)))

    def test_invalid_column_kinds_and_numerical_kde_failure_are_typed(self):
        with self.assertRaisesRegex(ValueError, "explicit_column_types_required"):
            metrics.privacy(self.fit, self.val, self.synth, ["unknown"] * 3)
        with patch("scipy.stats.gaussian_kde", side_effect=self.np.linalg.LinAlgError("generated_control")):
            got = metrics.privacy(self.fit, self.val, self.synth, ["continuous"] * 3)
        self.assertEqual(got["domias_kde"]["reason"], "numerically_singular_full_dimensional_kde")
        self.assertIsNone(got["domias_kde"]["value"])
        self.assertEqual(got["distance_mia"]["status"], "ok")

    def test_singular_domias_and_cross_partition_rejection(self):
        fit, val, synth = (self.np.column_stack((x[:, 0], x[:, 0])) for x in (self.fit, self.val, self.synth))
        p = metrics.privacy(fit, val, synth, ["continuous"] * 2)
        self.assertEqual(p["domias_kde"]["reason"], "singular_full_dimensional_kde")
        with self.assertRaisesRegex(ValueError, "overlap"):
            metrics.privacy(self.fit, self.fit.copy(), self.synth, ["continuous"] * 3)

    def test_full_generated_panel_is_not_certification(self):
        from jsonschema import Draft202012Validator
        panel = metrics.evaluate(self.fit, self.val, self.synth, ["continuous"] * 3)
        schema = json.loads((HERE / "expanded-validation-metrics.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(panel)
        json.dumps(panel, allow_nan=False)
        self.assertEqual(panel["detection"]["catboost"]["folds"], 5)
        self.assertIsNone(panel["mfs_v2"])
        self.assertIsNone(panel["ptf_v1"])
        self.assertIsNone(panel["superiority"])
        self.assertFalse(panel["official_tests_opened"])

    def test_all_author_buffers_and_licenses_match_frozen_pins(self):
        root = HERE / "fixtures" / "expanded-validation-metrics"
        manifest = json.loads((root / "author-source-pins.json").read_text())
        self.assertFalse(manifest["candidate_execution"])
        for item in manifest["files"]:
            relative = item["path"] + (".txt" if item["path"].endswith(".py") else "")
            payload = (root / "author-source" / item["repository"] / relative).read_bytes()
            self.assertEqual(len(payload), item["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])

    def test_overlap_rejected_before_auditor_training(self):
        with patch.object(metrics, "c2st") as auditor:
            with self.assertRaisesRegex(ValueError, "training_validation_projected_overlap"):
                metrics.evaluate(self.fit, self.fit.copy(), self.synth, ["continuous"] * 3)
            auditor.assert_not_called()

    def test_module_import_does_not_initialize_numerical_dependencies(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "scipy", "sklearn", "catboost", "torch"}:
                raise AssertionError("numerical_dependency_initialized_before_caller_custody_check")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            isolated_spec = importlib.util.spec_from_file_location("isolated_diagnostics", HERE / "expanded_validation_metrics.py")
            isolated = importlib.util.module_from_spec(isolated_spec)
            isolated_spec.loader.exec_module(isolated)

    def test_schema_preserves_unavailable_metrics_and_rejects_claims(self):
        from jsonschema import Draft202012Validator
        validator = Draft202012Validator(json.loads((HERE / "expanded-validation-metrics.schema.json").read_text()))
        panel = metrics.evaluate(self.fit, self.val, self.synth, ["continuous"] * 3)
        panel["alpha_beta"] = metrics.unavailable("negative_author_alpha_beta_delta")
        panel["privacy"]["domias_kde"] = metrics.unavailable("singular_full_dimensional_kde")
        validator.validate(panel)
        for field, value in (("mfs_v2", .99), ("ptf_v1", .99), ("official_tests_opened", True)):
            with self.subTest(field=field):
                self.assertFalse(validator.is_valid(dict(panel, **{field: value})))

    def test_invalid_numeric_values_never_echo_source_fields(self):
        sentinel = "generated_private_field_do_not_echo"
        for rows in ([[sentinel], [sentinel]], [[1], [2, 3]], [[float("nan")], [1]]):
            with self.subTest(rows_type=type(rows).__name__):
                with self.assertRaises(ValueError) as failure:
                    metrics.checked_arrays(rows)
                self.assertNotIn(sentinel, str(failure.exception))
                self.assertEqual(str(failure.exception), "invalid_numeric_matrix_values")


if __name__ == "__main__":
    unittest.main(verbosity=2)
