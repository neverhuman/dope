"""Auditor ABI smoke tests, using the pinned native CatBoost port."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from research.transfer.adapters.base import FeatureBatch, ImportanceResult
from research.transfer.adapters.auditors.conventional import ConventionalAdapter, ConventionalLearnerRecipe, file_sha256
from research.transfer.tests.test_adapters import pin
from research.transfer.tests.test_other_auditors import PortFixture, batch, query
from research.transfer.tdl.drivers import evaluate


class TdlAuditorTests(unittest.TestCase):
    def test_correlated_feature_substitution_preserves_grouped_permutation_drivers(self):
        fixture = json.loads((Path(__file__).resolve().parents[3] /
                              "tests/fixtures/tdl/correlated_features.json").read_text())
        rows, targets = fixture["real_test_rows"], fixture["targets"]
        vectors = []
        for selected in (fixture["real_model_feature"], fixture["synthetic_model_feature"]):
            def loss(table):
                return sum((row[selected] + .5 * row[2] - y)**2 / len(rows)
                           for row, y in zip(table, targets))
            self.assertEqual(loss(rows), 0.)
            importances = []
            for group in fixture["groups"]:
                perturbed = [row[:] for row in rows]
                for i, source in enumerate(fixture["permutation"]):
                    for feature in group:
                        perturbed[i][feature] = rows[source][feature]
                importances.append(loss(perturbed))
            self.assertEqual(importances, fixture["expected_group_importances"])
            vectors.append(importances)
        report = evaluate({"real_importances": vectors[0], "synthetic_importances": vectors[1],
                           "required_components": ["mass", "rank"], "real_real_ceiling": {"mass": 1., "rank": 1.}})
        self.assertEqual(report["driver_fidelity"], 1.)

    def test_catboost_fit_predict_grouped_importance_and_portable_receipt(self):
        recipe = ConventionalLearnerRecipe("catboost", "regression", (("iterations", 12), ("depth", 2)),
                                           pin("fixture development selection"), precision="float32")
        fixture = PortFixture(self, recipe, "catboost", pin("native conventional fixture"), mode="scratch")
        model = replace(fixture.identity, dependencies=(("catboost", "1.2.10", file_sha256(fixture.artifacts.dependencies[0])),))
        runtime = replace(fixture.runtime, factory=None, version_reader=lambda _: "1.2.10")
        port = fixture.adapter(ConventionalAdapter, recipe, model, runtime)
        receipt = port.fit(batch(targets=(1000., 3000., 5000.)), 7)
        self.assertEqual(receipt["consumed_rows"], 3)
        real_test = query()
        predictions = port.predict(real_test, 8)
        self.assertTrue(all(1000. <= v[0] <= 5000. for v in predictions.values))
        perturbed = FeatureBatch(real_test.row_ids, tuple(reversed(real_test.rows)), real_test.schema_sha256)
        def loss(prediction):
            return sum((value[0] - target)**2 / 2 for value, target in zip(prediction.values, (4000., 4500.)))
        importance = port.importance(real_test, ((0, (perturbed,)),), loss, pin("grouped conditional protocol"), 8)
        self.assertIsInstance(importance, ImportanceResult)
        self.assertEqual(importance.query_sha256, real_test.sha256)
        self.assertEqual(importance.group_ids, (0,))
        self.assertNotIn("/home/", json.dumps(port.receipt()))
        self.assertNotIn("4000", json.dumps(port.receipt()))
