"""Offline neural-generator conformance. No vendor runtime and no source rows."""

import hashlib
import json
from pathlib import Path
import unittest

from research.transfer.adapters.base import AdapterAdmission, AdapterError, FeatureBatch, TrainingBatch
from research.transfer.adapters.generators.neural import load_lock, neural_adapter, recipe_identity

ROOT = Path(__file__).resolve().parents[3]
LOCK = json.loads((ROOT / "research/transfer/locks/neural-generators.json").read_text())


def pin(text):
    return hashlib.sha256(text.encode()).hexdigest()


def registry(model):
    return {"format": "dope-adapter-registry-v1", "kind": "generator", "entries": [{
        "name": model.name, "status": "reviewed", "identity_sha256": model.sha256,
        "eligibility_sha256": pin("fixture eligibility"),
        "native_reference_sha256": pin("fixture reference"), "license_status": "compatible"}]}


def training():
    features = FeatureBatch(tuple(pin(f"train-{index}") for index in range(3)),
                            ((1.0,), (2.0,), (3.0,)), pin("neural schema"))
    return TrainingBatch(features, (10.0, 20.0, 123456789.0), "development_train")


def adapter(lock, name, recipe):
    identity = recipe_identity(lock, name, recipe)
    admission = AdapterAdmission.from_registry(registry(identity), identity, "generator")
    return neural_adapter(lock, name, recipe, admission)


class NeuralGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.lock = load_lock(json.dumps(LOCK))
        self.training = training()

    def test_scaled_tabsyn_samples_exports_and_reloads_without_the_full_schedule(self):
        model = adapter(self.lock, "tabsyn", "scaled")
        model.fit(self.training, 11)
        sample = model.sample(2, 5)
        self.assertEqual(len(sample.data.targets), 2)
        self.assertEqual(sample.data.features.schema_sha256, self.training.features.schema_sha256)
        self.assertFalse(set(sample.data.features.row_ids) & set(self.training.features.row_ids))
        state = model.export()
        self.assertFalse(state["vendor_executed"])
        self.assertEqual(state["schedule"], {"diffusion_epochs": 1000, "sampling_steps": 50, "vae_epochs": 200})
        self.assertFalse(state["full_schedule"])
        model.reload(json.loads(json.dumps(state)))
        self.assertEqual(model.export(), state)
        fitted = [item for item in model.attempts if item["status"] == "fitted"]
        self.assertEqual(fitted[0]["schedule"]["vae_epochs"], 200)
        self.assertNotEqual(fitted[0]["schedule"]["diffusion_epochs"], 10001)
        visible = {item["recipe"] for item in model.attempts if item["status"] in ("visible", "timeout")}
        self.assertIn("native_full", visible)
        self.assertIn("common_transfer", visible)

    def test_scaled_tabsyn_keeps_the_full_schedule_visible_and_unrun(self):
        model = adapter(self.lock, "tabsyn", "native_full")
        with self.assertRaises(AdapterError) as raised:
            model.fit(self.training, 11)
        self.assertEqual(str(raised.exception), "scaled_called_full_schedule")
        self.assertNotIn("123456789", str(raised.exception))
        recipes = {item["recipe"] for item in model.attempts}
        self.assertIn("native_full", recipes)
        self.assertTrue(any(item["status"] == "timeout" and item["eligible"] is False for item in model.attempts))
        self.assertTrue(any(item["lane"] == "unrestricted_quality" and item["within_budget"] is False
                            for item in model.attempts))

    def test_reloading_a_full_schedule_into_scaled_tabsyn_fails(self):
        model = adapter(self.lock, "tabsyn", "scaled")
        model.fit(self.training, 11)
        state = model.export()
        state["full_schedule"] = True
        with self.assertRaises(AdapterError) as raised:
            model.reload(state)
        self.assertEqual(str(raised.exception), "scaled_called_full_schedule")

    def test_common_transfer_is_a_different_identity_from_scaled(self):
        scaled = recipe_identity(self.lock, "tabsyn", "scaled")
        common = recipe_identity(self.lock, "tabsyn", "common_transfer")
        self.assertNotEqual(scaled.sha256, common.sha256)
        self.assertNotEqual(scaled.recipe_sha256, common.recipe_sha256)

    def test_tabdiff_and_tabddpm_keep_author_schedules_as_unbudgeted_references(self):
        for name, author_key, author_value in (
            ("tabdiff", "training_steps", 8000),
            ("tabddpm", "training_steps", 1000),
        ):
            with self.subTest(name=name):
                model = adapter(self.lock, name, "resource_matched")
                model.fit(self.training, 3)
                state = model.export()
                self.assertFalse(state["full_schedule"])
                self.assertNotIn(author_key, state["schedule"])
                model.reload(state)
                reference = next(item for item in model.attempts if item["recipe"] == "native_full")
                self.assertEqual(reference["schedule"][author_key], author_value)
                self.assertEqual(reference["lane"], "unrestricted_quality")
                self.assertFalse(reference["within_budget"])


if __name__ == "__main__":
    unittest.main()
