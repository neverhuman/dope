import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import measure_retained_samples as measure
from research.benchmark import publish_retained_metrics as publish


class InputTests(unittest.TestCase):
    def test_unapproved_metric_source_rejected_before_loader_initialization(self):
        with patch.object(measure.importlib.util, "spec_from_file_location") as loader:
            with self.assertRaisesRegex(ValueError, "unapproved_metric"):
                measure.load_source(dict(path="never-read.py", sha256="0" * 64), "frozen_pilot_metrics")
            loader.assert_not_called()

    def test_added_metric_bytecode_rejected_before_loader_initialization(self):
        ref = dict(path="pinned.py", sha256="6c03892685df856f7a491ca79acb1c28fe8270ead8bbfea1ba50228cb6f0e7e3")
        with patch.object(measure, "read_verified"), patch.object(Path, "exists", return_value=True), \
                patch.object(measure.importlib.util, "spec_from_file_location") as loader:
            with self.assertRaisesRegex(ValueError, "unexpected_metric_bytecode"):
                measure.load_source(ref, "frozen_pilot_metrics")
            loader.assert_not_called()

    def test_loader_executes_captured_bytes_despite_later_source_and_cache_mutation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[3] / "target") as tmp:
            path = Path(tmp) / "pinned.py"
            path.write_bytes(b"identity = 'verified'\n")
            original_spec = measure.importlib.util.spec_from_file_location

            def mutate_after_verification(*args, **kwargs):
                path.write_bytes(b"raise AssertionError('unverified initializer')\n")
                cache = Path(measure.importlib.util.cache_from_source(str(path)))
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(b"unverified bytecode")
                return original_spec(*args, **kwargs)

            ref = dict(path=str(path), sha256="6c03892685df856f7a491ca79acb1c28fe8270ead8bbfea1ba50228cb6f0e7e3")
            with patch.object(measure, "read_verified", return_value=path.read_bytes()), \
                    patch.object(measure.importlib.util, "spec_from_file_location", side_effect=mutate_after_verification):
                module = measure.load_source(ref, "frozen_pilot_metrics")
            self.assertEqual(module.identity, "verified")

    def test_constant_target_is_uninformative_without_division_by_zero(self):
        result = measure.utility_outcome(0., 0., 0.)
        self.assertFalse(result["informative"])
        self.assertIsNone(result["retention"])
        self.assertTrue(result["low_signal_noninferior"])

    def test_fifteen_percent_ram_floor_with_whole_reservation(self):
        with self.assertRaisesRegex(ValueError, "insufficient_ram"):
            measure.capacity("MemTotal: 100000000 kB\nMemAvailable: 15000000 kB\n")
        self.assertEqual(measure.capacity("MemTotal: 100000000 kB\nMemAvailable: 20000000 kB\n")["free_ram_floor_fraction"], .15)

    def test_digest_rejects_float_alias_before_numeric_import(self):
        identity = {"fit_seed": 11}
        job = {"job_identity": {"fit_seed": 11.0}, "job_sha256": measure.digest(identity)}
        with self.assertRaisesRegex(ValueError, "job_identity_mismatch"):
            measure.check_job(job)

    def test_input_digest_rejects_corruption_without_decoding(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[3] / "target") as tmp:
            p = Path(tmp) / "numeric.csv"
            p.write_bytes(b"1,2\n")
            with self.assertRaisesRegex(ValueError, "input_digest_mismatch"):
                measure.read_verified(dict(path=str(p), sha256="0" * 64))

    def test_sealed_test_rejected_before_read(self):
        identity = {"fit_seed": 11}
        cell = dict(official_tests_opened=False, column_order="projected_inputs_then_target",
                    train_ref={"path": "train.csv"}, validation_ref={"path": "test.csv"},
                    synthetic_ref={"path": "synthetic.csv"}, column_kinds=["continuous"])
        with self.assertRaisesRegex(ValueError, "sealed_test_forbidden"):
            measure.check_job(dict(job_identity=identity, job_sha256=measure.digest(identity), evaluator_input=cell))


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parents[1] / "results/tabsyn-eight-lineage-validation"
        self.panel = json.loads((self.root / "panel.json").read_text())

    def test_complete_actual_panel_and_dataset_aggregation(self):
        self.assertEqual((self.panel["sample_cells"], self.panel["lineages"]), (48, 8))
        groups = {(x["dataset"], x["row_multiplier"]): set() for x in self.panel["cells"]}
        for cell in self.panel["cells"]:
            self.assertTrue(cell["receipt_ref"]["path"].endswith(cell["job_sha256"] + ".json"))
            groups[cell["dataset"], cell["row_multiplier"]].add(cell["sample_seed"])
            self.assertEqual(cell["metrics"]["detection"]["catboost"]["folds"], 5)
            self.assertFalse(cell["metrics"]["detection"]["catboost"]["duplicate_group_leakage"])
        self.assertTrue(all(seeds == {101, 211, 307} for seeds in groups.values()))
        self.assertAlmostEqual(self.panel["summary"][1]["catboost_retention"]["median"], .9112758346655363)

    def test_committed_csv_reproduces_from_committed_metric_cells(self):
        panel = copy.deepcopy(self.panel)
        panel["metric_rows"] = [publish.flatten(cell) for cell in panel["cells"]]
        self.assertEqual(publish.render(panel)["figure-kpis.csv"], (self.root / "figure-kpis.csv").read_bytes())

    def test_json_schema_and_null_claims(self):
        import jsonschema
        jsonschema.Draft202012Validator(json.loads((self.root / "panel.schema.json").read_text())).validate(self.panel)
        for key in ("mfs_v2", "ptf_v1", "release_safe_l3", "superiority"):
            self.assertIsNone(self.panel[key])
            self.assertTrue(all(cell[key] is None for cell in self.panel["cells"]))

    def test_rewritten_receipt_rejected_against_frozen_hash(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[3] / "target") as tmp:
            path = Path(tmp) / "receipt.json"
            original = b'{"utility": 0.2}'
            path.write_bytes(b'{"utility": 0.9}')
            with self.assertRaisesRegex(ValueError, "frozen_receipt_changed"):
                publish.verified(path, dict(bytes=len(original), sha256=hashlib.sha256(original).hexdigest()))


if __name__ == "__main__":
    unittest.main()
