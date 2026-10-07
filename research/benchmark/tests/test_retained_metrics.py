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

    def multiple_fit_fixture(self, root, *, missing_sample=False):
        """Two independent fits of one existing public six-sample lineage."""
        def save(name, value):
            path = root / name
            body = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
            path.write_bytes(body)
            return dict(path=str(path), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())

        first_dataset = self.panel["cells"][0]["dataset"]
        originals = [cell for cell in self.panel["cells"] if cell["dataset"] == first_dataset]
        jobs, refs = [], []
        for seed in (23, 37):
            for original in originals:
                if missing_sample and seed == 37 and original["row_multiplier"] == 4 and original["sample_seed"] == 307:
                    continue
                receipt = copy.deepcopy(original)
                identity = dict(dataset=first_dataset, fit_seed=seed,
                                row_multiplier=receipt["row_multiplier"], sample_seed=receipt["sample_seed"])
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                receipt.update(job_identity=identity, job_sha256=digest, fit_seed=seed, status="ok",
                               input_refs={}, source_refs=self.panel["source_refs"])
                refs.append(save(digest + ".json", receipt))
                jobs.append(dict(job_identity=identity, job_sha256=digest, evaluator_input=receipt))
        manifest_ref = save("inputs.json", dict(jobs=jobs))
        lock_ref = save("lock.json", dict(actual_complete=True, official_tests_opened=False,
                        count=len(refs), receipts=refs, source_refs=self.panel["source_refs"]))
        execution_ref = save("execution.json", dict(receipt_lock_ref=lock_ref, metric_receipts=len(refs),
                             source_refs=dict(original_metric_input_manifest=manifest_ref,
                                              measurement_driver=manifest_ref, runtime_inventory=manifest_ref)))
        return lock_ref, root, manifest_ref, execution_ref

    def test_independent_fit_seeds_have_separate_dataset_summaries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[3] / "target") as tmp:
            panel = publish.build(*self.multiple_fit_fixture(Path(tmp)), by_fit_seed=True)
            self.assertEqual(panel["sample_cells"], 12)
            self.assertEqual({(row["fit_seed"], row["row_multiplier"]) for row in panel["summary"]},
                             {(23, 1), (23, 4), (37, 1), (37, 4)})
            self.assertTrue(all(row["datasets"] == 1 and row["cells"] == 3 for row in panel["summary"]))
            self.assertTrue(all(panel[key] is None for key in ("mfs_v2", "ptf_v1", "superiority")))

    def test_old_single_fit_mode_rejects_multiple_independent_fits(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[3] / "target") as tmp:
            with self.assertRaisesRegex(ValueError, "incomplete_three_sample_seed_group"):
                publish.build(*self.multiple_fit_fixture(Path(tmp)))

    def test_per_fit_summary_rejects_missing_sample_seed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[3] / "target") as tmp:
            with self.assertRaisesRegex(ValueError, "incomplete_three_sample_seed_group"):
                publish.build(*self.multiple_fit_fixture(Path(tmp), missing_sample=True), by_fit_seed=True)

    def test_committed_cpu_replication_metrics_keep_fit_and_split_bindings(self):
        root = Path(__file__).parents[1] / "results/cpu-fivefit-validation-v1"
        manifest = json.loads((root / "manifest.json").read_text())
        repo = Path(__file__).parents[3]
        producer = repo / manifest["producer_ref"]["path"]
        self.assertEqual(hashlib.sha256(producer.read_bytes()).hexdigest(), manifest["producer_ref"]["sha256"])
        self.assertEqual(manifest["sample_cells"], 192)
        self.assertFalse(manifest["five_fit_matrix_complete"])
        self.assertFalse(manifest["full_population_complete"])
        import jsonschema
        jsonschema.Draft202012Validator(json.loads((root / "manifest.schema.json").read_text())).validate(manifest)
        for method, expected in (("arf", 144), ("forest", 48)):
            panel = json.loads((root / method / "panel.json").read_text())
            fits = json.loads((repo / manifest["inputs"][method]["published_fit_panel"]["path"]).read_text())["fits"]
            by_identity = {(row["dataset"], row["fit_seed"], row["configuration_sha256"]): row for row in fits}
            self.assertEqual(panel["sample_cells"], expected)
            self.assertEqual({x["fit_seed"] for x in panel["summary"]},
                             {23} if method == "arf" else {23, 37, 53, 71})
            for cell in panel["cells"]:
                fitted = by_identity[cell["dataset"], cell["fit_seed"], cell["config_sha256"]]
                self.assertEqual(cell["charged_artifact_bytes"], fitted["artifact_bytes"])
                self.assertEqual(cell["input_hashes"]["train_ref"], fitted["train_sha256"])
                self.assertEqual(cell["input_hashes"]["validation_ref"], fitted["validation_sha256"])
                self.assertEqual(cell["selection_binding"], ",".join(fitted["configuration_labels"]))
            rebuilt = {**panel, "metric_rows": [publish.flatten(cell) for cell in panel["cells"]]}
            self.assertEqual(publish.render(rebuilt)["figure-kpis.csv"], (root / method / "figure-kpis.csv").read_bytes())
            jsonschema.Draft202012Validator(json.loads((root / "panel.schema.json").read_text())).validate(panel)
        for item in manifest["files"].values():
            body = (repo / item["path"]).read_bytes()
            self.assertEqual((len(body), hashlib.sha256(body).hexdigest()), (item["bytes"], item["sha256"]))


if __name__ == "__main__":
    unittest.main()
