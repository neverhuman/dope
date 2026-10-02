import json
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark import admission
from research.benchmark.admission import assess
from research.benchmark.manifest import digest
from research.benchmark.score import CONTRACT, artifact_inventory, sha256


ROOT = Path(__file__).resolve().parents[3]


class AdmissionTests(unittest.TestCase):
    def test_missing_final_locks_keep_test_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            result = assess(ROOT, Path(directory))
            self.assertFalse(result["admitted"])
            self.assertIn("budget.lock.json:missing_or_invalid", result["blockers"])

    def test_incomplete_method_lock_keeps_test_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            lock_root = Path(directory)
            (lock_root / "methods.lock.json").write_text(json.dumps({
                "complete": False, "frozen_for_final_evaluation": False,
                "methods": {"dope": {"status": "locked"}}}))
            result = assess(ROOT, lock_root)
            self.assertIn("methods.lock.json:not_frozen", result["blockers"])
            self.assertIn("methods.lock.json:source_or_config_gap", result["blockers"])

    def fixture(self, lock_root):
        locks = {
            "methods.lock.json": {"methods": {"dope": {
                "status": "locked", "source_sha256": "source", "adapter_sha256": "adapter",
                "default_config": {}, "tuning_search_space": {},
                "dependency_or_container_digest": "dependency", "license": "MIT",
                "license_evidence": "https://example.com/license",
                "native_objective": {"status": "inapplicable", "reason": "default-only unit fixture"},
                "fit_command": "fit", "sampling_command": "sample"}}},
            "datasets.lock.json": {"datasets": [{"id": "toy", "train_rows": 100,
                "license": {"status": "recorded"}, "source_row_hash": "rows",
                "projection_sha256": "projection",
                "split": {"hashes": {"train": "a", "validation": "b", "test": "c"}},
                "projected_files": {"train": "a", "validation": "b", "test": "c"}}]},
            "budget.lock.json": {"reporting_milestone_days": 14,
                                 "scratch_ceiling_bytes": 200_000_000_000,
                                 "tuning_trials_per_method_dataset": 8,
                                 "tuning_wall_time_hours_per_method_dataset": 12,
                                 "pilot_receipts_sha256": "receipt"},
            "evaluator.lock.json": {"metric_implementation_sha256": "metric"},
            "method-dataset-matrix.lock.json": {"cells": [{
                "id": "toy-dope-default-11", "dataset": "toy", "method": "dope",
                "panel": "public_core", "track": "common_numeric", "tier": "l3",
                "applicable": True, "fit_seed": 11, "sample_seeds": [101, 211, 307],
                "size_multipliers": [1, 2, 4, 8], "worker_dir": "/mnt/fast-scratch/dope-benchmark/toy",
                "scratch_reservation_bytes": 1000, "memory_reservation_bytes": 1000,
                "requires_gpu": False, "gpu_vram_mib": 0,
                "timeout_seconds": 600, "runtime_python": "/usr/bin/python3",
                "configuration": {"kind": "default"}}]},
        }
        # Synthetic metadata hashes are valid digest shapes, not benchmark evidence.
        method = locks["methods.lock.json"]["methods"].pop("dope")
        locks["methods.lock.json"]["methods"]["author_default_fixture"] = method
        for key in ("source_sha256", "adapter_sha256", "dependency_or_container_digest"):
            method[key] = digest(method[key])
        dataset = locks["datasets.lock.json"]["datasets"][0]
        for key in ("source_row_hash", "projection_sha256"):
            dataset[key] = digest(dataset[key])
        for mapping in (dataset["split"]["hashes"], dataset["projected_files"]):
            for key in mapping:
                mapping[key] = digest(mapping[key])
        locks["budget.lock.json"]["pilot_receipts_sha256"] = digest("fixture-pilot-receipt")
        evaluator = locks["evaluator.lock.json"]
        evaluator["metric_implementation_sha256"] = digest("fixture-evaluator")
        template = locks["method-dataset-matrix.lock.json"]["cells"][0]
        template.update({"method": "author_default_fixture", "method_source_sha256": method["source_sha256"],
            "dependency_or_container_digest": method["dependency_or_container_digest"],
            "adapter_sha256": method["adapter_sha256"], "projection_sha256": dataset["projection_sha256"],
            "train_sha256": dataset["projected_files"]["train"],
            "validation_sha256": dataset["projected_files"]["validation"],
            "evaluator_sha256": evaluator["metric_implementation_sha256"]})
        cells = []
        for seed in CONTRACT["fit_seeds"]:
            cell = copy.deepcopy(template)
            cell.update({"id": f"toy-author-default-{seed}", "fit_seed": seed})
            cells.append(cell)
        locks["method-dataset-matrix.lock.json"]["cells"] = cells
        for name, body in locks.items():
            body.update({"complete": True, "frozen_for_final_evaluation": True})
        self.write_locks(lock_root, locks)
        return locks

    def write_locks(self, lock_root, locks, seal=True):
        if seal:
            binding = admission.frozen_set_digest(locks)
            for body in locks.values():
                body["freeze_set_sha256"] = binding
        for name, body in locks.items():
            (lock_root / name).write_text(json.dumps(body))

    def test_complete_locks_admit_only_locked_applicable_cells(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            lock_root = Path(directory)
            locks = self.fixture(lock_root)
            self.assertTrue(assess(ROOT, lock_root)["admitted"])
            locks["method-dataset-matrix.lock.json"]["cells"][0]["method"] = "unavailable"
            (lock_root / "method-dataset-matrix.lock.json").write_text(
                json.dumps(locks["method-dataset-matrix.lock.json"]))
            self.assertIn("method-dataset-matrix.lock.json:cell_gap",
                          assess(ROOT, lock_root)["blockers"])

    def test_placeholder_digest_or_stale_lock_binding_never_admits(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            locks = self.fixture(root)
            locks["budget.lock.json"]["pilot_receipts_sha256"] = "receipt"
            self.write_locks(root, locks)
            self.assertIn("final_lock_set:invalid_sha256", assess(ROOT, root)["blockers"])
            locks["budget.lock.json"]["pilot_receipts_sha256"] = digest("other-receipts")
            self.write_locks(root, locks, seal=False)
            self.assertIn("final_lock_set:hash_binding_gap", assess(ROOT, root)["blockers"])

    def test_one_seed_or_duplicate_seed_is_not_a_complete_matrix(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            locks = self.fixture(root)
            altered = copy.deepcopy(locks)
            altered["method-dataset-matrix.lock.json"]["cells"] = altered["method-dataset-matrix.lock.json"]["cells"][:1]
            self.write_locks(root, altered)
            self.assertIn("method-dataset-matrix.lock.json:fit_schedule_gap", assess(ROOT, root)["blockers"])
            locks["method-dataset-matrix.lock.json"]["cells"][-1]["fit_seed"] = 11
            self.write_locks(root, locks)
            self.assertIn("method-dataset-matrix.lock.json:fit_schedule_gap", assess(ROOT, root)["blockers"])

    def test_omitted_pair_requires_an_explicit_exclusion(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            locks = self.fixture(root)
            locks["methods.lock.json"]["methods"]["unavailable_fixture"] = {"status": "unavailable", "note": "fixture"}
            self.write_locks(root, locks)
            self.assertIn("method-dataset-matrix.lock.json:pair_coverage_gap", assess(ROOT, root)["blockers"])
            locks["method-dataset-matrix.lock.json"]["cells"].append({"dataset": "toy", "method": "unavailable_fixture",
                "panel": "public_core", "track": "common_numeric", "tier": "l3", "applicable": False,
                "exclusion_reason": "fixture source unavailable"})
            self.write_locks(root, locks)
            self.assertTrue(assess(ROOT, root)["admitted"])

    def test_matrix_lineage_drift_is_rejected_even_after_resealing(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            locks = self.fixture(root)
            for field in ("method_source_sha256", "adapter_sha256", "projection_sha256",
                          "dependency_or_container_digest", "train_sha256", "validation_sha256", "evaluator_sha256"):
                altered = copy.deepcopy(locks)
                altered["method-dataset-matrix.lock.json"]["cells"][0][field] = digest("drift")
                self.write_locks(root, altered)
                self.assertIn("method-dataset-matrix.lock.json:lineage_binding_gap", assess(ROOT, root)["blockers"])

    def test_unimplemented_author_faithful_track_cannot_open_tests(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            locks = self.fixture(root)
            for cell in locks["method-dataset-matrix.lock.json"]["cells"]:
                cell["track"] = "author_faithful"
            self.write_locks(root, locks)
            self.assertIn("method-dataset-matrix.lock.json:cell_gap", assess(ROOT, root)["blockers"])

    def test_native_objective_requires_both_tracks_and_verified_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            locks = self.fixture(root)
            locks["methods.lock.json"]["methods"]["author_default_fixture"]["native_objective"] = {
                "status": "locked", "name": "fixture-native-kpi", "direction": "maximize",
                "implementation_sha256": digest("fixture-metric"),
                "tie_breaks": ["artifact_bytes_ascending", "config_sha256_ascending"]}
            self.write_locks(root, locks)
            self.assertIn("method-dataset-matrix.lock.json:fit_schedule_gap", assess(ROOT, root)["blockers"])
            cells = locks["method-dataset-matrix.lock.json"]["cells"]
            tuned = copy.deepcopy(cells)
            for cell in tuned:
                cell["id"] += "-tuned"
                cell["configuration"] = {"kind": "tuned", "values": {}, "selection_path": str(root / "missing-selection.json"),
                                         "selection_sha256": digest("missing-selection")}
            cells.extend(tuned)
            self.write_locks(root, locks)
            result = assess(ROOT, root)
            self.assertNotIn("method-dataset-matrix.lock.json:fit_schedule_gap", result["blockers"])
            self.assertIn("method-dataset-matrix.lock.json:native_selection_evidence_gap", result["blockers"])
            method = locks["methods.lock.json"]["methods"]["author_default_fixture"]
            objective = method["native_objective"]
            validation_hash = locks["datasets.lock.json"]["datasets"][0]["projected_files"]["validation"]
            artifact = root / "trial/artifact"
            artifact.mkdir(parents=True)
            for name in ("model.json", "projection.json"):
                (artifact / name).write_text("{}")
            inventory, charged = artifact_inventory(artifact, ["model.json", "projection.json"])
            model_hash = sha256(artifact / "model.json")
            metric = root / "trial/native-metric.json"
            metric.write_text(json.dumps({"method": "author_default_fixture", "partition": "validation",
                "objective": objective["name"], "implementation_sha256": objective["implementation_sha256"],
                "value": 1.0, "validation_sha256": validation_hash, "artifact_sha256": model_hash}))
            trial = {"status": "ok", "wall_seconds": 1, "config": {}, "native_kpi": 1.0,
                "artifact_bytes": charged, "artifact_inventory": inventory, "artifact_sha256": model_hash,
                "metric_receipt_path": str(metric), "metric_receipt_sha256": sha256(metric),
                "identity": {"dataset": "toy", "method": "author_default_fixture",
                    "round_sha256": digest("fixture-round"), "method_source_sha256": method["source_sha256"],
                    "validation_sha256": validation_hash, "config": {}}}
            attempt = root / "trial/attempt.json"
            attempt.write_text(json.dumps(trial))
            trial.update({"attempt_receipt_path": str(attempt), "attempt_receipt_sha256": sha256(attempt)})
            selection = root / "selection.json"
            selection.write_text(json.dumps({"format": "dope-benchmark-validation-selection",
                "partition": "validation", "validation_sha256": validation_hash, "test_opened": False,
                "dataset": "toy", "method": "author_default_fixture", "round_sha256": digest("fixture-round"),
                "objective": objective, "selected_config": {}, "selected_trial_index": 0, "trials": [trial]}))
            for cell in tuned:
                cell["configuration"].update({"selection_path": str(selection), "selection_sha256": sha256(selection)})
            self.write_locks(root, locks)
            with patch.object(admission, "SCRATCH_ROOT", root):
                self.assertTrue(assess(ROOT, root)["admitted"])
                sealed = root / "prepared/evaluator/toy/test.csv"
                sealed.parent.mkdir(parents=True)
                sealed.write_text("0,1\n1,0\n")
                original_open = Path.open

                def guarded_open(path, *args, **kwargs):
                    if path.resolve() == sealed.resolve():
                        raise AssertionError("native evidence opened sealed data")
                    return original_open(path, *args, **kwargs)

                # All three receipt readers reject a link to evaluator data
                # before hashing, including when the apparent suffix is JSON.
                for path in (selection, attempt, metric, artifact):
                    saved = path.with_name(path.name + ".saved")
                    path.rename(saved)
                    path.symlink_to(sealed if path != artifact else sealed.parent, target_is_directory=path == artifact)
                    with patch.object(Path, "open", guarded_open):
                        self.assertFalse(assess(ROOT, root)["admitted"])
                    path.unlink()
                    saved.rename(path)
                for cell in tuned:
                    cell["configuration"]["selection_path"] = str(sealed)
                self.write_locks(root, locks)
                with patch.object(Path, "open", guarded_open):
                    self.assertFalse(assess(ROOT, root)["admitted"])
                for cell in tuned:
                    cell["configuration"]["selection_path"] = str(selection)
                self.write_locks(root, locks)
                self.assertTrue(assess(ROOT, root)["admitted"])

                loop = root / "loop.json"
                loop.symlink_to(loop)
                for cell in tuned:
                    cell["configuration"].update(selection_path=str(loop), selection_sha256="0" * 64)
                self.write_locks(root, locks)
                self.assertIn("method-dataset-matrix.lock.json:native_selection_evidence_gap",
                              assess(ROOT, root)["blockers"])
                for cell in tuned:
                    cell["configuration"].update(selection_path=str(selection), selection_sha256=sha256(selection))

                # Rebind hashes so each malformed JSON object reaches its
                # shape check instead of merely failing an earlier checksum.
                pristine = {p: p.read_bytes() for p in (selection, attempt, metric)}
                for bad in (selection, attempt, metric):
                    bad.write_text("[]")
                    if bad != selection:
                        payload = json.loads(selection.read_text())
                        selected_trial = payload["trials"][0]
                        if bad == metric:
                            selected_trial["metric_receipt_sha256"] = sha256(metric)
                            attempt.write_text(json.dumps({k: v for k, v in selected_trial.items()
                                if k not in ("attempt_receipt_path", "attempt_receipt_sha256")}))
                        selected_trial["attempt_receipt_sha256"] = sha256(attempt)
                        selection.write_text(json.dumps(payload))
                    for cell in tuned:
                        cell["configuration"]["selection_sha256"] = sha256(selection)
                    self.write_locks(root, locks)
                    self.assertIn("method-dataset-matrix.lock.json:native_selection_evidence_gap",
                                  assess(ROOT, root)["blockers"])
                    for p, content in pristine.items():
                        p.write_bytes(content)
                for cell in tuned:
                    cell["configuration"]["selection_sha256"] = sha256(selection)
                self.write_locks(root, locks)
                payload = json.loads(selection.read_text())
                for field in ("wall_seconds", "native_kpi"):
                    payload["trials"][0][field] = 10**400
                    if field == "native_kpi":
                        measured = json.loads(pristine[metric])
                        measured["value"] = 10**400
                        metric.write_text(json.dumps(measured))
                        payload["trials"][0]["metric_receipt_sha256"] = sha256(metric)
                        attempt.write_text(json.dumps({k: v for k, v in payload["trials"][0].items()
                            if k not in ("attempt_receipt_path", "attempt_receipt_sha256")}))
                        payload["trials"][0]["attempt_receipt_sha256"] = sha256(attempt)
                    selection.write_text(json.dumps(payload))
                    for cell in tuned:
                        cell["configuration"]["selection_sha256"] = sha256(selection)
                    self.write_locks(root, locks)
                    self.assertFalse(assess(ROOT, root)["admitted"])
                    for p, content in pristine.items():
                        p.write_bytes(content)
                    payload = json.loads(pristine[selection])
                selection.write_bytes(pristine[selection])
                for cell in tuned:
                    cell["configuration"]["selection_sha256"] = sha256(selection)
                self.write_locks(root, locks)
                (artifact / "model.json").write_text('{"changed":true}')
                self.assertIn("method-dataset-matrix.lock.json:native_selection_evidence_gap", assess(ROOT, root)["blockers"])

    def test_admission_does_not_open_test_rows(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            self.fixture(root)
            original = Path.open

            def sealed_open(path, *args, **kwargs):
                if path.name == "test.csv":
                    raise AssertionError("admission opened a test partition")
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", sealed_open):
                self.assertTrue(assess(ROOT, root)["admitted"])

    def test_compact_runtime_digest_and_runner_agree(self):
        from research.benchmark import adapters, runner
        runtime = ROOT / "research/benchmark/numpy-runtime.lock.json"
        versions = json.loads(runtime.read_text())["versions"]
        for name in ("independent_marginals", "Chow-Liu"):
            with self.subTest(method=name), tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
                root = Path(directory)
                locks = self.fixture(root)
                methods = locks["methods.lock.json"]["methods"]
                method = methods.pop("author_default_fixture")
                methods[name] = method
                method.update({"adapter": name, "adapter_sha256": sha256(Path(adapters.__file__)),
                               "dependency_or_container_digest": sha256(runtime),
                               "dependency_versions": versions})
                for cell in locks["method-dataset-matrix.lock.json"]["cells"]:
                    cell.update({"method": name, "adapter_sha256": method["adapter_sha256"],
                                 "dependency_or_container_digest": method["dependency_or_container_digest"]})
                self.write_locks(root, locks)
                self.assertTrue(assess(ROOT, root)["admitted"])
                from importlib import metadata
                with patch.object(metadata, "version", side_effect=metadata.PackageNotFoundError("numpy")):
                    self.assertIn("methods.lock.json:compact_runtime_evidence_gap", assess(ROOT, root)["blockers"])
                # Final execution passes the same dependency proof and reaches
                # the deliberately absent worker manifest without fitting.
                with patch.object(runner, "call_adapter", side_effect=AssertionError("fit forbidden")):
                    with self.assertRaisesRegex(FileNotFoundError, "worker-manifest.json"):
                        runner.run({"method": name, "final": True, "fit_seed": 11,
                                    "track": "common_numeric", "worker_dir": str(root / "missing-worker")},
                                   locks["methods.lock.json"], root / "unused")
                method["dependency_or_container_digest"] = "0" * 64
                self.write_locks(root, locks)
                self.assertIn("methods.lock.json:compact_runtime_evidence_gap", assess(ROOT, root)["blockers"])

    def test_dp_schedule_requires_each_epsilon_for_all_fit_seeds(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            locks = self.fixture(root)
            method = locks["methods.lock.json"]["methods"]["author_default_fixture"]
            method.update({"group": "dp", "default_config": {"epsilon": 1},
                           "tuning_search_space": {"epsilon": [1, 4, 10]}})
            cells = locks["method-dataset-matrix.lock.json"]["cells"]
            for cell in cells:
                cell["dp_epsilon"] = 1
            self.write_locks(root, locks)
            self.assertIn("method-dataset-matrix.lock.json:fit_schedule_gap", assess(ROOT, root)["blockers"])
            for epsilon in (4, 10):
                added = copy.deepcopy(cells[:5])
                for cell in added:
                    cell["dp_epsilon"] = epsilon
                    cell["id"] += f"-epsilon-{epsilon}"
                cells.extend(added)
            self.write_locks(root, locks)
            self.assertTrue(assess(ROOT, root)["admitted"])


if __name__ == "__main__":
    unittest.main()
