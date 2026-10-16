"""Deterministic TDL conformance, accounting attacks, and standalone replay."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bench.metrics.qualification import require_citable
from research.transfer.tdl import aggregate, charge, drivers, frontier, transfer
from research.transfer.tdl.access import AccessTrace
from research.transfer.tdl.common import MetricError
from research.transfer.tdl.generators import DopeGenerator, LiteralGenerator, SpecifiedGenerator
from research.transfer.tdl.golden import close, strict_load

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/tdl"


def workspace():
    (ROOT / "target").mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="tdl-")


def transfer_input(**changes):
    data = dict(null_losses=[10., 10.], real_losses=[2., 2.], synthetic_losses=[2.8, 2.8],
                real_gain_lcb=7., constant_target=False)
    data.update(changes)
    return data


def driver_input(**changes):
    data = dict(real_importances=[8., 2., 0.], synthetic_importances=[8., 2., 0.],
                real_signs=[1, -1, 0], synthetic_signs=[1, -1, 0],
                real_interactions=[4., 1.], synthetic_interactions=[4., 1.],
                real_real_ceiling={name: .9 for name in ("mass", "rank", "sign", "interaction")})
    data.update(changes)
    return data


def aggregate_input():
    rows = []
    for source in ("s1", "s2"):
        for dataset in ("d1", "d2"):
            for seed in (1, 2, 3):
                for family, base in (("tree", .9), ("linear", .8)):
                    rows.append(dict(source_family=source, dataset=dataset, seed=seed,
                                     family=family, auditor=family, transfer=base + (seed - 2) * .1))
    return dict(rows=rows, required_families=["tree", "linear"],
                required_auditors={"tree": ["tree"], "linear": ["linear"]},
                required_seeds=[1, 2, 3], bootstrap_seed=1729,
                bootstrap_replicates=199, alpha=.05)


def frontier_input():
    def cell(size, transfer_, driver=.9):
        return dict(artifact_bytes=size, status="complete", families=["tree", "linear"],
                    gates={"closure": True, "privacy": True}, lcb_transfer=transfer_, driver_fidelity=driver)
    return dict(required_families=["tree", "linear"], required_gates=["closure", "privacy"],
                byte_grid=[1024, 2048, 4096],
                cells=[cell(1024, .85), cell(2048, .95), cell(4096, .90)])


class TransferTests(unittest.TestCase):
    def test_original_units_unclipped_ratio_and_informative_gate(self):
        self.assertAlmostEqual(transfer.evaluate(transfer_input())["raw_transfer"], .9)
        self.assertEqual(transfer.evaluate(transfer_input(synthetic_losses=[0., 0.]))["raw_transfer"], 1.25)
        self.assertEqual(transfer.evaluate(transfer_input(synthetic_losses=[18., 18.]))["raw_transfer"], -1.)
        for changes in ({"constant_target": True}, {"real_gain_lcb": 0.}, {"real_losses": [10., 10.]},
                        {"absolute_gain_floor": 9.}):
            self.assertEqual(transfer.evaluate(transfer_input(**changes))["status"], "uninformative")
        self.assertEqual(transfer.evaluate(transfer_input(real_gain_lcb=None))["status"], "incomplete")

    def test_nonfinite_overflow_boolean_and_unpaired_losses_refuse(self):
        for value in (float("nan"), float("inf"), "NaN", "Infinity", "1e309", True, 10**1000):
            result = transfer.evaluate(transfer_input(synthetic_losses=[value, 0.]))
            self.assertEqual(result["status"], "failed")
            self.assertNotIn(str(value), result["reason"])
        self.assertEqual(transfer.evaluate(transfer_input(real_losses=[2.]))["reason"], "unpaired_losses")
        # Finite individual terms can still overflow a ratio.
        result = transfer.evaluate(transfer_input(null_losses=[1e-308], real_losses=[0.],
                                  synthetic_losses=[1e308], real_gain_lcb=1e-308,
                                  absolute_gain_floor=0., relative_gain_floor=0.))
        self.assertEqual(result["status"], "failed")


class AggregateTests(unittest.TestCase):
    def test_paired_cluster_bootstrap_is_order_invariant_and_conservative(self):
        data = aggregate_input()
        report = aggregate.evaluate(data)
        self.assertEqual(report["status"], "complete")
        self.assertAlmostEqual(report["robust_transfer"], .8)
        self.assertAlmostEqual(report["mean_transfer"], .85)
        self.assertLess(report["lcb"], report["robust_transfer"])
        data["rows"].reverse()
        self.assertEqual(report, aggregate.evaluate(data))
        data["rows"] = [r for r in data["rows"] if r["family"] != "linear"]
        self.assertEqual(aggregate.evaluate(data)["status"], "incomplete")

    def test_extra_auditors_do_not_reweight_families_and_worse_family_cannot_raise_score(self):
        data = aggregate_input()
        original = aggregate.evaluate(data)
        data["rows"] += [dict(row, auditor="tree2") for row in data["rows"] if row["family"] == "tree"]
        data["required_auditors"]["tree"].append("tree2")
        self.assertTrue(close(aggregate.evaluate(data), original))
        data["required_families"].append("weak")
        data["required_auditors"]["weak"] = ["weak"]
        data["rows"] += [dict(row, family="weak", auditor="weak", transfer=-.2)
                         for row in data["rows"] if row["family"] == "linear"]
        self.assertLessEqual(aggregate.evaluate(data)["robust_transfer"], original["robust_transfer"])

    def test_missing_block_and_duplicate_observation_refuse(self):
        data = aggregate_input()
        data["rows"].pop()
        self.assertEqual(aggregate.evaluate(data)["status"], "incomplete")
        data = aggregate_input()
        data["rows"].append(data["rows"][0])
        self.assertEqual(aggregate.evaluate(data)["reason"], "duplicate_observation")


class DriverTests(unittest.TestCase):
    def test_grouped_correlated_features_and_ceiling_normalization(self):
        # Two mutually substitutable features are one evaluator-defined group.
        self.assertEqual(drivers.evaluate(driver_input())["driver_fidelity"], 1.)
        self.assertEqual(drivers.rank_agreement([1., 1., 0.], [1., 2., 0.]), 5 / 6)
        self.assertEqual(drivers.evaluate(driver_input(real_real_ceiling={"mass": .4}))["status"], "uninformative")

    def test_sign_reversal_interaction_strength_loss_and_label_shuffle_fail(self):
        for changes in ({"synthetic_signs": [-1, 1, 0]}, {"synthetic_interactions": [0., 0.]},
                        {"synthetic_interactions": [.04, .01]}, {"synthetic_importances": [0., 0., 0.]}):
            self.assertFalse(drivers.evaluate(driver_input(**changes))["driver_pass"])
        self.assertEqual(drivers.evaluate(driver_input(synthetic_signs=None))["status"], "incomplete")


class FrontierTests(unittest.TestCase):
    def test_smallest_tested_and_monotone_frontier(self):
        data = frontier_input()
        report = frontier.evaluate(data)
        self.assertEqual(report["status"], "qualified")
        self.assertEqual(report["tdl_bytes"], 2048)
        self.assertEqual([r["frontier_transfer"] for r in report["frontier"]], [.85, .95, .95])
        for cell in data["cells"]:
            cell["driver_fidelity"] = .1
        self.assertEqual(frontier.evaluate(data)["status"], "not_reached_in_tested_range")
        self.assertIsNone(frontier.evaluate(data)["tdl_bytes"])

    def test_missing_family_gate_or_failed_cell_propagates(self):
        for field, value in (("families", ["tree"]), ("gates", {"closure": True}),
                             ("lcb_transfer", None), ("status", "incomplete"),
                             ("status", "unqualified"), ("status", "failed")):
            data = frontier_input()
            data["cells"][0][field] = value
            self.assertIn(frontier.evaluate(data)["status"], {"incomplete", "unqualified", "failed"})


class ChargeAndTraceTests(unittest.TestCase):
    def test_full_framing_and_closure_charged_despite_denominator_attacks(self):
        data = dict(sections=[{"name": "z", "hex": "010203"}, {"name": "a", "hex": "04"}],
                    closure_files=[{"name": "runtime", "hex": "00" * 17, "scope": "runtime", "dataset_specific": False}])
        report = charge.evaluate(data)
        self.assertEqual(report["payload_bytes"], 8 + 2 * 12 + 2 + 4)
        self.assertEqual(report["sampling_closure_bytes"], report["payload_bytes"] + 17)
        changed = dict(data, encoded_numbers=0, real_rows=10**100, packed_scalars=1)
        self.assertEqual(report, charge.evaluate(changed))
        changed["sections"] = list(reversed(data["sections"]))
        self.assertEqual(charge.canonical_container(data["sections"]), charge.canonical_container(changed["sections"]))
        changed = copy.deepcopy(data)
        changed["closure_files"][0]["dataset_specific"] = True
        self.assertEqual(charge.evaluate(changed)["reason"], "undeclared_or_dataset_specific_closure")

    def test_hidden_sidecar_checkpoint_and_symlink_are_rejected_without_paths(self):
        with workspace() as directory:
            root = Path(directory)
            artifact, hidden = root / "artifact", root / "private-checkpoint"
            artifact.write_bytes(b"safe")
            hidden.write_bytes(b"private-header-extremum")
            trace = AccessTrace({artifact: "payload"})
            with trace.activate():
                self.assertEqual(artifact.read_bytes(), b"safe")
                with self.assertRaisesRegex(MetricError, "sampler_uncharged_read"):
                    hidden.read_bytes()
            report = json.dumps(trace.receipt())
            self.assertNotIn(directory, report)
            self.assertNotIn("private", report)
            self.assertEqual(trace.receipt()["status"], "failed")
            artifact.unlink()
            artifact.symlink_to(hidden)
            with trace.activate():
                with self.assertRaises(MetricError):
                    artifact.read_bytes()

    def test_guard_detects_descriptor_reads_and_process_escape(self):
        trace = AccessTrace({})
        with trace.activate():
            with self.assertRaises(MetricError):
                open(0)
            with self.assertRaisesRegex(MetricError, "sampler_external_access"):
                subprocess.run([sys.executable, "-c", "pass"])

    def test_fixture_admission_and_strict_json(self):
        for value in ({"citable": False}, {"citable": True, "evidence_status": "specified_unrun"}, {}):
            with self.assertRaises(MetricError):
                require_citable(value)
        for raw in ('{"x": NaN}', '{"x":1,"x":2}'):
            with self.assertRaises(MetricError):
                strict_load(raw)
        self.assertFalse(close(True, 1))
        self.assertFalse(close(float("inf"), 1.))
        self.assertFalse(close({"x": 1}, {"x": 1, "y": None}))


class StandaloneGeneratorTests(unittest.TestCase):
    def test_literal_controls_export_reload_in_fresh_process_with_access_trace(self):
        for method in ("copied_rows", "gzip_original"):
            with self.subTest(method=method), workspace() as directory:
                root = Path(directory)
                port = LiteralGenerator(method)
                port.fit(FIXTURES / "toy.csv", FIXTURES / "schema.json", 1)
                artifact, output = root / "payload", root / "sample.csv"
                inventory = port.export(artifact)
                self.assertEqual(inventory["payload_bytes"], artifact.stat().st_size)
                reference = charge.evaluate({"sections": [{"name": "literal_payload", "hex": port._payload.hex()}],
                                             "closure_files": []})
                self.assertEqual(inventory["payload_bytes"], reference["payload_bytes"])
                self.assertEqual(inventory["artifact_sha256"], reference["artifact_sha256"])
                self.assertEqual(LiteralGenerator.reload(artifact, method=method).sample(12, 2)[0],
                                 LiteralGenerator.reload(artifact, method=method).sample(12, 2)[0])
                result = subprocess.run([sys.executable, "-B", "-m", "research.transfer.tdl.generators",
                                         "--method", method, "--artifact", str(artifact), "--rows", "12",
                                         "--seed", "2", "--output", str(output)], capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(result.returncode, 0, result.stdout)
                receipt = json.loads(result.stdout)
                self.assertEqual(receipt["payload_bytes"], artifact.stat().st_size)
                trace = receipt["access_trace"]
                self.assertEqual(trace["status"], "passed")
                self.assertEqual(trace["coverage"], "all_threads_and_native_file_opens")
                self.assertIn({"operation": "read", "inventory_id": "payload"}, trace["events"])
                self.assertGreater(receipt["shared_bytes"], 0)
                self.assertEqual(receipt["sampling_closure_bytes"], receipt["payload_bytes"] + receipt["shared_bytes"])
                self.assertEqual(trace["byte_accounting"]["artifact_sha256"], inventory["artifact_sha256"])
                self.assertNotIn(str(root), result.stdout)
                self.assertEqual(len(output.read_text().splitlines()), 12)

    def test_dope_public_toy_fit_and_fresh_isolated_replay(self):
        binary = ROOT / "target/debug/dope-kernel"
        if not binary.is_file():
            subprocess.run(["cargo", "build", "--locked", "--bin", "dope-kernel"], cwd=ROOT, check=True)
        with workspace() as directory:
            root = Path(directory)
            port = DopeGenerator(binary, root / "fit")
            port.fit(FIXTURES / "toy.csv", FIXTURES / "schema.json", 1)
            artifact = root / "payload.raw"
            inventory = port.export(artifact)
            sampled, receipt = DopeGenerator.reload(artifact, binary=binary, workdir=root / "reload").sample(12, 2)
            self.assertEqual(len(sampled), 12)
            self.assertEqual(receipt["payload_bytes"], inventory["payload_bytes"])
            self.assertGreater(receipt["sampling_closure_bytes"], inventory["payload_bytes"])
            self.assertEqual(receipt["access_trace"]["filesystem_network_exec"], "denied")
            result = subprocess.run([sys.executable, "-B", "-m", "research.transfer.tdl.generators", "--method", "dope",
                                     "--artifact", str(artifact), "--binary", str(binary), "--rows", "12", "--seed", "2"],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(json.loads(result.stdout)["sample_sha256"], receipt["sample_sha256"])
            self.assertNotIn(str(root), result.stdout)
            with self.assertRaisesRegex(MetricError, "native_generator_failed"):
                port.fit(root / "missing.csv", FIXTURES / "schema.json", 1)
            with self.assertRaisesRegex(MetricError, "generator_not_loaded"):
                port.sample(12, 2)
            with self.assertRaisesRegex(MetricError, "generator_not_fitted"):
                port.export(root / "stale.raw")

    def test_planned_stubs_cannot_run(self):
        for method in ("gaussian_copula", "arf"):
            with self.assertRaisesRegex(MetricError, "generator_specified_unrun"):
                SpecifiedGenerator(method).fit()


class GoldenCorpusTests(unittest.TestCase):
    def test_every_shared_case_matches_reference_and_is_noncitable(self):
        from research.transfer.tdl.golden import cases, run_case
        paths = cases()
        self.assertGreaterEqual(len(paths), 40)
        seen = set()
        for path in paths:
            case = strict_load(path.read_text())
            with self.subTest(case=case["case_id"]):
                self.assertNotIn(case["case_id"], seen)
                seen.add(case["case_id"])
                self.assertTrue(close(run_case(case), case["expected"]))
                with self.assertRaisesRegex(MetricError, "noncitable_fixture"):
                    require_citable(case)

    def test_manifest_disallows_citation_and_json_overflow_cannot_enter(self):
        manifest = strict_load((FIXTURES / "manifest.json").read_text())
        with self.assertRaisesRegex(MetricError, "noncitable_fixture"):
            require_citable(manifest)
        with self.assertRaises(MetricError):
            strict_load('{"value":1e309}')
