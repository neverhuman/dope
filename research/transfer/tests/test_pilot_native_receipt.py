"""Authored fixture regressions for the actual Rust receipt transport; no SDK fit."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from research.transfer.campaign_artifacts import create_private, file_hash
from research.transfer.evidence import canonical_json, parse_json
from research.transfer.pilot_learning_contract import ARMS, CONTEXTS, LEARNERS, SCALES, identity
from research.transfer.pilot_native_receipt import NativeReceiptBridge, BASIS_KINDS, seed, wire


@dataclass
class Query:
    row_ids: tuple = ("query-a", "query-b")
    rows: tuple = ((1.0,), (2.0,))


@dataclass
class Context:
    name: str
    rows: int = 2


@dataclass
class Pool:
    requested_rows: int
    produced_rows: int
    draw_seed: int = 13
    sha256: str = "b" * 64


@dataclass
class Model:
    sha256: str = "c" * 64
    weights_sha256: str | None = None
    mode: str = "scratch"
    precision: str = "float64"
    determinism: str = "deterministic"


@dataclass
class Recipe:
    estimators: int = 1
    sha256: str = "d" * 64


class NativeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[3]
        parent = self.repo / "target/native-receipt-fixtures"
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        built = self.repo / "target/debug/dope-kernel"
        self.assertTrue(built.is_file(), "Run the planned Cargo native CLI test first")
        # Cargo hard-links its output into deps/; snapshot it as production does.
        self.binary = self.root / "dope-kernel"
        create_private(self.binary, built.read_bytes(), mode=0o700)
        self.paths = {k: self.root / k for k in ("learners", "drivers", "artifact", "schema", "csv")}
        self.paths["binary"] = self.binary
        for key in ("artifact", "schema", "csv"):
            create_private(self.paths[key], ("fixture-" + key).encode())
        documents = {}
        for kind in BASIS_KINDS:
            raw = ((self.repo / "production/behavior-contract-v1.json").read_bytes() if kind == "protocol"
                   else canonical_json({"fixture_kind": kind}))
            path = self.root / (kind + ".json")
            create_private(path, raw)
            documents[kind] = {"path": str(path), "sha256": file_hash(path)}
        basis = {"format": "dope-pilot-native-receipt-basis-v1", "bindings": {},
                 "global_seed": 7, "documents": documents}
        self.basis_path = self.root / "basis.json"
        create_private(self.basis_path, canonical_json(basis))
        self.bridge = NativeReceiptBridge(self.basis_path, file_hash(self.basis_path),
            file_hash(self.binary, 256 << 20), file_hash(Path(__file__)), issue_policy=self.fixture_policy)
        self.doc = {"cell_id": "a" * 64, "fit_seed": 11, "control_seed": 19, "comparator": "dope", "fit_rows": 3,
                    "csv_sha256": file_hash(self.paths["csv"]), "stage_wall_ms": {"receipt": 60000}}
        self.cell = NS(document=lambda: self.doc)
        self.graph = {"fixture": "actual frozen operation graph", "nodes": []}
        outcome = {"event": "outcome", "status": "completed", "wall_ns": 1000000,
                   "process_cpu_ns": 1000000, "reaped_children_cpu_seconds": 0.0,
                   "process_lifetime_peak_rss_bytes": 4096}
        physical = {"settled": True, "completed": True, "violation": False, "over_budget": False,
                    "graph": self.graph, "outcomes": [outcome]}
        self.journal = NS(state={(self.doc["cell_id"], "learners"): {"physical": physical},
                                (self.doc["cell_id"], "receipt"): {"pending": True}},
                          records=[{"fixture": "settled prefix"}],
                          spec=NS(document=lambda: {"budget": {"peak_ram_bytes": 1 << 30}}))
        pools = tuple((s, "synthetic", Pool(4 if s == SCALES[0] else 16, 4 if s == SCALES[0] else 16)) for s in SCALES)
        plan = NS(query=Query(), contexts=tuple(Context(c) for c in CONTEXTS), pools=pools, fit_seed=17)
        registry = {l: NS(model=Model(), recipe=Recipe(), sha256="e" * 64) for l in LEARNERS}
        comparisons, cases, links = [], [], []
        for scale in SCALES:
            for learner in LEARNERS:
                for context in CONTEXTS:
                    coordinate = {"scale": scale, "learner": learner, "context": context}
                    comparisons.append({**coordinate, "losses": {"real": [0.0, 1.0],
                        "synthetic": [1.0, 2.0], "null": [2.0, 3.0]}, "inference": None})
                    for arm in ARMS:
                        cases.append({**coordinate, "arm": arm, "status": "completed", "pool_sha256": "b" * 64,
                            "model_sha256": "c" * 64, "recipe_sha256": "d" * 64,
                            "consumed_rows": 2, "query_row_ids": plan.query.row_ids})
                        links.append({**coordinate, "arm": arm, "status": "completed"})
        private = {"cases": wire(cases), "physical_operations": [outcome]}
        self.execution = NS(plan=plan, cases=cases, registry=registry, private_result=lambda: private,
            evidence=NS(records=[outcome], guard=NS(graph=NS(document=lambda: self.graph))))
        create_private(self.paths["learners"], canonical_json({"execution": private, "paired": {
            "query_row_ids": list(plan.query.row_ids), "labels_sha256": "9" * 64, "comparisons": comparisons}}))
        create_private(self.paths["drivers"], canonical_json({"links": links}))

    @staticmethod
    def fixture_policy(path, bundle_sha256):
        # Explicit test-only issuer. Production must verify independent owner pins.
        bundle = parse_json(path.read_bytes())
        policy = {"format": "dope-pilot-native-receipt-policy-v1", "bundle_sha256": bundle_sha256,
            "policies": [{"identity": r["identity"], "executor_hash": r["executor_hash"],
                          "receipt_hash": identity(r), "freeze_hash": None} for r in bundle["receipts"]]}
        raw = canonical_json(policy)
        return raw, hashlib.sha256(raw).hexdigest()

    def verify(self):
        return self.bridge.verify(self.execution, self.journal, self.cell, self.paths, self.root)

    def test_actual_mapping_reaches_real_native_verifier_and_keeps_unqualified_scope(self):
        result = self.verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(len(result["receipts"]), 20)
        self.assertFalse(result["scientific_admission"])
        self.assertEqual(self.bridge.processes, 1)
        self.assertEqual(self.execution.evidence.records[0]["wall_ns"], 1000000)
        with self.assertRaisesRegex(ValueError, "replay"):
            self.verify()

    def test_nonzero_missing_and_tampered_native_output_refuse(self):
        with patch("research.transfer.pilot_native_receipt.invoke", return_value={"status": "failed", "returncode": 1}):
            with self.assertRaisesRegex(ValueError, "verifier_failed"):
                self.verify()
        self.assertFalse((self.root / "stage-receipt.json").exists())

    def test_unsettled_cost_and_changed_learner_bytes_refuse_before_native(self):
        self.journal.state[(self.doc["cell_id"], "learners")]["physical"]["settled"] = False
        with self.assertRaisesRegex(ValueError, "unsettled"):
            self.verify()
        self.assertEqual(self.bridge.processes, 0)

    def test_missing_policy_pin_and_wrong_binary_refuse(self):
        with self.assertRaisesRegex(ValueError, "trusted_policy"):
            NativeReceiptBridge(self.basis_path, file_hash(self.basis_path), "0" * 64, "f" * 64, issue_policy=None)
        self.bridge.binary_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "binary_changed"):
            self.verify()

    def test_control_generator_seed_is_distinct_from_compiler_seed(self):
        self.doc["comparator"] = "gaussian_copula"
        bundle = self.bridge.prepare(self.execution, self.journal, self.cell, self.paths,
                                     self.root / "control-proposal")
        self.assertTrue(all(r["identity"]["generator_fit_seed"] == 19 for r in bundle["receipts"]))

    def test_missing_native_output_refuses(self):
        with patch("research.transfer.pilot_native_receipt.invoke", return_value={"status": "completed", "returncode": 0}):
            with self.assertRaises(ValueError):
                self.verify()

    def test_changed_issuer_code_pin_refuses_before_native(self):
        self.bridge.issuer_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "issuer_code_changed"):
            self.verify()
        self.assertEqual(self.bridge.processes, 0)

    def test_uint64_is_refused_not_relabelled_under_immutable_v1(self):
        self.assertEqual(seed("0x000d0beef2026105"), int("000d0beef2026105", 16))
        with self.assertRaisesRegex(ValueError, "v1_seed_out_of_range"):
            seed("0xffffffffffffffff")

    def test_tampered_or_missing_native_output_refuses(self):
        with patch("research.transfer.pilot_native_receipt.invoke", return_value={"status": "completed", "returncode": 0}), \
             patch("research.transfer.pilot_native_receipt.output_document", return_value={"status": "verified"}):
            with self.assertRaisesRegex(ValueError, "output_mismatch"):
                self.verify()

    def test_changed_private_learner_evidence_refuses(self):
        data = parse_json(self.paths["learners"].read_bytes())
        data["execution"]["cases"][0]["status"] = "failed"
        self.paths["learners"].write_bytes(canonical_json(data))
        with self.assertRaisesRegex(ValueError, "input_changed"):
            self.verify()
        self.assertEqual(self.bridge.processes, 0)

    def test_wrong_policy_executor_reaches_real_rust_refusal(self):
        def wrong(path, key):
            raw, _ = self.fixture_policy(path, key)
            p = parse_json(raw)
            p["policies"][0]["executor_hash"] = "0" * 64
            raw = canonical_json(p)
            return raw, hashlib.sha256(raw).hexdigest()
        self.bridge.issue_policy = wrong
        with self.assertRaisesRegex(ValueError, "verifier_failed"):
            self.verify()


class ParentStageRefusalTests(unittest.TestCase):
    def test_native_failure_never_creates_receipt_or_report(self):
        from research.transfer.pilot_worker_runtime import PilotWorkerRuntime
        parent = Path(__file__).resolve().parents[3] / "target/native-receipt-fixtures"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            root = Path(temporary)
            paths = {k: root / k for k in ("artifact", "csv", "schema", "selection_csv", "query_csv", "receipt", "report")}
            for key in ("artifact", "csv", "schema", "selection_csv", "query_csv"):
                create_private(paths[key], ("fixture-" + key).encode())
            doc = {k + "_sha256": file_hash(paths[k]) for k in ("csv", "schema", "selection_csv", "query_csv")}
            # Deliberately bypass construction only to exercise the fail-closed
            # stage boundary with an injected verifier failure, without a model.
            worker = object.__new__(PilotWorkerRuntime)
            worker._bound, worker._capsule, worker._summaries = True, b"fixture", {}
            worker.capsule = lambda: b"fixture"
            worker._execution, worker.native_result, worker.kind = None, None, "development"
            def failed(*args):
                raise ValueError("native_receipt_verifier_failed")
            worker.native_receipt = NS(verify=failed, outcome=None)
            result = worker.stage("receipt", None, NS(document=lambda: doc), paths, root,
                                  file_hash(paths["artifact"]), None)
            self.assertEqual(result[0]["status"], "failed")
            self.assertIsNone(result[1])
            self.assertFalse(paths["receipt"].exists())
            self.assertFalse(paths["report"].exists())
