"""8-stage pilot runner bridge tests; synthetic fixtures never authorize release."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from research.transfer import pilot_runner
from research.transfer.campaign_admission import (
    ADMISSION_FORMAT, GATES, PLAN_FORMAT, PilotExecutionPlan,
)
from research.transfer.campaign_artifacts import create_private, directory_pin, file_hash
from research.transfer.campaign_runner import CampaignRunSpec, CampaignRunnerError, FORMAT, STAGES, StageJournal
from research.transfer.evidence import canonical_json, digest

GIB = 1 << 30


def _pilot_plan_doc():
    return {
        "format": PLAN_FORMAT,
        "version": 1,
        "operation": "development_pilot",
        "campaign_id": "8" * 64,
        "pins": {
            "source_commit": "1" * 40,
            "source_tree": "2" * 40,
            "environment_sha256": "3" * 64,
            "policy_sha256": "4" * 64,
            "schedule_sha256": "5" * 64,
            "operation_dag_sha256": "9" * 64,
        },
        "tasks": [
            {
                "task_id": "a" * 64,
                "role": "observational",
                "fit_rows": 64,
                "primary_n": 16,
                "input_sha256": "b" * 64,
            },
            {
                "task_id": "c" * 64,
                "role": "adverse_mechanism",
                "fit_rows": 64,
                "primary_n": 16,
                "input_sha256": "d" * 64,
            },
        ],
        "methods": list(pilot_runner.COMPARATORS),
        "byte_caps": [32768, 131072],
        "learners": list(pilot_runner.LEARNERS),
        "contexts": [
            {"context_id": "standard", "rows": 16},
            {"context_id": "stratified_anchor", "rows": 16},
        ],
        "seeds": {
            "encoding": "explicit-hex-uint64-v1",
            "master": "0x00000000000006c0",
            "substreams": {
                "fit": "0x00000000000006c1",
                "sample_primary": "0x00000000000006c2",
                "sample_4n": "0x00000000000006c5",
                "context": "0x00000000000006c3",
                "learner": "0x00000000000006c4",
            },
        },
        "stages": list(STAGES),
        "sample_policies": list(pilot_runner.SCALES),
        "budget": {
            "wall_ms": 80000,
            "cpu_ms": 80000,
            "peak_ram_bytes": 2 * GIB,
            "disk_bytes": 4 * GIB,
            "max_attempts": 1,
        },
    }


def _admission_contract_doc(pilot_plan: PilotExecutionPlan, *, gate_status="accepted_for_operation"):
    ev = "e" * 64 if gate_status == "accepted_for_operation" else None
    return {
        "format": ADMISSION_FORMAT,
        "version": 1,
        "operation": "development_pilot",
        "plan_sha256": pilot_plan.sha256,
        "pins": dict(pilot_plan.document()["pins"]),
        "gates": {
            gate: {"status": gate_status, "evidence_sha256": ev}
            for gate in GATES
        },
    }


def fixture(directory, stage_wall_ms=10000, csv_rows=64, binary_source=None, comparator="dope", candidate="chow_liu", version=1, byte_budget=32768):
    root = Path(directory).absolute()
    binary = root / "dope-kernel"
    worker = root / "pilot-worker"
    if binary_source is not None:
        create_private(binary, Path(binary_source).read_bytes(), 0o500)
    else:
        create_private(binary, b"unit-fixture-dope-kernel", 0o500)
    create_private(worker, b"unit-fixture-pilot-worker", 0o500)
    csv = root / "SECRET_fit.csv"
    sel_csv = root / "SECRET_selection.csv"
    qry_csv = root / "SECRET_query.csv"
    schema = root / "SECRET_schema.json"
    create_private(
        csv,
        ("SECRET_y,SECRET_x\n" + "".join(f"{100 + 200 * i / max(1, csv_rows - 1)},{10 + 20 * i / max(1, csv_rows - 1)}\n" for i in range(csv_rows))).encode(),
    )
    create_private(
        sel_csv,
        ("SECRET_y,SECRET_x\n" + "".join(f"{150 + i},{15 + i}\n" for i in range(16))).encode(),
    )
    create_private(
        qry_csv,
        ("SECRET_y,SECRET_x\n" + "".join(f"{250 + i},{25 + i}\n" for i in range(16))).encode(),
    )
    create_private(
        schema,
        canonical_json({
            "format": "dope-table-schema",
            "version": 1,
            "columns": [
                {"id": "c0001", "source_name": "SECRET_x", "role": "feature", "dtype": "continuous"},
                {"id": "c0002", "source_name": "SECRET_y", "role": "target", "dtype": "continuous"},
            ],
        }),
    )
    # Preserve the deliberate external target-first source, then use the shared
    # declared-target intake before constructing any canonical pilot binding.
    from research.transfer.target_last import normalize_csv
    original_schema = schema.read_bytes()
    for partition in (csv, sel_csv, qry_csv):
        normalized = normalize_csv(partition.read_bytes(), original_schema, target="SECRET_y")
        partition.write_bytes(normalized.csv)
    schema.write_bytes(normalized.schema)
    pilot_doc = _pilot_plan_doc()
    if byte_budget not in pilot_doc["byte_caps"]:
        pilot_doc["byte_caps"] = sorted([32768, byte_budget])
    if version == 2:
        pilot_doc["seeds"]["substreams"]["sample_primary"] = "0xfffffffffffffffe"
        pilot_doc["seeds"]["substreams"]["sample_4n"] = "0xffffffffffffffff"
    pilot_plan = PilotExecutionPlan.from_document(pilot_doc)
    admission_contract = _admission_contract_doc(pilot_plan)
    cell_doc = {
        "format": pilot_runner.FORMAT,
        "version": 1,
        "cell_id": "5" * 64,
        "task_id": "a" * 64,
        "task_role": "observational",
        "comparator": comparator,
        "candidate": candidate,
        "binary": str(binary),
        "binary_sha256": file_hash(binary, 256 << 20),
        "worker_binary": str(worker),
        "worker_binary_sha256": file_hash(worker, 256 << 20),
        "csv": str(csv),
        "csv_sha256": file_hash(csv),
        "selection_csv": str(sel_csv),
        "selection_csv_sha256": file_hash(sel_csv),
        "query_csv": str(qry_csv),
        "query_csv_sha256": file_hash(qry_csv),
        "schema": str(schema),
        "schema_sha256": file_hash(schema),
        "fit_seed": 0x6C1,
        "draw_seed": 0x6C2,
        "control_seed": 0x6C3,
        "driver_seed": 0x6C4,
        "fit_rows": 64,
        "selection_rows": 16,
        "query_rows": 16,
        "rows": 16,
        "byte_budget": byte_budget,
        "deadline_seconds": 1,
        "training_scales": list(pilot_runner.SCALES),
        "learners": list(pilot_runner.LEARNERS),
        "contexts": list(pilot_runner.CONTEXTS),
        "learner_paths": list(pilot_runner.LEARNER_PATHS),
        "learner_rights_sha256": {learner: "7" * 64 for learner in pilot_runner.LEARNERS},
        "stage_wall_ms": {stage: stage_wall_ms for stage in STAGES},
    }
    if version == 2:
        cell_doc["version"] = 2
        for name, stream in (("fit_seed", "fit"), ("draw_seed", "sample_primary"),
                             ("draw_4n_seed", "sample_4n"), ("control_seed", "context"),
                             ("driver_seed", "learner")):
            cell_doc[name] = pilot_doc["seeds"]["substreams"][stream]
    cell_plan = pilot_runner.PilotCellExecutionPlan.from_document(cell_doc)
    spec = CampaignRunSpec.from_document({
        "format": FORMAT,
        "version": 1,
        "phase": "development",
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "environment_sha256": "3" * 64,
        "protocol_sha256": "4" * 64,
        "roster_sha256": "9" * 64,
        "cells": [{
            "cell_id": cell_doc["cell_id"],
            "input_sha256": cell_plan.input_sha256,
            "command_sha256": cell_plan.sha256,
        }],
        "budget": {"wall_ms": stage_wall_ms * len(STAGES), "peak_ram_bytes": 2 * GIB},
        "max_attempts": 1,
    })
    evidence = root / "evidence"
    evidence.mkdir(mode=0o700)
    key, public = root / "test-private.pem", root / "test-public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(key)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(key), "-pubout", "-out", str(public)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    payload = {
        "scope": pilot_runner.SCOPE,
        "phase": "development",
        "scientific_admission": False,
        "run_spec_sha256": spec.sha256,
        "pilot_execution_plan_sha256": pilot_plan.sha256,
        "pilot_admission_contract_sha256": digest(admission_contract),
        "cell_execution_plan_sha256": cell_plan.sha256,
        "rights_admission_sha256": "a" * 64,
        "resource_lease_sha256": "b" * 64,
        "evidence_directory_sha256": directory_pin(evidence),
    }
    message, signature, admission = root / "message", root / "signature", root / "admission.json"
    create_private(message, canonical_json(payload))
    subprocess.run(
        ["openssl", "pkeyutl", "-sign", "-inkey", str(key), "-rawin", "-in", str(message), "-out", str(signature)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    create_private(admission, canonical_json({
        "payload": payload,
        "signature_ed25519_base64": base64.b64encode(signature.read_bytes()).decode(),
    }))
    return spec, pilot_plan, admission_contract, cell_plan, evidence, {
        "admission": admission,
        "public_key": public,
        "trusted_key_sha256": file_hash(public),
    }


def simulated_8stage_workers(cell_plan, observed_commands=None):
    doc = cell_plan.document()

    def call(argv, root, label, _wall, _ram):
        if observed_commands is not None:
            observed_commands.append((label, list(argv)))
        if label == "compile":
            create_private(root / "model.raw", b"unit-pilot-artifact")
        artifact_sha = file_hash(root / "model.raw", 256 << 20) if (root / "model.raw").exists() else None
        artifact_size = (root / "model.raw").stat().st_size if (root / "model.raw").exists() else 0
        payloads = {
            "doctor": {
                "name": "dope-kernel",
                "implementation": "native_rust",
                "gpu_training": False,
                "executable_sha256": doc["binary_sha256"],
            },
            "compile": {"unit_fixture": True, "comparator": doc["comparator"], "candidate": doc["candidate"]},
            "reload": {"comparator": doc["comparator"], "accounting": {"complete_sampling_bytes": artifact_size}},
            "sample": {
                "sample_sha256": "c" * 64,
                "filesystem_network_exec": "denied",
                "inherited_source_fds": "closed",
                "rows": doc["rows"],
                "seed": doc["draw_seed"],
            },
            "learners": {
                "comparator": doc["comparator"],
                "candidate": doc["candidate"],
                "training_scales": list(pilot_runner.SCALES),
                "learners": list(pilot_runner.LEARNERS),
                "contexts": list(pilot_runner.CONTEXTS),
                "learner_paths": list(pilot_runner.LEARNER_PATHS),
                "evaluation_count": 20,
                "artifact_sha256": artifact_sha,
                "selection_csv_sha256": doc["selection_csv_sha256"],
                "query_csv_sha256": doc["query_csv_sha256"],
                "paired_loss_digest": "d" * 64,
            },
            "drivers": {
                "path": "drivers-v1",
                "evidence_kind": "permutation_importance",
                "seed": doc["driver_seed"],
                "learners": list(pilot_runner.LEARNERS),
                "query_csv_sha256": doc["query_csv_sha256"],
                "driver_digest": "e" * 64,
            },
            "receipt": {
                "artifact_sha256": artifact_sha,
                "comparator": doc["comparator"],
                "byte_budget": doc["byte_budget"],
                "receipt_digest": "f" * 64,
            },
            "report": {
                "redacted": True,
                "sealed_holdout_opened": False,
                "scientific_qualification": "unmeasured",
                "release_authorization": "not_authorized",
                "report_digest": "1" * 64,
            },
        }
        if doc["version"] == 2 and label == "learners":
            payloads[label]["materialized_draw_manifest_sha256"] = file_hash(root / "materialized-draws/manifest.json")
        create_private(root / (label + ".stdout"), canonical_json(payloads[label]))
        create_private(root / (label + ".stderr"), b"")
        return {
            "status": "completed",
            "wall_ms": 2,
            "peak_ram_bytes": 1024,
            "returncode": 0,
            "stdout_sha256": file_hash(root / (label + ".stdout")),
            "stderr_sha256": file_hash(root / (label + ".stderr")),
        }

    return call


class PilotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.spec, self.pilot_plan, self.contract, self.cell_plan, self.root, self.auth = fixture(self.temporary.name)

    def execute(self):
        return pilot_runner.execute_pilot_cell(
            self.spec, self.pilot_plan, self.contract, self.cell_plan, self.root, **self.auth,
        )

    def test_full_8stage_pilot_execution_uses_exact_kernel_abi_and_separates_comparator_controls(self):
        commands = []
        with mock.patch.object(pilot_runner, "invoke", side_effect=simulated_8stage_workers(self.cell_plan, commands)):
            receipt = self.execute()
        self.assertEqual(receipt["status"], "ok")
        self.assertTrue(receipt["bridge_stages_complete"])
        self.assertFalse(receipt["pipeline_complete"])
        self.assertFalse(receipt["scientific_admission"])
        self.assertEqual(receipt["pipeline_prerequisites"], list(pilot_runner.PIPELINE_PREREQUISITES))
        self.assertEqual(commands[1][1][1], "compile-raw")
        self.assertIn("--candidate", commands[1][1])
        self.assertEqual(commands[2][1][1], "inspect-raw")
        self.assertEqual(commands[3][1][1], "sample-raw-isolated")
        self.assertNotIn("SECRET", json.dumps(receipt))
        learner_argv = next(argv for label, argv in commands if label == "learners")
        doc = self.cell_plan.document()
        for flag, name, expected in (("--fit-csv", "fit.csv", "csv_sha256"),
                                     ("--schema", "schema.json", "schema_sha256")):
            actual = Path(learner_argv[learner_argv.index(flag) + 1])
            self.assertEqual(actual, self.root / name)
            self.assertEqual(file_hash(actual), doc[expected])
            self.assertEqual(learner_argv[learner_argv.index(flag + "-sha256") + 1], doc[expected])
            self.assertEqual(actual.stat().st_mode & 0o777, 0o400)
        self.assertNotEqual(learner_argv[learner_argv.index("--fit-csv") + 1],
                            learner_argv[learner_argv.index("--query-csv") + 1])
        with tempfile.TemporaryDirectory() as copula_dir:
            spec, plan, contract, cell_plan, root, auth = fixture(
                copula_dir, comparator="gaussian_copula", candidate="gaussian_copula",
            )
            copula_cmds = []
            with mock.patch.object(pilot_runner, "invoke", side_effect=simulated_8stage_workers(cell_plan, copula_cmds)):
                copula_receipt = pilot_runner.execute_pilot_cell(spec, plan, contract, cell_plan, root, **auth)
            self.assertEqual(copula_receipt["status"], "ok")
            self.assertFalse(copula_receipt["pipeline_complete"])
            self.assertEqual(copula_cmds[1][1][1], "control-compile")
            self.assertEqual(copula_cmds[2][1][1], "control-reload")
            self.assertEqual(copula_cmds[3][1][1], "control-sample")

    def test_modified_private_fit_input_refuses_before_learner_execution(self):
        commands = []
        original = simulated_8stage_workers(self.cell_plan, commands)
        def tamper_after_sample(argv, root, label, wall, ram):
            result = original(argv, root, label, wall, ram)
            if label == "sample":
                path = root / "fit.csv"
                path.chmod(0o600)
                path.write_bytes(b"SECRET_SUBSTITUTED_SOURCE")
            return result
        with mock.patch.object(pilot_runner, "invoke", side_effect=tamper_after_sample):
            result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"][-1]["stage"], "learners")
        self.assertEqual(result["stages"][-1]["status"], "failed")
        self.assertFalse(any(label == "learners" for label, _ in commands))
        self.assertNotIn("SECRET", json.dumps(result))

    def test_real_dope_kernel_first_4_stages_abi_and_payload_verification(self):
        checkout = Path(__file__).resolve().parents[3]
        real_binary = checkout / "target/debug/dope-kernel"
        if not real_binary.is_file():
            subprocess.run(
                ["cargo", "build", "--locked", "--bin", "dope-kernel"], cwd=checkout, check=True,
            )
        self.assertTrue(real_binary.is_file(), "own-checkout dope-kernel build did not produce the binary")
        with tempfile.TemporaryDirectory() as real_dir:
            spec, plan, contract, cell_plan, root, auth = fixture(
                real_dir, stage_wall_ms=30000, binary_source=real_binary,
            )
            receipt = pilot_runner.execute_pilot_cell(spec, plan, contract, cell_plan, root, **auth)
            self.assertEqual(
                [r["stage"] for r in receipt["stages"][:4]],
                ["prepare", "compile", "reload", "sample"],
            )
            for row in receipt["stages"][:4]:
                self.assertEqual(row["status"], "completed")
            self.assertEqual(receipt["stages"][4]["stage"], "learners")
            self.assertEqual(receipt["stages"][4]["status"], "failed")
            self.assertFalse(receipt["pipeline_complete"])

    def test_native_command_preserves_signed_candidate_and_budgets(self):
        doc = self.cell_plan.document()
        doc["candidate"] = "independent"
        plan = pilot_runner.PilotCellExecutionPlan.from_document(doc)
        paths = {"binary": self.root / "dope-kernel", "artifact": self.root / "model.raw"}
        binary, args = pilot_runner._stage_command(paths, plan, "compile")
        self.assertEqual(binary, paths["binary"])
        self.assertEqual(args[0], "compile-raw")
        for flag, value in (
            ("--candidate", "independent"),
            ("--seed", str(doc["fit_seed"])),
            ("--max-complete-bytes", str(doc["byte_budget"])),
            ("--deadline-seconds", str(doc["deadline_seconds"])),
        ):
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_pending_rights_or_unaccepted_gate_refused_before_model_or_native_execution(self):
        blocked = _admission_contract_doc(self.pilot_plan, gate_status="missing")
        with mock.patch.object(pilot_runner, "invoke") as worker:
            with self.assertRaisesRegex(CampaignRunnerError, "pilot_rights_or_qualification_unavailable"):
                pilot_runner.execute_pilot_cell(
                    self.spec, self.pilot_plan, blocked, self.cell_plan, self.root, **self.auth,
                )
            worker.assert_not_called()

    def test_native_engineering_scope_is_rejected_for_pilot_execution(self):
        original = json.loads(self.auth["admission"].read_bytes())
        original["payload"]["scope"] = "native_engineering_stages_only"
        self.auth["admission"].write_bytes(canonical_json(original))
        with mock.patch.object(pilot_runner, "invoke") as worker:
            with self.assertRaises(CampaignRunnerError):
                self.execute()
            worker.assert_not_called()

    def test_insufficient_fit_rows_in_csv_is_refused_before_compile(self):
        with tempfile.TemporaryDirectory() as small_dir:
            spec, plan, contract, cell_plan, root, auth = fixture(small_dir, csv_rows=16)
            with mock.patch.object(pilot_runner, "invoke") as worker:
                receipt = pilot_runner.execute_pilot_cell(spec, plan, contract, cell_plan, root, **auth)
                worker.assert_not_called()
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["stages"][0]["reason"], "pilot_input_rows_insufficient")


class FrozenDriverPolicyTests(unittest.TestCase):
    def test_distinct_stream_requires_policy_and_preserves_legacy(self):
        import hashlib
        from research.transfer.pilot_cell_plan import _driver_seed_bound, FROZEN_DRIVER_POLICY_SHA256
        streams = {"learner": "0x000d0beef2026105"}
        master = "0x000d0beef2026100"
        derived = int.from_bytes(hashlib.sha256(
            b"dope-e03-development-driver-stream-v1\0" + int(master, 16).to_bytes(8, "little")
        ).digest()[:8], "little")
        seed = f"0x{derived:016x}"
        legacy = {"version": 2, "driver_seed": streams["learner"]}
        self.assertTrue(_driver_seed_bound(legacy, streams, master))
        self.assertNotEqual(seed, streams["learner"])
        self.assertFalse(_driver_seed_bound(legacy | {"driver_seed": seed}, streams, master))
        successor = legacy | {"driver_seed": seed, "driver_policy_sha256": FROZEN_DRIVER_POLICY_SHA256}
        self.assertTrue(_driver_seed_bound(successor, streams, master))
        self.assertFalse(_driver_seed_bound(successor | {"driver_policy_sha256": "0" * 64}, streams, master))
        self.assertFalse(_driver_seed_bound(successor | {"driver_seed": streams["learner"]}, streams, master))
        self.assertFalse(_driver_seed_bound(successor | {"version": 1}, streams, master))

    def test_policy_is_part_of_signed_cell_input_and_rejects_invalid_scope(self):
        from research.transfer.pilot_cell_plan import FROZEN_DRIVER_POLICY_SHA256
        with tempfile.TemporaryDirectory() as td:
            _, _, _, old, _, _ = fixture(td, version=2)
            doc = old.document()
            updated = pilot_runner.PilotCellExecutionPlan.from_document(
                doc | {"driver_policy_sha256": FROZEN_DRIVER_POLICY_SHA256})
            self.assertNotEqual(old.input_sha256, updated.input_sha256)
            self.assertNotEqual(old.sha256, updated.sha256)
            with self.assertRaisesRegex(CampaignRunnerError, "pilot_driver_policy_invalid"):
                pilot_runner.PilotCellExecutionPlan.from_document(doc | {"driver_policy_sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
