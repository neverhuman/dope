"""Measured closure fixtures: private state receives no unverified sharing credit."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from research.transfer.bundle import (
    BundleError, U64_MAX, account_inventory, recipient_accounting, verify_bundle,
)
from research.transfer.evidence import canonical_json
from research.transfer.tests.test_bundle import GOLDEN, stage

ROOT = Path(__file__).resolve().parents[3]


class RecipientClosureTests(unittest.TestCase):
    def test_shared_cold_amortized_goldens_equal_physical_file_ledger(self):
        goldens = json.loads((ROOT / "rust/bundle/recipient-goldens.json").read_bytes())
        self.assertEqual(hashlib.sha256((ROOT / "schemas/bundle-golden-v1.json").read_bytes()).hexdigest(),
                         goldens["legacy_fixture_sha256"])
        for case in GOLDEN["cases"]:
            directory, root, digest = stage(case)
            with directory, self.subTest(case=case["name"]):
                report = recipient_accounting(verify_bundle(root, digest))
                self.assertEqual(report, goldens["cases"][case["name"]])
                physical = sum(path.stat().st_size for path in root.iterdir())
                self.assertEqual(report["physical_file_bytes"], physical)
                self.assertEqual(sum(report["section_bytes"].values()) +
                                 report["manifest_framing_bytes"], physical)
                self.assertEqual(report["cold_closure_bytes"] +
                                 report["nonrecipient_delivery_bytes"], physical)
                self.assertEqual(report["amortized_k_closure_bytes"], report["cold_closure_bytes"])
                self.assertIsNone(report["shared_source_independent_decoder_bytes"])

    def test_hidden_projection_prior_and_checkpoint_are_rejected_without_path_echo(self):
        for name in ("projection.json", "private-prior.bin", "checkpoint.bin"):
            directory, root, digest = stage(GOLDEN["cases"][0])
            with directory, self.subTest(name=name):
                (root / name).write_bytes(b"PRIVATE_HEADER_97531,99999")
                with self.assertRaises(BundleError) as caught:
                    verify_bundle(root, digest)
                self.assertEqual(str(caught.exception), "undeclared_component")
                self.assertNotIn(name, str(caught.exception))
                self.assertNotIn(str(root), str(caught.exception))

    def test_declared_fitted_state_and_runtime_are_fully_paid_at_large_k(self):
        names = {"dictionary.bin": "codebook.bin", "transform.bin": "projection.json",
                 "shared.bin": "private-prior.bin", "runtime.bin": "checkpoint.bin",
                 "payload.bin": "q4.bin"}
        case = copy.deepcopy(GOLDEN["cases"][0])
        case["manifest"]["amortization_count"] = 1000000
        for component in case["manifest"]["components"]:
            original = component["path"]
            component["path"] = names.get(original, original)
        case["files_hex"] = {names.get(name, name): data for name, data in case["files_hex"].items()}
        directory, root, digest = stage(case)
        with directory:
            bundle = verify_bundle(root, digest)
            report = recipient_accounting(bundle)
            raw = canonical_json(case["manifest"])
            # Payload 4 + private prior 10 + transform 2 + codebook 3 + runtime 5.
            self.assertEqual(report["cold_closure_bytes"], len(raw) + 24)
            self.assertEqual(report["dataset_specific_bytes"], len(raw) + 24)
            self.assertEqual(report["amortized_k_closure_bytes"], len(raw) + 24)
            self.assertEqual(report["section_bytes"]["sampling_payload"], 4)
            self.assertEqual(report["declared_runtime_bytes"], 5)
            serialized = json.dumps(report)
            for name in names.values():
                self.assertNotIn(name, serialized)
            self.assertNotIn(str(root), serialized)

    def test_required_rows_and_auditor_weights_only_enter_when_required_or_reachable(self):
        for role_id in ("rows", "evidence"):
            for mode in ("optional", "required", "reachable"):
                def mutate(manifest):
                    component = next(c for c in manifest["components"] if c["id"] == role_id)
                    component["required"] = mode == "required"
                    if mode in ("required", "reachable"):
                        payload = next(c for c in manifest["components"] if c["id"] == "payload")
                        payload["depends_on"] = sorted(payload["depends_on"] + [role_id])
                directory, root, digest = stage(GOLDEN["cases"][0], mutate)
                with directory, self.subTest(role=role_id, mode=mode):
                    report = recipient_accounting(verify_bundle(root, digest))
                    required = 0 if mode == "optional" else (9 if role_id == "rows" else 7)
                    self.assertEqual(report["cold_closure_bytes"], (root / "manifest.json").stat().st_size + 24 + required)
                    self.assertEqual(report["generated_samples_bytes"], 9)

    def test_digest_and_path_cannot_substitute_for_required_bytes(self):
        for mode in ("missing", "changed", "empty"):
            directory, root, digest = stage(GOLDEN["cases"][0])
            with directory, self.subTest(mode=mode):
                path = root / "shared.bin"
                if mode == "missing":
                    path.unlink()
                else:
                    path.write_bytes(b"" if mode == "empty" else b"wrong-data")
                with self.assertRaises(BundleError):
                    verify_bundle(root, digest)

    def test_unverified_input_and_overflow_fail_closed(self):
        with self.assertRaisesRegex(BundleError, "^verification_required$"):
            recipient_accounting({"charged_sampling_bytes": 0})
        component = {"measured_bytes": U64_MAX, "role": "sampling_payload"}
        with self.assertRaisesRegex(BundleError, "^integer_overflow$"):
            account_inventory([component], 1, 1, "unmeasured")


if __name__ == "__main__":
    unittest.main()
