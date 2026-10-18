"""FOLLOW-L1a: unavailable measurements remain explicit in failed receipts."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from research.transfer.tdl.golden import run_case, strict_load

ROOT = Path(__file__).resolve().parents[3]
BYTE_FIELDS = ("payload_bytes", "section_bytes", "framing_bytes", "shared_bytes",
               "sampling_closure_bytes", "cold_bundle_bytes", "amortized_bytes", "artifact_sha256")
GROUP_FIELDS = {
    "byte_accounting": BYTE_FIELDS,
    "diagnostics": ("encoded_scalars", "encoded_symbols", "representation_closeness"),
    "auditor_reports": ("null_loss", "real_loss", "synthetic_loss", "real_gain", "raw_transfer", "clipped_transfer"),
    "driver_report": ("raw_components", "normalized_components", "driver_fidelity"),
    "aggregate": ("family_means", "robust_transfer", "mean_transfer", "lcb"),
}


def golden(status):
    return strict_load((ROOT / f"tests/golden/reports/follow_l1a_{status}.json").read_text())


def group(report, name):
    return report[name][0] if name == "auditor_reports" else report[name]


class ResultSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = {name: json.loads((ROOT / f"schemas/{name}-v1.schema.json").read_text())
                       for name in ("tdl-report", "byte-accounting")}
        for schema in cls.schemas.values():
            Draft202012Validator.check_schema(schema)
        cls.report = Draft202012Validator(cls.schemas["tdl-report"])
        cls.bytes = Draft202012Validator(cls.schemas["byte-accounting"])

    def test_two_golden_reports_and_their_byte_accounting_are_valid(self):
        for status in ("failed", "incomplete"):
            with self.subTest(status=status):
                report = golden(status)
                self.assertIs(report["citable"], False)
                self.report.validate(report)
                self.bytes.validate(dict(report["byte_accounting"], status=status))

    def test_failed_and_incomplete_measurement_fields_cannot_be_omitted(self):
        for status in ("failed", "incomplete"):
            for name, fields in GROUP_FIELDS.items():
                for field in fields:
                    with self.subTest(status=status, group=name, field=field):
                        report = golden(status)
                        del group(report, name)[field]
                        self.assertFalse(self.report.is_valid(report))
            for field in BYTE_FIELDS:
                with self.subTest(status=status, byte_field=field):
                    byte = dict(golden(status)["byte_accounting"], status=status)
                    del byte[field]
                    self.assertFalse(self.bytes.is_valid(byte))

    def test_null_measurements_require_a_nonempty_source_free_reason(self):
        for status in ("failed", "incomplete"):
            for name in GROUP_FIELDS:
                for reason in (None, "", " ", "private/header: 42"):
                    with self.subTest(status=status, group=name, reason=reason):
                        report = golden(status)
                        group(report, name)["reason"] = reason
                        self.assertFalse(self.report.is_valid(report))
                report = golden(status)
                del group(report, name)["reason"]
                self.assertFalse(self.report.is_valid(report), name)

    def test_complete_charge_goldens_still_validate(self):
        for path in (ROOT / "tests/golden/charge").glob("*.json"):
            case = strict_load(path.read_text())
            if case["expected"]["status"] == "complete":
                with self.subTest(case=case["case_id"]):
                    self.bytes.validate(case["expected"])

    def test_complete_accounting_cannot_replace_measurements_with_null(self):
        byte = strict_load((ROOT / "tests/golden/charge/canonical.json").read_text())["expected"]
        for field in BYTE_FIELDS:
            with self.subTest(field=field):
                changed = dict(byte, reason="missing_measurement")
                changed[field] = None
                self.assertFalse(self.bytes.is_valid(changed))

    def test_successful_metric_groups_cannot_use_failure_nulls(self):
        for name, status in (("auditor_reports", "informative"), ("driver_report", "complete"),
                             ("aggregate", "qualified")):
            with self.subTest(group=name):
                report = golden("failed")
                group(report, name)["status"] = status
                self.assertFalse(self.report.is_valid(report))

    def test_failed_accounting_can_retain_measured_numbers(self):
        measured = strict_load((ROOT / "tests/golden/charge/canonical.json").read_text())["expected"]
        for status in ("failed", "incomplete"):
            byte = dict(measured, status=status, reason="sample_failed")
            self.bytes.validate(byte)
            for field, value in (("payload_bytes", 0), ("payload_bytes", True), ("shared_bytes", -1),
                                 ("amortized_bytes", "unknown")):
                with self.subTest(status=status, field=field, value=value):
                    changed = dict(byte)
                    changed[field] = value
                    self.assertFalse(self.bytes.is_valid(changed))

    def test_failed_report_preserves_known_measurements(self):
        report = golden("incomplete")
        self.assertEqual(report["auditor_reports"][0]["null_loss"], 10.)
        self.assertEqual(report["auditor_reports"][0]["real_gain"], 8.)
        self.assertIsNone(report["auditor_reports"][0]["synthetic_loss"])
        self.report.validate(report)

    def test_standalone_failure_reason_is_required_even_with_known_numbers(self):
        measured = strict_load((ROOT / "tests/golden/charge/canonical.json").read_text())["expected"]
        for status in ("failed", "incomplete"):
            for reason in (None, "", " "):
                self.assertFalse(self.bytes.is_valid(dict(measured, status=status, reason=reason)))


class CandidateAdmissionTests(unittest.TestCase):
    """Independent schema/hash oracle, not a second scientific reducer."""

    @classmethod
    def setUpClass(cls):
        cls.protocol = strict_load((ROOT / "contracts/tdl-contract-v2.candidate.json").read_text())
        cls.result = cls.fixture("candidate-result.json")
        cls.schemas = {}
        for name in ("protocol", "result"):
            schema = strict_load((ROOT / f"schemas/tdl-{name}-v1.schema.json").read_text())
            Draft202012Validator.check_schema(schema)
            cls.schemas[name] = Draft202012Validator(schema)

    @staticmethod
    def fixture(name):
        return strict_load((ROOT / "tests/fixtures/tdl-v2" / name).read_text())

    @staticmethod
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    @classmethod
    def canonical(cls, value):
        def check(item):
            if type(item) is float:
                raise ValueError("candidate_noncanonical_number")
            if type(item) is int:
                cls.require(abs(item) <= (1 << 53) - 1, "nonfinite_or_invalid_number")
            if type(item) is list:
                for child in item:
                    check(child)
            if type(item) is dict:
                for child in item.values():
                    check(child)
        check(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False)

    @classmethod
    def identity(cls, kind, body):
        cls.require(kind in ("protocol", "task", "method", "panel", "artifact", "campaign",
                             "release_profile"), "candidate_identity_kind")
        encoded = cls.canonical(body).encode("utf-8")
        return kind + ":" + hashlib.sha256(b"dope.identity.v1\0" + kind.encode() + b"\0" + encoded).hexdigest()

    @classmethod
    def protocol_id(cls, protocol):
        return cls.identity("protocol", {k: v for k, v in protocol.items() if k != "protocol_id"})

    @classmethod
    def validate_protocol(cls, protocol):
        expected = cls.protocol_id(protocol)
        cls.require(cls.schemas["protocol"].is_valid(protocol), "candidate_schema_mismatch")
        parent = hashlib.sha256((ROOT / "contracts/tdl-contract-v1.json").read_bytes()).hexdigest()
        cls.require(protocol["parent_contract_sha256"] == parent, "candidate_parent_digest_mismatch")
        cls.require(protocol["protocol_id"] == expected, "candidate_protocol_digest_mismatch")
        panel = protocol["panel"]
        if panel is not None:
            models = panel["eligible_models"]
            cls.require(len({model["id"] for model in models}) == len(models) and
                        {model["family"] for model in models} == set(panel["required_families"]),
                        "candidate_inconsistent_panel")
            for role in ("development", "confirmation"):
                roster = protocol["roles"][role]
                if roster is not None:
                    cls.require(set(roster) == {model["id"] for model in models if model["role"] == role},
                                "candidate_inconsistent_panel")
        development, confirmation = (protocol["roles"][name] for name in ("development", "confirmation"))
        if development is not None and confirmation is not None:
            cls.require(not set(development) & set(confirmation), "candidate_inconsistent_panel")

    @classmethod
    def admit(cls, protocol, receipt, legacy_replay=False):
        if receipt.get("contract_id") == "tdl-contract-v1":
            cls.require(legacy_replay, "legacy_receipt_requires_replay")
            cls.require(set(receipt) == {"contract_id", "case"}, "invalid_legacy_replay")
            run_case(receipt["case"])
            return "LegacyReplay"
        cls.require(not legacy_replay, "candidate_not_legacy_receipt")
        cls.validate_protocol(protocol)
        cls.require(receipt.get("protocol_id") == protocol["protocol_id"], "candidate_receipt_protocol_mismatch")
        cls.canonical(receipt)
        cls.require(cls.schemas["result"].is_valid(receipt), "candidate_schema_mismatch")
        identities = receipt["identities"]
        for kind, entity in identities.items():
            body = {k: v for k, v in entity.items() if k != "id"}
            cls.require(entity["id"] == cls.identity(kind, body), "candidate_identity_mismatch")
        artifact, campaign = (identities[name]["bindings"] for name in ("artifact", "campaign"))
        cls.require(artifact["protocol_id"] == campaign["protocol_id"] == protocol["protocol_id"] and
                    artifact["task_id"] == identities["task"]["id"] and
                    artifact["method_id"] == identities["method"]["id"] and
                    campaign["panel_id"] == identities["panel"]["id"], "candidate_binding_mismatch")
        return "CandidateAdmission"

    def test_hash_goldens_and_byte_identical_historical_parent(self):
        parent = (ROOT / "contracts/tdl-contract-v1.json").read_bytes()
        self.assertEqual(len(parent), 4820)
        self.assertEqual(hashlib.sha256(parent).hexdigest(),
                         "281923c33da12e218d54c2226d138bbea2a22d86936bae5c3d7edfbfaf2c63ab")
        for case in self.fixture("canonical-hashes.json"):
            with self.subTest(kind=case["kind"], identity=case["identity"]):
                self.assertEqual(self.canonical(case["payload"]), case["canonical"])
                self.assertEqual(self.identity(case["kind"], case["payload"]), case["identity"])
        self.validate_protocol(self.protocol)
        self.assertEqual(self.admit(self.protocol, self.result), "CandidateAdmission")

    def test_every_protocol_design_field_is_hash_bound(self):
        original = self.protocol_id(self.protocol)
        for key in self.protocol:
            if key == "protocol_id":
                continue
            with self.subTest(field=key):
                changed = copy.deepcopy(self.protocol)
                changed[key] = {"mutation": changed[key]}
                self.assertNotEqual(self.protocol_id(changed), original)
        reordered = dict(reversed(list(self.protocol.items())))
        self.assertEqual(self.protocol_id(reordered), original)
        changed = copy.deepcopy(self.protocol)
        changed["byte_grid_bytes"].reverse()
        self.assertNotEqual(self.protocol_id(changed), original)
        body = copy.deepcopy({k: v for k, v in self.result["identities"]["artifact"].items() if k != "id"})
        original = self.identity("artifact", body)
        body["bindings"]["closure_sha256"] = "b" * 64
        self.assertNotEqual(self.identity("artifact", body), original)

    def test_shared_admission_counterexamples_have_source_free_reasons(self):
        for case in self.fixture("admission-cases.json"):
            with self.subTest(case=case["id"]):
                is_protocol = case["target"] == "protocol"
                value = copy.deepcopy(self.protocol if is_protocol else self.result)
                if "setup_panel" in case:
                    value["panel"] = copy.deepcopy(case["setup_panel"])
                parts = case["path"].strip("/").split("/")
                parent = value
                for part in parts[:-1]:
                    parent = parent[part]
                if case["remove"]:
                    del parent[parts[-1]]
                else:
                    parent[parts[-1]] = copy.deepcopy(case["value"])
                with self.assertRaises(ValueError) as error:
                    if case.get("resign_identity"):
                        entity = value["identities"]["artifact"]
                        entity["id"] = self.identity("artifact", {k: v for k, v in entity.items() if k != "id"})
                    if is_protocol:
                        if case["resign"]:
                            value["protocol_id"] = self.protocol_id(value)
                        self.validate_protocol(value)
                    else:
                        self.admit(self.protocol, value)
                self.assertEqual(str(error.exception), case["reason"])
                self.assertNotIn("CANARY_PRIVATE", str(error.exception))

    def test_strict_ingress_rejects_duplicates_and_nonfinite_numbers(self):
        for case in self.fixture("invalid-ingress.json"):
            with self.subTest(case=case["id"]):
                with self.assertRaises(ValueError) as error:
                    strict_load(case["text"])
                self.assertEqual(str(error.exception), case["reason"])
                self.assertNotIn("CANARY_PRIVATE", str(error.exception))

    def test_seven_state_axes_are_independent_and_do_not_certify_the_candidate(self):
        axes = self.protocol["state_axes"]
        self.assertEqual(len(axes), 7)
        for axis, options in axes.items():
            for option in options:
                with self.subTest(axis=axis, option=option):
                    receipt = copy.deepcopy(self.result)
                    receipt[axis] = option
                    self.assertEqual(self.admit(self.protocol, receipt), "CandidateAdmission")
                    self.assertFalse(receipt["citable"])
                    self.assertFalse(receipt["certified"])
                    for other in axes:
                        if other != axis:
                            self.assertEqual(receipt[other], self.result[other])

    def test_legacy_replay_is_explicit_and_uses_the_historical_reader(self):
        legacy = self.fixture("legacy-replay.json")
        with self.assertRaisesRegex(ValueError, "^legacy_receipt_requires_replay$"):
            self.admit(self.protocol, legacy)
        self.assertEqual(self.admit(self.protocol, legacy, legacy_replay=True), "LegacyReplay")
        self.assertEqual(run_case(legacy["case"]), legacy["case"]["expected"])
        with self.assertRaisesRegex(ValueError, "^candidate_not_legacy_receipt$"):
            self.admit(self.protocol, self.result, legacy_replay=True)
