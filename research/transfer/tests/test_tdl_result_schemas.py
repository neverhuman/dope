"""FOLLOW-L1a: unavailable measurements remain explicit in failed receipts."""
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from research.transfer.tdl.golden import strict_load

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
