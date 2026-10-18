import json
from pathlib import Path
import unittest

from research.transfer.calibration_receipt import development_protocol, verify_calibration_receipt
from research.transfer.evidence import digest
from research.transfer.power import stress_simulation


ROOT = Path(__file__).resolve().parents[3]


def receipt(**overrides):
    protocol = development_protocol()
    body = {
        "format": "dope-calibration-receipt",
        "version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_digest": digest(protocol),
        "repetitions": 2000,
        "bootstrap_draws": 1999,
        "frozen_for_confirmation": True,
        "manifest_digest": "a" * 64,
        "tolerances_frozen_before_outcomes": True,
        "mcse_reported": True,
        "wilson_reported": True,
    }
    body.update(overrides)
    sealed = dict(body)
    sealed.pop("binding_digest", None)
    sealed["binding_digest"] = digest(sealed)
    return sealed


class CalibrationReceiptTests(unittest.TestCase):
    def test_c02_protocol_is_separate_and_historical_v1_receipt_authenticates(self):
        protocol = json.loads((ROOT / "research/transfer/protocols/statistics-v2-development.json").read_text())
        self.assertEqual(protocol["protocol_id"], "transfer-statistics-v2-development")
        self.assertEqual(protocol["inference"]["default_draws"], 10000)
        self.assertFalse(protocol["frozen_for_confirmation"])
        self.assertFalse(protocol["qualification"])
        self.assertEqual(protocol["parent_protocol_digest"], digest(development_protocol()))
        historical = json.loads((ROOT / "tests/fixtures/vnext-successors/development-calibration-v1.json").read_text())
        self.assertTrue(verify_calibration_receipt(historical)["authenticates"])
        proposed = receipt(protocol_id=protocol["protocol_id"], protocol_digest=digest(protocol))
        self.assertFalse(verify_calibration_receipt(proposed)["qualifies"])
        self.assertEqual(verify_calibration_receipt(proposed)["reason"], "unknown_protocol")

    def test_s5_seeded_monte_carlo_smoke_cannot_freeze_confirmation(self):
        evidence = stress_simulation()
        self.assertEqual(set(evidence["scenarios"]), {"near_threshold", "low_gain", "heavy_tailed", "correlated", "few_source"})
        self.assertFalse(evidence["qualification"])
        self.assertEqual(evidence["calibration"], "unmeasured")
        for name, result in evidence["scenarios"].items():
            self.assertEqual(result["repetitions"], 20)
            self.assertFalse(result["qualification_authority"])
            if name == "few_source":
                self.assertEqual(result["failed_repetitions"], {"insufficient_families": 20})
                self.assertIsNone(result["metrics"]["upper_covered"]["mc95"])
            else:
                self.assertEqual(result["valid_repetitions"], 20)
                self.assertIsNotNone(result["metrics"]["upper_covered"]["mc_se"])
                self.assertIsNotNone(result["metrics"]["noninferior"]["mc95"])
        output = ROOT / "target/vnext/c02-seeded-stress-simulation.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        self.assertFalse(development_protocol()["frozen_for_confirmation"])

    def test_schema_has_no_qualification_field(self):
        schema = json.loads((ROOT / "schemas/calibration-receipt-v1.schema.json").read_text())
        self.assertEqual(schema["$id"], "urn:dope:calibration-receipt:v1")
        self.assertEqual(schema["properties"]["format"]["const"], "dope-calibration-receipt")
        self.assertEqual(schema["properties"]["version"]["const"], 1)
        self.assertIs(schema["additionalProperties"], False)
        self.assertNotIn("qualifies", schema["properties"])
        self.assertNotIn("calibration", schema["properties"])
        self.assertEqual(schema["required"], [
            "format", "version", "protocol_id", "protocol_digest", "repetitions", "bootstrap_draws",
            "frozen_for_confirmation", "manifest_digest", "tolerances_frozen_before_outcomes",
            "mcse_reported", "wilson_reported", "binding_digest",
        ])

    def test_development_protocol_does_not_qualify(self):
        protocol = development_protocol()
        self.assertEqual(protocol["protocol_id"], "transfer-statistics-v1-development")
        self.assertIs(protocol["frozen_for_confirmation"], False)
        self.assertEqual(protocol["development_validation"]["repetitions"], 120)
        decision = verify_calibration_receipt(receipt())
        self.assertIs(decision["authenticates"], True)
        self.assertIs(decision["qualifies"], False)
        self.assertEqual(decision["calibration"], "unmeasured")
        self.assertEqual(decision["reason"], "development_protocol_is_not_n13b")
        self.assertIs(decision["n13b"], False)
        self.assertEqual(decision["campaign"], "unmeasured")
        self.assertIs(decision["release"], False)

    def test_tamper_unknown_protocol_and_self_assertion_stay_unqualified(self):
        sealed = receipt()
        sealed["binding_digest"] = "b" * 64
        tampered = verify_calibration_receipt(sealed)
        self.assertIs(tampered["authenticates"], False)
        self.assertIs(tampered["qualifies"], False)
        self.assertEqual(tampered["reason"], "tampered_calibration_receipt")
        mismatched = verify_calibration_receipt(receipt(protocol_digest="c" * 64))
        self.assertIs(mismatched["authenticates"], False)
        self.assertEqual(mismatched["reason"], "protocol_digest_mismatch")
        self.assertIs(mismatched["qualifies"], False)
        unknown = verify_calibration_receipt(receipt(protocol_id="transfer-statistics-v1-final"))
        self.assertIs(unknown["authenticates"], False)
        self.assertEqual(unknown["reason"], "unknown_protocol")
        self.assertIs(unknown["qualifies"], False)
        with self.assertRaises(ValueError) as asserted:
            verify_calibration_receipt(receipt(qualifies=True, calibration="measured"))
        self.assertEqual(str(asserted.exception), "self_asserted_qualification")
        with self.assertRaises(ValueError) as version:
            verify_calibration_receipt(receipt(version=2))
        self.assertEqual(str(version.exception), "unknown_calibration_receipt_version")
        with self.assertRaises(ValueError) as malformed:
            verify_calibration_receipt([])
        self.assertEqual(str(malformed.exception), "invalid_calibration_receipt")


if __name__ == "__main__":
    unittest.main()
