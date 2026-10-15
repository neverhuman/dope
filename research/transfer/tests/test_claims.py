"""Synthetic claim-graph checks. These fixtures do not admit a campaign."""

from __future__ import annotations

import hashlib
import unittest

from research.transfer.claims import (
    ClaimError,
    VerifiedClaimGraph,
    public_projection,
    synthetic_fixture,
    verify_claim_graph,
)
from research.transfer.evidence import canonical_json, parse_json
from research.transfer.lineage import assert_public_provenance


SENTINEL = "sentinel-marker-99999"


def _pin(raw):
    return hashlib.sha256(raw).hexdigest()


def _replace_receipt(document, receipts, receipt_id, mutate):
    receipt = parse_json(receipts[receipt_id])
    mutate(receipt)
    raw = canonical_json(receipt)
    doc = parse_json(document)
    for claim in doc["claims"]:
        for item in claim["references"]:
            if item["receipt_id"] == receipt_id:
                item["sha256"] = _pin(raw)
    updated = dict(receipts)
    updated[receipt_id] = raw
    return canonical_json(doc), updated


def _named(claim_id, receipt_change=None, comparison="raw_gap_superiority"):
    fixture = synthetic_fixture(comparison)
    document = fixture["document"]
    receipts = fixture["receipts"]
    receipt_id = next(iter(receipts))
    if receipt_change is not None:
        document, receipts = _replace_receipt(document, receipts, receipt_id, receipt_change)
    doc = parse_json(document)
    doc["claims"][0]["claim_id"] = claim_id
    return canonical_json(doc), receipts, fixture["expected"]


def _graph(comparison="raw_gap_superiority"):
    fixture = synthetic_fixture(comparison)
    graph = verify_claim_graph(fixture["document"], fixture["receipts"], fixture["expected"])
    return fixture, graph, public_projection(graph)


class ClaimGraphTests(unittest.TestCase):
    def test_each_comparison_stays_unqualified(self):
        for comparison in ("raw_gap_superiority", "adjusted_z_noninferiority", "equivalence"):
            fixture, graph, projection = _graph(comparison)
            self.assertIs(type(graph), VerifiedClaimGraph)
            self.assertEqual(assert_public_provenance(projection), projection)
            claim = projection["claims"][0]
            self.assertEqual(claim["claim_id"], "rate-risk-frontier")
            self.assertFalse(claim["campaign_admitted"])
            self.assertFalse(claim["n21_admitted"])
            self.assertFalse(claim["qualified"])
            self.assertEqual(claim["qualification"], "synthetic")
            self.assertTrue(claim["vendor_executed"])
            self.assertEqual(claim["artifact_count"], 2)
            self.assertEqual(claim["fit_count"], 3)
            self.assertEqual(claim["sample_count"], 5)
            self.assertEqual(claim[comparison]["decision"], "supported")
            self.assertFalse(claim[comparison]["qualified"])
            body = {key: value for key, value in projection.items() if key != "projection_sha256"}
            self.assertEqual(projection["projection_sha256"], _pin(canonical_json(body)))
            reference = parse_json(fixture["document"])["claims"][0]["references"][0]
            pin = _pin(fixture["receipts"][reference["receipt_id"]])
            self.assertEqual(reference["sha256"], pin)
            self.assertEqual(claim["receipts"], [{"sha256": pin}])
            self.assertNotIn("receipt_pins", claim)
            self.assertNotIn("receipt_sha256", claim)

    def test_endpoints_do_not_share_a_decision(self):
        _, _, raw_gap = _graph("raw_gap_superiority")
        claim = raw_gap["claims"][0]
        self.assertEqual(claim["raw_gap_superiority"]["decision"], "supported")
        self.assertEqual(claim["raw_gap_superiority"]["bound_side"], "upper")
        self.assertEqual(claim["adjusted_z_noninferiority"]["decision"], "unavailable")
        self.assertIsNone(claim["adjusted_z_noninferiority"]["bound_side"])
        self.assertEqual(claim["equivalence"]["decision"], "unavailable")
        _, _, equivalence = _graph("equivalence")
        other = equivalence["claims"][0]
        self.assertEqual(other["equivalence"]["decision"], "supported")
        self.assertEqual(other["equivalence"]["bound_side"], "both")
        self.assertEqual(other["equivalence"]["equivalence_margin"], 0.5)
        self.assertEqual(other["raw_gap_superiority"]["decision"], "not_supported")
        _, _, adjusted = _graph("adjusted_z_noninferiority")
        z_claim = adjusted["claims"][0]
        self.assertEqual(z_claim["adjusted_z_noninferiority"]["decision"], "supported")
        self.assertEqual(z_claim["adjusted_z_noninferiority"]["bound_side"], "upper")
        self.assertEqual(z_claim["raw_gap_superiority"]["decision"], "unavailable")

    def test_changed_pin_and_mutated_bytes(self):
        fixture = synthetic_fixture("raw_gap_superiority")
        doc = parse_json(fixture["document"])
        doc["claims"][0]["references"][0]["sha256"] = "a" * 64
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "pin_mismatch")
        raw = fixture["receipts"]["receipt-raw-gap"]
        mutated = raw + b"\n"
        self.assertNotEqual(mutated, raw)
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(fixture["document"], {"receipt-raw-gap": mutated}, fixture["expected"])
        self.assertIn(str(caught.exception), {"pin_mismatch", "invalid_receipt", "invalid_bound"})

    def test_missing_duplicate_and_empty_claims(self):
        fixture = synthetic_fixture("raw_gap_superiority")
        doc = parse_json(fixture["document"])
        doc["claims"][0]["references"][0]["receipt_id"] = "receipt-absent"
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "missing_receipt")
        doc = parse_json(fixture["document"])
        doc["claims"][0]["references"].append(dict(doc["claims"][0]["references"][0]))
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "duplicate_reference")
        doc = parse_json(fixture["document"])
        doc["claims"].append(parse_json(canonical_json(doc["claims"][0])))
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "duplicate_claim")
        doc = parse_json(fixture["document"])
        doc["claims"] = []
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "missing_claim")

    def test_mixed_identities(self):
        cases = {
            "mixed_cohort": {"cohort_sha256": "b" * 64},
            "mixed_metric": {"metric_sha256": "c" * 64},
            "mixed_protocol": {"protocol_sha256": "d" * 64},
            "mixed_label": {"comparison_label": "equivalence"},
        }
        for code, change in cases.items():
            fixture = synthetic_fixture("raw_gap_superiority")
            receipt = parse_json(fixture["receipts"]["receipt-raw-gap"])
            receipt.update(change)
            receipt["receipt_id"] = "receipt-other"
            raw = canonical_json(receipt)
            doc = parse_json(fixture["document"])
            doc["claims"][0]["references"].append({"receipt_id": "receipt-other", "sha256": _pin(raw)})
            receipts = dict(fixture["receipts"])
            receipts["receipt-other"] = raw
            with self.assertRaises(ClaimError) as caught:
                verify_claim_graph(canonical_json(doc), receipts, fixture["expected"])
            self.assertEqual(str(caught.exception), code)

    def test_vendor_false_does_not_report_execution(self):
        fixture = synthetic_fixture("raw_gap_superiority")
        document, receipts = _replace_receipt(
            fixture["document"], fixture["receipts"], "receipt-raw-gap",
            lambda receipt: receipt.update(vendor_executed=False),
        )
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(document, receipts, fixture["expected"])
        self.assertEqual(str(caught.exception), "vendor_not_executed")
        self.assertNotIn("true", str(caught.exception))

    def test_count_conflation_and_margin_label(self):
        fixture = synthetic_fixture("raw_gap_superiority")
        doc = parse_json(fixture["document"])
        doc["claims"][0]["fit_count"] = 5
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "sample_fit_conflation")
        doc = parse_json(fixture["document"])
        doc["claims"][0]["artifact_count"] = 9
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "count_mismatch")
        document, receipts = _replace_receipt(
            fixture["document"], fixture["receipts"], "receipt-raw-gap",
            lambda receipt: receipt.update(lower=-0.25, upper=0.25),
        )
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(document, receipts, fixture["expected"])
        self.assertEqual(str(caught.exception), "label_mismatch")
        doc = parse_json(fixture["document"])
        doc["claims"][0]["equivalence_margin"] = 0.25
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "margin_mismatch")

    def test_unverified_bound_and_self_declared_flag(self):
        fixture = synthetic_fixture("raw_gap_superiority")
        document, receipts = _replace_receipt(
            fixture["document"], fixture["receipts"], "receipt-raw-gap",
            lambda receipt: receipt.update(lower=None),
        )
        doc = parse_json(document)
        doc["claims"][0]["equivalence_margin"] = None
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), receipts, fixture["expected"])
        self.assertEqual(str(caught.exception), "unverified_bound")
        doc = parse_json(fixture["document"])
        doc["claims"][0]["verified"] = True
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "self_declared_verification")

    def test_bool_and_budget_fail_before_the_interval(self):
        fixture = synthetic_fixture("raw_gap_superiority")
        doc = parse_json(fixture["document"])
        doc["claims"][0]["artifact_count"] = True
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(canonical_json(doc), fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "invalid_claim")
        document, receipts = _replace_receipt(
            fixture["document"], fixture["receipts"], "receipt-raw-gap",
            lambda receipt: receipt.update(lower=True),
        )
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(document, receipts, fixture["expected"])
        self.assertEqual(str(caught.exception), "invalid_bound")
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(b"x" * 65537, fixture["receipts"], fixture["expected"])
        self.assertEqual(str(caught.exception), "document_budget_exceeded")

    def test_projection_refuses_a_dict_and_omits_sentinels(self):
        with self.assertRaises(ClaimError) as caught:
            public_projection({SENTINEL: SENTINEL, "verified": True})
        self.assertEqual(str(caught.exception), "verification_required")
        self.assertNotIn(SENTINEL, str(caught.exception))
        with self.assertRaises(ClaimError) as caught:
            VerifiedClaimGraph({SENTINEL: SENTINEL})
        self.assertEqual(str(caught.exception), "verification_required")
        self.assertNotIn(SENTINEL, str(caught.exception))
        fixture = synthetic_fixture("raw_gap_superiority")
        document, receipts = _replace_receipt(
            fixture["document"], fixture["receipts"], "receipt-raw-gap",
            lambda receipt: receipt.update(header=SENTINEL, maximum=99999.5, per_query=[99999.5]),
        )
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(document, receipts, fixture["expected"])
        self.assertEqual(str(caught.exception), "invalid_receipt")
        self.assertNotIn(SENTINEL, str(caught.exception))
        self.assertNotIn("99999.5", str(caught.exception))
        _, _, projection = _graph()
        rendered = canonical_json(projection).decode()
        self.assertNotIn("header", rendered)
        self.assertNotIn(SENTINEL, rendered)
        self.assertNotIn("99999.5", rendered)

    def test_limitation_ids_keep_computed_states(self):
        preserved = (
            "unfavorable-baseline",
            "cap-failure-mass",
            "tail-harm",
            "rights-unfrozen",
            "historical-panel-reuse",
            "seed-dependence",
            "campaign-cost",
        )
        document, receipts, expected = _named(
            "unfavorable-baseline",
            lambda receipt: receipt.update(lower=0.25, upper=0.5),
        )
        projection = public_projection(verify_claim_graph(document, receipts, expected))
        self.assertEqual(assert_public_provenance(projection), projection)
        claim = projection["claims"][0]
        self.assertEqual(claim["raw_gap_superiority"]["decision"], "not_supported")
        self.assertFalse(claim["qualified"])
        self.assertFalse(claim["campaign_admitted"])
        self.assertFalse(claim["n21_admitted"])
        self.assertFalse(claim["raw_gap_superiority"]["qualified"])
        self.assertEqual(claim["receipts"], [{"sha256": _pin(receipts["receipt-raw-gap"])}])
        self.assertNotIn("receipt_sha256", claim)
        document, receipts, expected = _named(
            "tail-harm",
            lambda receipt: receipt.update(lower=-0.25, upper=0.25),
        )
        claim = public_projection(verify_claim_graph(document, receipts, expected))["claims"][0]
        self.assertEqual(claim["raw_gap_superiority"]["decision"], "inconclusive")
        self.assertEqual(claim["adjusted_z_noninferiority"]["decision"], "unavailable")
        self.assertEqual(claim["equivalence"]["decision"], "unavailable")
        self.assertFalse(claim["qualified"])
        for claim_id in preserved:
            document, receipts, expected = _named(claim_id, lambda receipt: receipt.update(lower=None))
            claim = public_projection(verify_claim_graph(document, receipts, expected))["claims"][0]
            self.assertEqual(claim["claim_id"], claim_id)
            self.assertEqual(claim["raw_gap_superiority"]["decision"], "unavailable")
            self.assertIsNone(claim["raw_gap_superiority"]["lower"])
            self.assertFalse(claim["qualified"])
            self.assertFalse(claim["campaign_admitted"])
        fixture = synthetic_fixture("raw_gap_superiority")
        document, receipts = _replace_receipt(
            fixture["document"], fixture["receipts"], "receipt-raw-gap",
            lambda receipt: receipt.update(lower=None),
        )
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(document, receipts, fixture["expected"])
        self.assertEqual(str(caught.exception), "unverified_bound")
        document, receipts = _replace_receipt(
            fixture["document"], fixture["receipts"], "receipt-raw-gap",
            lambda receipt: receipt.update(lower=0.25, upper=0.5),
        )
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(document, receipts, fixture["expected"])
        self.assertEqual(str(caught.exception), "label_mismatch")
        for headline in ("algorithmic-hypothesis-margin", "replay-receipt", "one-cell-loss", "full-refit"):
            document, receipts, expected = _named(headline, lambda receipt: receipt.update(lower=None))
            with self.assertRaises(ClaimError) as caught:
                verify_claim_graph(document, receipts, expected)
            self.assertEqual(str(caught.exception), "unverified_bound")
        for flag in ("verified", "qualified"):
            document, receipts, expected = _named("cap-failure-mass")
            doc = parse_json(document)
            doc["claims"][0][flag] = True
            with self.assertRaises(ClaimError) as caught:
                verify_claim_graph(canonical_json(doc), receipts, expected)
            self.assertEqual(str(caught.exception), "self_declared_verification")
        names = ("raw_gap_superiority", "adjusted_z_noninferiority", "equivalence")
        for comparison in names:
            document, receipts, expected = _named("seed-dependence", comparison=comparison)
            claim = public_projection(verify_claim_graph(document, receipts, expected))["claims"][0]
            self.assertEqual(claim[comparison]["decision"], "supported")
            self.assertFalse(claim["qualified"])
            self.assertFalse(claim[comparison]["qualified"])
            self.assertFalse(claim["campaign_admitted"])
            for other in names:
                if other != comparison:
                    self.assertNotEqual(claim[other]["decision"], "supported")

    def test_incomplete_denominator_and_identity(self):
        fixture = synthetic_fixture("raw_gap_superiority")
        document, receipts = _replace_receipt(
            fixture["document"], fixture["receipts"], "receipt-raw-gap",
            lambda receipt: receipt.update(denominator_observed=3),
        )
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(document, receipts, fixture["expected"])
        self.assertEqual(str(caught.exception), "incomplete_denominator")
        expected = dict(fixture["expected"])
        expected["loss"] = "mae"
        with self.assertRaises(ClaimError) as caught:
            verify_claim_graph(fixture["document"], fixture["receipts"], expected)
        self.assertEqual(str(caught.exception), "identity_mismatch")


if __name__ == "__main__":
    unittest.main()
