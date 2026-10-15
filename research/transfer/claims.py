"""Bind a claim graph to receipt bytes.

A matching pin records an observation. It does not admit a campaign, N21,
or a release. Callers cannot construct VerifiedClaimGraph.
"""

from __future__ import annotations

import hashlib
import math
import re

from .evidence import EvidenceError, canonical_json, parse_json
from .lineage import assert_public_provenance
from .statistics import interval_decision


FORMAT = "dope-claim-graph"
VERSION = 1
RECEIPT_FORMAT = "dope-claim-bound-receipt"
PROJECTION_FORMAT = "dope-claim-public-projection"
MAX_DOCUMENT_BYTES = 65536
MAX_RECEIPT_BYTES = 65536
MAX_CLAIMS = 8
MAX_RECEIPTS = 8
MAX_REFERENCES = 8

_TOKEN = object()
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UNIT = re.compile(r"^[a-z0-9_]{1,64}$")
_RECEIPT_ID = re.compile(r"^[a-z0-9-]{1,64}$")
_UNSAFE = re.compile(r"[\\{}$&#%^~]")
_LADDER = frozenset({
    "rate-risk-frontier",
    "cross-source-transfer",
    "cross-learner-transfer",
    "standard-transfer-bound",
    "algorithmic-hypothesis",
    "algorithmic-hypothesis-margin",
    "direct-predictor-control",
    "coreset-control",
    "distillation-control",
    "circuit-control",
    "unfavorable-baseline",
    "cap-failure-mass",
    "tail-harm",
    "rights-unfrozen",
    "historical-panel-reuse",
    "seed-dependence",
    "campaign-cost",
    "replay-receipt",
    "one-cell-loss",
    "full-refit",
})
_PRESERVED_STATES = frozenset({
    "unfavorable-baseline",
    "cap-failure-mass",
    "tail-harm",
    "rights-unfrozen",
    "historical-panel-reuse",
    "seed-dependence",
    "campaign-cost",
})
_COMPARISONS = (
    "raw_gap_superiority",
    "adjusted_z_noninferiority",
    "equivalence",
)
_LOSSES = frozenset({"mse", "rmse", "mae"})
_ENDPOINTS = frozenset({"gap", "normalized_gap", "z"})
_FORBIDDEN = frozenset({"verified", "qualified", "campaign_admitted", "n21_admitted"})
_DOCUMENT_KEYS = frozenset({"format", "version", "claims"})
_CLAIM_KEYS = frozenset({
    "claim_id", "protocol_sha256", "cohort_sha256", "recipe_sha256", "source_sha256",
    "loss", "loss_unit", "metric_sha256", "artifact_count", "fit_count", "sample_count",
    "asserted_comparison", "equivalence_margin", "references", "synthetic",
})
_EXPECTED_KEYS = frozenset({
    "protocol_sha256", "cohort_sha256", "recipe_sha256", "source_sha256",
    "loss", "loss_unit", "metric_sha256",
})
_RECEIPT_KEYS = frozenset({
    "format", "version", "receipt_id", "protocol_sha256", "cohort_sha256",
    "recipe_sha256", "source_sha256", "loss", "loss_unit", "metric_sha256",
    "comparison_label", "artifact_count", "fit_count", "sample_count",
    "denominator_observed", "denominator_expected", "vendor_executed", "endpoint",
    "lower", "upper", "estimate", "level", "equivalence_margin", "margin",
    "moment_assumption", "distribution_free", "synthetic",
})
_MARGIN_KEYS = frozenset({"relative", "absolute_scaled", "absolute_raw"})
_IDENTITY_FIELDS = (
    "protocol_sha256", "cohort_sha256", "recipe_sha256", "source_sha256",
    "loss", "loss_unit", "metric_sha256",
)
_FIXTURE_LITERALS = {
    "raw_gap_superiority": {
        "receipt_id": "receipt-raw-gap",
        "endpoint": "gap",
        "lower": -0.5,
        "upper": -0.25,
        "equivalence_margin": None,
    },
    "equivalence": {
        "receipt_id": "receipt-equivalence",
        "endpoint": "gap",
        "lower": -0.25,
        "upper": 0.25,
        "equivalence_margin": 0.5,
    },
    "adjusted_z_noninferiority": {
        "receipt_id": "receipt-adjusted-z",
        "endpoint": "z",
        "lower": -0.5,
        "upper": -0.25,
        "equivalence_margin": None,
    },
}


class ClaimError(ValueError):
    """A claim rule failed. The message is the code only."""


class VerifiedClaimGraph:
    """Private verifier output. Outside construction raises verification_required."""

    def __init__(self, projection, token=None):
        if token is not _TOKEN:
            raise ClaimError("verification_required")
        self._projection = projection


def _fail(code):
    raise ClaimError(code)


def _parse(data, limit, budget_code, shape_code):
    if type(data) is not bytes:
        _fail(shape_code)
    if len(data) > limit:
        _fail(budget_code)
    try:
        value = parse_json(data)
    except EvidenceError as exc:
        _fail(str(exc))
    if canonical_json(value) != data:
        _fail(shape_code)
    return value


def _public_text(value, pattern, code):
    if type(value) is not str:
        _fail(code)
    if _UNSAFE.search(value):
        _fail("unsafe_label")
    if pattern.fullmatch(value) is None:
        _fail(code)


def _sha_field(value, code):
    _public_text(value, _SHA, code)


def _count(value, code):
    if type(value) is not int or value < 0:
        _fail(code)


def _number_or_null(value, code):
    if value is None:
        return
    if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value):
        _fail(code)


def _same(left, right):
    return canonical_json(left) == canonical_json(right)


def _closed(value, keys, code):
    if type(value) is not dict or set(value) != keys:
        _fail(code)


def _check_expected(expected):
    _closed(expected, _EXPECTED_KEYS, "invalid_claim")
    for key in ("protocol_sha256", "cohort_sha256", "recipe_sha256", "source_sha256", "metric_sha256"):
        _sha_field(expected[key], "invalid_claim")
    if expected["loss"] not in _LOSSES:
        _fail("invalid_claim")
    _public_text(expected["loss_unit"], _UNIT, "invalid_claim")


def _check_margin(margin):
    _closed(margin, _MARGIN_KEYS, "invalid_receipt")
    for key in ("relative", "absolute_scaled"):
        _number_or_null(margin[key], "invalid_bound")
        if margin[key] is None:
            _fail("invalid_bound")
    _number_or_null(margin["absolute_raw"], "invalid_bound")


def _check_claim(claim):
    if type(claim) is not dict:
        _fail("invalid_claim")
    if _FORBIDDEN & set(claim):
        _fail("self_declared_verification")
    _closed(claim, _CLAIM_KEYS, "invalid_claim")
    if claim["claim_id"] not in _LADDER:
        _public_text(claim["claim_id"], re.compile(r"^.*$"), "unknown_claim")
        _fail("unknown_claim")
    _public_text(claim["claim_id"], re.compile(r"^[a-z0-9-]{1,64}$"), "unknown_claim")
    for key in ("protocol_sha256", "cohort_sha256", "recipe_sha256", "source_sha256", "metric_sha256"):
        _sha_field(claim[key], "invalid_claim")
    if claim["loss"] not in _LOSSES:
        _fail("invalid_claim")
    _public_text(claim["loss_unit"], _UNIT, "invalid_claim")
    for key in ("artifact_count", "fit_count", "sample_count"):
        _count(claim[key], "invalid_claim")
    if claim["asserted_comparison"] not in _COMPARISONS:
        _fail("invalid_claim")
    _number_or_null(claim["equivalence_margin"], "invalid_bound")
    if type(claim["synthetic"]) is not bool:
        _fail("invalid_claim")
    references = claim["references"]
    if type(references) is not list or not 1 <= len(references) <= MAX_REFERENCES:
        _fail("invalid_claim")
    seen_ids = set()
    seen_sha = set()
    for item in references:
        _closed(item, frozenset({"receipt_id", "sha256"}), "invalid_claim")
        _public_text(item["receipt_id"], _RECEIPT_ID, "invalid_claim")
        _sha_field(item["sha256"], "invalid_claim")
        if item["receipt_id"] in seen_ids or item["sha256"] in seen_sha:
            _fail("duplicate_reference")
        seen_ids.add(item["receipt_id"])
        seen_sha.add(item["sha256"])


def _check_receipt(receipt, receipt_id):
    _closed(receipt, _RECEIPT_KEYS, "invalid_receipt")
    if receipt["format"] != RECEIPT_FORMAT or type(receipt["version"]) is not int or receipt["version"] != 1:
        _fail("invalid_receipt")
    if receipt["receipt_id"] != receipt_id:
        _fail("invalid_receipt")
    _public_text(receipt["receipt_id"], _RECEIPT_ID, "invalid_receipt")
    for key in ("protocol_sha256", "cohort_sha256", "recipe_sha256", "source_sha256", "metric_sha256"):
        _sha_field(receipt[key], "invalid_receipt")
    if receipt["loss"] not in _LOSSES:
        _fail("invalid_receipt")
    _public_text(receipt["loss_unit"], _UNIT, "invalid_receipt")
    if receipt["comparison_label"] not in _COMPARISONS:
        _fail("invalid_receipt")
    for key in ("artifact_count", "fit_count", "sample_count"):
        _count(receipt[key], "invalid_receipt")
    for key in ("denominator_observed", "denominator_expected"):
        if type(receipt[key]) is not int:
            _fail("invalid_receipt")
    if receipt["endpoint"] not in _ENDPOINTS:
        _fail("invalid_receipt")
    for key in ("lower", "upper", "estimate", "level", "equivalence_margin"):
        _number_or_null(receipt[key], "invalid_bound")
    _check_margin(receipt["margin"])
    if receipt["moment_assumption"] != "finite_variance" or receipt["distribution_free"] is not False:
        _fail("assumption_mismatch")
    if type(receipt["synthetic"]) is not bool:
        _fail("invalid_receipt")
    if receipt["vendor_executed"] is False:
        _fail("vendor_not_executed")
    if receipt["vendor_executed"] is not True:
        _fail("vendor_execution_unverified")
    if receipt["denominator_observed"] < 1 or receipt["denominator_observed"] != receipt["denominator_expected"]:
        _fail("incomplete_denominator")


def _identities_agree(rows, field, code):
    values = [row[field] for row in rows]
    if any(not _same(value, values[0]) for value in values):
        _fail(code)


def _decision_state(kind, receipt):
    endpoint = receipt["endpoint"]
    lower = receipt["lower"]
    upper = receipt["upper"]
    margin = receipt["equivalence_margin"]
    if lower is None or upper is None:
        return "unavailable"
    if kind == "raw_gap_superiority" and endpoint not in ("gap", "normalized_gap"):
        return "unavailable"
    if kind == "adjusted_z_noninferiority" and endpoint != "z":
        return "unavailable"
    if kind == "equivalence" and margin is None:
        return "unavailable"
    try:
        decided = interval_decision(lower, upper, endpoint=endpoint, equivalence_margin=margin)
    except ValueError:
        _fail("invalid_bound")
    if kind == "raw_gap_superiority":
        held = decided["superior"]
    elif kind == "adjusted_z_noninferiority":
        held = decided["noninferior"]
    else:
        held = decided["equivalent"]
    if held:
        return "supported"
    if decided["inconclusive"]:
        return "inconclusive"
    return "not_supported"


def _decision_object(kind, receipt, state):
    if state == "unavailable":
        return {
            "decision": "unavailable",
            "qualified": False,
            "lower": None,
            "upper": None,
            "estimate": None,
            "level": None,
            "margin": None,
            "equivalence_margin": None,
            "bound_side": None,
        }
    side = "both" if kind == "equivalence" else "upper"
    return {
        "decision": state,
        "qualified": False,
        "lower": receipt["lower"],
        "upper": receipt["upper"],
        "estimate": receipt["estimate"],
        "level": receipt["level"],
        "margin": dict(receipt["margin"]) if kind == "adjusted_z_noninferiority" else None,
        "equivalence_margin": receipt["equivalence_margin"] if kind == "equivalence" else None,
        "bound_side": side,
    }


def _project_claim(claim, rows):
    states = {kind: _decision_state(kind, rows[0]) for kind in _COMPARISONS}
    asserted = claim["asserted_comparison"]
    if claim["claim_id"] not in _PRESERVED_STATES:
        if states[asserted] == "unavailable":
            _fail("unverified_bound")
        if states[asserted] != "supported":
            _fail("label_mismatch")
    return {
        "claim_id": claim["claim_id"],
        "protocol_sha256": claim["protocol_sha256"],
        "cohort_sha256": claim["cohort_sha256"],
        "recipe_sha256": claim["recipe_sha256"],
        "source_sha256": claim["source_sha256"],
        "metric_sha256": claim["metric_sha256"],
        "loss": claim["loss"],
        "loss_unit": claim["loss_unit"],
        "artifact_count": claim["artifact_count"],
        "fit_count": claim["fit_count"],
        "sample_count": claim["sample_count"],
        "vendor_executed": True,
        "qualification": "synthetic" if claim["synthetic"] else "development",
        "campaign_admitted": False,
        "n21_admitted": False,
        "qualified": False,
        "synthetic": claim["synthetic"],
        "receipts": [{"sha256": pin} for pin in sorted({item["sha256"] for item in claim["references"]})],
        "raw_gap_superiority": _decision_object("raw_gap_superiority", rows[0], states["raw_gap_superiority"]),
        "adjusted_z_noninferiority": _decision_object(
            "adjusted_z_noninferiority", rows[0], states["adjusted_z_noninferiority"]
        ),
        "equivalence": _decision_object("equivalence", rows[0], states["equivalence"]),
    }


def verify_claim_graph(document, receipts, expected):
    """Return a token-guarded graph. The projection stays unqualified."""
    if type(document) is dict and _FORBIDDEN & set(document):
        _fail("self_declared_verification")
    parsed = _parse(document, MAX_DOCUMENT_BYTES, "document_budget_exceeded", "invalid_claim")
    if type(parsed) is dict and _FORBIDDEN & set(parsed):
        _fail("self_declared_verification")
    _closed(parsed, _DOCUMENT_KEYS, "invalid_claim")
    if parsed["format"] != FORMAT or type(parsed["version"]) is not int or parsed["version"] != 1:
        _fail("invalid_claim")
    claims = parsed["claims"]
    if type(claims) is not list:
        _fail("invalid_claim")
    if not claims:
        _fail("missing_claim")
    if len(claims) > MAX_CLAIMS:
        _fail("document_budget_exceeded")
    for claim in claims:
        if type(claim) is dict and _FORBIDDEN & set(claim):
            _fail("self_declared_verification")
    for claim in claims:
        _check_claim(claim)
    ids = [claim["claim_id"] for claim in claims]
    if len(ids) != len(set(ids)):
        _fail("duplicate_claim")
    _check_expected(expected)
    if type(receipts) is not dict or any(type(key) is not str for key in receipts):
        _fail("invalid_claim")
    if len(receipts) > MAX_RECEIPTS:
        _fail("receipt_budget_exceeded")
    parsed_receipts = {}
    for key, raw in receipts.items():
        if type(raw) is not bytes or len(raw) > MAX_RECEIPT_BYTES:
            _fail("receipt_budget_exceeded" if type(raw) is bytes else "invalid_receipt")
    for key, raw in receipts.items():
        parsed_receipts[key] = _parse(raw, MAX_RECEIPT_BYTES, "receipt_budget_exceeded", "invalid_receipt")
    projected = []
    for claim in claims:
        rows = []
        for item in claim["references"]:
            receipt_id = item["receipt_id"]
            if receipt_id not in parsed_receipts:
                _fail("missing_receipt")
            raw = receipts[receipt_id]
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                _fail("pin_mismatch")
            receipt = parsed_receipts[receipt_id]
            _check_receipt(receipt, receipt_id)
            rows.append(receipt)
        _identities_agree(rows, "cohort_sha256", "mixed_cohort")
        _identities_agree(rows, "protocol_sha256", "mixed_protocol")
        _identities_agree(rows, "metric_sha256", "mixed_metric")
        _identities_agree(rows, "comparison_label", "mixed_label")
        if any(row["comparison_label"] != claim["asserted_comparison"] for row in rows):
            _fail("mixed_label")
        for field in _IDENTITY_FIELDS:
            if any(not _same(row[field], expected[field]) or not _same(claim[field], expected[field]) for row in rows):
                _fail("identity_mismatch")
        if any(not _same(row["margin"], rows[0]["margin"]) or not _same(row["equivalence_margin"], rows[0]["equivalence_margin"]) for row in rows):
            _fail("margin_mismatch")
        if not _same(claim["equivalence_margin"], rows[0]["equivalence_margin"]):
            _fail("margin_mismatch")
        bound_fields = ("endpoint", "lower", "upper", "estimate", "level")
        if any(any(not _same(row[field], rows[0][field]) for field in bound_fields) for row in rows):
            _fail("invalid_bound")
        for row in rows:
            if claim["fit_count"] == row["sample_count"] and claim["fit_count"] != row["fit_count"]:
                _fail("sample_fit_conflation")
            if any(claim[key] != row[key] for key in ("artifact_count", "fit_count", "sample_count")):
                _fail("count_mismatch")
        projected.append(_project_claim(claim, rows))
    body = {"format": PROJECTION_FORMAT, "version": 1, "claims": projected}
    body["projection_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    try:
        assert_public_provenance(body)
    except Exception as exc:
        if type(exc) is ClaimError:
            raise
        _fail("unsafe_label")
    return VerifiedClaimGraph(body, _TOKEN)


def public_projection(graph):
    """Return the public dict. Other types fail before their fields are read."""
    if type(graph) is not VerifiedClaimGraph:
        _fail("verification_required")
    body = _parse(canonical_json(graph._projection), MAX_DOCUMENT_BYTES, "document_budget_exceeded", "invalid_claim")
    return assert_public_provenance(body)


def _identity_pin(role):
    payload = {"name": "rate-risk-frontier", "role": role, "synthetic": True}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def synthetic_fixture(comparison):
    """Synthetic bytes for one asserted comparison. Campaign admission stays false."""
    if comparison not in _FIXTURE_LITERALS:
        _fail("invalid_claim")
    literal = _FIXTURE_LITERALS[comparison]
    pins = {role: _identity_pin(role) for role in ("protocol", "cohort", "recipe", "source", "metric")}
    receipt = {
        "format": RECEIPT_FORMAT,
        "version": 1,
        "receipt_id": literal["receipt_id"],
        "protocol_sha256": pins["protocol"],
        "cohort_sha256": pins["cohort"],
        "recipe_sha256": pins["recipe"],
        "source_sha256": pins["source"],
        "loss": "mse",
        "loss_unit": "original_mse",
        "metric_sha256": pins["metric"],
        "comparison_label": comparison,
        "artifact_count": 2,
        "fit_count": 3,
        "sample_count": 5,
        "denominator_observed": 4,
        "denominator_expected": 4,
        "vendor_executed": True,
        "endpoint": literal["endpoint"],
        "lower": literal["lower"],
        "upper": literal["upper"],
        "estimate": None,
        "level": None,
        "equivalence_margin": literal["equivalence_margin"],
        "margin": {"relative": 0, "absolute_scaled": 0, "absolute_raw": None},
        "moment_assumption": "finite_variance",
        "distribution_free": False,
        "synthetic": True,
    }
    raw = canonical_json(receipt)
    pin = hashlib.sha256(raw).hexdigest()
    document = {
        "format": FORMAT,
        "version": 1,
        "claims": [{
            "claim_id": "rate-risk-frontier",
            "protocol_sha256": pins["protocol"],
            "cohort_sha256": pins["cohort"],
            "recipe_sha256": pins["recipe"],
            "source_sha256": pins["source"],
            "loss": "mse",
            "loss_unit": "original_mse",
            "metric_sha256": pins["metric"],
            "artifact_count": 2,
            "fit_count": 3,
            "sample_count": 5,
            "asserted_comparison": comparison,
            "equivalence_margin": literal["equivalence_margin"],
            "references": [{"receipt_id": literal["receipt_id"], "sha256": pin}],
            "synthetic": True,
        }],
    }
    expected = {
        "protocol_sha256": pins["protocol"],
        "cohort_sha256": pins["cohort"],
        "recipe_sha256": pins["recipe"],
        "source_sha256": pins["source"],
        "loss": "mse",
        "loss_unit": "original_mse",
        "metric_sha256": pins["metric"],
    }
    return {
        "document": canonical_json(document),
        "receipts": {literal["receipt_id"]: raw},
        "expected": expected,
        "comparison": comparison,
    }
