"""Explicit fixture/evidence admission; qualification comes from locked receipts."""
from research.transfer.tdl.common import MetricError


def require_citable(record):
    if type(record) is not dict or record.get("citable") is not True:
        raise MetricError("noncitable_fixture")
    if record.get("evidence_status") not in {"sealed_test", "validation", "selection_exposed_validation", "historical"}:
        raise MetricError("uncitable_evidence_status")
    return record
