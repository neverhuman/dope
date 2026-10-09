"""Resolve published sample CSV paths. Does not open the files."""

from __future__ import annotations

from pathlib import Path


def resolve_sample_csv(cell: dict) -> str | None:
    """Return a candidate CSV path. The caller checks that the file exists.

    DOPE cells store metric_receipt.metric_path and have no sample_evidence.
    Density comparators store sample_evidence.metric_path. The CSV replaces
    the suffix .metric.json with .csv. ARF stores sample_evidence.sample_path.
    CTGAN and TVAE store sample_evidence.path when that path names a csv.
    Forest stores receipt.path to a receipt.json. The non-repeat sample is the
    sibling sample.csv listed beside that receipt. A missing candidate is
    skipped. No other path is constructed.
    """
    if cell.get("method") == "DOPE":
        metric = (cell.get("metric_receipt") or {}).get("metric_path")
        if isinstance(metric, str) and metric.endswith(".metric.json"):
            return metric[: -len(".metric.json")] + ".csv"
        return None
    evidence = cell.get("sample_evidence") or {}
    if isinstance(evidence, dict):
        sample = evidence.get("sample_path")
        if isinstance(sample, str) and sample.endswith(".csv"):
            return sample
        metric = evidence.get("metric_path")
        if isinstance(metric, str) and metric.endswith(".metric.json"):
            return metric[: -len(".metric.json")] + ".csv"
        path = evidence.get("path")
        if isinstance(path, str) and path.endswith(".csv"):
            return path
    receipt = cell.get("receipt") or {}
    receipt_path = receipt.get("path") if isinstance(receipt, dict) else None
    if isinstance(receipt_path, str) and receipt_path.endswith("/receipt.json"):
        return str(Path(receipt_path).with_name("sample.csv"))
    return None
