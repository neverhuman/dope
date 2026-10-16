"""Derive historical public cohort fields from the v2 source ledger."""
import json
from pathlib import Path

from research.benchmark.review_fixes.v2_panel import STRONGEST_MIN_LINEAGES

ROOT = Path(__file__).resolve().parents[2]
LEDGER = "research/benchmark/results/review-fixes-v2/scalar-cells.jsonl"
SOURCE = "docs/whitepaper/generated/v2-headline.json"


def expected_cohorts(root: Path = ROOT) -> dict:
    # This is the same distinct-dataset count checked against PreparedN by the
    # historical claim compiler; it does not count fits, cells or observations.
    with (root / LEDGER).open() as ledger:
        prepared = len({json.loads(line)["dataset"] for line in ledger if line.strip()})
    if not prepared:
        raise ValueError("v2 cohort ledger is empty")
    source = json.loads((root / SOURCE).read_text())
    expected = {}
    for arm in source["arms"]:
        observed = arm["level"]["n"] if arm["level"] is not None else 0
        if type(observed) is not int or not 0 <= observed <= prepared:
            raise ValueError("v2 cohort source has an invalid lineage count")
        expected[arm["code"]] = {"prepared_lineages": prepared,
                                 "complete": observed >= STRONGEST_MIN_LINEAGES}
    return expected


def cohort_failures(data: dict, root: Path = ROOT) -> list[str]:
    try:
        expected = expected_cohorts(root)
    except (OSError, ValueError, KeyError, TypeError):
        return ["cohort source ledger or headline is missing or invalid"]
    failures = []
    for row in data["results"]:
        if row.get("source", {}).get("path") != SOURCE:
            continue  # The publication registry separately refuses other sources.
        source = expected.get(row["source"].get("arm"))
        if source is None:
            failures.append(f"{row['id']}: cohort source arm is missing")
            continue
        cohort = row.get("cohort", {})
        for field, kind, label in (("prepared_lineages", int, "v2 ledger"),
                                   ("complete", bool, "v2 lineage rule")):
            value = cohort.get(field)
            if type(value) is not kind or value != source[field]:
                failures.append(f"{row['id']}: cohort {field} differs from {label}")
    return failures


def refresh_cohorts(data: dict, root: Path = ROOT) -> None:
    expected = expected_cohorts(root)
    for row in data["results"]:
        if row.get("source", {}).get("path") == SOURCE:
            row["cohort"].update(expected[row["source"]["arm"]])
