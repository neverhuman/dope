#!/usr/bin/env python3
"""Cross-file count checks over parsed generated artifacts.

Pure functions over dicts. A row key is (method, configuration, size,
auditor). Shapes:

  level        rows with the key fields and ``n``: a printed retention level
  informative  rows with the key fields and ``n_informative``
  paired       rows with the key fields and ``n`` for DOPE minus that arm;
               ``left_method`` and ``left_configuration`` override the DOPE side
  printed      rows ``{"file", "quantity", "n", "cohort"}``; ``cohort`` is the
               declared cohort label or None

Every check returns readable violations. load_generated_artifacts reads the
committed v2 panel and the headline JSON emitted from it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
GENERATED = PAPER / "generated"
V2_PANEL = PAPER.parents[1] / "research" / "benchmark" / "results" / "review-fixes-v2" / "panel.json"
REFERENCE = ("DOPE", "features12_steps2048")
V2_REFERENCE = ("DOPE", "headline")
CONTROLS = ("predictor_only", "real_bootstrap_4n")


def row_key(row, method=None, configuration=None) -> tuple:
    return (
        row["method"] if method is None else method,
        row["configuration"] if configuration is None else configuration,
        int(row["size"]),
        row["auditor"],
    )


def _label(key) -> str:
    method, configuration, size, auditor = key
    return f"{method} {configuration} {size}n {auditor}"


def informative_index(rows, field: str = "n_informative") -> dict[tuple, int]:
    """Key to informative count. Two rows that disagree on one key are an error."""
    index = {}
    for row in rows:
        key, count = row_key(row), int(row[field])
        if index.get(key, count) != count:
            raise ValueError(f"two informative counts for {_label(key)}")
        index[key] = count
    return index


def paired_n_violations(paired_rows, informative: dict, reference=REFERENCE) -> list[str]:
    """A paired n cannot exceed the informative n of either side."""
    failures = []
    for row in paired_rows:
        left = row_key(row, row.get("left_method", reference[0]), row.get("left_configuration", reference[1]))
        right = row_key(row)
        n = int(row["n"])
        for side, key in (("left", left), ("right", right)):
            if key not in informative:
                failures.append(f"paired {_label(right)}: no informative count for the {side} side {_label(key)}")
            elif n > informative[key]:
                failures.append(
                    f"paired {_label(right)}: n={n} exceeds the {side} informative n={informative[key]} ({_label(key)})"
                )
    return failures


def level_n_violations(level_rows, informative: dict) -> list[str]:
    """A printed level n equals the informative n of the same key."""
    failures = []
    for row in level_rows:
        key, n = row_key(row), int(row["n"])
        if key not in informative:
            failures.append(f"level {_label(key)}: no informative count")
        elif n != informative[key]:
            failures.append(f"level {_label(key)}: n={n} but informative n={informative[key]}")
    return failures


def cohort_violations(printed_rows) -> list[str]:
    """One quantity printed with two counts needs two distinct declared cohort labels."""
    groups: dict = {}
    for row in printed_rows:
        groups.setdefault(row["quantity"], []).append(row)
    failures = []
    for quantity, rows in groups.items():
        for index, first in enumerate(rows):
            for second in rows[index + 1:]:
                if int(first["n"]) == int(second["n"]):
                    continue
                labels = (first.get("cohort"), second.get("cohort"))
                if all(labels) and labels[0] != labels[1]:
                    continue
                failures.append(
                    f"{quantity}: n={first['n']} in {first['file']} and n={second['n']} in "
                    f"{second['file']} without two distinct cohort labels"
                )
    return failures


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def strongest_comparator(rows, min_lineages: int = 90, *, auditor: str = "catboost", size: int = 4,
                         exclude=(REFERENCE[0],)):
    """The arm with the highest median among arms with at least ``min_lineages`` paired lineages.

    Each row is one arm at one size and auditor, with its own retention
    ``median`` and its paired lineage count in ``paired_n`` (or ``n``). An
    equal median is broken by method, then configuration. None when no arm
    qualifies.
    """
    eligible = []
    for row in rows:
        paired = row.get("paired_n", row.get("n"))
        if (row.get("auditor") != auditor or int(row.get("size", -1)) != size
                or row.get("method") in exclude or paired is None or int(paired) < min_lineages
                or not _finite(row.get("median"))):
            continue
        eligible.append(row)
    if not eligible:
        return None
    return min(eligible, key=lambda row: (-float(row["median"]), str(row["method"]), str(row["configuration"])))


def check_artifacts(artifacts: dict) -> list[str]:
    """Run every check whose inputs are present."""
    failures = []
    informative = informative_index(artifacts.get("informative") or [])
    if artifacts.get("paired"):
        failures += paired_n_violations(artifacts["paired"], informative)
    if artifacts.get("level"):
        failures += level_n_violations(artifacts["level"], informative)
    if artifacts.get("printed"):
        failures += cohort_violations(artifacts["printed"])
    return failures


def load_generated_artifacts(panel_path: Path = V2_PANEL, generated: Path = GENERATED) -> dict | None:
    """The v2 panel's levels and contrasts in the shapes above, or None before the panel exists."""
    if not panel_path.is_file():
        return None
    panel = json.loads(panel_path.read_text())
    keys = ("method", "configuration", "size", "auditor")
    levels = [{**{key: row[key] for key in keys}, "n": row["n"], "median": row.get("median")}
              for row in panel["levels"]]
    paired = [{**{key: row[key] for key in keys}, "n": row["n"], "left_method": V2_REFERENCE[0],
               "left_configuration": V2_REFERENCE[1]} for row in panel["contrasts"]]
    headline_path = generated / "v2-headline.json"
    return {"informative": [{**row, "n_informative": row["n"]} for row in levels], "paired": paired,
            "level": levels, "strongest": panel.get("strongest"),
            "headline": json.loads(headline_path.read_text()) if headline_path.is_file() else None}


def strongest_violations(artifacts: dict) -> list[str]:
    """The panel's strongest comparator is the argmax the rule gives, and the headline JSON repeats it."""
    paired_n = {(row["method"], row["configuration"]): row["n"] for row in artifacts["paired"]
                if row["size"] == 4 and row["auditor"] == "catboost"}
    rows = [{**row, "paired_n": paired_n.get((row["method"], row["configuration"]))} for row in artifacts["level"]]
    best = strongest_comparator(rows, exclude=(V2_REFERENCE[0], *CONTROLS))
    stated = artifacts.get("strongest")
    expected = None if best is None else (best["method"], best["configuration"])
    found = None if stated is None else (stated["method"], stated["configuration"])
    failures = []
    if expected != found:
        failures.append(f"strongest comparator is {found} but the rule gives {expected}")
    headline = artifacts.get("headline")
    if headline is not None and headline.get("strongest") != stated:
        failures.append("v2-headline.json names a different strongest comparator than the panel")
    return failures


def check_v2() -> list[str]:
    """Count and strongest-comparator checks over the committed v2 panel; empty before it exists."""
    artifacts = load_generated_artifacts()
    if artifacts is None:
        return []
    failures = check_artifacts({key: artifacts[key] for key in ("informative", "paired")})
    return failures + strongest_violations(artifacts)
