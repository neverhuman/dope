#!/usr/bin/env python3
"""Turn B-lane matched-population publications into manuscript rows.

The receipt schema is the ARF matched publication from PR 123
(``publish_arf_population_matched.py``): ``version`` 1, ``cells``, and
``paired_descriptive``. Each method has one JSON file under
``research/benchmark/results/``. A missing file becomes the words
``not measured``. This module only reads those files. It does not fit,
sample, or write a receipt.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "benchmark" / "results"
OUT = REPO / "docs" / "whitepaper" / "generated" / "publication-rows.tex"

# Display name, results filename, cell method, format string.
PUBLICATIONS = (
    (
        "TabDDPM",
        "tabddpm-matched-population-validation.json",
        "TabDDPM",
        "dope-complete-original-tabddpm-matched-population-validation",
    ),
    (
        "TabSyn",
        "tabsyn-matched-population-validation.json",
        "TabSyn",
        "dope-complete-original-tabsyn-matched-population-validation",
    ),
    (
        "Forest-Flow",
        "forest-flow-matched-population-validation.json",
        "ForestDiffusion/Forest-Flow",
        "dope-complete-original-forest-flow-matched-population-validation",
    ),
    (
        "ARF",
        "arf-matched-population-validation.json",
        "ARF",
        "dope-complete-original-arf-matched-population-validation",
    ),
)

# The manuscript's displayed generator. Other profiles in a receipt are not rows.
DISPLAYED_CONFIGURATION = "features12_steps2048"

CELL_KEYS = (
    "dataset",
    "method",
    "configuration",
    "size_multiplier",
    "status",
    "charged_artifact_bytes",
    "counts_as_dope_win",
    "mfs_v2",
    "ptf_v1",
    "release_safe",
    "superiority",
)
PAIR_KEYS = (
    "configuration",
    "reference_method",
    "reference_configuration",
    "size_multiplier",
    "auditor",
    "paired_complete_informative_lineages",
    "dope_median_retention",
    "reference_median_retention",
    "median_paired_difference",
    "superiority",
)
NULL_CLAIMS = ("mfs_v2", "ptf_v1", "release_safe", "superiority")
NOT_MEASURED = "not measured"


def _spec(display):
    for spec in PUBLICATIONS:
        if spec[0] == display:
            return spec
    raise KeyError(display)


def locate(results_dir, filename):
    """Return a receipt inside ``results_dir``, or None when it is absent."""
    if Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError(f"receipt filename escapes the results directory: {filename}")
    directory = Path(results_dir)
    path = directory / filename
    if path.is_symlink():
        resolved = path.resolve()
        if resolved.parent != directory.resolve():
            raise ValueError(f"receipt symlink leaves the results directory: {filename}")
    if not path.is_file():
        return None
    return path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not a finite number")
    return number


def _sig3(value):
    if value is None:
        return "null"
    number = _number(value, "paired value")
    if abs(number) >= 0.01:
        return f"{number:.3f}"
    return f"{number:.3g}"


def _tex(text):
    return str(text).replace("_", r"\_")


def load_publication(path, method, format_name):
    """Validate one receipt. ``path`` is read and not written."""
    document = json.loads(Path(path).read_text())
    _require(document.get("format") == format_name, "publication format differs")
    _require(document.get("version") == 1, "publication version differs")
    cells = document.get("cells")
    pairs = document.get("paired_descriptive")
    _require(isinstance(cells, list) and cells, "publication cells are missing")
    _require(isinstance(pairs, list) and pairs, "publication pairs are missing")
    for cell in cells:
        _require(isinstance(cell, dict), "publication cell is not an object")
        for key in CELL_KEYS:
            _require(key in cell, f"publication cell lacks {key}")
        _require(cell["method"] == method, "publication cell method differs")
        _require(isinstance(cell["dataset"], str) and cell["dataset"], "publication dataset differs")
        _require(isinstance(cell["configuration"], str) and cell["configuration"], "publication configuration differs")
        _require(type(cell["size_multiplier"]) is int, "publication size differs")
        _require(isinstance(cell["status"], str) and cell["status"], "publication status differs")
        _require(type(cell["charged_artifact_bytes"]) is int and cell["charged_artifact_bytes"] > 0,
                 "publication charge differs")
        _require(cell["counts_as_dope_win"] is False, "publication counts a DOPE win")
        for key in NULL_CLAIMS:
            _require(cell[key] is None, f"publication {key} is not null")
    chosen = []
    for row in pairs:
        _require(isinstance(row, dict), "publication pair is not an object")
        for key in PAIR_KEYS:
            _require(key in row, f"publication pair lacks {key}")
        _require(row["reference_method"] == method, "publication pair method differs")
        _require(row["superiority"] is None, "publication superiority is not null")
        _require(type(row["size_multiplier"]) is int, "publication pair size differs")
        count = row["paired_complete_informative_lineages"]
        _require(type(count) is int and count >= 0, "publication lineage count differs")
        if (
            row["size_multiplier"] == 4
            and row["auditor"] == "catboost"
            and row["configuration"] == DISPLAYED_CONFIGURATION
        ):
            chosen.append(row)
    _require(chosen, "publication has no displayed size-4 CatBoost pair")
    return chosen


def _phrase_from_pairs(pairs):
    for row in pairs:
        if row["reference_configuration"] == "author_default":
            return _sig3(row["reference_median_retention"])
    return _sig3(pairs[0]["reference_median_retention"])


def phrase(display, results_dir=RESULTS):
    """CatBoost receipt median, or ``not measured`` when the file is absent."""
    _display, filename, method, format_name = _spec(display)
    path = locate(results_dir, filename)
    if path is None:
        return NOT_MEASURED
    return _phrase_from_pairs(load_publication(path, method, format_name))


def _row(display, configuration, lineages, dope, receipt, difference):
    label = display if configuration is None else f"{display}, {_tex(configuration)}"
    return f"{label} & {lineages} & {dope} & {receipt} & {difference} \\\\"


def rows_for(display, results_dir=RESULTS):
    """One LaTeX row per size-4 CatBoost pair, or one not-measured row."""
    _display, filename, method, format_name = _spec(display)
    path = locate(results_dir, filename)
    if path is None:
        return [_row(display, None, NOT_MEASURED, NOT_MEASURED, NOT_MEASURED, NOT_MEASURED)]
    lines = []
    for row in load_publication(path, method, format_name):
        lines.append(_row(
            display,
            row["reference_configuration"],
            str(row["paired_complete_informative_lineages"]),
            _sig3(row["dope_median_retention"]),
            _sig3(row["reference_median_retention"]),
            _sig3(row["median_paired_difference"]),
        ))
    return lines


def render(results_dir=RESULTS):
    body = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Publication & Lineages & DOPE & Receipt & Difference \\",
        r"\midrule",
    ]
    for display, _filename, _method, _format_name in PUBLICATIONS:
        body.extend(rows_for(display, results_dir))
    body.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(body)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(RESULTS))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
