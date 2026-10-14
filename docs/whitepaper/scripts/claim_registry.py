#!/usr/bin/env python3
"""Compile the paper's abstract claims from the registry below and refuse any drift.

Each registered claim names its estimand, the panel row or headline field it
rests on, the macros that print it, and its expected state. The compiler
re-derives every macro value with the emitter's own formatters from the pinned
panel and headline (it scores nothing itself), recomputes each verdict from
the interval rule, and checks that:

- every data macro in the abstract belongs to a registered claim;
- each claim's state still matches the evidence (a flipped verdict, a cohort
  that falls below the full-panel count, or an opened test split fails);
- a small cohort is printed with its lineage count and called separate;
- an inconclusive verdict is never worded as equivalence;
- title, abstract, README prose and CITATION.cff avoid the forbidden language.

``--check`` exits non-zero on any failure. ``--json`` prints the compiled
claims. Neither mode writes a file. check_paper.py runs the same compilation
inside the authenticated build snapshot, so the publication path enforces it;
the registry lives in this module so that snapshot carries it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parents[1]
sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import review_v2_emit as E  # noqa: E402
from research.benchmark.review_fixes.v2_panel import STRONGEST_MIN_LINEAGES  # noqa: E402

# The registry is data: estimands, states, claims and forbidden language.
REGISTRY = json.loads(r'''
{
  "format": "dope-paper-claim-registry",
  "version": 1,
  "scope": "Historical five-seed validation panel behind dope-mfs.tex. Successor campaign claims (N21) are out of scope until measured; they never enter this registry as facts.",
  "sources": {
    "panel": "research/benchmark/results/review-fixes-v2/panel.json",
    "headline": "docs/whitepaper/generated/v2-headline.json",
    "ledger": "research/benchmark/results/review-fixes-v2/scalar-cells.jsonl"
  },
  "estimands": {
    "retention_level": "Median over lineages of null-normalized TSTR/TRTR retention, with the 95% cluster, lineage and fit-seed nested bootstrap interval.",
    "paired_difference": "Hodges-Lehmann pseudo-median of paired lineage differences, DOPE minus comparator, with the 95% nested bootstrap interval. A raw-gap superiority (or inferiority) verdict requires that interval to exclude zero; otherwise the verdict is inconclusive. Equivalence is a separate rule (90% interval inside the margin) and is never inferred from an inconclusive verdict.",
    "charged_bytes": "Median over lineages of the per-lineage median charged artifact bytes over completed fits; every byte that sampling reads is charged.",
    "cap_share": "Completed DOPE fits whose charged bytes are within the 10,240-byte cap, over all completed fits.",
    "panel_size": "Distinct prepared lineages in the v2 scalar ledger.",
    "disclosure": "A required statement whose evidence is a flag in the panel, not an estimate."
  },
  "states": {
    "measured": "An estimate on validation rows; selection-exposed, as every estimate in this panel.",
    "above": "Raw-gap superiority on validation rows; interval excludes zero above it.",
    "below": "Raw-gap inferiority on validation rows; interval excludes zero below it.",
    "inconclusive": "The interval contains zero. The paper says 'cannot be separated from', which is not equivalence.",
    "null_with_reason": "The quantity is unmeasured and printed as null with the gates that keep it null.",
    "pending": "Reserved for a future pre-registered run; nothing is reported."
  },
  "claims": [
    {"id": "panel-size", "estimand": "panel_size", "kind": "count", "macro": "PreparedN", "state": "measured"},
    {"id": "dope-bytes", "estimand": "charged_bytes", "kind": "bytes", "arm": "Dope", "macro": "VDopeBytes", "state": "measured"},
    {"id": "dope-cap-share", "estimand": "cap_share", "kind": "cap_share", "arm": "Dope", "macro": "VDopeWithin", "state": "measured"},
    {"id": "dope-level-catboost-4n", "estimand": "retention_level", "kind": "level", "arm": "Dope",
     "auditor": "catboost", "size": 4, "macros": {"value": "VDopeCbFour", "lo": "VDopeCbFourLo", "hi": "VDopeCbFourHi"}, "state": "measured"},
    {"id": "dope-vs-strongest-catboost-4n", "estimand": "paired_difference", "kind": "contrast", "comparator": "strongest",
     "auditor": "catboost", "size": 4, "cohort": "full",
     "macros": {"verdict": "VVerdictStrong", "value": "VStrongDiff", "lo": "VStrongLo", "hi": "VStrongHi", "name": "VStrongestName"},
     "state": "inconclusive"},
    {"id": "strongest-bytes-ratio", "estimand": "charged_bytes", "kind": "bytes_ratio", "comparator": "strongest",
     "macros": {"bytes": "VStrongBytes", "ratio": "VStrongRatio"}, "state": "measured"},
    {"id": "dope-vs-arf-default-catboost-4n", "estimand": "paired_difference", "kind": "contrast", "arm": "Arf",
     "auditor": "catboost", "size": 4, "cohort": "full",
     "macros": {"verdict": "VVerdictArfCbFour", "value": "VDiffArfCbFour", "lo": "VLoArfCbFour", "hi": "VHiArfCbFour"},
     "state": "inconclusive"},
    {"id": "dope-vs-forest-flow-catboost-4n", "estimand": "paired_difference", "kind": "contrast",
     "arm": "Forest", "auditor": "catboost", "size": 4,
     "cohort": "small", "macros": {"verdict": "VVerdictForestCbFour", "value": "VDiffForestCbFour", "lo": "VLoForestCbFour",
     "hi": "VHiForestCbFour", "n": "VNForestCbFour"}, "state": "below"},
    {"id": "dope-vs-predictor-only-catboost-4n", "estimand": "paired_difference", "kind": "contrast", "arm": "Pred",
     "auditor": "catboost", "size": 4, "cohort": "full",
     "macros": {"verdict": "VVerdictPredCbFour", "value": "VDiffPredCbFour", "lo": "VLoPredCbFour", "hi": "VHiPredCbFour"},
     "state": "above"},
    {"id": "dope-vs-real-rows-catboost-4n", "estimand": "paired_difference", "kind": "contrast", "arm": "Boot",
     "auditor": "catboost", "size": 4, "cohort": "full",
     "macros": {"verdict": "VVerdictBootCbFour", "value": "VDiffBootCbFour", "lo": "VLoBootCbFour", "hi": "VHiBootCbFour"},
     "state": "below"},
    {"id": "selection-exposure", "estimand": "disclosure", "kind": "disclosure", "panel_flag": "official_tests_opened",
     "flag_value": false, "abstract_phrases": ["selection-exposed", "official test split is reserved"], "state": "pending"},
    {"id": "release-score", "estimand": "disclosure", "kind": "null", "panel_flag": "mfs_v3", "flag_value": null,
     "manuscript_phrases": ["The release score is null until its privacy and representation gates are measured"],
     "state": "null_with_reason"}
  ],
  "forbidden_language": {
    "applies_to": ["title", "abstract", "README prose outside the generated block", "CITATION.cff"],
    "patterns": ["intelligen", "state[- ]of[- ]the[- ]art", "\\bsota\\b", "differentially private", "privacy[- ]preserving",
                 "\\bequivalent to\\b", "\\bguarantee", "kilobyte[- ]scale", "kilobyte generator", "compression is intelligence",
                 "transferable knowledge"]
  }
}
''')
TEX = PAPER / "dope-mfs.tex"
GENERATED = PAPER / "generated"
MACRO_DEF = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")
MACRO_USE = re.compile(r"\\([A-Z][A-Za-z]+)")
STATE_OF_VERDICT = {"above": "above", "below": "below", "inseparable": "inconclusive"}


def generated_macros(root: Path = GENERATED) -> dict[str, str]:
    values = {}
    for path in sorted(root.glob("*.tex")):
        for line in path.read_text().splitlines():
            match = MACRO_DEF.match(line.strip())
            if match:
                values[match.group(1)] = match.group(2)
    return values


def abstract_of(tex: str) -> str:
    start, end = tex.index("\\begin{abstract}"), tex.index("\\end{abstract}")
    return tex[start + len("\\begin{abstract}"):end]


def title_of(tex: str) -> str:
    match = re.search(r"\\title\{\\texorpdfstring\{.*?\}\{([^{}]*)\}\}", tex, re.S)
    return match.group(1) if match else ""


def readme_prose(readme: str) -> str:
    begin, end = "<!-- BEGIN BENCHMARK -->", "<!-- END BENCHMARK -->"
    return readme.split(begin)[0] + readme.split(end)[1]


def _row(rows, code, auditor, size):
    found = [row for row in rows if E._code(row) == code and row["auditor"] == auditor and row["size"] == size]
    if len(found) != 1:
        raise ValueError(f"{len(found)} panel rows for {code} {auditor} size {size}")
    return found[0]


def _arm(headline, code):
    return next(arm for arm in headline["arms"] if arm["code"] == code)


def _strongest_code(panel):
    best = panel["strongest"]
    return E.M.arm(best["method"], best["configuration"]).code


def expected_values(claim: dict, panel: dict, headline: dict, lineages: int) -> tuple[dict[str, str], str | None]:
    """Macro values the evidence implies, and the verdict-derived state where one applies."""
    kind = claim["kind"]
    if kind == "count":
        return {claim["macro"]: str(lineages)}, None
    if kind == "bytes":
        return {claim["macro"]: E.exact_bytes(_arm(headline, claim["arm"])["bytes"]["median"])}, None
    if kind == "cap_share":
        stats = _arm(headline, claim["arm"])["bytes"]
        return {claim["macro"]: f"{stats['fits_within_cap']}/{stats['fits_ok']}"}, None
    if kind == "level":
        row = _row(panel["levels"], claim["arm"], claim["auditor"], claim["size"])
        names = claim["macros"]
        return {names["value"]: E.num(row["median"]), names["lo"]: E.num(row["lo"]), names["hi"]: E.num(row["hi"])}, None
    if kind == "bytes_ratio":
        strong = _arm(headline, _strongest_code(panel))["bytes"]["median"]
        dope = _arm(headline, "Dope")["bytes"]["median"]
        names = claim["macros"]
        return {names["bytes"]: E.human_bytes(strong), names["ratio"]: E.ratio(strong / dope)}, None
    if kind == "contrast":
        code = _strongest_code(panel) if claim.get("comparator") == "strongest" else claim["arm"]
        row = _row(panel["contrasts"], code, claim["auditor"], claim["size"])
        names = claim["macros"]
        values = {names["verdict"]: E.VERDICT[row["verdict"]], names["value"]: E.num(row["hl"]),
                  names["lo"]: E.num(row["hl_lo"]), names["hi"]: E.num(row["hl_hi"])}
        if "name" in names:
            values[names["name"]] = E.M.arm(row["method"], row["configuration"]).display
        if "n" in names:
            values[names["n"]] = str(row["n"])
        recomputed = "above" if row["hl_lo"] > 0 else ("below" if row["hl_hi"] < 0 else "inseparable")
        if recomputed != row["verdict"]:
            raise ValueError(f"{claim['id']}: panel verdict {row['verdict']} disagrees with its interval")
        return values, STATE_OF_VERDICT[row["verdict"]]
    return {}, None


def contrast_row(claim, panel):
    code = _strongest_code(panel) if claim.get("comparator") == "strongest" else claim["arm"]
    return _row(panel["contrasts"], code, claim["auditor"], claim["size"])


def compile_claims(registry: dict, panel: dict, headline: dict, macros: dict[str, str], tex: str, readme: str,
                   citation: str, lineages: int) -> tuple[list[dict], list[str]]:
    failures: list[str] = []
    compiled = []
    abstract = abstract_of(tex)
    covered = set()
    for claim in registry["claims"]:
        if claim["estimand"] not in registry["estimands"] or claim["state"] not in registry["states"]:
            failures.append(f"{claim['id']}: unknown estimand or state")
            continue
        try:
            values, derived = expected_values(claim, panel, headline, lineages)
        except (ValueError, KeyError, StopIteration) as error:
            failures.append(f"{claim['id']}: {error}")
            continue
        for name, value in values.items():
            covered.add(name)
            if macros.get(name) != value:
                failures.append(f"{claim['id']}: \\{name} is {macros.get(name)!r}, evidence gives {value!r}")
        if derived is not None and derived != claim["state"]:
            failures.append(f"{claim['id']}: registered state {claim['state']} but the evidence now says {derived}")
        if claim["kind"] == "contrast":
            row = contrast_row(claim, panel)
            small = row["n"] < STRONGEST_MIN_LINEAGES
            if small != (claim.get("cohort") == "small"):
                failures.append(f"{claim['id']}: cohort of {row['n']} lineages is registered as {claim.get('cohort')}")
            if small:
                count = "\\" + claim["macros"].get("n", "")
                sentence = next((part for part in abstract.split(". ") if "\\" + claim["macros"]["verdict"] in part), "")
                if count not in sentence or "separate" not in sentence:
                    failures.append(f"{claim['id']}: small cohort is printed without its count as a separate block")
            if row["verdict"] == "inseparable" and (row.get("equivalence") or {}).get("equivalent"):
                failures.append(f"{claim['id']}: an equivalent row must be registered under its own equivalence claim")
        if claim["kind"] == "disclosure":
            if panel.get(claim["panel_flag"]) is not claim["flag_value"]:
                failures.append(f"{claim['id']}: panel {claim['panel_flag']} is {panel.get(claim['panel_flag'])!r}")
            for phrase in claim["abstract_phrases"]:
                if phrase not in abstract:
                    failures.append(f"{claim['id']}: abstract lacks {phrase!r}")
        if claim["kind"] == "null":
            if panel.get(claim["panel_flag"]) is not claim["flag_value"]:
                failures.append(f"{claim['id']}: panel {claim['panel_flag']} is no longer null")
            for phrase in claim["manuscript_phrases"]:
                if phrase not in tex:
                    failures.append(f"{claim['id']}: manuscript lacks {phrase!r}")
        compiled.append({"id": claim["id"], "estimand": claim["estimand"], "state": claim["state"],
                         "macros": values})
    used = {name for name in MACRO_USE.findall(abstract) if name in macros}
    for name in sorted(used - covered):
        failures.append(f"abstract macro \\{name} belongs to no registered claim")
    texts = {"title": title_of(tex), "abstract": abstract, "README": readme_prose(readme)}
    if citation is not None:
        texts["CITATION.cff"] = citation
    for label, text in texts.items():
        for pattern in registry["forbidden_language"]["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                failures.append(f"{label} uses forbidden language /{pattern}/")
    return compiled, failures


def load_inputs():
    panel = json.loads(E.PANEL.read_text())
    headline = json.loads((GENERATED / "v2-headline.json").read_text())
    ledger = E.RESULTS / "review-fixes-v2" / "scalar-cells.jsonl"
    lineages = len({json.loads(line)["dataset"] for line in ledger.read_text().splitlines() if line.strip()})
    # The build snapshot carries no CITATION.cff; paper-tests checks it against the same patterns.
    citation = REPO / "CITATION.cff"
    return (REGISTRY, panel, headline, generated_macros(), TEX.read_text(), (REPO / "README.md").read_text(),
            citation.read_text() if citation.is_file() else None, lineages)


def publication_failures() -> list[str]:
    """Failures for check_paper.py, each prefixed so the build log names the registry."""
    return ["claim registry: " + failure for failure in compile_claims(*load_inputs())[1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    compiled, failures = compile_claims(*load_inputs())
    if args.json:
        print(json.dumps({"claims": compiled, "failures": failures}, indent=1))
    for failure in failures:
        print("claim registry:", failure, file=sys.stderr)
    if not args.json:
        print(f"{len(compiled)} claims compiled, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
