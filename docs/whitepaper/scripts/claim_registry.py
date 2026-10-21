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
    {"id": "dope-level-linear-4n", "estimand": "retention_level", "kind": "level", "arm": "Dope",
     "auditor": "linear", "size": 4, "macros": {"value": "VDopeLinFour", "lo": "VDopeLinFourLo", "hi": "VDopeLinFourHi"}, "state": "measured"},
    {"id": "dope-level-mlp-4n", "estimand": "retention_level", "kind": "level", "arm": "Dope",
     "auditor": "mlp", "size": 4, "macros": {"value": "VDopeMlpFour", "lo": "VDopeMlpFourLo", "hi": "VDopeMlpFourHi"}, "state": "measured"},
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
     "manuscript_phrases": ["The release score is null because its prerequisites and cleartext scan are unmeasured, and measured copy, membership, and attribute gates already fail on some cells"],
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
    if begin in readme and end in readme:
        return readme.split(begin)[0] + readme.split(end)[1]
    return re.sub(r"<!-- BEGIN TDL RESULTS -->.*?<!-- END TDL RESULTS -->", "", readme, flags=re.S)


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


# TDL publication rules extend this registry; tools/claims/check.py is only an entry point.
EVIDENCE_STATUSES = frozenset(("sealed_test", "validation", "selection_exposed_validation", "historical",
                               "specified_unrun", "unqualified", "failed"))
LEGACY_README_PROSE_SHA256 = "8d90e1fc7dd3a910a3c01b721fd29a8a0329d67ce646aeb54ddd1b788d1f279e"
PUBLIC_BANNED = (r"privacy[- ]preserving", r"\bminimum\b", r"\bbeats\s+TabSyn\b",
                 r"validated\s+across\s+(?:Kumo|Mitra|TabICL|TabPFN)",
                 r"1\.9\s*(?:KiB|KB)\s+(?:cold\s+closure|standalone)", r"\boutperforms\b",
                 r"\bHIPAA\b", r"\bde[-\s]?identified\b", r"\bMDL\b",
                 *REGISTRY["forbidden_language"]["patterns"])
# Result prose must use generated values. Structural names (TDL-v1, D-01, T90),
# URLs, citations, CLI code examples and TeX layout dimensions are not results.
PUBLIC_NUMBER = re.compile(r"(?<![\w./#-])(?:[+-]?\d+(?:[.,]\d+)*|\.\d+)(?:%|\s*(?:KiB|MiB|GiB|bytes))?(?![\w/-]|\.\d)")
DEMO_INSPECT = "examples/readme-regression/INSPECT.txt"
DEMO_NATIVE_PIN = "docs/whitepaper/receipts/native-byte-conformance-v1.json"
COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def public_prose(text: str, tex: bool = False) -> str:
    if tex:
        text = re.sub(r"(?m)(?<!\\)%.*$", "", text)
    text = re.sub(r"```(?:bash|sh|shell)\n.*?```", "", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", lambda m: m.group(1) if re.match(r"^[+-]?(?:\d|\.\d)", m.group(1)) else "", text)
    text = re.sub(r"https?://[^\s)>}]+", "", text)
    text = re.sub(r"\\(?:cite\w*|label|ref|eqref|url|input|includegraphics)(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"(?m)^\s*\\(?:documentclass|usepackage|setlength|hypersetup|geometry|definecolor).*", "", text)
    text = re.sub(r"(?m)^\s*(?:\d+[.)]\s|\(\d+\)~)", "", text)
    text = re.sub(r"\b(?:Figure|Table|Section)\s+\d+\b", "", text)
    text = re.sub(r"(\\(?:newcommand|renewcommand|providecommand)\{\\[A-Za-z]+\})\[\d+\]", r"\1", text)
    return text


def public_language_failures(text: str, label: str) -> list[str]:
    # Keep direct disclaimers such as "not HIPAA de-identification" valid.
    # Negation elsewhere must not exempt an affirmative claim in the same text.
    claims = re.sub(r"\b(?:no|not)\s+(?:HIPAA[-\s]+(?:safe|compliant|de-identification)|de[-\s]?identified|MDL)\b",
                    " ", text, flags=re.I)
    return [f"{label}: forbidden claim /{pattern}/" for pattern in PUBLIC_BANNED
            if re.search(pattern, claims, re.I)]


def lint_public_text(text: str, label: str, held_out_knowledge: bool = False) -> list[str]:
    failures = public_language_failures(text, label)
    prose = public_prose(text, tex=label.endswith(".tex"))
    if not held_out_knowledge and re.search(r"\bknowledge\b", text, re.I):
        failures.append(f"{label}: knowledge claim lacks held-out evidence")
    for match in PUBLIC_NUMBER.finditer(prose):
        failures.append(f"{label}: hand-typed result number {match.group()!r}; use generated input")
    return failures


def demo_metadata_failures(data: dict, inspected: dict, native_pin: dict) -> list[str]:
    """Check descriptive demo facts against the same sources as its result numbers."""
    text = "\n".join(data["demo_notes"]["paragraphs"])
    source = native_pin["source_data"]
    def normalize_license(value):
        return re.sub(r"[-\s]+", " ", value).casefold()
    licenses = re.findall(r"\bCC[ -]+[A-Z]+(?:[ -]+[A-Z]+)*[ -]+\d+(?:\.\d+)*\b", text, re.I)
    dois = re.findall(r"\bDOI\s+([^\s)]+)", text, re.I)
    counts = re.findall(r"\b([a-z]+|\d+)\s+positional features\b", text, re.I)
    values = {int(value) if value.isdigit() else COUNT_WORDS.index(value.casefold())
              if value.casefold() in COUNT_WORDS else None for value in counts}
    failures = []
    if {normalize_license(value) for value in licenses} != {normalize_license(source["license"])}:
        failures.append("demo_notes: license differs from native pin source_data")
    if {value.rstrip('.,;') for value in dois} != {source["doi"]}:
        failures.append("demo_notes: DOI differs from native pin source_data")
    if values != {inspected["positional_features"]}:
        failures.append("demo_notes: positional feature count differs from INSPECT.txt")
    return failures


def readme_hero_failures(readme: str) -> list[str]:
    """The generated hero figure is a README image above the first section heading."""
    first_section = re.search(r"(?m)^## ", readme)
    head = readme[:first_section.start()] if first_section else readme
    if not re.search(r"(?m)^!\[[^\]\n]+\]\(paper/generated/hero-frontier\.png\)", head):
        return ["README: the generated hero figure must appear above the first section"]
    return []


def result_failures(data: dict, auditors: dict, status: dict) -> list[str]:
    failures = []
    if data.get("citable") is not True or data.get("fixture") is True:
        failures.append("results: non-citable fixture cannot be published")
    entries = {row["id"]: row for row in auditors["entries"]}
    labels = {row["id"]: row["evidence_status"] for row in status["results"]}
    if len(labels) != len(status["results"]):
        failures.append("status: duplicate result id")
    ids = set()
    for row in data["results"]:
        name = row["id"]
        if name in ids:
            failures.append(f"{name}: duplicate result id")
        ids.add(name)
        if row.get("citable") is not True:
            failures.append(f"{name}: non-citable fixture result")
        evidence = row.get("evidence_status")
        if row.get("source", {}).get("path") == REGISTRY["sources"]["headline"] and (
                evidence != "selection_exposed_validation" or row.get("evaluation_split") != "validation" or row.get("final")):
            failures.append(f"{name}: historical source cannot be relabeled as sealed or final")
        if evidence not in EVIDENCE_STATUSES or labels.get(name) != evidence:
            failures.append(f"{name}: missing or inconsistent evidence status")
        if row.get("final") and (evidence != "sealed_test" or row.get("evaluation_split") != "sealed_test"):
            failures.append(f"{name}: final claim requires sealed-test evidence")
        if evidence == "sealed_test" and row.get("evaluation_split") != "sealed_test":
            failures.append(f"{name}: sealed evidence label disagrees with evaluation split")
        if row.get("publication_role") == "headline":
            auditor = entries.get(row.get("auditor"), {})
            if auditor.get("qualification", {}).get("state") != "qualified":
                failures.append(f"{name}: unqualified headline auditor")
            cohort = row["cohort"]
            if not cohort.get("complete") or cohort["observed_lineages"] != cohort["prepared_lineages"]:
                failures.append(f"{name}: incomplete-cohort headline claim")
            if not data.get("protocol_frozen") or not row.get("complete_budget_grid") or row.get("driver_gate") is not True:
                failures.append(f"{name}: headline requires frozen protocol, complete budget grid and driver gate")
        if row.get("pooled_cohort_ids") and set(row["pooled_cohort_ids"]) != {row["cohort"]["id"]}:
            failures.append(f"{name}: mixed-cohort claim")
        if row.get("claim_allowed") is True and (row.get("citable") is not True or evidence != "sealed_test" or not row.get("final")):
            failures.append(f"{name}: non-citable or non-sealed row cannot set claim_allowed: true")
        if row.get("tdl_bytes") is not None and (row.get("driver_gate") is not True or not row.get("complete_budget_grid")):
            failures.append(f"{name}: TDL bytes lack required gates")
    for idx, para in enumerate(data.get("demo_notes", {}).get("paragraphs", [])):
        failures.extend(public_language_failures(para, f"demo_notes[{idx}]"))
    if status.get("allowed_statuses") is None or set(status["allowed_statuses"]) != EVIDENCE_STATUSES:
        failures.append("status: evidence vocabulary differs from registry")
    return failures


def public_failures(root: Path = REPO, check_render: bool = True) -> list[str]:
    """Fail closed on public TDL claims; preserve the historical compiler above."""
    import hashlib
    from tools.claims import render
    from tools.claims.cohorts import cohort_failures
    from tools.claims.registry import validate as validate_registries
    from tools.claims.version import failures as version_failures

    data_path = root / "paper/generated/headline-results.json"
    if not data_path.is_file():
        return ["public results source is missing"]
    data = json.loads(data_path.read_text())
    status = json.loads((root / "evidence/status.json").read_text())
    auditors = json.loads((root / "research/transfer/registries/auditors.json").read_text())
    failures = result_failures(data, auditors, status)
    cohort_errors = cohort_failures(data, root)
    if cohort_errors:
        return failures + cohort_errors
    failures.extend(validate_registries(root))
    failures.extend(version_failures(root))
    for row in data["results"]:
        source = row["source"]
        path = root / source["path"]
        if source["path"] != REGISTRY["sources"]["headline"]:
            failures.append(f"{row['id']}: source is not admitted by this historical publication registry")
            continue
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            failures.append(f"{row['id']}: source must be a tracked repository input")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            failures.append(f"{row['id']}: source hash drift")
        original = next((arm for arm in json.loads(path.read_text())["arms"] if arm["code"] == source["arm"]), None)
        if original is not None and any(row.get(key) != original[key] for key in ("method", "configuration", "display")):
            failures.append(f"{row['id']}: method or configuration differs from pinned source")
        if row.get("auditor") != "catboost":
            failures.append(f"{row['id']}: historical source only supports CatBoost")
        level = row["retention"]
        if original is None or (level and {"median": level["estimate"], "lo": level["lo"], "hi": level["hi"], "n": row["cohort"]["observed_lineages"]} != original["level"]):
            failures.append(f"{row['id']}: result differs from pinned historical input")
        if original is not None and row["artifact_bytes"] is not None and row["artifact_bytes"]["median"] != original["bytes"]["median"]:
            failures.append(f"{row['id']}: bytes differ from pinned historical input")
    for pin in data.get("demo_notes", {}).get("pins", []):
        path = root / pin["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
            failures.append("historical demo source pin drift")
    if not {DEMO_INSPECT, DEMO_NATIVE_PIN}.issubset({pin["path"] for pin in data.get("demo_notes", {}).get("pins", [])}):
        failures.append("demo_notes: metadata source pin is missing")
    try:
        failures.extend(demo_metadata_failures(data, json.loads((root / DEMO_INSPECT).read_text()),
                                                json.loads((root / DEMO_NATIVE_PIN).read_text())))
    except (OSError, KeyError, ValueError, TypeError):
        failures.append("demo_notes: source metadata is missing or invalid")
    outputs = render.expected_outputs(root) if check_render else render.text_outputs(data, status)
    if check_render:
        for name, expected in outputs.items():
            path = root / name
            if not path.is_file() or path.read_bytes() != expected:
                failures.append(f"{name}: generated output or figure hash drift")
    # The equal-byte study renders its own README block and paper macros from
    # committed study outputs; every one of them must equal a fresh rendering.
    import equal_bytes_text
    # Public surfaces only; the renderer's own check covers its research outputs.
    equal_outputs = {name: text for name, text in equal_bytes_text.text_outputs().items()
                     if name.startswith("paper/")}
    for name, expected in equal_outputs.items():
        path = root / name
        if not path.is_file() or path.read_text() != expected:
            failures.append(f"{name}: equal-byte generated output drift")
    readme = (root / "README.md").read_text()
    # Existing v2 BENCHMARK is already compiled by the historical registry.
    # The L3 landing page uses the new marker. While lanes integrate, both are admitted.
    if render.BEGIN in readme:
        failures.extend(readme_hero_failures(readme))
        if readme.count(render.BEGIN) != 1 or readme.count(render.END) != 1 or readme.index(render.BEGIN) > readme.index(render.END):
            failures.append("README: generated result markers must be unique and ordered")
        else:
            block = readme[readme.index(render.BEGIN):readme.index(render.END) + len(render.END)]
            expected = outputs["paper/generated/readme-result-card.md"].decode().strip()
            if block != expected:
                failures.append("README: generated result card drift")
            prose = readme.replace(block, "")
            if "<!-- BEGIN BENCHMARK -->" in prose:
                import readme_benchmark
                historical = readme_benchmark.readme_block(prose)
                original = json.loads((root / REGISTRY["sources"]["headline"]).read_text())
                if historical != readme_benchmark.expected_readme_block(original):
                    failures.append("README: historical generated benchmark drift")
                prose = prose.replace(historical, "")
            if render.DEMO_BEGIN in prose:
                if prose.count(render.DEMO_BEGIN) != 1 or prose.count(render.DEMO_END) != 1 or prose.index(render.DEMO_BEGIN) > prose.index(render.DEMO_END):
                    failures.append("README: demo markers must be unique and ordered")
                else:
                    demo = prose[prose.index(render.DEMO_BEGIN):prose.index(render.DEMO_END) + len(render.DEMO_END)]
                    if demo != outputs["paper/generated/readme-demo-notes.md"].decode().strip():
                        failures.append("README: generated historical demo notes drift")
                    prose = prose.replace(demo, "")
            begin, end = equal_bytes_text.README_BEGIN, equal_bytes_text.README_END
            if begin in prose:
                if prose.count(begin) != 1 or prose.count(end) != 1 or prose.index(begin) > prose.index(end):
                    failures.append("README: equal-byte markers must be unique and ordered")
                else:
                    block = prose[prose.index(begin):prose.index(end) + len(end)]
                    if block != equal_outputs["paper/generated/readme-equal-bytes.md"].strip():
                        failures.append("README: generated equal-byte block drift")
                    prose = prose.replace(block, "")
            failures.extend(lint_public_text(prose, "README"))
    elif "<!-- BEGIN BENCHMARK -->" not in readme:
        failures.append("README: generated result card markers are missing")
    elif hashlib.sha256(readme_prose(readme).encode()).hexdigest() != LEGACY_README_PROSE_SHA256:
        failures.append("README: changed prose must adopt the generated TDL result card")
    verified_outputs = set(outputs) | {name for name in equal_outputs if name.endswith(".tex")}
    contract_generator = root / "paper/scripts/contract_macros.py"
    if contract_generator.is_file():
        import runpy
        contract_name = "paper/generated/contract-macros.tex"
        try:
            expected = runpy.run_path(str(contract_generator))["render"](
                json.loads((root / "contracts/tdl-contract-v1.json").read_text())).encode()
            if not (root / contract_name).is_file() or (root / contract_name).read_bytes() != expected:
                failures.append(f"{contract_name}: generated contract macro drift")
            else:
                verified_outputs.add(contract_name)
        except (OSError, KeyError, ValueError, TypeError):
            failures.append(f"{contract_name}: generator or contract is missing or invalid")
    for path in sorted((root / "paper").rglob("*.tex")):
        relative = path.relative_to(root)
        # Historical archives are exempt; renderer outputs are authenticated above.
        if relative.is_relative_to("paper/archive") or relative.as_posix() in verified_outputs:
            continue
        failures.extend(lint_public_text(path.read_text(), str(relative)))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--public-only", action="store_true", help="Check the new public artifacts without the archived manuscript")
    args = parser.parse_args()
    compiled, failures = ([], []) if args.public_only else compile_claims(*load_inputs())
    failures.extend(public_failures())
    if args.json:
        print(json.dumps({"claims": compiled, "failures": failures}, indent=1))
    for failure in failures:
        print("claim registry:", failure, file=sys.stderr)
    if not args.json:
        print(f"{len(compiled)} claims compiled, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
