#!/usr/bin/env python3
"""Public claim ledger and language firewall, including CR-04, CR-10 and CR-13."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "docs/whitepaper/scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from tools.claims import registry  # noqa: E402

NONSIGNIFICANT = re.compile(
    r"\b(?:no\s+significant\s+difference|non[-\s]?significant\s+(?:difference|comparison)|"
    r"not\s+statistically\s+significant|statistically\s+indistinguishable|"
    r"interval\s+contains\s+zero)\b|\bp\s*>\s*0\.05\b", re.I)
EQUIVALENCE = re.compile(r"\b(?:equivalen(?:ce|t)|parity|on\s+par|matches|ties)\b", re.I)
# Remove only the directly negated conclusion, retaining later affirmative clauses.
NEGATIVE_CONCLUSION = re.compile(
    r"\b(?:(?:does|do|did|can|could|should|would)\s+not|cannot|can't|never)\s+"
    r"(?:establish|prove|show|imply|demonstrate|claim|mean)\s+"
    r"(?:(?:statistical|practical)\s+)?(?:equivalen(?:ce|t)|parity|on\s+par|matches|ties)\b|"
    r"\b(?:not|no|without)\s+(?:evidence\s+of\s+)?(?:equivalen(?:ce|t)|parity|on\s+par|matches|ties)\b",
    re.I)
BADGE = re.compile(
    r"\b(?:HIPAA(?:[-\s]+(?:safe|compliant|de[-\s]?identification))?|anonymous|anonymized|"
    r"de[-\s]?identified|differentially\s+private|formal[-\s]+DP|safe[-\s]+shar(?:e|ing))\b", re.I)
NEGATIVE_BADGE = re.compile(
    r"\b(?:no|not|never|without)\s+(?:(?:a|an|any)\s+)?"
    r"(?:claim\s+of\s+|guarantee\s+of\s+)?(?:HIPAA[-\s]+(?:safe|compliant|de[-\s]?identification)|"
    r"HIPAA(?=\s+claim)|anonymous|anonymized|de[-\s]?identified|differentially\s+private|"
    r"formal[-\s]+DP|safe[-\s]+shar(?:e|ing))\b", re.I)
PRIVATE_PATH = re.compile(
    r"(?:/(?:home|Users|root|mnt|srv|private)/|\b(?:s3|gs|az|ssh|sftp)://|"
    r"\\\\[A-Za-z0-9_.-]+\\|\b(?:localhost|[A-Za-z0-9.-]+\.(?:internal|local))(?=[:/\s]|$)|"
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b)")
RAW_ECHO = re.compile(
    r"\b(?:raw[_ -]?(?:row|column|value|header)|source[_ -]?(?:column|value|header)|"
    r"column[_ -]?names?|(?:source[_ -]?)?(?:minimum|maximum|extrema))s?[\"']?\s*[:=]|"
    r"\bCANARY_[\w-]+|sentinel-marker-\d+|\b\d{3}-\d{2}-\d{4}\b|"
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.I)


def lint_s1_r074_no_significant_difference_equivalence(
        text: str, source: str, equivalence_procedure: dict | None = None,
        receipt_sha256s: list[str] | None = None) -> list[str]:
    if registry.equivalence_passed(equivalence_procedure) and registry.receipt_digests(receipt_sha256s):
        return []
    # Paragraph scope also catches an inference made in the sentence after the p-value.
    for paragraph in re.split(r"\n\s*\n", text):
        if NONSIGNIFICANT.search(paragraph) and EQUIVALENCE.search(NEGATIVE_CONCLUSION.sub("", paragraph)):
            return [f"{source}: CR-13/S1-R074: non-significance does not establish equivalence"]
    return []


def lint_cr04_redaction_and_unbacked_badges(text: str, source: str) -> list[str]:
    failures = []
    if BADGE.search(NEGATIVE_BADGE.sub("", text)):
        failures.append(f"{source}: CR-04: privacy or sharing badge lacks release authority")
    if PRIVATE_PATH.search(text):
        failures.append(f"{source}: CR-04: private host or bucket path")
    if RAW_ECHO.search(text):
        failures.append(f"{source}: CR-04: raw source field or value echo")
    return failures


def overlap_language_failures(text: str, source: str, uncertain_models: set[str]) -> list[str]:
    failures = []
    for paragraph in re.split(r"\n\s*\n", text):
        if not any(re.search(name, paragraph, re.I) for name in uncertain_models):
            continue
        affirmative = re.sub(
            r"\b(?:not|no|never)\s+(?:a\s+)?(?:contamination[- ]free|zero[- ]overlap)", "", paragraph, flags=re.I)
        if re.search(r"contamination[- ]free|zero[- ]overlap|no\s+pretraining\s+overlap", affirmative, re.I):
            failures.append(f"{source}: CR-10: contamination-free language contradicts overlap disclosure")
        if (re.search(r"unseen[- ]source|zero[- ]shot", paragraph, re.I) and
                re.search(r"\bwin(?:s)?\b|\bbeats\b|\boutperforms\b", paragraph, re.I) and
                not re.search(r"pretraining\s+overlap\s+(?:is\s+)?(?:unknown|detected)|"
                              r"(?:unknown|overlapping)\s+pretraining", paragraph, re.I)):
            failures.append(f"{source}: CR-10: unseen-source or zero-shot win requires an explicit overlap caveat")
    return failures


def lint_cr10_pretraining_overlap_caveats(data: dict, auditors: dict,
                                         text_surfaces: dict[str, str]) -> list[str]:
    failures, uncertain_models, observed_models, result_uncertainty = [], set(), set(), set()

    def names(entry):
        identity = ' '.join(str(entry.get(key, '')) for key in ('id', 'auditor', 'display'))
        return {name for name in ('kumo', 'tabicl', 'mitra', 'tabpfn') if re.search(name, identity, re.I)}

    entries = {row["id"]: row for row in auditors.get("entries", [])}
    for entry in entries.values():
        if registry.foundation_model(entry):
            try:
                if registry.pretraining_disclosure(entry)["state"] in ("unknown", "overlapping"):
                    uncertain_models.update(names(entry))
            except ValueError as error:
                failures.append(str(error))
    for row in data.get("results", []):
        entry = entries.get(row.get("auditor"), {})
        if not (registry.foundation_model(row) or registry.foundation_model(entry)):
            continue
        try:
            # A published result requires its own checkpoint disclosure, even for an unrun registry stub.
            if "pretraining_corpus_disclosure" not in row or not row.get("checkpoint_sha256"):
                raise ValueError("CR-10: foundation result requires per-checkpoint disclosure")
            disclosure = registry.pretraining_disclosure({**entry, **row})
            observed_models.update(names(row))
            if disclosure["state"] in ("unknown", "overlapping"):
                result_uncertainty.update(names(row))
            if disclosure["state"] in ("unknown", "overlapping") and row.get("claim_allowed") is True:
                failures.append("CR-10: uncertain pretraining overlap cannot grant an unqualified claim")
        except ValueError as error:
            failures.append(str(error))
    # Result checkpoints supersede unrun registry placeholders for the same model.
    uncertain_models = (uncertain_models - observed_models) | result_uncertainty
    for source, text in text_surfaces.items():
        failures.extend(overlap_language_failures(text, source, uncertain_models))
    return failures


def publication_failures(data: dict, status: dict, root: Path = ROOT) -> list[str]:
    """New checks also run through the historical renderer's existing publication gate."""
    surfaces = {}
    # The landing page, the historical results page and the citation file are public surfaces.
    for name in ("README.md", "HISTORY.md", "CITATION.cff"):
        if (root / name).is_file():
            surfaces[name] = (root / name).read_text()
    for index, path in enumerate(sorted((root / "paper").rglob("*.tex"))):
        relative = path.relative_to(root)
        if not relative.is_relative_to("paper/archive"):
            surfaces[f"paper surface {index}"] = path.read_text()
    for index, text in enumerate(data.get("demo_notes", {}).get("paragraphs", [])):
        surfaces[f"demo paragraph {index}"] = text
    # Include structured result/status text too, including arbitrary nested sidecars.
    surfaces["result metadata"] = json.dumps(data)
    surfaces["status metadata"] = json.dumps(status)
    failures = registry.validate_result_claims(data)
    failures.extend(lint_cr10_pretraining_overlap_caveats(data, json.loads(
        (root / "research/transfer/registries/auditors.json").read_text()), surfaces))
    for source, text in surfaces.items():
        if source != "result metadata":
            failures.extend(lint_s1_r074_no_significant_difference_equivalence(text, source))
        failures.extend(lint_cr04_redaction_and_unbacked_badges(text, source))
    for row in data.get("results", []):
        failures.extend(lint_s1_r074_no_significant_difference_equivalence(
            json.dumps(row), "result metadata", row.get("equivalence_procedure"), row.get("receipt_sha256s")))
    return failures


def public_failures(root: Path = ROOT, check_render: bool = True) -> list[str]:
    import claim_registry
    from tools.claims import render
    try:
        data = json.loads((root / "paper/generated/headline-results.json").read_text())
        status = json.loads((root / "evidence/status.json").read_text())
        failures = publication_failures(data, status, root)
        failures.extend(registry.validate_claim_ledger(render.claim_ledger(data, status, root=root), root,
                                                       check_sources=True))
        if failures:
            return failures
        # The historical registry quotes input in some errors. Keep its private diagnostic out of public output.
        failures.extend("public surfaces: historical firewall violation" for _ in
                        claim_registry.public_failures(root, check_render=check_render))
        return failures
    except (OSError, ValueError, KeyError, TypeError):
        return ["public claims: invalid or unavailable publication input"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-ledger", type=Path, help="Also validate a claim-v1 JSON ledger")
    args = parser.parse_args()
    failures = public_failures()
    if args.claim_ledger:
        try:
            failures.extend(registry.validate_claim_ledger(json.loads(args.claim_ledger.read_text()), ROOT))
        except (OSError, ValueError):
            failures.append("claim ledger: invalid or unavailable JSON")
    print("\n".join(failures) if failures else "Public claims firewall passed")
    return int(bool(failures))

if __name__ == "__main__":
    raise SystemExit(main())
