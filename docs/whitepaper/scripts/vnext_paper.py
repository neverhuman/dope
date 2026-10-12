"""Check the falsifiable vNext draft.

This script does not emit measurements. Headline quantities stay as
``\\TBD{claim-id}`` placeholders until a later claim-binding packet supplies
receipts. Venue selection stays closed.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


class PaperDraftError(ValueError):
    """The draft asserted an unbound result or left a required claim out."""


@dataclass(frozen=True)
class ClaimLadder:
    claim_id: str
    kind: str
    statement: str


@dataclass(frozen=True)
class ReproductionLevel:
    name: str
    heading: str
    recipe: str
    measured: bool = False


class VenueWrapper:
    """Venue choice waits for a recorded decision. This draft uses article."""

    venue = "undecided"

    def select(self, venue: str) -> None:
        raise PaperDraftError("venue stays undecided until a recorded decision")


CLAIMS: tuple[ClaimLadder, ...] = (
    ClaimLadder(
        "rate-risk-frontier",
        "headline",
        "Original-unit loss of the frozen artifact on withheld numeric-regression cells, under the declared byte constraint.",
    ),
    ClaimLadder(
        "cross-source-transfer",
        "headline",
        "Change in original-unit loss on a source family withheld from selection.",
    ),
    ClaimLadder(
        "cross-learner-transfer",
        "headline",
        "Change in original-unit loss on a learner family withheld from selection.",
    ),
    ClaimLadder(
        "standard-transfer-bound",
        "assumption",
        "Width of the assumed transfer inequality that does not depend on the new selection rule.",
    ),
    ClaimLadder(
        "algorithmic-hypothesis",
        "hypothesis",
        "Change in withheld original-unit loss attributed to development-only decoded selection and byte allocation.",
    ),
    ClaimLadder(
        "algorithmic-hypothesis-margin",
        "hypothesis",
        "Predeclared margin for the algorithmic hypothesis. Unbound in this draft.",
    ),
    ClaimLadder(
        "direct-predictor-control",
        "control",
        "Withheld original-unit loss of the generator minus the shipped development predictor.",
    ),
    ClaimLadder(
        "coreset-control",
        "control",
        "Withheld original-unit loss of the generator minus a real-row coreset at the same byte constraint.",
    ),
    ClaimLadder(
        "distillation-control",
        "control",
        "Withheld original-unit loss of the generator minus distilled prototypes at the same byte constraint.",
    ),
    ClaimLadder(
        "circuit-control",
        "control",
        "Withheld original-unit loss of the generator minus the symbolic dependence program at the same byte constraint.",
    ),
    ClaimLadder(
        "unfavorable-baseline",
        "limitation",
        "Original-unit loss difference against the strongest admitted baseline, retained when that baseline is ahead.",
    ),
    ClaimLadder(
        "cap-failure-mass",
        "limitation",
        "Share of artifacts that miss the byte constraint. Failures stay in the denominator.",
    ),
    ClaimLadder(
        "tail-harm",
        "limitation",
        "Upper tail of original-unit loss harm. The tail stays visible.",
    ),
    ClaimLadder(
        "rights-unfrozen",
        "limitation",
        "Confirmation flag while any required right or exposure axis is unresolved.",
    ),
    ClaimLadder(
        "historical-panel-reuse",
        "limitation",
        "A renamed historical panel is not a fresh source family.",
    ),
    ClaimLadder(
        "seed-dependence",
        "limitation",
        "Sensitivity of the withheld loss to the fit seed.",
    ),
    ClaimLadder(
        "campaign-cost",
        "recipe",
        "Compute spent on a confirmation campaign. Unmeasured in this draft.",
    ),
    ClaimLadder(
        "replay-receipt",
        "recipe",
        "Identity of the bound receipt used to replay a published summary.",
    ),
    ClaimLadder(
        "one-cell-loss",
        "recipe",
        "Original-unit loss recomputed for one frozen development cell.",
    ),
    ClaimLadder(
        "full-refit",
        "recipe",
        "Outcome of a full refit under frozen locks.",
    ),
)


LEVELS: tuple[ReproductionLevel, ...] = (
    ReproductionLevel(
        "inspect",
        "Inspect",
        "Read this draft, the claim ladder, and the development source registry. Do not fit a model and do not open a sealed split.",
    ),
    ReproductionLevel(
        "report-replay",
        "Report replay",
        "Recompute a published summary from a bound receipt. This draft has no bound receipt, so the step is unmeasured.",
    ),
    ReproductionLevel(
        "one-cell",
        "One-cell metric recompute",
        "Recompute original-unit loss for one frozen development cell after the metric contract exists. Unmeasured here.",
    ),
    ReproductionLevel(
        "full-refit",
        "Full refit",
        "Refit under frozen rights, code, and budget locks. Blocked while the source-family protocol is unfrozen.",
    ),
)


PSEUDO_LINES: tuple[str, ...] = (
    "procedure RateRiskCompile:",
    "inputs: development sources, learners, and byte constraint",
    "candidates := decoded generators allowed by rights",
    "sensitivity := development-learner transfer loss",
    "allocation := serialized bytes by development sensitivity",
    "artifact := freeze the allocated decoded program",
    "procedure Confirm:",
    "inputs: artifact, withheld sources, withheld learners",
    "require the source-family protocol to be frozen",
    "return original-unit loss on the withheld cells",
)

BANNED: tuple[str, ...] = (
    r"\bSOTA\b",
    r"state-of-the-art",
    r"state of the art",
    r"equivalen",
    r"\bprivacy\b",
    r"\bHIPAA\b",
    r"\bcausal",
    r"pretrain",
    r"\bthe first\b",
    r"\bfirst to\b",
)

NONCLAIM = re.compile(
    r"\\begin\{vnextnonclaims\}.*?\\end\{vnextnonclaims\}",
    re.DOTALL,
)


def method_pseudocode() -> str:
    return "\n".join(PSEUDO_LINES)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _digits(text: str) -> bool:
    return any(character.isdigit() for character in text)


def _banned(text: str) -> str | None:
    for pattern in BANNED:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return None


def _strip_tex_nonclaims(text: str) -> str:
    return NONCLAIM.sub("", text)


def _strip_md_nonclaims(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        if line.strip() == "## Non-claims":
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def _assert_registry_unfrozen(root: Path) -> None:
    sys.path.insert(0, str(root))
    from research.transfer.lineage import load_registry

    registry = load_registry()
    if registry.protocol_frozen() or registry.confirmation_allowed():
        raise PaperDraftError("draft confirmation must stay closed while rights are unresolved")


def check(root: Path | None = None) -> None:
    root = root or repo_root()
    tex_root = root / "docs" / "whitepaper" / "vnext"
    tex_files = sorted(tex_root.rglob("*.tex"))
    if not tex_files:
        raise PaperDraftError("vnext tex tree is missing")
    names = {path.name for path in tex_files}
    if "main.tex" not in names or "supplement.tex" not in names:
        raise PaperDraftError("main and supplement skeletons are required")
    markdown = [
        root / "docs" / "vnext" / "RELATED_WORK.md",
        root / "docs" / "vnext" / "REPRODUCE.md",
    ]
    for path in markdown:
        if not path.is_file():
            raise PaperDraftError("reproduction or related-work note is missing")

    tex = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)
    measurable = tex.replace(r"\newcommand{\TBD}[1]{[unbound: #1]}", "")
    if _digits(measurable):
        raise PaperDraftError("vnext tex contains a digit; measurements must use TBD placeholders")
    if "IEEEtran" in tex or "iclr" in tex.lower():
        raise PaperDraftError("venue class is not selected for this draft")
    banned = _banned(_strip_tex_nonclaims(tex))
    if banned is not None:
        raise PaperDraftError("vnext tex contains an unearned claim marker")
    for claim in CLAIMS:
        if _digits(claim.claim_id) or _digits(claim.statement) or _banned(claim.statement):
            raise PaperDraftError("claim ladder entry is not a safe placeholder")
        token = "\\TBD{" + claim.claim_id + "}"
        if token not in tex:
            raise PaperDraftError("claim placeholder missing from the draft")
    method = (tex_root / "sections" / "method.tex").read_text(encoding="utf-8")
    for line in PSEUDO_LINES:
        if line not in method:
            raise PaperDraftError("method pseudocode is missing a required step")

    related = _strip_md_nonclaims((root / "docs" / "vnext" / "RELATED_WORK.md").read_text(encoding="utf-8"))
    reproduce = (root / "docs" / "vnext" / "REPRODUCE.md").read_text(encoding="utf-8")
    reproduce_body = _strip_md_nonclaims(reproduce)
    for label, body in (("related work", related), ("reproduction", reproduce_body)):
        if _digits(body) or _banned(body):
            raise PaperDraftError(f"{label} note contains a digit or an unearned claim")
    for level in LEVELS:
        if level.measured:
            raise PaperDraftError("a reproduction level is marked measured")
        if f"## {level.heading}" not in reproduce:
            raise PaperDraftError("reproduction packet is missing a level")
    if "docs/testing.md" not in reproduce or "production/kpi-contract.json" not in reproduce:
        raise PaperDraftError("reproduction packet does not name the budget locks")
    if "unmeasured" not in reproduce_body:
        raise PaperDraftError("reproduction packet must keep replay, one-cell, and refit unmeasured")
    for phrase in ("direct predictor", "coreset", "distillation", "circuit"):
        if phrase not in related.lower() and phrase not in related:
            raise PaperDraftError("related work is missing a required alternative")
    _assert_registry_unfrozen(root)
    if VenueWrapper.venue != "undecided":
        raise PaperDraftError("venue wrapper moved without a recorded decision")


def main() -> None:
    check()
    print("vnext draft check passed")
    print("venue", VenueWrapper.venue)
    for claim in CLAIMS:
        print(claim.claim_id, claim.kind, "unbound")


if __name__ == "__main__":
    main()
