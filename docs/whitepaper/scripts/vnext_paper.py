"""Check the falsifiable vNext draft, or render a verified claim graph.

The default draft check does not emit measurements. Headline quantities stay as
``\\TBD{claim-id}`` placeholders in the tracked draft. Venue selection stays
closed.

The separately versioned measured mode (``--measured``) renders only the public
projection of a claim graph that ``research.transfer.claims`` verified against
receipt bytes and trusted identities. It writes under ``target/vnext/`` and never
edits the tracked draft. No claim it renders is qualified or campaign-admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import math
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


# Measured interface mode. The contract is G1's closed claim-graph interface v1
# (research.transfer.claims). This consumer re-checks the public projection it is
# handed and renders it; it never derives a decision, an interval, or a number.

MEASURED_FORMAT = "dope-vnext-measured-render"
MEASURED_VERSION = 1
PROJECTION_FORMAT = "dope-claim-public-projection"
PROJECTION_VERSION = 1
PROJECTION_KEYS = frozenset({"format", "version", "claims", "projection_sha256"})
PROJECTION_CLAIM_KEYS = frozenset({
    "claim_id", "protocol_sha256", "cohort_sha256", "recipe_sha256", "source_sha256", "metric_sha256",
    "loss", "loss_unit", "artifact_count", "fit_count", "sample_count", "vendor_executed", "qualification",
    "campaign_admitted", "n21_admitted", "qualified", "synthetic", "receipts",
    "raw_gap_superiority", "adjusted_z_noninferiority", "equivalence",
})
DECISION_KEYS = frozenset({
    "decision", "qualified", "lower", "upper", "estimate", "level", "margin", "equivalence_margin", "bound_side",
})
COMPARISONS = (
    ("raw_gap_superiority", "raw-gap superiority"),
    ("adjusted_z_noninferiority", "adjusted-Z noninferiority"),
    ("equivalence", "equivalence"),
)
DECISIONS = {
    "supported": "supported",
    "not_supported": "not supported",
    "inconclusive": "inconclusive",
    "unavailable": "unavailable",
}
BOUND_SIDES = {None: None, "upper": "upper bound", "both": "both bounds"}
MARGIN_KEYS = ("relative", "absolute_scaled", "absolute_raw")
QUALIFICATIONS = frozenset({"synthetic", "development"})
CAMPAIGN_ID = re.compile(r"[a-z][a-z0-9-]{0,63}")
RECEIPT_ID = re.compile(r"[a-z0-9-]{1,64}")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}


@dataclass(frozen=True)
class MeasuredRender:
    projection: bytes
    macros: str
    receipt: dict


def _transfer():
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from research.transfer import claims
    from research.transfer.evidence import canonical_json
    from research.transfer.lineage import LineageError, assert_public_provenance

    return claims, canonical_json, assert_public_provenance, LineageError


def tex_escape(text: str) -> str:
    """Escape one untrusted string for TeX. Control characters are refused, never echoed."""
    if type(text) is not str or any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise PaperDraftError("claim text is not a printable string")
    return "".join(TEX_ESCAPES.get(character, character) for character in text)


def _number(value) -> str:
    """Render a receipt-bound number exactly as stored; bool and nonfinite values are refused."""
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    raise PaperDraftError("claim number is not a finite receipt value")


def _optional(value) -> str:
    return "unavailable" if value is None else _number(value)


def _decision_text(label: str, decision: dict) -> str:
    text = f"{label} {DECISIONS[decision['decision']]}"
    if decision["lower"] is not None or decision["upper"] is not None:
        text += f", interval [{_optional(decision['lower'])}, {_optional(decision['upper'])}]"
    if decision["estimate"] is not None:
        text += f", estimate {_number(decision['estimate'])}"
    if decision["level"] is not None:
        text += f", level {_number(decision['level'])}"
    if decision["margin"] is not None:
        margin = decision["margin"]
        text += (f", margin relative {_optional(margin['relative'])}, scaled {_optional(margin['absolute_scaled'])},"
                 f" raw {_optional(margin['absolute_raw'])}")
    if decision["equivalence_margin"] is not None:
        text += f", equivalence margin {_number(decision['equivalence_margin'])}"
    if BOUND_SIDES[decision["bound_side"]] is not None:
        text += f", {BOUND_SIDES[decision['bound_side']]}"
    return text


def _claim_text(claim: dict) -> str:
    parts = [
        f"{tex_escape(claim['loss'])} in {tex_escape(claim['loss_unit'])}",
        f"{_number(claim['artifact_count'])} artifacts, {_number(claim['fit_count'])} fits, "
        f"{_number(claim['sample_count'])} samples",
    ]
    parts += [_decision_text(label, claim[key]) for key, label in COMPARISONS]
    return "; ".join(parts)


def _check_decision(decision) -> None:
    if type(decision) is not dict or set(decision) != DECISION_KEYS:
        raise PaperDraftError("decision does not match claim-graph interface v1")
    if decision["decision"] not in DECISIONS or decision["bound_side"] not in BOUND_SIDES:
        raise PaperDraftError("decision state is not registered")
    if decision["qualified"] is not False:
        raise PaperDraftError("measured mode cannot render a qualified decision")
    for key in ("lower", "upper", "estimate", "level", "equivalence_margin"):
        _optional(decision[key])
    if decision["margin"] is not None:
        if type(decision["margin"]) is not dict or tuple(sorted(decision["margin"])) != tuple(sorted(MARGIN_KEYS)):
            raise PaperDraftError("decision margin does not match claim-graph interface v1")
        for key in MARGIN_KEYS:
            _optional(decision["margin"][key])
    if decision["decision"] == "supported" and (decision["lower"] is None or decision["upper"] is None):
        raise PaperDraftError("a supported decision has an unverified interval")


def _checked_projection(projection, canonical_json, assert_public_provenance, lineage_error) -> list[dict]:
    """Defense in depth: the consumer does not trust a projection it did not check."""
    try:
        assert_public_provenance(projection)
    except lineage_error:
        raise PaperDraftError("public projection failed the provenance check") from None
    if set(projection) != PROJECTION_KEYS or projection["format"] != PROJECTION_FORMAT \
            or type(projection["version"]) is not int or projection["version"] != PROJECTION_VERSION:
        raise PaperDraftError("public projection does not match claim-graph interface v1")
    body = {key: value for key, value in projection.items() if key != "projection_sha256"}
    try:
        content = canonical_json(body)
    except ValueError:
        raise PaperDraftError("public projection is not finite JSON") from None
    if hashlib.sha256(content).hexdigest() != projection["projection_sha256"]:
        raise PaperDraftError("public projection digest does not match its content")
    rows = projection["claims"]
    if type(rows) is not list or not rows:
        raise PaperDraftError("public projection has no claim")
    ladder = {claim.claim_id for claim in CLAIMS}
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != PROJECTION_CLAIM_KEYS:
            raise PaperDraftError("claim does not match claim-graph interface v1")
        if row["claim_id"] not in ladder:
            raise PaperDraftError("claim is not on the vnext claim ladder")
        if row["claim_id"] in seen:
            raise PaperDraftError("claim appears twice in the projection")
        seen.add(row["claim_id"])
        if row["qualified"] is not False or row["campaign_admitted"] is not False or row["n21_admitted"] is not False:
            raise PaperDraftError("measured mode cannot render a qualified or campaign-admitted claim")
        if row["qualification"] not in QUALIFICATIONS or row["synthetic"] is not (row["qualification"] == "synthetic"):
            raise PaperDraftError("claim qualification is not registered")
        if row["vendor_executed"] is not True:
            raise PaperDraftError("claim has no verified vendor execution")
        for key in ("artifact_count", "fit_count", "sample_count"):
            if type(row[key]) is not int or row[key] < 0:
                raise PaperDraftError("claim count is not a separate non-negative integer")
        receipts = row["receipts"]
        if type(receipts) is not list or not receipts \
                or any(type(item) is not dict or set(item) != {"sha256"} for item in receipts):
            raise PaperDraftError("claim receipts are not a list of sha256 objects")
        pins = [item["sha256"] for item in receipts]
        if pins != sorted(set(pins)) or any(type(pin) is not str or SHA256_HEX.fullmatch(pin) is None for pin in pins):
            raise PaperDraftError("claim receipt pins are not a sorted unique sha256 list")
        for key, _label in COMPARISONS:
            _check_decision(row[key])
    return rows


def render_measured(graph, campaign_id: str) -> MeasuredRender:
    """Render one verified claim graph. A dict, a number, or a verified flag is refused."""
    claims, canonical_json, assert_public_provenance, lineage_error = _transfer()
    if type(graph) is not claims.VerifiedClaimGraph:
        raise PaperDraftError("measured mode accepts only a verified claim graph")
    if type(campaign_id) is not str or CAMPAIGN_ID.fullmatch(campaign_id) is None:
        raise PaperDraftError("campaign id is not a lowercase slug")
    try:
        projection = claims.public_projection(graph)
    except AttributeError:
        raise PaperDraftError("claim graph was not produced by the verifier") from None
    rows = {row["claim_id"]: row for row in
            _checked_projection(projection, canonical_json, assert_public_provenance, lineage_error)}
    lines = [
        "% Generated by docs/whitepaper/scripts/vnext_paper.py --measured. Do not edit.",
        f"% Claim-graph projection {projection['projection_sha256']}.",
        "% No claim below is qualified or admitted to a campaign.",
        "\\makeatletter",
        "\\providecommand{\\TBD}[1]{[unbound: #1]}",
        "\\providecommand{\\VNextUnqualified}[2]{[unqualified #1 value: #2]}",
        "\\providecommand{\\VNextClaim}[1]{\\@ifundefined{vnext@claim@#1}{\\TBD{#1}}{\\@nameuse{vnext@claim@#1}}}",
    ]
    ladder = []
    for claim in CLAIMS:
        row = rows.get(claim.claim_id)
        if row is None:
            ladder.append({"claim_id": claim.claim_id, "state": "unbound", "qualification": None})
            continue
        lines.append(f"\\@namedef{{vnext@claim@{claim.claim_id}}}"
                     f"{{\\VNextUnqualified{{{row['qualification']}}}{{{_claim_text(row)}}}}}")
        ladder.append({"claim_id": claim.claim_id, "state": "bound", "qualification": row["qualification"]})
    lines.append("\\makeatother")
    macros = "\n".join(lines) + "\n"
    projection_bytes = canonical_json(projection)
    receipt = {
        "format": MEASURED_FORMAT,
        "version": MEASURED_VERSION,
        "interface": "dope-claim-graph-interface-v1",
        "campaign_id": campaign_id,
        "projection_sha256": projection["projection_sha256"],
        "claims_json_sha256": hashlib.sha256(projection_bytes).hexdigest(),
        "claims_tex_sha256": hashlib.sha256(macros.encode("utf-8")).hexdigest(),
        "ladder": ladder,
        "bound_claims": len(rows),
        "qualified_claims": 0,
        "campaign_admitted": False,
        "n21_admitted": False,
        "figure_inputs": {"state": "unbound", "reason": "no claim is admitted to a campaign"},
    }
    try:
        assert_public_provenance(receipt)
    except lineage_error:
        raise PaperDraftError("render receipt failed the provenance check") from None
    return MeasuredRender(projection_bytes, macros, receipt)


def verify_and_render(document: bytes, receipts: dict[str, bytes], expected: dict, campaign_id: str) -> MeasuredRender:
    claims = _transfer()[0]
    return render_measured(claims.verify_claim_graph(document, receipts, expected), campaign_id)


def write_measured(render: MeasuredRender, output_dir: Path) -> None:
    canonical_json = _transfer()[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "claims.json").write_bytes(render.projection)
    (output_dir / "claims.tex").write_text(render.macros, encoding="utf-8")
    (output_dir / "render.json").write_bytes(canonical_json(render.receipt) + b"\n")


def _bounded_read(path: Path, limit: int) -> bytes:
    """One byte past the interface budget is enough for the verifier to refuse an oversized input."""
    with path.open("rb") as stream:
        return stream.read(limit + 1)


def _measured_main(args, parser) -> int:
    claims = _transfer()[0]
    from research.transfer.evidence import EvidenceError, parse_json

    if args.expected is None or args.campaign_id is None or not args.receipt:
        parser.error("--measured needs --receipt, --expected, and --campaign-id")
    if len(args.receipt) > claims.MAX_RECEIPTS:
        parser.error("too many receipts for claim-graph interface v1")
    receipt_paths: dict[str, Path] = {}
    for item in args.receipt:
        receipt_id, separator, path = item.partition("=")
        if not separator or RECEIPT_ID.fullmatch(receipt_id) is None or receipt_id in receipt_paths:
            parser.error("--receipt takes a unique ID=PATH with a lowercase receipt id")
        receipt_paths[receipt_id] = Path(path)
    try:
        receipts = {receipt_id: _bounded_read(path, claims.MAX_RECEIPT_BYTES)
                    for receipt_id, path in receipt_paths.items()}
        document = _bounded_read(args.measured, claims.MAX_DOCUMENT_BYTES)
        expected = parse_json(_bounded_read(args.expected, claims.MAX_DOCUMENT_BYTES))
    except OSError:
        print("vnext measured mode refused: input_unreadable", file=sys.stderr)
        return 1
    try:
        render = verify_and_render(document, receipts, expected, args.campaign_id)
    except (claims.ClaimError, EvidenceError, PaperDraftError) as error:
        print(f"vnext measured mode refused: {error}", file=sys.stderr)
        return 1
    output_dir = args.output_dir or repo_root() / "target" / "vnext" / args.campaign_id
    write_measured(render, output_dir)
    print("vnext measured render written; no claim is qualified or campaign-admitted")
    print("projection", render.receipt["projection_sha256"])
    print("bound claims", render.receipt["bound_claims"], "of", len(CLAIMS))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the vNext draft, or render a verified claim graph.")
    parser.add_argument("--measured", type=Path, metavar="CLAIM_GRAPH",
                        help="claim-graph document to verify and render (measured interface mode)")
    parser.add_argument("--receipt", action="append", default=[], metavar="ID=PATH",
                        help="one bound receipt referenced by the claim graph")
    parser.add_argument("--expected", type=Path, metavar="TRUSTED_IDENTITIES",
                        help="trusted expected identities from the custodian, never from the claim graph")
    parser.add_argument("--campaign-id", help="lowercase slug naming the target/vnext output folder")
    parser.add_argument("--output-dir", type=Path, help="override target/vnext/<campaign-id>")
    args = parser.parse_args(argv)
    if args.measured is not None:
        return _measured_main(args, parser)
    if args.receipt or args.expected or args.campaign_id or args.output_dir:
        parser.error("these options need --measured")
    check()
    print("vnext draft check passed")
    print("venue", VenueWrapper.venue)
    for claim in CLAIMS:
        print(claim.claim_id, claim.kind, "unbound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
