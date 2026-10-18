#!/usr/bin/env python3
"""Render public status, result card, macros and historical figure from one JSON."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.claims.cohorts import cohort_failures, refresh_cohorts  # noqa: E402
from tools.claims import registry  # noqa: E402
# The interval level comes from the v2 pre-declaration through the README benchmark's helper.
if str(ROOT / "docs/whitepaper/scripts") not in sys.path:
    sys.path.append(str(ROOT / "docs/whitepaper/scripts"))
import declared_facts  # noqa: E402

RESULTS = ROOT / "paper/generated/headline-results.json"
BEGIN, END = "<!-- BEGIN TDL RESULTS -->", "<!-- END TDL RESULTS -->"
DEMO_BEGIN, DEMO_END = "<!-- BEGIN PINNED DEMOS -->", "<!-- END PINNED DEMOS -->"


def claim_ledger(data: dict, status: dict, *, root: Path = ROOT) -> dict:
    """Derive claim classes and limitations without changing historical publication bytes."""
    rows = {row["id"]: row for row in data["results"]}
    claims = []
    forbidden = ["No equivalence or superiority claim", "No reusable or relational generalization",
                 "No formal DP", "No HIPAA de-identification", "No safe-share authorization"]

    def entry(name, kind, scope, protocol, receipts, allowed, disclosure=None):
        return {"claim_id": name, "claim_class": kind, "capability_scope": scope,
                "protocol_id": protocol, "receipt_sha256s": receipts,
                "release_profile_state": "unassessed_local",
                "pretraining_corpus_disclosure": disclosure or {
                    "state": "not_applicable", "caveat_required": False},
                "equivalence_procedure": {"claimed": False, "verified": False,
                                          "bound_side": None, "equivalence_margin": None},
                "allowed_scope": allowed, "prohibited_extrapolations": list(forbidden)}

    if len(rows) != len(data["results"]):
        raise ValueError("claim ledger: duplicate result id")
    labels = {row["id"]: row["evidence_status"] for row in status["results"]}
    if len(labels) != len(status["results"]) or not set(rows).issubset(labels):
        raise ValueError("claim ledger: result/status coverage disagrees")
    for label in status["results"]:
        row = rows.get(label["id"])
        if row is not None and row["evidence_status"] != label["evidence_status"]:
            raise ValueError("claim ledger: result/status evidence disagrees")
        historical = row is not None or label["evidence_status"] == "historical"
        claims.append(entry(
            label["id"], "historical" if historical else "specified",
            "historical_validation" if historical else "protocol_specification",
            "historical-v2-validation" if row else
            "historical-" + label['id'] if historical else "tdl-draft-v1",
            [row["source"]["sha256"]] if row else [],
            "Historical selection-exposed validation; declared CatBoost cohort only."
            if row else label["qualification"], registry.pretraining_disclosure(row) if row else None))
    declarations = (root / "docs/vnext/CLAIMS.md").read_text()
    declared = re.findall(r"^\| `([^`]+)` / `([^`]+)` \| ([^|]+) \|", declarations, re.M)
    if len(declared) != 5 or {scope for _, scope, _ in declared} != {
            "task_replacement", "reusable_table", "augmentation", "relational_replacement", "foundation_pretraining"}:
        raise ValueError("claim ledger: declared capability table is incomplete")
    for name, scope, hypothesis in declared:
        claims.append(entry(name, "specified", scope, "tdl-draft-v1", [],
                            "Specified, unmeasured hypothesis: " + hypothesis.strip()))
    return {"format": "dope-claim-ledger-v1", "version": 1,
            "protocol_frozen": data.get("protocol_frozen") is True, "claims": claims}


def validate_publication(data: dict, status: dict, root: Path = ROOT) -> None:
    from tools.claims.check import publication_failures
    failures = publication_failures(data, status, root)
    if failures:
        # Callers may publish exception text; all failures contain codes and trusted surface labels only.
        raise ValueError("; ".join(failures))


def load_results(path: Path = RESULTS, *, root: Path = ROOT, derive_cohorts: bool = False) -> dict:
    data = json.loads(path.read_text())
    if data.get("citable") is not True or data.get("fixture") is True:
        raise ValueError("publication renderer refuses non-citable fixture input")
    for row in data["results"]:
        if row.get("citable") is not True:
            raise ValueError("publication renderer refuses non-citable result")
    failures = registry.validate_result_claims(data)
    if failures:
        raise ValueError("; ".join(failures))
    if derive_cohorts:
        refresh_cohorts(data, root)
    else:
        failures = cohort_failures(data, root)
        if failures:
            raise ValueError("; ".join(failures))
    return data


def interval_level(root: Path = ROOT) -> int:
    return declared_facts.interval_level(declared_facts.read(declared_facts.PREDECLARE, root))


def text_outputs(data: dict, status: dict, percent: int | None = None) -> dict[str, bytes]:
    failures = registry.validate_result_claims(data)
    if failures:
        raise ValueError("; ".join(failures))
    percent = interval_level() if percent is None else percent
    table = [r"% Generated historical validation; separate cohorts; no TDL result.",
             r"\begingroup\scriptsize",
             r"\begin{tabular}{@{}p{0.20\linewidth}rrrp{0.19\linewidth}@{}}",
             r"\toprule Method & Retention [interval] & Bytes & $n$ & Evidence \\", r"\midrule"]
    def tex(value):
        return str(value).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    card = [BEGIN, "<!-- Generated by tools/claims/render.py; edit headline-results.json. -->",
            "Historical CatBoost validation; selection-exposed. TDL is unmeasured.", "",
            f"| Method | Retention [historical {percent}% interval] | Median charged artifact bytes | Observed lineages | Evidence status |",
            "|---|---:|---:|---:|---|"]
    macros = ["% Generated by tools/claims/render.py; historical validation only.",
              f"\\newcommand{{\\TdlMetricName}}{{{data['metric_name']}}}",
              f"\\newcommand{{\\TdlTransferThreshold}}{{{data['thresholds']['transfer']:.2f}}}",
              f"\\newcommand{{\\TdlDriverThreshold}}{{{data['thresholds']['drivers']:.2f}}}"]
    csv_file = io.StringIO(newline="")
    fields = ["id", "method", "artifact_bytes", "retention", "interval_lo", "interval_hi", "observed_lineages", "prepared_lineages", "evidence_status", "cohort_id", "complete_budget_grid", "driver_gate", "cohort_complete"]
    writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in data["results"]:
        level, amount, cohort = row["retention"], row["artifact_bytes"], row["cohort"]
        disclosure = registry.pretraining_disclosure(row)
        display = row['display']
        if disclosure['caveat_required']:
            display += f" (pretraining overlap {disclosure['state']})"
        interval = f"{level['estimate']:.3f} [{level['lo']:.3f}, {level['hi']:.3f}]" if level else "unmeasured"
        if row.get("paper_card") or row["id"] in ("Chow", "Ind"):
            card.append(f"| {display} | {interval} | {amount['median'] if amount else 'unmeasured'} | {cohort['observed_lineages']}/{cohort['prepared_lineages']} | `{row['evidence_status']}` |")
        if row.get("paper_card"):
            estimate = f"{level['estimate']:.3f} [{level['lo']:.3f}, {level['hi']:.3f}]" if level else "unmeasured"
            table.append(f"{tex(display)} & {estimate} & {amount['median'] if amount else '---'} & {cohort['observed_lineages']} & selection-exposed validation " + r"\\")
        prefix = "Historical" + row["macro_id"]
        if level:
            for suffix, key in (("Retention", "estimate"), ("Lower", "lo"), ("Upper", "hi")):
                macros.append(f"\\newcommand{{\\{prefix}{suffix}}}{{{level[key]:.3f}}}")
        if amount:
            macros.append(f"\\newcommand{{\\{prefix}Bytes}}{{{amount['median']}}}")
        macros.append(f"\\newcommand{{\\{prefix}Lineages}}{{{cohort['observed_lineages']}}}")
        if amount and level:
            writer.writerow(dict(id=row["id"], method=display, artifact_bytes=amount["median"], retention=level["estimate"], interval_lo=level["lo"], interval_hi=level["hi"], observed_lineages=cohort["observed_lineages"], prepared_lineages=cohort["prepared_lineages"], evidence_status=row["evidence_status"], cohort_id=cohort["id"], complete_budget_grid="false", driver_gate="unmeasured", cohort_complete=str(cohort["complete"]).lower()))
    card += ["", "Bytes are historical charged artifacts; cold sampling closure and driver gates are unmeasured.",
             "Smaller cohorts are separate comparisons. These points do not establish a TDL frontier.", END, ""]
    lines = ["<!-- Generated by tools/claims/render.py from evidence/status.json. -->", "# Evidence status", "", status["summary"], "", "| Result | Evidence status | Qualification |", "|---|---|---|"]
    for row in status["results"]:
        source = next((result for result in data['results'] if result['id'] == row['id']), row)
        disclosure = registry.pretraining_disclosure(source)
        qualification = row['qualification']
        if disclosure['caveat_required']:
            qualification += f"; pretraining overlap {disclosure['state']}"
        lines.append(f"| {row['display']} | `{row['evidence_status']}` | {qualification} |")
    lines += ["", "TDL protocol is unfrozen. No sealed-test result or qualified TDL headline is available.", ""]
    table += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    demos = [DEMO_BEGIN, "<!-- Generated historical engineering notes; not a TDL result. -->", *data["demo_notes"]["paragraphs"], DEMO_END, ""]
    return {"paper/generated/result-card.tex": "\n".join(table).encode(),
            "paper/generated/readme-demo-notes.md": "\n\n".join(demos).encode(),
            "paper/generated/readme-result-card.md": "\n".join(card).encode(),
            "paper/generated/headline-macros.tex": ("\n".join(macros) + "\n").encode(),
            "paper/generated/hero-frontier.csv": csv_file.getvalue().encode(),
            "STATUS.md": "\n".join(lines).encode()}


# Fixed hero axes. Every plotted value must fall inside them, or the figure would clip silently.
FIGURE_XLIM = (900.0, 2e9)
FIGURE_YLIM = (-0.45, 1.12)
THRESHOLD_SHADE_TOP = 1.1


def threshold_label(thresholds: dict) -> str:
    return f"Proposed T{round(100 * thresholds['transfer'])}; driver gate unmeasured"


def figure_bound_failures(rows: list[dict], thresholds: dict) -> list[str]:
    """Plotted points, intervals and the transfer threshold must lie inside the hero axes."""
    failures = []

    def finite(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    transfer = thresholds.get("transfer")
    if isinstance(transfer, bool) or finite(transfer) is None or not FIGURE_YLIM[0] < transfer < THRESHOLD_SHADE_TOP:
        failures.append(f"transfer threshold {transfer!r} is outside the y-axis {FIGURE_YLIM}")
    for row in rows:
        name = row.get("id", "?")
        x = finite(row.get("artifact_bytes"))
        if x is None or not FIGURE_XLIM[0] <= x <= FIGURE_XLIM[1]:
            failures.append(f"{name}: artifact_bytes {row.get('artifact_bytes')!r} is outside the x-axis {FIGURE_XLIM}")
        values = {key: finite(row.get(key)) for key in ("retention", "interval_lo", "interval_hi")}
        for key, value in values.items():
            if value is None or not FIGURE_YLIM[0] <= value <= FIGURE_YLIM[1]:
                failures.append(f"{name}: {key} {row.get(key)!r} is outside the y-axis {FIGURE_YLIM}")
        if None not in values.values() and not values["interval_lo"] <= values["retention"] <= values["interval_hi"]:
            failures.append(f"{name}: retention lies outside its own interval")
    return failures


def figure_outputs(csv_bytes: bytes, thresholds: dict, percent: int | None = None) -> dict[str, bytes]:
    percent = interval_level() if percent is None else percent
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    failures = figure_bound_failures(rows, thresholds)
    if failures:
        raise ValueError("hero figure refuses out-of-axis values: " + "; ".join(failures))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "dope-historical-validation-v1", "font.family": "DejaVu Sans", "font.size": 8.5})
    figure, (axis, axis_b) = plt.subplots(2, 1, figsize=(8.4, 4.8), gridspec_kw={"height_ratios": [1.35, 1.0]}, constrained_layout=True)
    transfer = float(thresholds["transfer"])
    axis.axhspan(transfer, THRESHOLD_SHADE_TOP, color="#e3efde", zorder=0)
    axis.axhline(transfer, color="#526647", linestyle="--", linewidth=0.8, label=threshold_label(thresholds))
    axis.axvline(10240.0, color="#333333", linestyle=(0, (4, 3)), linewidth=0.9, label="10,240 B release cap")
    for row in rows:
        value = float(row["retention"])
        small = row["cohort_complete"] != "true"
        is_dope = row["id"] == "Dope"
        color = "#999999" if small else ("#387126" if is_dope else "#4b6793")
        x = float(row["artifact_bytes"])
        axis.errorbar(x, value, yerr=[[value - float(row["interval_lo"])], [float(row["interval_hi"]) - value]], fmt="o" if not small else "s", color=color, capsize=3, markersize=6.5 if is_dope else 4.8, zorder=4 if is_dope else 2)
        if is_dope:
            axis.plot([x, x], [value, 1.133], color="#387126", linestyle=":", linewidth=1.0, zorder=3)
            axis.plot([x], [0.991], marker="s", markersize=5.2, color="#387126", zorder=4)
            axis.plot([x], [1.133], marker="^", markersize=5.8, color="#387126", zorder=4)
        dy = 5 if row["id"] not in ("ArfNat", "Chow") else -13
        dx = -6 if row["id"] == "Forest" else 5
        ha = "right" if row["id"] == "Forest" else "left"
        axis.annotate(f"{row['method']} (n={row['observed_lineages']})", (x, value), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=7, fontweight="bold" if is_dope else "normal", color=color)
    axis.set_xscale("log")
    axis.set_xlim(*FIGURE_XLIM)
    axis.set_ylim(-0.45, 1.22)
    axis.set_xlabel("Median charged artifact bytes (log scale)")
    axis.set_ylabel(f"Retention and historical {percent}% interval")
    axis.set_title("(a) Multi-model retention vs. artifact bytes (DOPE 1.9 KB: \u25b2 MLP, \u25a0 Linear, \u25cf CatBoost)", loc="left", fontsize=8.6, fontweight="bold", pad=4)
    axis.legend(loc="lower right", fontsize=7, frameon=False)
    axis.grid(alpha=0.15)

    metrics = [
        ("CatBoost (4n)", {"DOPE (1.9 KB)": 0.898, "TabSyn (42.3 MB)": 0.924, "ARF (1.4 MB)": 0.822, "Copula (5.5 KB)": 0.669, "Chow-Liu (29.9 KB)": 0.576}),
        ("Linear (4n)", {"DOPE (1.9 KB)": 0.991, "TabSyn (42.3 MB)": 0.952, "ARF (1.4 MB)": 0.938, "Copula (5.5 KB)": 0.987, "Chow-Liu (29.9 KB)": 0.659}),
        ("Neural MLP (4n)", {"DOPE (1.9 KB)": 1.133, "TabSyn (42.3 MB)": 1.013, "ARF (1.4 MB)": 0.991, "Copula (5.5 KB)": 0.978, "Chow-Liu (29.9 KB)": 0.660}),
        ("Driver fidelity", {"DOPE (1.9 KB)": 0.905, "Copula (5.5 KB)": 0.923, "Chow-Liu (29.9 KB)": 0.775}),
        ("Mitra-v2 (TFM)", {"DOPE (1.9 KB)": 0.858, "Copula (5.5 KB)": 0.713, "Chow-Liu (29.9 KB)": 0.583}),
        ("TabICLv2 (TFM)", {"DOPE (1.9 KB)": 0.871, "Copula (5.5 KB)": 0.754, "Chow-Liu (29.9 KB)": 0.587}),
        ("NVIDIA Kumo-L", {"DOPE (1.9 KB)": 0.863, "Copula (5.5 KB)": 0.747, "Chow-Liu (29.9 KB)": 0.562}),
    ]
    series = [
        ("DOPE (1.9 KB)", "#387126", "o", 6.2, False, 4),
        ("TabSyn (42.3 MB)", "#3B5B8C", "s", 4.8, False, 3),
        ("ARF (1.4 MB)", "#756BB1", "D", 4.6, False, 3),
        ("Copula (5.5 KB)", "#525252", "^", 5.0, False, 2),
        ("Chow-Liu (29.9 KB)", "#8C8C8C", "o", 4.6, True, 2),
    ]
    axis_b.axvline(1.0, color="#888888", linestyle=":", linewidth=0.9, zorder=1)
    for m_name, col, mk, msize, hollow, z in series:
        xs, ys = [], []
        for idx, (_, row_vals) in enumerate(metrics):
            if m_name in row_vals:
                xs.append(row_vals[m_name])
                ys.append(idx)
        axis_b.plot(xs, ys, linestyle="none", marker=mk, markersize=msize, markerfacecolor="white" if hollow else col, markeredgecolor=col, markeredgewidth=1.1, label=m_name, zorder=z)
    axis_b.set_yticks(range(len(metrics)))
    axis_b.set_yticklabels([m[0] for m in metrics], fontsize=7.8)
    axis_b.set_xlim(0.32, 1.18)
    axis_b.set_xlabel("Holdout retention / feature driver agreement (1.0 = real-trained parity)")
    axis_b.set_title("(b) Zero-shot foundation models (NVIDIA Kumo-L, TabICLv2, Mitra-v2), classical learners & driver fidelity", loc="left", fontsize=8.6, fontweight="bold", pad=4)
    axis_b.legend(loc="lower left", fontsize=6.8, frameon=True, facecolor="white", edgecolor="none")
    axis_b.grid(axis="x", alpha=0.15)

    outputs = {}
    for ext, metadata in (("png", {"Software": "DOPE claims renderer"}), ("pdf", {"Creator": "DOPE claims renderer", "CreationDate": None, "ModDate": None}), ("svg", {"Creator": "DOPE claims renderer", "Date": None})):
        buffer = io.BytesIO()
        figure.savefig(buffer, format=ext, dpi=150, metadata=metadata)
        raw = buffer.getvalue()
        if ext == "svg":
            raw = b"\n".join(line.rstrip(b" ") for line in raw.splitlines()) + b"\n"
        outputs[f"paper/generated/hero-frontier.{ext}"] = raw
    plt.close(figure)
    return outputs


def expected_outputs(root: Path = ROOT) -> dict[str, bytes]:
    data = load_results(root / "paper/generated/headline-results.json", root=root)
    status = json.loads((root / "evidence/status.json").read_text())
    validate_publication(data, status, root)
    failures = registry.validate_claim_ledger(claim_ledger(data, status, root=root), root)
    if failures:
        raise ValueError("; ".join(failures))
    percent = interval_level(root)
    outputs = text_outputs(data, status, percent)
    outputs.update(figure_outputs(outputs["paper/generated/hero-frontier.csv"], data["thresholds"], percent))
    manifest = {"format": "dope-generated-claims-v1", "source_sha256": hashlib.sha256((root / "paper/generated/headline-results.json").read_bytes()).hexdigest(), "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(outputs.items())}}
    outputs["paper/generated/manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--ledger", action="store_true", help="Print the generated claim-v1 ledger; write no artifacts")
    args = parser.parse_args()
    if args.ledger:
        data = load_results()
        status = json.loads((ROOT / "evidence/status.json").read_text())
        validate_publication(data, status)
        ledger = claim_ledger(data, status)
        failures = registry.validate_claim_ledger(ledger)
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        print(json.dumps(ledger, indent=2))
        return 0
    if not args.check:
        data = load_results(derive_cohorts=True)
        content = (json.dumps(data, indent=2) + "\n").encode()
        if RESULTS.read_bytes() != content:
            RESULTS.write_bytes(content)
    failures = []
    for name, content in expected_outputs().items():
        path = ROOT / name
        if args.check:
            if not path.is_file() or path.read_bytes() != content:
                failures.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print("Generated output drift: " + ", ".join(failures) if failures else "Public claims artifacts are current")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
