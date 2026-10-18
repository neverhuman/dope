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
from tools.claims import foundation, registry  # noqa: E402
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
    f_summary = foundation.load_foundation_summary()
    f_rows = foundation.foundation_table_rows(f_summary)
    table = [r"% Generated historical validation; separate cohorts; no TDL result.",
             r"\begingroup\scriptsize",
             r"\begin{tabular}{@{}p{0.20\linewidth}rrrp{0.19\linewidth}@{}}",
             r"\toprule Method & Retention [interval] & Bytes & $n$ & Evidence \\", r"\midrule"]
    def tex(value):
        return str(value).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    card = [BEGIN, "<!-- Generated by tools/claims/render.py; edit headline-results.json. -->",
            "Historical validation; selection-exposed. TDL and cross-model feature-driver preservation (`D80`) are unmeasured.", "",
            "#### Historical Foundation-Model Block (`mfs-v3-density-panel.json`, 97 matched mixed/simulated lineages, fit seed 11, sample seeds 101/211/307, 4n context)", "",
            f"| Method | Kumo Large [historical {percent}% interval] | TabICLv2 [historical {percent}% interval] | Mitra-v2 [historical {percent}% interval] | Matched charged bytes | Cross-model driver preservation | Target-correlation proxy (94/97 lineages) |",
            "|---|---:|---:|---:|---:|---|---:|"]
    for f_row in f_rows:
        k_c, t_c, m_c = f_row["kumo_tabular_l"], f_row["tabicl2"], f_row["mitra_v2"]
        px = f_row["target_correlation_proxy"]
        card.append(
            f"| {f_row['display']} | {100.0 * k_c['median']:.1f}% [{100.0 * k_c['lo']:.1f}%, {100.0 * k_c['hi']:.1f}%] "
            f"| {100.0 * t_c['median']:.1f}% [{100.0 * t_c['lo']:.1f}%, {100.0 * t_c['hi']:.1f}%] "
            f"| {100.0 * m_c['median']:.1f}% [{100.0 * m_c['lo']:.1f}%, {100.0 * m_c['hi']:.1f}%] "
            f"| {f_row['matched_artifact_bytes']:,} B | `{f_row['cross_model_driver_preservation']}` "
            f"| {px['median']:.3f} [{px['q1']:.3f}, {px['q3']:.3f}] (`{px['proxy_kind']}`) |"
        )
    card += [
        "",
        f"{f_summary['strongest_foundation_comparator']['scope_note']} "
        f"{f_summary['discrepancy_audit']['resolution']} "
        "Foundation-model auditor qualification is pending and pretraining overlap is unknown.",
        "",
        "#### Historical Classical CatBoost Block (`v2-headline.json`, selection-exposed)",
        "",
        f"| Method | Retention [historical {percent}% interval] | Median charged artifact bytes | Observed lineages | Evidence status |",
        "|---|---:|---:|---:|---|",
    ]
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
    card += ["", "Bytes are historical charged artifacts; cold sampling closure and cross-model feature-driver preservation (`D80`) are unmeasured.",
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


def _draw_foundation_hero(f_summary: dict, six_bars: list[dict], percent: int, *, profile: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Paper profile uses exact 5.60 in width matching \herowidth (0.86\linewidth = 5.59 in) so 9.5pt is ~9.5pt on page 1.
    # Web profile uses 6.20 in width at 180 dpi (1116 px) with identical proportions for crisp README & 390px mobile display.
    width_in, height_in = (5.60, 3.12) if profile == "paper" else (6.20, 3.45)
    base_pt = 9.5 if profile == "paper" else 10.3
    matplotlib.rcParams.update({
        "svg.hashsalt": "dope-historical-validation-v1",
        "font.family": "DejaVu Sans",
        "font.size": base_pt,
    })
    figure, axis = plt.subplots(1, 1, figsize=(width_in, height_in))
    figure.subplots_adjust(left=0.19, right=0.950, top=0.75, bottom=0.25)

    y_positions = [5.10, 4.42, 3.10, 2.42, 1.10, 0.42]
    group_centers = [4.76, 2.76, 0.76]
    group_labels = [bar["auditor_display"] for bar in six_bars[::2]]
    dope_bytes = six_bars[0]["matched_artifact_bytes"]
    copula_bytes = six_bars[1]["matched_artifact_bytes"]

    for y_pos, bar in zip(y_positions, six_bars):
        is_dope = bar["method"] == "DOPE"
        color = "#2f6b25" if is_dope else "#7a7a7a"
        hatch = "" if is_dope else "//"
        val = bar["retention_pct"]
        lo = bar["interval_lo_pct"]
        hi = bar["interval_hi_pct"]
        rect = axis.barh(
            y_pos, val, height=0.54, color=color, edgecolor="#1a1a1a",
            linewidth=0.85, zorder=2,
        )[0]
        if hatch:
            rect.set_hatch(hatch)
        axis.errorbar(
            val, y_pos, xerr=[[val - lo], [hi - val]], fmt="none",
            ecolor="#111111", elinewidth=1.25, capsize=3.0, capthick=1.0, zorder=3,
        )
        label_text = f"{val:.1f}%"
        if hi <= 91.0:
            axis.text(
                hi + 1.2, y_pos, label_text, va="center", ha="left",
                fontsize=base_pt - 0.1, fontweight="bold" if is_dope else "normal", color="#111111",
            )
        else:
            axis.text(
                max(2.0, lo - 1.2), y_pos, label_text, va="center", ha="right",
                fontsize=base_pt - 0.1, fontweight="bold", color="#ffffff" if is_dope else "#111111",
            )

    axis.set_xlim(0.0, 100.0)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=base_pt - 0.2)
    axis.set_ylim(-0.15, 5.75)
    axis.set_yticks(group_centers)
    axis.set_yticklabels(group_labels, fontsize=base_pt + 0.1, fontweight="bold")
    axis.set_xlabel(
        f"Predictive transfer retention (0–100%, whiskers: historical {percent}% interval)",
        fontsize=base_pt - 0.2, labelpad=4,
    )
    figure.suptitle(
        f_summary["hero_title"],
        x=0.50, y=0.955, ha="center", fontsize=base_pt + 0.8, fontweight="bold",
    )
    axis.grid(axis="x", alpha=0.22, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor="#2f6b25", edgecolor="#1a1a1a", linewidth=0.85,
              label=f"DOPE ({dope_bytes:,} B)"),
        Patch(facecolor="#7a7a7a", edgecolor="#1a1a1a", linewidth=0.85, hatch="//",
              label=f"Gaussian copula ({copula_bytes:,} B)"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.875), ncol=2,
                  fontsize=base_pt - 0.1, frameon=False, handlelength=1.5, columnspacing=1.6)
    figure.text(
        0.50, 0.025,
        "Historical validation (97 lineages, 4n) · Cross-model driver preservation: not yet measured",
        ha="center", va="bottom", fontsize=base_pt - 1.5, color="#222222",
    )
    return figure, plt


def figure_outputs(csv_bytes: bytes, thresholds: dict, percent: int | None = None) -> dict[str, bytes]:
    percent = interval_level() if percent is None else percent
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    failures = figure_bound_failures(rows, thresholds)
    if failures:
        raise ValueError("hero figure refuses out-of-axis values: " + "; ".join(failures))
    f_summary = foundation.load_foundation_summary()
    six_bars = foundation.hero_six_bar_rows(f_summary)
    for bar in six_bars:
        if not (0.0 <= bar["interval_lo_pct"] <= bar["retention_pct"] <= bar["interval_hi_pct"] <= 100.0):
            raise ValueError(f"hero figure refuses out-of-axis foundation bar: {bar}")

    outputs = {}
    for ext, profile, metadata in (
        ("png", "web", {"Software": "DOPE claims renderer"}),
        ("svg", "web", {"Creator": "DOPE claims renderer", "Date": None}),
        ("pdf", "paper", {"Creator": "DOPE claims renderer", "CreationDate": None, "ModDate": None}),
    ):
        figure, plt = _draw_foundation_hero(f_summary, six_bars, percent, profile=profile)
        buffer = io.BytesIO()
        figure.savefig(buffer, format=ext, dpi=180, metadata=metadata)
        plt.close(figure)
        raw = buffer.getvalue()
        if ext == "svg":
            raw = b"\n".join(line.rstrip(b" ") for line in raw.splitlines()) + b"\n"
        outputs[f"paper/generated/hero-frontier.{ext}"] = raw
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
