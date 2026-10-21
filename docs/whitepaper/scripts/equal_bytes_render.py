"""Render the equal-byte study: figures, paper macros and the README block.

Inputs are the committed study summary (research/benchmark/equal_bytes/results)
and, for the generator markers in Figure 1, the frozen v2 ledger
(research/benchmark/results/review-fixes-v2/scalar-cells.jsonl plus
docs/whitepaper/generated/v2-headline.json for median charged bytes).

    python3 docs/whitepaper/scripts/equal_bytes_render.py          # write
    python3 docs/whitepaper/scripts/equal_bytes_render.py --check  # text outputs must match
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from equal_bytes_text import (BLUE, GENERATED, GRAY, INK, MUTED, ORANGE, README_BEGIN, README_END, REPO, RESULTS, TAU,
                              coverage, scaling_rows, text_outputs, write_text)


def style(plt) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "dope",
                         "pdf.fonttype": 42})


def save(fig, stem: str) -> None:
    fig.savefig(GENERATED / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(GENERATED / f"{stem}.png", dpi=200, metadata={"Software": None})


def hero(summary: dict, generators: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    budgets = np.logspace(2, 9.3, 400)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True, sharey=True)
    order = ["DOPE", "Forest-Flow", "TabSyn", "ARF", "TVAE", "Gaussian copula", "Chow-Liu tree", "CTGAN",
             "Independent marginals"]
    shades = {"Forest-Flow": "#5b5a56", "TabSyn": "#74736f", "ARF": "#8a8986", "TVAE": "#a3a29d",
              "Gaussian copula": "#b8b7b2", "Chow-Liu tree": "#c9c8c3", "CTGAN": "#d4d3cf",
              "Independent marginals": "#dddcd8"}
    never = []
    # Tables a generator did not run on stay in the denominator, as the frontier defines.
    panel = max(point["lineages"] for point in generators)
    for point in sorted(generators, key=lambda g: order.index(g["label"].split(",")[0])):
        own = point["method"] == "DOPE"
        lineages = [[value] for value in point["per_lineage"].values()]
        values = coverage(lineages + [[]] * (panel - len(lineages)), TAU, budgets)
        if max(values) == 0.0:
            never.append(point["label"])
            continue
        label = "DOPE (published run)" if own else (
            point["label"] if point["lineages"] == panel else f"{point['label']} (ran on {point['lineages']})")
        left.step(budgets, values, where="post", color=BLUE if own else shades[point["label"]],
                  linewidth=2.2 if own else 1.3, label=label, zorder=3 if own else 2)
    if never:
        left.text(1.3e2, 0.72, "Never reach the bar: " + ", ".join(never), fontsize=6.3, color=MUTED)
    left.set_title(f"(a) Against published generators ({panel} tables)", fontsize=8.5, loc="left", color=INK)
    lineages = summary["lineages"]
    groups = [("2,000 fit rows", [x for x in lineages if x["n_fit"] >= 2000], "-", 2.1),
              ("fewer rows", [x for x in lineages if x["n_fit"] < 2000], ":", 1.6)]
    for group, members, dash, width in groups:
        for label, color, points in (
                ("DOPE program", BLUE, [list(x["dope"].values()) for x in members]),
                ("Real rows, best codec", ORANGE,
                 [[{"bytes": p["bytes"], "robust": p["robust"]} for p in x["data"]] for x in members])):
            right.step(budgets, coverage(points, TAU, budgets), where="post", color=color, linestyle=dash,
                       linewidth=width, label=f"{label}, {group} ({len(members)})")
    right.set_title(f"(b) Versus rows ({len(lineages)} mixed-origin tables)", fontsize=8.5,
                    loc="left", color=INK)
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlim(1e2, 2e9 if ax is left else 2e5)
        ax.set_ylim(0, 1.42)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(color="#e6e5e1", linewidth=0.6)
        ax.set_xlabel("Byte budget (log scale)")
        ax.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=2 if ax is left else 1,
                  columnspacing=0.8, handlelength=1.6)
    left.set_ylabel("Share of tables where every learner\nkeeps 80% of the real gain")
    save(fig, "transfer-per-byte")
    plt.close(fig)


def paired(summary: dict, panel: dict | None) -> None:
    """Equal-byte gap against how much larger the whole table is than the program."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    real = {x["dataset"] for x in summary["lineages"]}
    rows = panel["lineages"] if panel else summary["lineages"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9), constrained_layout=True)
    for simulated in (True, False):
        own = [x for x in rows if "headline" in x and (x["dataset"] not in real) == simulated]
        size = [x["full_table_bytes"]["8"] / x["headline"]["bytes"] for x in own]
        gap = [x["headline"]["robust"] - (x["headline"]["data_at_equal_bytes"] or 0.0) for x in own]
        if simulated:
            ax.scatter(size, gap, s=16, facecolor="white", edgecolor=GRAY, linewidth=0.9,
                       label=f"Other panel lineages ({len(own)})", zorder=2)
        else:
            large = [x["n_fit"] >= 2000 for x in own]
            ax.scatter([v for v, big in zip(size, large) if big], [g for g, big in zip(gap, large) if big], s=26,
                       color=BLUE, label=f"Historical cohort, 2,000 fit rows ({sum(large)})", zorder=3)
            ax.scatter([v for v, big in zip(size, large) if not big], [g for g, big in zip(gap, large) if not big],
                       s=26, color=ORANGE, label=f"Historical cohort, fewer rows ({len(large) - sum(large)})", zorder=3)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.axvline(1.0, color="#d4d3cf", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Whole table's bytes over the program's bytes (log scale)")
    ax.set_ylabel("Program minus real rows,\nworst-learner retention")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.grid(color="#e6e5e1", linewidth=0.6)
    save(fig, "equal-bytes-paired")
    plt.close(fig)


def scaling_figure(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    fig, ax = plt.subplots(figsize=(3.4, 2.8), constrained_layout=True)
    names = sorted({r["dataset"] for r in rows})
    for name in names:
        own = [r for r in rows if r["dataset"] == name]
        n = [r["n"] for r in own]
        ax.plot(n, [r["full_table_bytes"] for r in own], color=ORANGE, linewidth=1.1, alpha=0.75)
        ax.plot(n, [r["dope_bytes"] for r in own], color=BLUE, linewidth=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fit rows n (log scale)")
    ax.set_ylabel("Bytes (log scale)")
    ax.plot([], [], color=ORANGE, label="Whole table, best lossless codec")
    ax.plot([], [], color=BLUE, label="DOPE program (headline shape)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax.grid(color="#e6e5e1", linewidth=0.6)
    save(fig, "equal-bytes-scaling")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-readme", action="store_true", help="also replace the README block in place")
    args = parser.parse_args()
    failures: list[str] = []
    for name, text in text_outputs().items():
        write_text(REPO / name, text, args.check, failures)
    if not args.check:
        summary = json.loads((RESULTS / "summary.json").read_text())
        generators = json.loads((RESULTS / "generators.json").read_text())
        hero(summary, generators)
        panel_path = RESULTS / "panel-summary.json"
        paired(summary, json.loads(panel_path.read_text()) if panel_path.is_file() else None)
        scaling_figure(scaling_rows())
    if args.write_readme:
        readme = REPO / "README.md"
        text = readme.read_text()
        start, end = text.index(README_BEGIN), text.index(README_END) + len(README_END)
        block = (REPO / "paper/generated/readme-equal-bytes.md").read_text().strip()
        readme.write_text(text[:start] + block + text[end:])
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
