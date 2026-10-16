#!/usr/bin/env python3
"""README benchmark block from the v2 headline JSON that the paper also reads.

The table rows, the strongest comparator, and every interval come from
docs/whitepaper/generated/v2-headline.json, written by review_v2_emit.py.
The panel size, fit-seed count and interval level come from the pre-declaration
(research/benchmark/review_fixes/predeclare_v2.json), and the byte cap from the
embedded release contract (production/kpi-contract.json); no count is typed here.
The two images are drawn by v2_figures.py from that same file, so the README
and the paper cannot disagree. `--check` compares the committed block.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_spec import TEASER_X_LIM  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HEADLINE = REPO / "docs" / "whitepaper" / "generated" / "v2-headline.json"
README = REPO / "README.md"
PNGS = (REPO / "docs" / "readme" / "pareto.png", REPO / "docs" / "readme" / "paired.png")
BEGIN = "<!-- BEGIN BENCHMARK -->"
END = "<!-- END BENCHMARK -->"
PREDECLARE = REPO / "research" / "benchmark" / "review_fixes" / "predeclare_v2.json"
CONTRACT = REPO / "production" / "kpi-contract.json"
COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
ORDER = ("Dope", "TabSyn", "Arf", "ArfNat", "Gauss", "Chow", "Ind", "Tvae", "Ctgan", "Forest", "TabDdpm",
         "TabDiff", "Great", "Synthpop", "Smote", "Boot", "Pred")


def fmt(value, places=3) -> str:
    if value is None:
        return "—"
    text = f"{value:.{places}f}"
    return text.replace("-", "−") if not text.startswith("-0.000") else "0.000"


def signed(value) -> str:
    return ("+" if value is not None and value > 0 else "") + fmt(value)


def size(value) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{int(round(value))} B"


def panel_facts(headline: dict, predeclare: dict, contract: dict) -> dict:
    """Panel size, fit seeds, interval level and byte cap from their sources; disagreement fails."""
    sizes = {method["lineages"] for method in predeclare["methods"]["primary"].values()}
    if len(sizes) != 1:
        raise ValueError("predeclare_v2 primary methods disagree on the panel size")
    panel = sizes.pop()
    dope = next(arm for arm in headline["arms"] if arm["code"] == "Dope")
    if dope["bytes"]["n"] != panel:
        raise ValueError("v2-headline.json DOPE lineage count differs from the declared panel")
    seeds = len(predeclare["dope"]["fit_seeds"])
    if seeds not in COUNT_WORDS:
        raise ValueError("fit-seed count has no README wording")
    match = re.search(r"percentile (\d+)% interval", predeclare["estimands"]["uncertainty"])
    if match is None:
        raise ValueError("predeclare_v2 does not state the interval level")
    return {"panel": panel, "fit_seeds": COUNT_WORDS[seeds], "level": int(match.group(1)),
            "cap": contract["tier_byte_limits"]["l3"]}


def load_facts(headline: dict) -> dict:
    return panel_facts(headline, json.loads(PREDECLARE.read_text()), json.loads(CONTRACT.read_text()))


def benchmark_block(headline: dict, facts: dict | None = None) -> str:
    facts = load_facts(headline) if facts is None else facts
    cap = f"{facts['cap']:,}"
    arms = {arm["code"]: arm for arm in headline["arms"]}
    dope = arms["Dope"]
    lines = [
        f"Validation panel: {facts['panel']} PMLB regression tables, CatBoost auditor, synthetic size 4n. "
        f"DOPE uses {facts['fit_seeds']} fit seeds; intervals are {facts['level']}% nested bootstrap intervals "
        "(simulated-family clusters, then lineages, then fit seeds). "
        "The difference column is DOPE minus the method on shared lineages (Hodges–Lehmann).",
        "",
        f"| Method | Lineages | CatBoost retention, 4n | DOPE − method | W/T/L | Median artifact | Lineage medians within {cap} B |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for code in ORDER:
        arm = arms.get(code)
        if arm is None or arm["role"] == "diagnostic":
            continue
        level = arm["level"]
        name = f"**{arm['display']}**" if code == "Dope" else arm["display"]
        if arm["role"] == "control":
            name += " (control)"
        row = arm["contrasts"]["catboost"]
        diff = "—" if row is None else f"{signed(row['hl'])} [{fmt(row['hl_lo'])}, {fmt(row['hl_hi'])}]"
        wtl = "—" if row is None else f"{row['wins']}/{row['ties']}/{row['losses']}"
        stats = arm["bytes"]
        within = "—" if stats is None else f"{round(100 * stats['within_cap'] / stats['n'])}%"
        artifact = "none" if arm["role"] == "control" else ("—" if stats is None else size(stats["median"]))
        lines.append(f"| {name} | {level['n']} | {fmt(level['median'])} [{fmt(level['lo'])}, {fmt(level['hi'])}] "
                     f"| {diff} | {wtl} | {artifact} | {within} |")
    key = (headline["strongest"]["method"], headline["strongest"]["configuration"])
    strongest = next(arm for arm in headline["arms"] if (arm["method"], arm["configuration"]) == key)
    others = [arm["bytes"]["median"] for arm in headline["arms"]
              if arm["role"] == "generator" and arm["bytes"] is not None]
    lines += [
        "",
        "![CatBoost retention at synthetic size 4n against median charged artifact bytes on a log axis. "
        f"DOPE's median artifact is {size(dope['bytes']['median'])}; the other generators' medians run from "
        f"{size(min(others))} to {size(max(others))}.](docs/readme/pareto.png)",
        "",
        f"![DOPE minus each comparator, Hodges–Lehmann retention difference with {facts['level']}% nested intervals, "
        "under the CatBoost, linear, and MLP auditors.](docs/readme/paired.png)",
        "",
        f"In the paired chart the axis runs from {TEASER_X_LIM[0]:g} to {TEASER_X_LIM[1]:g}: an interval end beyond it "
        "is drawn as an arrow at the edge, and a point beyond it is drawn at the edge with its value printed.",
        "",
        f"The strongest full-panel comparator is {strongest['display']}. These are validation estimates on a "
        "training-derived cut; the official test split is reserved for one pre-registered run. Resampled real "
        "rows are a row-count diagnostic, neither a generator nor an upper bound. The release score (MFS-v3) is "
        "null for every method "
        "because its prerequisites and cleartext scan are unmeasured, and measured copy, membership, and "
        "attribute gates already fail on some cells. DOPE's median artifact is "
        f"{size(dope['bytes']['median'])}, and {dope['bytes']['fits_within_cap']} of {dope['bytes']['fits_ok']} "
        f"individual DOPE fits are within the {cap}-byte release cap; the paper names the lineages above it.",
    ]
    return "\n".join(lines)


def splice_readme(readme: str, block: str) -> str:
    if readme.count(BEGIN) != 1 or readme.count(END) != 1:
        raise ValueError("benchmark markers are not unique")
    start = readme.index(BEGIN) + len(BEGIN)
    finish = readme.index(END)
    if finish < start:
        raise ValueError("benchmark markers are out of order")
    return readme[:start] + "\n" + block.rstrip() + "\n" + readme[finish:]


def readme_block(readme: str) -> str:
    return readme[readme.index(BEGIN):readme.index(END) + len(END)]


def expected_readme_block(headline: dict) -> str:
    return BEGIN + "\n" + benchmark_block(headline).rstrip() + "\n" + END


def main() -> None:
    headline = json.loads(HEADLINE.read_text())
    if "--write-readme" in sys.argv:
        README.write_text(splice_readme(README.read_text(), benchmark_block(headline)))
    if "--check" in sys.argv:
        if readme_block(README.read_text()) != expected_readme_block(headline):
            raise SystemExit("README benchmark block does not match v2-headline.json")
        missing = [str(path) for path in PNGS if not path.is_file()]
        if missing:
            raise SystemExit(f"README figures are missing: {missing}")
        print("README benchmark block matches v2-headline.json")
    if "--print" in sys.argv or len(sys.argv) == 1:
        print(benchmark_block(headline))


if __name__ == "__main__":
    main()
