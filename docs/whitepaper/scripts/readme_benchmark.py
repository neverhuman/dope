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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import declared_facts  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HEADLINE = REPO / "docs" / "whitepaper" / "generated" / "v2-headline.json"
README = REPO / "README.md"
PNGS = (REPO / "docs" / "readme" / "pareto.png", REPO / "docs" / "readme" / "paired.png")
BEGIN = "<!-- BEGIN BENCHMARK -->"
END = "<!-- END BENCHMARK -->"
PREDECLARE = REPO / declared_facts.PREDECLARE
CONTRACT = REPO / declared_facts.CONTRACT
COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def fmt(value, places=3) -> str:
    if value is None:
        return "—"
    text = f"{value:.{places}f}"
    return text.replace("-", "−") if not text.startswith("-0.000") else "0.000"


def signed(value) -> str:
    return ("+" if value is not None and value > 0 else "") + fmt(value)


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
    return {"panel": panel, "fit_seeds": COUNT_WORDS[seeds], "level": declared_facts.interval_level(predeclare),
            "cap": declared_facts.byte_cap(contract)}


def load_facts(headline: dict) -> dict:
    return panel_facts(headline, json.loads(PREDECLARE.read_text()), json.loads(CONTRACT.read_text()))


def benchmark_block(headline: dict, facts: dict | None = None) -> str:
    facts = load_facts(headline) if facts is None else facts
    arms = {arm["code"]: arm for arm in headline["arms"]}
    dope = arms["Dope"]
    key = (headline["strongest"]["method"], headline["strongest"]["configuration"])
    strongest = next(arm for arm in headline["arms"] if (arm["method"], arm["configuration"]) == key)
    contrast = strongest["contrasts"]["catboost"]
    level = dope["level"]
    lines = [
        "![Historical validation retention at synthetic size 4n versus median charged artifact bytes (log x-axis); points show models and feature-driver agreement.](docs/readme/pareto.png)",
        "",
        f"Historical panel: {facts['panel']} PMLB regression tables, {facts['fit_seeds']} DOPE fit seeds, "
        f"{facts['level']}% nested intervals. At 4n, DOPE CatBoost retention is "
        f"{fmt(level['median'])} [{fmt(level['lo'])}, {fmt(level['hi'])}]; strongest full-panel comparator "
        f"{strongest['display']} is {fmt(strongest['level']['median'])} "
        f"[{fmt(strongest['level']['lo'])}, {fmt(strongest['level']['hi'])}]. "
        f"Paired DOPE-minus-comparator difference: {signed(contrast['hl'])} "
        f"[{fmt(contrast['hl_lo'])}, {fmt(contrast['hl_hi'])}]. These are selection-exposed validation estimates; "
        "the official test is sealed, MFS-v3 is null, and copy and inference gates fail on some cells.",
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
