#!/usr/bin/env python3
"""README benchmark block from the committed density table.

The markdown cells are the density-table cells. The PNG is
``draw_retention_bars(..., readme=True)``, the same bars as the paper PDF.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DENSITY = REPO / "docs" / "whitepaper" / "generated" / "density-table.tex"
PANEL = REPO / "docs" / "whitepaper" / "generated" / "panel-stats.json"
README = REPO / "README.md"
PNG = REPO / "docs" / "readme" / "retention-bars.png"
BEGIN = "<!-- BEGIN BENCHMARK -->"
END = "<!-- END BENCHMARK -->"
_COMPARATORS = {
    "Gaussian copula": "Gaussian copula",
    "Chow--Liu": "Chow–Liu",
    "Indep.\\ marginals": "Independent marginals",
}


def parse_density_rows(tex):
    """Return the eight-cell rows of the density table, in file order."""
    rows = []
    for line in tex.splitlines():
        stripped = line.strip()
        if not stripped.endswith("\\\\"):
            continue
        if stripped.startswith("\\") or stripped.startswith("Auditor"):
            continue
        parts = [part.strip() for part in stripped[:-2].split("&")]
        if len(parts) != 8:
            raise ValueError(f"density row does not have 8 cells: {stripped}")
        rows.append(parts)
    if len(rows) != 9:
        raise ValueError(f"density table has {len(rows)} rows, expected 9")
    return rows


def _plain_holm(cell):
    text = cell.replace("$", "").replace("\\times", "×")
    return text.replace("10^{", "10^").replace("}", "")


def benchmark_block(tex):
    """CatBoost markdown plus the linear non-separation sentence."""
    rows = parse_density_rows(tex)
    catboost = [row for row in rows if row[0] == "CatBoost"]
    if len(catboost) != 3:
        raise ValueError("CatBoost block is not three comparators")
    counts = {row[2] for row in catboost}
    if counts != {"97"}:
        raise ValueError(f"CatBoost lineage count is {counts}")
    lines = [
        "CatBoost retention on 97 lineages, synthetic size 4n, fit seed 11. Each cell is a dimensionless retention.",
        "",
        "| Comparator | n | DOPE | Other | Difference | W/T/L | Holm p |",
        "| --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in catboost:
        comparator = _COMPARATORS.get(row[1])
        if comparator is None:
            raise ValueError(f"unexpected CatBoost comparator: {row[1]}")
        lines.append(
            "| "
            + " | ".join(
                (
                    comparator,
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    _plain_holm(row[7]),
                )
            )
            + " |"
        )
    linear = next(
        row
        for row in rows
        if row[0] == "Linear" and row[1] == "Gaussian copula"
    )
    lines.extend(
        [
            "",
            (
                "The linear auditor does not separate DOPE from the Gaussian copula "
                f"(n={linear[2]}, difference {linear[5]}, W/T/L {linear[6]}, "
                f"Holm p {_plain_holm(linear[7])}). That interval contains zero. "
                "MFS-v2 and MFS-v3 are null on every method in the manuscript."
            ),
        ]
    )
    return "\n".join(lines)


def splice_readme(readme, block):
    if readme.count(BEGIN) != 1 or readme.count(END) != 1:
        raise ValueError("benchmark markers are not unique")
    start = readme.index(BEGIN) + len(BEGIN)
    finish = readme.index(END)
    if finish < start:
        raise ValueError("benchmark markers are out of order")
    return readme[:start] + "\n" + block.rstrip() + "\n" + readme[finish:]


def readme_block(readme):
    start = readme.index(BEGIN)
    finish = readme.index(END) + len(END)
    return readme[start:finish]


def expected_readme_block(tex):
    return BEGIN + "\n" + benchmark_block(tex).rstrip() + "\n" + END


def write_png(dest=None):
    from render_figures import GEN, draw_retention_bars, load_json

    stats = load_json(GEN / "panel-stats.json")
    path = Path(dest) if dest is not None else PNG
    return draw_retention_bars(stats, path, readme=True)


def main():
    import sys

    tex = DENSITY.read_text()
    block = benchmark_block(tex)
    if "--write-readme" in sys.argv:
        README.write_text(splice_readme(README.read_text(), block))
    if "--write-png" in sys.argv:
        print(write_png())
    if "--check" in sys.argv:
        actual = readme_block(README.read_text())
        expect = expected_readme_block(tex)
        if actual != expect:
            raise SystemExit("README benchmark block does not match density-table.tex")
        if not PNG.is_file():
            raise SystemExit("README retention figure is missing")
    if "--print" in sys.argv or len(sys.argv) == 1:
        print(block)


if __name__ == "__main__":
    main()
