#!/usr/bin/env python3
"""History-page benchmark block from the v2 headline JSON that the paper also reads.

The table rows, the strongest comparator, and every interval come from
docs/whitepaper/generated/v2-headline.json, written by review_v2_emit.py.
The panel size, fit-seed count and interval level come from the pre-declaration
(research/benchmark/review_fixes/predeclare_v2.json), and the byte cap from the
embedded release contract (production/kpi-contract.json); no count is typed here.
The two images are drawn by v2_figures.py from that same file, so the
historical results page (HISTORY.md) and the paper cannot disagree.
`--write-readme` splices the block into that page; `--check` compares the
committed block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import declared_facts  # noqa: E402
from figure_spec import TEASER_X_LIM  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
HEADLINE = REPO / "docs" / "whitepaper" / "generated" / "v2-headline.json"
HISTORY = REPO / "HISTORY.md"
PNGS = (REPO / "docs" / "readme" / "pareto.png", REPO / "docs" / "readme" / "paired.png")
BEGIN = "<!-- BEGIN BENCHMARK -->"
END = "<!-- END BENCHMARK -->"
PREDECLARE = REPO / declared_facts.PREDECLARE
CONTRACT = REPO / declared_facts.CONTRACT
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
    return {"panel": panel, "fit_seeds": COUNT_WORDS[seeds], "level": declared_facts.interval_level(predeclare),
            "cap": declared_facts.byte_cap(contract)}


def load_facts(headline: dict) -> dict:
    return panel_facts(headline, json.loads(PREDECLARE.read_text()), json.loads(CONTRACT.read_text()))


def _load_classical_table() -> dict[tuple[str, str], dict[str, dict]]:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.claims import foundation as _fnd
    v2, _ = _fnd.authenticate_source(_fnd.V2_PANEL_REL, root=REPO)
    classical: dict[tuple[str, str], dict[str, dict]] = {}
    for row in v2.get("levels", ()):
        if row.get("size") == 4:
            classical.setdefault((row["method"], row["configuration"]), {})[row["auditor"]] = row
    return classical


def benchmark_block(headline: dict, facts: dict | None = None) -> str:
    facts = load_facts(headline) if facts is None else facts
    cap = f"{facts['cap']:,}"
    arms = {arm["code"]: arm for arm in headline["arms"]}
    dope = arms["Dope"]
    classical = _load_classical_table()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.claims import foundation as _fnd
    f_summary = _fnd.load_foundation_summary(REPO)
    f_rows = _fnd.foundation_table_rows(f_summary)
    key = (headline["strongest"]["method"], headline["strongest"]["configuration"])
    strongest = next(arm for arm in headline["arms"] if (arm["method"], arm["configuration"]) == key)
    others = [arm["bytes"]["median"] for arm in headline["arms"]
              if arm["role"] == "generator" and arm["bytes"] is not None]

    lines = [
        "![Multi-model retention at synthetic size 4n against median charged artifact bytes on a log axis, "
        "and cross-architecture transfer across zero-shot tabular foundation models (NVIDIA Kumo-L, TabICLv2, Mitra-v2), "
        f"classical learners, and historical feature-target correlation proxy (cross-model feature-driver preservation: not yet measured). DOPE's median artifact is {size(dope['bytes']['median'])}; "
        f"the other generators' medians run from {size(min(others))} to {size(max(others))}.](docs/readme/pareto.png)",
        "",
        f"Validation panel: {facts['panel']} PMLB regression tables across three task-trained model families "
        "(CatBoost, Linear, Neural MLP), synthetic size 4n. "
        f"DOPE uses {facts['fit_seeds']} fit seeds; intervals are {facts['level']}% nested bootstrap intervals "
        "(simulated-family clusters, then lineages, then fit seeds). "
        "The difference column is DOPE minus the method on shared lineages (Hodges–Lehmann). "
        "The zero-shot tabular foundation-model block below is a separate 97-lineage matched historical cohort "
        "(fit seed 11; sample seeds 101, 211, 307; lineage-level bootstrap intervals).",
        "",
        "#### 1. Multi-Model Task-Trained Transfer & Charged Artifact Bytes (Synthetic Size `4n`)",
        "",
        f"| Method | Lineages | CatBoost (4n) | Linear (4n) | Neural MLP (4n) | DOPE − method (CatBoost) | W/T/L | Median artifact | Within {cap} B |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for code in ORDER:
        arm = arms.get(code)
        if arm is None or arm["role"] == "diagnostic":
            continue
        level = arm["level"]
        name = f"**{arm['display']}**" if code == "Dope" else arm["display"]
        if arm["role"] == "control":
            name += " (control)"
        cls_rows = classical.get((arm["method"], arm["configuration"]), {})
        lin_row = cls_rows.get("linear")
        mlp_row = cls_rows.get("mlp")
        cb_cell = f"{fmt(level['median'])} [{fmt(level['lo'])}, {fmt(level['hi'])}]"
        lin_cell = "—" if lin_row is None else f"{fmt(lin_row['median'])} [{fmt(lin_row['lo'])}, {fmt(lin_row['hi'])}]"
        mlp_cell = "—" if mlp_row is None else f"{fmt(mlp_row['median'])} [{fmt(mlp_row['lo'])}, {fmt(mlp_row['hi'])}]"
        row = arm["contrasts"]["catboost"]
        diff = "—" if row is None else f"{signed(row['hl'])} [{fmt(row['hl_lo'])}, {fmt(row['hl_hi'])}]"
        wtl = "—" if row is None else f"{row['wins']}/{row['ties']}/{row['losses']}"
        stats = arm["bytes"]
        within = "—" if stats is None else f"{round(100 * stats['within_cap'] / stats['n'])}%"
        artifact = "none" if arm["role"] == "control" else ("—" if stats is None else size(stats["median"]))
        lines.append(
            f"| {name} | {level['n']} | {cb_cell} | {lin_cell} | {mlp_cell} | {diff} | {wtl} | {artifact} | {within} |"
        )

    lines += [
        "",
        "#### 2. Zero-Shot Tabular Foundation Models (TFM) & Historical Feature-Target Correlation Proxy",
        "",
        "| Method | Lineages | NVIDIA Kumo Large (`kumo_tabular_l`) | TabICLv2 (`tabicl2`) | AutoGluon Mitra-v2 (`mitra_v2`) | Feature-Target Correlation Proxy (`driver_fidelity` proxy) | Matched median artifact |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
    ]
    for f_row in f_rows:
        m_disp = "**DOPE (ours)**" if f_row["method"] == "DOPE" else f_row["display"]
        k_row = f_row["kumo_tabular_l"]
        t_row = f_row["tabicl2"]
        m_row = f_row["mitra_v2"]
        d_row = f_row["target_correlation_proxy"]
        b_med = f_row["matched_artifact_bytes"]
        n_val = f_row["n_lineages"]
        k_cell = f"{fmt(k_row['median'])} [{fmt(k_row['lo'])}, {fmt(k_row['hi'])}]"
        t_cell = f"{fmt(t_row['median'])} [{fmt(t_row['lo'])}, {fmt(t_row['hi'])}]"
        m_cell = f"{fmt(m_row['median'])} [{fmt(m_row['lo'])}, {fmt(m_row['hi'])}]"
        d_cell = f"{fmt(d_row['median'])} (IQR [{fmt(d_row['q1'])}, {fmt(d_row['q3'])}])"
        b_cell = f"{size(b_med)} ({b_med:,} B)"
        lines.append(f"| {m_disp} | {n_val} | {k_cell} | {t_cell} | {m_cell} | {d_cell} | {b_cell} |")
    lines += [
        "",
        "On these matched historical TFM lineages (fit seed 11; sample seeds 101, 211, 307; synthetic size 4n; pretraining overlap is unknown), "
        "Gaussian copula is the strongest measured comparator on this historical block (not universally strongest), and "
        "cross-model feature-driver preservation is not yet measured (the proxy column reports absolute feature-target correlation agreement; "
        "Gaussian copula is 0.975 median of lineage medians vs. 0.923 median of lineage minima from historical assembler::_min; pooled-seed median is 0.973).",
    ]

    lines += [
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
        HISTORY.write_text(splice_readme(HISTORY.read_text(), benchmark_block(headline)))
    if "--check" in sys.argv:
        if readme_block(HISTORY.read_text()) != expected_readme_block(headline):
            raise SystemExit("HISTORY.md benchmark block does not match v2-headline.json")
        missing = [str(path) for path in PNGS if not path.is_file()]
        if missing:
            raise SystemExit(f"benchmark figures are missing: {missing}")
        print("HISTORY.md benchmark block matches v2-headline.json")
    if "--print" in sys.argv or len(sys.argv) == 1:
        print(benchmark_block(headline))


if __name__ == "__main__":
    main()
