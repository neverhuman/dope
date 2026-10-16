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
from figure_spec import TEASER_X_LIM  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HEADLINE = REPO / "docs" / "whitepaper" / "generated" / "v2-headline.json"
README = REPO / "README.md"
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


V2_PANEL = REPO / "research" / "benchmark" / "results" / "review-fixes-v2" / "panel.json"
MFS_PANEL = REPO / "research" / "benchmark" / "results" / "mfs-v3-density-panel.json"


def _load_extra_tables() -> tuple[dict, dict, dict]:
    classical: dict[tuple[str, str], dict[str, dict]] = {}
    if V2_PANEL.is_file():
        v2 = json.loads(V2_PANEL.read_text())
        for row in v2.get("levels", ()):
            if row.get("size") == 4:
                classical.setdefault((row["method"], row["configuration"]), {})[row["auditor"]] = row
    tfm: dict[str, dict] = {}
    driver: dict[str, tuple[float, float, float, int]] = {}
    if MFS_PANEL.is_file():
        mfs = json.loads(MFS_PANEL.read_text())
        tfm = mfs.get("retention_summary", {})
        by_m: dict[str, list[float]] = {}
        for lin in mfs.get("lineages", ()):
            vals = sorted(
                float(s["holdout"]["driver_fidelity"])
                for s in lin.get("seeds", ())
                if isinstance(s.get("holdout", {}).get("driver_fidelity"), (int, float))
            )
            if vals:
                mid = len(vals) // 2
                med = vals[mid] if len(vals) % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])
                by_m.setdefault(lin["method"], []).append(med)
        for m, vals in by_m.items():
            vals.sort()
            n = len(vals)
            mid = n // 2
            med = vals[mid] if n % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])
            q1 = vals[max(0, int(round(0.25 * (n - 1))))]
            q3 = vals[min(n - 1, int(round(0.75 * (n - 1))))]
            driver[m] = (med, q1, q3, n)
    return classical, tfm, driver


def benchmark_block(headline: dict, facts: dict | None = None) -> str:
    facts = load_facts(headline) if facts is None else facts
    cap = f"{facts['cap']:,}"
    arms = {arm["code"]: arm for arm in headline["arms"]}
    dope = arms["Dope"]
    classical, tfm, driver = _load_extra_tables()
    key = (headline["strongest"]["method"], headline["strongest"]["configuration"])
    strongest = next(arm for arm in headline["arms"] if (arm["method"], arm["configuration"]) == key)
    others = [arm["bytes"]["median"] for arm in headline["arms"]
              if arm["role"] == "generator" and arm["bytes"] is not None]

    lines = [
        "![Multi-model retention at synthetic size 4n against median charged artifact bytes on a log axis, "
        "and cross-architecture transfer across zero-shot tabular foundation models (NVIDIA Kumo-L, TabICLv2, Mitra-v2), "
        f"classical learners, and feature driver fidelity. DOPE's median artifact is {size(dope['bytes']['median'])}; "
        f"the other generators' medians run from {size(min(others))} to {size(max(others))}.](docs/readme/pareto.png)",
        "",
        f"Validation panel: {facts['panel']} PMLB regression tables across three task-trained model families "
        "(CatBoost, Linear, Neural MLP) and three zero-shot tabular foundation models "
        "(NVIDIA Kumo Tabular Large, TabICLv2, AutoGluon Mitra-v2), synthetic size 4n. "
        f"DOPE uses {facts['fit_seeds']} fit seeds; intervals are {facts['level']}% nested bootstrap intervals "
        "(simulated-family clusters, then lineages, then fit seeds). "
        "The difference column is DOPE minus the method on shared lineages (Hodges–Lehmann).",
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

    if tfm:
        lines += [
            "",
            "#### 2. Zero-Shot Tabular Foundation Models (TFM) & Holdout Feature Driver Fidelity",
            "",
            "| Method | Lineages | NVIDIA Kumo Large (`kumo_tabular_l`) | TabICLv2 (`tabicl2`) | AutoGluon Mitra-v2 (`mitra_v2`) | Holdout Feature Driver Fidelity | Median artifact |",
            "| --- | ---: | --- | --- | --- | --- | ---: |",
        ]
        tfm_arms = (
            ("DOPE", "**DOPE (ours)**", arms["Dope"]["bytes"]),
            ("GaussianCopula", "Gaussian copula", arms["Gauss"]["bytes"]),
            ("Chow-Liu", "Chow-Liu tree", arms["Chow"]["bytes"]),
            ("independent_marginals", "Independent marginals", arms["Ind"]["bytes"]),
        )
        for m_key, m_disp, b_stats in tfm_arms:
            k_row = tfm.get("kumo_tabular_l", {}).get("methods", {}).get(m_key)
            t_row = tfm.get("tabicl2", {}).get("methods", {}).get(m_key)
            m_row = tfm.get("mitra_v2", {}).get("methods", {}).get(m_key)
            d_row = driver.get(m_key)
            n_val = k_row["n"] if k_row else "—"
            k_cell = "—" if not k_row else f"{fmt(k_row['median'])} [{fmt(k_row['lo'])}, {fmt(k_row['hi'])}]"
            t_cell = "—" if not t_row else f"{fmt(t_row['median'])} [{fmt(t_row['lo'])}, {fmt(t_row['hi'])}]"
            m_cell = "—" if not m_row else f"{fmt(m_row['median'])} [{fmt(m_row['lo'])}, {fmt(m_row['hi'])}]"
            d_cell = "—" if not d_row else f"{fmt(d_row[0])} (IQR [{fmt(d_row[1])}, {fmt(d_row[2])}])"
            b_cell = "—" if not b_stats else size(b_stats["median"])
            lines.append(f"| {m_disp} | {n_val} | {k_cell} | {t_cell} | {m_cell} | {d_cell} | {b_cell} |")

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
