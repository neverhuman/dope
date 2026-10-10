#!/usr/bin/env python3
"""TabSyn retention next to DOPE on the lineages both actually measured.

Reads metric JSON from the predeclared sample root. A lineage with no artifact
stays absent. Absence is not a win. The published-sample parser check has to
have matched before a table is emitted. Official test files are not opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from research.benchmark.review_fixes.review_tex import longtable  # noqa: E402
from research.benchmark.review_fixes.receipt_panel import (  # noqa: E402
    AUDITORS,
    GENERATED,
    RESULTS,
    _clean,
    _load,
    ci,
    fmt,
    marginal,
    names_of,
    paired,
    primary_map,
    reduce_cells,
    tex_name,
    write_tex,
)
from research.benchmark.review_fixes.stats import holm  # noqa: E402

CONFIGURATION = "scaled_200_vae_1000_diffusion"
PANEL = RESULTS / "review-fixes-tabsyn-v1" / "panel.json"


def _metrics(root: Path) -> list[dict]:
    found = []
    for path in sorted(root.rglob("sample-*.metric.json")):
        try:
            found.append(json.loads(path.read_text()))
        except json.JSONDecodeError as error:
            raise SystemExit(f"malformed TabSyn metric {path.name}") from error
    if not found:
        raise SystemExit("no TabSyn metric cells")
    return found


def _adapt(cells: list[dict], names: dict) -> tuple[list[dict], list[str]]:
    adapted = []
    seen = set()
    for cell in cells:
        dataset = cell.get("dataset")
        if dataset not in names:
            raise SystemExit(f"metric dataset is outside the lineage record: {dataset}")
        if cell.get("official_tests_opened") is True:
            raise SystemExit("TabSyn metric opened an official test")
        seen.add(dataset)
        adapted.append({
            "method": "TabSyn",
            "configuration": CONFIGURATION,
            "dataset": dataset,
            "size_multiplier": cell.get("size"),
            "sample_seed": cell.get("sample_seed"),
            "status": cell.get("status"),
            "utility": cell.get("utility"),
            "null_loss": cell.get("null_loss"),
        })
    absent = sorted(set(names) - seen)
    return adapted, absent


def _levels(method: str, configuration: str, values: dict, names: dict, cohort: str) -> list[dict]:
    rows = []
    for size in (1, 4):
        for auditor in AUDITORS:
            summary = marginal(
                values[size][auditor], names,
                f"tabsyn|{cohort}|{method}|{auditor}|{size}",
            )
            summary.update({
                "method": method,
                "configuration": configuration,
                "cohort": cohort,
                "size": size,
                "auditor": auditor,
            })
            rows.append(summary)
    return rows


def _contrasts(dope: dict, tabsyn: dict, names: dict) -> list[dict]:
    rows = []
    for size in (1, 4):
        drafted = []
        family = []
        for auditor in AUDITORS:
            item = paired(
                dope[size][auditor],
                tabsyn[size][auditor],
                names,
                f"tabsyn-paired|{size}|{auditor}",
            )
            item.update({
                "method": "TabSyn",
                "configuration": CONFIGURATION,
                "size": size,
                "auditor": auditor,
                "contrast": "DOPE minus TabSyn",
            })
            drafted.append(item)
            family.append(item["wilcoxon_p"])
        for item, p_value in zip(drafted, holm(family)):
            item["holm_p"] = p_value
            item["holm_family_n"] = len(family)
            rows.append(item)
    return rows


def _matched(dope: dict, tabsyn: dict) -> dict:
    matched = {size: {auditor: {} for auditor in AUDITORS} for size in (1, 4)}
    for size in (1, 4):
        for auditor in AUDITORS:
            keys = set(dope[size][auditor]) & set(tabsyn[size][auditor])
            matched[size][auditor] = {key: tabsyn[size][auditor][key] for key in keys}
    return matched


def _matched_dope(dope: dict, tabsyn: dict) -> dict:
    matched = {size: {auditor: {} for auditor in AUDITORS} for size in (1, 4)}
    for size in (1, 4):
        for auditor in AUDITORS:
            keys = set(dope[size][auditor]) & set(tabsyn[size][auditor])
            matched[size][auditor] = {key: dope[size][auditor][key] for key in keys}
    return matched


def render_tabsyn(payload: dict) -> str:
    levels = []
    for row in payload["levels"]:
        levels.append(
            f"{tex_name(row['method'])} & {tex_name(row['configuration'])} & {tex_name(row['cohort'])} & "
            f"{row['size']}$n$ & {tex_name(row['auditor'])} & {row['n']} & {fmt(row['median'])} & {ci(row)} \\\\"
        )
    contrasts = []
    for row in payload["contrasts"]:
        flag = "yes" if row["tost"]["equivalent"] else "no"
        contrasts.append(
            f"{row['size']}$n$ & {tex_name(row['auditor'])} & {tex_name(row['method'])} & "
            f"{tex_name(row['configuration'])} & {row['n']} & {fmt(row['median'])} & {ci(row)} & "
            f"{fmt(row['mean'])} & {flag} & {row['wins']}/{row['ties']}/{row['losses']} & "
            f"{fmt(row['holm_p'], 3)} \\\\"
        )
    absent = ", ".join(payload["artifact_absent"]) or "none"
    levels_caption = (
        "\\textbf{TabSyn and DOPE retention levels, fit seed 11.} TabSyn runs the scaled schedule (200 VAE and "
        "1{,}000 diffusion epochs). A lineage median needs three informative sample seeds at size $n$ or $4n$; "
        "own-lineage rows use every informative TabSyn lineage, and matched rows use the lineages shared with "
        "informative DOPE features12\\_\\allowbreak steps2048. Intervals are the family-cluster bootstrap. "
        f"Lineages absent from the admitted cohort: {tex_name(absent)}; absence is no win. No Holm family."
    )
    contrasts_caption = (
        "\\textbf{DOPE minus TabSyn, fit seed 11.} Median paired retention difference on the matched lineages "
        "with its family-cluster bootstrap interval, the mean difference, the v1 mean-difference equivalence test at "
        "$\\pm 0.02$, and W/T/L. The Holm family has three tests per size (one comparator, three auditors), separate "
        "from every other comparator block; the Wilcoxon and equivalence tests treat lineages as independent."
    )
    return longtable(
        levels_caption,
        "Method & Configuration & Cohort & Size & Auditor & $n$ lineages & Median & Hierarchical CI",
        levels,
        "lllllrrr",
        label="tab:review-tabsyn",
    ) + longtable(
        contrasts_caption,
        "Size & Auditor & Comparator & Configuration & $n$ & Median diff. & Hierarchical CI & Mean & TOST & W/T/L & Holm $p$",
        contrasts,
        "lllllrrllrl",
        label="tab:review-tabsyn-contrasts",
    )


def payload_from_cells(cells: list[dict], gate: dict, sources=None) -> dict:
    if gate.get("match") is not True or gate.get("official_tests_opened") is True:
        raise SystemExit("parser check did not match; refusing to emit a TabSyn table")
    record = json.loads(sources["research/benchmark/results/s3-lineage-record.json"]) if sources is not None else _load("s3-lineage-record.json")[0]
    names = names_of(record)
    adapted, absent = _adapt(cells, names)
    tabsyn = primary_map(reduce_cells(adapted, "TabSyn", CONFIGURATION))
    density = json.loads(sources["research/benchmark/results/density-matched-population-validation.json"]) if sources is not None else _load("density-matched-population-validation.json")[0]
    dope = primary_map(reduce_cells(density["cells"], "DOPE", "features12_steps2048"))
    predeclare = json.loads(sources["research/benchmark/review_fixes/predeclare.json"]) if sources is not None else json.loads((REPO / "research/benchmark/review_fixes/predeclare.json").read_text())
    if predeclare["tabsyn"]["config_sha256"] != "a34729383eac6783c7e754a791094054d54f1062655db9a306cd4cca296c5ebc":
        raise SystemExit("TabSyn config pin differs from the sampler")
    if predeclare["tabsyn"]["sample_seeds"] != [101, 211, 307] or predeclare["tabsyn"]["sizes"] != [1, 4]:
        raise SystemExit("TabSyn seed or size pin differs from the sampler")
    levels = _levels("TabSyn", CONFIGURATION, tabsyn, names, "own lineages")
    matched_tabsyn = _matched(dope, tabsyn)
    matched_dope = _matched_dope(dope, tabsyn)
    levels.extend(_levels("TabSyn", CONFIGURATION, matched_tabsyn, names, "matched lineages"))
    levels.extend(_levels("DOPE", "features12_steps2048", matched_dope, names, "matched lineages"))
    return _clean({
        "format": "dope-review-fix-tabsyn-v1",
        "version": 1,
        "official_tests_opened": False,
        "formal_dp": False,
        "missing_is_not_a_win": True,
        "configuration": CONFIGURATION,
        "artifact_absent": absent,
        "parser_check": {
            "dataset": gate.get("dataset"),
            "sample_seed": gate.get("sample_seed"),
            "row_multiplier": gate.get("row_multiplier"),
            "match": True,
            "official_tests_opened": False,
        },
        "levels": levels,
        "contrasts": _contrasts(dope, tabsyn, names),
        "claims": {"mfs_v2": None, "ptf_v1": None, "release_safe_l3": None, "superiority": None, "formal_dp": False},
    })


def _payload(cells_root: Path, parser_check: Path) -> dict:
    return payload_from_cells(_metrics(cells_root), json.loads(parser_check.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path)
    parser.add_argument("--parser-check", type=Path)
    parser.add_argument("--from-panel", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.from_panel or (args.check and args.cells is None):
        panel_path = args.from_panel or PANEL
        payload = json.loads(panel_path.read_text())
        rendered = render_tabsyn(payload)
        tex_path = GENERATED / "review-tabsyn.tex"
        json_path = GENERATED / "review-tabsyn.json"
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.check:
            if not tex_path.is_file() or tex_path.read_text() != rendered:
                raise SystemExit("review-tabsyn.tex does not match the panel")
            if not json_path.is_file() or json_path.read_text() != text:
                raise SystemExit("review-tabsyn.json does not match the panel")
            print("tabsyn check ok")
            return
        if text != panel_path.read_text():
            raise SystemExit("tabsyn panel is not canonical json")
        json_path.write_text(text)
        write_tex("review-tabsyn.tex", rendered)
        return
    if args.cells is None or args.parser_check is None:
        raise SystemExit("TabSyn emission needs --cells and --parser-check")
    payload = _payload(args.cells, args.parser_check)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    PANEL.write_text(text)
    (GENERATED / "review-tabsyn.json").write_text(text)
    write_tex("review-tabsyn.tex", render_tabsyn(payload))
    for row in payload["levels"]:
        if row["auditor"] == "catboost" and row["cohort"] in ("own lineages", "matched lineages"):
            print(
                f"{row['method']} {row['cohort']} CatBoost size {row['size']}n "
                f"median {fmt(row['median'])} {ci(row)} on {row['n']} lineages"
            )
    for row in payload["contrasts"]:
        if row["auditor"] == "catboost":
            print(
                f"DOPE minus TabSyn CatBoost size {row['size']}n "
                f"median {fmt(row['median'])} {ci(row)} on {row['n']} lineages"
            )
    print("artifact_absent", " ".join(payload["artifact_absent"]) or "none")


if __name__ == "__main__":
    main()
