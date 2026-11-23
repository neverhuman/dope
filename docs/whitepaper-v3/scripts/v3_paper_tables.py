"""Results tables of the DOPE v3.0.0 paper, rendered from `dope_kpi` receipts.

The cell strings follow `dope_kpi render-readme` (`engine/dope-arena/src/kpi/render.rs`)
character for character, so the paper's table can be compared with the README block
(`bench/real-world/results/scoreboard/v3.0.0/readme-block.md`) cell by cell before any
LaTeX is written. A mismatch, a missing row or an extra row is an error, never a warning.
"""
from __future__ import annotations

import re
from typing import Any

TASKS = ("regression", "binary", "multiclass")
GUARD_NAMES = {
    "task_ci_low": "tasks",
    "mlp_delta": "mlp",
    "realism_coverage_delta": "realism",
    "privacy_coverage_delta": "privacy",
    "exact_copy_excess": "copies",
}
CHECK_WORDS = {"pass": "pass", "fail": "FAIL", "not_evaluated": "n/e"}


class TableError(ValueError):
    """The receipts and the README block disagree, or a receipt is malformed."""


def fixed3(x: float) -> str:
    text = f"{x:.3f}"
    return "0.000" if text == "-0.000" else text


def signed3(x: float) -> str:
    text = fixed3(x)
    return text if text == "0.000" or text.startswith("-") else "+" + text


def opt3(x: Any) -> str:
    return "-" if x is None else fixed3(x)


def p_value(p: float) -> str:
    return "<0.001" if p < 0.001 else fixed3(p)


def bytes_cell(arm: dict[str, Any]) -> str:
    b = arm["bytes"]
    median, share = b.get("median"), b.get("share_within_budget")
    if median is None or share is None:
        return "-"
    med = f"{median:.0f}" if float(median).is_integer() else f"{median:.1f}"
    return f"{med} / {b['max']} / {fixed3(share)}"


def guard_cell(arm: dict[str, Any]) -> str:
    copies = arm.get("exact_copy")
    copy_text = (
        f"{fixed3(copies['mean'])} ({fixed3(copies['max'])})"
        if isinstance(copies, dict)
        else opt3(arm.get("exact_copy_excess"))
    )
    return f"{opt3(arm.get('realism_coverage'))} / {opt3(arm.get('privacy_coverage'))} / {copy_text}"


def arm_cells(arm: dict[str, Any]) -> list[str]:
    """The README row of one arm: 11 cells, arm id first (without backticks)."""
    p = arm["primary"]
    if all(p.get(k) is not None for k in ("trg", "ci_low", "ci_high")):
        trg = f"{fixed3(p['trg'])} [{fixed3(p['ci_low'])}, {fixed3(p['ci_high'])}]"
    else:
        trg = "-"
    if arm["bytes"].get("scored_above_budget", 0) > 0:
        trg += " †"
    by_task = {t["task"]: t for t in p.get("tasks", [])}
    tasks = [
        f"{fixed3(by_task[t]['value'])} ({by_task[t]['families']})" if t in by_task else "-"
        for t in TASKS
    ]
    retention = " / ".join(opt3(m.get("median")) for m in arm.get("median_retention", []))
    cells = f"{p['failed_cells']} / {p['cells']}"
    if p.get("not_applicable_cells", 0) > 0:
        cells += f" ({p['not_applicable_cells']} n/a)"
    return [arm["arm_id"], arm["artifact_class"], bytes_cell(arm), trg, *tasks, retention,
            opt3(arm.get("c2st_median")), guard_cell(arm), cells]


def guard_word(guard: dict[str, Any], comp: dict[str, Any]) -> str:
    name, value = guard["name"], guard.get("value")
    if value is None:
        text = ""
    elif name == "bytes":
        text = f" {value:.0f}"
    elif name == "exact_copy_excess":
        copies = comp.get("exact_copy")
        text = f" {fixed3(value)} ({fixed3(copies['max'])})" if copies else f" {fixed3(value)}"
    else:
        text = f" {signed3(value)}"
    return f"{GUARD_NAMES.get(name, name)}{text} {CHECK_WORDS[guard['check']]}"


def calibration_cell(comp: dict[str, Any]) -> str:
    cal = comp.get("null_calibration")
    if not cal:
        return "-"
    rule = cal["flip_guard_rule"]
    if rule["acceptable"]:
        return fixed3(rule["rate"])
    return f"{fixed3(rule['rate'])} (> {fixed3(cal['max_false_promote_rate'])})"


def comparison_cells(comp: dict[str, Any]) -> list[str]:
    """The README row of one comparison: 6 cells (first one without backticks)."""
    if comp.get("kind") == "seed_group" or comp.get("baseline_above_budget", 0) > 0:
        raise TableError(f"comparison {comp.get('id')}: seed groups and over-budget baselines are not rendered")
    guards = "; ".join(guard_word(g, comp) for g in comp["decision"]["guards"])
    delta = f"{signed3(comp['delta'])} [{signed3(comp['ci_low'])}, {signed3(comp['ci_high'])}]"
    return [f"{comp['id']} vs {comp['baseline']}", delta, f"{p_value(comp['p_holm'])} ({comp['holm_m']})",
            guards, calibration_cell(comp), comp["decision"]["outcome"]]


def _md_cells(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
    return [c.replace("`", "").replace("**", "") for c in cells]


def parse_readme_block(text: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Arm rows and comparison rows of the README block, keyed by arm id / comparison label."""
    arms: dict[str, list[str]] = {}
    comps: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        cells = _md_cells(line)
        (comps if " vs " in cells[0] else arms)[cells[0]] = cells
    if not arms or not comps:
        raise TableError("README block holds no arm or no comparison rows")
    return arms, comps


def cross_check(readme_text: str, arms: list[dict[str, Any]], comps: list[dict[str, Any]],
                extra_arm_ids: set[str]) -> None:
    """Every README row equals its receipt rendering; every receipt row is in the README.

    `extra_arm_ids` are third-party arms that may be absent from the README block."""
    md_arms, md_comps = parse_readme_block(readme_text)
    rendered = {a["arm_id"]: arm_cells(a) for a in arms}
    for arm_id, cells in md_arms.items():
        if arm_id not in rendered:
            raise TableError(f"README arm {arm_id} is in no receipt")
        if rendered[arm_id] != cells:
            raise TableError(f"README row of {arm_id} differs from its receipt rendering")
    missing = set(rendered) - set(md_arms) - extra_arm_ids
    if missing:
        raise TableError(f"receipt arms missing from the README block: {sorted(missing)}")
    rendered_c = {c[0]: c for c in (comparison_cells(x) for x in comps)}
    if rendered_c != md_comps:
        raise TableError("README comparison rows differ from the compare receipt")


def tex(cell: str) -> str:
    """LaTeX for one README cell string (numbers keep their digits; signs become math minus)."""
    out = re.sub(r"(?<!\S)-(?!\S)", "--", cell)
    out = re.sub(r"(?<![\w.])-(?=\d)", r"\\ensuremath{-}", out)
    out = out.replace("_", r"\_").replace("%", r"\%").replace("<", r"\textless{}")
    out = out.replace("> ", r"\textgreater{} ").replace("†", r"\textdagger{}")
    return out


def _arm_label(cells: list[str], highlight: str) -> str:
    label = r"\texttt{" + tex(cells[0]) + "}"
    return r"\textbf{" + label + "}" if cells[0] == highlight else label


def _groups(arms: list[dict[str, Any]], third_party: list[dict[str, Any]]) -> list[list[list[str]]]:
    source_free = [arm_cells(a) for a in arms if a["artifact_class"] != "control"]
    controls = [arm_cells(a) for a in arms if a["artifact_class"] == "control"]
    return [g for g in (source_free, [arm_cells(a) for a in third_party], controls) if g]


def _tabular(spec: str, header: str, groups: list[list[list[str]]], cols: list[int], highlight: str) -> list[str]:
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", header + r" \\", r"\midrule"]
    for gi, group in enumerate(groups):
        if gi:
            lines.append(r"\midrule")
        for cells in group:
            row = [_arm_label(cells, highlight)] + [tex(cells[i]) for i in cols]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def render_arm_table(arms: list[dict[str, Any]], third_party: list[dict[str, Any]], highlight: str) -> str:
    """Two panels of one table: the README columns wrapped after the per-task cells."""
    groups = _groups(arms, third_party)
    head_a = (r"Arm & Class & \makecell{Bytes median / max\\ / share $\le$ budget} & TRG [95\% CI] & "
              r"\makecell{Regression\\(families)} & \makecell{Binary\\(families)} & "
              r"\makecell{Multiclass\\(families)}")
    head_b = (r"Arm & \makecell{Median retention\\linear / gbdt / mlp} & C2ST & "
              r"\makecell{Realism / privacy /\\copies mean (max)} & \makecell{Failed\\/ cells}")
    tp = ("% third-party baseline rows: " + ", ".join(a["arm_id"] for a in third_party)
          if third_party else "% third-party baseline rows: none (primary-methods.json absent)")
    lines = ["% Generated by docs/whitepaper-v3/scripts/render_v3_paper.py -- DO NOT EDIT BY HAND.", tp,
             r"\textbf{(a)}\par\smallskip"]
    lines += _tabular("@{}llrrrrr@{}", head_a, groups, [1, 2, 3, 4, 5, 6], highlight)
    lines += [r"\par\medskip\textbf{(b)}\par\smallskip"]
    lines += _tabular("@{}lrrrr@{}", head_b, groups, [7, 8, 9, 10], highlight)
    note = above_budget_note(arms + third_party)
    if note:
        lines += [r"\par\smallskip", r"\parbox{\linewidth}{\raggedright " + _tt_ids(tex(note)) + "}"]
    return "\n".join(lines) + "\n"


def above_budget_note(arms: list[dict[str, Any]]) -> str:
    """The README's dagger note for arms scored with files above the budget ('' when there are none)."""
    above = [a for a in arms if a["bytes"].get("scored_above_budget", 0) > 0]
    if not above:
        return ""
    budget = above[0]["bytes"]["budget"]
    items = "; ".join(f"`{a['arm_id']}` {a['bytes']['scored_above_budget']} files, TRG at {budget} B "
                      f"{opt3(a.get('trg_at_budget'))}" for a in above)
    return f"\u2020 Scored with packets above {budget} B (the TRG reads the rows as scored): {items}."


def _tt_ids(text: str) -> str:
    return re.sub(r"`([^`]*)`", r"\\texttt{\1}", text)


def render_methods_table(comps: list[dict[str, Any]]) -> str:
    """The README comparison row of each third-party method against the release encoder."""
    lines = ["% Generated by docs/whitepaper-v3/scripts/render_v3_paper.py -- DO NOT EDIT BY HAND."]
    if not comps:
        return "\n".join(lines + ["% third-party comparisons: none (compare-methods.json absent)"]) + "\n"
    base = {c["baseline"] for c in comps}
    if len(base) != 1:
        raise TableError(f"third-party comparisons name several baselines: {sorted(base)}")
    lines += [r"\begin{tabular}{@{}lrr>{\raggedright\arraybackslash}p{0.34\linewidth}rl@{}}", r"\toprule",
              r"\makecell[l]{Method vs\\\texttt{" + tex(base.pop()) + r"}} & $\Delta$TRG [95\% CI] & "
              r"\makecell{Holm $p$\\($m$)} & Guards & \makecell{Sign-flip\\PROMOTE} & Outcome \\", r"\midrule"]
    for comp in comps:
        cells = comparison_cells(comp)
        guards = re.sub(r"\bFAIL\b", r"\\textbf{FAIL}", tex(cells[3]))
        lines.append(r"\texttt{" + tex(comp["id"]) + "} & " + " & ".join([tex(cells[1]), tex(cells[2]), guards, tex(cells[4])])
                     + r" & \textbf{" + cells[5] + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _threshold(name: str, value: float) -> str:
    sym = {"ci_low": ">", "holm_p": "<", "min_delta": r"\ge", "task_ci_low": ">", "mlp_delta": r"\ge",
           "realism_coverage_delta": ">", "privacy_coverage_delta": ">", "exact_copy_excess": r"\le",
           "bytes": r"\le"}[name]
    num = f"{value:.0f}" if name == "bytes" else (signed3(value) if name.endswith("delta") or name.endswith("ci_low") else fixed3(value))
    return "$" + sym + "$ " + tex(num)


def render_comparison_table(comp: dict[str, Any]) -> str:
    """The README comparison row, then one row per promotion condition and guard."""
    cells = comparison_cells(comp)
    lines = ["% Generated by docs/whitepaper-v3/scripts/render_v3_paper.py -- DO NOT EDIT BY HAND.",
             r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
             r"Comparison & $\Delta$TRG [95\% CI] & Holm $p$ ($m$) & Sign-flip PROMOTE & Outcome \\",
             r"\midrule",
             r"\texttt{" + tex(comp["id"]) + r"} vs \texttt{" + tex(comp["baseline"]) + "} & "
             + " & ".join(tex(c) for c in cells[1:3] + cells[4:5]) + r" & \textbf{" + cells[5] + r"} \\",
             r"\bottomrule", r"\end{tabular}", r"\par\medskip",
             r"\begin{tabular}{@{}llrl@{}}", r"\toprule",
             r"Rule item & Threshold & Value & Check \\", r"\midrule"]
    labels = {"ci_low": "interval lower bound", "holm_p": "Holm-adjusted $p$", "min_delta": "minimum gain"}
    for item in comp["decision"]["conditions"]:
        val = fixed3(item["value"]) if item["name"] == "holm_p" else signed3(item["value"])
        lines.append(f"condition: {labels[item['name']]} & {_threshold(item['name'], item['threshold'])} & "
                     f"{tex(val)} & {CHECK_WORDS[item['check']]} " + r"\\")
    for guard in comp["decision"]["guards"]:
        word = guard_word(guard, comp)
        name, _, rest = word.partition(" ")
        value, _, check = rest.rpartition(" ")
        lines.append(f"guard: {name} & {_threshold(guard['name'], guard['threshold'])} & {tex(value)} & "
                     + (r"\textbf{FAIL}" if check == "FAIL" else check) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
