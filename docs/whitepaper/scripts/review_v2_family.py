#!/usr/bin/env python3
"""Emit the five-seed source-family sensitivity table and its macros.

The input is the pinned v2 family file from research/benchmark/review_fixes/v2_family.py;
the strongest comparator is read from the pinned v2 panel, where it is computed.
Formatting and verdict words are the v2 emitter's, so the two never disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import method_registry as M  # noqa: E402
import review_v2_emit as E  # noqa: E402

FAMILY = E.RESULTS / "review-fixes-v2" / "family.json"
SCOPE_DISPLAY = {"excluding_simulated": "excl.\\ simulated", "feynman": "feynman", "strogatz": "strogatz",
                 "fri": "fri", "bng": "bng"}
SIZE_DISPLAY = {1: "$n$", 4: "$4n$"}


def _key(row: dict) -> tuple:
    return row["scope"], row["method"], row["configuration"], row["size"], row["auditor"]


def _estimate(row: dict, field: str) -> str:
    if row.get("ci_status"):
        return f"${E.num(row[field])}$ & not identified"
    return f"${E.num(row[field])}$ & ${E.interval(row['lo' if field == 'median' else 'hl_lo'], row['hi' if field == 'median' else 'hl_hi'])}$"


def table(family: dict) -> str:
    levels = {_key(row): row for row in family["levels"]}
    contrasts = [row for row in family["contrasts"] if row["auditor"] == "catboost"]
    head = "Scope & Size & Auditor & Row & $n$ & Estimate & 95\\% nested CI & W/T/L \\\\"
    lines = [
        "\\begin{longtable}{lllllrll}",
        "\\caption{\\textbf{Source-family sensitivity, five-seed panel.} A DOPE row is DOPE's median retention "
        "within the scope; a comparator row is the Hodges--Lehmann DOPE-minus-comparator difference on the "
        "lineages both arms complete there. The excluding-simulated rows drop the four simulated families; "
        "inside one family the family is fixed and its lineages are resampled, and fit seeds are resampled "
        "inside each lineage separately for each arm. A one-seed comparator contributes its single fit. "
        "Linear and MLP contrasts are in the family file. These rows are a sensitivity outside every Holm "
        "family.}\\label{tab:v2-family}\\\\",
        "\\toprule", head, "\\midrule", "\\endfirsthead", "\\toprule", head, "\\midrule", "\\endhead",
    ]
    for scope in family["scopes"]:
        for size in (1, 4):
            for auditor in ("catboost", "linear", "mlp"):
                level = levels.get((scope, "DOPE", "headline", size, auditor))
                if level is not None:
                    lines.append(f"{SCOPE_DISPLAY[scope]} & {SIZE_DISPLAY[size]} & "
                                 f"{M.AUDITOR_DISPLAY[auditor]} & {M.arm('DOPE', 'headline').display} & "
                                 f"{level['n']} & {_estimate(level, 'median')} & --- \\\\")
                if auditor != "catboost":
                    continue
                for row in contrasts:
                    if row["scope"] == scope and row["size"] == size:
                        lines.append(f"{SCOPE_DISPLAY[scope]} & {SIZE_DISPLAY[size]} & CatBoost & "
                                     f"$-$ {M.arm(row['method'], row['configuration']).display} & {row['n']} & "
                                     f"{_estimate(row, 'hl')} & {row['wins']}/{row['ties']}/{row['losses']} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}", ""]
    return "\n".join(lines)


def _contrast_macros(prefix: str, row: dict | None) -> str:
    if row is None:
        return "".join(E.cmd(prefix + part, E.DASH) for part in ("Diff", "Lo", "Hi", "N", "Verdict"))
    return (E.cmd(prefix + "Diff", E.num(row["hl"])) + E.cmd(prefix + "Lo", E.num(row["hl_lo"]))
            + E.cmd(prefix + "Hi", E.num(row["hl_hi"])) + E.cmd(prefix + "N", row["n"])
            + E.cmd(prefix + "Verdict", E.VERDICT[row["verdict"]]))


def macros(family: dict, strongest: dict | None) -> str:
    """Excluding-simulated macros at 4n under CatBoost: DOPE's level, each comparator, and the strongest."""
    scope = "excluding_simulated"
    level = next(row for row in family["levels"]
                 if (row["scope"], row["size"], row["auditor"]) == (scope, 4, "catboost"))
    text = (E.cmd("VExclDopeCbFour", E.num(level["median"])) + E.cmd("VExclDopeCbFourLo", E.num(level["lo"]))
            + E.cmd("VExclDopeCbFourHi", E.num(level["hi"])) + E.cmd("VExclDopeCbFourN", level["n"]))
    rows = {(row["method"], row["configuration"]): row for row in family["contrasts"]
            if (row["scope"], row["size"], row["auditor"]) == (scope, 4, "catboost")}
    for arm, row in rows.items():
        text += _contrast_macros("VExcl" + M.arm(*arm).code + "CbFour", row)
    best = None if strongest is None else rows.get((strongest["method"], strongest["configuration"]))
    return text + _contrast_macros("VExclStrong", best)


def outputs(family: dict, panel: dict) -> dict[str, str]:
    return {"v2-family-table.tex": table(family), "v2-family-numbers.tex": macros(family, panel.get("strongest"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-panel", type=Path, default=FAMILY, help="the pinned v2 family file")
    parser.add_argument("--out", type=Path, default=E.GENERATED)
    parser.add_argument("--check", action="store_true", help="fail when a committed output differs")
    args = parser.parse_args()
    family = json.loads(args.from_panel.read_text())
    panel = json.loads(E.PANEL.read_text())
    if family.get("official_tests_opened") is not False or panel.get("official_tests_opened") is not False:
        raise SystemExit("v2 family or panel official test flag is not closed")
    texts = outputs(family, panel)
    if args.check:
        stale = sorted(name for name, text in texts.items()
                       if not (args.out / name).is_file() or (args.out / name).read_text() != text)
        if stale:
            raise SystemExit("v2 family outputs differ from the family file: " + ", ".join(stale))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (args.out / name).write_text(text)


if __name__ == "__main__":
    main()
