#!/usr/bin/env python3
"""Emit the all-five-fits lineage sensitivity table and its macros.

The input is the pinned v2 all-five file from research/benchmark/review_fixes/v2_all_five.py;
the strongest comparator is read from the pinned v2 panel, where it is computed.
Formatting and verdict words are the v2 emitter's.
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

ALL_FIVE = E.RESULTS / "review-fixes-v2" / "all-five.json"
SIZE_DISPLAY = {1: "$n$", 4: "$4n$"}


def _cell(row: dict, point: str, lo: str, hi: str) -> str:
    if row.get("ci_status"):
        return f"${E.num(row[point])}$ & not identified"
    return f"${E.num(row[point])}$ & ${E.interval(row[lo], row[hi])}$"


def table(data: dict) -> str:
    head = "Size & Row & Lineages & Fits & Estimate & 95\\% nested CI & W/T/L \\\\"
    lines = [
        "\\begin{longtable}{llrrrll}",
        "\\caption{\\textbf{All-five-fits sensitivity, CatBoost.} The pre-declared lineage sensitivity of the "
        "five-seed arms: a lineage enters a row only when all five fit seeds have a complete fit. A level row is "
        "the median lineage value of one arm; a DOPE-minus row is the Hodges--Lehmann difference on lineages where "
        "both arms have all five fits. Intervals are the panel's nested bootstrap. Linear and MLP rows are in the "
        "all-five file. These rows are a sensitivity outside every Holm family.}\\label{tab:v2-all-five}\\\\",
        "\\toprule", head, "\\midrule", "\\endfirsthead", "\\toprule", head, "\\midrule", "\\endhead",
    ]
    for size in (1, 4):
        for row in data["levels"]:
            if row["size"] == size and row["auditor"] == "catboost":
                lines.append(f"{SIZE_DISPLAY[size]} & {M.arm(row['method'], row['configuration']).display} & "
                             f"{row['n']} & {row['n_fits']} & {_cell(row, 'median', 'lo', 'hi')} & --- \\\\")
        for row in data["contrasts"]:
            if row["size"] == size and row["auditor"] == "catboost":
                lines.append(f"{SIZE_DISPLAY[size]} & DOPE $-$ {M.arm(row['method'], row['configuration']).display} & "
                             f"{row['n']} & --- & {_cell(row, 'hl', 'hl_lo', 'hl_hi')} & "
                             f"{row['wins']}/{row['ties']}/{row['losses']} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}", ""]
    return "\n".join(lines)


def _dashes(prefix: str) -> str:
    return "".join(E.cmd(prefix + part, E.DASH) for part in ("Diff", "Lo", "Hi", "N", "Verdict"))


def macros(data: dict, strongest: dict | None) -> str:
    """DOPE's all-five level and its contrast with the computed strongest comparator, CatBoost at 4n."""
    level = next((row for row in data["levels"] if (row["method"], row["configuration"], row["size"],
                                                   row["auditor"]) == ("DOPE", "headline", 4, "catboost")), None)
    text = "".join(E.cmd(f"VAllFiveDopeCbFour{part}", E.DASH if level is None else value) for part, value in (
        ("", None if level is None else E.num(level["median"])), ("Lo", None if level is None else E.num(level["lo"])),
        ("Hi", None if level is None else E.num(level["hi"])), ("N", None if level is None else level["n"])))
    row = None if strongest is None else next(
        (item for item in data["contrasts"] if (item["method"], item["configuration"], item["size"], item["auditor"])
         == (strongest["method"], strongest["configuration"], 4, "catboost")), None)
    if row is None:
        return text + _dashes("VAllFiveStrong")
    return (text + E.cmd("VAllFiveStrongDiff", E.num(row["hl"])) + E.cmd("VAllFiveStrongLo", E.num(row["hl_lo"]))
            + E.cmd("VAllFiveStrongHi", E.num(row["hl_hi"])) + E.cmd("VAllFiveStrongN", row["n"])
            + E.cmd("VAllFiveStrongVerdict", E.VERDICT[row["verdict"]]))


def outputs(data: dict, panel: dict) -> dict[str, str]:
    return {"v2-all-five-table.tex": table(data), "v2-all-five-numbers.tex": macros(data, panel.get("strongest"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-panel", type=Path, default=ALL_FIVE, help="the pinned v2 all-five file")
    parser.add_argument("--out", type=Path, default=E.GENERATED)
    parser.add_argument("--check", action="store_true", help="fail when a committed output differs")
    args = parser.parse_args()
    data = json.loads(args.from_panel.read_text())
    panel = json.loads(E.PANEL.read_text())
    if data.get("official_tests_opened") is not False or panel.get("official_tests_opened") is not False:
        raise SystemExit("v2 all-five or panel official test flag is not closed")
    texts = outputs(data, panel)
    if args.check:
        stale = sorted(name for name, text in texts.items()
                       if not (args.out / name).is_file() or (args.out / name).read_text() != text)
        if stale:
            raise SystemExit("v2 all-five outputs differ from the all-five file: " + ", ".join(stale))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (args.out / name).write_text(text)


if __name__ == "__main__":
    main()
