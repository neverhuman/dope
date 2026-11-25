"""Number formatting and equal-cap sweep macros for the v3 paper renderer."""
from __future__ import annotations

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCOREBOARD = "bench/real-world/results/scoreboard/v3.0.0"


class RenderError(RuntimeError):
    pass


def num(x: float, digits: int = 3, signed: bool = False) -> str:
    text = format(float(x), f"{'+' if signed else ''}.{digits}f")
    if re.fullmatch(r"[+-]?0\.?0*", text):
        text = text.lstrip("+-")
    return text.replace("-", r"\ensuremath{-}")


def ci(v: float, lo: float, hi: float, signed: bool = False, digits: int = 3) -> str:
    return f"{num(v, digits, signed)} [{num(lo, digits, signed)}, {num(hi, digits, signed)}]"


def grouped(x: float) -> str:
    whole = f"{x:,.0f}" if float(x).is_integer() else f"{x:,.1f}"
    return whole.replace(",", "{,}").replace("-", r"\ensuremath{-}")


def sweep_macros() -> list[tuple[str, str]]:
    """Point estimates on the equal-cap sweep. The plotted series stay in the bound CSVs."""
    served_path = REPO / f"{SCOREBOARD}/evidence/addendum-k/served.csv"
    rows_path = REPO / f"{SCOREBOARD}/evidence/addendum-k/rows.csv"
    with served_path.open(encoding="utf-8") as served_file:
        served = {int(row["cap"]): row for row in csv.DictReader(served_file)}
    with rows_path.open(encoding="utf-8") as rows_file:
        stored = {int(row["cap"]): row for row in csv.DictReader(rows_file)}
    needed = (1024, 2048, 16384)
    if set(needed) - set(served) or set(needed) - set(stored):
        raise RenderError("addendum-k sweep is missing a 1 KiB, 2 KiB or 16 KiB cap")
    return [
        ("Sweep1024Score", num(float(served[1024]["score"]), 1)),
        ("Sweep16384Score", num(float(served[16384]["score"]), 1)),
        ("Sweep16384Median", grouped(float(served[16384]["median"]))),
        ("Rows16384Score", num(float(stored[16384]["score"]), 1)),
        ("Rows16384Count", grouped(float(stored[16384]["retained_rows_median"]))),
    ]


def short(x: float) -> str:
    return format(float(x), "g").replace("-", r"\ensuremath{-}")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}\\%"


def tt(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"
