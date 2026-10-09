"""Small TeX helpers for review tables: a captioned longtable and a row-key guard."""

from __future__ import annotations

from research.benchmark.review_fixes.receipt_panel import _nfields


def assert_distinct(keys, label: str) -> None:
    """Refuse a table in which two printed rows carry the same identifying cells."""
    seen = set()
    for key in keys:
        if key in seen:
            raise ValueError(f"{label} prints two rows with the same identity: {key}")
        seen.add(key)


def longtable(caption: str, header: str, rows: list[str], align: str, note: str = "", label: str = "") -> str:
    """A longtable whose caption states its Holm family, as check_paper.py requires."""
    if "Holm" not in caption:
        raise ValueError("review table caption must state its Holm family or that it has none")
    expected = _nfields(header)
    if len(align) != expected:
        raise ValueError(f"column spec {align!r} does not match header fields {expected}: {header}")
    if not rows:
        raise ValueError("review table has no rows")
    for row in rows:
        if not row.lstrip().startswith("%") and _nfields(row) != expected:
            raise ValueError(f"row has {_nfields(row)} fields, header has {expected}: {row}")
    head = f"\\toprule\n{header} \\\\\n\\midrule\n"
    return (
        note
        + f"\\begin{{longtable}}{{{align}}}\n\\caption{{{caption}}}"
        + (f"\\label{{{label}}}" if label else "") + "\\\\\n"
        + head + "\\endfirsthead\n"
        + head + "\\endhead\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{longtable}\n"
    )
