"""Render the informative-lineage denominator table from a receipt panel.

The informative count is the level n of the panel's retention rows, the same
lineages the size tables print. The other counts come from the panel's
denominator rows. That sensitivity path sees only lineage rows with three
finite stored null, TRTR, and TSTR losses, the complete-loss rows. The
Forest-Flow cells store no null loss, so that path is empty for Forest-Flow
while its retention rows are not. Rendering reads the payload only; no panel
byte changes.
"""

from __future__ import annotations


def _stacked(top: str, bottom: str) -> str:
    return f"\\shortstack[r]{{{top}\\\\{bottom}}}"


# Ten columns at \scriptsize with 2pt separation: one-line headers set 457pt
# against the 397pt text width; these stacked headers set 390pt.
HEADER = " & ".join((
    "Method", "Configuration", "Size", "Informative", _stacked("Complete-", "loss rows"),
    _stacked("Non-", "informative"), "Undefined", "Capped", _stacked("Sensitivity", "$n$"),
    _stacked("Capped", "median"),
))
ALIGN = "lllrrrrrrr"
AUDITOR = "catboost"


def _key(row) -> tuple:
    return (row["method"], row["configuration"], int(row["size"]), row["auditor"])


def level_counts(retention_rows) -> dict[tuple, int]:
    """(method, configuration, size, auditor) to the retention-row lineage count."""
    counts = {}
    for row in retention_rows:
        key = _key(row)
        if key in counts:
            raise ValueError(f"duplicate retention row {key}")
        counts[key] = int(row["n"])
    return counts


def denominator_rows(payload, auditor: str = AUDITOR) -> list[dict]:
    """Denominator rows for one auditor, with the level count beside the loss path."""
    levels = level_counts(payload["retention"])
    rows = []
    for row in payload["denominator"]:
        if row["auditor"] != auditor:
            continue
        key = _key(row)
        if key not in levels:
            raise ValueError(f"denominator row has no retention row {key}")
        counted = int(row["n_informative"]) + int(row["n_noninformative"])
        if int(row["sensitivity_n"]) != counted:
            raise ValueError(f"sensitivity n disagrees with its counts {key}")
        if int(row["n_informative"]) > levels[key]:
            raise ValueError(f"complete-loss informative count exceeds the level count {key}")
        rows.append({
            **row,
            "informative": levels[key],
            "complete_loss_rows": counted + int(row["n_undefined"]),
        })
    return rows


def denominator_lines(payload, fmt, tex_name, auditor: str = AUDITOR) -> list[str]:
    """Table rows. ``fmt`` and ``tex_name`` are the receipt panel's formatters."""
    lines = []
    for row in denominator_rows(payload, auditor):
        lines.append(
            f"{tex_name(row['method'])} & {tex_name(row['configuration'])} & {row['size']}$n$ & "
            f"{row['informative']} & {row['complete_loss_rows']} & {row['n_noninformative']} & "
            f"{row['n_undefined']} & {row['n_capped']} & {row['sensitivity_n']} & "
            f"{fmt(row['sensitivity_median'])} \\\\"
        )
    return lines
