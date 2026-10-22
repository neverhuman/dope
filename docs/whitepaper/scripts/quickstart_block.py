#!/usr/bin/env python3
"""History-page block for the quickstart: the committed reference run on UCI Adult.

The block is a pure rendering of examples/quickstart/results/adult-reference.json,
the report.json that `python3 examples/quickstart/quickstart.py --dataset adult`
writes. claim_registry.py checks that the historical results page (HISTORY.md)
carries it byte for byte, so its quickstart numbers can never drift from a committed run.

    python3 -B docs/whitepaper/scripts/quickstart_block.py --write-readme   # also writes the paper macros
    python3 -B docs/whitepaper/scripts/quickstart_block.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REPORT = "examples/quickstart/results/adult-reference.json"
ABLATION = "examples/quickstart/results/ablation.json"
MACROS = "paper/generated/quickstart-macros.tex"
ABLATION_TEX = "paper/generated/quickstart-ablation.tex"
SHORT = {
    "one-hot": "one-hot columns",
    "target-ordered · robust · chow_liu": "target-ordered codes",
    "target-ordered · min-max · chow_liu": "min-max scaling",
    "target-ordered · quantile · chow_liu": "a quantile map",
    "target-ordered · robust · default search": "the built-in tournament",
    "target-ordered · robust · transfer-in-the-loop (K=3)": "transfer-in-the-loop selection",
}
BEGIN = "<!-- BEGIN QUICKSTART -->"
END = "<!-- END QUICKSTART -->"
HISTORY = "HISTORY.md"


def _size(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{n:,.0f} bytes"


def readme_block(report: dict) -> str:
    sizes = report["bytes"]
    encoder = sizes["encoder"]
    auditors = " · ".join(f"{row['auditor']} {row['retention']:.3f}" for row in report["auditors"]
                          if row["retention"] is not None)
    selection = report.get("selection") or {}
    scores = selection.get("scores") or {}
    chosen = report["backend"]
    picked = (f"`{chosen}` (3-fold transfer on the fit rows: "
              + ", ".join(f"{name} {value:.3f}" for name, value in sorted(scores.items(), key=lambda kv: -kv[1]))
              + ")") if scores else f"`{chosen}`"
    state = sizes["state"]
    lines = [
        BEGIN,
        f"Historical prototype run on {report['dataset']}: {report['fit_rows']:,} fit rows, a {report['task']} target "
        f"`{report['target']}`, and {report['holdout_rows']:,} development query rows (the official test file).",
        "",
        "| Step | What happens | Result |",
        "| --- | --- | --- |",
        f"| encode | fit-row transform; incomplete prototype size only | `encoder.bin` "
        f"{encoder['total_bytes']:,} bytes ({encoder['transform_bytes']:,} transforms + "
        f"{encoder['label_bytes']:,} compressed names and labels) |",
        f"| select | backends scored inside the fit rows | {picked} |",
        f"| compile | `dope-kernel compile` reads only the encoded fit rows | `model.dpk` {sizes['model']:,} bytes |",
        f"| sample | native draws decoded with a live fitted Python encoder | "
        f"{report['synthetic_rows']:,} synthetic rows |",
        f"| evaluate | learners trained on real vs. synthetic rows, scored on the holdout | retention {auditors} |",
        "",
        f"Historical prototype subtotal: **{state:,} bytes** (model plus diagnostic binary dump). "
        "This omits complete reloadable encoder state: its CSV/gzip ratios are not complete transfer ratios. "
        "The current command reloads and charges the full encoder.json plus model.dpk; "
        "no corrected Adult total or speed has been measured. Historical compiled program:",
        "",
        "```text",
        pretty_program(report["program"]),
        "```",
        "",
        "<sub>Local engineering demonstration on one public table and one split, with the auditors in "
        "<code>examples/quickstart/dope_eval.py</code>; not the paper benchmark and no privacy claim. "
        f"Rendered from <code>{REPORT}</code>.</sub>",
        END,
    ]
    return "\n".join(lines)


def _short(config: str) -> str | None:
    for key, name in SHORT.items():
        if config == key or config.startswith(key + " ") or config.startswith(key + " ·"):
            return name
    return None


def ablation_sentence(ablation: dict) -> str:
    """Mean worst-auditor retention over seeds for each encoding and selection choice, per table."""
    by_config: dict[str, dict[str, float]] = {}
    for row in ablation["summary"]:
        name = _short(row["config"])
        if name is not None:
            by_config.setdefault(name, {})[row["dataset"]] = row["worst_mean"]
    tables = sorted({row["dataset"] for row in ablation["summary"]})
    titles = {"adult": "Adult", "abalone": "Abalone"}
    parts = [f"{name} " + ", ".join(f"{values[t]:.3f}" for t in tables if t in values)
             for name, values in by_config.items()]
    return (f"mean worst-auditor retention over {len(ablation['seeds'])} seeds ("
            + ", ".join(titles.get(t, t) for t in tables) + "): " + "; ".join(parts) + ".")


def pretty_program(sexpr: str) -> str:
    """Put each top-level clause of the program's S-expression on its own line."""
    depth, start, pieces = 0, 0, []
    for index, char in enumerate(sexpr):
        if char == "(":
            depth += 1
            if depth == 2:
                pieces.append(sexpr[start:index].rstrip())
                start = index
        elif char == ")":
            depth -= 1
    pieces.append(sexpr[start:])
    return pieces[0] + "".join("\n  " + piece.strip() for piece in pieces[1:])


def tex_macros(report: dict, ablation: dict) -> str:
    sizes = report["bytes"]
    retention = {row["auditor"]: row["retention"] for row in report["auditors"]}
    tree = "catboost" if "catboost" in retention else "gbdt"
    macro = {
        "QSFitRows": f"{report['fit_rows']:,}", "QSHoldoutRows": f"{report['holdout_rows']:,}",
        "QSBackend": report["backend"].replace("_", "\\_"), "QSStateBytes": f"{sizes['state']:,}",
        "QSModelBytes": f"{sizes['model']:,}", "QSEncoderBytes": f"{sizes['encoder']['total_bytes']:,}",
        "QSCsvRatio": f"{sizes['fit_csv'] / sizes['state']:,.0f}",
        "QSGzipRatio": f"{sizes['fit_csv_gzip9'] / sizes['state']:,.0f}",
        "QSRetCatBoost": f"{retention[tree]:.3f}", "QSRetLinear": f"{retention['linear']:.3f}",
        "QSRetMlp": f"{retention['mlp']:.3f}",
        "QSAblationSummary": ablation_sentence(ablation).replace("_", "\\_"),
    }
    body = "".join(f"\\providecommand{{\\{k}}}{{{v}}}\n" for k, v in macro.items())
    return ("% Generated by docs/whitepaper/scripts/quickstart_block.py; do not edit.\n"
            "% Historical prototype subtotal omits complete receiver state; ratios are not complete transfer ratios.\n" + body)


def ablation_tex(ablation: dict) -> str:
    """Supplement table: every ablation configuration on both tables."""
    tables = sorted({row["dataset"] for row in ablation["summary"]})
    titles = {"adult": "Adult", "abalone": "Abalone"}
    out = ["% Generated by docs/whitepaper/scripts/quickstart_block.py; do not edit.",
           "\\begin{tabular}{@{}l" + "rr" * len(tables) + "@{}}", "\\toprule",
           "Encoding $\\cdot$ scaling $\\cdot$ backend & "
           + " & ".join(f"\\multicolumn{{2}}{{c}}{{{titles.get(t, t)}}}" for t in tables) + " \\\\",
           " & " + " & ".join("worst & prototype bytes" for _ in tables) + " \\\\", "\\midrule"]
    configs = []
    for row in ablation["summary"]:
        if row["config"] not in configs:
            configs.append(row["config"])
    cells = {(row["dataset"], row["config"]): row for row in ablation["summary"]}
    for config in configs:
        values = []
        for table in tables:
            row = cells.get((table, config))
            values += ([f"{row['worst_mean']:.3f} $\\pm$ {row['worst_sd']:.3f}", f"{row['state_bytes_mean']:,.0f}"]
                       if row else ["--", "--"])
        label = config.replace("·", "$\\cdot$").replace("_", "\\_")
        out.append(label + " & " + " & ".join(values) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def load_ablation(root: Path = REPO) -> dict:
    return json.loads((root / ABLATION).read_text())


def load_report(root: Path = REPO) -> dict:
    return json.loads((root / REPORT).read_text())


def failures(readme: str, root: Path = REPO) -> list[str]:
    if readme.count(BEGIN) != 1 or readme.count(END) != 1:
        return [f"{HISTORY}: quickstart block markers must appear exactly once"]
    start, end = readme.index(BEGIN), readme.index(END) + len(END)
    if readme[start:end] != readme_block(load_report(root)):
        return [f"{HISTORY}: generated quickstart block drift"]
    return []


def strip(readme: str) -> str:
    if BEGIN in readme and END in readme:
        return readme[:readme.index(BEGIN)] + readme[readme.index(END) + len(END):]
    return readme


def splice(readme: str, block: str) -> str:
    start, end = readme.index(BEGIN), readme.index(END) + len(END)
    return readme[:start] + block + readme[end:]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args(argv)
    readme_path = REPO / HISTORY
    readme = readme_path.read_text()
    if args.write_readme:
        readme_path.write_text(splice(readme, readme_block(load_report())))
        (REPO / MACROS).write_text(tex_macros(load_report(), load_ablation()))
        (REPO / ABLATION_TEX).write_text(ablation_tex(load_ablation()))
        return 0
    for name, text in ((MACROS, tex_macros(load_report(), load_ablation())),
                       (ABLATION_TEX, ablation_tex(load_ablation()))):
        if not (REPO / name).is_file() or (REPO / name).read_text() != text:
            print(f"{name}: generated quickstart TeX drift", file=sys.stderr)
            return 1
    found = failures(readme)
    for failure in found:
        print(failure, file=sys.stderr)
    if not found:
        print(f"{HISTORY} quickstart block is current")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
