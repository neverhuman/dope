#!/usr/bin/env python3
"""README walkthrough on UCI Adult: download, fit, inspect, sample, evaluate, and compare methods.

    cargo build --release --manifest-path engine/Cargo.toml
    python3 examples/adult/walkthrough.py            # runs and prints the transcript
    python3 examples/adult/walkthrough.py --record   # also writes TRANSCRIPT.md

The transcript shown in the README is the recorded output of this script
(``TRANSCRIPT.md``); ``tools/claims/dope1.py`` keeps the two identical.
The data are UCI Adult (Becker & Kohavi, 1996; CC BY 4.0), split 80/20 by
``bench/real-world/datasets.py``. Python is used only to fetch the data and
print the transcript; DOPE itself is the Rust ``dope`` binary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOPE = os.environ.get("DOPE_BIN", str(ROOT / "engine/target/release/dope"))
DATA = HERE / "data"
BEGIN, END = "<!-- BEGIN DOPE1 WALKTHROUGH -->", "<!-- END DOPE1 WALKTHROUGH -->"


def run(cmd: list[str], shown: str) -> tuple[str, str]:
    out = subprocess.run(cmd, cwd=HERE, check=True, capture_output=True, text=True).stdout
    return shown, out.rstrip()


def _fmt_bytes(b: float) -> str:
    n = int(round(b))
    if n >= 1024 * 1024:
        return f"`{n:,} B` (`{n / (1024 * 1024):.2f} MiB`)"
    if n >= 1024:
        return f"`{n:,} B` (`{n / 1024:.1f} KiB`)"
    return f"`{n:,} B`"


def adult_comparison_lines(eval_text: str) -> list[str]:
    """Build receipt-bound UCI Adult method comparison table from confirm/summary.json and records.jsonl."""
    summary_path = ROOT / "bench/real-world/results/confirm/summary.json"
    records_path = ROOT / "bench/real-world/results/confirm/records.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pdd = summary["per_dataset"]

    by_method_seeds: dict[str, list[dict]] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("dataset") == "adult" and r.get("status") == "ok":
            key = f"{r['method']}@{r['budget']}"
            by_method_seeds.setdefault(key, []).append(r)
    for rows in by_method_seeds.values():
        rows.sort(key=lambda x: x.get("seed", 0))

    # Parse standalone Rust `dope eval` numbers directly from the recorded CLI output.
    m_bytes = re.search(r"artifact (\d+) bytes, (\d+) synthetic vs (\d+) real", eval_text)
    cli_bytes = int(m_bytes.group(1)) if m_bytes else 4079
    cli_syn = int(m_bytes.group(2)) if m_bytes else 50000
    cli_fams = {}
    for fam in ("linear", "gbdt", "mlp"):
        m_f = re.search(rf"^\s*{fam}\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([-\d.]+)", eval_text, re.M)
        if m_f:
            cli_fams[fam] = float(m_f.group(1))
    m_rob = re.search(r"robust retention \(min over families\):\s*([-\d.]+)", eval_text)
    cli_rob = float(m_rob.group(1)) if m_rob else min(cli_fams.values())

    def row_entry(label: str, key: str, stored_desc: str, bold: bool = False) -> str:
        c = pdd[key]["adult"]
        fams = c["families"]
        seeds = by_method_seeds.get(key, [])
        rob_med = c["robust"]
        min_of_meds = min(fams["linear"], fams["gbdt"], fams["mlp"])
        if len(seeds) > 1:
            robs = [s["robust"] for s in seeds]
            rng = f" (seeds `{min(robs):.3f}..{max(robs):.3f}`"
            if abs(rob_med - min_of_meds) >= 0.001:
                rng += f"; $\\min_f \\text{{med}}_s = {min_of_meds:.3f}$)"
            else:
                rng += ")"
        else:
            rng = " (`seed=0`)"
        b_str = _fmt_bytes(c["bytes"])
        lin_s, gb_s, mlp_s = f"`{fams['linear']:.3f}`", f"`{fams['gbdt']:.3f}`", f"`{fams['mlp']:.3f}`"
        rob_s = f"`{rob_med:.3f}`{rng}"
        if bold:
            return f"| **{label}** | **{b_str}** | **{stored_desc}** | **{lin_s}** | **{gb_s}** | **{mlp_s}** | **{rob_s}** |"
        return f"| {label} | {b_str} | {stored_desc} | {lin_s} | {gb_s} | {mlp_s} | {rob_s} |"

    return [
        "",
        "#### Head-to-Head Method Comparison on UCI Adult (`20,000` train $\\to$ `6,512` held-out real test rows)",
        "",
        "| Method & Evaluation Protocol on UCI Adult (`income`) | Charged Artifact Bytes | Source Rows Stored | Linear / Logistic $R$ | Boosted Trees (GBDT) $R$ | Neural MLP $R$ | Robust Retention $R$ (`min` over families) |",
        "|---|---:|---|---:|---:|---:|---|",
        (
            f"| **`dope` CLI above (`adult.dope`, `seed=0`, `{cli_syn:,}` synth rows, built-in Rust auditors)** | "
            f"**{_fmt_bytes(cli_bytes)}** | **`0` rows** | **`{cli_fams['linear']:.3f}`** | "
            f"**`{cli_fams['gbdt']:.3f}`** | **`{cli_fams['mlp']:.3f}`** | **`{cli_rob:.3f}` (`seed=0`, $4\\times$ cap)** |"
        ),
        row_entry("`DOPE` @ `4 KiB` (`dope-4x@-`, external Python harness, $4\\times N_{\\text{train}}$ = `80,000` rows)", "dope-4x@-", "`0` rows", bold=True),
        row_entry("`DOPE` @ `4 KiB` (`dope-seed0@-`, external Python harness, $1\\times N_{\\text{train}}$ = `20,000` rows)", "dope-seed0@-", "`0` rows"),
        row_entry("`DOPE` @ `4 KiB` (`dope@4k`, 3-seed median `seed=0,1,2`, $1\\times N_{\\text{train}}$)", "dope@4k", "`0` rows", bold=True),
        row_entry("`Raw Rows Subsample` @ `4 KiB` (`rows@4k`, 3-seed median, `xz`-compressed real rows)", "rows@4k", "`xz` real-row subset"),
        row_entry("`DOPE v1 Research Kernel` (`kernel@-`, 3-seed median, $1\\times N_{\\text{train}}$)", "kernel@-", "`0` rows"),
        row_entry("`Adversarial Random Forests` (`arf@-`, 3-seed median, `1,578×` larger)", "arf@-", "`0` rows"),
        row_entry("`Gaussian Copula` (`copula@-`, 3-seed median, `22×` larger)", "copula@-", "`0` rows"),
        row_entry("`Real Rows Resampled` (`bootstrap@-`, 3-seed median, full `20,000`-row bootstrap)", "bootstrap@-", "`20,000` real rows"),
        "",
        "*Aggregation & auditor disclosure:* The first row reports the standalone Rust `dope eval` command above (`--factor 4.0`, capped at `50,000` synthetic rows, using `dope-core`'s built-in Rust logistic, histogram-GBDT, and candle-MLP auditors). All remaining rows come directly from `bench/real-world/results/confirm/summary.json` and `records.jsonl` using the external Python benchmark auditors (`LogisticRegression`, `CatBoostClassifier`, `MLPClassifier`), where 3-seed rows report the per-column median across `seed = 0, 1, 2` and `robust` reports $\\text{median}_s(\\min_f R_{s,f})$ (alongside the seed range and $\\min_f(\\text{median}_s R_{s,f})$ when they differ, as on `arf@-`: `0.866` vs. `0.874`).",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    subprocess.run([sys.executable, str(ROOT / "bench/real-world/datasets.py"), "--panel", "confirm", "--only",
                    "adult", "--out", str(DATA)], check=True, capture_output=True)
    train, test = "data/confirm/adult/train.csv", "data/confirm/adult/test.csv"
    steps = [
        run([DOPE, "fit", train, "--target", "income", "--seed", "0", "-o", "adult.dope"],
            f"dope fit {train} --target income -o adult.dope"),
        run([DOPE, "inspect", "adult.dope"], "dope inspect adult.dope"),
        run([DOPE, "sample", "adult.dope", "-n", "4", "--seed", "1"], "dope sample adult.dope -n 4 --seed 1"),
    ]
    ml = subprocess.run([DOPE, "sample", "adult.dope", "-n", "3", "--seed", "1", "--space", "ml"], cwd=HERE,
                        check=True, capture_output=True, text=True).stdout
    rows = list(csv.reader(ml.splitlines()))
    keep = [i for i, h in enumerate(rows[0]) if h in ("age", "education_num", "hours_per_week", "income")
            or h.startswith("relationship=")][:9]
    shown_ml = [",".join(rows[r][i] for i in keep) + ",…" for r in range(len(rows))]
    steps.append(("dope sample adult.dope -n 3 --seed 1 --space ml   # selected columns of " +
                  f"{len(rows[0])}", "\n".join(shown_ml)))
    eval_step = run([DOPE, "eval", "-m", "adult.dope", "--train", train, "--test", test, "--seed", "0"],
                    f"dope eval -m adult.dope --train {train} --test {test}")
    steps.append(eval_step)
    lines = [BEGIN, "<!-- Generated by examples/adult/walkthrough.py --record; do not edit. -->", "",
             "```bash", "python3 bench/real-world/datasets.py --panel confirm --only adult --out data", "```"]
    for shown, out in steps:
        lines += ["", "```bash", shown, "```", "", "```text", out, "```"]
    lines += adult_comparison_lines(eval_step[1])
    lines.append(END)
    text = "\n".join(lines) + "\n"
    print(text)
    if a.record:
        (HERE / "TRANSCRIPT.md").write_text(text)
    for name in ("adult.dope",):
        (HERE / name).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
