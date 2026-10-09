#!/usr/bin/env python3
"""Emit the predeclare_v2 macros, tables, and headline JSON for the paper.

Inputs are the pinned v2 panel, the v2 fit ledger, the byte ledgers, and the
review privacy panel. Every data-dependent word in the manuscript (a verdict,
the strongest comparator, the Pareto set, the size ranking) is computed here,
so the prose cannot contradict the numbers. Nothing here opens a test file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import method_registry as M  # noqa: E402
import v2_bytes  # noqa: E402

RESULTS = REPO / "research" / "benchmark" / "results"
PANEL = RESULTS / "review-fixes-v2" / "panel.json"
FITS = RESULTS / "review-fixes-v2" / "fits.jsonl"
PRIVACY = RESULTS / "review-fixes-privacy-v1" / "panel.json"
GENERATED = HERE.parent / "generated"
VERDICT = {"above": "is above", "below": "is below", "inseparable": "cannot be separated from", None: "---"}
MARGIN = 0.02
HEADLINE_ORDER = ("TabSyn", "Arf", "ArfNat", "Gauss", "Chow", "Ind", "Tvae", "Ctgan", "Forest",
                  "TabDdpm", "TabDiff", "Great", "Synthpop", "Smote", "DopeDef", "Pred", "Boot")
LEVEL_ORDER = ("Dope", "DopeDef") + HEADLINE_ORDER[:-3] + ("Pred", "Boot")


DASH = "\\mbox{---}"


def num(value, places: int = 3) -> str:
    if value is None:
        return DASH
    text = f"{value:.{places}f}"
    return "0." + "0" * places if text in {"-0." + "0" * places} else text


def pval(value) -> str:
    if value is None:
        return DASH
    if value < 0.001:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"{mantissa}\\times 10^{{{int(exponent)}}}"
    return f"{value:.3f}"


def exact_bytes(value) -> str:
    return f"{int(round(value)):,}".replace(",", "{,}")


def human_bytes(value) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value / scale:.1f}\\,\\mathrm{{{unit}}}"
    return f"{exact_bytes(value)}\\,\\mathrm{{B}}"


def ratio(value) -> str:
    exponent = int(math.floor(math.log10(value)))
    return f"{value / 10 ** exponent:.1f}\\times 10^{{{exponent}}}"


def cmd(name: str, value) -> str:
    if not name.isalpha():
        raise ValueError(f"macro name {name!r} must be letters only")
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def interval(lo, hi) -> str:
    return f"[{num(lo)}, {num(hi)}]"


def _code(row) -> str:
    return M.arm(row["method"], row["configuration"]).code


def macros(panel: dict, byte_stats: dict, fits: list[dict], privacy: dict) -> str:
    lines = [cmd("VMargin", num(MARGIN, 2))]
    for row in panel["levels"]:
        name = _code(row) + M.macro_suffix(row["auditor"], row["size"])
        lines += [cmd("VLev" + name, num(row["median"])), cmd("VLev" + name + "Lo", num(row.get("lo"))),
                  cmd("VLev" + name + "Hi", num(row.get("hi"))), cmd("VLev" + name + "N", row["n"])]
        if row["method"] == "DOPE" and row["configuration"] == "headline":
            short = "VDope" + M.macro_suffix(row["auditor"], row["size"])
            lines += [cmd(short, num(row["median"])), cmd(short + "Lo", num(row.get("lo"))),
                      cmd(short + "Hi", num(row.get("hi"))), cmd(short + "N", row["n"])]
    for row in panel["contrasts"]:
        name = _code(row) + M.macro_suffix(row["auditor"], row["size"])
        wtl = f"{row['wins']}/{row['ties']}/{row['losses']}"
        equivalent = (row.get("equivalence") or {}).get("equivalent")
        lines += [cmd("VDiff" + name, num(row["hl"])), cmd("VLo" + name, num(row["hl_lo"])),
                  cmd("VHi" + name, num(row["hl_hi"])), cmd("VN" + name, row["n"]),
                  cmd("VMed" + name, num(row["median"])), cmd("VWtl" + name, wtl),
                  cmd("VHolm" + name, pval(row.get("holm_p"))), cmd("VVerdict" + name, VERDICT[row["verdict"]]),
                  cmd("VEquiv" + name, "---" if equivalent is None else ("yes" if equivalent else "no"))]
    lines += _strongest(panel, byte_stats)
    lines += _seed_macros(panel)
    lines += _byte_macros(byte_stats, fits)
    lines += _sentences(panel, byte_stats)
    lines += _privacy_macros(privacy)
    lines += _fidelity_macros(panel)
    lines += _simulated_count()
    return "".join(lines)


def _contrast(panel, code, auditor="catboost", size=4):
    for row in panel["contrasts"]:
        if _code(row) == code and row["auditor"] == auditor and row["size"] == size:
            return row
    return None


def _strongest(panel, byte_stats) -> list[str]:
    best = panel.get("strongest")
    if best is None:
        raise SystemExit("no comparator reaches the full-panel paired count; the headline is undefined")
    arm = M.arm(best["method"], best["configuration"])
    row = _contrast(panel, arm.code)
    stats = byte_stats[f"{best['method']}|{best['configuration']}"]
    dope = byte_stats["DOPE|headline"]
    linear = _contrast(panel, arm.code, "linear")
    return [cmd("VStrongLinDiff", num(linear["hl"])), cmd("VStrongLinLo", num(linear["hl_lo"])),
            cmd("VStrongLinHi", num(linear["hl_hi"])),
            cmd("VStrongestName", arm.display), cmd("VStrongDiff", num(row["hl"])),
            cmd("VStrongLo", num(row["hl_lo"])), cmd("VStrongHi", num(row["hl_hi"])),
            cmd("VStrongN", row["n"]), cmd("VStrongWtl", f"{row['wins']}/{row['ties']}/{row['losses']}"),
            cmd("VStrongHolm", pval(row.get("holm_p"))), cmd("VVerdictStrong", VERDICT[row["verdict"]]),
            cmd("VStrongBytes", human_bytes(stats["median"])),
            cmd("VStrongRatio", ratio(stats["median"] / dope["median"]))]


def _seed_macros(panel) -> list[str]:
    seed = panel["seed_rank"]
    later = {(row["size"], row["auditor"]): row for row in panel["seeds_23_71"]}[(4, "catboost")]
    two = panel["two_by_two"]
    out = [cmd("VSeedTop", seed["seed11_top"]), cmd("VSeedN", seed["n"]), cmd("VSeedP", pval(seed["p_one_sided"])),
           cmd("VLaterCbFour", num(later["median"])), cmd("VLaterCbFourLo", num(later.get("lo"))),
           cmd("VLaterCbFourHi", num(later.get("hi"))), cmd("VLaterCbFourN", later["n"]),
           cmd("VTwoN", two["n_common"])]
    for key, name in (("bhist_seed11", "VTwoBhistEleven"), ("bhist_seeds23_71", "VTwoBhistLater"),
                      ("bnew_seed11", "VTwoBnewEleven"), ("bnew_seeds23_71", "VTwoBnewLater"),
                      ("historical_seed11", "VTwoHist")):
        out.append(cmd(name, num(two.get(key))))
        out.append(cmd(name + "Own", num(two.get(key + "_own_median"))))
        out.append(cmd(name + "OwnN", two.get(key + "_own_n", 0)))
    return out


def _byte_macros(byte_stats, fits) -> list[str]:
    out = []
    for key, stats in sorted(byte_stats.items()):
        if stats is None:
            continue
        code = M.arm(*key.split("|", 1)).code
        out += [cmd("VBytes" + code, human_bytes(stats["median"])),
                cmd("VBytesExact" + code, exact_bytes(stats["median"])),
                cmd("VWithin" + code, f"{stats['within_cap']}/{stats['n']}")]
    dope = byte_stats["DOPE|headline"]
    out += [cmd("VDopeBytes", exact_bytes(dope["median"])),
            cmd("VDopeWithin", f"{dope['fits_within_cap']}/{dope['fits_ok']}"),
            cmd("VDopeFitsOk", f"{dope['fits_ok']}/{dope['fits_total']}")]
    return out


def _simulated_count() -> list[str]:
    sys.path.insert(0, str(REPO))
    from research.benchmark.review_fixes.stats import SIMULATED, family_of
    record = json.loads((RESULTS / "s3-lineage-record.json").read_text())
    count = sum(1 for row in record["rows"] if family_of(row["display_name"]) in SIMULATED)
    return [cmd("VSimulatedN", count)]


def _sentences(panel, byte_stats) -> list[str]:
    levels = {(_code(row), row["auditor"], row["size"]): row for row in panel["levels"]}
    generators = [arm for arm in M.ARMS if arm.role in {"headline", "generator"}
                  and f"{arm.method}|{arm.configuration}" in byte_stats
                  and byte_stats[f"{arm.method}|{arm.configuration}"] is not None
                  and (arm.code, "catboost", 4) in levels]
    points = [(arm, byte_stats[f"{arm.method}|{arm.configuration}"]["median"],
               levels[(arm.code, "catboost", 4)]["median"]) for arm in generators]
    pareto = [arm.display for arm, size, value in points
              if not any(other_size <= size and other_value >= value and (other_size, other_value) != (size, value)
                         for _, other_size, other_value in points)]
    dope_bytes = byte_stats["DOPE|headline"]["median"]
    heavy = {"CTGAN", "TVAE", "TabSyn", "ARF", "ForestDiffusion/Forest-Flow", "TabDDPM", "TabDiff", "GReaT"}
    larger = [math.log10(size / dope_bytes) for arm, size, _ in points if arm.method in heavy]
    full = [arm.code for arm in generators if levels[(arm.code, "catboost", 4)]["n"] >= 90]
    order = {size: sorted(full, key=lambda code: -levels[(code, "catboost", size)]["median"])
             for size in (1, 4) if all((code, "catboost", size) in levels for code in full)}
    same = len(order) == 2 and order[1] == order[4]
    lifted = sum(1 for code in full if code != "Dope"
                 and levels[(code, "catboost", 4)]["median"] > levels[(code, "catboost", 1)]["median"])
    equivalent = [row for row in panel["contrasts"] if (row.get("equivalence") or {}).get("equivalent")]
    return [cmd("VParetoSet", ", ".join(pareto)), cmd("VOrdersLo", f"{min(larger):.1f}" if larger else "---"),
            cmd("VOrdersHi", f"{max(larger):.1f}" if larger else "---"),
            cmd("VSizeRankSame", "is the same at both sizes" if same else "changes between the two sizes"),
            cmd("VSizeLifted", f"{lifted} of {len(full) - 1}"), cmd("VEquivCount", len(equivalent)),
            cmd("VEquivSentence", "No comparison meets that rule." if not equivalent else
                f"{len(equivalent)} comparisons meet that rule; supplement~E lists them.")]


def _privacy_macros(privacy) -> list[str]:
    out = []
    keys = {"c2st_catboost_auc": "Ctwo", "dcr_fit_median": "DcrFit", "dcr_validation_median": "DcrVal",
            "nndr_fit_median": "Nndr", "distance_mia_auc": "Mia"}
    for row in privacy["summaries"]:
        if row["method"] != "DOPE" or row["size"] != 4 or row["metric"] not in keys:
            continue
        base = "V" + keys[row["metric"]] + "Dope"
        out += [cmd(base, num(row["median"])), cmd(base + "Lo", num(row["lo"])), cmd(base + "Hi", num(row["hi"])),
                cmd(base + "N", row["n"])]
    return out


def _fidelity_macros(panel) -> list[str]:
    out = []
    keys = {"marginal_ks_mean": "Ks", "pair_correlation_fidelity": "Corr", "c2st_auc_logistic": "Logit"}
    for row in panel.get("fidelity", []):
        if (row["method"], row["configuration"], row["size"]) == ("DOPE", "headline", 4):
            base = "V" + keys[row["metric"]] + "Dope"
            out += [cmd(base, num(row["median"])), cmd(base + "Lo", num(row.get("lo"))),
                    cmd(base + "Hi", num(row.get("hi")))]
    return out


def headline_table(panel, byte_stats) -> str:
    rows = []
    strongest = panel["strongest"]
    lead = M.arm(strongest["method"], strongest["configuration"]).code
    codes = [lead] + [code for code in HEADLINE_ORDER if code != lead]
    for code in codes:
        cells = [_contrast(panel, code, auditor) for auditor in ("catboost", "linear", "mlp")]
        if cells[0] is None:
            continue
        arm = M.BY_CODE[code]
        stats = byte_stats.get(f"{arm.method}|{arm.configuration}")
        diffs = " & ".join("---" if row is None else f"${num(row['hl'])}$ \\scriptsize$[{num(row['hl_lo'])},{num(row['hl_hi'])}]$"
                           for row in cells)
        lead_row = cells[0]
        rows.append(f"{arm.display} & {lead_row['n']} & {diffs} & {lead_row['wins']}/{lead_row['ties']}/{lead_row['losses']}"
                    f" & ${pval(lead_row.get('holm_p'))}$ & {'---' if stats is None else '$' + human_bytes(stats['median']) + '$'} \\\\")
    head = ("\\begin{tabular}{@{}lrlllrrr@{}}\n\\toprule\n"
            "Comparator & $n$ & CatBoost & Linear & MLP & W/T/L & Holm $p$ & Bytes \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def size_table(panel) -> str:
    levels = {(_code(row), row["auditor"], row["size"]): row for row in panel["levels"]}
    rows = []
    for code in LEVEL_ORDER:
        if (code, "catboost", 4) not in levels:
            continue
        cells = []
        for auditor in ("catboost", "linear", "mlp"):
            for size in (1, 4):
                row = levels.get((code, auditor, size))
                cells.append("---" if row is None else
                             f"\\shortstack{{${num(row['median'])}$\\\\\\tiny$[{num(row.get('lo'), 2)},{num(row.get('hi'), 2)}]$}}")
        rows.append(f"{M.BY_CODE[code].display} & {levels[(code, 'catboost', 4)]['n']} & " + " & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{@{}lrcccccc@{}}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{CatBoost} & \\multicolumn{2}{c}{Linear} & \\multicolumn{2}{c}{MLP} \\\\\n"
            "Method & $n$ & $n$ rows & $4n$ rows & $n$ rows & $4n$ rows & $n$ rows & $4n$ rows \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def fidelity_table(privacy) -> str:
    order = ("DOPE|features12_steps2048", "ARF|author_default", "ARF|native_selected", "GaussianCopula|native_selected",
             "Chow-Liu|native_selected", "independent_marginals|native_selected", "TVAE|native_selected",
             "CTGAN|native_selected")
    metrics = ("c2st_catboost_auc", "dcr_fit_median", "dcr_validation_median", "nndr_fit_median", "distance_mia_auc")
    index = {(row["method"] + "|" + row["configuration"], row["metric"]): row
             for row in privacy["summaries"] if row["size"] == 4}
    rows = []
    for key in order:
        method, configuration = key.split("|")
        name = "DOPE (fit seed 11)" if method == "DOPE" else M.arm(method, configuration).display
        cells = []
        for metric in metrics:
            row = index.get((key, metric))
            cells.append("---" if row is None else f"${num(row['median'])}$ \\scriptsize$[{num(row['lo'], 2)},{num(row['hi'], 2)}]$")
        count = (index.get((key, "c2st_catboost_auc")) or {}).get("n", "---")
        rows.append(f"{name} & {count} & " + " & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{@{}lrlllll@{}}\n\\toprule\n"
            "Method & $n$ & C2ST & DCR fit & DCR val. & NNDR & MIA AUC \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def _longtable(columns: str, head: str, rows: list[str], caption: str, label: str) -> str:
    return (f"\\begin{{longtable}}{{{columns}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{head} \\\\\n"
            "\\midrule\n\\endfirsthead\n\\toprule\n" + head + " \\\\\n\\midrule\n\\endhead\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")


def contrasts_table(panel) -> str:
    rows = []
    order = {code: index for index, code in enumerate(HEADLINE_ORDER)}
    for row in sorted(panel["contrasts"], key=lambda r: (r["size"], order.get(_code(r), 99), r["auditor"])):
        arm = M.arm(row["method"], row["configuration"])
        equivalence = row.get("equivalence") or {}
        yuen = (row.get("yuen") or {}).get("equivalent")
        holm = "---" if row.get("holm_p") is None else f"${pval(row['holm_p'])}$ ({row['holm_family_n']})"
        rows.append(" & ".join([
            "$n$" if row["size"] == 1 else "$4n$", M.AUDITOR_DISPLAY[row["auditor"]], arm.display, str(row["n"]),
            f"${num(row['median'])}$ \\scriptsize${interval(row['median_lo'], row['median_hi'])}$",
            f"${num(row['hl'])}$ \\scriptsize${interval(row['hl_lo'], row['hl_hi'])}$",
            f"{row['wins']}/{row['ties']}/{row['losses']}", holm,
            "---" if equivalence.get("equivalent") is None else ("yes" if equivalence["equivalent"] else "no"),
            "---" if yuen is None else ("yes" if yuen else "no")]) + " \\\\")
    head = "Size & Auditor & Comparator & $n$ & Median diff. & Hodges--Lehmann & W/T/L & Holm $p$ (family) & HL eq. & Yuen eq."
    caption = ("\\textbf{All DOPE-minus-comparator contrasts.} Nested bootstrap 95\\% intervals. The Holm family size "
               "is in parentheses; families are declared per size in predeclare\\_v2. HL eq.\\ is the pre-declared "
               "equivalence rule (90\\% interval of the Hodges--Lehmann difference inside $\\pm 0.02$); Yuen eq.\\ is the "
               "trimmed-mean sensitivity.")
    return _longtable("@{}llp{1.25in}rllrlll@{}", head, rows, caption, "tab:v2-contrasts")


def levels_table(panel) -> str:
    rows = []
    order = {code: index for index, code in enumerate(LEVEL_ORDER)}
    for row in sorted(panel["levels"], key=lambda r: (order.get(_code(r), 99), _code(r), r["size"], r["auditor"])):
        arm = M.arm(row["method"], row["configuration"])
        rows.append(" & ".join([arm.display, "$n$" if row["size"] == 1 else "$4n$", M.AUDITOR_DISPLAY[row["auditor"]],
                                str(row["n"]), str(row.get("n_fits", "---")),
                                f"${num(row['median'])}$", f"${interval(row.get('lo'), row.get('hi'))}$"]) + " \\\\")
    head = "Arm & Size & Auditor & Lineages & Fits & Median & 95\\% nested interval"
    caption = ("\\textbf{Retention level of every arm.} Median of lineage values; a five-seed arm needs three complete "
               "fits per lineage. No Holm family.")
    return _longtable("@{}p{1.6in}llrrrl@{}", head, rows, caption, "tab:v2-levels")


def seeds_table(panel) -> str:
    seed = panel["seed_rank"]
    two = panel["two_by_two"]
    rows = [f"Seed 11 highest of five (CatBoost, $4n$) & {seed['seed11_top']} of {seed['n']} & expected {num(seed['expected'], 1)}; one-sided $p={pval(seed['p_one_sided'])}$ \\\\"]
    for row in panel["seeds_23_71"]:
        rows.append(f"Seeds 23--71 only, {M.AUDITOR_DISPLAY[row['auditor']]}, {'$n$' if row['size'] == 1 else '$4n$'} & "
                    f"{row['n']} lineages & ${num(row['median'])}$ ${interval(row.get('lo'), row.get('hi'))}$ \\\\")
    for key, label in (("bhist_seed11", "historical binary, seed 11"), ("bhist_seeds23_71", "historical binary, seeds 23--71"),
                       ("bnew_seed11", "later binary, seed 11"), ("bnew_seeds23_71", "later binary, seeds 23--71"),
                       ("historical_seed11", "historical seed-11 population")):
        rows.append(f"2$\\times$2 cell: {label} & {two['n_common']} common ({two.get(key + '_own_n', 0)} own) & "
                    f"${num(two.get(key))}$ (own lineages ${num(two.get(key + '_own_median'))}$) \\\\")
    return ("\\begin{tabular}{@{}p{2.6in}ll@{}}\n\\toprule\nDiagnostic & Count & Value \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def headline_json(panel, byte_stats) -> dict:
    out = {"format": "dope-v2-headline", "strongest": panel["strongest"], "arms": []}
    levels = {(_code(row), row["auditor"], row["size"]): row for row in panel["levels"]}
    for arm in M.ARMS:
        level = levels.get((arm.code, "catboost", 4))
        if level is None:
            continue
        stats = byte_stats.get(f"{arm.method}|{arm.configuration}")
        out["arms"].append({
            "code": arm.code, "method": arm.method, "configuration": arm.configuration,
            "display": arm.display, "role": arm.role, "color": arm.color, "marker": arm.marker,
            "level": {key: level.get(key) for key in ("median", "lo", "hi", "n")},
            "bytes": None if stats is None else {key: stats[key] for key in ("median", "q1", "q3", "n", "within_cap")},
            "contrasts": {auditor: None if (row := _contrast(panel, arm.code, auditor)) is None else
                          {key: row[key] for key in ("hl", "hl_lo", "hl_hi", "n", "wins", "ties", "losses", "verdict")}
                          for auditor in ("catboost", "linear", "mlp")}})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-panel", type=Path, default=PANEL)
    parser.add_argument("--out", type=Path, default=GENERATED)
    args = parser.parse_args()
    panel = json.loads(args.from_panel.read_text())
    if panel.get("official_tests_opened") is not False:
        raise SystemExit("v2 panel official test flag is not closed")
    fits = [json.loads(line) for line in FITS.read_text().splitlines() if line.strip()]
    privacy = json.loads(PRIVACY.read_text())
    byte_stats = v2_bytes.all_arms(fits)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v2-numbers.tex").write_text(macros(panel, byte_stats, fits, privacy))
    (args.out / "v2-headline-table.tex").write_text(headline_table(panel, byte_stats))
    (args.out / "v2-size-table.tex").write_text(size_table(panel))
    (args.out / "v2-fidelity-table.tex").write_text(fidelity_table(privacy))
    (args.out / "v2-contrasts.tex").write_text(contrasts_table(panel))
    (args.out / "v2-levels.tex").write_text(levels_table(panel))
    (args.out / "v2-seeds.tex").write_text(seeds_table(panel))
    (args.out / "v2-headline.json").write_text(json.dumps(headline_json(panel, byte_stats), indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
