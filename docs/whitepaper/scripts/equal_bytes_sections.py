"""Equal-byte renderer sections: full panel, protocol sensitivity, copula ablation, UCI Adult."""
from __future__ import annotations

import json
import sys

import numpy as np

from equal_bytes_common import ADULT, RESULTS, REPO, fmt, human_bytes, latex_table

COPULA = RESULTS / "ablations" / "gaussian-copula"
SUBSET_ARMS = (("codes", "8-bit codes"), ("csv", "CSV"))


def gap_cells(prefix: str, block: dict) -> dict:
    """Macros for one aggregate block: equal-byte gap with its interval, wins and losses, all and large tables."""
    out = {f"{prefix}Ratio": fmt(block["data_equivalence_ratio_median"], 1)}
    for scope, key in (("", "equal_byte"), ("Large", "large_equal_byte")):
        gap = block[f"{key}_gap_median"]
        out.update({f"{prefix}{scope}Gap": fmt(gap[0]), f"{prefix}{scope}GapLo": fmt(gap[1]),
                    f"{prefix}{scope}GapHi": fmt(gap[2]), f"{prefix}{scope}Wins": str(block[f"{key}_wins"]),
                    f"{prefix}{scope}Losses": str(block[f"{key}_losses"])})
    return out


def panel_macros() -> dict:
    """The full 97-lineage panel (simulated families included, two-rung ladder)."""
    path = RESULTS / "panel-summary.json"
    if not path.is_file():
        return {}
    summary = json.loads(path.read_text())
    p = summary["aggregate"]
    losses = [x for x in summary["lineages"] if "headline" in x and x["headline"]["robust"]
              - (x["headline"]["data_at_equal_bytes"] or 0.0) < -0.01]
    names = {"Lineages": str(p["lineages"]), "Bytes": f"{p['headline_bytes_median']:,.0f}",
             "Robust": fmt(p["headline_robust_median"][0]),
             "RatioLarge": fmt(p["data_equivalence_ratio_large_median"], 1), "Large": str(p["large_lineages"]),
             "ByteFactor": fmt(p["byte_reduction_factor_median"], 1), "BeyondFull": str(p["beyond_full_table"]),
             "RhoRows": fmt(p["ratio_vs_rows_spearman"][0]), "RhoRowsN": str(p["ratio_vs_rows_spearman"][2]),
             "RhoProxy": fmt(p["ratio_vs_proxy_large_spearman"][0]),
             "RhoProxyN": str(p["ratio_vs_proxy_large_spearman"][2]),
             "RhoSize": fmt(p["gap_vs_table_size_spearman"][0]), "RhoSizeN": str(p["gap_vs_table_size_spearman"][2]),
             "LossMaxRows": f"{max((x['n_fit'] for x in losses), default=0):,}",
             "LossMaxFull": f"{max((x['full_table_bytes']['8'] for x in losses), default=0):,}"}
    for key, threshold in (("Eighty", "T80"), ("Ninety", "T90")):
        names[f"Tdl{key}Dope"] = str(p["tdl_wins"][threshold]["dope_smaller"])
        names[f"Tdl{key}Data"] = str(p["tdl_wins"][threshold]["data_smaller"])
    out = {f"EbPanel{k}": v for k, v in names.items()}
    out.update(gap_cells("EbPanel", p))
    out.update(gap_cells("EbPanelNoMlp", summary["sensitivity"]["without_mlp"]))
    return out


def sensitivity_macros(summary: dict) -> dict:
    """The headline KPIs without the MLP and with real rows trained as given."""
    out = gap_cells("EbNoMlp", summary["sensitivity"]["without_mlp"])
    out.update(gap_cells("EbAsGiven", summary["sensitivity"]["as_given"]))
    for name, key in (("NoMlp", "without_mlp"), ("AsGiven", "as_given")):
        block = summary["sensitivity"][key]
        out[f"Eb{name}Robust"] = fmt(block["headline_robust_median"][0])
        out[f"Eb{name}TdlNinetyDope"] = str(block["tdl_wins"]["T90"]["dope_smaller"])
        out[f"Eb{name}TdlNinetyData"] = str(block["tdl_wins"]["T90"]["data_smaller"])
    return out


def copula_points() -> list[dict]:
    """Headline rung against the exact Gaussian copula rung (G4), table by table."""
    sys.path.insert(0, str(REPO))
    from research.benchmark.equal_bytes import summarize
    rows = []
    for path in sorted(COPULA.glob("*.json")):
        ablation = json.loads(path.read_text())
        record = json.loads((RESULTS / "lineages" / path.name).read_text())
        merged = dict(record, dope=record["dope"] + [f for f in ablation["dope"] if f.get("rung") == "G4"])
        points = summarize.dope_points(merged)
        if points.get("L4", {}).get("robust") is None or points.get("G4", {}).get("robust") is None:
            continue
        rows.append({"dataset": record["dataset"], "delta": points["G4"]["robust"] - points["L4"]["robust"],
                     "extra_bytes": points["G4"]["bytes"] - points["L4"]["bytes"]})
    return rows


def copula_macros() -> dict:
    rows = copula_points()
    if not rows:
        return {}
    sys.path.insert(0, str(REPO))
    from research.benchmark.equal_bytes import summarize
    delta = summarize.bootstrap_median([r["delta"] for r in rows])
    return {"EbCopulaTables": str(len(rows)), "EbCopulaDelta": fmt(delta[0], 3), "EbCopulaDeltaLo": fmt(delta[1], 3),
            "EbCopulaDeltaHi": fmt(delta[2], 3), "EbCopulaWins": str(sum(r["delta"] > 0.01 for r in rows)),
            "EbCopulaLosses": str(sum(r["delta"] < -0.01 for r in rows)),
            "EbCopulaExtraBytes": f"{np.median([r['extra_bytes'] for r in rows]):,.0f}"}


def adult_rows() -> dict:
    adult = json.loads(ADULT.read_text())
    null = adult["null_log_loss"]
    real = adult["arms"]["real rows"]
    rows = {}
    for arm, scores in adult["arms"].items():
        rows[arm] = {name: (null - s["log_loss"]) / (null - real[name]["log_loss"]) for name, s in scores.items()}
    return {"adult": adult, "retention": rows}


def subset_arm(adult: dict, codec: str) -> str:
    return f"real rows in {adult['artifact_bytes']} B ({codec})"


def adult_macros() -> dict:
    data = adult_rows()
    adult, retention = data["adult"], data["retention"]
    out = {
        "AdultBytes": f"{adult['artifact_bytes']:,}",
        "AdultCsvBytes": f"{adult['train_file']['csv']:,}",
        "AdultGzipBytes": f"{adult['train_file']['gzip']:,}",
        "AdultCsvFactor": f"{adult['train_file']['csv'] / adult['artifact_bytes']:,.0f}",
        "AdultGzipFactor": f"{adult['train_file']['gzip'] / adult['artifact_bytes']:,.0f}",
        **{f"AdultDope{name.capitalize()}": fmt(value) for name, value in retention["DOPE synthetic"].items()},
        **{f"AdultAucReal{name.capitalize()}": fmt(s["auc"], 3) for name, s in adult["arms"]["real rows"].items()},
        **{f"AdultAucDope{name.capitalize()}": fmt(s["auc"], 3) for name, s in adult["arms"]["DOPE synthetic"].items()},
    }
    for codec, _ in SUBSET_ARMS:
        label = codec.capitalize()
        out[f"AdultSubsetRows{label}"] = f"{adult['subset_rows'][codec]:,}"
        for name, value in retention[subset_arm(adult, codec)].items():
            out[f"AdultSubset{label}{name.capitalize()}"] = fmt(value)
    return out


def adult_readme_rows() -> list[str]:
    data = adult_rows()
    adult, retention = data["adult"], data["retention"]
    cells = lambda arm: " | ".join(fmt(retention[arm][n]) for n in ("logistic", "mlp", "catboost"))
    rows = [f"| Real training rows (CSV, gzip) | {adult['train_file']['csv']:,}, {adult['train_file']['gzip']:,} "
            f"| {cells('real rows')} |",
            f"| DOPE program | {adult['artifact_bytes']:,} | {cells('DOPE synthetic')} |"]
    for codec, label in SUBSET_ARMS:
        rows.append(f"| Real rows in the same bytes, {label} ({adult['subset_rows'][codec]:,} rows) "
                    f"| {adult['artifact_bytes']:,} | {cells(subset_arm(adult, codec))} |")
    return rows


def adult_table() -> str:
    data = adult_rows()
    adult, retention = data["adult"], data["retention"]
    arms = [("Real training rows", f"{human_bytes(adult['train_file']['csv'])} (gzip "
             f"{human_bytes(adult['train_file']['gzip'])})", "real rows"),
            ("DOPE table artifact", f"{adult['artifact_bytes']:,}", "DOPE synthetic")]
    arms += [(f"Real rows, {label} ({adult['subset_rows'][codec]:,})", f"{adult['artifact_bytes']:,}",
              subset_arm(adult, codec)) for codec, label in SUBSET_ARMS]
    rows = []
    for label, size, arm in arms:
        cells = [f"{fmt(retention[arm][n])} / {fmt(adult['arms'][arm][n]['auc'], 3)}"
                 for n in ("logistic", "mlp", "catboost")]
        rows.append([label, size] + cells)
    return latex_table(["Training data", "Bytes", "Logistic", "MLP", "CatBoost"], rows, "lrrrr")
