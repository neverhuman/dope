"""Summarize the equal-byte study into the KPIs reported in the paper and README.

Retention is null-normalized against the matched real reference (the MLP's
better of the real rows as given or resampled to the synthetic size)
and clamped to [0, 1] per auditor, the TDL contract's frontier clipping; a
lineage's robust retention is the minimum over its informative auditors (real
gain at least 1% of the null loss). Seeds are aggregated by the median in both
arms. For each arm the frontier F(B) is the best robust retention among its
configurations whose median charged bytes are at most B, and TDL(tau) is the
smallest power-of-two budget whose frontier value reaches tau. The bytes real
rows need to match a retention level interpolate the real-row frontier
linearly in log bytes between the two budgets that bracket it.

    python3 -B -m research.benchmark.equal_bytes.summarize RESULTS_DIR --out SUMMARY.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

AUDITORS = ("linear", "mlp", "catboost")
GRID = [128 * 2 ** k for k in range(10)]
THRESHOLDS = (0.8, 0.9)
HEADLINE = "L4"
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 1729


EXCLUDE: tuple[str, ...] = ()  # auditors left out of the robust minimum (sensitivity runs)
AS_GIVEN = False  # sensitivity: real rows trained as given, no resampling to the synthetic size


def reference(record: dict) -> dict:
    """Real-row losses that synthetic and subset arms are normalized against."""
    return record["real"] if AS_GIVEN else record.get("real_matched", record["real"])


def subset_loss(record: dict, row: dict) -> dict:
    if AS_GIVEN:
        return record["real"] if row.get("anchored") else row.get("loss_as_given", row["loss"])
    return row["loss"]


def informative(record: dict) -> list[str]:
    null = record["null"]
    return [a for a in AUDITORS if a not in EXCLUDE and null - reference(record)[a] >= 0.01 * abs(null)]


def retention(record: dict, auditor: str, loss: float) -> float:
    gain = record["null"] - reference(record)[auditor]
    if gain <= 0:
        return 0.0
    return min(max((record["null"] - loss) / gain, 0.0), 1.0)


def robust(record: dict, per_auditor: dict) -> float | None:
    auditors = informative(record)
    return min(per_auditor[a] for a in auditors) if auditors else None


RUNGS: tuple[str, ...] | None = None  # restrict the DOPE ladder (set by --rungs)


def dope_points(record: dict, arm: str = "dope") -> dict:
    """Median bytes and robust retention for each rung of a DOPE arm."""
    points = {}
    for rung in sorted({row["rung"] for row in record.get(arm, [])}):
        if arm == "dope" and RUNGS is not None and rung not in RUNGS:
            continue
        fits = [row for row in record[arm] if row["rung"] == rung and "loss" in row]
        if not fits:
            continue
        per_auditor = {a: float(np.median([retention(record, a, f["loss"][a]) for f in fits])) for a in AUDITORS}
        points[rung] = {"bytes": float(np.median([f["bytes"] for f in fits])),
                        "robust": robust(record, per_auditor), "auditors": per_auditor}
        historical = [f["historical_codec_bytes"] for f in fits if f.get("historical_codec_bytes") is not None]
        if historical:
            points[rung]["historical_codec_bytes"] = float(np.median(historical)) + record["projection_bytes"]
    return points


def data_points(record: dict) -> list[dict]:
    """Seed-median retention per (budget, bits) cell of the real-row arm."""
    cells = {}
    for row in record.get("data", []):
        cells.setdefault((row["budget"], row["bits"]), []).append(row)
    points = []
    for (budget, bits), rows in sorted(cells.items()):
        per_auditor = {a: float(np.median([retention(record, a, subset_loss(record, r)[a]) for r in rows]))
                       for a in AUDITORS}
        points.append({"budget": budget, "bits": bits, "bytes": float(np.median([r["bytes"] for r in rows])),
                       "k": int(np.median([r["k"] for r in rows])), "robust": robust(record, per_auditor),
                       "auditors": per_auditor})
    return points


def frontier(points, budget: float) -> float | None:
    values = [p["robust"] for p in points if p["bytes"] <= budget and p["robust"] is not None]
    return max(values) if values else None


def bytes_to_reach(points, level: float) -> float | None:
    """Bytes at which the frontier reaches `level`, interpolated in log bytes between bracketing cells."""
    best, below = -math.inf, None
    for point in sorted((p for p in points if p["robust"] is not None), key=lambda p: p["bytes"]):
        if point["robust"] <= best:
            continue
        if point["robust"] >= level:
            if below is None:
                return point["bytes"]
            share = (level - below["robust"]) / (point["robust"] - below["robust"])
            return float(math.exp(math.log(below["bytes"]) + share * (math.log(point["bytes"]) - math.log(below["bytes"]))))
        best, below = point["robust"], point
    return None


def tdl(points, threshold: float) -> int | None:
    for budget in GRID:
        value = frontier(points, budget)
        if value is not None and value >= threshold:
            return budget
    return None


def bootstrap_median(values: list[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.median(rng.choice(data, size=(BOOTSTRAP_DRAWS, len(data)), replace=True), axis=1)
    return float(np.median(data)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def two_part_proxy(record: dict, robust: float, program_bytes: float) -> float:
    """Data-equivalence ratio predicted by the two-part-code argument (docs/math.md).

    Real rows needed to reach retention T scale with the noise-to-signal ratio
    over (1 - T), capped by the rows available; each row costs (d + 1) values.
    """
    noise = min(record["real"][a] for a in AUDITORS) / record["null"]
    rows_needed = noise / max(1.0 - noise, 1e-9) / max(1.0 - robust, 1e-3)
    rows_needed = 1.0 / (1.0 / rows_needed + 1.0 / record["n_fit"])
    return rows_needed * (record["d"] + 1) / program_bytes


def lineage_summary(record: dict) -> dict:
    dope = dope_points(record)
    default = dope_points(record, "dope_cpu_default").get("cpu_default")
    data = data_points(record)
    head = dope.get(HEADLINE)
    out = {"dataset": record["dataset"], "n_fit": record["n_fit"], "d": record["d"],
           "informative_auditors": informative(record), "dope": dope, "cpu_default": default,
           "projection_json_bytes": record["projection_json_bytes"],
           "projection_bytes": record["projection_bytes"],
           "full_table_bytes": record["full_table_bytes"], "data": data,
           "tdl": {}}
    for threshold in THRESHOLDS:
        key = f"T{int(threshold * 100)}"
        out["tdl"][key] = {"dope": tdl(list(dope.values()), threshold), "data": tdl(data, threshold)}
    if head and head["robust"] is not None:
        at_equal = frontier(data, head["bytes"])
        needed = bytes_to_reach(data, head["robust"])
        out["headline"] = {"bytes": head["bytes"], "robust": head["robust"],
                           "data_at_equal_bytes": at_equal,
                           "data_bytes_to_match": needed,
                           "data_equivalence_ratio": needed / head["bytes"] if needed else None,
                           "beyond_full_table": needed is None,
                           "two_part_proxy": two_part_proxy(record, head["robust"], head["bytes"])}
    return out


def per_learner_gaps(lineages: list[dict]) -> dict:
    """Per learner: median program-minus-rows retention at the program's bytes, wins, losses, tables."""
    out = {}
    for auditor in AUDITORS:
        diffs = []
        for x in lineages:
            if auditor not in x["informative_auditors"]:
                continue
            head = x["dope"][HEADLINE]
            rows = [p["auditors"][auditor] for p in x["data"] if p["bytes"] <= head["bytes"]]
            diffs.append(head["auditors"][auditor] - (max(rows) if rows else 0.0))
        out[auditor] = [float(np.median(diffs)) if diffs else None, int(sum(d > 0.01 for d in diffs)),
                        int(sum(d < -0.01 for d in diffs)), len(diffs)]
    return out


def aggregate(lineages: list[dict]) -> dict:
    heads = [x for x in lineages if "headline" in x]
    gap = [x["headline"]["robust"] - (x["headline"]["data_at_equal_bytes"] if x["headline"]["data_at_equal_bytes"]
           is not None else 0.0) for x in heads]
    ratios = [x["headline"]["data_equivalence_ratio"] if x["headline"]["data_equivalence_ratio"] is not None
              else math.inf for x in heads]
    packed = [x["dope"][HEADLINE]["bytes"] for x in heads]
    historical = [x["dope"][HEADLINE]["historical_codec_bytes"] - x["projection_bytes"] + x["projection_json_bytes"]
                  for x in heads]
    defaults = [x for x in heads if x["cpu_default"] and x["cpu_default"]["robust"] is not None]
    tdl_wins = {}
    for key in ("T80", "T90"):
        both = [(x["tdl"][key]["dope"], x["tdl"][key]["data"]) for x in lineages]
        tdl_wins[key] = {
            "dope_smaller": sum(1 for d, r in both if d is not None and (r is None or d < r)),
            "equal": sum(1 for d, r in both if d is not None and d == r),
            "data_smaller": sum(1 for d, r in both if r is not None and (d is None or r < d)),
            "neither": sum(1 for d, r in both if d is None and r is None),
        }
    large = [x for x in heads if x["n_fit"] >= 2000]
    large_gap = [g for x, g in zip(heads, gap) if x["n_fit"] >= 2000]
    from scipy.stats import spearmanr
    # Prop. 3: the gap should rise with how much larger the whole table is than the program.
    size = [x["full_table_bytes"]["8"] / x["headline"]["bytes"] for x in heads]
    by_size = spearmanr(size, gap)
    # Where the whole table fits in the program's bytes the gap is at most zero by construction.
    outgrown = [(v, g) for v, g in zip(size, gap) if v > 1]
    by_size_outgrown = spearmanr(*zip(*outgrown))
    finite = [x for x in heads if x["headline"]["data_equivalence_ratio"] is not None]
    by_rows = spearmanr([x["headline"]["data_equivalence_ratio"] for x in finite], [x["n_fit"] for x in finite])
    large_finite = [x for x in finite if x["n_fit"] >= 2000]
    by_proxy = spearmanr([x["headline"]["data_equivalence_ratio"] for x in large_finite],
                         [x["headline"]["two_part_proxy"] for x in large_finite])
    return {
        # Rounded so scipy versions agree on the committed bytes.
        "ratio_vs_rows_spearman": [round(float(by_rows.statistic), 6), float(f"{by_rows.pvalue:.4g}"), len(finite)],
        "ratio_vs_proxy_large_spearman": [round(float(by_proxy.statistic), 6), float(f"{by_proxy.pvalue:.4g}"),
                                          len(large_finite)],
        "lineages": len(lineages),
        "headline_rung": HEADLINE,
        "headline_bytes_median": float(np.median(packed)),
        "headline_robust_median": bootstrap_median([x["headline"]["robust"] for x in heads]),
        "historical_style_bytes_median": float(np.median(historical)),
        "byte_reduction_factor_median": float(np.median(np.array(historical) / np.array(packed))),
        "equal_byte_gap_median": bootstrap_median(gap),
        "equal_byte_wins": int(sum(g > 0.01 for g in gap)), "equal_byte_losses": int(sum(g < -0.01 for g in gap)),
        "data_equivalence_ratio_median": float(np.median(ratios)),
        "data_equivalence_ratio_large_median": float(np.median([r for x, r in zip(heads, ratios) if x["n_fit"] >= 2000])),
        "large_lineages": len(large),
        "large_equal_byte_gap_median": bootstrap_median(large_gap) if large_gap else None,
        "large_equal_byte_wins": int(sum(g > 0.01 for g in large_gap)),
        "large_equal_byte_losses": int(sum(g < -0.01 for g in large_gap)),
        "gap_vs_table_size_spearman": [round(float(by_size.statistic), 6), float(f"{by_size.pvalue:.4g}"), len(heads)],
        "gap_vs_table_size_spearman_outgrown": [round(float(by_size_outgrown.statistic), 6),
                                                float(f"{by_size_outgrown.pvalue:.4g}"), len(outgrown)],
        "large_per_learner_gap": per_learner_gaps(large),
        "beyond_full_table": int(sum(1 for r in ratios if math.isinf(r))),
        "cpu_default_robust_median": float(np.median([x["cpu_default"]["robust"] for x in defaults])),
        "trained_minus_default_median": bootstrap_median(
            [x["headline"]["robust"] - x["cpu_default"]["robust"] for x in defaults]),
        "tdl_wins": tdl_wins,
    }


SENSITIVITY_KEYS = ("headline_robust_median", "equal_byte_gap_median", "equal_byte_wins", "equal_byte_losses",
                    "data_equivalence_ratio_median", "large_equal_byte_gap_median", "large_equal_byte_wins",
                    "large_equal_byte_losses", "large_per_learner_gap", "tdl_wins")


def sensitivity(records: list[dict]) -> dict:
    """The headline KPIs without the MLP, and with real rows trained as given."""
    global EXCLUDE, AS_GIVEN
    out = {}
    for name, exclude, as_given in (("without_mlp", ("mlp",), False), ("as_given", (), True)):
        EXCLUDE, AS_GIVEN = exclude, as_given
        try:
            result = aggregate([lineage_summary(r) for r in records])
        finally:
            EXCLUDE, AS_GIVEN = (), False
        out[name] = {k: result[k] for k in SENSITIVITY_KEYS}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rungs", help="comma-separated DOPE rungs to keep, for a uniform ladder across tables")
    args = parser.parse_args()
    global RUNGS
    RUNGS = tuple(args.rungs.split(",")) if args.rungs else None
    paths = sorted((p for d in args.results for p in d.glob("*.json")), key=lambda p: p.name)
    records = [json.loads(p.read_text()) for p in paths]
    lineages = [lineage_summary(r) for r in records]
    summary = {"format": "dope-equal-bytes-summary-v2", "evidence_status": "development_validation",
               "aggregate": aggregate(lineages), "sensitivity": sensitivity(records), "lineages": lineages}
    args.out.write_text(json.dumps(summary, indent=1, sort_keys=True, default=lambda v: None) + "\n")
    print(json.dumps(summary["aggregate"], indent=1))


if __name__ == "__main__":
    main()
