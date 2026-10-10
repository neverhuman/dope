"""Reduce the predeclare_v2 scalar ledger into the v2 review panel.

Every estimand, family, and minimum follows predeclare_v2.json. The DOPE
headline arm is compared with every other arm on shared lineages. Missing
arms stay in their Holm family as untested members, so a family never shrinks
after the data are seen. Output floats are unrounded; emitters format them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from research.benchmark.review_fixes import stats_v2 as S
from research.benchmark.review_fixes.stats import cluster_of, family_of

HERE = Path(__file__).resolve().parent
PREDECLARE = HERE / "predeclare_v2.json"
AUDITORS = ("catboost", "linear", "mlp")
SIZES = (1, 4)
SAMPLE_SEEDS = (101, 211, 307)
HEADLINE = ("DOPE", "headline")
MIN_PAIRS = 6
STRONGEST_MIN_LINEAGES = 90
CAP = 10240
FAMILIES = {
    "primary": [("ARF", "author_default"), ("TabSyn", "scaled_200_vae_1000_diffusion"),
                ("TabDDPM", "author_default"), ("ForestDiffusion/Forest-Flow", "author_default_B1")],
    "density": [("GaussianCopula", "native_selected"), ("Chow-Liu", "native_selected"),
                ("independent_marginals", "native_selected")],
    "deep_secondary": [("CTGAN", "author_default"), ("TVAE", "author_default"),
                       ("TabDiff", "author_default"), ("GReaT", "author_default_distilgpt2")],
    "other_secondary": [("ARF", "native_selected"), ("synthpop_CART", "author_default"),
                        ("SMOTE", "k5_uniform_interpolation_on_features_and_target")],
    "controls": [("predictor_only", "control"), ("real_bootstrap_4n", "control")],
}
FIVE_SEED = {("DOPE", "headline"), ("DOPE", "product_default"), ("ARF", "author_default"),
             ("ARF", "native_selected"), ("predictor_only", "control")}
NOT_A_COMPARATOR = {"predictor_only", "real_bootstrap_4n"}


# Arms reported outside every Holm family; an arm in neither list is refused.
DESCRIPTIVE = {
    ("DOPE", "product_default"), ("DOPE", "headline_bnew"), ("DOPE", "fourseed_bnew"), ("DOPE", "historical_seed11"),
    ("DOPE", "historical_seed11_features12_steps512"), ("DOPE", "historical_seed11_features24_steps512"),
    ("DOPE", "historical_seed11_features24_steps2048"), ("CTGAN", "native_selected"), ("TVAE", "native_selected"),
    ("GaussianCopula", "default"), ("Chow-Liu", "default"), ("independent_marginals", "default"),
    ("ForestDiffusion/Forest-Flow", "default"), ("ForestDiffusion/Forest-Flow", "native_selected"),
    ("predictor_only", "control_split2027"), ("predictor_only", "control_split2999"),
    ("predictor_only", "control_split4099"), ("predictor_only", "control_split8191"),
}


def family_of_arm(arm: tuple[str, str]) -> str:
    for name, members in FAMILIES.items():
        if arm in members:
            return name
    if arm in DESCRIPTIVE:
        return "descriptive"
    raise ValueError(f"arm {arm} is in no declared family and no descriptive list")


def min_fits(arm: tuple[str, str]) -> int:
    return S.MIN_FITS_FIVE_SEED if arm in FIVE_SEED else 1


def fit_values(cells: list[dict], names: dict[str, str]) -> dict:
    """(arm, size, auditor) -> fit items. A fit needs three ok informative sample seeds."""
    triples: dict[tuple, dict[int, float | None]] = defaultdict(dict)
    for cell in cells:
        arm = (cell["method"], cell["configuration"])
        for auditor in AUDITORS:
            row = cell["utility"].get(auditor) or {}
            value = row.get("retention") if row.get("informative") is True else None
            key = (arm, cell["dataset"], cell["fit_seed"], cell["size"], auditor)
            if cell["sample_seed"] in triples[key]:
                raise ValueError(f"duplicate sample cell for {key}")
            triples[key][cell["sample_seed"]] = value
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for (arm, dataset, fit_seed, size, auditor), samples in sorted(triples.items(), key=lambda kv: repr(kv[0])):
        if set(samples) != set(SAMPLE_SEEDS) or any(samples[seed] is None for seed in SAMPLE_SEEDS):
            continue
        ordered = sorted(samples[seed] for seed in SAMPLE_SEEDS)
        name = names.get(dataset, dataset)
        grouped[(arm, size, auditor)].append({
            "dataset": dataset, "cluster": cluster_of(name, dataset), "family": family_of(name),
            "fit_seed": -1 if fit_seed is None else fit_seed, "value": float(ordered[1]),
        })
    return grouped


FIDELITY = ("marginal_ks_mean", "pair_correlation_fidelity", "c2st_auc_logistic")


def fidelity_values(cells: list[dict], names: dict[str, str]) -> dict:
    """(arm, size, metric) -> fit items; a fit value is the median over its three sample seeds."""
    triples: dict[tuple, dict[int, float]] = defaultdict(dict)
    for cell in cells:
        arm = (cell["method"], cell["configuration"])
        for metric in FIDELITY:
            if cell.get(metric) is not None:
                triples[(arm, cell["dataset"], cell["fit_seed"], cell["size"], metric)][cell["sample_seed"]] = cell[metric]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for (arm, dataset, fit_seed, size, metric), samples in sorted(triples.items(), key=lambda kv: repr(kv[0])):
        if set(samples) != set(SAMPLE_SEEDS):
            continue
        name = names.get(dataset, dataset)
        grouped[(arm, size, metric)].append({
            "dataset": dataset, "cluster": cluster_of(name, dataset),
            "fit_seed": -1 if fit_seed is None else fit_seed,
            "value": float(np.median([samples[seed] for seed in SAMPLE_SEEDS]))})
    return grouped


def _verdict(lo, hi) -> str | None:
    if lo is None or hi is None:
        return None
    if lo > 0:
        return "above"
    if hi < 0:
        return "below"
    return "inseparable"


def contrast(left: list[dict], right: list[dict], label: str, right_arm: tuple, draws: int) -> dict:
    paired = S.nested_paired(left, right, label, S.MIN_FITS_FIVE_SEED, min_fits(right_arm), draws)
    if paired["n"] < MIN_PAIRS:
        paired["wilcoxon_p"] = None
        paired["ci_status"] = "not_identified_below_minimum_pairs"
    paired["equivalence"] = S.equivalence_hl(paired)
    paired["yuen"] = S.yuen_tost(paired["differences"])
    paired["verdict"] = _verdict(paired["hl_lo"], paired["hl_hi"])
    return paired


def _level_task(task: tuple) -> dict:
    kind, key, items, label, minimum, draws = task
    return {"kind": kind, "key": key, **S.nested_level(items, label, minimum, draws)}


def _contrast_task(task: tuple) -> dict:
    key, left, right, label, arm, draws = task
    result = contrast(left, right, label, arm, draws)
    result.pop("differences")
    return {"key": key, **result}


def _run(function, tasks: list, jobs: int) -> list:
    """Every statistic seeds its own generator from its label, so order and workers never change a value."""
    if jobs <= 1 or len(tasks) < 2:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(function, tasks, chunksize=4))


def over_cap(fits: list[dict], cap: int = CAP) -> list[str]:
    """Lineages with at least one ok headline fit charged above the release cap."""
    return sorted({fit["dataset"] for fit in fits if fit["configuration"] == HEADLINE[1] and fit["status"] == "ok"
                   and isinstance(fit.get("charged_bytes"), int) and fit["charged_bytes"] > cap})


def cap_sensitivity(values: dict, excluded: list[str], comparators: list[tuple], draws: int) -> dict:
    """Post hoc sensitivity requested in review: the headline level and contrasts without over-cap lineages."""
    def keep(items):
        return [item for item in items if item["dataset"] not in excluded]
    levels = [{"size": 4, "auditor": auditor,
               **S.nested_level(keep(values.get((HEADLINE, 4, auditor), [])), f"cap|level|{auditor}",
                                S.MIN_FITS_FIVE_SEED, draws)} for auditor in AUDITORS]
    contrasts = []
    for arm in comparators:
        for auditor in AUDITORS:
            left, right = keep(values.get((HEADLINE, 4, auditor), [])), keep(values.get((arm, 4, auditor), []))
            if left and right:
                result = contrast(left, right, f"cap|paired|{arm}|{auditor}", arm, draws)
                result.pop("differences")
                contrasts.append({"method": arm[0], "configuration": arm[1], "size": 4, "auditor": auditor, **result})
    return {"cap": CAP, "excluded": excluded, "levels": levels, "contrasts": contrasts}


def reduce(cells: list[dict], names: dict[str, str], draws: int = S.DRAWS, jobs: int | None = None,
           predeclare: bytes | None = None, fits: list[dict] | None = None) -> dict:
    jobs = min(os.cpu_count() or 1, 16) if jobs is None else jobs
    values = fit_values(cells, names)
    arms = sorted({key[0] for key in values})
    level_tasks = [("level", (arm, size, auditor), values[(arm, size, auditor)], f"level|{arm}|{size}|{auditor}",
                    min_fits(arm), draws)
                   for arm in arms for size in SIZES for auditor in AUDITORS if values.get((arm, size, auditor))]
    fidelity_items = fidelity_values(cells, names)
    level_tasks += [("fidelity", key, items, f"fidelity|{key[0]}|{key[1]}|{key[2]}", min_fits(key[0]), draws)
                    for key, items in sorted(fidelity_items.items(), key=lambda kv: repr(kv[0]))]
    contrast_tasks = [((arm, size, auditor), values[(HEADLINE, size, auditor)], values[(arm, size, auditor)],
                       f"paired|{arm}|{size}|{auditor}", arm, draws)
                      for arm in arms if arm != HEADLINE for size in SIZES for auditor in AUDITORS
                      if values.get((HEADLINE, size, auditor)) and values.get((arm, size, auditor))]
    levels, fidelity = [], []
    for result in _run(_level_task, level_tasks, jobs):
        kind, key = result.pop("kind"), result.pop("key")
        if kind == "level":
            arm, size, auditor = key
            levels.append({"method": arm[0], "configuration": arm[1], "size": size, "auditor": auditor, **result})
        else:
            arm, size, metric = key
            fidelity.append({"method": arm[0], "configuration": arm[1], "size": size, "metric": metric, **result})
    contrasts = []
    for result in _run(_contrast_task, contrast_tasks, jobs):
        arm, size, auditor = result.pop("key")
        contrasts.append({"method": arm[0], "configuration": arm[1], "size": size, "auditor": auditor,
                          "family": family_of_arm(arm), **result})
    _holm(contrasts)
    seeds = [item for item in values.get((HEADLINE, 4, "catboost"), [])]
    later = {key: [item for item in items if item["fit_seed"] != 11]
             for key, items in values.items() if key[0] == HEADLINE}
    sensitivity = [{"size": size, "auditor": auditor,
                    **S.nested_level(later.get((HEADLINE, size, auditor), []),
                                     f"level|seeds23-71|{size}|{auditor}", S.MIN_FITS_FIVE_SEED, draws)}
                   for size in SIZES for auditor in AUDITORS]
    best = strongest(levels, contrasts)
    comparators = ([] if best is None else [(best["method"], best["configuration"])]) + [("ARF", "author_default")]
    cap_rows = None if fits is None else cap_sensitivity(values, over_cap(fits), comparators, draws)
    return {
        "format": "dope-review-fix-v2-panel", "version": 1, "fidelity": fidelity, "cap_sensitivity": cap_rows,
        "predeclare_sha256": hashlib.sha256(PREDECLARE.read_bytes() if predeclare is None else predeclare).hexdigest(),
        "draws": draws, "levels": levels, "contrasts": contrasts,
        "strongest": best,
        "seed_rank": S.seed_rank_test(seeds), "seeds_23_71": sensitivity,
        "two_by_two": two_by_two(values), "sample_identity": sample_identity(cells),
        "official_tests_opened": False, "formal_dp": False, "mfs_v3": None, "superiority": None,
    }


def _holm(contrasts: list[dict]) -> None:
    for size in SIZES:
        for name, members in FAMILIES.items():
            # Declared members stay in the family even when an arm has no cell at this size.
            slots = [(arm, auditor) for arm in members for auditor in AUDITORS]
            index = {(row["method"], row["configuration"], row["auditor"]): row
                     for row in contrasts if row["size"] == size and row["family"] == name}
            p_values = [(index.get((arm[0], arm[1], auditor)) or {}).get("wilcoxon_p") for arm, auditor in slots]
            for (arm, auditor), adjusted in zip(slots, S.holm(p_values)):
                row = index.get((arm[0], arm[1], auditor))
                if row is not None:
                    row["holm_p"] = adjusted
                    row["holm_family_n"] = len(slots)
    for row in contrasts:
        row.setdefault("holm_p", None)
        row.setdefault("holm_family_n", None)


def strongest(levels: list[dict], contrasts: list[dict]) -> dict | None:
    """Highest CatBoost 4n level among comparator arms paired with DOPE on >= 90 lineages."""
    paired_n = {(row["method"], row["configuration"]): row["n"] for row in contrasts
                if row["size"] == 4 and row["auditor"] == "catboost"}
    candidates = [row for row in levels if row["size"] == 4 and row["auditor"] == "catboost"
                  and row["method"] not in NOT_A_COMPARATOR and row["method"] != "DOPE"
                  and paired_n.get((row["method"], row["configuration"]), 0) >= STRONGEST_MIN_LINEAGES]
    if not candidates:
        return None
    best = max(candidates, key=lambda row: (row["median"], row["method"], row["configuration"]))
    return {"method": best["method"], "configuration": best["configuration"], "median": best["median"],
            "paired_n": paired_n[(best["method"], best["configuration"])]}


def two_by_two(values: dict) -> dict:
    """Binary by seed group on CatBoost 4n, on lineages present in all four cells."""
    def lineage_medians(arm, seeds):
        grouped = defaultdict(list)
        for item in values.get((arm, 4, "catboost"), []):
            if item["fit_seed"] in seeds:
                grouped[item["dataset"]].append(item["value"])
        return {key: float(np.median(vals)) for key, vals in grouped.items()}
    cells = {"bhist_seed11": lineage_medians(HEADLINE, {11}),
             "bhist_seeds23_71": lineage_medians(HEADLINE, {23, 37, 53, 71}),
             "bnew_seed11": lineage_medians(("DOPE", "headline_bnew"), {11}),
             "bnew_seeds23_71": lineage_medians(("DOPE", "fourseed_bnew"), {23, 37, 53, 71}),
             "historical_seed11": lineage_medians(("DOPE", "historical_seed11"), {11})}
    common = set.intersection(*(set(cell) for cell in cells.values())) if all(cells.values()) else set()
    out = {"n_common": len(common)}
    for name, medians in cells.items():
        shared = [medians[key] for key in sorted(common)]
        out[name] = float(np.median(shared)) if shared else None
        out[name + "_own_n"] = len(medians)
        out[name + "_own_median"] = float(np.median(list(medians.values()))) if medians else None
    return out


def check_fits(cells: list[dict], fits: list[dict]) -> None:
    """Every DOPE cell refit under v2 comes from an ok fit in the fit ledger, so bytes and retention share fits."""
    ok = {(fit["configuration"], fit["dataset"], fit["fit_seed"]) for fit in fits
          if fit.get("official_tests_opened") is False and fit["status"] == "ok"}
    if not ok:
        raise ValueError("fit ledger has no ok fit")
    refit = {fit["configuration"] for fit in fits}
    for cell in cells:
        if cell["method"] == "DOPE" and cell["configuration"] in refit:
            if (cell["configuration"], cell["dataset"], cell["fit_seed"]) not in ok:
                raise ValueError("a DOPE cell has no ok fit in the fit ledger")


def sample_identity(cells: list[dict]) -> dict:
    """Fits whose synthetic samples are byte-identical between the headline refit and a diagnostic DOPE arm.

    A fit counts when it shares at least one sample cell with the other arm and every shared cell
    has the same synthetic digest.
    """
    digests: dict[tuple, dict] = defaultdict(dict)
    for cell in cells:
        if cell["method"] == "DOPE":
            key = (cell["dataset"], cell["fit_seed"], cell["sample_seed"], cell["size"])
            digests[cell["configuration"]][key] = cell["synthetic_sha256"]
    out = {}
    for other in ("headline_bnew", "fourseed_bnew", "historical_seed11"):
        shared: dict[tuple, bool] = {}
        for key, digest in digests.get(other, {}).items():
            mine = digests.get(HEADLINE[1], {}).get(key)
            if mine is not None:
                fit = key[:2]
                shared[fit] = shared.get(fit, True) and mine == digest
        out[other] = {"fits": len(shared), "identical": sum(shared.values())}
    return out


def load_ledger(path: Path) -> list[dict]:
    cells = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(cell.get("official_tests_opened") is not False for cell in cells):
        raise ValueError("ledger official test flag is not closed")
    return cells


def names_of(record: dict) -> dict[str, str]:
    return {row["dataset"]: row["display_name"] for row in record["rows"]}


def canonical(panel: dict) -> str:
    return json.dumps(panel, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the pinned v2 panel from the v2 scalar ledger.")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--lineage-record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()
    names = names_of(json.loads(args.lineage_record.read_text()))
    fits = [json.loads(line) for line in args.ledger.with_name("fits.jsonl").read_text().splitlines() if line.strip()]
    args.out.write_text(canonical(reduce(load_ledger(args.ledger), names, jobs=args.jobs, fits=fits)))
    print(f"wrote {args.out} sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
