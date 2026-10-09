#!/usr/bin/env python3
"""Lineage-bootstrap intervals and paired tests from committed validation ledgers.

Reads published records only. Does not open official test files, does not fit
a generator, and does not treat a small p-value as PTF-v1, MFS-v2, or a
release-safe claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "benchmark" / "results"
OUT = REPO / "docs" / "whitepaper" / "generated"
SEED = 20261005
DRAWS = 10_000
AUDITORS = ("catboost", "linear", "mlp")
SEEDS = (101, 211, 307)
DENSITY_METHODS = ("DOPE", "GaussianCopula", "Chow-Liu", "independent_marginals")
DENSITY_COMPARATORS = ("GaussianCopula", "Chow-Liu", "independent_marginals")
BYTE_CAP = 10240
Q_NEMENYI_K4 = 2.569

ANCHORS = {
    ("density", "catboost", "DOPE"): (97, 0.939876353161328),
    ("density", "catboost", "GaussianCopula"): (None, 0.6556811260),
    ("density", "catboost", "Chow-Liu"): (None, 0.5859856074),
    ("density", "catboost", "independent_marginals"): (None, -0.0119708477),
    ("neural", "catboost", "DOPE"): (21, 0.9533658441),
    ("neural", "catboost", "CTGAN"): (21, -0.0812588018),
    ("neural", "catboost", "TVAE"): (21, 0.6690331042),
}


INPUT_SHA256 = {
    "density-matched-population-validation.json": "5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a",
    "expanded-validation-diagnostics.json": "cd20ac0d07a1dcab7c5e46db195adb463172b5174387bb71a054b2e5f340beeb",
    "mfs-v3-density-panel.json": "d2cbbb5c08d045c161e3809e150efeacafadcfbee5e2accde6ed39cc46fe9c25",
    "s3-coreset-procedure.json": "73251e43cf7838687427569931bf6127f42e932374af33137c57fd8fe8544704",
    "s3-data.lock.json": "857b61324d0a5b7dbbb98e0c147fcc79c5bb8616e7ca61afd6785469c17f038f",
    "s3-lineage-record.json": "bbd0852c49f595315104261efa4eaa6f50db257e323b8c002aac6737a2dbe7d7",
    "s3-matched-forest-confirmation-validation.json": "97a25be902954cad16c5b5802d0e0ba468dea104221efa2465434c94e6774a1f",
    "sdv-matched-population-validation.json": "4dc367eb78159a0386882d23ae1f99b0aa2cc9f425d1aae499b6f3453222a44e"
}

def load(name):
    """Authenticate the complete input bytes before decoding a committed ledger."""
    if name not in INPUT_SHA256:
        raise ValueError("unregistered panel input")
    path = RESULTS / name
    if path.is_symlink():
        raise ValueError("symlinked panel input")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA256[name]:
        raise ValueError("panel input digest mismatch")
    return json.loads(raw)


KNOWN_METHODS = set(DENSITY_METHODS) | {"CTGAN", "TVAE", "ForestDiffusion/Forest-Flow"}


def validate_rows(rows, sampled=False):
    """Reject unknown methods and duplicate identities, including blank measurements."""
    seen = set()
    for row in rows:
        if row.get("method") not in KNOWN_METHODS:
            raise ValueError("unknown panel method")
        identity = tuple(row.get(key) for key in ("method", "dataset", "configuration", "size_multiplier", "fit_seed"))
        if sampled:
            identity += (row.get("sample_seed"),)
        if identity in seen:
            raise ValueError("duplicate method-lineage-auditor identity")
        seen.add(identity)


HASH_INPUT_SHA256 = {
    "docs/whitepaper/generated/beyond-fit.json": "a1447f908bc9c51876d345597039607c734b507bf85aa8e08db0639e80a1f036",
    "docs/whitepaper/generated/fit-trace.json": "d8c31b9e4eb59cb1a087cf5dcdde4a91a73eca00f3704fbff647a7da6ce65522",
    "docs/whitepaper/generated/loss-curves.json": "0124915497c98e560ef99c1e5ecf7ad876f9174439ff9c763cf1cf906b4aebe1",
    "docs/whitepaper/generated/original-field-map.json": "5b164d434414d3389f639d3645f40b11a999b32c56daa2cfe36d25b2205082b6",
    "docs/whitepaper/generated/provenance.json": "dea64ea61eaacac0e287298bba3ba76aa7900819aa00feb3a5d66cdcb4fb8a17",
    "docs/whitepaper/generated/replay-cost.json": "faf77e7ce1ed1d22127701bf68bbfe2f8d183f89cbd3475dd80094888c36f086",
    "docs/whitepaper/scripts/paper_receipts.py": "a2d6d45e3ff06a38ed61ab029fadfd14529978bcfc89618aa9ceb3fbe57238ee",
    "research/benchmark/expanded_validation_metrics.py": "a952062c0f83805f6442a440a5ae15293a843349d423eaa106a9c56a20d14c0f",
    "research/benchmark/methods.lock.json": "899c355cb55769710ed369fea6b082589b31858860fb74929bf4190de28aca33",
    "research/benchmark/results/arf-native-closure-watch-v1.receipt.json": "7e8eeb1ba94ca747367e98d5a016d2ed3ed4c40c5a98ca76d7a5fe27ce9c173c",
    "research/benchmark/results/beyondarena-s3-inventory.json": "504166764631dbdf5e02d116334af5cc587731fcf3efe130d62240e6dd512d64",
    "research/benchmark/results/density-matched-population-validation.json": "5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a",
    "research/benchmark/results/expanded-validation-diagnostics.json": "cd20ac0d07a1dcab7c5e46db195adb463172b5174387bb71a054b2e5f340beeb",
    "research/benchmark/results/paper-original-metadata-v1/index.json": "6d73d5f7b37849526a98a451b41a49e045ef21e911058d920940c5ea989f7fd9",
    "research/benchmark/results/s3-coreset-procedure.json": "73251e43cf7838687427569931bf6127f42e932374af33137c57fd8fe8544704",
    "research/benchmark/results/s3-data.lock.json": "857b61324d0a5b7dbbb98e0c147fcc79c5bb8616e7ca61afd6785469c17f038f",
    "research/benchmark/results/s3-lineage-record.json": "bbd0852c49f595315104261efa4eaa6f50db257e323b8c002aac6737a2dbe7d7",
    "research/benchmark/results/s3-matched-forest-confirmation-validation.json": "97a25be902954cad16c5b5802d0e0ba468dea104221efa2465434c94e6774a1f",
    "research/benchmark/results/sdv-matched-population-validation.json": "4dc367eb78159a0386882d23ae1f99b0aa2cc9f425d1aae499b6f3453222a44e"
}


def authenticate_hash_inventory():
    """Pin the transitive disclosure inputs as well as the numerical ledgers."""
    from expanded_panel import HASH_PATHS
    if set(HASH_PATHS) != set(HASH_INPUT_SHA256):
        raise ValueError("unregistered hash inventory input")
    for relative, expected in HASH_INPUT_SHA256.items():
        path = REPO / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("hash inventory input digest mismatch")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _generator(array, salt):
    """One stream per multiset. Identical retention vectors share one interval."""
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()
    key = int.from_bytes(digest[:8], "little")
    return np.random.default_rng([SEED, int(salt), key])


def summarize(values, rng=None, salt=0):
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    array = np.sort(array)
    n = int(array.size)
    if n == 0:
        return {"n": 0, "median": None, "lo": None, "hi": None}
    median = float(np.median(array))
    if n == 1:
        return {"n": 1, "median": median, "lo": median, "hi": median}
    local = _generator(array, salt)
    index = local.integers(0, n, size=(DRAWS, n))
    samples = np.median(array[index], axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"n": n, "median": median, "lo": float(lo), "hi": float(hi)}


def holm(p_values):
    count = len(p_values)
    order = sorted(range(count), key=lambda i: (p_values[i] is None, p_values[i]))
    adjusted = [None] * count
    running = 0.0
    for rank, index in enumerate(order):
        if p_values[index] is None:
            continue
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def rank_biserial(diff):
    nonzero = diff[np.isfinite(diff) & (diff != 0)]
    if nonzero.size == 0:
        return None
    ranks = stats.rankdata(np.abs(nonzero))
    plus = float(ranks[nonzero > 0].sum())
    minus = float(ranks[nonzero < 0].sum())
    total = plus + minus
    if total == 0:
        return None
    return (plus - minus) / total


def paired_test(left, right, rng, family_size):
    keys = sorted(set(left) & set(right))
    diff = np.array([left[key] - right[key] for key in keys], dtype=float)
    result = {
        "n": int(diff.size),
        "datasets": keys,
        "differences": [float(value) for value in diff],
        "median_difference": None,
        "lo": None,
        "hi": None,
        "familywise_lo": None,
        "familywise_hi": None,
        "wilcoxon_statistic": None,
        "wilcoxon_p": None,
        "rank_biserial": None,
        "sign_p": None,
        "dope_median": None,
        "dope_lo": None,
        "dope_hi": None,
        "other_median": None,
        "other_lo": None,
        "other_hi": None,
        "wins": int(np.sum(diff > 0)) if diff.size else 0,
        "ties": int(np.sum(diff == 0)) if diff.size else 0,
        "losses": int(np.sum(diff < 0)) if diff.size else 0,
    }
    if diff.size == 0:
        return result
    left_summary = summarize([left[key] for key in keys], salt=0)
    right_summary = summarize([right[key] for key in keys], salt=0)
    result["dope_median"] = left_summary["median"]
    result["dope_lo"] = left_summary["lo"]
    result["dope_hi"] = left_summary["hi"]
    result["other_median"] = right_summary["median"]
    result["other_lo"] = right_summary["lo"]
    result["other_hi"] = right_summary["hi"]
    point = summarize(diff, salt=1)
    result["median_difference"] = point["median"]
    result["lo"] = point["lo"]
    result["hi"] = point["hi"]
    if diff.size > 1 and family_size:
        ordered = np.sort(diff[np.isfinite(diff)])
        local = _generator(ordered, 2 + int(family_size))
        index = local.integers(0, ordered.size, size=(DRAWS, ordered.size))
        samples = np.median(ordered[index], axis=1)
        tail = (0.05 / family_size) / 2
        flo, fhi = np.quantile(samples, [tail, 1 - tail])
        result["familywise_lo"] = float(flo)
        result["familywise_hi"] = float(fhi)
    result["rank_biserial"] = rank_biserial(diff)
    nonzero = diff[diff != 0]
    if nonzero.size:
        positive = int(np.sum(nonzero > 0))
        result["sign_p"] = float(stats.binomtest(positive, int(nonzero.size), 0.5, alternative="greater").pvalue)
        try:
            signed = stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
            result["wilcoxon_statistic"] = float(signed.statistic)
            result["wilcoxon_p"] = float(signed.pvalue)
        except ValueError:
            result["wilcoxon_p"] = 1.0
            result["wilcoxon_statistic"] = 0.0
    else:
        result["sign_p"] = 1.0
        result["wilcoxon_p"] = 1.0
        result["rank_biserial"] = 0.0
    return result


def complete_map(rows, auditor, size, method="DOPE"):
    found = {}
    if method == "DOPE":
        identities = [row["dataset"] for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate lineage identity")
        for row in rows:
            group = row["sizes"][str(size)]["utility"][auditor]
            if group.get("complete_informative_sample_group") and finite(group.get("median_retention")):
                found[row["dataset"]] = float(group["median_retention"])
        return found
    for dataset, sizes in rows.items():
        group = sizes[str(size)]["utility"][auditor]
        if group.get("complete_informative_sample_group") and finite(group.get("median_retention")):
            found[dataset] = float(group["median_retention"])
    return found


def summary_map(summary, method, configuration, size, auditor):
    validate_rows(summary)
    found = {}
    for row in summary:
        if row.get("method") != method or row.get("configuration") != configuration:
            continue
        if row.get("size_multiplier") != size:
            continue
        group = (row.get("utility") or {}).get(auditor) or {}
        if group.get("complete_informative_sample_group") and finite(group.get("median_retention")):
            found[row["dataset"]] = float(group["median_retention"])
    return found


def attach_holm(pairs, field, target):
    names = list(pairs)
    adjusted = holm([pairs[name][field] for name in names])
    for name, value in zip(names, adjusted):
        pairs[name][target] = value


def friedman_block(maps):
    keys = set.intersection(*(set(item) for item in maps.values())) if maps else set()
    labels = [label for label in DENSITY_METHODS if label in maps]
    rows = []
    for key in sorted(keys):
        if all(key in maps[label] for label in labels):
            rows.append([maps[label][key] for label in labels])
    matrix = np.asarray(rows, dtype=float)
    report = {"n": int(matrix.shape[0]) if matrix.size else 0, "methods": labels, "statistic": None, "p": None, "cd": None, "average_ranks": {}, "separated_pairs": []}
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return report
    ranks = np.vstack([stats.rankdata(-row, method="average") for row in matrix])
    average = ranks.mean(axis=0)
    report["average_ranks"] = {label: float(value) for label, value in zip(labels, average)}
    stat, p_value = stats.friedmanchisquare(*[matrix[:, index] for index in range(matrix.shape[1])])
    report["statistic"] = float(stat)
    report["p"] = float(p_value)
    k = matrix.shape[1]
    n = matrix.shape[0]
    if k == 4:
        cd = Q_NEMENYI_K4 * math.sqrt(k * (k + 1) / (6 * n))
        report["cd"] = float(cd)
        separated = []
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                if j <= i:
                    continue
                gap = abs(float(average[i] - average[j]))
                if gap > cd:
                    separated.append({"left": left, "right": right, "rank_gap": gap})
        report["separated_pairs"] = separated
    return report


def block_from_maps(method_maps, comparators, rng):
    """method_maps: auditor -> method -> {dataset: retention}."""
    family = []
    for auditor in AUDITORS:
        for comparator in comparators:
            family.append((auditor, comparator))
    raw_pairs = {}
    for auditor, comparator in family:
        raw_pairs[(auditor, comparator)] = paired_test(
            method_maps[auditor]["DOPE"],
            method_maps[auditor][comparator],
            rng,
            family_size=len(family),
        )
    attach_holm(raw_pairs, "wilcoxon_p", "holm_p")
    attach_holm(raw_pairs, "sign_p", "sign_holm_p")
    auditors = {}
    for auditor in AUDITORS:
        methods = {name: summarize(values.values(), rng) for name, values in method_maps[auditor].items()}
        pairs = {}
        for comparator in comparators:
            item = dict(raw_pairs[(auditor, comparator)])
            item.pop("datasets", None)
            pairs[comparator] = item
        auditors[auditor] = {"methods": methods, "pairs": pairs, "friedman": None}
    if set(DENSITY_COMPARATORS).issubset(set(next(iter(method_maps.values())))):
        for auditor in AUDITORS:
            auditors[auditor]["friedman"] = friedman_block(method_maps[auditor])
    return auditors


def marginal_from_rows(record, rng):
    expected = {row["dataset"] for row in record["rows"]}
    if set(record["comparators"]) != set(DENSITY_COMPARATORS):
        raise ValueError("unknown or missing comparator")
    if any(set(rows) != expected for rows in record["comparators"].values()):
        raise ValueError("missing comparator lineage")
    method_maps = {auditor: {} for auditor in AUDITORS}
    for auditor in AUDITORS:
        method_maps[auditor]["DOPE"] = complete_map(record["rows"], auditor, 4)
        for name in DENSITY_COMPARATORS:
            method_maps[auditor][name] = complete_map(record["comparators"][name], auditor, 4, method="comparator")
    return block_from_maps(method_maps, DENSITY_COMPARATORS, rng)


def group_cells(cells, method, configuration, size):
    validate_rows(cells, sampled=True)
    grouped = defaultdict(dict)
    for cell in cells:
        if cell.get("method") != method or cell.get("configuration") != configuration:
            continue
        if cell.get("size_multiplier") != size:
            continue
        grouped[cell["dataset"]][cell["sample_seed"]] = cell
    return grouped


def threshold_sweep(cells, method, configuration, thresholds):
    report = {}
    for auditor in AUDITORS:
        report[auditor] = {}
        grouped = group_cells(cells, method, configuration, 4)
        for threshold in thresholds:
            kept = []
            for by_seed in grouped.values():
                if set(by_seed) != set(SEEDS):
                    continue
                scores = []
                usable = True
                for seed in SEEDS:
                    cell = by_seed[seed]
                    utility = (cell.get("utility") or {}).get(auditor) or {}
                    null = cell.get("null_loss")
                    real = utility.get("trtr_loss")
                    synthetic = utility.get("tstr_loss")
                    if cell.get("status") != "ok" or not all(finite(value) for value in (null, real, synthetic)):
                        usable = False
                        break
                    improvement = null - real
                    if improvement < threshold * abs(null) or improvement == 0:
                        usable = False
                        break
                    scores.append((null - synthetic) / improvement)
                if usable:
                    kept.append(float(np.median(scores)))
            report[auditor][str(threshold)] = {
                "n": len(kept),
                "median": float(np.median(kept)) if kept else None,
            }
    return report


def fidelity_summary(cells, method, configuration, rng):
    grouped = group_cells(cells, method, configuration, 4)
    fields = ("marginal_ks_mean", "pair_correlation_fidelity", "c2st_auc")
    report = {}
    for field in fields:
        values = []
        for by_seed in grouped.values():
            samples = [cell.get(field) for cell in by_seed.values() if finite(cell.get(field))]
            if len(samples) == len(SEEDS):
                values.append(float(np.median(samples)))
        report[field] = summarize(values, rng)
    return report


def byte_summary(rows, rng):
    charged = []
    misses = []
    within = 0
    known = 0
    for row in rows:
        group = row["sizes"]["4"]
        amount = group.get("charged_artifact_bytes")
        if not isinstance(amount, int):
            continue
        known += 1
        charged.append(amount)
        flag = group.get("artifact_within_l3_cap")
        if flag is True and amount <= BYTE_CAP:
            within += 1
        elif amount > BYTE_CAP or flag is False:
            misses.append({"dataset": row["dataset"], "display_name": row.get("display_name"), "charged_bytes": amount})
    return {"known": known, "within_cap": within, "median": summarize(charged, rng), "over_cap": misses}


def anchor_value(block, auditor, method):
    """Published comparison medians are paired-subset medians at size 4."""
    if method == "DOPE" and auditor in block and "methods" in block[auditor] and block is not None:
        pass
    pairs = block[auditor]["pairs"]
    if method == "DOPE":
        # Density DOPE marginal matches the paired subset. Neural DOPE does not.
        if "CTGAN" in pairs:
            return pairs["CTGAN"]["n"], pairs["CTGAN"]["dope_median"]
        return block[auditor]["methods"]["DOPE"]["n"], block[auditor]["methods"]["DOPE"]["median"]
    for comparator, pair in pairs.items():
        if comparator == method or method in comparator:
            return pair["n"], pair["other_median"]
    got = block[auditor]["methods"][method]
    return got["n"], got["median"]


def check_anchors(density, neural):
    failures = []
    for (block_name, auditor, method), (expect_n, expect_median) in ANCHORS.items():
        source = density if block_name == "density" else neural
        got_n, got_median = anchor_value(source, auditor, method)
        if expect_n is not None and got_n != expect_n:
            failures.append(f"{block_name} {auditor} {method} n {got_n} != {expect_n}")
        if got_median is None or abs(got_median - expect_median) > 5e-4:
            failures.append(f"{block_name} {auditor} {method} median {got_median} != {expect_median}")
    return failures


def paired_median_tex(value, difference_lo, difference_hi, side):
    """Bold one median when the paired difference interval excludes zero.

    ``side`` is ``dope`` or ``other``. The difference is DOPE minus the
    other method. An interval that contains zero, including the linear
    auditor against the copula, stays plain. A privacy column never
    reaches this helper.
    """
    text = sig3(value)
    if difference_lo is None or difference_hi is None:
        return text
    higher = (side == "dope" and difference_lo > 0) or (side == "other" and difference_hi < 0)
    if higher:
        return f"\\textbf{{{text}}}"
    return text


def sig3(value):
    """Three decimals on the retention scale, three significant digits below 0.1."""
    if value is None or not finite(value):
        return "---"
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.3g}"


def tex_p(value):
    if value is None or not finite(value):
        return "---"
    if value < 1e-3:
        exponent = math.floor(math.log10(value))
        mantissa = value / 10**exponent
        return f"{mantissa:.2f}\\times 10^{{{exponent}}}"
    return f"{value:.3f}"


def tex_bytes(value):
    if value is None:
        return "---"
    return f"{int(round(value)):,}".replace(",", "{,}")


def command(name, body):
    return f"\\newcommand{{\\{name}}}{{{body}}}\n"


def write_numbers(payload):
    import paper_emit

    paper_emit.emit(payload, OUT, sig3, tex_p, tex_bytes, command)


def write_density_table(density):
    labels = {
        "catboost": "CatBoost",
        "linear": "Linear",
        "mlp": "MLP",
        "GaussianCopula": "Gaussian copula",
        "Chow-Liu": "Chow--Liu",
        "independent_marginals": "Indep.\\ marginals",
    }
    rows = []
    for auditor in AUDITORS:
        for comparator in DENSITY_COMPARATORS:
            pair = density[auditor]["pairs"][comparator]
            rows.append(
                f"{labels[auditor]} & {labels[comparator]} & {pair['n']} & "
                f"{paired_median_tex(pair['dope_median'], pair['lo'], pair['hi'], 'dope')} [{sig3(pair['dope_lo'])}, {sig3(pair['dope_hi'])}] & "
                f"{paired_median_tex(pair['other_median'], pair['lo'], pair['hi'], 'other')} [{sig3(pair['other_lo'])}, {sig3(pair['other_hi'])}] & "
                f"{sig3(pair['median_difference'])} [{sig3(pair['lo'])}, {sig3(pair['hi'])}] & "
                f"{pair['wins']}/{pair['ties']}/{pair['losses']} & "
                f"${tex_p(pair['holm_p'])}$ \\\\"
            )
    body = "\n".join(rows)
    (OUT / "density-table.tex").write_text(
        "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
        "Auditor & Comparator & $n$ & DOPE & Other & Difference & W/T/L & Holm $p$ \\\\\n"
        "\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def write_headline_table(blocks):
    """One row per block and auditor: the tuned baseline with the highest paired median."""
    specs = (
        ("density", "Density", DENSITY_COMPARATORS, {
            "GaussianCopula": "Gaussian copula",
            "Chow-Liu": "Chow--Liu",
            "independent_marginals": "Indep.\\ marginals",
        }),
        ("neural", "Neural", ("CTGAN", "TVAE"), {"CTGAN": "CTGAN", "TVAE": "TVAE"}),
        ("forest", "Forest", ("Forest-Flow",), {"Forest-Flow": "Forest-Flow"}),
    )
    auditor_labels = {"catboost": "CatBoost", "linear": "Linear", "mlp": "MLP"}
    rows = []
    for block_id, label, comparators, names in specs:
        block = blocks[block_id]
        for auditor in AUDITORS:
            ranked = []
            for comparator in comparators:
                pair = block[auditor]["pairs"][comparator]
                if not pair["n"] or pair["other_median"] is None:
                    continue
                ranked.append((pair["other_median"], comparator, pair))
            if not ranked:
                continue
            ranked.sort(key=lambda item: (item[0], item[1]))
            _median, comparator, pair = ranked[-1]
            rows.append(
                f"{label} & {auditor_labels[auditor]} & {names[comparator]} & {pair['n']} & "
                f"{sig3(pair['median_difference'])} [{sig3(pair['lo'])}, {sig3(pair['hi'])}] & "
                f"{pair['wins']}/{pair['ties']}/{pair['losses']} & ${tex_p(pair['holm_p'])}$ \\\\"
            )
    (OUT / "headline-table.tex").write_text(
        "\\begin{tabular}{@{}lllrllr@{}}\n\\toprule\n"
        "Block & Auditor & Baseline & $n$ & Difference & W/T/L & Holm $p$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def tex_name(value):
    return str(value).replace("_", r"\_").replace("&", r"\&")


def write_neural_table(neural):
    labels = {"catboost": "CatBoost", "linear": "Linear", "mlp": "MLP", "CTGAN": "CTGAN", "TVAE": "TVAE"}
    rows = []
    for auditor in AUDITORS:
        for comparator in ("CTGAN", "TVAE"):
            pair = neural[auditor]["pairs"][comparator]
            rows.append(
                f"{labels[auditor]} & {labels[comparator]} & {pair['n']} & "
                f"{paired_median_tex(pair['dope_median'], pair['lo'], pair['hi'], 'dope')} [{sig3(pair['dope_lo'])}, {sig3(pair['dope_hi'])}] & "
                f"{paired_median_tex(pair['other_median'], pair['lo'], pair['hi'], 'other')} [{sig3(pair['other_lo'])}, {sig3(pair['other_hi'])}] & "
                f"{sig3(pair['median_difference'])} [{sig3(pair['lo'])}, {sig3(pair['hi'])}] & "
                f"{pair['wins']}/{pair['ties']}/{pair['losses']} & "
                f"${tex_p(pair['holm_p'])}$ \\\\"
            )
    (OUT / "neural-table.tex").write_text(
        "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
        "Auditor & Comparator & $n$ & DOPE & Other & Difference & W/T/L & Holm $p$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def write_forest_table(forest_doc):
    names = {item["id"]: item["display_name"] for item in forest_doc["dataset_metadata"]}
    packed = defaultdict(dict)
    for row in forest_doc["summary"]:
        if row.get("size_multiplier") != 4:
            continue
        if row.get("method") == "DOPE" and row.get("configuration") == "features12_steps2048":
            slot = "dope"
        elif row.get("method") == "ForestDiffusion/Forest-Flow" and row.get("configuration") == "native_selected":
            slot = "forest"
        else:
            continue
        packed[row["dataset"]][slot] = row
    lines = []
    for dataset in sorted(packed, key=lambda item: names.get(item, item)):
        dope = packed[dataset].get("dope") or {}
        forest = packed[dataset].get("forest") or {}
        dope_u = (dope.get("utility") or {}).get("catboost") or {}
        forest_u = (forest.get("utility") or {}).get("catboost") or {}
        lines.append(
            f"{tex_name(names.get(dataset, dataset))} & "
            f"{tex_bytes(dope.get('charged_artifact_bytes'))} & {sig3(dope_u.get('median_retention'))} & "
            f"{tex_bytes(forest.get('charged_artifact_bytes'))} & {sig3(forest_u.get('median_retention'))} \\\\"
        )
    (OUT / "forest-table.tex").write_text(
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
        "Lineage & DOPE bytes & DOPE & Forest-Flow bytes & Forest-Flow \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def _byte_pack(amounts, within):
    if not amounts:
        return {"n": 0, "within": 0, "median": None, "lo": None, "hi": None}
    return {
        "n": len(amounts),
        "within": within,
        "median": float(np.median(amounts)),
        "lo": int(min(amounts)),
        "hi": int(max(amounts)),
    }


def _comparator_bytes(block):
    amounts = []
    within = 0
    for sizes in block.values():
        group = sizes["4"]
        amount = group.get("charged_artifact_bytes")
        if not isinstance(amount, int):
            continue
        amounts.append(amount)
        if group.get("artifact_within_l3_cap") is True and amount <= BYTE_CAP:
            within += 1
    return _byte_pack(amounts, within)


def _neural_bytes(summary, method):
    amounts = []
    within = 0
    for row in summary:
        if row.get("method") != method or row.get("configuration") != "native_selected":
            continue
        if row.get("size_multiplier") != 4:
            continue
        if (row.get("statuses") or {}).get("ok") != 3:
            continue
        amount = row.get("charged_artifact_bytes")
        if not isinstance(amount, int):
            continue
        amounts.append(amount)
        if row.get("artifact_within_l3_cap") is True and amount <= BYTE_CAP:
            within += 1
    return _byte_pack(amounts, within)


def _paired_sufficiency(record, auditor):
    def grab(rows):
        found = {}
        for row in rows:
            group = row["sizes"]["4"]["utility"][auditor]
            if group.get("complete_informative_sample_group") and finite(group.get("median_retention")):
                found[row["dataset"]] = float(group["median_retention"])
        return found

    left = grab(record["rows"])
    right = grab(record["sufficiency_rows"])
    keys = sorted(set(left) & set(right))
    if not keys:
        return {"n": 0, "median_2048": None, "median_8192": None, "median_difference": None}
    base = np.array([left[key] for key in keys], dtype=float)
    extra = np.array([right[key] for key in keys], dtype=float)
    return {
        "n": len(keys),
        "median_2048": float(np.median(base)),
        "median_8192": float(np.median(extra)),
        "median_difference": float(np.median(extra - base)),
    }


def _read_json(path):
    if path.parent != RESULTS:
        raise ValueError("unregistered panel input path")
    return load(path.name)


def _public_forest_cost(cost):
    """Keep numeric cost fields. Machine names are not a paper statistic."""
    hidden = {"forest_gpu_host", "shared_evaluator_host"}
    return {key: value for key, value in cost.items() if key not in hidden}


def collect_sidecars(record, neural_doc, forest_doc, forest_bytes):
    from paper_receipts import checked_extracts
    original_extracts, original_loss = checked_extracts()
    misses = []
    not_informative = []
    dope_amounts = []
    for row in record["rows"]:
        group = row["sizes"]["4"]
        amount = group.get("charged_artifact_bytes")
        utility = group["utility"]["catboost"]
        if isinstance(amount, int):
            dope_amounts.append(amount)
            if amount > BYTE_CAP:
                misses.append({"name": row["display_name"], "bytes": amount})
        if utility.get("blank_reason") == "not_informative":
            not_informative.append(row["display_name"])
    misses.sort(key=lambda item: -item["bytes"])
    dope_forest = []
    for row in forest_doc["summary"]:
        if row.get("size_multiplier") != 4 or row.get("configuration") != "features12_steps2048":
            continue
        if row.get("method") != "DOPE" or not isinstance(row.get("charged_artifact_bytes"), int):
            continue
        dope_forest.append(row["charged_artifact_bytes"])
    lock = _read_json(RESULTS / "s3-data.lock.json") or {}
    excluded = [entry["id"] for entry in lock.get("entries", []) if not entry.get("split")]
    split = next((entry.get("split") or {} for entry in lock.get("entries", []) if entry.get("split")), {})
    return {
        "bytes": {
            "GaussianCopula": _comparator_bytes(record["comparators"]["GaussianCopula"]),
            "Chow-Liu": _comparator_bytes(record["comparators"]["Chow-Liu"]),
            "independent_marginals": _comparator_bytes(record["comparators"]["independent_marginals"]),
            "CTGAN": _neural_bytes(neural_doc["summary"], "CTGAN"),
            "TVAE": _neural_bytes(neural_doc["summary"], "TVAE"),
        },
        "over_cap": misses,
        "dope_byte_min": min(dope_amounts) if dope_amounts else None,
        "not_informative": not_informative,
        "sufficiency": {auditor: _paired_sufficiency(record, auditor) for auditor in AUDITORS},
        "forest_cost": _public_forest_cost(forest_doc.get("cost") or {}),
        "dope_forest_byte_median": float(np.median(dope_forest)) if dope_forest else None,
        "forest_byte_median": float(np.median(forest_bytes)) if forest_bytes else None,
        "lock": {
            "catalog_sha256": lock.get("catalog_sha256"),
            "eligible": lock.get("eligible_entries"),
            "prepared": lock.get("prepared"),
            "excluded": lock.get("excluded_after_checks"),
            "unknown_rights": lock.get("unknown_rights_excluded"),
            "final_evaluation_authorized": lock.get("final_evaluation_authorized"),
            "overlap_ids": excluded,
            "split_kind": split.get("kind"),
            "split_seed": split.get("seed"),
        },
        "replay": original_extracts["replay-cost.json"],
        "beyond": original_extracts["beyond-fit.json"],
        "provenance_counts": original_extracts["provenance-counts.json"],
        "loss": original_loss,
    }


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    rng = np.random.default_rng(SEED)
    record = load("s3-lineage-record.json")
    density_cells = load("density-matched-population-validation.json")
    neural_doc = load("sdv-matched-population-validation.json")
    forest_doc = load("s3-matched-forest-confirmation-validation.json")
    for doc in (density_cells, neural_doc, forest_doc):
        validate_rows(doc["cells"], sampled=True)
        validate_rows(doc["summary"])
    density = marginal_from_rows(record, rng)
    neural_maps = {auditor: {} for auditor in AUDITORS}
    for auditor in AUDITORS:
        neural_maps[auditor]["DOPE"] = summary_map(neural_doc["summary"], "DOPE", "features12_steps2048", 4, auditor)
        neural_maps[auditor]["CTGAN"] = summary_map(neural_doc["summary"], "CTGAN", "native_selected", 4, auditor)
        neural_maps[auditor]["TVAE"] = summary_map(neural_doc["summary"], "TVAE", "native_selected", 4, auditor)
    neural = block_from_maps(neural_maps, ("CTGAN", "TVAE"), rng)
    forest_maps = {auditor: {} for auditor in AUDITORS}
    for auditor in AUDITORS:
        forest_maps[auditor]["DOPE"] = summary_map(
            forest_doc["summary"], "DOPE", "features12_steps2048", 4, auditor
        )
        forest_maps[auditor]["Forest-Flow"] = summary_map(
            forest_doc["summary"], "ForestDiffusion/Forest-Flow", "native_selected", 4, auditor
        )
    forest = block_from_maps(forest_maps, ("Forest-Flow",), rng)
    thresholds = (0.0, 0.001, 0.01, 0.05, 0.1)
    configs = {
        "DOPE": "features12_steps2048",
        "GaussianCopula": "native_selected",
        "Chow-Liu": "native_selected",
        "independent_marginals": "native_selected",
    }
    robustness = {
        method: threshold_sweep(density_cells["cells"], method, config, thresholds)
        for method, config in configs.items()
    }
    fidelity = {
        method: fidelity_summary(density_cells["cells"], method, config, rng)
        for method, config in configs.items()
    }
    failures = check_anchors(density, neural)
    primary = robustness["DOPE"]["catboost"]["0.01"]["median"]
    published = density["catboost"]["methods"]["DOPE"]["median"]
    if primary is None or published is None or abs(primary - published) > 5e-4:
        failures.append(f"threshold 0.01 median {primary} disagrees with lineage record {published}")
    forest_bytes = [
        row.get("charged_artifact_bytes")
        for row in forest_doc["summary"]
        if row.get("method") == "ForestDiffusion/Forest-Flow"
        and row.get("configuration") == "native_selected"
        and row.get("size_multiplier") == 4
        and isinstance(row.get("charged_artifact_bytes"), int)
    ]
    payload = {
        "format": "dope-paper-panel-stats",
        "version": 1,
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        "official_tests_opened": {
            "s3-lineage-record": record.get("official_tests_opened"),
            "density": density_cells.get("official_tests_opened"),
            "neural": neural_doc.get("official_tests_opened"),
            "forest": forest_doc.get("official_tests_opened"),
        },
        "claims": {"ptf_v1": None, "mfs_v2": None, "release_safe_l3": None, "paired_superiority_claim": False},
        "blocks": {"density": density, "neural": neural, "forest": forest},
        "threshold_robustness": robustness,
        "fidelity": fidelity,
        "bytes": {"DOPE": byte_summary(record["rows"], rng)},
        "forest_bytes": {
            "min": min(forest_bytes) if forest_bytes else None,
            "max": max(forest_bytes) if forest_bytes else None,
            "n": len(forest_bytes),
            "within": sum(1 for amount in forest_bytes if amount <= BYTE_CAP),
        },
        "anchor_failures": failures,
        "notes": {
            "interval": "percentile lineage bootstrap of the median; three sample seeds stay inside each lineage median",
            "holm_family_density": "9 tests, 3 comparators times 3 auditors, size 4, density block only",
            "holm_family_neural": "CTGAN and TVAE times auditors, size 4, not pooled with density",
            "holm_family_forest": "Forest-Flow native_selected, size 4, six-lineage confirmation, not pooled",
            "friedman": "Nemenyi CD uses Demsar q_alpha 2.569 for four methods at alpha 0.05",
            "superiority": "Retention tests do not authorize PTF-v1, MFS-v2, or release-safe L3",
            "wins": "lineage counts of positive, zero, and negative paired differences, DOPE minus the baseline",
            "strongest_baseline": "highest paired median retention inside the block and auditor; blocks are not pooled",
            "holm_family_fidelity_density": "12 tests, 4 metrics times 3 density comparators, grouped validation, not pooled with retention",
            "holm_family_fidelity_neural": "8 tests, 4 metrics times CTGAN and TVAE on the neural lineages only, not pooled",
            "fidelity_privacy": "DCR share and DOMIAS AUC are empirical. A larger value sits nearer the fit view. Not a utility win, not DP, and not HIPAA.",
        },
    }
    from expanded_panel import build_expanded

    diagnostics = load("expanded-validation-diagnostics.json")
    procedure = load("s3-coreset-procedure.json")
    validate_rows(diagnostics["cells"], sampled=True)
    authenticate_hash_inventory()
    expanded = build_expanded(REPO, record, density_cells, diagnostics, procedure, density)
    failures.extend(expanded["failures"])
    payload["expanded"] = {key: value for key, value in expanded.items() if key != "failures"}
    payload["sidecars"] = collect_sidecars(record, neural_doc, forest_doc, forest_bytes)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "panel-stats.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_numbers(payload)
    write_density_table(density)
    write_neural_table(neural)
    write_forest_table(forest_doc)
    write_headline_table(payload["blocks"])
    print(json.dumps({
        "anchors_ok": not failures,
        "failures": failures,
        "dope_catboost": density["catboost"]["methods"]["DOPE"],
        "holm_gauss": density["catboost"]["pairs"]["GaussianCopula"]["holm_p"],
        "friedman": density["catboost"]["friedman"],
        "forest_n": forest["catboost"]["pairs"]["Forest-Flow"]["n"],
        "forest_bytes": payload["forest_bytes"],
        "dope_within": payload["bytes"]["DOPE"]["within_cap"],
    }, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
