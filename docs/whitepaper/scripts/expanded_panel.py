"""Fidelity, raw auditor loss, coreset bounds, and ledger hashes for the paper.

Reads committed ledgers only. Does not open an official test file and does not
refit a generator. Empirical distance and density-ratio figures stay empirical.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

METRICS = ("alpha_precision", "beta_recall", "share_closer_to_train", "domias_auc")
DENSITY_COMPARATORS = ("GaussianCopula", "Chow-Liu", "independent_marginals")
NEURAL_COMPARATORS = ("CTGAN", "TVAE")
DOPE_CONFIG = "features12_steps2048"
SEEDS = (101, 211, 307)
PRIMARY_THRESHOLD = 0.01

HASH_PATHS = (
    "research/benchmark/expanded_validation_metrics.py",
    "research/benchmark/methods.lock.json",
    "research/benchmark/results/expanded-validation-diagnostics.json",
    "research/benchmark/results/s3-coreset-procedure.json",
    "research/benchmark/results/s3-lineage-record.json",
    "research/benchmark/results/density-matched-population-validation.json",
    "research/benchmark/results/sdv-matched-population-validation.json",
    "research/benchmark/results/s3-matched-forest-confirmation-validation.json",
    "research/benchmark/results/s3-data.lock.json",
    "research/benchmark/results/arf-native-closure-watch-v1.receipt.json",
    "research/benchmark/results/beyondarena-s3-inventory.json",
    "docs/whitepaper/generated/fit-trace.json",
    "docs/whitepaper/generated/loss-curves.json",
    "docs/whitepaper/generated/replay-cost.json",
    "docs/whitepaper/generated/provenance.json",
    "docs/whitepaper/generated/beyond-fit.json",
)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_value(cell, metric):
    if metric == "domias_auc":
        if cell.get("domias_status") != "ok":
            return None
        return cell.get("domias_auc")
    if metric in ("alpha_precision", "beta_recall"):
        if cell.get("alpha_beta_status") != "ok":
            return None
        return cell.get(metric)
    return cell.get(metric)


def _lineage_map(cells, method, configuration, metric):
    grouped = defaultdict(dict)
    for cell in cells:
        if cell.get("method") != method or cell.get("configuration") != configuration:
            continue
        if cell.get("status") != "ok" or cell.get("size_multiplier") != 4:
            continue
        grouped[cell["dataset"]][cell["sample_seed"]] = cell
    found = {}
    for dataset, by_seed in grouped.items():
        if set(by_seed) != set(SEEDS):
            continue
        values = []
        for seed in SEEDS:
            value = _metric_value(by_seed[seed], metric)
            if not _finite(value):
                values = None
                break
            values.append(float(value))
        if values is not None:
            found[dataset] = float(sorted(values)[len(values) // 2])
    return found


def _pair_block(panel, cells, dope_config, comparators, configurations):
    methods = {"DOPE": {}}
    for metric in METRICS:
        methods["DOPE"][metric] = panel.summarize(_lineage_map(cells, "DOPE", dope_config, metric).values())
    pairs = {}
    order = []
    for comparator in comparators:
        pairs[comparator] = {}
        configuration = configurations[comparator]
        for metric in METRICS:
            left = _lineage_map(cells, "DOPE", dope_config, metric)
            right = _lineage_map(cells, comparator, configuration, metric)
            methods.setdefault(comparator, {})[metric] = panel.summarize(right.values())
            result = panel.paired_test(left, right, None, len(comparators) * len(METRICS))
            result.pop("datasets", None)
            result.pop("differences", None)
            pairs[comparator][metric] = result
            order.append((comparator, metric))
    adjusted = panel.holm([pairs[comp][metric]["wilcoxon_p"] for comp, metric in order])
    for (comp, metric), value in zip(order, adjusted):
        pairs[comp][metric]["holm_p"] = value
    return {"methods": methods, "pairs": pairs, "holm_family": len(order)}


def _informative_loss(cells, method, configuration, auditor, threshold):
    grouped = defaultdict(dict)
    for cell in cells:
        if cell.get("method") != method or cell.get("configuration") != configuration:
            continue
        if cell.get("size_multiplier") != 4:
            continue
        grouped[cell["dataset"]][cell["sample_seed"]] = cell
    retention, real_rmse, synthetic_rmse, real_r2, synthetic_r2 = [], [], [], [], []
    for by_seed in grouped.values():
        if set(by_seed) != set(SEEDS):
            continue
        scores = {"retention": [], "real_rmse": [], "synthetic_rmse": [], "real_r2": [], "synthetic_r2": []}
        usable = True
        for seed in SEEDS:
            cell = by_seed[seed]
            utility = (cell.get("utility") or {}).get(auditor) or {}
            null = cell.get("null_loss")
            real = utility.get("trtr_loss")
            synthetic = utility.get("tstr_loss")
            if cell.get("status") != "ok" or not all(_finite(value) for value in (null, real, synthetic)):
                usable = False
                break
            if min(null, real, synthetic) < 0 or null == 0:
                usable = False
                break
            improvement = null - real
            if improvement < threshold * abs(null) or improvement == 0:
                usable = False
                break
            scores["retention"].append((null - synthetic) / improvement)
            scores["real_rmse"].append(math.sqrt(real))
            scores["synthetic_rmse"].append(math.sqrt(synthetic))
            scores["real_r2"].append(1.0 - real / null)
            scores["synthetic_r2"].append(1.0 - synthetic / null)
        if usable:
            retention.append(float(sorted(scores["retention"])[1]))
            real_rmse.append(float(sorted(scores["real_rmse"])[1]))
            synthetic_rmse.append(float(sorted(scores["synthetic_rmse"])[1]))
            real_r2.append(float(sorted(scores["real_r2"])[1]))
            synthetic_r2.append(float(sorted(scores["synthetic_r2"])[1]))
    return {
        "n": len(retention),
        "retention": retention,
        "real_rmse": real_rmse,
        "synthetic_rmse": synthetic_rmse,
        "real_r2": real_r2,
        "synthetic_r2": synthetic_r2,
    }


def _coreset(record, procedure):
    fit_rows = [row["fit_rows"] for row in record["rows"] if isinstance(row.get("fit_rows"), int)]
    by_id = {row["dataset"]: row for row in procedure["rows"]}
    if set(by_id) != {row["dataset"] for row in record["rows"]}:
        raise ValueError("coreset procedure lineages differ from the lineage record")
    bins = {}
    not_downsampled = 0
    for row in procedure["rows"]:
        if not row["downsampled"]:
            not_downsampled += 1
            continue
        if row["downsampling_method"] != "hybrid_coreset":
            raise ValueError("unexpected coreset method")
        if row["minmax_features_preserved"] is not True or row["minmax_target_preserved"] is not True:
            raise ValueError("hybrid coreset did not preserve feature and target extrema")
        bins[row["bins"]] = bins.get(row["bins"], 0) + 1
    return {
        "lineages": len(fit_rows),
        "fit_rows_min": min(fit_rows),
        "fit_rows_max": max(fit_rows),
        "not_downsampled": not_downsampled,
        "bins": {str(key): bins[key] for key in sorted(bins)},
        "sampling_method": procedure["sampling_method"],
        "final_max_rows": procedure["final_max_rows"],
        "minimum_final_rows": procedure["minimum_final_rows"],
        "minimum_test_to_train_ratio_when_small": procedure["minimum_test_to_train_ratio_when_small"],
        "train_target_rows": procedure["train_target_rows"],
        "test_target_rows": procedure["test_target_rows"],
        "split_kind": "official_test_grouped_training_80_20",
        "split_seed": 1729,
    }


def _hashes(repo: Path):
    rows = []
    for relative in HASH_PATHS:
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"paper input missing: {relative}")
        rows.append({"path": relative, "sha256": _sha256(path)})
    return rows


def build_expanded(repo, record, density_cells, ledger, procedure, density_block):
    import compute_panel as panel

    failures = []
    if ledger.get("official_tests_opened") is not False or procedure.get("official_tests_opened") is not False:
        failures.append("expanded ledger or coreset procedure opened official tests")
    if ledger.get("formal_dp") is not False or ledger.get("hipaa_deidentification") is not False:
        failures.append("expanded ledger claims formal DP or HIPAA de-identification")
    if (
        ledger.get("mfs_v2") is not None
        or ledger.get("ptf_v1") is not None
        or ledger.get("superiority") is not None
        or ledger.get("release_safe_l3") is not None
    ):
        failures.append("expanded ledger carries a certification claim")
    if ledger.get("paired_superiority_claim") is not False:
        failures.append("expanded ledger claims paired superiority")
    harness = repo / ledger["harness"]
    if _sha256(harness) != ledger["harness_sha256"]:
        failures.append("harness bytes differ from the measurement ledger")
    cells = ledger["cells"]
    if any(cell.get("status") != "ok" for cell in cells):
        failures.append("expanded ledger contains a cell that is not ok")
    configurations = {
        "DOPE": DOPE_CONFIG,
        "GaussianCopula": "native_selected",
        "Chow-Liu": "native_selected",
        "independent_marginals": "native_selected",
        "CTGAN": "native_selected",
        "TVAE": "native_selected",
    }
    density = _pair_block(panel, cells, DOPE_CONFIG, DENSITY_COMPARATORS, configurations)
    neural_ids = {cell["dataset"] for cell in cells if cell.get("block") == "neural"}
    neural_cells = [cell for cell in cells if cell["dataset"] in neural_ids]
    neural = _pair_block(panel, neural_cells, DOPE_CONFIG, NEURAL_COMPARATORS, configurations)
    loss = {}
    for auditor in panel.AUDITORS:
        raw = _informative_loss(density_cells["cells"], "DOPE", DOPE_CONFIG, auditor, PRIMARY_THRESHOLD)
        published = density_block[auditor]["methods"]["DOPE"]
        if raw["n"] != published["n"]:
            failures.append(f"{auditor} informative n {raw['n']} != published {published['n']}")
        retention_summary = panel.summarize(raw["retention"])
        if retention_summary["median"] is None or abs(retention_summary["median"] - published["median"]) > 5e-4:
            failures.append(
                f"{auditor} recomputed retention {retention_summary['median']} != {published['median']}"
            )
        loss[auditor] = {
            "n": raw["n"],
            "real_rmse": panel.summarize(raw["real_rmse"]),
            "synthetic_rmse": panel.summarize(raw["synthetic_rmse"]),
            "real_r2": panel.summarize(raw["real_r2"]),
            "synthetic_r2": panel.summarize(raw["synthetic_r2"]),
        }
    counts = {}
    for threshold in (0.001, 0.01, 0.05):
        counts[str(threshold)] = {
            auditor: _informative_loss(density_cells["cells"], "DOPE", DOPE_CONFIG, auditor, threshold)["n"]
            for auditor in panel.AUDITORS
        }
    if counts["0.01"]["catboost"] != 97 or counts["0.01"]["linear"] != 92 or counts["0.01"]["mlp"] != 75:
        failures.append(f"primary informative counts are {counts['0.01']}, expected 97/92/75")
    return {
        "source_commit": ledger["source_commit"],
        "harness_sha256": ledger["harness_sha256"],
        "arf_measured": ledger["arf"]["measured"],
        "arf_reason": ledger["arf"]["reason"],
        "empirical_only": True,
        "formal_dp": False,
        "hipaa_deidentification": False,
        "blocks": {"density": density, "neural": neural},
        "loss": loss,
        "threshold_counts": counts,
        "coreset": _coreset(record, procedure),
        "hashes": _hashes(repo),
        "failures": failures,
    }
