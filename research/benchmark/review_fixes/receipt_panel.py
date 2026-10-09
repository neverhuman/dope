"""Build the review-fix tables from the committed validation ledgers.

Reads published records only. Does not open an official test file, does not fit
a generator, and does not treat a result as a release score.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.benchmark.review_fixes.stats import (
    MARGIN,
    cap_retention,
    cluster_of,
    delta_method_se,
    family_of,
    holm,
    median_ci,
    tost_mean,
    wilcoxon_p,
    win_tie_loss,
)

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "benchmark" / "results"
GENERATED = REPO / "docs" / "whitepaper" / "generated"
FIGURES = REPO / "docs" / "whitepaper" / "figures"
WORKERS = Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker")
AUDITORS = ("catboost", "linear", "mlp")
SEEDS = (101, 211, 307)
DENSITY = (
    ("DOPE", "features12_steps2048"),
    ("GaussianCopula", "native_selected"),
    ("Chow-Liu", "native_selected"),
    ("independent_marginals", "native_selected"),
)
NEURAL = (("CTGAN", "native_selected"), ("TVAE", "native_selected"))
ARF = (("ARF", "author_default"), ("ARF", "native_selected"))
FOREST = (("ForestDiffusion/Forest-Flow", "native_selected"),)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()), _sha(path)


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def reduce_cells(cells, method: str, configuration: str, fit_seed: int = 11):
    grouped: dict[tuple, dict] = {}
    for cell in cells:
        if cell.get("method") != method or cell.get("configuration") != configuration:
            continue
        if cell.get("fit_seed") not in (None, fit_seed):
            continue
        size = cell.get("size_multiplier")
        seed = cell.get("sample_seed")
        dataset = cell.get("dataset")
        if size not in (1, 4) or seed not in SEEDS or not isinstance(dataset, str):
            continue
        slot = grouped.setdefault((dataset, size), {})
        if seed in slot:
            raise ValueError(f"duplicate cell {method} {configuration} {dataset} {size} {seed}")
        slot[seed] = cell
    return {key: _collapse(seeds) for key, seeds in grouped.items()}


def _series(seeds, field: str):
    if set(seeds) != set(SEEDS):
        return []
    values = []
    for seed in SEEDS:
        cell = seeds[seed]
        value = cell.get(field)
        if cell.get("status") != "ok" or not _finite(value):
            return []
        values.append(float(value))
    return values


def _collapse(seeds: dict) -> dict:
    complete = set(seeds) == set(SEEDS)
    auditors = {}
    for name in AUDITORS:
        retentions, tstr, trtr, informative = [], [], [], []
        ok = complete
        for seed in SEEDS:
            cell = seeds.get(seed) or {}
            utility = ((cell.get("utility") or {}).get(name) or {})
            if cell.get("status") != "ok":
                ok = False
            retentions.append(utility.get("retention"))
            tstr.append(utility.get("tstr_loss"))
            trtr.append(utility.get("trtr_loss"))
            informative.append(utility.get("informative") is True)
        losses_ok = ok and all(_finite(value) for value in tstr + trtr)
        primary = ok and all(informative) and all(_finite(value) for value in retentions)
        auditors[name] = {
            "median_retention": float(np.median(retentions)) if primary else None,
            "tstr": [float(value) for value in tstr] if losses_ok else [],
            "trtr": [float(value) for value in trtr] if losses_ok else [],
            "informative": primary,
            "losses_ok": losses_ok,
        }
    return {
        "auditors": auditors,
        "nulls": _series(seeds, "null_loss"),
        "c2st": _series(seeds, "c2st_auc"),
        "ks": _series(seeds, "marginal_ks_mean"),
        "pair": _series(seeds, "pair_correlation_fidelity"),
        "complete": complete,
    }


def record_map(record, method: str, size: int, auditor: str) -> dict:
    found = {}
    if method == "DOPE":
        rows = record["rows"]
        iterator = ((row["dataset"], row["sizes"]) for row in rows)
    else:
        iterator = record["comparators"][method].items()
    for dataset, sizes in iterator:
        group = sizes[str(size)]["utility"][auditor]
        if group.get("complete_informative_sample_group") and _finite(group.get("median_retention")):
            found[dataset] = float(group["median_retention"])
    return found


def primary_map(reduced) -> dict:
    found = {size: {auditor: {} for auditor in AUDITORS} for size in (1, 4)}
    for (dataset, size), row in reduced.items():
        for auditor, block in row["auditors"].items():
            if block["informative"] and block["median_retention"] is not None:
                found[size][auditor][dataset] = float(block["median_retention"])
    return found


def assert_matches(reduced_map, record_values, label: str) -> None:
    shared = set(reduced_map) & set(record_values)
    for dataset in shared:
        if abs(reduced_map[dataset] - record_values[dataset]) > 1e-9:
            raise ValueError(f"parser mismatch {label} {dataset}")
    if not shared:
        raise ValueError(f"parser found no overlap for {label}")


def names_of(record) -> dict:
    return {row["dataset"]: row["display_name"] for row in record["rows"]}


def load_tasks(datasets) -> dict:
    """Task and positive target span. The span endpoints are not returned."""
    tasks = {}
    for dataset in datasets:
        projection = json.loads((WORKERS / dataset / "projection.json").read_text())
        task = projection["task"]
        span = None
        if task == "regression":
            width = float(projection["target_map"]["max"]) - float(projection["target_map"]["min"])
            if math.isfinite(width) and width >= 0:
                span = width
        tasks[dataset] = {"task": task, "span": span}
    return tasks


def lineage_error(losses, task_info, kind: str):
    if not losses or task_info is None:
        return None
    center = float(np.median(losses))
    if task_info["task"] == "binary":
        return {"kind": "log_loss", "value": center}
    if task_info["span"] is None or center < 0:
        return None
    scale = 1.0 if kind == "projected" else task_info["span"]
    return {"kind": "rmse", "value": math.sqrt(center) * scale}


def sensitivity(row, auditor: str):
    block = row["auditors"][auditor]
    if not block["losses_ok"] or len(row["nulls"]) != 3:
        return {"status": "incomplete"}
    null = float(np.median(row["nulls"]))
    trtr = float(np.median(block["trtr"]))
    tstr = float(np.median(block["tstr"]))
    gap = null - trtr
    if gap == 0 or not math.isfinite(gap):
        return {"status": "undefined"}
    raw = (null - tstr) / gap
    capped, was_capped = cap_retention(raw)
    return {"status": "capped" if was_capped else "raw", "value": capped, "raw": raw}


def paired(left: dict, right: dict, names: dict, label: str) -> dict:
    keys = sorted(set(left) & set(right))
    diff = [left[key] - right[key] for key in keys]
    clusters = [cluster_of(names.get(key, key), key) for key in keys]
    summary = median_ci(diff, label, clusters)
    summary.update(win_tie_loss(diff))
    summary["wilcoxon_p"] = wilcoxon_p(diff)
    summary["tost"] = tost_mean(diff)
    summary["mean"] = summary["tost"]["mean"]
    return summary


def marginal(values: dict, names: dict, label: str) -> dict:
    keys = sorted(values)
    clusters = [cluster_of(names.get(key, key), key) for key in keys]
    return median_ci([values[key] for key in keys], label, clusters)


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "---"
    return f"{float(value):.{digits}f}"


def ci(summary) -> str:
    if summary.get("lo") is None:
        return "---"
    return f"[{fmt(summary['lo'])}, {fmt(summary['hi'])}]"


def tex_name(value: str) -> str:
    return str(value).replace("_", r"\_").replace("&", r"\&")


def write_tex(name: str, body: str) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / name).write_text(body)


def _row_retention(method, configuration, auditor, size, summary) -> str:
    return (
        f"{tex_name(method)} & {tex_name(configuration)} & {tex_name(auditor)} & {size}$n$ & "
        f"{summary['n']} & {fmt(summary['median'])} & {ci(summary)} \\\\"
    )


def fivefit_rows(panel_path: Path) -> list:
    if not panel_path.exists():
        return []
    document = json.loads(panel_path.read_text())
    rows = []
    for row in document.get("summary") or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "method": row.get("method"),
            "fit_seed": row.get("fit_seed"),
            "datasets": row.get("datasets"),
            "row_multiplier": row.get("row_multiplier"),
            "selection_binding": row.get("selection_binding"),
            "catboost_retention": row.get("catboost_retention"),
            "linear_retention": row.get("linear_retention"),
            "mlp_retention": row.get("mlp_retention"),
            "c2st_catboost_auc": row.get("c2st_catboost_auc"),
            "distance_mia_auc": row.get("distance_mia_auc"),
            "dcr_validation_median": row.get("dcr_validation_median"),
            "dcr_fit_median": row.get("dcr_fit_median"),
            "nndr_fit_median": row.get("nndr_fit_median"),
        })
    return rows


def stored_fidelity_rows(reductions, names) -> list:
    """Stored logistic C2ST, marginal KS, and pair correlation.

    Each row counts lineages with three finite stored values. The cohorts are
    not matched across methods, and the C2ST column is not the GBDT detector.
    """
    rows = []
    sources = list(DENSITY) + list(NEURAL) + list(ARF) + list(FOREST)
    for method, configuration in sources:
        reduced = reductions[(method, configuration)]
        for size in (1, 4):
            for field, key in (("c2st", "c2st"), ("ks", "ks"), ("pair", "pair")):
                values = {}
                for (dataset, row_size), row in reduced.items():
                    if row_size == size and len(row[field]) == 3:
                        values[dataset] = float(np.median(row[field]))
                summary = marginal(values, names, f"fidelity|{method}|{configuration}|{size}|{field}")
                summary.update({
                    "method": method,
                    "configuration": configuration,
                    "size": size,
                    "metric": key,
                    "c2st_kind": "stored_logistic" if key == "c2st" else None,
                    "coverage": "unmatched; lineages with three finite stored values",
                })
                rows.append(summary)
    return rows


def build(stop_after_anchor: bool = False) -> dict:
    predeclare_path = REPO / "research" / "benchmark" / "review_fixes" / "predeclare.json"
    predeclare = json.loads(predeclare_path.read_text())
    if predeclare["tost"]["margin_retention"] != MARGIN:
        raise ValueError("TOST margin in code and predeclaration differ")
    record, record_sha = _load("s3-lineage-record.json")
    density, density_sha = _load("density-matched-population-validation.json")
    sdv, sdv_sha = _load("sdv-matched-population-validation.json")
    arf, arf_sha = _load("arf-matched-population-validation.json")
    forest, forest_sha = _load("s3-matched-forest-confirmation-validation.json")
    names = names_of(record)
    reductions = {}
    for method, configuration in DENSITY:
        reductions[(method, configuration)] = reduce_cells(density["cells"], method, configuration)
    for method, configuration in NEURAL:
        reductions[(method, configuration)] = reduce_cells(sdv["cells"], method, configuration)
    reductions[("DOPE", "features12_steps2048_sdv")] = reduce_cells(
        sdv["cells"], "DOPE", "features12_steps2048")
    for method, configuration in ARF:
        reductions[(method, configuration)] = reduce_cells(arf["cells"], method, configuration)
    for method, configuration in FOREST:
        reductions[(method, configuration)] = reduce_cells(forest["cells"], method, configuration)
    del density, sdv, arf, forest

    maps = {key: primary_map(value) for key, value in reductions.items()}
    for method, configuration in DENSITY:
        for size in (1, 4):
            for auditor in AUDITORS:
                assert_matches(
                    maps[(method, configuration)][size][auditor],
                    record_map(record, method, size, auditor),
                    f"{method} {configuration} {size} {auditor}",
                )
    anchor = maps[("DOPE", "features12_steps2048")][4]["catboost"]
    anchor_values = [anchor[key] for key in sorted(anchor)]
    anchor_median = float(np.median(anchor_values)) if anchor_values else None
    expected = predeclare["parser_checks"]["dope_catboost_4n_median"]
    expected_n = predeclare["parser_checks"]["dope_catboost_4n_n"]
    if len(anchor_values) != expected_n or anchor_median is None or abs(anchor_median - expected) > 1e-12:
        raise ValueError(f"published DOPE anchor mismatch {len(anchor_values)} {anchor_median}")
    if stop_after_anchor:
        return _clean({
            "format": "dope-review-fix-parser-check",
            "anchor": {"n": len(anchor_values), "median": anchor_median},
            "predeclaration_sha256": _sha(predeclare_path),
        })
    tasks = load_tasks(names)

    compared = disagreements = 0
    for size in (1, 4):
        for auditor in AUDITORS:
            left = maps[("DOPE", "features12_steps2048")][size][auditor]
            right = maps[("DOPE", "features12_steps2048_sdv")][size][auditor]
            for dataset in set(left) & set(right):
                compared += 1
                if abs(left[dataset] - right[dataset]) > 1e-9:
                    disagreements += 1

    published = list(DENSITY) + list(NEURAL) + list(ARF) + list(FOREST)
    retention_rows = []
    size_contrast_rows = []
    for method, configuration in published:
        for auditor in AUDITORS:
            for size in (1, 4):
                values = maps[(method, configuration)][size][auditor]
                summary = marginal(values, names, f"marginal|{method}|{configuration}|{auditor}|{size}")
                summary.update({"method": method, "configuration": configuration, "auditor": auditor, "size": size})
                retention_rows.append(summary)
            left = maps[(method, configuration)][1][auditor]
            right = maps[(method, configuration)][4][auditor]
            contrast = paired(left, right, names, f"size-contrast|{method}|{configuration}|{auditor}")
            contrast.update({"method": method, "configuration": configuration, "auditor": auditor})
            size_contrast_rows.append(contrast)

    blocks = {
        "density": (DENSITY[1:], "density"),
        "neural": (NEURAL, "neural"),
        "arf": (ARF, "arf"),
        "forest": (FOREST, "forest"),
    }
    dope = maps[("DOPE", "features12_steps2048")]
    paired_rows = []
    for block_name, (comparators, _label) in blocks.items():
        for size in (1, 4):
            family = []
            drafted = []
            for auditor in AUDITORS:
                for method, configuration in comparators:
                    item = paired(
                        dope[size][auditor],
                        maps[(method, configuration)][size][auditor],
                        names,
                        f"paired|{block_name}|{size}|{auditor}|{method}|{configuration}",
                    )
                    item.update({
                        "block": block_name,
                        "size": size,
                        "auditor": auditor,
                        "method": method,
                        "configuration": configuration,
                    })
                    drafted.append(item)
                    family.append(item["wilcoxon_p"])
            adjusted = holm(family)
            for item, p_value in zip(drafted, adjusted):
                item["holm_p"] = p_value
                paired_rows.append(item)

    family_rows = []
    for family in ("feynman", "strogatz", "fri", "bng", "other"):
        for size in (1, 4):
            for auditor in AUDITORS:
                members = {
                    dataset: value
                    for dataset, value in dope[size][auditor].items()
                    if family_of(names.get(dataset, dataset)) == family
                }
                summary = marginal(members, names, f"family|{family}|{size}|{auditor}|DOPE")
                summary.update({"family": family, "size": size, "auditor": auditor, "method": "DOPE"})
                family_rows.append(summary)
                for method, configuration in DENSITY[1:]:
                    other = {
                        dataset: maps[(method, configuration)][size][auditor][dataset]
                        for dataset in members
                        if dataset in maps[(method, configuration)][size][auditor]
                    }
                    left = {dataset: members[dataset] for dataset in other}
                    item = paired(left, other, names, f"family-paired|{family}|{size}|{auditor}|{method}")
                    item.update({
                        "family": family, "size": size, "auditor": auditor,
                        "method": method, "configuration": configuration,
                    })
                    family_rows.append(item)

    excluding_rows = []
    for size in (1, 4):
        family = []
        drafted = []
        for auditor in AUDITORS:
            for method, configuration in DENSITY[1:]:
                left = {
                    dataset: value for dataset, value in dope[size][auditor].items()
                    if family_of(names.get(dataset, dataset)) == "other"
                }
                right = {
                    dataset: maps[(method, configuration)][size][auditor][dataset]
                    for dataset in left
                    if dataset in maps[(method, configuration)][size][auditor]
                }
                left = {dataset: left[dataset] for dataset in right}
                item = paired(left, right, names, f"excluding|{size}|{auditor}|{method}|{configuration}")
                item.update({
                    "size": size, "auditor": auditor, "method": method, "configuration": configuration,
                })
                drafted.append(item)
                family.append(item["wilcoxon_p"])
        adjusted = holm(family)
        for item, p_value in zip(drafted, adjusted):
            item["holm_p"] = p_value
            excluding_rows.append(item)

    denominator_rows = []
    rmse_rows = []
    delta_points = []
    for method, configuration in published:
        reduced = reductions[(method, configuration)]
        for size in (1, 4):
            for auditor in AUDITORS:
                informative = noninformative = undefined = capped = incomplete = 0
                capped_values = []
                rmse_values = {}
                trtr_values = {}
                for (dataset, row_size), row in reduced.items():
                    if row_size != size:
                        continue
                    decision = sensitivity(row, auditor)
                    status = decision["status"]
                    if status == "incomplete":
                        incomplete += 1
                        continue
                    if status == "undefined":
                        undefined += 1
                        continue
                    if row["auditors"][auditor]["informative"]:
                        informative += 1
                    else:
                        noninformative += 1
                    if status == "capped":
                        capped += 1
                    capped_values.append((dataset, decision["value"]))
                    task = tasks.get(dataset)
                    synthetic = lineage_error(row["auditors"][auditor]["tstr"], task, "original")
                    real = lineage_error(row["auditors"][auditor]["trtr"], task, "original")
                    if synthetic is not None and synthetic["kind"] == "rmse":
                        rmse_values[dataset] = synthetic["value"]
                    if real is not None and real["kind"] == "rmse":
                        trtr_values[dataset] = real["value"]
                    if method == "DOPE" and auditor == "catboost" and len(row["nulls"]) == 3:
                        se = delta_method_se(
                            row["auditors"][auditor]["tstr"],
                            float(np.median(row["nulls"])),
                            float(np.median(row["auditors"][auditor]["trtr"])) if row["auditors"][auditor]["trtr"] else None,
                        )
                        gap = None
                        if row["auditors"][auditor]["trtr"]:
                            gap = abs(float(np.median(row["nulls"])) - float(np.median(row["auditors"][auditor]["trtr"])))
                        if se is not None and gap is not None:
                            delta_points.append({
                                "dataset": dataset,
                                "size": size,
                                "family": family_of(names.get(dataset, dataset)),
                                "gap": gap,
                                "se": se,
                                "informative": row["auditors"][auditor]["informative"],
                            })
                sens_map = {dataset: value for dataset, value in capped_values}
                sens = marginal(sens_map, names, f"sensitivity|{method}|{configuration}|{size}|{auditor}")
                denominator_rows.append({
                    "method": method,
                    "configuration": configuration,
                    "size": size,
                    "auditor": auditor,
                    "n_informative": informative,
                    "n_noninformative": noninformative,
                    "n_undefined": undefined,
                    "n_capped": capped,
                    "n_incomplete": incomplete,
                    "sensitivity_median": sens["median"],
                    "sensitivity_lo": sens["lo"],
                    "sensitivity_hi": sens["hi"],
                    "sensitivity_n": sens["n"],
                })
                synthetic_summary = marginal(rmse_values, names, f"rmse|{method}|{configuration}|{size}|{auditor}")
                real_summary = marginal(trtr_values, names, f"trtr-rmse|{method}|{configuration}|{size}|{auditor}")
                rmse_rows.append({
                    "method": method,
                    "configuration": configuration,
                    "size": size,
                    "auditor": auditor,
                    "n_regression": synthetic_summary["n"],
                    "synthetic_rmse_median": synthetic_summary["median"],
                    "synthetic_rmse_lo": synthetic_summary["lo"],
                    "synthetic_rmse_hi": synthetic_summary["hi"],
                    "real_rmse_median": real_summary["median"],
                    "real_rmse_n": real_summary["n"],
                    "task_counts": _task_counts(rmse_values, reduced, size, tasks),
                })

    fidelity_rows = stored_fidelity_rows(reductions, names)

    seed_rows = fivefit_rows(RESULTS / "cpu-fivefit-validation-v1" / "arf" / "panel.json")
    seed_rows += fivefit_rows(RESULTS / "cpu-fivefit-validation-v1" / "forest" / "panel.json")
    task_counts = defaultdict(int)
    for info in tasks.values():
        task_counts[info["task"]] += 1

    payload = {
        "format": "dope-review-fix-receipt-panel",
        "version": 1,
        "predeclaration_sha256": _sha(predeclare_path),
        "sources": {
            "s3-lineage-record.json": record_sha,
            "density-matched-population-validation.json": density_sha,
            "sdv-matched-population-validation.json": sdv_sha,
            "arf-matched-population-validation.json": arf_sha,
            "s3-matched-forest-confirmation-validation.json": forest_sha,
        },
        "fit_seed": 11,
        "fit_seed_stage": "not_identified",
        "task_counts": dict(task_counts),
        "anchor": {"n": len(anchor_values), "median": anchor_median},
        "sdv_dope_disagreements": {"compared": compared, "disagreements": disagreements},
        "retention": retention_rows,
        "size_contrast": size_contrast_rows,
        "paired": paired_rows,
        "family": family_rows,
        "excluding_simulated": excluding_rows,
        "denominator": denominator_rows,
        "rmse": rmse_rows,
        "delta_method_points": delta_points,
        "fidelity_stored": fidelity_rows,
        "existing_multiseed": seed_rows,
        "claims": {
            "mfs_v2": None,
            "ptf_v1": None,
            "release_safe_l3": None,
            "superiority": None,
            "formal_dp": False,
        },
    }
    payload["findings"] = _findings(payload)
    return _clean(payload)


def _task_counts(rmse_values, reduced, size, tasks) -> dict:
    counts = defaultdict(int)
    for (dataset, row_size) in reduced:
        if row_size == size and dataset in tasks:
            counts[tasks[dataset]["task"]] += 1
    counts["regression_with_original_rmse"] = len(rmse_values)
    return dict(counts)


def _findings(payload) -> list:
    def one(method, configuration, auditor, size):
        for row in payload["retention"]:
            if (row["method"], row["configuration"], row["auditor"], row["size"]) == (method, configuration, auditor, size):
                return row
        return None

    lines = []
    for size, words in ((1, "size n"), (4, "size 4n")):
        row = one("DOPE", "features12_steps2048", "catboost", size)
        lines.append(
            f"DOPE CatBoost {words}: median retention {fmt(row['median'])} "
            f"{ci(row)} on {row['n']} informative lineages (fit seed 11)."
        )
    equivalents = [
        row for row in payload["paired"]
        if row["auditor"] == "catboost" and row["tost"]["equivalent"]
    ]
    lines.append(
        "CatBoost TOST at the pre-declared ±0.02 mean-difference margin, treating lineages as iid, passes for "
        + (", ".join(f"{row['method']} {row['configuration']} size {row['size']}n" for row in equivalents) or "no published comparator")
        + "."
    )
    return lines


def _nfields(line: str) -> int:
    body = line.strip()
    if body.endswith("\\\\"):
        body = body[:-2].rstrip()
    return body.replace(r"\&", "").count("&") + 1


def _table(header: str, rows: list[str], align: str) -> str:
    expected = _nfields(header)
    if len(align) != expected:
        raise ValueError(f"column spec {align!r} does not match header fields {expected}: {header}")
    for row in rows:
        if row.lstrip().startswith("%"):
            continue
        got = _nfields(row)
        if got != expected:
            raise ValueError(f"row has {got} fields, header has {expected}: {row}")
    body = "\n".join(rows) if rows else "% no rows"
    return (
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"
        f"{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def _metric_median(block) -> str:
    if isinstance(block, dict):
        return fmt(block.get("median"))
    return fmt(block)


def _measured_prefix(row) -> str:
    block = row.get("catboost_retention")
    measured = block.get("measured_datasets") if isinstance(block, dict) else None
    datasets = row.get("datasets")
    if isinstance(datasets, list):
        datasets = len(datasets)
    if measured is None:
        return str(datasets)
    return f"{measured}/{datasets}"


def _metric_label(row) -> str:
    if row.get("metric") == "c2st":
        return "stored logistic C2ST"
    if row.get("metric") == "ks":
        return "marginal KS"
    if row.get("metric") == "pair":
        return "pair correlation"
    return str(row.get("metric"))


def emit_tex(payload) -> None:
    note = (
        "% Wilcoxon and the TOST t-tests treat paired lineages as iid. "
        "The interval is the family-cluster bootstrap of the median. "
        "Holm families are separate for size n and size 4n "
        "(density 9, neural 6, ARF 6, Forest-Flow 3).\n"
    )
    retention = [
        _row_retention(row["method"], row["configuration"], row["auditor"], row["size"], row)
        for row in payload["retention"]
    ]
    write_tex("review-size-retention.tex", _table(
        "Method & Configuration & Auditor & Size & $n$ & Median & Hierarchical 95\\% CI",
        retention,
        "lllllrr",
    ))
    paired_lines = []
    for row in payload["paired"]:
        tost = row["tost"]
        flag = "yes" if tost["equivalent"] else "no"
        paired_lines.append(
            f"{row['size']}$n$ & {tex_name(row['auditor'])} & {tex_name(row['method'])} & "
            f"{tex_name(row['configuration'])} & {row['n']} & {fmt(row['median'])} & {ci(row)} & "
            f"{fmt(row['mean'])} & {flag} & {row['wins']}/{row['ties']}/{row['losses']} & {fmt(row['holm_p'], 3)} \\\\"
        )
    write_tex("review-paired.tex", note + _table(
        "Size & Auditor & Comparator & Configuration & $n$ & Median diff. & Hierarchical CI & Mean & TOST & W/T/L & Holm $p$",
        paired_lines,
        "lllllrrllrl",
    ))
    tost_lines = []
    for row in payload["paired"]:
        tost = row["tost"]
        tost_lines.append(
            f"{row['size']}$n$ & {tex_name(row['auditor'])} & {tex_name(row['method'])} & "
            f"{fmt(tost['mean'])} & [{fmt(tost['ci90_lo'])}, {fmt(tost['ci90_hi'])}] & "
            f"{fmt(tost['p'], 3)} & {'yes' if tost['equivalent'] else 'no'} \\\\"
        )
    write_tex("review-tost.tex", note + _table(
        "Size & Auditor & Comparator & Mean diff. & 90\\% CI & TOST $p$ & Equivalent at $\\pm 0.02$",
        tost_lines,
        "lllrrrl",
    ))
    family_lines = []
    for row in payload["family"]:
        if row.get("method") != "DOPE" and row["auditor"] != "catboost":
            continue
        label = "DOPE" if row["method"] == "DOPE" else row["method"]
        family_lines.append(
            f"{tex_name(row['family'])} & {row['size']}$n$ & {tex_name(row['auditor'])} & "
            f"{tex_name(label)} & {row['n']} & {fmt(row['median'])} & {ci(row)} \\\\"
        )
    for row in payload["excluding_simulated"]:
        if row["auditor"] != "catboost":
            continue
        family_lines.append(
            f"excluding simulated & {row['size']}$n$ & catboost & {tex_name(row['method'])} & "
            f"{row['n']} & {fmt(row['median'])} & {ci(row)} \\\\"
        )
    write_tex("review-family.tex", _table(
        "Family & Size & Auditor & Contrast & $n$ & Median & Hierarchical CI",
        family_lines,
        "lllllrr",
    ))
    denom_lines = []
    for row in payload["denominator"]:
        if row["auditor"] != "catboost":
            continue
        denom_lines.append(
            f"{tex_name(row['method'])} & {tex_name(row['configuration'])} & {row['size']}$n$ & "
            f"{row['n_informative']} & {row['n_noninformative']} & {row['n_undefined']} & "
            f"{row['n_capped']} & {row['sensitivity_n']} & {fmt(row['sensitivity_median'])} \\\\"
        )
    write_tex("review-denominator.tex", _table(
        "Method & Configuration & Size & Informative & Noninformative & Undefined & Capped & Sensitivity $n$ & Capped median",
        denom_lines,
        "lllrrrrrr",
    ))
    rmse_lines = []
    for row in payload["rmse"]:
        if row["auditor"] != "catboost":
            continue
        rmse_lines.append(
            f"{tex_name(row['method'])} & {row['size']}$n$ & {row['n_regression']} & "
            f"{fmt(row['synthetic_rmse_median'])} & {fmt(row['real_rmse_median'])} \\\\"
        )
    write_tex("review-rmse.tex", _table(
        "Method & Size & Regression lineages & Synthetic RMSE & Real TRTR RMSE",
        rmse_lines,
        "llrrr",
    ))
    fidelity_lines = []
    for row in payload["fidelity_stored"]:
        fidelity_lines.append(
            f"{tex_name(row['method'])} & {row['size']}$n$ & {tex_name(_metric_label(row))} & "
            f"{row['n']} & {fmt(row['median'])} & {ci(row)} \\\\"
        )
    methods = ", ".join(sorted({str(row["method"]) for row in payload["fidelity_stored"]}))
    fidelity_note = (
        "% Stored logistic C2ST, marginal KS, and pair correlation from the population ledgers. "
        "Each n counts lineages with three finite stored values. Cohorts are not matched, "
        "and this table is not a cross-method ranking. The stored C2ST is not the GBDT detector. "
        f"Methods in this file: {methods}. TabSyn is not in these ledgers.\n"
    )
    write_tex("review-fidelity-stored.tex", fidelity_note + _table(
        "Method & Size & Metric & $n$ lineages & Median & Hierarchical CI",
        fidelity_lines,
        "lllrrr",
    ))
    seed_lines = []
    for row in payload["existing_multiseed"]:
        seed_lines.append(
            f"{tex_name(row['method'])} & {row['fit_seed']} & {row['row_multiplier']} & "
            f"{_measured_prefix(row)} & {_metric_median(row.get('catboost_retention'))} & "
            f"{_metric_median(row.get('distance_mia_auc'))} & "
            f"{_metric_median(row.get('dcr_validation_median'))} \\\\"
        )
    write_tex("review-seeds-existing.tex", (
        "% Existing cpu-fivefit prefix. It is not pooled with the 97-lineage table. "
        "Measured/prefix counts finite CatBoost retentions over the prefix size.\n"
    ) + _table(
        "Method & Fit seed & Size & Measured/prefix & CatBoost retention & Distance MIA AUC & Holdout DCR median",
        seed_lines,
        "llllrrr",
    ))
    _figure(payload["delta_method_points"])


def _figure(points) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(4.2, 3.2))
    for size, marker in ((1, "o"), (4, "s")):
        chosen = [point for point in points if point["size"] == size]
        axis.scatter(
            [point["gap"] for point in chosen],
            [point["se"] for point in chosen],
            s=12,
            marker=marker,
            label=f"{size}n",
            alpha=0.8,
        )
    axis.set_xlabel("Absolute null minus real loss")
    axis.set_ylabel("Sample SD of three TSTR losses / |null-trtr|")
    axis.set_title("DOPE CatBoost, three sample seeds")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        FIGURES / "review-retention-se.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _require_anchor(payload) -> None:
    predeclare = json.loads((REPO / "research" / "benchmark" / "review_fixes" / "predeclare.json").read_text())
    expected_n = predeclare["parser_checks"]["dope_catboost_4n_n"]
    expected = predeclare["parser_checks"]["dope_catboost_4n_median"]
    anchor = payload.get("anchor") or {}
    median = anchor.get("median")
    if anchor.get("n") != expected_n or median is None or abs(float(median) - expected) > 1e-12:
        raise SystemExit(f"panel anchor mismatch {anchor}")


def _same_median(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return float(left) == float(right)


def patch_stored_fidelity(panel_path: Path) -> None:
    """Replace only the stored-fidelity rows. Other panel numbers stay put."""
    payload = json.loads(panel_path.read_text())
    _require_anchor(payload)
    campaign = _load_reductions()
    fresh = stored_fidelity_rows(campaign["reductions"], campaign["names"])
    old = {
        (row["method"], row["configuration"], row["size"], row["metric"]): row.get("median")
        for row in payload.get("fidelity_stored") or []
    }
    for row in fresh:
        key = (row["method"], row["configuration"], row["size"], row["metric"])
        if key in old and not _same_median(old[key], row.get("median")):
            raise SystemExit(f"stored fidelity drift {key}")
    payload["fidelity_stored"] = fresh
    payload["fidelity_coverage"] = (
        "unmatched stored values; not a cross-method ranking; "
        "stored C2ST is logistic; TabSyn is absent from these ledgers"
    )
    text = json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n"
    temporary = panel_path.with_suffix(".json.tmp")
    temporary.write_text(text)
    temporary.replace(panel_path)
    print(f"patched fidelity rows {len(fresh)} anchor n={payload['anchor']['n']}")


def _load_reductions() -> dict:
    panel_path = RESULTS / "review-fixes-receipts-v1" / "panel.json"
    if panel_path.is_file():
        pins = json.loads(panel_path.read_text()).get("sources") or {}
        for name, recorded in pins.items():
            if _sha(RESULTS / name) != recorded:
                raise SystemExit(f"source drift {name}")
    record, _record_sha = _load("s3-lineage-record.json")
    density, _density_sha = _load("density-matched-population-validation.json")
    sdv, _sdv_sha = _load("sdv-matched-population-validation.json")
    arf, _arf_sha = _load("arf-matched-population-validation.json")
    forest, _forest_sha = _load("s3-matched-forest-confirmation-validation.json")
    names = names_of(record)
    reductions = {}
    for method, configuration in DENSITY:
        reductions[(method, configuration)] = reduce_cells(density["cells"], method, configuration)
    for method, configuration in NEURAL:
        reductions[(method, configuration)] = reduce_cells(sdv["cells"], method, configuration)
    for method, configuration in ARF:
        reductions[(method, configuration)] = reduce_cells(arf["cells"], method, configuration)
    for method, configuration in FOREST:
        reductions[(method, configuration)] = reduce_cells(forest["cells"], method, configuration)
    return {"names": names, "reductions": reductions}


def emit_from_panel(panel_path: Path) -> None:
    payload = json.loads(panel_path.read_text())
    _require_anchor(payload)
    for name, recorded in (payload.get("sources") or {}).items():
        actual = _sha(RESULTS / name)
        if actual != recorded:
            raise SystemExit(f"source drift {name}")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if text != panel_path.read_text():
        raise SystemExit("panel is not canonical json; refuse to emit a second serialization")
    (GENERATED / "review-receipts.json").write_text(text)
    emit_tex(payload)
    for line in payload.get("findings") or []:
        print(line)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-parser", action="store_true")
    parser.add_argument("--from-panel", type=Path)
    parser.add_argument("--patch-fidelity", type=Path)
    args = parser.parse_args()
    if args.check_parser:
        payload = build(stop_after_anchor=True)
        anchor = payload["anchor"]
        print(f"parser check ok n={anchor['n']} median={anchor['median']}")
        return
    if args.patch_fidelity:
        patch_stored_fidelity(args.patch_fidelity)
        return
    if args.from_panel:
        emit_from_panel(args.from_panel)
        return
    if os.environ.get("DOPE_RF_REBUILD") != "1":
        raise SystemExit("refusing a private ledger rebuild; pass --from-panel")
    payload = build(stop_after_anchor=False)
    out_dir = RESULTS / "review-fixes-receipts-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out_dir / "panel.json").write_text(text)
    (GENERATED / "review-receipts.json").write_text(text)
    emit_tex(payload)
    for line in payload["findings"]:
        print(line)


if __name__ == "__main__":
    main()
