"""Per-lineage retention record for the rights-cleared regression panel.

Reads published validation cells only. Does not open official test partitions,
does not score MFS-v2, and does not choose a profile from the retentions.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

AUDITORS = ("catboost", "linear", "mlp")
SEEDS = (101, 211, 307)
SIZES = (1, 4)
BYTE_CAP = 10240
DISPLAYED = "features12_steps2048"
SUFFICIENCY = "features12_steps8192"
COMPARATORS = (
    ("GaussianCopula", "native_selected"),
    ("Chow-Liu", "native_selected"),
    ("independent_marginals", "native_selected"),
)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def group_retention(cells):
    """Median retention when every sample seed is informative. Otherwise null."""
    if len(cells) != len(SEEDS):
        raise ValueError("sample group is not the three frozen seeds")
    by_seed = {}
    for cell in cells:
        seed = cell["sample_seed"]
        if seed not in SEEDS or seed in by_seed:
            raise ValueError("sample seed differs from the frozen schedule")
        by_seed[seed] = cell
    charges = {(cell.get("charged_artifact_bytes"), cell.get("artifact_within_l3_cap")) for cell in cells}
    if len(charges) != 1:
        raise ValueError("sample group artifact charges differ")
    bytes_value, within_cap = next(iter(charges))
    if bytes_value is not None and (type(bytes_value) is not int or bytes_value < 0):
        raise ValueError("invalid artifact charge")
    if within_cap is True and bytes_value is not None and bytes_value > BYTE_CAP:
        raise ValueError("cap flag disagrees with charged bytes")
    utility = {}
    for auditor in AUDITORS:
        scores = []
        reasons = []
        for seed in SEEDS:
            cell = by_seed[seed]
            record = (cell.get("utility") or {}).get(auditor) or {}
            informative = cell.get("status") == "ok" and record.get("informative") is True
            value = record.get("retention") if informative else None
            if value is not None and not _finite(value):
                raise ValueError("invalid common retention")
            scores.append(value)
            if value is None:
                if cell.get("status") != "ok":
                    reasons.append(cell.get("unavailable_reason") or cell.get("status") or "unavailable")
                elif record.get("informative") is not True:
                    reasons.append("not_informative")
                else:
                    reasons.append("retention_absent")
        complete = all(score is not None for score in scores)
        utility[auditor] = {
            "sample_retention_values": scores,
            "complete_informative_sample_group": complete,
            "median_retention": median(scores) if complete else None,
            "blank_reason": None if complete else reasons[0],
        }
    status_counts = {}
    for cell in cells:
        status_counts[cell.get("status")] = status_counts.get(cell.get("status"), 0) + 1
    return {
        "statuses": status_counts,
        "charged_artifact_bytes": bytes_value,
        "artifact_within_l3_cap": within_cap,
        "copy_counts": [by_seed[seed].get("copy_counts") for seed in SEEDS],
        "real_vs_real_control_counts": [by_seed[seed].get("real_vs_real_control_counts") for seed in SEEDS],
        "utility": utility,
    }


def lineage_record(dataset_ids, groups, names, shapes):
    """One row per dataset. `groups` maps (dataset, size) to a sample group."""
    rows = []
    for dataset in sorted(dataset_ids):
        shape = shapes.get(dataset, {})
        row = {
            "dataset": dataset,
            "display_name": names.get(dataset, dataset),
            "fit_rows": shape.get("fit_rows"),
            "features": shape.get("features"),
            "sizes": {},
        }
        for size in SIZES:
            row["sizes"][str(size)] = groups[(dataset, size)]
        rows.append(row)
    return rows


def index_cells(cells, method, configuration):
    grouped = defaultdict(list)
    datasets = set()
    for cell in cells:
        if cell.get("method") != method or cell.get("configuration") != configuration:
            continue
        if cell.get("mfs_v2") is not None:
            raise ValueError("cell carries an MFS-v2 value")
        datasets.add(cell["dataset"])
        grouped[(cell["dataset"], cell["size_multiplier"])].append(cell)
    groups = {}
    for dataset in datasets:
        for size in SIZES:
            groups[(dataset, size)] = group_retention(grouped[(dataset, size)])
    return datasets, groups


def panel_median(rows, auditor, size):
    scores = []
    for row in rows:
        group = row["sizes"][str(size)]["utility"][auditor]
        if group["complete_informative_sample_group"]:
            scores.append(group["median_retention"])
    return len(scores), (median(scores) if scores else None)


def write_auditor_figures(directory, stem, rows, comparators, title):
    """One PDF per auditor. Comparators map label to (dataset, size) -> group or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "DOPE": "#0072B2",
        "GaussianCopula": "#E69F00",
        "Chow-Liu": "#009E73",
        "independent_marginals": "#D55E00",
    }
    paths = []
    for auditor in AUDITORS:
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
        low, high = -1.5, 1.6
        for axis, size in zip(axes, SIZES):
            outside = 0
            for label, lookup in (("DOPE", None), *comparators):
                xs, ys = [], []
                for row in rows:
                    group = row["sizes"][str(size)] if label == "DOPE" else lookup(row["dataset"], size)
                    if group is None:
                        continue
                    utility = group["utility"][auditor]
                    charged = group["charged_artifact_bytes"]
                    if (not utility["complete_informative_sample_group"] or charged is None
                            or charged <= 0):
                        continue
                    value = utility["median_retention"]
                    if value < low or value > high:
                        outside += 1
                        continue
                    xs.append(math.log10(charged))
                    ys.append(value)
                axis.scatter(xs, ys, s=16, c=colors.get(label, "#000000"), label=label,
                             alpha=0.9, linewidths=0)
            axis.axvline(math.log10(BYTE_CAP), color="#666666", linestyle="--", linewidth=0.8)
            axis.set_ylim(low, high)
            axis.set_title(f"{size}n; {outside} points outside [−1.5, 1.6]")
            axis.set_xlabel("log10 charged bytes")
            axis.set_ylabel(f"{auditor} retention")
            axis.legend(loc="lower right", fontsize=6, frameon=True)
        figure.suptitle(f"{title}: {auditor}", fontsize=10)
        figure.tight_layout()
        path = directory / f"{stem}-{auditor}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)
        paths.append(path)
    return paths


DIFFERENCE_TITLES = {
    "GaussianCopula": "Gaussian copula",
    "Chow-Liu": "Chow–Liu",
    "independent_marginals": "Independent",
}


def write_difference_strip(path, rows, comparators):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for label, lookup in comparators:
        points = []
        for row in rows:
            own = row["sizes"]["4"]["utility"]["catboost"]
            other = lookup(row["dataset"], 4)
            if other is None or not own["complete_informative_sample_group"]:
                continue
            baseline = other["utility"]["catboost"]
            if not baseline["complete_informative_sample_group"]:
                continue
            points.append((own["median_retention"] - baseline["median_retention"], row["display_name"]))
        points.sort()
        panels.append((label, points))
    height = max(8.0, 0.16 * max((len(points) for _, points in panels), default=1))
    figure, axes = plt.subplots(1, len(panels) or 1, figsize=(11.2, height), sharey=False)
    if len(panels) == 1:
        axes = [axes]
    for axis, (label, points) in zip(axes, panels):
        title = DIFFERENCE_TITLES.get(label, label)
        if not points:
            axis.set_title(title)
            axis.text(0.5, 0.5, "no complete pairs", ha="center", va="center")
            continue
        outside = sum(value < -1.5 or value > 1.5 for value, _name in points)
        axis.barh(range(len(points)), [item[0] for item in points], color="#0072B2", height=0.72)
        axis.set_yticks(range(len(points)))
        axis.set_yticklabels([item[1] for item in points], fontsize=5.5)
        axis.tick_params(axis="y", pad=1, length=0)
        axis.axvline(0, color="#333333", linewidth=0.6)
        axis.set_xlim(-1.5, 1.5)
        axis.set_xlabel("DOPE minus comparator", fontsize=8)
        axis.set_title(f"{title} ({outside} outside)", fontsize=9)
    figure.suptitle("CatBoost retention difference at 4n, features12_steps2048", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.18)
    plt.close(figure)


def _tex_num(value):
    if value is None:
        return "---"
    return f"{value:.4f}"


def _tex_name(value):
    return str(value).replace("_", r"\_").replace("&", r"\&")


def write_longtable(path, caption, rows, columns):
    """columns is a list of (heading, callable(row) -> tex cell)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * len(columns)
    lines = [
        "\\scriptsize",
        "\\begin{longtable}{" + align + "}",
        "\\caption{" + caption + "}\\\\",
        "\\toprule",
        " & ".join(heading for heading, _ in columns) + "\\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        " & ".join(heading for heading, _ in columns) + "\\\\",
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]
    for row in rows:
        lines.append(" & ".join(str(getter(row)) for _, getter in columns) + "\\\\")
    lines.append("\\end{longtable}")
    lines.append("\\normalsize")
    path.write_text("\n".join(lines) + "\n")


def displayed_columns():
    def size_group(row, size):
        return row["sizes"][str(size)]

    return [
        ("id", lambda row: row["dataset"]),
        ("name", lambda row: _tex_name(row["display_name"])),
        ("rows", lambda row: "---" if row["fit_rows"] is None else str(row["fit_rows"])),
        ("feat.", lambda row: "---" if row["features"] is None else str(row["features"])),
        ("bytes", lambda row: "---" if size_group(row, 4)["charged_artifact_bytes"] is None
            else f"{size_group(row, 4)['charged_artifact_bytes']:,}"),
        ("cap", lambda row: {True: "pass", False: "fail"}.get(
            size_group(row, 4)["artifact_within_l3_cap"], "---")),
        ("CB $n$", lambda row: _tex_num(size_group(row, 1)["utility"]["catboost"]["median_retention"])),
        ("CB $4n$", lambda row: _tex_num(size_group(row, 4)["utility"]["catboost"]["median_retention"])),
        ("lin $4n$", lambda row: _tex_num(size_group(row, 4)["utility"]["linear"]["median_retention"])),
        ("MLP $4n$", lambda row: _tex_num(size_group(row, 4)["utility"]["mlp"]["median_retention"])),
    ]


def load_sufficiency(reconciliation_path):
    """Build lineage rows for features12_steps8192 from the closed validation round."""
    recon = json.loads(Path(reconciliation_path).read_text())
    if recon.get("official_tests_opened") is not False or recon.get("mfs_v2") is not None:
        raise ValueError("sufficiency round carries a sealed-test or fitness claim")
    selected = [cell for cell in recon["cells"] if cell.get("profile") == SUFFICIENCY]
    flat = []
    status_counts = {}
    for cell in selected:
        status_counts[cell["status"]] = status_counts.get(cell["status"], 0) + 1
        utility = None
        copy_counts = None
        controls = None
        if cell["status"] == "ok":
            metric_path = Path(cell["sample_evidence"]["metric_path"])
            if metric_path.name == "test.csv" or "evaluator" in metric_path.parts:
                raise ValueError("sufficiency metric path is not a sample receipt")
            metric = json.loads(metric_path.read_text())
            if metric.get("mfs_v2") is not None:
                raise ValueError("sufficiency metric carries MFS-v2")
            utility = metric["utility"]
            copy_counts = metric.get("copy_counts")
            controls = metric.get("real_vs_real_control_counts")
        charged = cell.get("charged_artifact_bytes")
        within = None if charged is None else charged <= BYTE_CAP
        flat.append({
            "dataset": cell["dataset"],
            "method": "DOPE",
            "configuration": SUFFICIENCY,
            "sample_seed": cell["sample_seed"],
            "size_multiplier": cell["size_multiplier"],
            "status": cell["status"],
            "unavailable_reason": cell.get("unavailable_reason"),
            "utility": utility,
            "charged_artifact_bytes": charged,
            "artifact_within_l3_cap": within,
            "copy_counts": copy_counts,
            "real_vs_real_control_counts": controls,
            "mfs_v2": None,
        })
    datasets, groups = index_cells(flat, "DOPE", SUFFICIENCY)
    rows = lineage_record(datasets, groups, {}, {})
    return {"rows": rows, "status_counts": status_counts, "lineages": len(datasets)}


def load_catalog_names(path):
    names, features = {}, {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        provenance = row.get("provenance") or {}
        names[row["id"]] = provenance.get("dataset_id") or row["id"]
        shape = row.get("shape") or {}
        features[row["id"]] = shape.get("features")
    return names, features


def load_fit_rows(worker_root, dataset_ids):
    rows = {}
    root = Path(worker_root)
    for dataset in dataset_ids:
        manifest = root / dataset / "worker-manifest.json"
        if not manifest.is_file():
            rows[dataset] = None
            continue
        rows[dataset] = json.loads(manifest.read_text())["train_rows"]
    return rows


def attach_shapes(dataset_ids, names, features, fit_rows):
    return {
        dataset: {"fit_rows": fit_rows.get(dataset), "features": features.get(dataset)}
        for dataset in dataset_ids
    }


def assert_matches_published(published_pairs, rows, profile):
    for item in published_pairs:
        if item.get("dope_profile") != profile or item.get("size_multiplier") != 4:
            continue
        if item.get("auditor") != "catboost" or item.get("configuration") != "native_selected":
            continue
        count, value = panel_median(rows, "catboost", 4)
        # The published median is paired, so it can be smaller than the DOPE-only count.
        if item["method"] == "GaussianCopula":
            if count < item["complete_paired_lineages"]:
                raise ValueError("DOPE informative count is below the published paired count")
            own = [row["sizes"]["4"]["utility"]["catboost"]["median_retention"]
                   for row in rows
                   if row["dataset"] in set(item["dataset_ids"])]
            if median(own) != item["dope_matched_median"]:
                raise ValueError("recomputed CatBoost median differs from the published pair")


def build_record(density_cells, published_pairs, names, shapes, sufficiency_rows=None):
    datasets, groups = index_cells(density_cells, "DOPE", DISPLAYED)
    if len(datasets) != 100:
        raise ValueError("displayed profile does not cover 100 lineages")
    rows = lineage_record(datasets, groups, names, shapes)
    assert_matches_published(published_pairs, rows, DISPLAYED)
    comparators = {}
    for method, configuration in COMPARATORS:
        comp_datasets, comp_groups = index_cells(density_cells, method, configuration)
        if comp_datasets != datasets:
            raise ValueError("comparator lineage set differs from DOPE")
        comparators[method] = {
            dataset: {str(size): comp_groups[(dataset, size)] for size in SIZES}
            for dataset in datasets
        }
    record = {
        "format": "dope-s3-lineage-record",
        "version": 1,
        "displayed_profile": DISPLAYED,
        "sufficiency_profile": SUFFICIENCY,
        "sufficiency_adopted_as_displayed_model": False,
        "lineages": 100,
        "mfs_v2": None,
        "ptf_v1": None,
        "release_safe_l3": None,
        "official_tests_opened": False,
        "paired_superiority": None,
        "counts_as_dope_win": False,
        "pooled_with_neural_block": False,
        "pooled_with_beyondarena": False,
        "rows": rows,
        "comparators": comparators,
        "sufficiency_rows": sufficiency_rows,
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--sufficiency-reconciliation", type=Path)
    args = parser.parse_args()
    for path in (args.density, args.catalog, args.workers):
        if "test.csv" in str(path):
            parser.error("official test input is not a source for this record")
    density = json.loads(args.density.read_text())
    if density.get("official_tests_opened") is not False or density.get("mfs_v2") is not None:
        raise ValueError("density publication carries a sealed-test or fitness claim")
    names, features = load_catalog_names(args.catalog)
    datasets = {cell["dataset"] for cell in density["cells"] if cell.get("method") == "DOPE"
                and cell.get("configuration") == DISPLAYED}
    fit_rows = load_fit_rows(args.workers, datasets)
    record = build_record(density["cells"], density["paired_descriptive"], names,
                          attach_shapes(datasets, names, features, fit_rows))
    if args.sufficiency_reconciliation:
        sufficiency = load_sufficiency(args.sufficiency_reconciliation)
        if sufficiency["lineages"] != 100:
            raise ValueError("sufficiency profile does not cover 100 lineages")
        by_id = {row["dataset"]: row for row in record["rows"]}
        for row in sufficiency["rows"]:
            source = by_id.get(row["dataset"], {})
            row["display_name"] = source.get("display_name", row["dataset"])
            row["fit_rows"] = source.get("fit_rows")
            row["features"] = source.get("features")
        record["sufficiency_rows"] = sufficiency["rows"]
        record["sufficiency_profile_status_counts"] = sufficiency["status_counts"]
        write_longtable(
            args.table_dir / "s3-8192-lineages.tex",
            "Sufficiency budget, all 100 lineages. The cap column is charged bytes "
            "at or under 10{,}240. It is not a release certificate.",
            sufficiency["rows"],
            displayed_columns(),
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    comparators = [
        (method, lambda dataset, size, method=method: record["comparators"][method][dataset][str(size)])
        for method, _ in COMPARATORS
    ]
    write_auditor_figures(args.figure_dir, "s3-retention-bytes", record["rows"], comparators,
                          "features12_steps2048")
    write_difference_strip(args.figure_dir / "s3-paired-differences.pdf", record["rows"], comparators)
    write_longtable(
        args.table_dir / "s3-2048-lineages.tex",
        "Displayed profile, all 100 lineages. The cap column is charged bytes "
        "at or under 10{,}240. It is not a release certificate.",
        record["rows"],
        displayed_columns(),
    )
    print(json.dumps({
        "lineages": len(record["rows"]),
        "catboost_4n": panel_median(record["rows"], "catboost", 4),
        "linear_4n": panel_median(record["rows"], "linear", 4),
        "mlp_4n": panel_median(record["rows"], "mlp", 4),
        "mfs_v2": None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
