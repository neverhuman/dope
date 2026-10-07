"""Publish matched validation scalars from complete, hash-bound metric receipts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
import statistics

from research.benchmark.publish_retained_metrics import flatten, verified


IDENTITY = ("dataset", "method", "fit_seed", "sample_seed", "row_multiplier",
            "config_sha256", "selection_binding", "charged_artifact_bytes")
CLAIMS = ("mfs_v2", "ptf_v1", "release_safe_l3", "superiority")


def at_pointer(document, pointer):
    for part in pointer.strip("/").split("/"):
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def alias_identity(alias):
    if "logical_alias" not in alias:
        return {key: alias[key] for key in IDENTITY}
    original = alias["logical_alias"]
    selected = alias["selection_binding"]
    if selected["common_metrics_used_for_selection"] is not False:
        raise ValueError("shared_metric_selection_forbidden")
    if original["configuration"] not in ("default", "native_selected"):
        raise ValueError("unknown_native_configuration")
    return dict(dataset=original["dataset"], method=original["method"],
                fit_seed=original["fit_seed"], sample_seed=original["sample_seed"],
                row_multiplier=original["size_multiplier"],
                config_sha256=selected["config_sha256"],
                selection_binding="author_default" if original["configuration"] == "default" else "native_selected",
                charged_artifact_bytes=sum(ref["bytes"] for ref in alias["artifact_inventory"]))


def read_join(join_ref, cache, receipt_mirror=None):
    def read(ref):
        key = (ref["path"], ref["sha256"], ref["bytes"])
        if key not in cache:
            path = Path(ref["path"])
            if not path.is_file() and receipt_mirror is not None:
                path = Path(receipt_mirror) / path.name
            cache[key] = verified(path, ref)
        return cache[key]

    joined = read(join_ref)
    aliases = read(joined["alias_map_ref"])
    original = read(joined["input_manifest_ref"])
    jobs = {job["job_sha256"]: job for job in original["jobs"]}
    lock = read(joined["physical_receipt_lock_ref"])
    if lock["actual_complete"] is not True or lock["official_tests_opened"] is not False:
        raise ValueError("incomplete_metric_matrix")
    locked = {ref["sha256"]: ref for ref in lock["receipts"]}
    if len(locked) != lock["count"]:
        raise ValueError("duplicate_physical_receipt")
    if "utility_receipt_lock_ref" in joined:
        utility_lock = read(joined["utility_receipt_lock_ref"])
        if utility_lock["actual_complete"] is not True or utility_lock["utility_auditor_seed"] != 1729:
            raise ValueError("incomplete_utility_recomputation")
        locked.update((ref["sha256"], ref) for ref in utility_lock["receipts"])
        utility_inputs = read(utility_lock["manifest_ref"])
        jobs.update((job["job_sha256"], job) for job in utility_inputs["jobs"])
    result = []
    for row in joined["rows"]:
        if row["official_tests_opened"] is not False or any(row[key] is not None for key in CLAIMS):
            raise ValueError("invalid_validation_claim")
        alias = at_pointer(aliases, row["alias_json_pointer"])
        if row["alias_origin_ref"] != joined["alias_map_ref"]:
            raise ValueError("alias_manifest_binding_mismatch")
        identity = alias_identity(alias)
        if any(row[key] != val for key, val in identity.items()):
            raise ValueError("logical_configuration_binding_mismatch")
        ref = row["metric_receipt_ref"]
        if ref["sha256"] not in locked or locked[ref["sha256"]] != ref:
            raise ValueError("metric_receipt_not_in_frozen_lock")
        receipt = read(ref)
        if receipt["job_sha256"] != row["metric_job_sha256"] or receipt["status"] != "ok":
            raise ValueError("physical_metric_identity_mismatch")
        canonical = json.dumps(receipt["job_identity"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != receipt["job_sha256"]:
            raise ValueError("noncanonical_metric_job")
        job = jobs[receipt["job_sha256"]]
        if receipt["job_identity"] != job["job_identity"]:
            raise ValueError("metric_job_not_in_frozen_matrix")
        for name in ("dataset", "method", "fit_seed", "sample_seed", "row_multiplier"):
            if receipt[name] != identity[name]:
                raise ValueError("metric_input_identity_mismatch")
        for key in ("train", "validation", "synthetic", "projection"):
            if receipt["input_refs"][key + "_ref"] != job["evaluator_input"][key + "_ref"]:
                raise ValueError("metric_input_not_in_frozen_matrix")
            if receipt["input_refs"][key + "_ref"]["sha256"] != row[key + "_sha256"]:
                raise ValueError("aliased_numeric_input_mismatch")
        if alias["synthetic_ref"]["sha256"] != receipt["input_refs"]["synthetic_ref"]["sha256"]:
            raise ValueError("alias_synthetic_input_mismatch")
        if "metrics" not in receipt:
            for key in ("expanded_metric_ref", "expanded_metric_origin_ref"):
                if receipt[key] != job["evaluator_input"][key]:
                    raise ValueError("retained_diagnostic_not_in_frozen_job")
            expanded = read(receipt["expanded_metric_ref"])
            origin = read(receipt["expanded_metric_origin_ref"])
            if origin["job_sha256"] != receipt["job_sha256"] or origin["status"] != "ok" or \
                    origin["diagnostics"] != expanded or receipt["utility_auditor_seed"] != 1729:
                raise ValueError("retained_diagnostic_binding_mismatch")
            receipt = {**receipt, "metrics": {**expanded, "utility": receipt["utility"]}}
        if receipt["metrics"] != row["measurement"]:
            raise ValueError("logical_metric_value_changed")
        for record in (receipt, receipt["metrics"]):
            if record.get("official_tests_opened") is not False or any(record.get(key) is not None for key in CLAIMS):
                raise ValueError("raw_metric_validation_claim_changed")
        scalar = flatten({**receipt, **identity})
        scalar.update(metric_receipt_ref=ref, metric_job_sha256=receipt["job_sha256"],
                      metrics=receipt["metrics"], dependencies=receipt["dependencies"], source_refs=receipt["source_refs"],
                      alias_origin_ref=row["alias_origin_ref"], alias_json_pointer=row["alias_json_pointer"],
                      input_hashes={key: row[key + "_sha256"] for key in ("train", "validation", "synthetic", "projection")})
        result.append(scalar)
    if len(result) != joined["logical_cells"]:
        raise ValueError("logical_matrix_count_mismatch")
    return result


def summarize(rows, datasets, scope):
    result = []
    group_keys = sorted({(x["method"], x["selection_binding"], x["row_multiplier"]) for x in rows})
    excluded = {*IDENTITY, "metric_receipt_ref", "metric_job_sha256", "alias_origin_ref", "alias_json_pointer", "input_hashes", "metrics", "dependencies", "source_refs"}
    for method, selection, n in group_keys:
        group = [x for x in rows if (x["method"], x["selection_binding"], x["row_multiplier"]) == (method, selection, n)
                 and x["dataset"] in datasets]
        by_dataset = {d: [x for x in group if x["dataset"] == d] for d in datasets}
        if any(len(cells) != 3 or {x["sample_seed"] for x in cells} != {101, 211, 307} or
               len({(x["fit_seed"], x["config_sha256"]) for x in cells}) != 1 for cells in by_dataset.values()):
            raise ValueError("incomplete_matched_three_seed_group")
        aggregate = dict(method=method, selection_binding=selection, row_multiplier=n, scope=scope,
                         datasets=len(datasets), sample_cells=len(group),
                         aggregation="mean_of_three_sample_seeds_then_median_of_datasets")
        for metric in rows[0]:
            if metric in excluded:
                continue
            values = [statistics.mean([x[metric] for x in cells]) for cells in by_dataset.values()
                      if all(x[metric] is not None for x in cells)]
            aggregate[metric] = dict(median=statistics.median(values) if values else None,
                                    measured_datasets=len(values), unavailable_datasets=len(datasets) - len(values))
        result.append(aggregate)
    return result


def build(config, receipt_mirror=None):
    cache = {}
    rows = [row for ref in config["joined_panel_refs"] for row in read_join(ref, cache, receipt_mirror)]
    executions = {name: verified(ref["path"], ref) for name, ref in config["execution_refs"].items()}
    for joined_ref, name in zip(config["joined_panel_refs"], ("arf", "copula_chow")):
        joined = cache[(joined_ref["path"], joined_ref["sha256"], joined_ref["bytes"])]
        execution = executions[name]
        if execution["receipt_lock_ref"]["sha256"] != joined["physical_receipt_lock_ref"]["sha256"] or \
                execution["source_refs"]["original_metric_input_manifest"]["sha256"] != joined["input_manifest_ref"]["sha256"]:
            raise ValueError("execution_matrix_binding_mismatch")
    for execution in executions.values():
        if execution["official_tests_opened"] is not False or any(execution.get(key) is not None for key in CLAIMS):
            raise ValueError("execution_validation_claim_changed")
    utility = executions["utility_only"]
    if utility["receipt_lock_ref"]["sha256"] != joined["utility_receipt_lock_ref"]["sha256"] or utility["utility_only_jobs"] != 5:
        raise ValueError("utility_execution_matrix_binding_mismatch")
    ts = verified(config["tabsyn_panel_ref"]["path"], config["tabsyn_panel_ref"])
    ts_rows = [flatten(cell) for cell in ts["cells"]]
    common = sorted({x["dataset"] for x in ts_rows})
    if len(ts_rows) != 48 or len(common) != 8:
        raise ValueError("complete_eight_lineage_tabsyn_panel_required")
    populations = {method: sorted({x["dataset"] for x in rows if x["method"] == method})
                   for method in sorted({x["method"] for x in rows})}
    if set(populations) != {"ARF", "GaussianCopula", "Chow-Liu"}:
        raise ValueError("complete_requested_method_panel_required")
    if len(rows) != 3600:
        raise ValueError("complete_requested_logical_matrix_required")
    if any(len(datasets) != 100 for datasets in populations.values()):
        raise ValueError("complete_hundred_lineage_panel_required")
    if len({tuple(datasets) for datasets in populations.values()}) != 1:
        raise ValueError("classical_population_mismatch")
    summary = []
    for method, datasets in populations.items():
        summary += summarize([x for x in rows if x["method"] == method], datasets, "common_100_lineages")
    for cell, scalar in zip(ts["cells"], ts_rows):
        scalar.update(metric_receipt_ref=cell["receipt_ref"], metric_job_sha256=cell["job_sha256"],
                      metrics=cell["metrics"], dependencies=cell["dependencies"], source_refs=ts["source_refs"],
                      alias_origin_ref=None, alias_json_pointer=None,
                      input_hashes={key: value for key, value in ((name, cell["input_hashes"][name + "_ref"])
                                                                for name in ("train", "validation", "synthetic", "projection"))})
    matched = [x for x in rows + ts_rows if x["dataset"] in common]
    for dataset in populations["ARF"]:
        cohort = [x for x in rows + ts_rows if x["dataset"] == dataset]
        for key in ("train", "validation", "projection"):
            if len({x["input_hashes"][key] for x in cohort}) != 1:
                raise ValueError("cross_method_split_mismatch")
        for key in ("null_loss", "catboost_trtr_loss", "linear_trtr_loss", "mlp_trtr_loss"):
            values = [x[key] for x in cohort if x[key] is not None]
            if not all(math.isclose(x, values[0], rel_tol=1e-12, abs_tol=1e-12) for x in values):
                raise ValueError("cross_method_real_auditor_mismatch")
    summary += summarize(matched, common, "matched_tabsyn_eight_lineages")
    measurements = {}
    for row in rows + ts_rows:
        record = {key: row.pop(key) for key in ("metrics", "dependencies", "source_refs")}
        record.update(receipt_ref=row["metric_receipt_ref"], job_sha256=row["metric_job_sha256"])
        key = row["metric_receipt_ref"]["sha256"]
        if key in measurements and measurements[key] != record:
            raise ValueError("aliased_measurement_changed")
        measurements[key] = record
    return dict(format="dope-retained-classical-validation-comparison", version=1,
                input_refs=config, execution_records=executions, rows=rows + ts_rows, summary=summary, measurements=measurements,
                logical_sample_cells=len(rows) + len(ts_rows), complete_classical_lineages=100,
                tabsyn_lineages=len(common), official_tests_opened=False, selection_use=False,
                production_certified=False, comparison_scope="training_derived_validation_only",
                mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None)


def render(panel):
    scalar = io.StringIO(newline="")
    fields = list(IDENTITY) + sorted(key for key in panel["rows"][0]
                                    if key not in IDENTITY and not key.endswith("_ref") and key != "input_hashes")
    writer = csv.DictWriter(scalar, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(panel["rows"])
    physical = b"".join((json.dumps(dict(receipt_sha256=key, **value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
                        for key, value in sorted(panel["measurements"].items()))
    body = {key: value for key, value in panel.items() if key != "measurements"}
    body["physical_metrics_file"] = dict(path="physical-metrics.jsonl", bytes=len(physical), sha256=hashlib.sha256(physical).hexdigest())
    metadata = {key: value for key, value in body.items() if key != "rows"}
    metadata_json = json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False)
    row_json = ",\n".join("    " + json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                           for row in panel["rows"])
    panel_json = metadata_json[:-2] + ',\n  "rows": [\n' + row_json + '\n  ]\n}\n'
    outputs = {"panel.json": panel_json.encode(),
               "physical-metrics.jsonl": physical,
               "figure-kpis.csv": scalar.getvalue().encode()}
    columns = [("catboost_retention", "CatBoost retention"), ("marginal_error_mean", "Marginal KS/TV"),
               ("c2st_catboost_auc", "CatBoost C2ST AUC"), ("distance_mia_auc", "Distance MIA AUC")]
    for scope, filename, title in [("common_100_lineages", "classical-100.svg", "Classical methods: 100 validation lineages"),
                                    ("matched_tabsyn_eight_lineages", "matched-eight.svg", "Matched outcomes: eight complete TabSyn lineages")]:
        groups = [x for x in panel["summary"] if x["scope"] == scope]
        height = 170 + 28 * len(groups)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               '<g font-family="DejaVu Sans, sans-serif" fill="#18212b">',
               f'<text x="20" y="32" font-size="21">{html.escape(title)}</text>',
               '<text x="20" y="58" font-size="12">Mean of three sample seeds per dataset, then dataset median; fit seed 11; values (measured datasets).</text>']
        svg.append('<text x="20" y="86" font-size="13">Method / configuration / size</text>')
        for i, (_, label) in enumerate(columns):
            svg.append(f'<text x="{410 + 175 * i}" y="86" font-size="13">{label}</text>')
        for r, group in enumerate(groups):
            y = 112 + 28 * r
            label = f'{group["method"]} / {group["selection_binding"]} / {group["row_multiplier"]}n'
            svg.append(f'<text x="20" y="{y}" font-size="13">{html.escape(label)}</text>')
            for i, (metric, _) in enumerate(columns):
                outcome = group[metric]
                value = "NA" if outcome["median"] is None else f'{outcome["median"]:.6f}'
                svg.append(f'<text x="{410 + 175 * i}" y="{y}" font-size="13">{value} ({outcome["measured_datasets"]})</text>')
        svg.append(f'<text x="20" y="{height - 24}" font-size="12">Validation only. Empirical attacks; no formal DP, production score or superiority claim.</text></g></svg>\n')
        outputs[filename] = "\n".join(svg).encode()
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--config-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-mirror", type=Path)
    args = parser.parse_args()
    config = verified(args.config, dict(sha256=args.config_sha256, bytes=args.config_bytes))
    panel = build(config, args.receipt_mirror)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, body in render(panel).items():
        (args.output / name).write_bytes(body)


if __name__ == "__main__":
    main()
