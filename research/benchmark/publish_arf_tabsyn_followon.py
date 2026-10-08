"""Reduce frozen, rights-safe scalar exports; never open training or test rows.

Original metric receipts were authenticated during export. Portable regeneration
uses the committed scalar projection, whose hash is frozen in inputs.lock.json;
it does not recompute metrics or rehash retained model weights.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent / "results/arf-tabsyn-followon-validation-v1"
FIT_SEEDS = [11, 23, 37, 53, 71]
SAMPLE_SEEDS = [101, 211, 307]
IDENTITY = ("dataset", "method", "fit_seed", "sample_seed", "row_multiplier",
            "config_sha256", "selection_binding", "charged_artifact_bytes")
METRICS = ("marginal_error_mean", "pairwise_pearson_difference_mean", "contingency_tv_mean",
           "alpha_precision", "beta_recall", "c2st_catboost_auc", "c2st_logistic_auc",
           "dcr_fit_median", "dcr_validation_median", "nndr_fit_median", "distance_mia_auc",
           "domias_kde_auc", "null_loss", "catboost_trtr_loss", "catboost_tstr_loss",
           "catboost_retention", "linear_trtr_loss", "linear_tstr_loss", "linear_retention",
           "mlp_trtr_loss", "mlp_tstr_loss", "mlp_retention")
CLAIMS = dict(official_tests_opened=False, mfs_v2=None, ptf_v1=None,
              release_safe_l3=None, superiority=None)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_locked(path, ref):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlinked_scalar_export")
    data = path.read_bytes()
    if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("frozen_scalar_export_changed")
    return data


def reduce_rows(rows):
    seen, physical, splits = set(), {}, defaultdict(set)
    configurations = defaultdict(set)
    by_fit = defaultdict(list)
    for i, row in enumerate(rows):
        if any(row.get(k) != v or type(row.get(k)) is not type(v) for k, v in CLAIMS.items()):
            raise ValueError("invalid_validation_claim")
        for key, allowed in (("fit_seed", FIT_SEEDS), ("sample_seed", SAMPLE_SEEDS), ("row_multiplier", [1, 4])):
            if type(row[key]) is not int or row[key] not in allowed:
                raise ValueError("invalid_seed_or_size")
        if type(row["charged_artifact_bytes"]) is not int or row["charged_artifact_bytes"] < 0:
            raise ValueError("invalid_artifact_charge")
        if not re.fullmatch("[0-9a-f]{16}", row["dataset"]) or not re.fullmatch("[0-9a-f]{64}", row["config_sha256"]):
            raise ValueError("invalid_scalar_identity")
        if row["method"] not in ("ARF", "TabSyn") or row["selection_binding"] not in (
                ("author_default", "native_selected") if row["method"] == "ARF" else ("scaled_author_default",)):
            raise ValueError("invalid_native_selection_binding")
        if row["method"] == "TabSyn" and (row["fit_seed"], row["row_multiplier"]) != (11, 1):
            raise ValueError("unexpected_tabsyn_followon_schedule")
        identity = {key: row[key] for key in IDENTITY}
        if digest(identity) != row["logical_identity_sha256"]:
            raise ValueError("logical_identity_changed")
        key = tuple(row[k] for k in ("dataset", "method", "selection_binding", "fit_seed", "row_multiplier", "sample_seed"))
        if key in seen:
            raise ValueError("duplicate_logical_cell")
        seen.add(key)
        configurations[(row["dataset"], row["method"], row["selection_binding"])].add(row["config_sha256"])
        job = dict(row["metric_job_identity"])
        if "worker_key" in job or type(job.get("worker_sha256")) is not str or \
                not re.fullmatch("[0-9a-f]{64}", job["worker_sha256"]):
            raise ValueError("invalid_public_worker_digest")
        # Upstream calls this digest worker_key; restore its original spelling
        # only for canonical verification, avoiding credential-like sidecars.
        job["worker_key"] = job.pop("worker_sha256")
        if digest(job) != row["metric_job_sha256"] or job["fit_seed"] != row["fit_seed"] or \
                type(job["fit_seed"]) is not int:
            raise ValueError("physical_job_identity_changed")
        if row["method"] == "ARF" and (job["final"] is not False or
                job["split"] != "official_training_derived_validation" or job["track"] != "common-numeric"):
            raise ValueError("invalid_arf_validation_job")
        if set(row["input_hashes"]) != {"train", "validation", "projection", "synthetic"} or \
                any(type(h) is not str or not re.fullmatch("[0-9a-f]{64}", h) for h in row["input_hashes"].values()):
            raise ValueError("invalid_partition_binding")
        if job["sample_sha256"] != row["input_hashes"]["synthetic"]:
            raise ValueError("physical_sample_binding_changed")
        if row["method"] == "TabSyn" and any(job[k] != row[k] or type(job[k]) is not type(row[k])
                for k in ("dataset", "method", "fit_seed", "sample_seed", "row_multiplier", "config_sha256")):
            raise ValueError("tabsyn_job_identity_changed")
        splits[row["dataset"]].add(tuple(row["input_hashes"][k] for k in ("train", "validation", "projection")))
        values = {key: row[key] for key in METRICS}
        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in values.values()):
            raise ValueError("nonfinite_or_invalid_metric")
        ref = row["metric_receipt_ref"]
        if type(ref["bytes"]) is not int or ref["bytes"] <= 0 or not re.fullmatch("[0-9a-f]{64}", ref["sha256"]):
            raise ValueError("invalid_metric_receipt_ref")
        measurement = (ref, row["metric_job_sha256"], job, values, row["input_hashes"])
        if ref["sha256"] in physical and physical[ref["sha256"]] != measurement:
            raise ValueError("aliased_measurement_changed")
        physical[ref["sha256"]] = measurement
        by_fit[(row["dataset"], row["method"], row["selection_binding"], row["fit_seed"], row["row_multiplier"])].append((i, row))
    if any(len(hashes) != 1 for hashes in splits.values()):
        raise ValueError("cross_fit_or_method_split_changed")
    if any(len(configs) != 1 for configs in configurations.values()):
        raise ValueError("configuration_changed_between_fits")
    arf_ids = {r["dataset"] for r in rows if r["method"] == "ARF"}
    expected = {(d, "ARF", label, seed, n, ss) for d in arf_ids for label in ("author_default", "native_selected")
                for seed in FIT_SEEDS for n in (1, 4) for ss in SAMPLE_SEEDS}
    if len(arf_ids) != 100 or {k for k in seen if k[1] == "ARF"} != expected:
        raise ValueError("incomplete_arf_five_fit_matrix")
    per_fit, groups = [], defaultdict(list)
    for (dataset, method, label, seed, n), group in sorted(by_fit.items()):
        seeds = sorted(r["sample_seed"] for _, r in group)
        if len({r["config_sha256"] for _, r in group}) != 1:
            raise ValueError("mixed_configuration_sample_group")
        complete = seeds == SAMPLE_SEEDS
        metrics = {key: mean(r[key] for _, r in group) if complete and all(r[key] is not None for _, r in group) else None
                   for key in METRICS}
        record = dict(dataset=dataset, method=method, selection_binding=label, fit_seed=seed, row_multiplier=n,
                      complete_sample_group=complete, measured_sample_seeds=seeds,
                      missing_sample_seeds=[s for s in SAMPLE_SEEDS if s not in seeds],
                      source_row_indices=[i for i, _ in group], metrics=metrics)
        per_fit.append(record)
        if method == "ARF":
            groups[(dataset, label, n)].append((len(per_fit)-1, record))
    summary = []
    for (dataset, label, n), group in sorted(groups.items()):
        if [r["fit_seed"] for _, r in group] != FIT_SEEDS or not all(r["complete_sample_group"] for _, r in group):
            raise ValueError("incomplete_arf_fit_summary")
        metrics = {}
        for metric in METRICS:
            values = [r["metrics"][metric] for _, r in group]
            count = sum(v is not None for v in values)
            metrics[metric] = dict(fit_seed_means=values, measured_fits=count,
                                  mean=mean(values) if count == 5 else None,
                                  fit_sample_sd=stdev(values) if count == 5 else None,
                                  sd_is_confidence_interval=False)
        summary.append(dict(dataset=dataset, method="ARF", selection_binding=label, row_multiplier=n,
                            fit_seeds=FIT_SEEDS, source_per_fit_indices=[i for i, _ in group], metrics=metrics))
    ts = [r for r in rows if r["method"] == "TabSyn"]
    ts_groups = [r for r in per_fit if r["method"] == "TabSyn"]
    arf_physical = {r["metric_receipt_ref"]["sha256"] for r in rows if r["method"] == "ARF"}
    ts_physical = {r["metric_receipt_ref"]["sha256"] for r in ts}
    if len(arf_physical) != 5892 or len(ts_physical) != 95 or len(ts) != 95 or \
            len({r["dataset"] for r in ts}) != 41 or not {r["dataset"] for r in ts} <= arf_ids or \
            sum(r["complete_sample_group"] for r in ts_groups) != 19 or len(ts_groups) != 41:
        raise ValueError("frozen_followon_coverage_changed")
    return dict(format="arf-tabsyn-followon-validation-v1", rows=rows, per_fit=per_fit, summary=summary,
                arf_logical_cells=6000, arf_physical_metric_receipts=len(arf_physical), arf_alias_cells=108,
                arf_lineages=100, arf_fit_seeds=FIT_SEEDS, arf_common_n4n_five_fit_complete=True,
                tabsyn_new_cells=95, tabsyn_new_lineages=41, tabsyn_complete_n_groups=19, tabsyn_partial_n_groups=22,
                production_certified=False, current_bulk_model_hashes_rechecked=False, native_selection_changed=False,
                comparison_scope="training_derived_validation_only", **CLAIMS)


def build(root=ROOT):
    root = Path(root)
    if any(p.is_symlink() for p in (root / "inputs.lock.json", root, *root.parents)):
        raise ValueError("symlinked_scalar_export")
    lock = json.loads((root / "inputs.lock.json").read_bytes())
    if any(lock.get(k) != v or type(lock.get(k)) is not type(v) for k, v in CLAIMS.items()) or \
            lock["original_metric_receipts_verified_before_decode"] is not True or \
            lock["seed11_alias_fit_and_sample_receipts_reverified"] is not True or \
            any(type(lock[k]) is not int or lock[k] != 0 for k in ("new_fits", "new_samples", "new_auditor_evaluations")):
        raise ValueError("invalid_scalar_import_lock")
    if lock["scalar_export_ref"]["path"] != "scalar-cells.jsonl":
        raise ValueError("unexpected_scalar_export_path")
    raw = read_locked(root / "scalar-cells.jsonl", lock["scalar_export_ref"])
    rows = [json.loads(line) for line in raw.splitlines()]
    panel = reduce_rows(rows)
    panel["inputs"] = lock
    return panel


def render(panel):
    collections = ("rows", "per_fit", "summary")
    metadata = json.dumps({k: v for k, v in panel.items() if k not in collections}, sort_keys=True, indent=2, allow_nan=False)
    arrays = []
    for key in collections:
        records = ",\n".join("    " + json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) for r in panel[key])
        arrays.append('  "' + key + '": [\n' + records + '\n  ]')
    data = metadata[:-2] + ',\n' + ',\n'.join(arrays) + '\n}\n'
    output = io.StringIO(newline="")
    fields = list(IDENTITY) + list(METRICS) + ["metric_receipt_path", "metric_receipt_sha256"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in panel["rows"]:
        writer.writerow(dict(row, metric_receipt_path=row["metric_receipt_ref"]["path"],
                             metric_receipt_sha256=row["metric_receipt_ref"]["sha256"]))
    return {"panel.json": data.encode(), "figure-kpis.csv": output.getvalue().encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, data in render(build()).items():
        path = ROOT / name
        if args.check:
            if path.read_bytes() != data:
                raise SystemExit("followon_publication_drift: " + name)
        else:
            path.write_bytes(data)


if __name__ == "__main__":
    main()
