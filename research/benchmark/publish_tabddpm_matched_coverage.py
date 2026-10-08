"""Join existing common metrics and account for missing TabDDPM fit seeds.

This is a reduction of pinned publications, not a fit or evaluation runner.
Conditional cell ETAs are kept outside the scientific paired measurements.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, median

from research.benchmark.publish_arf_tabsyn_followon import FIT_SEEDS, SAMPLE_SEEDS, METRICS, CLAIMS
from research.benchmark.publish_retained_metrics import flatten

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research/benchmark/results/tabddpm-matched-fit-coverage-v1"
SOURCES = {
    "tabddpm": ("research/benchmark/results/tabddpm-twelve-lineage-retained-validation/panel.json",
                "877317666864c712b84feed7170f9dccd3587780ae3b5ebbb6cecf4513adda13"),
    "followon": ("research/benchmark/results/arf-tabsyn-followon-validation-v1/panel.json",
                 "069eea5d428fe76e5fcb1c100845d067d3add4cfb2df2c53d08e1451fa29324a"),
    "tabsyn": ("research/benchmark/results/tabsyn-eight-lineage-validation/panel.json",
               "3e49c1f8f99615f21abee3504afb645e6b61eaab2d88be2e279be0c0c1b91d76"),
}
ANCHOR = "2026-10-08T08:00:00-06:00"
PAIRED_METRICS = ("catboost_retention", "marginal_error_mean", "c2st_catboost_auc", "distance_mia_auc")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_sources():
    docs = {}
    for key, (path, sha) in SOURCES.items():
        file = REPO / path
        require(not any(p.is_symlink() for p in (file, *file.parents)), "symlinked_publication")
        raw = file.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == sha, "frozen_publication_changed")
        docs[key] = json.loads(raw)
    return docs


def reference(key, pointer):
    path, sha = SOURCES[key]
    return dict(path=path, sha256=sha, pointer=pointer)


def split_hashes(cell):
    hashes = cell["input_hashes"]
    return {k: hashes.get(k, hashes.get(k + "_ref")) for k in ("train", "validation", "projection")}


def build(docs):
    td, follow, old = (docs[k] for k in ("tabddpm", "followon", "tabsyn"))
    for document in (td, follow, old):
        require(all(document.get(k) == v and type(document.get(k)) is type(v) for k, v in CLAIMS.items()),
                "invalid_validation_claim")
    protocols = [td["physical_panel"]["source_refs"], follow["inputs"]["metric_source_refs"], old["source_refs"]]
    require(all({k: v["sha256"] for k, v in p.items()} ==
                {k: v["sha256"] for k, v in protocols[0].items()} for p in protocols), "evaluator_source_changed")
    physical = {c["receipt_ref"]["sha256"]: c for c in td["physical_panel"]["cells"]}
    cells, splits = [], {}
    for i, row in enumerate(follow["rows"]):
        if row["method"] == "ARF":
            hashes = split_hashes(row)
            require(row["dataset"] not in splits or splits[row["dataset"]] == hashes, "reference_split_changed")
            splits[row["dataset"]] = hashes
        if row["fit_seed"] == 11:
            cells.append(dict(row, publication_ref=reference("followon", f"/rows/{i}")))
    require(len(splits) == 100, "reference_population_changed")
    for i, cell in enumerate(old["cells"]):
        cells.append(dict(flatten(cell), input_hashes=split_hashes(cell),
                          metric_receipt_ref=cell["receipt_ref"], publication_ref=reference("tabsyn", f"/cells/{i}")))
    for i, row in enumerate(td["rows"]):
        cell = physical[row["metric_receipt"]["sha256"]]
        require(all(row[k] == v for k, v in flatten(cell).items() if k != "selection_binding"),
                "tabddpm_scalar_changed")
        for detector in cell["metrics"]["detection"].values():
            require(detector["status"] != "ok" or (detector["folds"] == 5 and
                    len(detector["fold_aucs"]) == 5 and detector["duplicate_group_leakage"] is False),
                    "detection_protocol_changed")
        cells.append(dict(row, input_hashes=split_hashes(cell), metric_receipt_ref=row["metric_receipt"],
                          publication_ref=reference("tabddpm", f"/rows/{i}")))
    indexed = {}
    for row in cells:
        require(all(row[k] is None or type(row[k]) in (int, float) and math.isfinite(row[k]) for k in METRICS),
                "invalid_metric_scalar")
        require(type(row["fit_seed"]) is int and row["fit_seed"] == 11 and
                type(row["sample_seed"]) is int and row["sample_seed"] in SAMPLE_SEEDS and
                type(row["row_multiplier"]) is int and row["row_multiplier"] in (1, 4), "invalid_matched_identity")
        require(row["dataset"] in splits and split_hashes(row) == splits[row["dataset"]], "matched_partition_changed")
        key = tuple(row[k] for k in ("method", "selection_binding", "dataset", "row_multiplier", "sample_seed"))
        require(key not in indexed, "duplicate_matched_cell")
        indexed[key] = row
    matched, summaries = [], []
    for role in ("author_default", "native_selected"):
        for method, right_role in (("ARF", role), ("TabSyn", "scaled_author_default")):
            for n in (1, 4):
                groups = []
                for dataset in sorted(splits):
                    pairs = [(indexed.get(("TabDDPM", role, dataset, n, s)),
                              indexed.get((method, right_role, dataset, n, s))) for s in SAMPLE_SEEDS]
                    if not all(left and right for left, right in pairs):
                        continue
                    require(len({left["config_sha256"] for left, _ in pairs}) ==
                            len({right["config_sha256"] for _, right in pairs}) == 1, "mixed_sample_configuration")
                    for left, right in pairs:
                        for metric in ("null_loss", "catboost_trtr_loss", "linear_trtr_loss", "mlp_trtr_loss"):
                            require(left[metric] == right[metric], "real_control_changed")
                        matched.append(dict(dataset=dataset, fit_seed=11, row_multiplier=n,
                            sample_seed=left["sample_seed"], tabddpm_selection=role, reference_method=method,
                            reference_selection=right_role, input_hashes=splits[dataset],
                            tabddpm_publication=left["publication_ref"], reference_publication=right["publication_ref"],
                            tabddpm_receipt=left["metric_receipt_ref"], reference_receipt=right["metric_receipt_ref"]))
                    values = {metric: (mean(left[metric] for left, _ in pairs), mean(right[metric] for _, right in pairs))
                              if all(left[metric] is not None and right[metric] is not None for left, right in pairs)
                              else (None, None) for metric in PAIRED_METRICS}
                    groups.append((dataset, values))
                for metric in PAIRED_METRICS:
                    usable = [(d, v[metric]) for d, v in groups if v[metric][0] is not None]
                    summaries.append(dict(tabddpm_selection=role, reference_method=method, reference_selection=right_role,
                        fit_seed=11, row_multiplier=n, metric=metric, complete_sample_lineages=len(groups),
                        informative_lineages=len(usable), datasets=[d for d, _ in usable],
                        tabddpm_median=median(v[0] for _, v in usable) if usable else None,
                        reference_median=median(v[1] for _, v in usable) if usable else None,
                        median_paired_difference=median(v[0]-v[1] for _, v in usable) if usable else None,
                        confidence_interval=None, superiority=None))
    measured = {(r["dataset"], r["selection_binding"], r["fit_seed"], r["row_multiplier"], r["sample_seed"]): i
                for i, r in enumerate(td["rows"])}
    require(len(measured) == len(td["rows"]) == 132, "retained_coverage_changed")
    grid, minutes = [], 0
    for dataset in sorted(splits):
        # A conservative total native cap, including historical work. This does
        # not grant eight additional trials or declare any missing configuration.
        minutes += 80
        native_winner_minutes = minutes
        for role in ("author_default", "native_selected"):
            for seed in FIT_SEEDS:
                already = [k for k in measured if k[:3] == (dataset, role, seed)]
                require(not already or len(already) == 6, "partial_retained_fit_group")
                if not already and (role == "author_default" or seed != 11):
                    minutes += 10
                cell_minutes = native_winner_minutes if role == "native_selected" and seed == 11 else minutes
                for n in (1, 4):
                    for s in SAMPLE_SEEDS:
                        index = measured.get((dataset, role, seed, n, s))
                        grid.append(dict(dataset=dataset, method="TabDDPM", selection_binding=role,
                            fit_seed=seed, row_multiplier=n, sample_seed=s, input_hashes=splits[dataset],
                            status="measured" if index is not None else "pending_no_gpu_admission",
                            measurement=reference("tabddpm", f"/rows/{index}") if index is not None else None,
                            eligible_gpu_minutes_to_metrics=None if index is not None else cell_minutes+360,
                            conditional_eta_mdt=None if index is not None else
                            (datetime.fromisoformat(ANCHOR)+timedelta(minutes=cell_minutes+360)).isoformat()))
    return dict(format="tabddpm-matched-fit-coverage-v1", source_refs={k: reference(k, "") for k in SOURCES},
        validation_population=100, fit_seeds=FIT_SEEDS, sample_seeds=SAMPLE_SEEDS, row_multipliers=[1, 4],
        planned_logical_slots=6000, measured_logical_slots=132, pending_logical_slots=5868,
        coverage_by_fit_seed={str(s): sum(k[2] == s for k in measured) for s in FIT_SEEDS},
        physical_metric_receipts=126, retained_models=21, grid=grid, matched_cells=matched, paired_descriptive=summaries,
        detection_protocol=dict(source_sha256=protocols[0]["expanded_validation_metrics"]["sha256"],
            folds=5, splitter_seed=1729, splitter="StratifiedGroupKFold", identical_fold_membership_certified=False),
        planning=dict(anchor_mdt=ANCHOR, gpu_admitted=False, total_cap_minutes=minutes,
            provisional_evaluation_minutes=360, total_gpu_slot_hours=(minutes+360)/60,
            historical_trials_count_toward_eight_trial_cap=True, twelve_hour_cell_cap_unchanged=True,
            selected_missing_configuration_known=False, runnable_job_manifest=False),
        new_fits=0, new_samples=0, new_auditor_evaluations=0, production_certified=False,
        full_population_complete=False, five_fit_complete=False, **CLAIMS)


def render(panel):
    buf = io.StringIO(newline="")
    fields = ["dataset", "selection_binding", "fit_seed", "row_multiplier", "sample_seed", "status",
              "measurement_path", "measurement_sha256", "measurement_pointer",
              "eligible_gpu_minutes_to_metrics", "conditional_eta_mdt"]
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in panel["grid"]:
        ref = row["measurement"] or {}
        writer.writerow({k: ref.get(k.removeprefix("measurement_"), "") if k.startswith("measurement_")
                         else row.get(k) for k in fields})
    text = ["# Matched retained TabDDPM measurements", "",
            "Training-derived validation only. Average the three sample seeds within each lineage before",
            "forming a paired dataset summary. Fit seed 11 only; no confidence interval or superiority claim.", "",
            "| TabDDPM selection | Reference | Size | Metric | Supported lineages | TabDDPM median | Reference median | Median paired difference |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in panel["paired_descriptive"]:
        text.append("| "+" | ".join(str(row[k]) for k in ("tabddpm_selection", "reference_method", "row_multiplier",
            "metric", "informative_lineages", "tabddpm_median", "reference_median", "median_paired_difference"))+" |")
    return {"panel.json": json.dumps(panel, indent=2, sort_keys=True)+"\n", "cell-coverage.csv": buf.getvalue(),
            "matched-summary.md": "\n".join(text)+"\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = render(build(load_sources()))
    OUT.mkdir(parents=True, exist_ok=True)
    files = {name: text.encode() for name, text in outputs.items()}
    files.update({name: (OUT/name).read_bytes() for name in ("README.md", "panel.schema.json", "manifest.schema.json")})
    manifest = dict(format="tabddpm-matched-fit-coverage-manifest-v1",
        files={name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name, raw in files.items()},
        producer=dict(path="research/benchmark/publish_tabddpm_matched_coverage.py",
                      sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    outputs["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True)+"\n"
    for name, text in outputs.items():
        if args.check:
            require((OUT/name).read_text() == text, "committed_reduction_changed")
        else:
            (OUT/name).write_text(text)
    print("Matched TabDDPM receipt reduction verified; no fits or auditor evaluations.")


if __name__ == "__main__":
    main()
