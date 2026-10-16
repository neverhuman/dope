"""Render the frozen scalar-only, fit11 linear-regression comparison."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from statistics import mean, median

RESULTS = Path(__file__).with_name("results") / "matched-linear-fit11-retention-v1"
PINS = {
    "panel.json": (244428, "5914fb8aea11e8090cd7cac70a748bccb75cf7a9f1092c25616b78cd1c5bb048"),
    "panel.schema.json": (9830, "0e22eec6b6c0c2abb733cfc4807a6cd8591c4be27e89eae6fd9a7fd52fdea767"),
    "source-proof.json": (73187, "04c6881be665a049eb8d425f1b8c32408bfeb44557d38e5d72e127ee74f172c9"),
    "source-proof.schema.json": (8828, "6bbb1dc7d94531fab944fe2109ffa3f1564bcb6de3eec2065e756bc6cd595b33"),
}
SUMMARY_FIELDS = (
    "peer_method", "peer_configuration", "size_multiplier",
    "paired_complete_informative_lineages",
    "DOPE_median_of_same_cohort_three_sample_means",
    "peer_median_of_same_cohort_three_sample_means", "median_paired_difference",
)
COST_SCOPE = dict(physical_cost_estimated=False, logical_aliases_are_independent=False,
                  training_or_campaign_cost=None, wall_seconds=None, energy=None, hardware=None)
ARTIFACT_SCOPE = dict(comparison_unconstrained_by_stored_byte_cap=True,
                      stored_byte_threshold_bytes=10240, charged_bytes_are_model_plus_projection=True,
                      artifact_bytes_define_production_eligibility=False)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def strict_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(strict_equal(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_JSON_key")
        result[key] = value
    return result


def reject_constant(_value):
    raise ValueError("nonfinite_JSON_constant")


def read_public(path, expected_bytes, expected_sha256):
    """Authenticate immutable public bytes before JSON decoding."""
    require(type(expected_bytes) is int and 0 < expected_bytes <= 1048576, "invalid_source_length")
    require(type(expected_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha256), "invalid_source_pin")
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "source_alias_forbidden")
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "source_not_regular")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        opened = os.fstat(descriptor)
        require(key(opened) == key(before), "source_identity_changed")
        blocks, length = [], 0
        while True:
            block = os.read(descriptor, 65536)
            if not block:
                break
            length += len(block)
            require(length <= expected_bytes, "source_length_mismatch")
            blocks.append(block)
        raw = b"".join(blocks)
        require(key(os.fstat(descriptor)) == key(opened) == key(path.lstat()), "source_identity_changed")
    finally:
        os.close(descriptor)
    require(length == expected_bytes, "source_length_mismatch")
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "source_hash_mismatch")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def validate_shape(value, schema):
    """Validate the finite JSON-schema subset used by the two committed schemas."""
    if "const" in schema:
        require(strict_equal(value, schema["const"]), "schema_constant_mismatch")
    if "enum" in schema:
        require(any(type(value) is type(v) and value == v for v in schema["enum"]), "schema_enum_mismatch")
    kinds = schema.get("type", [])
    kinds = [kinds] if type(kinds) is str else kinds
    allowed = {
        "object": type(value) is dict, "array": type(value) is list,
        "string": type(value) is str, "integer": type(value) is int,
        "number": type(value) in (int, float) and math.isfinite(value),
        "boolean": type(value) is bool, "null": value is None,
    }
    require(not kinds or any(allowed.get(k, False) for k in kinds), "schema_type_mismatch")
    if type(value) is dict and schema.get("type") == "object":
        require(set(schema["required"]) <= set(value), "schema_required_field_missing")
        require(schema.get("additionalProperties", True) or set(value) <= set(schema["properties"]), "schema_extra_field")
        for name, member in value.items():
            validate_shape(member, schema["properties"][name])
    if type(value) is list:
        require(schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)), "schema_array_length")
        for member in value:
            validate_shape(member, schema.get("items", {}))
    if "pattern" in schema:
        require(type(value) is str and re.fullmatch(schema["pattern"], value), "schema_pattern_mismatch")
    if "minimum" in schema:
        require(value >= schema["minimum"], "schema_minimum_mismatch")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_source_proof(panel, proof):
    require(proof["format"] == "measured_descriptive_validation_source_proof" and type(proof["version"]) is int and proof["version"] == 1, "source_proof_format")
    require(proof["hash_before_metadata_decode"] is True and proof["projection_task_only_extraction"] is True, "source_custody_scope")
    require(strict_equal(proof["panel_ref"], dict(path="panel.json", bytes=PINS["panel.json"][0], sha256=PINS["panel.json"][1])), "panel_source_binding")
    for key in ("raw_rows_model_weights_samples_TEST_or_headers_emitted", "new_fit_sample_evaluation_or_provider_initialization", "MFS_PTF_release_superiority_claimed"):
        require(proof[key] is False, "source_scope_upgrade")
    refs = proof["projection_task_refs"]
    require(len(refs) == len({r["dataset"] for r in refs}) == 23, "projection_task_coverage")
    tasks = {r["dataset"]: r for r in refs}
    for cell in panel["cells"]:
        row = tasks[cell["dataset"]]
        require(row["task"] == "regression" and row["sha256"] == cell["projection_sha256"], "projection_task_binding")
    metrics = proof["metric_refs"]
    require(len(metrics) == len({r["sha256"] for r in metrics}), "duplicate_metric_source")
    expected = {c[k] for c in panel["cells"] for k in ("DOPE_metric_sha256", "peer_metric_sha256")}
    require({r["sha256"] for r in metrics} == expected, "metric_source_coverage")
    sources = {r["name"]: r["sha256"] for r in proof["source_code_refs"]}
    require(len(proof["source_code_refs"]) == len(sources) == 3, "duplicate_utility_source")
    require(sources == {
        "pilot_metrics": "6c03892685df856f7a491ca79acb1c28fe8270ead8bbfea1ba50228cb6f0e7e3",
        "original_DOPE_metric_entry": "2340d48b755299165b999bdaaa9a73329b256c53d8c7342a1ff45f087296607e",
        "retained_metric_driver": "553750022364da3c19410887b96b71d4a235bf0cb22cebb2b98df4a76f9a6b1e",
    }, "utility_source_binding")
    runtime = proof["declared_relevant_runtime"]
    require(runtime["numerical_package_refs_exact"] == 1347 and runtime["stdlib_source_or_extension_refs_exact"] == 558, "declared_runtime_scope")
    require(runtime["whole_runtime_equivalence_claimed"] is False and runtime["historical_full_runtime_closure_upgraded"] is False, "runtime_scope_upgrade")
    require(len(runtime["additional_retained_refs"]) == 42 and len(runtime["required_system_provider_refs"]) == 8, "provider_scope")
    require(dict(Counter(r["role"] for r in runtime["additional_retained_refs"])) == dict(native_provider=40, stdlib_source_or_extension=1, elf_loader=1), "additional_provider_roles")


def recompute(panel):
    require(panel["format"] == panel["status"] == "measured_descriptive_validation", "panel_format")
    require(type(panel["version"]) is int and panel["version"] == 1, "panel_version")
    require(panel["DOPE_configuration"] == "features12_steps2048" and type(panel["fit_seed"]) is int and panel["fit_seed"] == 11, "fit_configuration_scope")
    require(panel["sample_seeds"] == [101, 211, 307] and all(type(v) is int for v in panel["sample_seeds"]), "sample_seed_scope")
    require(panel["size_multipliers"] == [1, 4] and all(type(v) is int for v in panel["size_multipliers"]), "sample_size_scope")
    require(strict_equal(panel["cost_scope"], COST_SCOPE) and strict_equal(panel["artifact_scope"], ARTIFACT_SCOPE), "cost_or_artifact_scope")
    require(panel["gated_scores"] == dict(mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None), "gated_score_not_null")
    require(panel["between_fit_aggregation"] is None and panel["effective_auditor_seed"] is None, "fit_or_seed_pooling")
    for key in ("official_tests_opened", "new_fit_sample_or_evaluation", "five_fit_variance_claim", "counts_as_DOPE_win", "production_certified", "historical_full_runtime_closure_upgraded"):
        require(panel[key] is False, "claim_upgrade")
    require(panel["within_lineage_aggregation"] == "arithmetic_mean_of_exact_three_sample_retention_values_for_both_methods" and panel["across_lineage_aggregation"] == "median_on_same_complete_informative_paired_cohort", "aggregation_scope")
    seen, groups, peer_aliases, dope_aliases = set(), {}, {}, {}
    for cell in panel["cells"]:
        require(cell["peer_method"] in ("TabDDPM", "Forest-Flow") and cell["peer_configuration"] in ("author_default", "native_selected"), "comparator_scope")
        require(cell["peer_method"] != "Forest-Flow" or cell["peer_configuration"] == "author_default", "Forest_selection_scope")
        require(type(cell["fit_seed"]) is int and cell["fit_seed"] == 11 and cell["task"] == "regression" and cell["auditor"] == "linear" and cell["effective_auditor_seed"] is None, "mixed_auditor_task_fit")
        require(type(cell["sample_seed"]) is int and cell["sample_seed"] in (101, 211, 307) and type(cell["size_multiplier"]) is int and cell["size_multiplier"] in (1, 4), "sample_identity")
        key = tuple(cell[k] for k in ("peer_method", "peer_configuration", "dataset", "size_multiplier", "sample_seed"))
        require(key not in seen, "duplicate_cell")
        seen.add(key)
        for name in ("DOPE_charged_artifact_bytes", "peer_charged_artifact_bytes"):
            require(type(cell[name]) is int and cell[name] > 0, "invalid_artifact_charge")
        nl, real = cell["null_loss"], cell["real_validation_loss"]
        require(finite(nl) and finite(real) and nl >= 0 and real >= 0, "nonfinite_control")
        require(type(cell["informative"]) is bool and cell["informative"] == (nl - real >= .01 * abs(nl)), "informative_control")
        for name in ("DOPE_retention", "peer_retention"):
            require((finite(cell[name]) and nl > real) if cell["informative"] else cell[name] is None, "retention_applicability")
        common = tuple(cell[k] for k in ("dataset", "fit_seed", "sample_seed", "size_multiplier", "task", "null_loss", "real_validation_loss", "informative", "TRAIN_sha256", "VAL_sha256", "projection_sha256"))
        for aliases, pin, value, charge in ((peer_aliases, "peer_metric_sha256", "peer_retention", "peer_charged_artifact_bytes"), (dope_aliases, "DOPE_metric_sha256", "DOPE_retention", "DOPE_charged_artifact_bytes")):
            payload = (common, cell[value], cell[charge])
            require(cell[pin] not in aliases or aliases[cell[pin]] == payload, "inconsistent_physical_alias")
            aliases[cell[pin]] = payload
        groups.setdefault(key[:-1], []).append(cell)
    paired, excluded = [], []
    for (method, role, dataset, size), cells in sorted(groups.items()):
        require(len(cells) == 3 and sorted(c["sample_seed"] for c in cells) == [101, 211, 307], "incomplete_sample_group")
        if all(c["informative"] for c in cells):
            paired.append(dict(peer_method=method, peer_configuration=role, dataset=dataset, size_multiplier=size,
                               DOPE_three_sample_mean=mean(c["DOPE_retention"] for c in cells),
                               peer_three_sample_mean=mean(c["peer_retention"] for c in cells)))
        else:
            excluded.append(dict(peer_method=method, peer_configuration=role, dataset=dataset, size_multiplier=size,
                                 reason="uninformative_linear_real_data_control"))
    summary = []
    for method, role, size in sorted({(r["peer_method"], r["peer_configuration"], r["size_multiplier"]) for r in paired}):
        rows = [r for r in paired if (r["peer_method"], r["peer_configuration"], r["size_multiplier"]) == (method, role, size)]
        summary.append(dict(peer_method=method, peer_configuration=role, size_multiplier=size,
                            paired_complete_informative_lineages=len(rows),
                            DOPE_median_of_same_cohort_three_sample_means=median(r["DOPE_three_sample_mean"] for r in rows),
                            peer_median_of_same_cohort_three_sample_means=median(r["peer_three_sample_mean"] for r in rows),
                            median_paired_difference=median(r["DOPE_three_sample_mean"] - r["peer_three_sample_mean"] for r in rows)))
    coverage = dict(logical_comparator_cells=len(seen), distinct_peer_metric_receipts=len(peer_aliases),
                    distinct_DOPE_metric_receipts=len(dope_aliases), projection_tasks_authenticated_as_regression=len({c["dataset"] for c in panel["cells"]}), CatBoost_or_MLP_cells_pooled=0)
    require(strict_equal(coverage, panel["coverage"]) and coverage == dict(logical_comparator_cells=210, distinct_peer_metric_receipts=204, distinct_DOPE_metric_receipts=138, projection_tasks_authenticated_as_regression=23, CatBoost_or_MLP_cells_pooled=0), "coverage_mismatch")
    require(paired == panel["paired_lineage_groups"] and excluded == panel["exclusions"] and summary == panel["summary"], "stored_aggregate_mismatch")
    return summary


def table(summary):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
    return stream.getvalue().encode()


def markdown(summary):
    lines = [
        "# Matched fit11 linear-regression retention", "",
        "Measured descriptive validation on the same official-training-derived inputs. DOPE uses fixed `features12_steps2048`; TabDDPM retains its original author-default/native-selected roles; Forest retains author-default fit11. Official tests remain sealed.", "",
        "Only the shared linear **regression** auditor is compared. All 23 projection task fields are hash-authenticated as regression. The shared pilot implementation constructs `LinearRegression()` without a seed. DOPE supplied 101/211/307 and the retained evaluator supplied 1729; CatBoost and MLP consume these different seeds and are excluded. Other pilot/A952 diagnostics are not pooled.", "",
        "Each method first averages its three sample-seed retentions (101/211/307) within each lineage and size. Dataset medians and paired differences use the exact same complete informative cohort on both sides. This is one fit seed, with no fit-variance or superiority claim.", "",
        "| Comparator | Configuration | Size | Paired lineages | DOPE median | Comparator median | Median paired difference |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['peer_method']} | {row['peer_configuration']} | {row['size_multiplier']}n | {row['paired_complete_informative_lineages']} | {row['DOPE_median_of_same_cohort_three_sample_means']:.6f} | {row['peer_median_of_same_cohort_three_sample_means']:.6f} | {row['median_paired_difference']:.6f} |")
    lines += ["",
        "The difference between two aggregate medians need not equal the median of paired lineage differences. For example, Forest 4n has aggregate medians 0.995956 and 0.995869 while its median paired difference is -0.004079. Pairing occurs before taking the difference median.", "",
        "Cohorts differ across comparator rows. Two Forest lineages (`19f4780b53b3fa41`, `a04f964bc53281e0`) have uninformative linear real-data controls and are excluded at both sizes. Their null outcomes remain in `panel.json`; neither is treated as a win or imputed result. Forest's other fit seeds are not pooled here.", "",
        "The comparison is unconstrained by the 10,240-byte stored-artifact threshold. The original [TabDDPM panel](../tabddpm-twelve-lineage-retained-validation/README.md) records all 21 retained models above that threshold. The Forest [first two](../forest-first-two-fivefit-v1/README.md), [next six](../forest-next-six-fivefit-v1/README.md) and [third five](../forest-third-five-fivefit-v1/README.md) cohorts likewise record all 65 models above it. Per-cell model-plus-projection byte charges remain in this panel. A stored-byte charge does not establish production eligibility, quality, privacy, or a DOPE win.", "",
        "There are 210 logical comparator cells, 204 distinct comparator metric receipts and 138 distinct DOPE metric receipts. Default/native aliases and DOPE reuse are not independent repetitions or multiplied costs. No training/campaign cost, hardware, energy or parent clock is inferred by this panel.", "",
        "The source proof binds the committed input panels, original metric references, the shared pilot and both drivers, 23 task-only projection checks, and relevant declared numerical-provider metadata. It retains all 42 additional comparator runtime references (40 native providers, one stdlib source/extension and one loader). Whole-runtime equivalence and historical runtime certification are not claimed.", "",
        "`MFS-v2`, `PTF-v1`, release-safe L3 and superiority are null. No rows, headers, transforms, model weights, sample payloads or TEST values are published; no new fit, generation or evaluation was performed.", "",
        "## Reproduction", "",
        "Authenticate the committed JSON/schema bytes, recompute every three-sample mean and same-cohort median, and check `figure-kpis.csv` plus this README byte for byte:", "",
        "```sh", "python3 -B -m research.benchmark.render_matched_linear_fit11 --check", "```", "",
        "The command uses only the standard library and committed public files. `--write` regenerates the two rendered files. The private source receipts are needed only to reproduce the original metadata projection, not to check these committed scalar aggregates.", "",
    ]
    return "\n".join(lines).encode()


def load(results):
    values = {name: read_public(Path(results) / name, *pin) for name, pin in PINS.items()}
    validate_shape(values["panel.json"], values["panel.schema.json"])
    validate_shape(values["source-proof.json"], values["source-proof.schema.json"])
    verify_source_proof(values["panel.json"], values["source-proof.json"])
    return values["panel.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    panel = load(args.results_dir)
    summary = recompute(panel)
    outputs = {"figure-kpis.csv": table(summary), "README.md": markdown(summary)}
    load(args.results_dir)
    for name, raw in outputs.items():
        path = args.results_dir / name
        if args.check:
            require(not path.is_symlink() and path.read_bytes() == raw, "rendered_leaf_mismatch")
        else:
            require(not path.is_symlink(), "rendered_leaf_alias")
            path.write_bytes(raw)
    print("PASS: 210 scalar cells; six same-cohort summaries; CSV/Markdown exact")


if __name__ == "__main__":
    main()
