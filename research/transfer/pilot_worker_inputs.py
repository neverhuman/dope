"""Private captured worker inputs; exact pins require trusted executor admission.

Contexts use owner-supplied row IDs/indices or a signed, precommitted per-pool
training-only policy. Query targets remain metric-side; they are never context
selection inputs. Policy-derived indices are recorded after exact draws exist.
"""
from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import math

from research.transfer.adapters.base import FeatureBatch, TrainingBatch
from research.transfer.campaign_artifacts import MAX_INPUT, read_regular
from research.transfer.evidence import parse_json
from research.transfer.pilot_cell_plan import PilotCellExecutionPlan
from research.transfer.pilot_learning_contract import (
    Context, LearningPlan, Pool, QueryLabels, SCALES, count, identity, pin, require,
)
from research.transfer.pilot_materialized_draws import load_pair
from research.transfer.tdl.target_last import schema_columns
from research.transfer.pilot_native_seed import NativeSeed

ROLES = ("fit", "selection", "query")


@dataclass(frozen=True)
class AssemblyBinding:
    cell_plan_sha256: str
    roles_sha256: str
    contexts_sha256: str
    draw_manifest_sha256: str
    source_sha256: str
    environment_sha256: str
    policy_sha256: str
    owner_recipe_decision_sha256: str
    native_seed_sha256: str
    learner_seed: str
    null_reference_sha256: str
    real_source_artifact_sha256: str
    real_source_seed: int | str
    real_seed_semantics: str
    owner_receipt_sha256: str
    resource_receipt_sha256: str

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.endswith("_sha256"):
                pin(value)
        require(NativeSeed(self.learner_seed).sha256 == self.native_seed_sha256,
                "assembly_learner_seed_identity_mismatch")
        require(self.real_seed_semantics in ("actual_source_stream", "deterministic_partition_zero"),
                "assembly_source_seed_unqualified")
        if self.real_seed_semantics == "deterministic_partition_zero":
            require(type(self.real_source_seed) is int and self.real_source_seed == 0,
                    "assembly_source_seed_unqualified")
        elif type(self.real_source_seed) is str:
            require(len(self.real_source_seed) == 18 and self.real_source_seed.startswith("0x")
                    and all(c in "0123456789abcdef" for c in self.real_source_seed[2:]),
                    "assembly_source_seed_invalid")
        else:
            count(self.real_source_seed)

    @property
    def sha256(self):
        return identity(asdict(self))


@dataclass(frozen=True, repr=False)
class Partition:
    features: FeatureBatch
    targets: tuple
    physical_unit_ids: tuple
    csv_sha256: str


def _captured(path, expected):
    pin(expected)
    raw = read_regular(path, MAX_INPUT)
    require(hashlib.sha256(raw).hexdigest() == expected, "assembly_input_changed")
    return raw


def _membership(raw, doc):
    value = parse_json(raw)
    require(type(value) is dict and set(value) == {"format", "schema_sha256", "partitions"}
            and value["format"] == "dope-pilot-private-physical-membership-v1"
            and value["schema_sha256"] == doc["schema_sha256"]
            and type(value["partitions"]) is dict and set(value["partitions"]) == set(ROLES),
            "assembly_membership_invalid")
    rows, all_ids, all_units = {}, set(), set()
    for role in ROLES:
        item = value["partitions"][role]
        csv_key = {"fit": "csv", "selection": "selection_csv", "query": "query_csv"}[role]
        require(type(item) is dict and set(item) == {"csv_sha256", "rows"}
                and item["csv_sha256"] == doc[csv_key + "_sha256"]
                and type(item["rows"]) is list and bool(item["rows"]),
                "assembly_membership_invalid")
        ids, units = [], []
        for row in item["rows"]:
            require(type(row) is dict and set(row) == {"row_id", "physical_unit_sha256"},
                    "assembly_membership_invalid")
            pin(row["row_id"])
            pin(row["physical_unit_sha256"])
            ids.append(row["row_id"])
            units.append(row["physical_unit_sha256"])
        require(len(ids) == len(set(ids)) and not set(ids) & all_ids
                and not set(units) & all_units, "assembly_partition_overlap")
        all_ids.update(ids)
        all_units.update(units)
        rows[role] = (tuple(ids), tuple(units))
    return rows


def _layout(raw):
    schema = parse_json(raw)
    require(type(schema) is dict and schema.get("format") == "dope-table-schema"
            and type(schema.get("version")) is int and schema["version"] == 1
            and type(schema.get("columns")) is list and bool(schema["columns"]),
            "assembly_schema_invalid")
    schema_columns(schema)  # reject noncanonical schemas before reading labels
    columns = schema["columns"]
    require(all(type(c) is dict and c.get("role") in ("feature", "target")
                and c.get("dtype") == "continuous" and type(c.get("source_name")) is str
                and bool(c["source_name"]) and type(c.get("id")) is str for c in columns),
            "assembly_numeric_schema_required")
    names = [c["source_name"] for c in columns]
    require(len(set(names)) == len(names) and len({c["id"] for c in columns}) == len(columns),
            "assembly_schema_invalid")
    target = [c["source_name"] for c in columns if c["role"] == "target"]
    features = [c["source_name"] for c in columns if c["role"] == "feature"]
    require(len(target) == 1 and bool(features), "assembly_regression_required")
    return names, features, target[0]


def _partition(raw, expected_rows, membership, layout, schema_sha256):
    names, features, target = layout
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        header = next(reader)
        require(len(header) == len(set(header)) and header == names and header[-1] == target,
                "assembly_header_mismatch")
        feature_indices, target_index = [header.index(n) for n in features], header.index(target)
        rows, targets = [], []
        for source in reader:
            require(len(source) == len(header) and len(rows) < expected_rows,
                    "assembly_row_count_mismatch")
            try:
                values = tuple(float(v) for v in source)
            except (ValueError, OverflowError):
                raise ValueError("assembly_numeric_invalid") from None
            require(all(math.isfinite(v) for v in values), "assembly_nonfinite_input")
            rows.append(tuple(values[i] for i in feature_indices))
            targets.append(values[target_index])
        require(len(rows) == expected_rows == len(membership[0]), "assembly_row_count_mismatch")
    except (UnicodeError, csv.Error, StopIteration, OverflowError):
        raise ValueError("assembly_csv_invalid") from None
    return Partition(FeatureBatch(membership[0], tuple(rows), schema_sha256), tuple(targets),
                     membership[1], hashlib.sha256(raw).hexdigest())


POOL_CONTEXT_FORMAT = "dope-pilot-private-pool-context-policy-v2"
POOL_POLICY_FORMAT = "dope-pilot-frozen-pool-context-policy-v1"


def pool_context_policy_sha256(name, rows, seed, selection_policy):
    """Identity of a rule frozen before any draws; not an admission decision."""
    require((name, selection_policy) in (("standard", "first_rows"),
                ("stratified_anchor", "fit_target_stratified")), "assembly_context_policy_invalid")
    count(rows, 1)
    require(type(seed) is str and len(seed) == 18 and seed.startswith("0x")
            and all(c in "0123456789abcdef" for c in seed[2:]), "assembly_context_seed_invalid")
    return identity({"format": POOL_POLICY_FORMAT, "name": name, "rows": rows,
                     "seed": seed, "selection_policy": selection_policy,
                     "seed_semantics": "identity_only_no_prng_target_rank_then_input_index"})


def _contexts(context_doc, pool_map):
    """Resolve fixed rules on each actual TRAINING pool, never real indices on synthetic rows.

    V1 explicit selections remain supported. V2 rule bytes and owner authority
    are frozen before generation; only their deterministic indices are resolved
    after the pool exists. Target ranks use training labels only. No query or
    selection labels are passed to this function.
    """
    if context_doc["format"] != POOL_CONTEXT_FORMAT:
        contexts = []
        indexed = context_doc["format"] == "dope-pilot-private-context-indices-v1"
        source_key = "indices" if indexed else "row_ids"
        for item in context_doc["contexts"]:
            require(type(item) is dict and set(item) == {"name", "rows", "seed", "policy_sha256", "selections"}
                    and type(item["selections"]) is list, "assembly_contexts_invalid")
            selections = []
            for selection in item["selections"]:
                require(type(selection) is dict and set(selection) == {"scale", "arm", source_key}
                        and type(selection[source_key]) is list, "assembly_contexts_invalid")
                values = selection[source_key]
                if indexed:
                    require((selection["scale"], selection["arm"]) in pool_map,
                            "assembly_contexts_invalid")
                    ids = pool_map[selection["scale"], selection["arm"]].batch.features.row_ids
                    require(all(type(i) is int and 0 <= i < len(ids) for i in values),
                            "assembly_context_index_invalid")
                    values = [ids[i] for i in values]
                selections.append((selection["scale"], selection["arm"], tuple(values)))
            contexts.append(Context(item["name"], item["rows"], item["seed"], item["policy_sha256"], tuple(selections)))
        return contexts, []
    contexts, evidence = [], []
    require(set(context_doc) == {"format", "contexts"} and type(context_doc["contexts"]) is list,
            "assembly_contexts_invalid")
    for item in context_doc["contexts"]:
        require(type(item) is dict and set(item) == {
            "name", "rows", "seed", "policy_sha256", "selection_policy"}, "assembly_contexts_invalid")
        expected = pool_context_policy_sha256(item["name"], item["rows"], item["seed"],
                                                item["selection_policy"])
        require(item["policy_sha256"] == expected, "assembly_context_policy_mismatch")
        selections = []
        for (scale, arm), pool in pool_map.items():
            batch = pool.batch
            ids, targets = batch.features.row_ids, batch.targets
            require(item["rows"] <= len(ids) == len(targets), "assembly_context_pool_too_small")
            require(all(type(v) in (int, float) and math.isfinite(v) for v in targets),
                    "assembly_context_target_invalid")
            if item["selection_policy"] == "first_rows":
                indices = list(range(item["rows"]))
            else:
                order = sorted(range(len(ids)), key=lambda i: (targets[i], i))
                indices = [order[(j * len(ids)) // item["rows"]] for j in range(item["rows"])]
            chosen = tuple(ids[i] for i in indices)
            selections.append((scale, arm, chosen))
            evidence.append({"context": item["name"], "scale": scale, "arm": arm,
                "policy_sha256": expected, "training_pool_sha256": identity(asdict(batch)),
                "selected_indices_sha256": identity(indices), "selected_row_ids_sha256": identity(chosen),
                "rows": len(chosen)})
        contexts.append(Context(item["name"], item["rows"], int(item["seed"], 16),
                                expected, tuple(selections)))
    return contexts, evidence


def assemble(cell_plan, binding, native_seed, paths, manifest_sha256, verify_assembly, *, control_runtime=None):
    """Capture exact fit/select/query roles; construct the existing typed plan.

    `paths` names private snapshots, plus exact owner-supplied membership/context
    locators. IDs are supplied before execution; never hash role labels to invent
    apparent disjointness. Physical-unit overlap, extra rows and unavailable
    numeric inputs refuse. Query targets stay separate from the learner plan.
    """
    require(type(cell_plan) is PilotCellExecutionPlan and type(binding) is AssemblyBinding
            and type(native_seed) is NativeSeed and callable(verify_assembly),
            "assembly_admission_required")
    # The verifier must authenticate actual role/source/rights/resource evidence.
    # Constructing this dataclass or matching its hashes does not grant admission.
    require(binding.cell_plan_sha256 == cell_plan.sha256
            and binding.native_seed_sha256 == native_seed.sha256
            and binding.learner_seed == native_seed.original_stream_hex
            and binding.draw_manifest_sha256 == manifest_sha256
            and verify_assembly(binding) is None, "assembly_not_admitted")
    doc = cell_plan.document()
    require(doc["version"] == 2 and set(paths) == {
        "csv", "selection_csv", "query_csv", "schema", "roles", "contexts", "artifact", "evidence"},
        "assembly_v2_paths_required")
    # The compiler stream remains in the cell plan; learner stream is signed
    # independently through the worker authority and AssemblyBinding.
    require(native_seed.original_stream_hex == binding.learner_seed,
            "assembly_learner_stream_mismatch")
    schema_raw = _captured(paths["schema"], doc["schema_sha256"])
    roles = _membership(_captured(paths["roles"], binding.roles_sha256), doc)
    tables = {}
    for role, key, count_key in (("fit", "csv", "fit_rows"),
                                 ("selection", "selection_csv", "selection_rows"),
                                 ("query", "query_csv", "query_rows")):
        tables[role] = _partition(_captured(paths[key], doc[key + "_sha256"]), doc[count_key],
                                  roles[role], _layout(schema_raw), doc["schema_sha256"])
    raw_contexts = _captured(paths["contexts"], binding.contexts_sha256)
    context_doc = parse_json(raw_contexts)
    require(type(context_doc) is dict and set(context_doc) == {"format", "contexts"}
            and context_doc["format"] in ("dope-pilot-private-contexts-v1",
                                          "dope-pilot-private-context-indices-v1",
                                          "dope-pilot-private-pool-context-policy-v2")
            and type(context_doc["contexts"]) is list, "assembly_contexts_invalid")
    draws = load_pair(paths["evidence"], manifest_sha256, artifact=paths["artifact"],
                      comparator=doc["comparator"], control_runtime=control_runtime)
    forbidden = tuple(row for role in ROLES for row in tables[role].features.row_ids)
    fit = tables["fit"]
    training = TrainingBatch(fit.features, fit.targets, "development_train")
    real = Pool(training, "fit_real", fit.csv_sha256, binding.real_source_artifact_sha256,
                len(fit.targets), len(fit.targets), binding.real_source_seed)
    null = Pool(training, "fit_null", fit.csv_sha256, binding.real_source_artifact_sha256,
                len(fit.targets), len(fit.targets), binding.real_source_seed)
    pools = []
    for scale, draw, seed in zip(SCALES, draws, (doc["draw_seed"], doc["draw_4n_seed"])):
        require(draw.spec.scale == scale and draw.spec.method == doc["comparator"]
                and draw.spec.input_sha256 == cell_plan.input_sha256
                and draw.spec.schema_sha256 == doc["schema_sha256"] and draw.spec.seed_hex == seed
                and draw.schema_document == schema_raw, "assembly_draw_binding_mismatch")
        pools.extend(((scale, "real", real), (scale, "null", null),
                      (scale, "synthetic", draw.pool(forbidden_row_ids=forbidden))))
    contexts, context_evidence = _contexts(context_doc, {(scale, arm): pool for scale, arm, pool in pools})
    query = tables["query"]
    plan = LearningPlan(doc["task_id"], doc["comparator"], doc["candidate"], draws[0].spec.artifact_sha256,
                        cell_plan.input_sha256, binding.source_sha256, binding.environment_sha256,
                        binding.policy_sha256, binding.owner_recipe_decision_sha256, native_seed.sha256,
                        doc["rows"], native_seed.seed32, native_seed.seed32, tuple(pools), tuple(contexts),
                        query.features, "stream_drivers_then_release", binding.null_reference_sha256, native_seed)
    labels = QueryLabels(query.features.row_ids, query.targets, query.csv_sha256)
    evidence = {"binding_sha256": binding.sha256, "cell_plan_sha256": cell_plan.sha256,
                "membership_sha256": binding.roles_sha256, "contexts_sha256": binding.contexts_sha256,
                "source_seed_semantics": binding.real_seed_semantics,
                "compiler_fit_seed": doc["fit_seed"], "learner_seed": binding.learner_seed,
                "native_seed_sha256": native_seed.sha256,
                "partition_rows": {r: len(tables[r].targets) for r in ROLES},
                "selection_row_ids_sha256": identity(list(tables["selection"].features.row_ids)),
                "query_labels_sha256": labels.sha256, "learning_plan_sha256": plan.sha256,
                "context_format": context_doc["format"], "context_policy_derivations": context_evidence,
                "scientific_admission": False, "pipeline_complete": False}
    return plan, labels, evidence
