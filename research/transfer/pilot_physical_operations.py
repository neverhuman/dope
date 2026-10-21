"""Prospective physical suballocations of an existing development StageJournal.

No second journal, scheduler, permission grant, retry or refund is introduced.
The parent holds its original lock throughout. A reservation may issue a guard
only when freshly appended; reloading an interrupted journal never issues one.
Native fits/updates below are conservative reservations, not observed counts.
"""
from dataclasses import dataclass
import time

from .campaign_runner import CampaignRunnerError, _hash, _integer, _require
from .evidence import canonical_json, digest, parse_json

FORMAT = "dope-pilot-physical-operation-graph-v1"
MAX_NODES = 16384


@dataclass(frozen=True, repr=False)
class PhysicalOperationGraph:
    _encoded: bytes

    @classmethod
    def from_document(cls, value):
        try:
            encoded = canonical_json(value)
            value = parse_json(encoded)
        except (ValueError, TypeError):
            raise CampaignRunnerError("physical_graph_invalid") from None
        _require(type(value) is dict and set(value) == {
            "format", "learning_plan_sha256", "registry_sha256", "null_reference_sha256",
            "driver_plan_sha256", "campaign_plan_sha256", "cell_id", "operation_admission_sha256",
            "nodes", "aliases"}, "physical_graph_invalid")
        _require(value["format"] == FORMAT and len(encoded) <= 8 * 1024 * 1024,
                 "physical_graph_invalid")
        for name in ("learning_plan_sha256", "registry_sha256", "null_reference_sha256",
                     "campaign_plan_sha256", "cell_id", "operation_admission_sha256"):
            _require(_hash(value[name]), "physical_graph_invalid")
        _require(value["driver_plan_sha256"] is None or _hash(value["driver_plan_sha256"]),
                 "physical_graph_invalid")
        nodes, seen = value["nodes"], set()
        _require(type(nodes) is list and 0 < len(nodes) <= MAX_NODES, "physical_graph_invalid")
        for node in nodes:
            _require(type(node) is dict and set(node) == {
                "key", "kind", "parents", "wall_ms", "reserved_child_fits",
                "reserved_optimizer_updates"}, "physical_graph_invalid")
            _require(_hash(node["key"]) and node["key"] not in seen
                     and node["kind"] in ("prepare", "predict")
                     and type(node["parents"]) is list
                     and all(_hash(p) and p in seen for p in node["parents"])
                     and len(set(node["parents"])) == len(node["parents"])
                     and _integer(node["wall_ms"], 1)
                     and _integer(node["reserved_child_fits"])
                     and _integer(node["reserved_optimizer_updates"]), "physical_graph_invalid")
            seen.add(node["key"])
        aliases = value["aliases"]
        _require(type(aliases) is list and 0 < len(aliases) <= MAX_NODES,
                 "physical_alias_invalid")
        logical, referenced = set(), set()
        for alias in aliases:
            _require(type(alias) is dict and set(alias) == {"logical_key", "operations"}
                     and _hash(alias["logical_key"]) and alias["logical_key"] not in logical
                     and type(alias["operations"]) is list and bool(alias["operations"])
                     and all(_hash(k) and k in seen for k in alias["operations"])
                     and len(set(alias["operations"])) == len(alias["operations"]),
                     "physical_alias_invalid")
            logical.add(alias["logical_key"])
            referenced.update(alias["operations"])
        _require(referenced == seen, "physical_orphan_operation")
        return cls(encoded)

    def document(self):
        # Public dataclass construction is not a validation bypass.
        return parse_json(type(self).from_document(parse_json(self._encoded))._encoded)

    @property
    def sha256(self):
        return digest(self.document())


def accept_physical_event(record, item):
    """Validate on a copied StageJournal state, before any durable mutation."""
    value = record["data"]
    _require(type(value) is dict and item.get("pending")
             and type(value.get("attempt")) is int and value["attempt"] == item["attempts"],
             "physical_stage_not_pending")
    common = {"cell_id", "stage", "attempt"}
    if record["type"] == "physical_reserve":
        _require(set(value) == common | {"graph"} and "physical" not in item,
                 "physical_reservation_replay")
        graph = PhysicalOperationGraph.from_document(value["graph"])
        doc = graph.document()
        _require(doc["campaign_plan_sha256"] == record["plan_sha256"]
                 and doc["cell_id"] == value["cell_id"], "physical_campaign_binding_mismatch")
        _require(sum(n["wall_ms"] for n in doc["nodes"]) <= item["reservation"],
                 "physical_stage_budget_exceeded")
        item["physical"] = {"graph": doc, "reservation_sha256": record["sha256"],
                            "settled": False, "over_budget": False}
        return False
    if record["type"] == "physical_interrupt":
        _require(set(value) == common | {"interruption_receipt_sha256"}
                 and _hash(value["interruption_receipt_sha256"])
                 and "physical" in item and not item["physical"]["settled"],
                 "physical_interruption_invalid")
        # A crash does not prove that any node was unexecuted. Never convert an
        # absent observation into not_run, zero cost, a refund or a retry grant.
        outcomes = [{"key": n["key"], "status": "unknown", "wall_ms": None,
                     "observed_child_fits": None, "observed_optimizer_updates": None}
                    for n in item["physical"]["graph"]["nodes"]]
        item["physical"].update(settled=True, completed=False, violation=True,
                                interruption=True, outcomes=outcomes,
                                interruption_receipt_sha256=value["interruption_receipt_sha256"])
        return False
    _require(set(value) == common | {"outcomes", "violation"}
             and type(value["violation"]) is bool and "physical" in item
             and not item["physical"]["settled"], "physical_settlement_replay")
    nodes = {n["key"]: n for n in item["physical"]["graph"]["nodes"]}
    outcomes, observed = value["outcomes"], {}
    _require(type(outcomes) is list and len(outcomes) == len(nodes), "physical_outcomes_incomplete")
    over = False
    for row in outcomes:
        _require(type(row) is dict and set(row) == {
            "key", "status", "wall_ms", "observed_child_fits", "observed_optimizer_updates"}
            and _hash(row["key"]) and row["key"] in nodes and row["key"] not in observed
            and row["status"] in ("completed", "failed", "over_budget", "not_run"),
            "physical_outcome_invalid")
        for name in ("wall_ms", "observed_child_fits", "observed_optimizer_updates"):
            _require(row[name] is None or _integer(row[name]), "physical_outcome_invalid")
        if row["status"] == "not_run":
            _require(all(row[n] is None for n in (
                "wall_ms", "observed_child_fits", "observed_optimizer_updates")),
                "physical_outcome_invalid")
        else:
            _require(row["wall_ms"] is not None, "physical_outcome_invalid")
        node = nodes[row["key"]]
        exceeded = ((row["wall_ms"] or 0) > node["wall_ms"] or
                    (row["observed_child_fits"] or 0) > node["reserved_child_fits"] or
                    (row["observed_optimizer_updates"] or 0) > node["reserved_optimizer_updates"])
        _require(not exceeded or row["status"] == "over_budget", "physical_overrun_not_recorded")
        over = over or row["status"] == "over_budget"
        observed[row["key"]] = row
    for key, node in nodes.items():
        if observed[key]["status"] != "not_run":
            _require(all(observed[p]["status"] == "completed" for p in node["parents"]),
                     "physical_dependency_incomplete")
    item["physical"].update(settled=True, over_budget=over, violation=value["violation"],
                            completed=all(r["status"] == "completed" for r in outcomes), outcomes=outcomes)
    return over


class PhysicalOperationGuard:
    """An in-process executor consumes the parent's freshly reserved finite DAG.

    This is trusted Python control flow, not isolation from an arbitrary same-user
    process. Subprocess workers need an admitted parent transport; they must not
    reopen the parent's locked journal or reconstruct a guard from a crash log.
    """
    def __init__(self, graph, reservation_sha256, journal, cell_id, stage, *, _fresh_issue=None):
        _require(_fresh_issue is _ISSUE, "fresh_physical_reservation_required")
        self.graph = PhysicalOperationGraph.from_document(graph.document())
        self.reservation_sha256 = reservation_sha256
        self.journal, self.cell_id, self.stage = journal, cell_id, stage
        self.nodes = {n["key"]: n for n in self.graph.document()["nodes"]}
        self.outcomes, self.active = {}, None
        self.start_ns = time.monotonic_ns()
        self.wall_ns = sum(n["wall_ms"] for n in self.nodes.values()) * 1000000
        self.violation = False
        self.refusals = []  # Private opaque keys/static reasons; never native exception text.

    def begin(self, key, kind):
        item = self.journal.state.get((self.cell_id, self.stage), {})
        physical = item.get("physical", {})
        live = (self.journal.fd is not None and not self.journal.read_only and item.get("pending")
                and not physical.get("settled", True)
                and physical.get("reservation_sha256") == self.reservation_sha256
                and digest(physical["graph"]) == self.graph.sha256
                and self.nodes == {n["key"]: n for n in physical["graph"]["nodes"]})
        ready = (live and not self.violation and self.active is None and key in self.nodes
                 and key not in self.outcomes and self.nodes[key]["kind"] == kind
                 and all(self.outcomes.get(p, {}).get("status") == "completed"
                         for p in self.nodes[key]["parents"])
                 and time.monotonic_ns() - self.start_ns < self.wall_ns)
        if not ready:
            self.violation = True
            self.refusals.append({"key": key if _hash(key) else None, "kind": kind,
                                  "live_reservation": bool(live), "active_operation": self.active,
                                  "declared": key in self.nodes, "already_executed": key in self.outcomes,
                                  "parents_completed": key in self.nodes and all(
                                      self.outcomes.get(p, {}).get("status") == "completed"
                                      for p in self.nodes[key]["parents"]),
                                  "elapsed_wall_ms": (time.monotonic_ns() - self.start_ns) // 1000000})
            raise CampaignRunnerError("physical_operation_not_admitted")
        self.active = key

    def finish(self, key, status, wall_ns, output):
        _require(self.active == key, "physical_operation_not_active")
        wall_ms = (wall_ns + 999999) // 1000000
        # Existing Mitra attempts declare requested children; they do NOT measure
        # actual completed children/updates. Keep observations null until traced.
        attempts = output.get("native_attempts", ()) if type(output) is dict else ()
        requested = [a.get("requested_child_fits") for a in attempts if type(a) is dict
                     and "requested_child_fits" in a]
        invalid = any(not _integer(n) for n in requested)
        exceeded = (wall_ms > self.nodes[key]["wall_ms"] or invalid or
                    sum(n for n in requested if _integer(n)) > self.nodes[key]["reserved_child_fits"])
        self.outcomes[key] = {"key": key, "status": "over_budget" if exceeded else status,
                              "wall_ms": wall_ms, "observed_child_fits": None,
                              "observed_optimizer_updates": None}
        self.active = None
        if exceeded:
            self.violation = True
            raise CampaignRunnerError("physical_operation_over_budget")

    def settlement(self):
        _require(self.active is None, "physical_operation_still_active")
        return [{"key": key, "status": "not_run", "wall_ms": None,
                 "observed_child_fits": None, "observed_optimizer_updates": None}
                if key not in self.outcomes else dict(self.outcomes[key]) for key in self.nodes]


_ISSUE = object()


def issue_guard(graph, reservation_sha256, journal, cell_id, stage):
    # Called only by StageJournal.reserve_operations after a new fsynced event.
    _require(journal._fresh_physical_issue == reservation_sha256,
             "fresh_physical_reservation_required")
    journal._fresh_physical_issue = None
    return PhysicalOperationGuard(graph, reservation_sha256, journal, cell_id, stage, _fresh_issue=_ISSUE)


def reserve_learning_operations(execution, journal, cell_id, wall_by_role, *, driver_plan=None, labels=None):
    """Build exact functional keys without importing/calling a native model.

    Optional streamed permutation keys reuse the existing permutation routine.
    Wall allocations come from the actual admitted parent, not a new default.
    The pinned Mitra route reserves8 children and400 steps (8x50 cap) for EVERY
    new physical prediction. Exact-query driver baselines and repeated inputs
    alias the same fit/query/predict-seed key; they create no extra model call.
    Prior329 observed updates are not transferred to a new call. Other routes
    reserve at most one prepare.
    """
    from .pilot_learning_contract import ARMS, LEARNERS, SCALES, identity, require
    from .pilot_learning_runtime import registry_digest
    from .adapters.base import FeatureBatch
    from .drivers import permutation_importance
    from dataclasses import asdict
    require(execution.plan.sha256 == execution._plan_sha256 and
            registry_digest(execution.registry) == execution._registry_sha256,
            "admitted_inputs_changed")
    spec = journal.spec.document()
    cells = {c["cell_id"]: c for c in spec["cells"]}
    require(cell_id in cells and execution.plan.input_sha256 == cells[cell_id]["input_sha256"]
            and execution.plan.environment_sha256 == spec["environment_sha256"],
            "physical_campaign_input_environment_mismatch")
    require(set(wall_by_role) == {*LEARNERS, "null_reference"})
    require(all(type(v) is dict and set(v) == {"prepare", "predict"}
                and all(_integer(ms, 1) for ms in v.values()) for v in wall_by_role.values()))
    require(driver_plan is None or (labels is not None and
            labels.row_ids == execution.plan.query.row_ids and
            execution.plan.model_retention == "stream_drivers_then_release"))
    nodes, aliases, seen = [], [], {}
    def add(key, kind, parents, registration):
        mitra_predict = registration.role == "mitra" and kind == "predict"
        if not execution.fixture:
            require(registration.model.resources.wall_seconds * 1000 <= wall_by_role[registration.role][kind],
                    "native_timeout_exceeds_physical_allocation")
            if mitra_predict:
                require(registration.recipe.bag_folds == 8, "mitra_physical_recipe_mismatch")
        node = {"key": key, "kind": kind, "parents": parents,
                "wall_ms": wall_by_role[registration.role][kind],
                "reserved_child_fits": 8 if mitra_predict else int(kind == "prepare"),
                "reserved_optimizer_updates": 400 if mitra_predict else 0}
        if key in seen:
            require(seen[key] == node, "physical_alias_binding_changed")
        else:
            nodes.append(node)
            seen[key] = node
        return key
    plan = execution.plan
    for scale in SCALES:
        for learner in LEARNERS:
            for context in plan.contexts:
                for arm in ARMS:
                    registration = execution.registration_for(learner, arm)
                    _, _, inputs = execution.functional_inputs(registration, scale, arm, context)
                    fit = add(identity({"operation": "prepare", **inputs}), "prepare", [], registration)
                    predict = add(identity({"operation": "predict", "fit_key": fit,
                                           "query": plan.query.sha256, "seed": plan.predict_seed}),
                                  "predict", [fit], registration)
                    operations = [fit, predict]
                    if driver_plan is not None:
                        def record(columns):
                            features = FeatureBatch(plan.query.row_ids, tuple(tuple(r) for r in zip(*columns)),
                                                    plan.query.schema_sha256)
                            key = identity({"operation": "predict", "fit_key": fit,
                                            "query": features.sha256, "seed": plan.predict_seed})
                            if key != predict:
                                add(key, "predict", [fit, predict], registration)
                            if key not in operations:
                                operations.append(key)
                            return 0.0
                        permutation_importance(record, tuple(zip(*plan.query.rows)), dict(driver_plan.groups),
                                               learner_id=registration.role, query_population=plan.query.sha256,
                                               seed=driver_plan.native_seed, repeats=driver_plan.repeats)
                    aliases.append({"logical_key": identity({"scale": scale, "learner": learner,
                                    "context": context.name, "arm": arm}), "operations": operations})
    graph = PhysicalOperationGraph.from_document({"format": FORMAT, "learning_plan_sha256": plan.sha256,
        "registry_sha256": registry_digest(execution.registry), "null_reference_sha256": execution.null_reference.sha256,
        "campaign_plan_sha256": journal.spec.sha256, "cell_id": cell_id,
        "operation_admission_sha256": identity(asdict(execution.admission)),
        "driver_plan_sha256": None if driver_plan is None else driver_plan.sha256,
        "nodes": nodes, "aliases": aliases})
    guard = journal.reserve_operations(cell_id, "learners", graph)
    execution.evidence.guard = guard
    return guard
