"""Versioned immutable pilot cell plans; declarations do not grant execution."""
from research.transfer.campaign_admission import (
    GATES, PilotContractError, PilotExecutionPlan, _identifier, inspect_admission_contract, seed_uint64,
)
from research.transfer.campaign_runner import CampaignRunSpec, CampaignRunnerError, STAGES, _command_path, _require
from research.transfer.evidence import canonical_json, digest, parse_json

FORMAT = "dope-pilot-cell-execution-plan"
COMPARATORS = ("dope", "gaussian_copula", "compressed_real_rows")
SCALES = ("primary_n", "sensitivity_4n")
LEARNERS = ("kumo_tabular", "tabicl", "mitra", "gradient_boosting", "neural")
CONTEXTS = ("standard", "stratified_anchor")
LEARNER_PATHS = ("real", "null", "synthetic")


def _hex(value, width, code):
    _require(type(value) is str and len(value) == width and all(c in "0123456789abcdef" for c in value), code)
    return value


def _positive(value, limit, code):
    _require(type(value) is int and 1 <= value <= limit, code)
    return value


def _seed(doc, key):
    return seed_uint64(doc[key]) if doc["version"] == 2 else doc[key]


class PilotCellExecutionPlan:
    """Immutable 8-stage cell execution plan bound to PilotExecutionPlan and CampaignRunSpec."""

    def __init__(self, document, encoded):
        self._document, self._encoded, self.sha256 = document, encoded, digest(document)
        keys = (
            "task_id", "task_role", "comparator", "candidate", "byte_budget",
            "csv_sha256", "selection_csv_sha256", "query_csv_sha256", "schema_sha256",
            "fit_seed", "draw_seed", "control_seed", "driver_seed",
            "fit_rows", "selection_rows", "query_rows", "rows",
        )
        if document["version"] == 2:
            keys += ("draw_4n_seed",)
        self.input_sha256 = digest({
            "format": FORMAT + "-input", "version": document["version"], **{k: document[k] for k in keys},
            "training_scales": list(document["training_scales"]),
            "learners": list(document["learners"]), "contexts": list(document["contexts"]),
        })

    @classmethod
    def from_document(cls, value):
        _require(type(value) is dict and type(value.get("version")) is int
                 and value["version"] in (1, 2), "pilot_cell_plan_invalid")
        extra = {"draw_4n_seed"} if value["version"] == 2 else set()
        _require(
            type(value) is dict and set(value) == {
                "format", "version", "cell_id", "task_id", "task_role", "comparator",
                "candidate", "binary", "binary_sha256", "worker_binary", "worker_binary_sha256",
                "csv", "csv_sha256", "selection_csv", "selection_csv_sha256",
                "query_csv", "query_csv_sha256", "schema", "schema_sha256",
                "fit_seed", "draw_seed", "control_seed", "driver_seed",
                "fit_rows", "selection_rows", "query_rows", "rows", "byte_budget",
                "deadline_seconds", "training_scales", "learners", "contexts",
                "learner_paths", "learner_rights_sha256", "stage_wall_ms",
            } | extra,
            "pilot_cell_plan_invalid",
        )
        _require(value["format"] == FORMAT, "pilot_cell_plan_invalid")
        # The real confined control route needs a separate typed runtime binding.
        # Preserve v1 decoding, but never imply its worker response is that route.
        _require(value["version"] == 1 or value["comparator"] == "dope",
                 "pilot_v2_control_runtime_unavailable")
        for key in (
            "cell_id", "task_id", "binary_sha256", "worker_binary_sha256",
            "csv_sha256", "selection_csv_sha256", "query_csv_sha256", "schema_sha256",
        ):
            _hex(value[key], 64, "pilot_cell_plan_invalid")
        _require(len({value["csv_sha256"], value["selection_csv_sha256"], value["query_csv_sha256"]}) == 3, "pilot_cell_plan_invalid")
        _require(value["task_role"] in ("observational", "adverse_mechanism"), "pilot_cell_plan_invalid")
        _require(value["comparator"] in COMPARATORS and _identifier(value["candidate"]), "pilot_cell_plan_invalid")
        if value["comparator"] != "dope":
            _require(value["candidate"] == value["comparator"], "pilot_cell_plan_invalid")
        path_keys = ("binary", "worker_binary", "csv", "selection_csv", "query_csv", "schema")
        for key in path_keys:
            _require(_command_path(value[key]), "pilot_cell_plan_invalid")
        _require(len({value[k] for k in path_keys}) == len(path_keys), "pilot_cell_plan_invalid")
        for key in ("fit_seed", "draw_seed", "control_seed", "driver_seed", *extra):
            if value["version"] == 2:
                try:
                    seed_uint64(value[key])
                except PilotContractError:
                    raise CampaignRunnerError("pilot_cell_plan_invalid") from None
            else:
                _require(type(value[key]) is int and 0 <= value[key] < (1 << 64), "pilot_cell_plan_invalid")
        if value["version"] == 2:
            _require(value["draw_seed"] != value["draw_4n_seed"], "pilot_draw_seeds_must_differ")
        for key in ("fit_rows", "selection_rows", "query_rows", "rows"):
            _positive(value[key], 1_000_000, "pilot_cell_plan_invalid")
        _require(value["fit_rows"] >= value["rows"], "pilot_cell_plan_invalid")
        _positive(value["byte_budget"], 1 << 30, "pilot_cell_plan_invalid")
        _positive(value["deadline_seconds"], 86_400, "pilot_cell_plan_invalid")
        _require(
            tuple(value["training_scales"]) == SCALES and tuple(value["learners"]) == LEARNERS
            and tuple(value["contexts"]) == CONTEXTS and tuple(value["learner_paths"]) == LEARNER_PATHS,
            "pilot_cell_plan_invalid",
        )
        rights, walls = value["learner_rights_sha256"], value["stage_wall_ms"]
        _require(type(rights) is dict and set(rights) == set(LEARNERS), "pilot_cell_plan_invalid")
        for learner in LEARNERS:
            _hex(rights[learner], 64, "pilot_cell_plan_invalid")
        _require(type(walls) is dict and set(walls) == set(STAGES), "pilot_cell_plan_invalid")
        for stage in STAGES:
            _positive(walls[stage], 86_400_000, "pilot_cell_plan_invalid")
        encoded = canonical_json(value)
        return cls(parse_json(encoded), encoded)

    def document(self):
        return parse_json(self._encoded)

    def verify_binding(self, spec: CampaignRunSpec, pilot_plan: PilotExecutionPlan, admission_contract: dict):
        _require(
            isinstance(spec, CampaignRunSpec) and isinstance(pilot_plan, PilotExecutionPlan) and type(admission_contract) is dict,
            "pilot_binding_mismatch",
        )
        plan_doc = pilot_plan.document()
        try:
            contract_status = inspect_admission_contract(
                pilot_plan, admission_contract, expected_pins=plan_doc["pins"], required_learners=list(LEARNERS),
            )
        except PilotContractError:
            raise CampaignRunnerError("pilot_binding_mismatch") from None
        _require(
            contract_status["status"] == "metadata_complete_requires_owner_verification"
            and not contract_status["missing_gates"]
            and all(admission_contract["gates"][g]["status"] == "accepted_for_operation" for g in GATES),
            "pilot_rights_or_qualification_unavailable",
        )
        spec_doc, pins = spec.document(), plan_doc["pins"]
        _require(
            spec_doc["phase"] == "development" and spec_doc["max_attempts"] == 1
            and spec_doc["source_commit"] == pins["source_commit"]
            and spec_doc["source_tree"] == pins["source_tree"]
            and spec_doc["environment_sha256"] == pins["environment_sha256"]
            and spec_doc["roster_sha256"] == pins["operation_dag_sha256"],
            "pilot_binding_mismatch",
        )
        ctx_names = tuple(c["context_id"] for c in plan_doc["contexts"])
        _require(
            tuple(plan_doc["methods"]) == COMPARATORS and tuple(plan_doc["sample_policies"]) == SCALES
            and tuple(plan_doc["learners"]) == LEARNERS and ctx_names == CONTEXTS
            and tuple(plan_doc["stages"]) == STAGES and self._document["byte_budget"] in plan_doc["byte_caps"],
            "pilot_binding_mismatch",
        )
        matched_tasks = [t for t in plan_doc["tasks"] if t["task_id"] == self._document["task_id"]]
        _require(len(matched_tasks) == 1, "pilot_binding_mismatch")
        task, streams = matched_tasks[0], plan_doc["seeds"]["substreams"]
        _require(
            task["role"] == self._document["task_role"]
            and task["fit_rows"] == self._document["fit_rows"]
            and task["primary_n"] == self._document["rows"]
            and seed_uint64(streams["fit"]) == _seed(self._document, "fit_seed")
            and seed_uint64(streams["sample_primary"]) == _seed(self._document, "draw_seed")
            and seed_uint64(streams["context"]) == _seed(self._document, "control_seed")
            and seed_uint64(streams["learner"]) == _seed(self._document, "driver_seed"),
            "pilot_binding_mismatch",
        )
        if self._document["version"] == 2:
            _require(seed_uint64(streams["sample_4n"]) == _seed(self._document, "draw_4n_seed"),
                     "pilot_binding_mismatch")
        cells = spec_doc["cells"]
        _require(len(cells) == 1, "pilot_binding_mismatch")
        cell = cells[0]
        _require(
            cell["cell_id"] == self._document["cell_id"]
            and cell["input_sha256"] == self.input_sha256
            and cell["command_sha256"] == self.sha256
            and sum(self._document["stage_wall_ms"].values()) <= spec_doc["budget"]["wall_ms"],
            "pilot_binding_mismatch",
        )


