"""Private development worker configuration; an authenticated executor grants use."""
from dataclasses import asdict, dataclass
from pathlib import Path

from .pilot_learning_contract import identity, pin, require
from .pilot_worker_inputs import AssemblyBinding
from .pilot_native_seed import NativeSeed, ORIGINAL_STREAM_HEX, OWNER_GRANT_SHA256


@dataclass(frozen=True)
class WorkerAuthority:
    input_sha256: str
    roles_path: str
    contexts_path: str
    roles_sha256: str
    contexts_sha256: str
    source_sha256: str
    environment_sha256: str
    policy_sha256: str
    owner_recipe_decision_sha256: str
    native_seed_sha256: str
    learner_seed: str  # explicit canonical uint64; never the compiler fit_seed by default
    null_reference_sha256: str
    real_source_artifact_sha256: str
    real_source_seed: int | str
    real_seed_semantics: str
    owner_receipt_sha256: str
    resource_receipt_sha256: str
    rights_admission_sha256: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.endswith("_sha256"):
                pin(value)
        require(all(type(p) is str and Path(p).is_absolute()
                    for p in (self.roles_path, self.contexts_path)), "worker_private_path_invalid")
        self.verify_learner_seed("fixture")  # representation/hash only; development is stricter
        # Validate source-seed vocabulary without treating prospective pins as admission.
        self.binding("0" * 64, "0" * 64)

    def verify_learner_seed(self, kind):
        require(kind in ("fixture", "development"), "worker_seed_scope_invalid")
        seed = NativeSeed(self.learner_seed)
        require(seed.sha256 == self.native_seed_sha256, "worker_learner_seed_identity_mismatch")
        if kind == "development":
            require(self.learner_seed == ORIGINAL_STREAM_HEX
                    and self.owner_recipe_decision_sha256 == OWNER_GRANT_SHA256,
                    "worker_owner_learner_stream_mismatch")
        return seed

    def binding(self, cell_plan_sha256, draw_manifest_sha256):
        fields = asdict(self)
        for name in ("input_sha256", "roles_path", "contexts_path", "rights_admission_sha256"):
            del fields[name]
        return AssemblyBinding(cell_plan_sha256=cell_plan_sha256,
                               draw_manifest_sha256=draw_manifest_sha256, **fields)

    def verify_parent(self, spec, cell_plan, authorization):
        doc = spec.document()
        require(self.input_sha256 == cell_plan.input_sha256
                and self.environment_sha256 == doc["environment_sha256"]
                and self.source_sha256 == identity({"source_commit": doc["source_commit"],
                                                    "source_tree": doc["source_tree"]})
                and self.rights_admission_sha256 == authorization["rights_admission_sha256"]
                and self.resource_receipt_sha256 == authorization["resource_lease_sha256"],
                "worker_parent_binding_mismatch")
