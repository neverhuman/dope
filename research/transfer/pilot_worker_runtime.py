"""Trusted parent pilot runtime: actual learners, streamed drivers and private losses.

The signed worker-binary field pins a configuration capsule for this in-process
route. The capsule binds code, inputs, registry and physical allocations. Required
executor callbacks authenticate current rights/owner/resource evidence. Neither
configuration hashes nor fixture execution establish scientific admission.
"""
from dataclasses import asdict
from pathlib import Path
import resource
import sys
import time

from .campaign_artifacts import create_private, file_hash, read_regular
from .evidence import canonical_json
from .pilot_learning_contract import DriverPlan, LEARNERS, identity, require
from .pilot_learning_execution import LearningExecution
from .pilot_learning_reports import execute_streamed, publish_private, reduce_learning
from .pilot_learning_runtime import OperationAdmission, registry_digest
from .pilot_native_seed import NativeSeed
from .pilot_native_receipt import NativeReceiptBridge
from .pilot_physical_operations import reserve_learning_operations
from .pilot_worker_authority import WorkerAuthority
from .pilot_worker_inputs import assemble

# The signed capsule captures this repository closure. SDK distributions, weights
# and actual adapter modules additionally use Registration's existing verification.
MODULES = ("pilot_worker_runtime.py", "pilot_worker_authority.py", "pilot_worker_inputs.py",
           "pilot_runner.py", "pilot_native_receipt.py", "pilot_cell_plan.py", "pilot_materialized_draws.py",
           "pilot_native_seed.py", "pilot_learning_contract.py", "pilot_learning_execution.py",
           "pilot_learning_runtime.py", "pilot_learning_reports.py", "pilot_physical_operations.py",
           "campaign_runner.py", "campaign_artifacts.py", "campaign_native.py", "evidence.py",
           "drivers.py", "adapters/base.py", "adapters/auditors/conventional.py",
           "pilot_constant_null.py", "tdl/target_last.py", "tdl/transfer.py", "tdl/drivers.py")


class PilotWorkerRuntime:
    def __init__(self, authority, registry, null_reference, driver_plan, wall_by_role, *,
                 verify_assembly, verify_operation, kind="development", control_runtime=None,
                 adapter_lifetime_policy=None, native_receipt=None):
        require(type(authority) is WorkerAuthority and type(driver_plan) is DriverPlan
                and kind in ("fixture", "development")
                and callable(verify_assembly) and callable(verify_operation), "worker_authority_required")
        if control_runtime is not None:
            from .pilot_control_runtime import PilotControlRuntime
            require(type(control_runtime) is PilotControlRuntime, "worker_control_runtime_invalid")
        from .pilot_physical_operations import LIFETIME_POLICY
        require(adapter_lifetime_policy in (None, LIFETIME_POLICY), "physical_lifetime_policy_invalid")
        self.adapter_lifetime_policy = adapter_lifetime_policy
        self.control_runtime = control_runtime
        authority.verify_learner_seed(kind)
        require(native_receipt is None or type(native_receipt) is NativeReceiptBridge,
                "native_receipt_bridge_invalid")
        require(kind == "fixture" or native_receipt is not None, "native_receipt_bridge_required")
        self.native_receipt, self.native_result = native_receipt, None
        self.authority, self.registry, self.null_reference = authority, dict(registry), null_reference
        self.driver_plan, self.wall_by_role, self.kind = driver_plan, wall_by_role, kind
        self.verify_assembly, self.verify_operation = verify_assembly, verify_operation
        self._summaries, self._execution = {}, None
        self._bound = False
        self._capsule = self.capsule()

    def capsule(self):
        root = Path(__file__).parent
        capsule = {"format": "dope-trusted-parent-pilot-worker-v1", "kind": self.kind,
            "authority": asdict(self.authority), "registry_sha256": registry_digest(self.registry),
            "null_reference_sha256": self.null_reference.sha256,
            "driver_plan": {"groups": [[n, list(v)] for n, v in self.driver_plan.groups],
                            "seed": self.driver_plan.seed, "repeats": self.driver_plan.repeats,
                            "protocol_sha256": self.driver_plan.protocol_sha256},
            "wall_by_role": self.wall_by_role,
            "repository_modules": {name: file_hash(root / name) for name in MODULES}}
        if self.native_receipt is not None:
            capsule["native_receipt"] = self.native_receipt.capsule()
        if self.adapter_lifetime_policy is not None:
            capsule.update(format="dope-trusted-parent-pilot-worker-v2",
                           adapter_lifetime_policy=self.adapter_lifetime_policy)
        if self.control_runtime is not None:
            from .pilot_control_runtime import MODULES as CONTROL_MODULES
            capsule["control_runtime"] = self.control_runtime.capsule()
            capsule["repository_modules"].update({name: file_hash(root / name) for name in CONTROL_MODULES})
        return canonical_json(capsule)

    @property
    def development_admission_used(self):
        return (self.kind == "development" and self._execution is not None
                and self._execution.admission.kind == "development"
                and self._execution.fixture is False)

    def bind(self, spec, cell_plan, auth, paths):
        doc = cell_plan.document()
        require(not self._bound and self.capsule() == self._capsule
                and read_regular(paths["worker"]) == self._capsule
                and file_hash(paths["worker"]) == doc["worker_binary_sha256"]
                and doc["version"] == 2 and self.driver_plan.seed == doc["driver_seed"],
                "worker_capsule_mismatch")
        self.authority.verify_learner_seed(self.kind)
        self.authority.verify_parent(spec, cell_plan, auth)
        require(all(self.registry[role].admission.eligibility_sha256 ==
                    doc["learner_rights_sha256"][role] for role in LEARNERS),
                "worker_learner_rights_mismatch")
        if doc["comparator"] == "dope":
            require(self.control_runtime is None, "worker_control_runtime_mismatch")
        else:
            require(self.control_runtime is not None, "worker_control_runtime_required")
            self.control_runtime.bind(spec, cell_plan, auth, paths)
        if self.native_receipt is not None:
            self.native_receipt.bind(self.authority, cell_plan)
        self._bound = True

    def _learners(self, journal, cell_plan, paths, root, manifest_sha256):
        require(self._execution is None and manifest_sha256 is not None, "worker_replay_refused")
        doc, authority = cell_plan.document(), self.authority
        authority.verify_learner_seed(self.kind)
        captured = {key: paths[key] for key in ("csv", "selection_csv", "query_csv", "schema", "artifact")}
        captured.update(roles=Path(authority.roles_path), contexts=Path(authority.contexts_path), evidence=root)
        plan, labels, assembly = assemble(cell_plan, authority.binding(cell_plan.sha256, manifest_sha256),
            NativeSeed(authority.learner_seed), captured, manifest_sha256, self.verify_assembly,
            control_runtime=self.control_runtime)
        admission = OperationAdmission(self.kind, plan.sha256, registry_digest(self.registry),
            authority.owner_receipt_sha256, authority.resource_receipt_sha256, LEARNERS,
            self.null_reference.sha256)
        execution = LearningExecution(plan, self.registry, self.null_reference, admission,
                                      self.verify_operation, root)
        self._execution = execution
        guard = reserve_learning_operations(execution, journal, doc["cell_id"], self.wall_by_role,
                                             driver_plan=self.driver_plan, labels=labels,
                                             adapter_lifetime_policy=self.adapter_lifetime_policy)
        try:
            execution, drivers = execute_streamed(execution, labels, self.driver_plan)
            paired = reduce_learning(execution, labels, {})  # no invented inference/ceiling
        finally:
            # Also settle attempted failures and explicit not-run slots; never refund them.
            journal.settle_operations(doc["cell_id"], "learners", guard)
        learners, driver_summary = publish_private(execution, paired, drivers, root)
        create_private(root / "private-input-assembly.json", canonical_json(assembly))
        learners.update(materialized_draw_manifest_sha256=manifest_sha256,
                        selection_csv_sha256=doc["selection_csv_sha256"],
                        query_csv_sha256=doc["query_csv_sha256"],
                        physical_graph_sha256=guard.graph.sha256,
                        physical_reservation_sha256=guard.reservation_sha256)
        driver_summary.update(seed=doc["driver_seed"], query_csv_sha256=doc["query_csv_sha256"],
                              physical_cost_stage="learners", no_additional_model_calls=True)
        self._summaries.update(learners=learners, drivers=driver_summary)
        return learners, file_hash(paths["learners"])

    def stage(self, stage, journal, cell_plan, paths, root, artifact_sha256, manifest_sha256):
        """Called only inside the already reserved parent stage, once per stage."""
        start = time.monotonic_ns()
        doc, status, payload, output_sha256 = cell_plan.document(), "failed", None, None
        try:
            require(self._bound and stage not in self._summaries.get("served", ())
                    and self.capsule() == self._capsule, "worker_state_changed_or_replayed")
            self._summaries.setdefault("served", []).append(stage)
            require(file_hash(paths["artifact"]) == artifact_sha256, "worker_artifact_changed")
            require(all(file_hash(paths[key]) == doc[key + "_sha256"]
                        for key in ("csv", "schema", "selection_csv", "query_csv")),
                    "worker_private_input_changed")
            if stage == "learners":
                payload, output_sha256 = self._learners(journal, cell_plan, paths, root, manifest_sha256)
                require(payload["status"] == "completed", "worker_learning_incomplete")
            elif stage == "drivers":
                payload = self._summaries["drivers"]
                require(payload["status"] == "executed_unqualified", "worker_drivers_incomplete")
                output_sha256 = file_hash(paths["drivers"])
            elif stage == "receipt":
                if self.native_receipt is not None:
                    self.native_result = self.native_receipt.verify(
                        self._execution, journal, cell_plan, paths, root)
                require(self.kind == "fixture" or self.native_result is not None,
                        "native_receipt_verification_required")
                payload = {"artifact_sha256": artifact_sha256, "comparator": doc["comparator"],
                           "byte_budget": doc["byte_budget"],
                           "learner_evidence_sha256": file_hash(paths["learners"]),
                           "driver_evidence_sha256": file_hash(paths["drivers"]),
                           "physical_cost_stage": "learners", "fixture": self.kind == "fixture",
                           "scientific_admission": False, "pipeline_complete": False}
                payload["native_receipt"] = self.native_result
                payload["native_receipt_verified"] = self.native_result is not None
                payload["receipt_digest"] = identity(payload)
                create_private(paths["receipt"], canonical_json(payload), 0o400)
                output_sha256 = file_hash(paths["receipt"])
                self._receipt_file_sha256 = output_sha256
            elif stage == "report":
                from .evidence import parse_json
                require(file_hash(paths["receipt"]) == self._receipt_file_sha256,
                        "native_receipt_report_binding_changed")
                prior = parse_json(read_regular(paths["receipt"]))
                require(prior["native_receipt"] == self.native_result
                        and (self.kind == "fixture" or self.native_result is not None),
                        "native_receipt_report_binding_changed")
                payload = {"redacted": True, "sealed_holdout_opened": False,
                           "scientific_qualification": "unmeasured", "release_authorization": "not_authorized",
                           "stage_receipt_sha256": file_hash(paths["receipt"]),
                           "fixture": self.kind == "fixture", "pipeline_complete": False}
                payload["native_receipt"] = self.native_result
                payload["native_receipt_verified"] = self.native_result is not None
                payload["report_digest"] = identity(payload)
                create_private(paths["report"], canonical_json(payload), 0o400)
                output_sha256 = file_hash(paths["report"])
            else:
                raise ValueError("worker_stage_invalid")
            # Validate inside the same failure boundary, including unexpected
            # verifier errors. The local import avoids a module initialization
            # cycle with the runner that owns the shared native payload contract.
            from .pilot_runner import _verify_stage_payload
            _verify_stage_payload(stage, payload, doc, artifact_sha256,
                                  paths["artifact"].stat().st_size, manifest_sha256)
            status = "completed"
        except (ValueError, TypeError, LookupError, ArithmeticError, RuntimeError, OSError):
            # Contract, callback and I/O failures have one terminal failed state.
            # The parent journal persists it after verifying physical settlement;
            # creating another file here could itself fail and hide that outcome.
            # Unclassified bugs and process interrupts propagate with the original
            # pending reservation; no retry, refund or successful fallback exists.
            payload, output_sha256 = None, None
        if self._execution is not None and self._execution.evidence.guard is not None\
                and self._execution.evidence.guard.violation:
            status = "over_budget"
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if stage == "receipt" and self.native_receipt is not None and self.native_receipt.outcome is not None:
            peak = max(peak, self.native_receipt.outcome["peak_ram_bytes"])
        outcome = {"status": status, "returncode": 0 if status == "completed" else 1,
                   "wall_ms": max(1, (time.monotonic_ns() - start + 999999) // 1000000),
                   "peak_ram_bytes": peak,
                   "resource_scope": "parent_lifetime_rss_upper_bound_not_child_or_gpu_peak"}
        return outcome, payload, output_sha256
