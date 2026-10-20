"""Actual adapter calls and private physical evidence, without query labels."""
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time

from research.transfer.adapters.base import PredictionResult
from research.transfer.campaign_artifacts import create_private, private_root
from research.transfer.evidence import canonical_json
from .pilot_learning_contract import (ARMS, LEARNERS, SCALES, LearningPlan, count, finite,
                                      identity, require)
from .pilot_learning_runtime import authorize, registry_digest


class PhysicalEvidence:
    """Append-only evidence, not another scheduler or permission/quota authority.

    The caller must hold the existing StageJournal stage reservation before entry.
    A fresh exclusive directory refuses replay. Each intent is durable before work.
    """
    def __init__(self, directory, guard=None):
        self.guard = guard
        root = private_root(directory)
        self.root = root / "physical-operations"
        self.root.mkdir(mode=0o700)  # existing directory is an error, never resume/reissue
        self.records = []
        self.previous = "0" * 64

    def append(self, event):
        row = {"sequence": len(self.records), "previous_sha256": self.previous, **event}
        # Convert immutable tuple vocabulary to the existing finite JSON codec.
        row = json.loads(json.dumps(row, allow_nan=False))
        encoded = canonical_json(row)
        create_private(self.root / (str(len(self.records)).zfill(6) + ".json"), encoded)
        self.previous = identity(row)
        self.records.append(row)
        return self.previous

    def call(self, key, kind, inputs, operation):
        if self.guard is not None:
            self.guard.begin(key, kind)
        self.append({"event": "intent", "key": key, "kind": kind, "inputs": inputs})
        wall, cpu = time.monotonic_ns(), time.process_time_ns()
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        status, output = "failed", None
        try:
            value, output = operation()
            status = "completed"
            return value
        finally:
            # Settle the parent's precharged slot before allowing another call.
            elapsed = time.monotonic_ns() - wall
            refusal = None
            if self.guard is not None:
                try:
                    self.guard.finish(key, status, elapsed, output)
                except Exception as error:
                    status, refusal = "over_budget", error
            # Never include backend exception strings; native messages may carry data.
            self.append({"event": "outcome", "key": key, "kind": kind, "status": status,
                         "reason": None if status == "completed" else "operation_failed",
                         "wall_ns": elapsed,
                         "process_cpu_ns": time.process_time_ns() - cpu,
                         "reaped_children_cpu_seconds": (resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime +
                             resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime - children.ru_utime - children.ru_stime),
                         "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                         "resource_scope": "self_cpu_and_lifetime_rss_upper_bound_not_child_or_gpu_peak",
                         "output": output})
            if refusal is not None:
                raise refusal


def validate_prediction(result, query, consumed):
    require(type(result) is PredictionResult and result.kind == "point")
    require(result.row_ids == query.row_ids and len(result.values) == len(query.rows))
    require(count(result.consumed_rows, 1) == consumed)
    count(result.memory_bytes)
    count(result.gpu_bytes)
    require(all(type(row) is tuple and len(row) == 1 for row in result.values))
    values = tuple(finite(row[0]) for row in result.values)
    return {"row_ids": result.row_ids, "values": values, "consumed_rows": result.consumed_rows,
            "memory_bytes": result.memory_bytes, "gpu_bytes": result.gpu_bytes}


def attempts(adapter):
    # Mitra explicitly exposes context registration and each predict-time bagged fit.
    return tuple(dict(row) for row in getattr(adapter, "attempts", ()))


class LearningExecution:
    def __init__(self, plan, registry, null_reference, admission, verify_operation, evidence_dir):
        require(type(plan) is LearningPlan)
        self.fixture = authorize(plan, registry, null_reference, admission, verify_operation)
        self.plan, self.registry, self.admission = plan, dict(registry), admission
        self._plan_sha256, self._registry_sha256 = plan.sha256, registry_digest(registry)
        self.null_reference = null_reference
        self.evidence = PhysicalEvidence(evidence_dir)
        self.prepared, self.predictions, self.failed, self.cases = {}, {}, {}, []
        planned = [{"scale": scale, "learner": learner, "context": context.name, "arm": arm}
                   for scale in SCALES for learner in LEARNERS for context in plan.contexts for arm in ARMS]
        require(len(planned) == 60)
        self.evidence.append({"event": "planned", "plan_sha256": plan.sha256, "private_plan": asdict(plan),
                              "admission": asdict(admission), "coordinates": planned,
                              "registry": {name: {"model_identity": asdict(entry.model),
                                  "adapter_admission": asdict(entry.admission), "recipe": asdict(entry.recipe),
                                  "runtime_sha256": entry.runtime_sha256, "module_pins": entry.module_pins,
                                  "operation_qualification_sha256": entry.operation_qualification_sha256}
                                  for name, entry in {**self.registry, "null_reference": self.null_reference}.items()},
                              "native_seed_provenance": plan.native_seed.provenance,
                              "planned_path_count": 60, "planned_comparison_count": 20})

    def _require_budget(self):
        guard = self.evidence.guard
        if guard is None:
            require(self.fixture, "physical_reservation_required")
            return
        from .pilot_physical_operations import PhysicalOperationGuard
        require(type(guard) is PhysicalOperationGuard, "physical_reservation_required")
        graph = guard.graph.document()
        require(graph["learning_plan_sha256"] == self.plan.sha256 and
                graph["registry_sha256"] == registry_digest(self.registry) and
                graph["null_reference_sha256"] == self.null_reference.sha256 and
                graph["operation_admission_sha256"] == identity(asdict(self.admission)),
                "physical_reservation_binding_mismatch")

    def registration_for(self, learner, arm):
        require(learner in LEARNERS and arm in ARMS)
        entry = self.null_reference if arm == "null" else self.registry[learner]
        if arm == "null":
            require(entry.sha256 == self.plan.null_reference_sha256, "null_reference_changed")
        return entry

    def _require_allocation(self, registration, key, kind):
        self._require_budget()
        if self.evidence.guard is None:
            return  # fixture-only compatibility; development already refused
        node = self.evidence.guard.nodes.get(key)
        require(node is not None and node["kind"] == kind, "physical_operation_not_reserved")
        if not self.fixture:
            require(registration.model.resources.wall_seconds * 1000 <= node["wall_ms"],
                    "native_timeout_exceeds_physical_allocation")
        if registration.role == "mitra" and kind == "predict":
            require(node["reserved_child_fits"] >= 8 and node["reserved_optimizer_updates"] >= 400,
                    "mitra_physical_allocation_incomplete")
            if not self.fixture:
                require(registration.recipe.bag_folds == 8, "mitra_physical_recipe_mismatch")

    def functional_inputs(self, registration, scale, arm, context):
        require((registration.role == "null_reference") == (arm == "null"), "arm_registration_mismatch")
        if arm == "null":
            require(registration.sha256 == self.plan.null_reference_sha256, "null_reference_changed")
        pool, training = self.plan.selected(scale, arm, context)
        require(registration.model.context_policy == "all" and registration.model.context_cap >= context.rows)
        require(registration.model.query_batch_size >= len(self.plan.query.rows))
        inputs = {"task_id": self.plan.task_id, "comparator": self.plan.comparator,
                  "comparison_arm": arm, "source_role": pool.source_role, "pool_sha256": pool.sha256,
                  "source_sha256": self.plan.source_sha256, "policy_sha256": self.plan.policy_sha256,
                  "input_sha256": self.plan.input_sha256, "registration_sha256": registration.sha256,
                  "training_sha256": training.sha256, "query_sha256": self.plan.query.sha256,
                  "context": context.name, "context_policy_sha256": context.policy_sha256,
                  "context_seed": context.seed, "context_cap": context.rows,
                  "fit_seed": self.plan.fit_seed, "predict_seed": self.plan.predict_seed,
                  "seed_derivation_sha256": self.plan.seed_derivation_sha256,
                  "owner_recipe_decision_sha256": self.plan.owner_recipe_decision_sha256}
        return pool, training, inputs

    def predict(self, adapter, registration, query, consumed, key, *, parent, extra=None):
        self._require_allocation(registration, key, "predict")
        def operation():
            start = len(attempts(adapter))
            try:
                result = adapter.predict(query, self.plan.predict_seed)
            finally:
                self.evidence.append({"event": "native_attempts", "parent": key,
                                      "attempts": attempts(adapter)[start:]})
            payload = validate_prediction(result, query, consumed)
            payload.update(module_origins=registration.origins(adapter, self.fixture),
                           native_attempts=attempts(adapter)[start:],
                           predict_seed=self.plan.predict_seed, query_sha256=query.sha256)
            return payload, payload
        return self.evidence.call(key, "predict", {"parent": parent, "query_sha256": query.sha256,
                                                  "query_row_ids": query.row_ids, **(extra or {})}, operation)

    def execute(self, on_case=None):
        self._require_budget()
        require(not self.cases, "execution_replay_refused")
        streaming = self.plan.model_retention == "stream_drivers_then_release"
        require(not streaming or callable(on_case), "streaming_driver_consumer_required")
        require(self.plan.sha256 == self._plan_sha256 and
                registry_digest(self.registry) == self._registry_sha256 and
                self.null_reference.sha256 == self.plan.null_reference_sha256, "admitted_inputs_changed")
        for scale in SCALES:
            for learner in LEARNERS:
                for context in self.plan.contexts:
                    for arm in ARMS:
                        registration = self.registration_for(learner, arm)
                        case = {"scale": scale, "learner": learner, "context": context.name,
                                "arm": arm, "status": "failed", "development_exposed": True}
                        try:
                            pool, training, inputs = self.functional_inputs(registration, scale, arm, context)
                            fit_key = identity({"operation": "prepare", **inputs})
                            predict_key = identity({"operation": "predict", "fit_key": fit_key,
                                                    "query": self.plan.query.sha256, "seed": self.plan.predict_seed})
                            case.update(fit_key=fit_key, prediction_key=predict_key,
                                        available_pool_rows=len(pool.batch.targets),
                                        pool_requested_rows=pool.requested_rows, pool_produced_rows=pool.produced_rows,
                                        requested_context_rows=context.rows, provided_rows=len(training.targets),
                                        consumed_rows=None, sample_policy=scale, pool_sha256=pool.sha256,
                                        train_row_ids=training.features.row_ids, query_row_ids=self.plan.query.row_ids,
                                        model_sha256=registration.model.sha256, recipe_sha256=registration.recipe.sha256,
                                        runtime_sha256=registration.runtime_sha256, registration_sha256=registration.sha256,
                                        registration_role=registration.role)
                            if fit_key in self.failed or predict_key in self.failed:
                                case["reason"] = "prior_physical_failure_no_retry"
                            else:
                                reused_fit = fit_key in self.prepared
                                if not reused_fit:
                                    def prepare():
                                        self._require_allocation(registration, fit_key, "prepare")
                                        adapter = registration.build(self.fixture)
                                        try:
                                            adapter.prepare(training, self.plan.fit_seed)
                                        finally:
                                            self.evidence.append({"event": "native_attempts", "parent": fit_key,
                                                                  "attempts": attempts(adapter)})
                                        receipt = adapter.receipt()
                                        require(count(receipt["provided_rows"], 1) == len(training.targets))
                                        require(count(receipt["consumed_rows"], 1) == context.rows)
                                        require(receipt["context_sha256"] == training.sha256)
                                        return adapter, {"receipt": receipt, "native_attempts": attempts(adapter)}
                                    try:
                                        self.prepared[fit_key] = self.evidence.call(fit_key, "prepare", inputs, prepare)
                                    except Exception:
                                        self.failed[fit_key] = True
                                        raise
                                reused_prediction = predict_key in self.predictions
                                if not reused_prediction:
                                    try:
                                        self.predictions[predict_key] = self.predict(
                                            self.prepared[fit_key], registration, self.plan.query, context.rows,
                                            predict_key, parent=fit_key)
                                    except Exception:
                                        self.failed[predict_key] = True
                                        raise
                                payload = self.predictions[predict_key]
                                case.update(status="completed", reason=None, consumed_rows=payload["consumed_rows"],
                                            produced_predictions=len(payload["values"]),
                                            reused_prepare=reused_fit, reused_prediction=reused_prediction,
                                            costs_charged_at=(fit_key, predict_key))
                        except Exception:
                            case["reason"] = "required_operation_failed"
                        self.cases.append(case)
                        self.evidence.append({"event": "logical_case", "case": case})
                        if on_case is not None:
                            on_case(self)  # trusted metric-side consumer; no labels passed to an adapter
                        if streaming and "fit_key" in case:
                            self.prepared[case["fit_key"]] = None
        return self

    def private_result(self):
        return {"format": "dope-pilot-learning-evidence-v1", "plan_sha256": self.plan.sha256,
                "fixture": self.fixture, "cases": self.cases, "predictions": self.predictions,
                "planned_path_count": 60, "planned_comparison_count": 20,
                "completed_path_count": sum(c["status"] == "completed" for c in self.cases),
                "physical_operations": self.evidence.records, "pipeline_complete": False}
