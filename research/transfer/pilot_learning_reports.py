"""Private metric/driver side: query targets never enter learner interfaces."""
from dataclasses import asdict
import json
from research.transfer.adapters.base import FeatureBatch
from research.transfer.campaign_artifacts import create_private, private_root
from research.transfer.evidence import canonical_json
from research.transfer.tdl import transfer as transfer_reducer
from research.transfer.tdl import drivers as driver_reducer
from .pilot_learning_contract import (ARMS, CONTEXTS, LEARNERS, SCALES, DriverPlan, QueryLabels,
                                      finite, identity, require)


def losses(prediction, labels):
    require(type(labels) is QueryLabels and tuple(prediction["row_ids"]) == labels.row_ids)
    values = prediction["values"]
    require(len(values) == len(labels.values))
    # Use the existing original-unit loss implementation, not another metric.
    result, _ = transfer_reducer._prediction_losses({
        "task": "regression", "targets": list(labels.values), "prediction_offset": 0.0,
        "prediction_scale": 1.0, "real_predictions": list(values),
        "synthetic_predictions": list(values), "null_predictions": list(values)})
    return result[0]


def reduce_learning(execution, labels, inference):
    """inference: external per-coordinate real_gain_lcb and evidence_sha256.

    Absent inference remains incomplete. Negative or uninformative results are kept.
    No selection, repeated fit or invented confidence interval is performed here.
    """
    require(type(labels) is QueryLabels and labels.row_ids == execution.plan.query.row_ids)
    require(type(inference) is dict)
    comparisons = []
    for scale in SCALES:
        for learner in LEARNERS:
            for context in CONTEXTS:
                coordinate = (scale, learner, context)
                cases = {case["arm"]: case for case in execution.cases
                         if (case["scale"], case["learner"], case["context"]) == coordinate}
                row = {"scale": scale, "learner": learner, "context": context,
                       "status": "incomplete", "reason": "missing_required_path", "losses": {}}
                if set(cases) == set(ARMS) and all(case["status"] == "completed" for case in cases.values()):
                    for arm in ARMS:
                        row["losses"][arm] = losses(execution.predictions[cases[arm]["prediction_key"]], labels)
                    bound = inference.get(coordinate)
                    lower = None
                    if bound is not None:
                        from .pilot_learning_contract import pin
                        require(set(bound) == {"real_gain_lcb", "evidence_sha256", "labels_sha256"})
                        pin(bound["evidence_sha256"])
                        require(bound["labels_sha256"] == labels.sha256)
                        lower = finite(bound["real_gain_lcb"])
                    row["inference"] = bound
                    reduction = transfer_reducer.evaluate_tdl({
                        "task": "paired_losses", "loss_space": "original_mse", "required_model_present": True,
                        "real_losses": row["losses"]["real"], "synthetic_losses": row["losses"]["synthetic"],
                        "null_losses": row["losses"]["null"], "real_gain_lcb": lower,
                        "constant_target": len(set(labels.values)) == 1})
                    row.update(status=reduction["status"], reason=reduction.get("reason"), reduction=reduction)
                comparisons.append(row)
    private = {"labels_sha256": labels.sha256, "query_row_ids": labels.row_ids,
               "query_targets": labels.values, "comparisons": comparisons, "planned_denominator": 20}
    execution.evidence.append({"event": "paired_losses", "private_evidence": private})
    return private


def execute_drivers(execution, labels, plan):
    from research.transfer.drivers import permutation_importance
    require(type(plan) is DriverPlan and type(labels) is QueryLabels)
    query = execution.plan.query
    require(labels.row_ids == query.row_ids)
    require(all(index < len(query.rows[0]) for _, group in plan.groups for index in group))
    records = getattr(execution, "_driver_records", {})
    execution._driver_records = records
    bindings = getattr(execution, "_driver_bindings", {})
    execution._driver_bindings = bindings
    links = []
    for case in execution.cases:
        coordinate = {key: case[key] for key in ("scale", "learner", "context", "arm")}
        if case["status"] != "completed":
            links.append({**coordinate, "status": "incomplete", "reason": "learner_path_failed"})
            continue
        key = identity({"fit": case["fit_key"], "query": query.sha256,
                        "labels": labels.sha256, "driver_plan": asdict(plan)})
        binding_key = identity({"coordinate": coordinate, "driver_key": key})
        if binding_key in bindings:
            links.append(bindings[binding_key])
            continue
        reused = key in records
        if not reused:
            adapter = execution.prepared[case["fit_key"]]
            require(adapter is not None, "fitted_state_released_no_implicit_refit")
            registration = execution.registration_for(case["learner"], case["arm"])
            require(registration.sha256 == case["registration_sha256"], "driver_registration_changed")
            calls = []
            record = {"status": "failed", "fit_key": case["fit_key"], "driver_plan": asdict(plan),
                      "labels_sha256": labels.sha256, "calls": calls}
            records[key] = record  # A failed physical driver is retained and never retried.
            execution.evidence.append({"event": "driver_intent", "key": key, "planned_predictions":
                                       1 + len(plan.groups) * plan.repeats})
            try:
                def loss(columns):
                    number = len(calls)
                    group = None if number == 0 else plan.groups[(number - 1) // plan.repeats][0]
                    repeat = None if number == 0 else (number - 1) % plan.repeats
                    perturbed = FeatureBatch(query.row_ids, tuple(tuple(row) for row in zip(*columns)),
                                             query.schema_sha256)
                    call_key = identity({"driver": key, "invocation": number, "features": perturbed.sha256})
                    call = {"key": call_key, "group": group, "repetition": repeat,
                            "permutation_seed": plan.seed, "perturbed_rows": perturbed.rows,
                            "perturbed_row_ids": perturbed.row_ids, "query_sha256": perturbed.sha256,
                            "status": "started"}
                    calls.append(call)
                    result = execution.predict(adapter, registration, perturbed, case["consumed_rows"],
                                               call_key, parent=case["fit_key"], extra=call)
                    paired = losses(result, labels)
                    value = finite(sum(paired) / len(paired))
                    call.update(status="completed", losses=paired, loss=value, prediction=result)
                    return value
                native = permutation_importance(loss, tuple(zip(*query.rows)), dict(plan.groups),
                                                learner_id=registration.role, query_population=query.sha256,
                                                seed=plan.native_seed, repeats=plan.repeats)
                require(len(calls) == 1 + len(plan.groups) * plan.repeats)
                record.update(status="completed", result=native,
                              applicability={"mass": "descriptive", "rank": "descriptive",
                                             "sign": "not_measured", "interaction": "not_measured",
                                             "causal": "not_qualified"})
            except Exception:
                record["reason"] = "driver_operation_failed"
            execution.evidence.append({"event": "driver_outcome", "key": key, "private_evidence": record})
        link = {**coordinate, "key": key, "status": records[key]["status"],
                "reused": reused, "costs_charged_at": key}
        bindings[binding_key] = link
        links.append(link)
    comparisons = []
    for scale in SCALES:
        for learner in LEARNERS:
            for context in CONTEXTS:
                arms = {link["arm"]: link for link in links
                        if (link["scale"], link["learner"], link["context"]) == (scale, learner, context)}
                payload = {"required_components": ["mass", "rank"], "real_real_ceiling": None}
                if all(arms.get(arm, {}).get("status") == "completed" for arm in ("real", "synthetic")):
                    for arm in ("real", "synthetic"):
                        raw = records[arms[arm]["key"]]["result"]["importance"]
                        payload[arm + "_importances"] = [raw[name] for name, _ in plan.groups]
                comparisons.append({"scale": scale, "learner": learner, "context": context,
                                    "reduction": driver_reducer.evaluate(payload)})
    return {"records": records, "links": links, "comparisons": comparisons,
            "planned_path_count": 60, "planned_comparison_count": 20, "independent_ceiling": "missing"}


def execute_streamed(execution, labels, plan):
    """Metric-side orchestrator: fit, predict, permute, release; no implicit refits."""
    require(execution.plan.model_retention == "stream_drivers_then_release")
    latest = None
    def consume(current):
        nonlocal latest
        latest = execute_drivers(current, labels, plan)
    execution.execute(on_case=consume)
    return execution, latest


def publish_private(execution, paired, drivers, directory):
    """Write private actual records; summaries expose counts/commitments only."""
    root = private_root(directory)
    learner_evidence = {"execution": execution.private_result(), "paired": paired}
    for name, value in (("private-learners.json", learner_evidence), ("private-drivers.json", drivers)):
        create_private(root / name, canonical_json(json.loads(json.dumps(value, allow_nan=False))))
    completed = sum(case["status"] == "completed" for case in execution.cases)
    learners = {"format": "dope-pilot-learners-v2", "plan_sha256": execution.plan.sha256,
                "comparator": execution.plan.comparator, "candidate": execution.plan.candidate,
                "artifact_sha256": execution.plan.artifact_sha256,
                "null_reference_sha256": execution.plan.null_reference_sha256,
                "training_scales": list(SCALES), "learners": list(LEARNERS), "contexts": list(CONTEXTS),
                "learner_paths": list(ARMS), "evaluation_count": 20, "evaluation_count_kind": "planned",
                "planned_path_count": 60, "completed_comparison_count": sum(
                    all(any(c["status"] == "completed" and
                            (c["scale"], c["learner"], c["context"], c["arm"]) == (s, l, x, a)
                            for c in execution.cases) for a in ARMS)
                    for s in SCALES for l in LEARNERS for x in CONTEXTS),
                "completed_path_count": completed, "private_evidence_sha256": identity(learner_evidence),
                "status": "completed" if completed == 60 else "incomplete", "pipeline_complete": False,
                "development_exposed": list(LEARNERS), "fixture": execution.fixture}
    if len(paired["comparisons"]) == 20 and all(len(row["losses"]) == 3 for row in paired["comparisons"]):
        learners["paired_loss_digest"] = identity(paired)
    driver_summary = {"format": "dope-pilot-drivers-v2", "path": "drivers-v1",
                      "evidence_kind": "permutation_importance", "learners": list(LEARNERS),
                      "planned_path_count": 60, "pipeline_complete": False, "status": "incomplete"}
    if len(drivers["links"]) == 60 and all(link["status"] == "completed" for link in drivers["links"]):
        driver_summary.update(status="executed_unqualified", driver_digest=identity(drivers))
    create_private(root / "learning-summary.json", canonical_json({"learners": learners, "drivers": driver_summary}))
    return learners, driver_summary
