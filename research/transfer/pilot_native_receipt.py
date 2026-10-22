"""Private typed v1 receipt mapping and bounded native verification.

The executor supplies pinned basis documents before work, then separately
approves the completed proposal via issue_policy(bundle_path, bundle_sha256).
That hook returns (policy_bytes, independently_trusted_policy_sha256); it must
verify the actual executor/identities/cost observations, not echo candidate pins.
No default hook, model execution, scientific approval or cache publication exists.
"""
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import time

from .campaign_artifacts import MAX_BINARY, create_private, file_hash, read_regular
from .campaign_native import invoke, output_document
from .evidence import canonical_json, parse_json
from .pilot_learning_contract import ARMS, CONTEXTS, LEARNERS, SCALES, identity, pin, require
from .tdl.common import mean

BASIS_KINDS = ("protocol", "source_family", "task", "feature_transform", "target_transform",
               "generator", "metric", "environment", "evaluation_code", "source", "executor")
BINDINGS = ("artifact", "learners", "drivers", "physical", "cell")


def wire(value):
    return json.loads(json.dumps(value, allow_nan=False))


def seed(value):
    value = int(value, 16) if type(value) is str else value
    require(type(value) is int and 0 <= value < 1 << 53, "native_receipt_v1_seed_out_of_range")
    return value  # Never truncate uint64 or silently substitute another stream.


class _Nodes:
    def __init__(self, root):
        self.root, self.nodes = root, {}

    def add(self, kind, raw, requires=()):
        key = hashlib.sha256(raw).hexdigest()
        descriptor = {"sha256": key, "path": key + ".blob", "bytes": len(raw),
                      "kind": kind, "requires": sorted(set(requires))}
        require(bool(raw), "native_receipt_empty_dependency")
        if key in self.nodes:
            require(self.nodes[key] == descriptor, "native_receipt_dependency_alias")
        else:
            destination = self.root / descriptor["path"]
            if destination.exists():
                require(read_regular(destination, MAX_BINARY) == raw, "native_receipt_snapshot_changed")
            else:
                create_private(destination, raw, 0o400)
            self.nodes[key] = descriptor
        return key

    def doc(self, kind, value, requires=()):
        return self.add(kind, canonical_json(wire(value)), requires)


class NativeReceiptBridge:
    def __init__(self, basis_path, basis_sha256, binary_sha256, issuer_sha256, *, issue_policy):
        for value in (basis_sha256, binary_sha256, issuer_sha256):
            pin(value)
        require(callable(issue_policy), "native_receipt_trusted_policy_required")
        self.basis_path = Path(basis_path)
        require(self.basis_path.is_absolute(), "native_receipt_basis_path_invalid")
        self.basis_sha256, self.binary_sha256 = basis_sha256, binary_sha256
        self.issuer_sha256, self.issue_policy = issuer_sha256, issue_policy
        self.processes, self.outcome = 0, None
        self._check_issuer()
        self._basis = self._read_basis()
        self._used = False

    def _check_issuer(self):
        # The signed worker capsule pins the actual existing executor hook source.
        source = inspect.getsourcefile(self.issue_policy)
        require(source is not None and file_hash(Path(source)) == self.issuer_sha256,
                "native_receipt_issuer_code_changed")

    def _read_basis(self):
        raw = read_regular(self.basis_path)
        require(hashlib.sha256(raw).hexdigest() == self.basis_sha256, "native_receipt_basis_changed")
        value = parse_json(raw)
        require(set(value) == {"format", "bindings", "documents", "global_seed"}
                and value["format"] == "dope-pilot-native-receipt-basis-v1"
                and set(value["documents"]) == set(BASIS_KINDS), "native_receipt_basis_invalid")
        seed(value["global_seed"])
        return value

    def capsule(self):
        self._check_issuer()
        require(self._read_basis() == self._basis, "native_receipt_basis_changed")
        return {"format": "dope-pilot-native-receipt-bridge-v1", "basis_path": str(self.basis_path),
                "basis_sha256": self.basis_sha256, "binary_sha256": self.binary_sha256,
                "issuer_sha256": self.issuer_sha256}

    def bind(self, authority, cell):
        basis = self._read_basis()
        expected = {key: getattr(authority, key) for key in (
            "input_sha256", "source_sha256", "environment_sha256", "owner_receipt_sha256",
            "resource_receipt_sha256")}
        require(basis["bindings"] == expected and cell.document()["binary_sha256"] == self.binary_sha256,
                "native_receipt_parent_mismatch")
        for kind in ("source", "environment"):
            require(basis["documents"][kind]["sha256"] == expected[kind + "_sha256"],
                    "native_receipt_basis_identity_mismatch")

    def _documents(self):
        documents = {}
        for kind, item in self._read_basis()["documents"].items():
            require(set(item) == {"path", "sha256"}, "native_receipt_basis_invalid")
            pin(item["sha256"])
            path = Path(item["path"])
            require(path.is_absolute() and all(not p.is_symlink() for p in (path, *path.parents)),
                    "native_receipt_basis_path_invalid")
            raw = read_regular(path)
            require(hashlib.sha256(raw).hexdigest() == item["sha256"], "native_receipt_basis_changed")
            documents[kind] = raw
        return documents

    def prepare(self, execution, journal, cell, paths, root):
        """Map existing outputs only; no learner, permutation or metric rerun."""
        doc = cell.document()
        physical = journal.state[(doc["cell_id"], "learners")]["physical"]
        require(physical["settled"] and physical["completed"] and not physical["violation"]
                and not physical["over_budget"] and physical["graph"] == execution.evidence.guard.graph.document(),
                "native_receipt_unsettled_cost")
        learner_raw, driver_raw = read_regular(paths["learners"]), read_regular(paths["drivers"])
        learners, drivers = parse_json(learner_raw), parse_json(driver_raw)
        require(learners["execution"] == wire(execution.private_result())
                and learners["paired"]["query_row_ids"] == list(execution.plan.query.row_ids)
                and len(drivers["links"]) == 60
                and all(link["status"] == "completed" for link in drivers["links"]),
                "native_receipt_input_changed")
        comparisons = learners["paired"]["comparisons"]
        coordinates = [(s, l, c) for s in SCALES for l in LEARNERS for c in CONTEXTS]
        require([(r["scale"], r["learner"], r["context"]) for r in comparisons] == coordinates,
                "native_receipt_comparison_coverage")
        documents = self._documents()
        root.mkdir(mode=0o700)  # No replay or reuse of an earlier proposal directory.
        cost = {"format": "dope-pilot-settled-learning-cost-v1", "physical": physical,
                "journal_prefix": journal.records,
                "scope": "settled_learning_and_drivers; receipt_report_settle_in_outer_journal",
                "whole_job_cpu_gpu_cost": None}
        receipts, bindings = [], None
        for row in comparisons:
            nodes = _Nodes(root)
            bound = {"artifact": nodes.add("artifact", read_regular(paths["artifact"])),
                     "learners": nodes.add("source", learner_raw), "drivers": nodes.add("source", driver_raw),
                     "physical": nodes.doc("source", cost), "cell": nodes.doc("configuration", doc)}
            require(bindings is None or bindings == bound, "native_receipt_binding_changed")
            bindings = bound
            receipt = self._comparison(nodes, documents, execution, learners, row, bound, paths, doc)
            receipts.append(receipt)
        bundle = {"format": "dope-pilot-native-receipt-bundle-v1", "receipts": receipts, "bindings": bindings}
        create_private(root / "bundle.json", canonical_json(bundle), 0o400)
        return bundle

    def _comparison(self, nodes, documents, execution, learners, row, bound, paths, doc):
        plan = execution.plan
        cases = {c["arm"]: c for c in execution.cases if
                 (c["scale"], c["learner"], c["context"]) == (row["scale"], row["learner"], row["context"])}
        require(set(cases) == set(ARMS) and all(c["status"] == "completed" for c in cases.values()))
        case = cases["synthetic"]
        context = next(c for c in plan.contexts if c.name == row["context"])
        pool = next(p for s, a, p in plan.pools if s == row["scale"] and a == "synthetic")
        registration = execution.registry[row["learner"]]
        require(case["model_sha256"] == registration.model.sha256
                and case["recipe_sha256"] == registration.recipe.sha256
                and case["pool_sha256"] == pool.sha256
                and case["query_row_ids"] == plan.query.row_ids
                and case["consumed_rows"] == context.rows, "native_receipt_case_identity_changed")
        values = {k + "_hash": nodes.add(k, raw) for k, raw in documents.items()
                  if k not in ("evaluation_code", "protocol", "executor")}
        values["evaluation_code_hash"] = nodes.doc("evaluation_code", {
            "implementation_sha256": hashlib.sha256(documents["evaluation_code"]).hexdigest(),
            "bridge_source_sha256": file_hash(Path(__file__)), "bindings": bound}, bound.values())
        # Include original implementation bytes, not only a claimed digest.
        implementation = nodes.add("source", documents["evaluation_code"])
        nodes.nodes[values["evaluation_code_hash"]]["requires"].append(implementation)
        values["schema_hash"] = nodes.add("schema", read_regular(paths["schema"]))
        values["data_hash"] = nodes.add("data", read_regular(paths["csv"]))
        values["query_hash"] = nodes.doc("query", {"features": asdict(plan.query),
            "row_ids": learners["paired"]["query_row_ids"], "labels_sha256": learners["paired"]["labels_sha256"]})
        values["configuration_hash"] = bound["cell"]
        values["sample_hash"] = nodes.doc("sample", asdict(pool))
        values["learner_hash"] = nodes.doc("learner", {"model": asdict(registration.model),
            "registration_sha256": registration.sha256, "context": asdict(context), "cases": cases})
        values["checkpoint_hash"] = nodes.doc("checkpoint", {
            "weights_sha256": registration.model.weights_sha256,
            "mode": registration.model.mode, "registration_sha256": registration.sha256})
        values["recipe_hash"] = nodes.doc("recipe", asdict(registration.recipe))
        values.update(partition_role="exploratory", global_seed=seed(self._basis["global_seed"]),
            generator_fit_seed=seed(doc["fit_seed"] if doc["comparator"] == "dope" else doc["control_seed"]), sample_seed=seed(pool.draw_seed),
            learner_seed=plan.fit_seed, produced_rows=pool.produced_rows, requested_rows=pool.requested_rows,
            consumed_rows=case["consumed_rows"], context_rows=context.rows, fit_rows=doc["fit_rows"],
            query_rows=len(plan.query.row_ids), ensemble_count=1, precision=registration.model.precision,
            determinism=registration.model.determinism)
        # ensemble_count is the actual adapter recipe, never an invented default.
        recipe = asdict(registration.recipe)
        require("estimators" in recipe or row["learner"] in ("gradient_boosting", "neural"),
                "native_receipt_ensemble_mapping_missing")
        if "estimators" in recipe:
            values["ensemble_count"] = recipe["estimators"]
        split = {k: values[k] for k in ("data_hash", "query_hash", "partition_role", "fit_rows", "query_rows")}
        values["split_hash"] = nodes.doc("split", {"format": "dope-split-manifest", "version": 1, **split},
                                          [values["data_hash"], values["query_hash"]])
        protocol = parse_json(documents["protocol"])
        commitments = set()
        for section in ("freeze_requirements", "workload", "disclosure"):
            for field, value in protocol[section].items():
                if (section == "freeze_requirements" or field.endswith("_hash")) and value is not None:
                    if not str(value).startswith("TBD-"):
                        require(value in nodes.nodes, "native_receipt_protocol_leaf_missing")
                        commitments.add(value)
        values["protocol_hash"] = nodes.add("protocol", documents["protocol"], commitments)
        state = nodes.doc("fit_state", {"artifact_sha256": bound["artifact"],
            "generator_seed": values["generator_fit_seed"], "input_sha256": doc["csv_sha256"]}, [bound["artifact"]])
        fit_fields = ("data_hash", "split_hash", "generator_hash", "configuration_hash",
                      "feature_transform_hash", "target_transform_hash", "generator_fit_seed", "global_seed")
        fit = {k: values[k] for k in fit_fields}
        values["fit_hash"] = nodes.doc("fit", {"format": "dope-generator-fit", "version": 1,
            "state_hash": state, **fit}, [state] + [v for k, v in fit.items() if k.endswith("_hash")])
        values["artifact_inventory_hash"] = nodes.doc("artifact_inventory", {
            "format": "dope-artifact-inventory", "version": 1, "files": [bound["artifact"]]}, [bound["artifact"]])
        require(set(row["losses"]) == set(ARMS) and all(len(v) == values["query_rows"] for v in row["losses"].values()))
        paired = {"format": "dope-paired-losses", "version": 1, "query_hash": values["query_hash"],
                  "metric_hash": values["metric_hash"], "loss": "mse", "loss_space": "original_target_units",
                  "loss_unit": "target_units_squared", **row["losses"]}
        paired_hash = nodes.doc("paired_evidence", paired, [values["query_hash"], values["metric_hash"]])
        executor = nodes.add("executor", documents["executor"])
        outcomes = [r for r in execution.evidence.records if r["event"] == "outcome"]
        require(outcomes and all(r["status"] == "completed" for r in outcomes))
        # Shared operation cost appears once in the bound physical DAG. These
        # per-comparison references are not additive costs across twenty receipts.
        attempt = {"attempt_id": bound["physical"], "parent_attempt_id": None, "exit_code": 0,
            "wall_time_ms": max(1, (sum(r["wall_ns"] for r in outcomes) + 999999) // 1000000),
            "cpu_time_ms": int(sum(r["process_cpu_ns"] / 1e6 + r["reaped_children_cpu_seconds"] * 1000 for r in outcomes)),
            "peak_memory_bytes": max(r["process_lifetime_peak_rss_bytes"] for r in outcomes)}
        return {"format": "dope-transfer-run", "version": 1, "identity": values,
            "executor_hash": executor, "freeze_hash": None, "dependencies": list(nodes.nodes.values()),
            "sampling_payload_bytes": nodes.nodes[bound["artifact"]]["bytes"], "paired_evidence_hash": paired_hash,
            "loss": "mse", "loss_space": "original_target_units", "loss_unit": "target_units_squared",
            "losses": {arm: mean(row["losses"][arm]) for arm in ARMS}, "execution": "complete", "attempt": attempt,
            "projection": "private", "release": "restricted_local", "transfer": "unmeasured",
            "empirical_privacy": "unmeasured", "formal_mechanism": "none", "legal_route": "unreviewed"}

    def verify(self, execution, journal, cell, paths, root):
        self._check_issuer()
        require(not self._used, "native_receipt_replay_refused")
        self._used = True
        doc = cell.document()
        require(journal.state[(doc["cell_id"], "receipt")]["pending"], "native_receipt_stage_not_reserved")
        start = time.monotonic_ns()
        directory = root / "native-receipts"
        require(file_hash(paths["binary"], MAX_BINARY) == self.binary_sha256, "native_receipt_binary_changed")
        bundle = self.prepare(execution, journal, cell, paths, directory)
        bundle_hash = file_hash(directory / "bundle.json", 16 << 20)
        policy_bytes, policy_hash = self.issue_policy(directory / "bundle.json", bundle_hash)
        pin(policy_hash)
        require(type(policy_bytes) is bytes and hashlib.sha256(policy_bytes).hexdigest() == policy_hash,
                "native_receipt_policy_pin_mismatch")
        create_private(directory / "policy.json", policy_bytes, 0o400)
        remaining = doc["stage_wall_ms"]["receipt"] - (time.monotonic_ns() - start + 999999) // 1000000
        require(remaining > 0, "native_receipt_deadline")
        self.processes += 1
        self.outcome = invoke([str(paths["binary"]), "verify-pilot-receipts", "--root", str(directory),
            "--bundle", "bundle.json", "--policy", "policy.json", "--expected-bundle-sha256", bundle_hash,
            "--expected-policy-sha256", policy_hash, "--expected-executable-sha256", self.binary_sha256],
            root, "native-receipt", remaining, journal.spec.document()["budget"]["peak_ram_bytes"])
        require(self.outcome["status"] == "completed" and self.outcome["returncode"] == 0,
                "native_receipt_verifier_failed")
        result = output_document(root, "native-receipt")
        expected = {"format": "dope-pilot-native-verified-v1", "status": "verified",
            "bundle_sha256": bundle_hash, "policy_sha256": policy_hash, "executable_sha256": self.binary_sha256,
            "receipts": [{"receipt_sha256": identity(r), "identity_sha256": identity(r["identity"])} for r in bundle["receipts"]],
            "scientific_admission": False, "release_authorization": "not_authorized"}
        require(result == expected and file_hash(paths["binary"], MAX_BINARY) == self.binary_sha256
                and file_hash(directory / "bundle.json", 16 << 20) == bundle_hash
                and file_hash(directory / "policy.json", 16 << 20) == policy_hash
                and all(file_hash(paths[k]) == bundle["bindings"][k]
                        for k in ("artifact", "learners", "drivers")),
                "native_receipt_output_mismatch")
        return result
