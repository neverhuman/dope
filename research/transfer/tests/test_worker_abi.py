"""Shared Python/Rust wire fixtures; no external runtime or protected data."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest

from research.transfer.adapters import worker_abi as abi
from research.transfer.adapters.base import FeatureBatch, TrainingBatch
from research.transfer.evidence import canonical_json, digest
from research.transfer.tests.test_adapters import features, learner, training


def pin(text):
    return hashlib.sha256(text.encode()).hexdigest()


def settings():
    pins = {key: pin(key) for key in (
        "model_sha256", "panel_sha256", "worker_image_sha256", "checkpoint_sha256",
        "prepared_sha256", "split_sha256", "dependencies_sha256", "preprocessing_sha256", "recipe_sha256")}
    pins.update(runtime_version="1.2.3+fixture", precision="float64", device="cpu")
    resources = dict(threads=1, wall_ms=1000, memory_bytes=1024, vram_bytes=0,
                     disk_bytes=0, network=False)
    policy = dict(primary_n=3, sample_multiplier=1, context_cap=3, truncation="none",
                  subsampling="none", predict_time_fit=False, query_policy="batch", query_batch_size=16,
                  order_sensitive=False)
    return pins, resources, policy


def fixture(method="predict", output=None, policy_changes=None):
    pins, resources, policy = settings()
    policy.update(policy_changes or {})
    active = method in abi.DATA_METHODS
    train = training()
    if output is not None and output["kind"] == "probability":
        train = replace(train, targets=(0, 1, 0))
    if policy["sample_multiplier"] == 4:
        train = TrainingBatch(features(tuple(range(12))), tuple(range(12)), "synthetic_train")
    ids = train.features.row_ids
    if policy["truncation"] == "head":
        ids = ids[:policy["context_cap"]]
    elif policy["subsampling"] == "seeded":
        ids = ids[::-1][:policy["context_cap"]]
    request, buffers = abi.build_request(pins, resources, policy, method, 7,
        training=train if active else None, context_row_ids=ids if active else (),
        query=features((3., 4.), "query") if method in abi.PREDICT_METHODS else None, output=output)
    prediction, result_buffers = None, {}
    if method in abi.PREDICT_METHODS:
        values = ((70.,), (90.,)) if output is None else ((.25, .75), (.75, .25))
        prediction, result_buffers["prediction"] = abi.pack_buffer(
            tuple(request["inputs"]["query_features"]["row_ids"]), values)
    response = {"format": "dope-model-worker-v1", "type": "response",
        **{key: copy.deepcopy(request[key]) for key in ("request_sha256", "pins", "counts",
           "context_sha256", "query_sha256", "output")}, "status": "ok", "error": None,
        "prediction": prediction, "usage": dict(wall_ms=1, memory_bytes=128, vram_bytes=0),
        "qualified": False, "qualification": "fixture_only"}
    return request, buffers, response, result_buffers


def seal(request):
    # Deliberately independent from the implementation helper.
    inputs = request["inputs"]
    request["context_sha256"] = digest(dict(train_features=inputs["train_features"],
        train_labels=inputs["train_labels"], context_row_ids=request["context_row_ids"]))
    request["query_sha256"] = digest(inputs["query_features"])
    request["request_sha256"] = digest({k: v for k, v in request.items() if k != "request_sha256"})


def wire_cases():
    """Rust consumes every case, including adversarial cases with recomputed hashes."""
    cases = []
    def add(name, values, request_ok=True, response_ok=True):
        request, buffers, response, result_buffers = values
        cases.append(dict(name=name, request=canonical_json(request).decode(),
            buffers={k: list(v) for k, v in buffers.items()},
            response=canonical_json(response).decode(),
            response_buffers={k: list(v) for k, v in result_buffers.items()},
            expected_request=request_ok, expected_response=response_ok,
            expected_receipt=(abi.validate_response(response, result_buffers, request, buffers, settings()[0])
                              if request_ok and response_ok else None)))
    for method in sorted(abi.DATA_METHODS | {"capabilities", "qualify", "release_context", "shutdown"}):
        add(method, fixture(method))
    probability = dict(kind="probability", class_order=[pin("class_a"), pin("class_b")], units="probability")
    add("probability", fixture(output=probability))
    add("distribution", fixture(output=dict(kind="distribution", class_order=[], units="distribution_target_original")))
    for name, changes in (("four_n", dict(sample_multiplier=4, truncation="head")),
                          ("head", dict(context_cap=2, truncation="head")),
                          ("seeded_subset", dict(context_cap=2, subsampling="seeded")),
                          ("predict_fit", dict(predict_time_fit=True)),
                          ("transductive", dict(query_policy="transductive", order_sensitive=True))):
        add(name, fixture(policy_changes=changes))
    sentinel = "private_header_/secret/checkpoint_987654"
    for path, value in (
        (("query_labels",), [sentinel]), (("inputs", "query_labels"), [sentinel]),
        (("inputs", "query_features", "targets"), [sentinel]),
        (("pins", "checkpoint_sha256"), pin("wrong checkpoint")),
        (("pins", "runtime_version"), ">=1.2"), (("resources", "network"), True),
        (("counts", "requested_rows"), 4), (("counts", "produced_rows"), 2),
        (("counts", "provided_rows"), 2), (("counts", "consumed_rows"), 2),
        (("policy", "sample_multiplier"), 4), (("policy", "context_cap"), 2),
        (("policy", "query_batch_size"), 1), (("feature_schema_sha256",), None),
        (("training_role",), "confirmation"), (("seed",), True), (("fit_seed",), -1),
        (("inputs", "train_labels", "columns"), 2)):
        values = fixture()
        target = values[0]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        seal(values[0])
        add("request_" + "_".join(path), values, False, False)
    values = fixture()
    values[0]["inputs"]["train_labels"]["row_ids"].reverse()
    seal(values[0])
    add("training_label_row_order", values, False, False)
    values = fixture()
    values[0]["context_row_ids"].reverse()
    seal(values[0])
    add("context_order", values, False, False)
    values = fixture()
    values[1]["query_labels"] = b"private labels"
    add("undeclared_label_buffer", values, False, False)
    values = fixture()
    values[1]["train_features"] = b"\0" * 24
    add("poisoned_bytes", values, False, False)
    values = fixture()
    values[1]["train_labels"] = struct.pack("<ddd", float("nan"), 1., 2.)
    values[0]["inputs"]["train_labels"]["sha256"] = hashlib.sha256(values[1]["train_labels"]).hexdigest()
    seal(values[0])
    add("nonfinite_training_buffer", values, False, False)
    for path, value in (
        (("request_sha256",), pin("stale cache")), (("qualified",), True),
        (("qualification",), "gpu_qualified"), (("output", "units"), "rank"),
        (("usage", "memory_bytes"), 1025), (("usage", "wall_ms"), 1001),
        (("usage", "vram_bytes"), 1), (("counts", "consumed_rows"), 2),
        (("counts", "requested_rows"), 12), (("error",), sentinel),
        (("prediction", "columns"), 2), (("pins", "checkpoint_sha256"), pin("wrong"))):
        values = fixture()
        target = values[2]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        add("response_" + "_".join(path), values, True, False)
    values = fixture()
    values[2]["prediction"]["row_ids"].reverse()
    add("cached_query_order", values, True, False)
    values = fixture()
    values[2]["debug"] = sentinel
    add("response_private_debug", values, True, False)
    for bad in (float("nan"), float("inf")):
        values = fixture()
        data = struct.pack("<dd", bad, 90.)
        values[3]["prediction"] = data
        values[2]["prediction"]["sha256"] = hashlib.sha256(data).hexdigest()
        add("nonfinite_" + str(bad), values, True, False)
    values = fixture(output=probability)
    descriptor, data = abi.pack_buffer(tuple(values[2]["prediction"]["row_ids"]), ((.2, .2), (.1, .1)))
    values[2]["prediction"] = descriptor
    values[3]["prediction"] = data
    add("invalid_probability_sum", values, True, False)
    values = fixture()
    values[2].update(status="unavailable", error="dependency_unavailable", prediction=None)
    values[2]["counts"]["consumed_rows"] = 0
    add("unavailable_still_unqualified", (values[0], values[1], values[2], {}))
    return cases


class WorkerAbiTests(unittest.TestCase):
    def test_shared_wire_cases(self):
        for case in wire_cases():
            with self.subTest(case=case["name"]):
                request = abi.decode_envelope(case["request"].encode()) if case["expected_request"] else json.loads(case["request"])
                buffers = {k: bytes(v) for k, v in case["buffers"].items()}
                if not case["expected_request"]:
                    with self.assertRaisesRegex(abi.AdapterError, "^worker_abi_invalid$") as caught:
                        abi.validate_request(request, buffers, settings()[0])
                    self.assertTrue(caught.exception.__suppress_context__)
                    continue
                self.assertEqual(abi.validate_request(request, buffers, settings()[0]), request["request_sha256"])
                response = json.loads(case["response"])
                result_buffers = {k: bytes(v) for k, v in case["response_buffers"].items()}
                if case["expected_response"]:
                    self.assertEqual(abi.validate_response(response, result_buffers, request, buffers, settings()[0]),
                                     case["expected_receipt"])
                else:
                    with self.assertRaisesRegex(abi.AdapterError, "^worker_abi_invalid$"):
                        abi.validate_response(response, result_buffers, request, buffers, settings()[0])

    def test_bad_json_size_duplicates_scalars_and_diagnostics_are_redacted(self):
        for data in (b'{"type":"request","type":"response"}', b'{"secret":NaN}', b'null', b'[]',
                     b'"/private/checkpoint"', b'{}', b'"\xff"', b' ' * (abi.MAX_ENVELOPE + 1)):
            with self.subTest(data=data[:32]), self.assertRaisesRegex(abi.AdapterError, "^worker_abi_invalid$"):
                abi.decode_envelope(data)
        for rows in ((("private cell",),), ((10**400,),), ((float("nan"),),), ((1., 2.), (3.,))):
            with self.assertRaisesRegex(abi.AdapterError, "^worker_abi_invalid$"):
                abi.pack_buffer((pin("row"),), rows)

    def test_adapter_exports_full_provided_context_and_fit_seed(self):
        from research.transfer.tests.test_adapters import identity
        port = learner(identity(context_cap=2, context_policy="head"))
        port.prepare(training(), 19)
        pins, resources, _ = settings()
        pins["model_sha256"] = port.identity.sha256
        request, buffers = port.worker_request(pins, resources, seed=7, primary_n=3,
                                              query=features((3., 4.), "query"))
        self.assertEqual(request["fit_seed"], 19)
        self.assertEqual(request["counts"], dict(requested_rows=3, produced_rows=3, provided_rows=3, consumed_rows=2))
        self.assertEqual(request["context_row_ids"], list(training().features.row_ids[:2]))
        self.assertEqual(set(buffers), {"train_features", "train_labels", "query_features"})
        self.assertFalse(hasattr(features((3.,), "query"), "targets"))

    def test_n_four_n_and_all_functional_policies_change_identity(self):
        pins, resources, policy = settings()
        original, _ = abi.build_request(pins, resources, policy, "predict", 7,
                                       training=training(), query=features((3.,), "query"))
        changes = [({"predict_time_fit": True}, {}), ({"order_sensitive": True}, {}),
                   ({"query_policy": "transductive"}, {}), ({"subsampling": "seeded"}, {}),
                   ({"context_cap": 2, "truncation": "head"}, {"context_row_ids": training().features.row_ids[:2]})]
        for change, extra in changes:
            request, _ = abi.build_request(pins, resources, {**policy, **change}, "predict", 7,
                training=training(), query=features((3.,), "query"), **extra)
            self.assertNotEqual(request["request_sha256"], original["request_sha256"])
        larger = TrainingBatch(features(tuple(range(12))), tuple(range(12)), "synthetic_train")
        four, _ = abi.build_request(pins, resources, {**policy, "sample_multiplier": 4, "truncation": "head"},
            "predict", 7, training=larger, query=features((99.,), "query"), context_row_ids=larger.features.row_ids[:3])
        self.assertEqual(four["counts"], dict(requested_rows=12, produced_rows=12, provided_rows=12, consumed_rows=3))
        self.assertNotEqual(four["request_sha256"], original["request_sha256"])
        for field in ("seed", "fit_seed"):
            request = copy.deepcopy(original)
            request[field] += 1
            seal(request)
            self.assertNotEqual(request["request_sha256"], original["request_sha256"])

    def test_label_injection_and_schema_mismatch_rejected_by_builder(self):
        pins, resources, policy = settings()
        for query in (training(), {"features": features((3.,), "query"), "targets": (123.,)},
                      replace(features((3.,), "query"), schema_sha256=pin("other schema"))):
            with self.assertRaisesRegex(abi.AdapterError, "^worker_abi_invalid$"):
                abi.build_request(pins, resources, policy, "predict", 7, training=training(), query=query)

    def test_versions_and_owner_pins_are_not_worker_self_reports(self):
        request, buffers, _, _ = fixture()
        for version in ("latest", "1.2", "1.2.*", ">=1.2.3", "1.2.3 ", "v1.2.3"):
            changed = copy.deepcopy(request)
            changed["pins"]["runtime_version"] = version
            seal(changed)
            # Even an owner pin cannot admit a floating/ambiguous version.
            with self.assertRaisesRegex(abi.AdapterError, "^worker_abi_invalid$"):
                abi.validate_request(changed, buffers, changed["pins"])
        for key in request["pins"]:
            pins = copy.deepcopy(request["pins"])
            pins[key] = pin("other")
            with self.subTest(pin=key), self.assertRaisesRegex(abi.AdapterError, "^worker_abi_invalid$"):
                abi.validate_request(request, buffers, pins)

    def test_every_owner_pin_and_resource_limit_binds_cache_identity(self):
        request, buffers, response, result_buffers = fixture()
        for section in ("pins", "resources"):
            for key in request[section]:
                if key == "network":
                    continue
                changed = copy.deepcopy(request)
                value = changed[section][key]
                changed[section][key] = ({"runtime_version": "1.2.4", "precision": "float32", "device": "cuda"}.get(key, pin("other"))
                    if type(value) is str else value + 1)
                seal(changed)
                self.assertNotEqual(request["request_sha256"], changed["request_sha256"])
                with self.subTest(field=key), self.assertRaisesRegex(abi.AdapterError, "^worker_abi_invalid$"):
                    abi.validate_response(response, result_buffers, changed, buffers, changed["pins"])

    def test_schema_and_source_shape_contract(self):
        root = Path(__file__).resolve().parents[3]
        self.assertEqual(abi.SCHEMA["$schema"], "https://json-schema.org/draft/2020-12/schema")
        for path in ("research/transfer/adapters/base.py", "research/transfer/adapters/worker_abi.py",
                     "research/transfer/tests/test_worker_abi.py", "rust/evaluation.rs"):
            self.assertLess(len((root / path).read_text().splitlines()), 500, path)

    def test_typed_feature_schema_changes_cache_identity(self):
        pins, resources, policy = settings()
        train, query = training(), features((3.,), "query")
        original, _ = abi.build_request(pins, resources, policy, "predict", 7, training=train, query=query)
        other = pin("other typed schema")
        changed, _ = abi.build_request(pins, resources, policy, "predict", 7,
            training=replace(train, features=replace(train.features, schema_sha256=other)),
            query=replace(query, schema_sha256=other))
        self.assertNotEqual(original["request_sha256"], changed["request_sha256"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--emit-fixtures"]:
        print(canonical_json(wire_cases()).decode())
    else:
        unittest.main()
