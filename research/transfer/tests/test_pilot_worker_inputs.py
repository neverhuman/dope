"""Private role capture tests; no vendor model imports or scientific qualification."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import unittest

from research.transfer.pilot_worker_inputs import AssemblyBinding, _layout, _membership, _partition
from research.transfer.pilot_native_seed import NativeSeed, ORIGINAL_STREAM_HEX, OWNER_GRANT_SHA256
from research.transfer.pilot_worker_authority import WorkerAuthority


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def encoded(value):
    return json.dumps(value).encode()


def fixture():
    doc = {"schema_sha256": digest("schema"), "csv_sha256": digest("fit"),
           "selection_csv_sha256": digest("selection"), "query_csv_sha256": digest("query")}
    parts = {}
    for role, key in (("fit", "csv"), ("selection", "selection_csv"), ("query", "query_csv")):
        parts[role] = {"csv_sha256": doc[key + "_sha256"], "rows": [
            {"row_id": digest(role + str(i)), "physical_unit_sha256": digest("unit" + role + str(i))}
            for i in range(2)]}
    value = {"format": "dope-pilot-private-physical-membership-v1",
             "schema_sha256": doc["schema_sha256"], "partitions": parts}
    schema = {"format": "dope-table-schema", "version": 1, "columns": [
        {"id": "C0", "role": "feature", "dtype": "continuous", "source_name": "private_feature"},
        {"id": "C1", "role": "target", "dtype": "continuous", "source_name": "private_target"}]}
    return doc, value, schema


class PrivateInputCaptureTests(unittest.TestCase):
    def setUp(self):
        self.doc, self.value, self.schema = fixture()
        self.membership = _membership(encoded(self.value), self.doc)["fit"]
        self.layout = _layout(encoded(self.schema))

    def parse(self, raw, expected=2):
        return _partition(raw, expected, self.membership, self.layout, self.doc["schema_sha256"])

    def test_private_header_reordering_preserves_feature_and_target_roles(self):
        value = self.parse(b"private_target,private_feature\n4,1\n5,2\n")
        self.assertEqual(value.features.rows, ((1.0,), (2.0,)))
        self.assertEqual(value.targets, (4.0, 5.0))
        self.assertEqual(value.features.row_ids, self.membership[0])

    def test_exact_count_refuses_extra_and_missing_rows(self):
        for raw in (b"private_feature,private_target\n1,4\n",
                    b"private_feature,private_target\n1,4\n2,5\n3,6\n"):
            with self.assertRaises(ValueError):
                self.parse(raw)

    def test_same_physical_unit_with_different_row_ids_is_still_overlap(self):
        self.value["partitions"]["query"]["rows"][0]["physical_unit_sha256"] = (
            self.value["partitions"]["fit"]["rows"][0]["physical_unit_sha256"])
        with self.assertRaisesRegex(ValueError, "assembly_partition_overlap"):
            _membership(encoded(self.value), self.doc)

    def test_repeated_measurements_inside_one_role_preserve_shared_physical_unit(self):
        rows = self.value["partitions"]["fit"]["rows"]
        rows[1]["physical_unit_sha256"] = rows[0]["physical_unit_sha256"]
        result = _membership(encoded(self.value), self.doc)["fit"]
        self.assertEqual(result[1], (rows[0]["physical_unit_sha256"],) * 2)
        self.assertNotEqual(result[0][0], result[0][1])

    def test_same_row_id_with_distinct_unit_labels_is_overlap(self):
        self.value["partitions"]["selection"]["rows"][0]["row_id"] = (
            self.value["partitions"]["fit"]["rows"][0]["row_id"])
        with self.assertRaisesRegex(ValueError, "assembly_partition_overlap"):
            _membership(encoded(self.value), self.doc)

    def test_duplicate_row_id_inside_one_role_refuses(self):
        rows = self.value["partitions"]["fit"]["rows"]
        rows[1]["row_id"] = rows[0]["row_id"]
        with self.assertRaises(ValueError):
            _membership(encoded(self.value), self.doc)

    def test_wrong_csv_or_schema_membership_binding_refuses(self):
        for field in ("csv", "schema"):
            value = deepcopy(self.value)
            if field == "csv":
                value["partitions"]["fit"]["csv_sha256"] = digest("another_input")
            else:
                value["schema_sha256"] = digest("another_schema")
            with self.assertRaises(ValueError):
                _membership(encoded(value), self.doc)

    def test_nonnumeric_nonfinite_and_malformed_bytes_have_static_errors(self):
        for value in (b"PRIVATE_SENTINEL", b"nan", b"inf", b"1e999", b"\xff"):
            raw = b"private_feature,private_target\n" + value + b",4\n2,5\n"
            with self.assertRaises(ValueError) as caught:
                self.parse(raw)
            self.assertIn(str(caught.exception), ("assembly_csv_invalid", "assembly_numeric_invalid", "assembly_nonfinite_input"))
            self.assertNotIn("PRIVATE_SENTINEL", str(caught.exception))

    def test_duplicate_extra_or_missing_headers_refuse_without_echo(self):
        for header in (b"private_target,private_target", b"private_feature,foreign_column",
                       b"private_feature,private_target,extra"):
            with self.assertRaises(ValueError) as caught:
                self.parse(header + b"\n1,4\n2,5\n")
            self.assertEqual(str(caught.exception), "assembly_header_mismatch")

    def test_missing_or_multiple_targets_and_duplicate_schema_names_refuse(self):
        for modification in ("none", "two", "duplicate"):
            schema = deepcopy(self.schema)
            if modification == "none":
                schema["columns"][1]["role"] = "feature"
            elif modification == "two":
                schema["columns"][0]["role"] = "target"
            else:
                schema["columns"][1]["source_name"] = "private_feature"
            with self.assertRaises(ValueError):
                _layout(encoded(schema))


class ExplicitLearnerSeedBindingTests(unittest.TestCase):
    def authority(self, seed):
        pin = digest("seed-binding-fixture")
        return WorkerAuthority(input_sha256=pin, roles_path="/private/roles.json",
            contexts_path="/private/contexts.json", roles_sha256=pin, contexts_sha256=pin,
            source_sha256=pin, environment_sha256=pin, policy_sha256=pin,
            owner_recipe_decision_sha256=OWNER_GRANT_SHA256,
            native_seed_sha256=NativeSeed(seed).sha256, learner_seed=seed,
            null_reference_sha256=pin, real_source_artifact_sha256=pin,
            real_source_seed=0, real_seed_semantics="deterministic_partition_zero",
            owner_receipt_sha256=pin, resource_receipt_sha256=pin, rights_admission_sha256=pin)

    def test_full_uint64_exact_identity_survives_authority_and_assembly(self):
        for seed in ("0x0000000000000000", "0xffffffffffffffff", ORIGINAL_STREAM_HEX):
            with self.subTest(seed=seed):
                authority = self.authority(seed)
                binding = authority.binding(digest("cell"), digest("draw"))
                self.assertIs(type(binding), AssemblyBinding)
                self.assertEqual(binding.learner_seed, seed)
                self.assertEqual(binding.native_seed_sha256, NativeSeed(seed).sha256)
                self.assertEqual(authority.verify_learner_seed("fixture").original_stream_hex, seed)
        self.assertEqual(self.authority(ORIGINAL_STREAM_HEX).verify_learner_seed("development").seed32,
                         2858805439)

    def test_hash_mismatch_refuses_both_authority_and_assembly(self):
        authority = self.authority(ORIGINAL_STREAM_HEX)
        binding = authority.binding(digest("cell"), digest("draw"))
        for value in (authority, binding):
            with self.subTest(type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "learner_seed_identity_mismatch"):
                    replace(value, native_seed_sha256=NativeSeed("0x000d0beef2026101").sha256)

    def test_seed_wire_requires_canonical_hex_not_lossy_number_or_implicit_default(self):
        authority = self.authority(ORIGINAL_STREAM_HEX)
        for seed in (True, 2858805439, 18446744073709551615, 1.0,
                     "0x1", "0xFFFFFFFFFFFFFFFF", "0x10000000000000000"):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    replace(authority, learner_seed=seed)
        from dataclasses import asdict
        fields = asdict(authority)
        del fields["learner_seed"]
        with self.assertRaises(TypeError):
            WorkerAuthority(**fields)

    def test_development_checks_owner_stream_and_fixed_recipe_not_just_matching_hashes(self):
        authority = self.authority("0x000d0beef2026101")
        with self.assertRaisesRegex(ValueError, "worker_owner_learner_stream_mismatch"):
            authority.verify_learner_seed("development")
        authority = replace(self.authority(ORIGINAL_STREAM_HEX),
                            owner_recipe_decision_sha256=digest("different-owner-recipe"))
        with self.assertRaisesRegex(ValueError, "worker_owner_learner_stream_mismatch"):
            authority.verify_learner_seed("development")
