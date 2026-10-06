"""Generated metadata controls; no files, learned bodies, native code or GPU.

Requested width, fit seed and readout mode are recorded caller inputs. The
inspector establishes the emitted opcode family and byte count, not those
training settings, runtime qualification, validation efficacy or eligibility.
"""

import hashlib
import itertools
import json
import unittest
from unittest.mock import patch

from research.benchmark import research_ablation_binding as binding


def pin(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def generated(opcode="sparse_linear", width="linear"):
    inspection = {
        "target": opcode, "format": "dope-kernel-v3", "version": 3,
        "decoder_id": 1, "rows_fitted": 24, "artifact_bytes": 137,
        "schema": ["PRIVATE_HEADER_SENTINEL"],
        "sexpr": "PRIVATE_COEFFICIENT_SENTINEL",
    }
    inspector_bytes = json.dumps(inspection, sort_keys=True).encode()
    log = b"".join(
        f"{step}\t1.00000000\t0.50000000\n".encode() for step in range(512)
    )
    request = {
        "requested_profile": f"grid_features6_width{width}_steps512_readoutoff",
        "fit_seed": 11, "candidate_slot": "compact_neural_residual",
        "source_bundle_ref": pin(b"generated opaque source reference"),
        "binary_ref": pin(b"generated opaque binary reference"),
        "runtime_ref": pin(b"generated opaque runtime reference"),
        "artifact_ref": {"bytes": 137, "sha256": "a" * 64},
        "inspection_ref": pin(inspector_bytes), "loss_log_ref": pin(log),
    }
    return request, inspector_bytes, log


class StrSubclass(str):
    pass


class IntSubclass(int):
    pass


class DictSubclass(dict):
    pass


class BytesSubclass(bytes):
    pass


class ResearchAblationBindingTests(unittest.TestCase):
    def reject(self, request, inspector_bytes, log):
        with self.assertRaises(binding.BindingGap):
            binding.bind_ablation_metadata(request, inspector_bytes, log)

    def changed_inspection(self, key, value):
        request, inspector_bytes, log = generated()
        inspection = json.loads(inspector_bytes)
        inspection[key] = value
        inspector_bytes = json.dumps(inspection, sort_keys=True).encode()
        request["inspection_ref"] = pin(inspector_bytes)
        return request, inspector_bytes, log

    def test_exact54_profiles(self):
        names = set()
        for features, width, steps, readout in itertools.product(
            [6, 12, 24], ["linear", "8", "16"], [512, 2048, 8192], ["on", "off"]
        ):
            name = f"grid_features{features}_width{width}_steps{steps}_readout{readout}"
            settings = binding.profile_settings(name)
            self.assertEqual(settings["feature_limit"], features)
            self.assertEqual(settings["requested_width"], width)
            self.assertEqual(settings["optimizer_steps"], steps)
            self.assertEqual(settings["readout_refit"], readout == "on")
            names.add(name)
        self.assertEqual(len(names), 54)

    def test_unknown_or_noncanonical_profile_refuses(self):
        for name in [
            "features12_steps2048", "grid_features06_width8_steps512_readouton",
            "grid_features6_width0_steps512_readouton",
            "grid_features6_width32_steps512_readouton",
            "grid_features6_width8_steps513_readouton",
            "grid_features6_width8_steps512_readouttrue",
            "grid_features6_width8_steps512_readouton ",
        ]:
            with self.subTest(profile=name), self.assertRaises(binding.BindingGap):
                binding.profile_settings(name)

    def test_linear_opcodes_are_separate_from_candidate_slot(self):
        for opcode in ["sparse_linear", "sparse_logistic"]:
            with self.subTest(opcode=opcode):
                result = binding.bind_ablation_metadata(*generated(opcode))
                self.assertEqual(result["actual_target_opcode"], opcode)
                self.assertEqual(result["requested_architecture"], "linear")
                self.assertEqual(result["candidate_slot"], "compact_neural_residual")
                self.assertIsNone(result["production_eligibility"])

    def test_neural_opcode_family_is_recorded(self):
        for width in ["8", "16"]:
            with self.subTest(width=width):
                result = binding.bind_ablation_metadata(
                    *generated("compact_neural_residual", width)
                )
                self.assertEqual(result["requested_width"], width)
                self.assertEqual(result["actual_target_opcode"], "compact_neural_residual")

    def test_architecture_opcode_disagreement_refuses(self):
        for opcode, width in [
            ("compact_neural_residual", "linear"),
            ("sparse_linear", "8"), ("sparse_logistic", "16"),
        ]:
            with self.subTest(opcode=opcode, width=width):
                self.reject(*generated(opcode, width))

    def test_unknown_opcode_refuses(self):
        for opcode in ["sparse_gam", "linear", "compact_neural_residual ", None]:
            with self.subTest(opcode=opcode):
                self.reject(*generated(opcode))

    def test_inspection_drift_refuses_before_json_parse(self):
        request, inspector_bytes, log = generated()
        # Same length drift excludes a size-only check; malformed JSON must not parse.
        drift = b"!" + inspector_bytes[1:]
        with patch.object(binding.json, "loads") as loads:
            with self.assertRaisesRegex(binding.BindingGap, "byte pin mismatch"):
                binding.bind_ablation_metadata(request, drift, log)
            loads.assert_not_called()

    def test_loss_drift_refuses_before_either_payload_parse(self):
        request, _, log = generated()
        malformed_inspector = b"{"
        request["inspection_ref"] = pin(malformed_inspector)
        # Pin verification precedes even the otherwise-invalid JSON parser.
        drift = b"!" + log[1:]
        with patch.object(binding.json, "loads") as loads:
            with self.assertRaisesRegex(binding.BindingGap, "byte pin mismatch"):
                binding.bind_ablation_metadata(request, malformed_inspector, drift)
            loads.assert_not_called()

    def test_requested_identity_requires_exact_builtin_types(self):
        cases = [
            ("candidate_slot", StrSubclass("compact_neural_residual")),
            ("candidate_slot", b"compact_neural_residual"),
            ("requested_profile", StrSubclass("grid_features6_widthlinear_steps512_readoutoff")),
            ("fit_seed", IntSubclass(11)), ("fit_seed", 11.0),
            ("fit_seed", True), ("fit_seed", -1), ("fit_seed", 2**64),
        ]
        for key, value in cases:
            with self.subTest(field=key, value_type=type(value).__name__):
                request, inspector_bytes, log = generated()
                request[key] = value
                self.reject(request, inspector_bytes, log)

    def test_pin_request_and_payload_types_are_exact(self):
        request, inspector_bytes, log = generated()
        self.reject(DictSubclass(request), inspector_bytes, log)
        self.reject(request, BytesSubclass(inspector_bytes), log)
        self.reject(request, inspector_bytes, BytesSubclass(log))
        for key, value in [
            ("bytes", True), ("bytes", 137.0), ("bytes", IntSubclass(137)),
            ("sha256", StrSubclass("a" * 64)), ("sha256", "A" * 64),
        ]:
            with self.subTest(pin_field=key, value_type=type(value).__name__):
                request, inspector_bytes, log = generated()
                request["artifact_ref"][key] = value
                self.reject(request, inspector_bytes, log)
        request, inspector_bytes, log = generated()
        request["artifact_ref"] = DictSubclass(request["artifact_ref"])
        self.reject(request, inspector_bytes, log)

    def test_native_inspection_identity_and_charge_refuse_mismatch(self):
        for key, value in [
            ("format", "dope-kernel-v2"), ("version", 4), ("version", 3.0),
            ("decoder_id", True), ("decoder_id", 2), ("rows_fitted", 0),
            ("rows_fitted", 24.0), ("artifact_bytes", 138),
            ("artifact_bytes", 137.0),
        ]:
            with self.subTest(field=key, value=value):
                self.reject(*self.changed_inspection(key, value))

    def test_nonfinite_inspection_constants_refuse(self):
        request, inspector_bytes, log = generated()
        for constant in (b"NaN", b"Infinity", b"-Infinity"):
            payload = b'{"private_field":' + constant + b"," + inspector_bytes[1:]
            request["inspection_ref"] = pin(payload)
            self.reject(request, payload, log)

    def test_duplicate_inspection_field_refuses(self):
        request, inspector_bytes, log = generated()
        inspector_bytes = b'{"target":"sparse_linear",' + inspector_bytes[1:]
        request["inspection_ref"] = pin(inspector_bytes)
        self.reject(request, inspector_bytes, log)

    def test_incomplete_extra_or_unterminated_log_refuses(self):
        for change in [
            lambda log: log.split(b"\n", 1)[1],
            lambda log: log + b"512\t1.00000000\t0.50000000\n",
            lambda log: log[:-1],
        ]:
            request, inspector_bytes, log = generated()
            log = change(log)
            request["loss_log_ref"] = pin(log)
            self.reject(request, inspector_bytes, log)

    def test_step_sequence_or_column_shape_refuses(self):
        for row in [
            b"1\t1.00000000\t0.50000000\n",
            b"00\t1.00000000\t0.50000000\n",
            b"0\t1.00000000\n", b"0\t1.00000000\t0.50000000\textra\n",
        ]:
            request, inspector_bytes, log = generated()
            log = row + log.split(b"\n", 1)[1]
            request["loss_log_ref"] = pin(log)
            self.reject(request, inspector_bytes, log)

    def test_nonfinite_headered_or_non_ascii_log_refuses(self):
        for row in [
            b"0\tnan\t0.5\n", b"0\t1\tinf\n", b"0\t-inf\t1\n",
            b"step\ttrain\tvalidation\n", b"0\t\xff\t1\n",
        ]:
            request, inspector_bytes, log = generated()
            log = row + log.split(b"\n", 1)[1]
            request["loss_log_ref"] = pin(log)
            self.reject(request, inspector_bytes, log)

    def test_public_output_does_not_forward_private_inspection_or_loss_values(self):
        result = binding.bind_ablation_metadata(*generated())
        expected = {
            "version", "requested_profile", "requested_architecture", "requested_width",
            "feature_limit", "optimizer_steps", "readout_refit", "fit_seed", "candidate_slot",
            "actual_target_opcode", "source_bundle_ref", "binary_ref", "runtime_ref",
            "artifact_ref", "inspection_ref", "loss_log_ref", "validation_log_records",
            "validation_log_phase", "identity_scope", "production_eligibility",
        }
        self.assertEqual(set(result), expected)
        public = json.dumps(result, sort_keys=True)
        for private in ["PRIVATE_HEADER_SENTINEL", "PRIVATE_COEFFICIENT_SENTINEL",
                        "schema", "sexpr", "1.00000000", "0.50000000"]:
            self.assertNotIn(private, public)

    def test_requested_settings_are_recorded_inputs_not_artifact_proven_facts(self):
        request, inspector_bytes, log = generated("compact_neural_residual", "8")
        first = binding.bind_ablation_metadata(request, inspector_bytes, log)
        request["requested_profile"] = "grid_features6_width16_steps512_readouton"
        request["fit_seed"] = 23
        second = binding.bind_ablation_metadata(request, inspector_bytes, log)
        # Identical inspector/log pins: this API can attest only the opcode family,
        # byte charge and complete log, not infer training width/seed/readout.
        self.assertEqual(first["inspection_ref"], second["inspection_ref"])
        self.assertEqual(first["actual_target_opcode"], second["actual_target_opcode"])
        self.assertEqual(second["requested_width"], "16")
        self.assertEqual(second["fit_seed"], 23)
        self.assertTrue(second["readout_refit"])
        for field in ["actual_width", "verified_fit_seed", "verified_readout_refit"]:
            self.assertNotIn(field, second)
        self.assertIsNone(second["production_eligibility"])

    def test_missing_extra_or_private_request_fields_refuse(self):
        for modification in [
            lambda request: request.pop("runtime_ref"),
            lambda request: request.update(private_path="PRIVATE_PATH_SENTINEL"),
            lambda request: request["runtime_ref"].update(path="PRIVATE_PATH_SENTINEL"),
        ]:
            request, inspector_bytes, log = generated()
            modification(request)
            with self.assertRaises(binding.BindingGap) as caught:
                binding.bind_ablation_metadata(request, inspector_bytes, log)
            self.assertNotIn("PRIVATE_PATH_SENTINEL", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
