"""Synthetic DOPERAW1 framing. No native CLI, source rows, or prediction goldens."""

import hashlib
import re
from pathlib import Path
import tempfile
import unittest

from research.transfer.bundle import (
    CAPABILITIES,
    HANDLERS,
    KINDS,
    BundleError,
    account_inventory,
    preflight_dpk,
    verify_bundle,
)
from research.transfer.evidence import canonical_json
from research.transfer.blake3_digest import blake3_hash
from research.transfer.raw_profile import (
    RAW_CAPABILITIES,
    RAW_HANDLER,
    RAW_KIND,
    account_raw_artifact,
    declare_raw_columns,
)

ROOT = Path(__file__).resolve().parents[3]
SECRET = "secret_x_offset_scale_99999"


def workspace_temp():
    parent = ROOT / "target"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def dpk(features=2):
    artifact = bytearray(56)
    artifact[:4] = b"DPK3"
    artifact[4] = 3
    artifact[5] = 2
    artifact[6] = 1
    artifact[8] = 8
    artifact[12:16] = features.to_bytes(4, "little")
    return bytes(artifact)


def framed(state, payload, *, state_len=None, payload_len=None, magic=b"DOPERAW1", digest=None):
    artifact = bytearray()
    artifact += magic
    artifact += (len(state) if state_len is None else state_len).to_bytes(8, "little")
    artifact += (len(payload) if payload_len is None else payload_len).to_bytes(8, "little")
    artifact += blake3_hash(state + payload) if digest is None else digest
    artifact += state
    artifact += payload
    return bytes(artifact)


def component(identity, role, kind, path, blob, *, required, consumed, depends_on=()):
    return {
        "bytes": len(blob),
        "consumed_by_sampler": consumed,
        "depends_on": list(depends_on),
        "executable": False,
        "handler": "none",
        "id": identity,
        "kind": kind,
        "path": path,
        "required": required,
        "role": role,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def stage(manifest, files):
    raw = canonical_json(manifest)
    directory = workspace_temp()
    root = Path(directory.name)
    (root / "manifest.json").write_bytes(raw)
    for name, blob in files.items():
        (root / name).write_bytes(blob)
    return directory, root, hashlib.sha256(raw).hexdigest()


class RawChargeTests(unittest.TestCase):
    def test_length_identity_leaves_runtime_and_closure_unmeasured(self):
        state = SECRET.encode()
        payload = dpk()
        artifact = framed(state, payload)
        profile = account_raw_artifact(artifact)
        self.assertEqual(profile["byte_profile_id"], "native-raw-complete-v1")
        self.assertEqual(profile["framing_bytes"], 56)
        self.assertEqual(profile["transform_schema_bytes"], len(state))
        self.assertEqual(profile["native_payload_bytes"], len(payload))
        self.assertEqual(
            profile["framing_bytes"] + profile["transform_schema_bytes"] + profile["native_payload_bytes"],
            profile["complete_sampling_bytes"],
        )
        self.assertEqual(profile["complete_sampling_bytes"], len(artifact))
        self.assertEqual(profile["transport_archive_bytes"], len(artifact))
        self.assertIsNone(profile["generic_runtime_bytes"])
        self.assertIsNone(profile["cold_sampling_closure_bytes"])
        self.assertEqual(profile["mapped_runtime_cold_closure"], "separate_unmeasured")
        self.assertEqual(profile["content_hash"], "blake3_state_plus_dpk")
        self.assertEqual(profile["symbolic_backend"], "unverified_native_inspect")
        self.assertEqual(profile["feature_support"], "fit_range_bounded_historical_native_marginals")
        self.assertEqual(profile["target_support"], "unclipped_affine_regression_not_tail_qualified")
        self.assertEqual(profile["capability"], "fully_observed_continuous_regression")
        self.assertEqual(profile["confirmation"], "unqualified")
        self.assertEqual(profile["rights"], "unqualified")
        self.assertEqual(profile["official_assets"], "unqualified")
        self.assertIs(profile["formal_dp"], False)
        self.assertIs(profile["hipaa_deidentification"], False)
        self.assertIs(profile["release_authorized"], False)
        self.assertEqual(profile["tail_fidelity"], "unqualified")
        self.assertEqual(profile["learner_transfer"], "unqualified")
        self.assertIs(profile["declared_columns_checked"], False)
        rendered = repr(dict(profile))
        self.assertNotIn(SECRET, rendered)
        self.assertNotIn("SECRET_HASH", rendered)
        self.assertNotIn("offset", profile)
        self.assertNotIn("scale", profile)
        with self.assertRaises(TypeError):
            profile["framing_bytes"] = 0

    def test_wrong_digest_is_not_an_inspect_replay(self):
        state = SECRET.encode()
        payload = dpk()
        good = framed(state, payload)
        flipped_digest = bytearray(good)
        flipped_digest[24] ^= 1
        flipped_state = bytearray(good)
        flipped_state[56] ^= 1
        cases = (
            bytes(flipped_digest),
            bytes(flipped_state),
            framed(state, payload, digest=b"SECRET_HASH" + b"!" * 21),
        )
        for artifact in cases:
            with self.subTest(prefix=artifact[24:28]):
                with self.assertRaises(BundleError) as raised:
                    account_raw_artifact(artifact)
                self.assertEqual(raised.exception.code, "raw_artifact_integrity_failed")
                self.assertEqual(str(raised.exception), "raw_artifact_integrity_failed")
                self.assertNotIn(SECRET, str(raised.exception))
                self.assertNotIn("SECRET_HASH", str(raised.exception))

    def test_digest_matches_unkeyed_blake3_state_plus_payload(self):
        vectors = {
            0: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
            65: "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee",
            1024: "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
            1025: "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444",
        }
        for length, expected in vectors.items():
            material = bytes(index % 251 for index in range(length))
            self.assertEqual(blake3_hash(material).hex(), expected)
        payload = dpk()
        state = SECRET.encode()
        self.assertEqual(framed(state, payload)[24:56], blake3_hash(state + payload))

    def test_bad_framing_refuses_without_echoing_the_artifact(self):
        state = SECRET.encode()
        payload = dpk()
        cases = {
            "raw_artifact_invalid": [
                b"",
                b"DOPERAW1",
                bytearray(framed(state, payload)),
                framed(state, payload, magic=b"DOPERAW2"),
                framed(b"", payload),
            ],
            "raw_artifact_integrity_failed": [
                framed(state, payload, state_len=len(state) + 1),
                framed(state, payload, payload_len=len(payload) + 1),
                framed(state, payload) + b"\x00",
                framed(state, payload)[:-1],
                framed(state, payload, state_len=2**64 - 1, payload_len=len(payload)),
            ],
            "raw_payload_limits_invalid": [
                framed(state, b"DPK3" + SECRET.encode()),
                framed(state, dpk()[:-1]),
            ],
        }
        for code, artifacts in cases.items():
            for artifact in artifacts:
                with self.subTest(code=code):
                    with self.assertRaises(BundleError) as raised:
                        account_raw_artifact(bytes(artifact) if type(artifact) is not bytearray else artifact)
                    self.assertEqual(raised.exception.code, code)
                    self.assertEqual(str(raised.exception), code)
                    self.assertNotIn(SECRET, str(raised.exception))

    def test_historical_inventory_keys_do_not_gain_the_raw_charge(self):
        accounting = account_inventory(
            [{
                "id": "payload",
                "role": "sampling_payload",
                "measured_bytes": 4,
                "required": True,
                "depends_on": [],
            }],
            10,
            1,
            "unmeasured",
        )
        self.assertNotIn("raw_profile", accounting)
        self.assertNotIn("byte_profile_id", accounting)
        self.assertNotIn("generic_runtime_bytes", accounting)


class RawColumnTests(unittest.TestCase):
    def test_continuous_declaration_keeps_support_limits_and_drops_other_roles(self):
        declared = declare_raw_columns([
            {"id": "c0001", "role": "feature", "dtype": "continuous", "label": "secret_label", "units": "secret_unit"},
            {"id": "c0002", "role": "feature", "dtype": "continuous", "mask_states": []},
            {"id": "secret_entity", "role": "excluded_sensitive_identifier", "dtype": "nominal"},
            {"id": "c0003", "role": "target", "dtype": "continuous"},
        ])
        self.assertEqual(declared["feature_count"], 2)
        self.assertEqual(declared["target_count"], 1)
        self.assertEqual(declared["feature_support"], "fit_range_bounded_historical_native_marginals")
        self.assertEqual(declared["target_support"], "unclipped_affine_regression_not_tail_qualified")
        self.assertEqual(declared["mixed_columns"], "unsupported")
        self.assertEqual(declared["missing_mask"], "unsupported")
        self.assertEqual(declared["confirmation"], "unqualified")
        self.assertIs(declared["declared_columns_checked"], True)
        self.assertEqual(declared["rights"], "unqualified")
        self.assertEqual(declared["official_assets"], "unqualified")
        self.assertEqual(declared["content_hash"], "unverified_native_inspect")
        self.assertEqual(declared["symbolic_backend"], "unverified_native_inspect")
        rendered = repr(dict(declared))
        self.assertNotIn("secret_label", rendered)
        self.assertNotIn("secret_unit", rendered)
        self.assertNotIn("secret_entity", rendered)

    def test_mixed_and_mask_states_are_explicit_refusals(self):
        def column(dtype="continuous", mask=None, role="feature"):
            item = {"id": "secret_name", "role": role, "dtype": dtype}
            if mask is not None:
                item["mask_states"] = mask
            return item

        for dtype in ("nominal", "integer", "boolean", "ordinal", "datetime", "text"):
            with self.subTest(dtype=dtype):
                with self.assertRaises(BundleError) as raised:
                    declare_raw_columns([column(dtype), column(role="target")])
                self.assertEqual(raised.exception.code, "raw_profile_requires_continuous_columns")
                self.assertNotIn("secret_name", str(raised.exception))
                self.assertNotIn(dtype, str(raised.exception))
        for state in ("missing", "unknown_category", "structural_absence", "padding", "censored"):
            with self.subTest(mask=state):
                with self.assertRaises(BundleError) as raised:
                    declare_raw_columns([column(mask=[state]), column(role="target")])
                self.assertEqual(raised.exception.code, "raw_profile_requires_observed_cells")
                self.assertNotIn(state, str(raised.exception))
                self.assertNotIn("secret_name", str(raised.exception))
        with self.assertRaises(BundleError) as raised:
            declare_raw_columns([{"id": "c0001", "role": "feature", "dtype": "continuous", "source_name": SECRET}])
        self.assertEqual(raised.exception.code, "schema_mismatch")
        self.assertNotIn(SECRET, str(raised.exception))
        with self.assertRaises(BundleError) as raised:
            declare_raw_columns([column(), column(), column(role="target"), column(role="target")])
        self.assertEqual(raised.exception.code, "raw_sampler_closure_mismatch")


class RawHandlerTests(unittest.TestCase):
    def test_handler_vocabulary_matches_the_profile_id(self):
        self.assertEqual(RAW_HANDLER, "native-raw-complete-v1")
        self.assertEqual(RAW_KIND, "doperaw1")
        self.assertEqual(RAW_CAPABILITIES, ["byte-inventory", "native-raw-sample"])
        self.assertIn(RAW_HANDLER, HANDLERS)
        self.assertIn(RAW_KIND, KINDS)
        self.assertTrue(set(RAW_CAPABILITIES) <= CAPABILITIES)
        source = (ROOT / "research/transfer/raw_profile.py").read_text()
        digest_source = (ROOT / "research/transfer/blake3_digest.py").read_text()
        for blob in (source, digest_source):
            for token in ("subprocess", "socket", "urllib", "pickle", "requests", "115.625", "162.76041666666666"):
                self.assertNotIn(token, blob)
            self.assertIsNone(re.search(r"^\s*(import|from)\s+blake3\b", blob, re.M))
        bundle = (ROOT / "research/transfer/bundle.py").read_text()
        self.assertLessEqual(bundle.count("\n"), 500)
        self.assertNotIn("115.625", bundle)

    def test_verified_bundle_charges_the_file_exactly_and_keeps_historical_roles(self):
        state = b"fitted-state"
        payload = dpk(features=3)
        artifact = framed(state, payload)
        runtime = b"rtun"
        manifest = {
            "amortization_count": 1,
            "bundle_id": "synthetic-raw",
            "capabilities": ["byte-inventory", "native-raw-sample"],
            "components": [
                component("payload", "sampling_payload", "doperaw1", "model.raw", artifact, required=True, consumed=True),
                component("runtime", "runtime", "opaque", "runtime.bin", runtime, required=False, consumed=False),
            ],
            "format": "dope.generator-bundle/v1",
            "sampling_handler": "native-raw-complete-v1",
            "transport_framing": "included",
            "version": 1,
        }
        directory, root, digest = stage(manifest, {"model.raw": artifact, "runtime.bin": runtime})
        with directory:
            verified = verify_bundle(root, digest)
        profile = verified.accounting["raw_profile"]
        self.assertEqual(profile["complete_sampling_bytes"], len(artifact))
        self.assertEqual(profile["transport_archive_bytes"], len(artifact))
        self.assertEqual(profile["native_payload_bytes"], len(payload))
        self.assertEqual(profile["transform_schema_bytes"], len(state))
        self.assertEqual(verified.accounting["sampling_payload_bytes"], len(artifact))
        self.assertEqual(verified.accounting["transform_bytes"], 0)
        self.assertEqual(verified.accounting["runtime_package_bytes"], len(runtime))
        self.assertIsNone(profile["generic_runtime_bytes"])
        self.assertIsNone(profile["cold_sampling_closure_bytes"])
        self.assertGreater(
            verified.accounting["complete_delivery_bytes"],
            profile["transport_archive_bytes"],
        )
        self.assertIsNone(verified.features)
        self.assertEqual(verified.retained_payload, artifact)
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(verified.retained_payload)
        self.assertEqual(raised.exception.code, "unsupported_dpk_version")
        self.assertEqual(preflight_dpk(artifact[56 + len(state):])["features"], 3)
        with self.assertRaises(TypeError):
            verified.accounting["raw_profile"]["complete_sampling_bytes"] = 0

    def test_wrong_kind_capability_or_bare_dpk_handler_does_not_take_the_raw_charge(self):
        state = b"fitted-state"
        payload = dpk()
        artifact = framed(state, payload)

        def manifest_for(handler, capabilities, kind):
            return {
                "amortization_count": 1,
                "bundle_id": "synthetic-raw",
                "capabilities": capabilities,
                "components": [
                    component("payload", "sampling_payload", kind, "model.raw", artifact, required=True, consumed=True),
                ],
                "format": "dope.generator-bundle/v1",
                "sampling_handler": handler,
                "transport_framing": "unmeasured",
                "version": 1,
            }

        refusals = [
            ("native-raw-complete-v1", ["byte-inventory", "native-raw-sample"], "dpk3", "sampler_closure_mismatch"),
            ("native-raw-complete-v1", ["native-raw-sample", "byte-inventory"], "doperaw1", "sampler_closure_mismatch"),
            ("native-raw-complete-v1", ["byte-inventory"], "doperaw1", "sampler_closure_mismatch"),
            ("native-inline-dpk", ["byte-inventory", "native-dpk-sample"], "doperaw1", "sampler_closure_mismatch"),
            ("inventory-only", ["byte-inventory"], "doperaw1", "sampler_closure_mismatch"),
            ("native-inline-dpk", ["byte-inventory", "native-dpk-sample"], "dpk3", "unsupported_dpk_version"),
        ]
        for handler, capabilities, kind, code in refusals:
            with self.subTest(handler=handler, kind=kind, code=code):
                directory, root, digest = stage(
                    manifest_for(handler, capabilities, kind),
                    {"model.raw": artifact},
                )
                with directory:
                    with self.assertRaises(BundleError) as raised:
                        verify_bundle(root, digest)
                self.assertEqual(raised.exception.code, code)
                self.assertNotIn(SECRET, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
