"""Synthetic DOPERAW1 framing and the retained DOPERAW2 fixture.

The 707-byte fixture is the file next to this module. It is read by its recorded
hash. It is not rendered or regenerated.
No native CLI, source rows, or prediction goldens.
"""

import hashlib
import json
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
    account_mixed_artifact,
    account_raw_artifact,
    declare_raw_columns,
)

ROOT = Path(__file__).resolve().parents[3]
ROW_MARKER = "row-marker-offset-scale-99999"
WRONG_DIGEST = b"0123456789abcdef0123456789abcdef"
RETAINED_RAW = Path(__file__).resolve().parent / "fixtures" / "retained-doperaw2-707.raw"
RETAINED_SHA256 = "fac2d6b32a59ac67a86571b57c0ee24669a6d865c0c91246ed2e80c3dc567d22"
RETAINED_MIXED = Path(__file__).resolve().parent / "fixtures" / "retained-dopmix01-1523.mixed"
RETAINED_MIXED_SHA256 = "d13c986a437af7e17c8b4b274b19fe9131b05d783b4847ad9ead0dc397a97912"
RETAINED_MIXED_NATIVE_SHA256 = "f5d6686b28caa71f68d05f4d48b0d39eea3133394fd449c1a6435ff7974d3a60"


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


def noise_dpk(body, *, minor=3, task=0, features=1, compress=False):
    if compress:
        section = bytes([5 | 0x80, len(body), len(body)]) + body
    else:
        section = bytes([5, len(body)]) + body
    artifact = bytearray(56 + len(section))
    artifact[:4] = b"DPK3"
    artifact[4] = 3
    artifact[5] = minor
    artifact[6] = 1
    artifact[7] = task
    artifact[8] = 4
    artifact[12:16] = features.to_bytes(4, "little")
    artifact[32:40] = len(section).to_bytes(8, "little")
    artifact[56:] = section
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
        state = ROW_MARKER.encode()
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
        self.assertNotIn(ROW_MARKER, rendered)
        self.assertNotIn(WRONG_DIGEST.decode(), rendered)
        self.assertNotIn("offset", profile)
        self.assertNotIn("scale", profile)
        with self.assertRaises(TypeError):
            profile["framing_bytes"] = 0

    def test_wrong_digest_is_not_an_inspect_replay(self):
        state = ROW_MARKER.encode()
        payload = dpk()
        good = framed(state, payload)
        flipped_digest = bytearray(good)
        flipped_digest[24] ^= 1
        flipped_state = bytearray(good)
        flipped_state[56] ^= 1
        cases = (
            bytes(flipped_digest),
            bytes(flipped_state),
            framed(state, payload, digest=WRONG_DIGEST),
        )
        for artifact in cases:
            with self.subTest(prefix=artifact[24:28]):
                with self.assertRaises(BundleError) as raised:
                    account_raw_artifact(artifact)
                self.assertEqual(raised.exception.code, "raw_artifact_integrity_failed")
                self.assertEqual(str(raised.exception), "raw_artifact_integrity_failed")
                self.assertNotIn(ROW_MARKER, str(raised.exception))
                self.assertNotIn(WRONG_DIGEST.decode(), str(raised.exception))

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
        state = ROW_MARKER.encode()
        self.assertEqual(framed(state, payload)[24:56], blake3_hash(state + payload))

    def test_bad_framing_refuses_without_echoing_the_artifact(self):
        state = ROW_MARKER.encode()
        payload = dpk()
        cases = {
            "raw_artifact_invalid": [
                b"",
                b"DOPERAW1",
                bytearray(framed(state, payload)),
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
                framed(state, b"DPK3" + ROW_MARKER.encode()),
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
                    self.assertNotIn(ROW_MARKER, str(raised.exception))

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
            {"id": "c0001", "role": "feature", "dtype": "continuous", "label": "column-label-marker", "units": "column-unit-marker"},
            {"id": "c0002", "role": "feature", "dtype": "continuous", "mask_states": []},
            {"id": "excluded-entity-marker", "role": "excluded_sensitive_identifier", "dtype": "nominal"},
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
        self.assertNotIn("column-label-marker", rendered)
        self.assertNotIn("column-unit-marker", rendered)
        self.assertNotIn("excluded-entity-marker", rendered)

    def test_mixed_and_mask_states_are_explicit_refusals(self):
        def column(dtype="continuous", mask=None, role="feature"):
            item = {"id": "column-name-marker", "role": role, "dtype": dtype}
            if mask is not None:
                item["mask_states"] = mask
            return item

        for dtype in ("nominal", "integer", "boolean", "ordinal", "datetime", "text"):
            with self.subTest(dtype=dtype):
                with self.assertRaises(BundleError) as raised:
                    declare_raw_columns([column(dtype), column(role="target")])
                self.assertEqual(raised.exception.code, "raw_profile_requires_continuous_columns")
                self.assertNotIn("column-name-marker", str(raised.exception))
                self.assertNotIn(dtype, str(raised.exception))
        for state in ("missing", "unknown_category", "structural_absence", "padding", "censored"):
            with self.subTest(mask=state):
                with self.assertRaises(BundleError) as raised:
                    declare_raw_columns([column(mask=[state]), column(role="target")])
                self.assertEqual(raised.exception.code, "raw_profile_requires_observed_cells")
                self.assertNotIn(state, str(raised.exception))
                self.assertNotIn("column-name-marker", str(raised.exception))
        with self.assertRaises(BundleError) as raised:
            declare_raw_columns([{"id": "c0001", "role": "feature", "dtype": "continuous", "source_name": ROW_MARKER}])
        self.assertEqual(raised.exception.code, "schema_mismatch")
        self.assertNotIn(ROW_MARKER, str(raised.exception))
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
                self.assertNotIn(ROW_MARKER, str(raised.exception))


class ResidualRawTests(unittest.TestCase):
    def test_retained_fixture_charges_full_bytes_without_fitted_numbers(self):
        artifact = RETAINED_RAW.read_bytes()
        self.assertEqual(len(artifact), 707)
        self.assertEqual(hashlib.sha256(artifact).hexdigest(), RETAINED_SHA256)
        self.assertEqual(artifact[:8], b"DOPERAW2")
        self.assertEqual(int.from_bytes(artifact[8:16], "little"), 392)
        self.assertEqual(int.from_bytes(artifact[16:24], "little"), 259)
        self.assertEqual(artifact[24:56], blake3_hash(artifact[56:]))
        self.assertTrue(artifact.endswith(b"\x05\x03\x04\x01\x00"))
        profile = account_raw_artifact(artifact)
        self.assertEqual(profile["byte_profile_id"], "native-raw-complete-residual-v1")
        self.assertEqual(profile["version"], 2)
        self.assertEqual(profile["profile"], "raw-numeric-residual-v1")
        self.assertEqual(profile["capability"], "fully_observed_numeric_features_continuous_regression")
        self.assertIsNone(profile["mathematical_parameter_count"])
        self.assertEqual(profile["framing_bytes"], 56)
        self.assertEqual(profile["transform_schema_bytes"], 392)
        self.assertEqual(profile["native_payload_bytes"], 259)
        self.assertEqual(profile["complete_sampling_bytes"], 707)
        self.assertEqual(profile["transport_archive_bytes"], 707)
        self.assertIsNone(profile["generic_runtime_bytes"])
        self.assertIsNone(profile["cold_sampling_closure_bytes"])
        self.assertEqual(profile["content_hash"], "blake3_state_plus_dpk")
        self.assertIs(profile["formal_dp"], False)
        self.assertIs(profile["hipaa_deidentification"], False)
        self.assertIs(profile["release_authorized"], False)
        self.assertEqual(profile["confirmation"], "unqualified")
        self.assertEqual(profile["certification"], "unqualified_development_artifact")
        rendered = repr(dict(profile))
        state = json.loads(artifact[56:448])
        self.assertNotIn(state["format"], rendered)
        self.assertNotIn(state["visibility"], rendered)
        for column in state["columns"]:
            self.assertNotIn(column["id"], rendered)
            for key in ("offset", "scale"):
                self.assertNotIn(json.dumps(column[key]), rendered)
        for key in ("offset", "scale", "knot", "label", "domain"):
            self.assertNotIn(key, profile)
        with self.assertRaises(TypeError):
            profile["mathematical_parameter_count"] = 0

    def test_each_uncompressed_kind_charges_without_keeping_its_integers(self):
        bodies = {
            0: bytes([4, 1, 0]),
            1: bytes([4, 1, 1, 0x80, 0x01, 0x00]),
            2: bytes([4, 1, 2, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]),
            3: bytes([4, 1, 3]) + bytes(17),
        }
        state = ROW_MARKER.encode()
        for kind, body in bodies.items():
            with self.subTest(kind=kind):
                profile = account_raw_artifact(framed(state, noise_dpk(body), magic=b"DOPERAW2"))
                self.assertEqual(profile["version"], 2)
                self.assertEqual(profile["profile"], "raw-numeric-residual-v1")
                self.assertIsNone(profile["mathematical_parameter_count"])
                self.assertNotIn(128, profile.values())
                self.assertNotIn(65535, profile.values())
                self.assertNotIn(ROW_MARKER, repr(dict(profile)))
                self.assertNotIn("knot", profile)

    def test_profile_mismatch_stays_closed(self):
        state = ROW_MARKER.encode()
        cases = [
            noise_dpk(bytes([4, 1, 0]), minor=2),
            noise_dpk(bytes([4, 1, 0]), task=1),
            noise_dpk(bytes([4, 1, 0]), compress=True),
            noise_dpk(bytes([4, 1, 0, 0x00])),
            noise_dpk(bytes([4, 1, 3]) + bytes(16)),
            noise_dpk(bytes([0, 0, 0])),
            dpk(),
        ]
        for payload in cases:
            with self.subTest(payload=payload[:8]):
                with self.assertRaises(BundleError) as raised:
                    account_raw_artifact(framed(state, payload, magic=b"DOPERAW2"))
                self.assertEqual(raised.exception.code, "raw_residual_profile_mismatch")
                self.assertEqual(str(raised.exception), "raw_residual_profile_mismatch")
                self.assertNotIn(ROW_MARKER, str(raised.exception))

    def test_doperaw1_keeps_historical_noise_and_refuses_explicit_residual(self):
        state = ROW_MARKER.encode()
        historical = account_raw_artifact(framed(state, noise_dpk(bytes([0, 0, 0]), minor=2)))
        self.assertEqual(historical["byte_profile_id"], "native-raw-complete-v1")
        self.assertEqual(historical["version"], 1)
        self.assertEqual(historical["profile"], "raw-continuous-symbolic-v1")
        self.assertEqual(historical["capability"], "fully_observed_continuous_regression")
        self.assertNotIn("mathematical_parameter_count", historical)
        with self.assertRaises(BundleError) as raised:
            account_raw_artifact(framed(state, noise_dpk(bytes([4, 1, 0]))))
        self.assertEqual(raised.exception.code, "raw_residual_profile_mismatch")
        self.assertNotIn(ROW_MARKER, str(raised.exception))
        flipped = bytearray(framed(state, noise_dpk(bytes([4, 1, 0])), magic=b"DOPERAW2"))
        flipped[24] ^= 1
        with self.assertRaises(BundleError) as raised:
            account_raw_artifact(bytes(flipped))
        self.assertEqual(raised.exception.code, "raw_artifact_integrity_failed")

    def test_verified_bundle_charges_the_retained_residual_file(self):
        artifact = RETAINED_RAW.read_bytes()
        manifest = {
            "amortization_count": 1,
            "bundle_id": "synthetic-residual",
            "capabilities": ["byte-inventory", "native-raw-sample"],
            "components": [
                component(
                    "payload",
                    "sampling_payload",
                    "doperaw1",
                    "model.raw",
                    artifact,
                    required=True,
                    consumed=True,
                ),
            ],
            "format": "dope.generator-bundle/v1",
            "sampling_handler": "native-raw-complete-v1",
            "transport_framing": "included",
            "version": 1,
        }
        directory, root, digest = stage(manifest, {"model.raw": artifact})
        with directory:
            verified = verify_bundle(root, digest)
        profile = verified.accounting["raw_profile"]
        self.assertEqual(profile["byte_profile_id"], "native-raw-complete-residual-v1")
        self.assertEqual(profile["complete_sampling_bytes"], 707)
        self.assertEqual(profile["transform_schema_bytes"], 392)
        self.assertEqual(profile["native_payload_bytes"], 259)
        self.assertIsNone(profile["mathematical_parameter_count"])
        self.assertEqual(verified.retained_payload, artifact)
        self.assertIsNone(verified.features)
        self.assertEqual(preflight_dpk(artifact[448:])["minor"], 3)
        self.assertEqual(preflight_dpk(artifact[448:])["features"], 1)
        rendered = repr(dict(profile))
        self.assertNotIn("dope-affine-state", rendered)
        self.assertNotIn(ROW_MARKER, rendered)


def _private_tokens(value):
    found = []
    if type(value) is str:
        found.append(value)
    elif type(value) in (int, float) and type(value) is not bool:
        found.append(json.dumps(value))
    elif type(value) is dict:
        for item in value.values():
            found.extend(_private_tokens(item))
    elif type(value) is list:
        for item in value:
            found.extend(_private_tokens(item))
    return found


class MixedChargeTests(unittest.TestCase):
    def test_retained_fixture_charges_full_bytes_without_fitted_fields(self):
        artifact = RETAINED_MIXED.read_bytes()
        self.assertEqual(len(artifact), 1523)
        self.assertEqual(hashlib.sha256(artifact).hexdigest(), RETAINED_MIXED_SHA256)
        self.assertEqual(artifact[:8], b"DOPMIX01")
        self.assertEqual(int.from_bytes(artifact[8:16], "little"), 1085)
        self.assertEqual(int.from_bytes(artifact[16:24], "little"), 382)
        self.assertEqual(artifact[24:56], blake3_hash(artifact[56:]))
        native = artifact[56 + 1085:]
        self.assertEqual(hashlib.sha256(native).hexdigest(), RETAINED_MIXED_NATIVE_SHA256)
        self.assertEqual(native[:4], b"DPK3")
        self.assertEqual(native[5], 2)
        profile = account_mixed_artifact(artifact)
        self.assertEqual(profile["byte_profile_id"], "native-mixed-complete-v1")
        self.assertEqual(profile["format"], "dope-mixed-sampling-artifact")
        self.assertEqual(profile["version"], 1)
        self.assertEqual(profile["profile"], "numeric_split_v1")
        self.assertIsNone(profile["mathematical_parameter_count"])
        self.assertEqual(profile["framing_bytes"], 56)
        self.assertEqual(profile["transform_schema_bytes"], 1085)
        self.assertEqual(profile["native_payload_bytes"], 382)
        self.assertEqual(profile["complete_sampling_bytes"], 1523)
        self.assertEqual(profile["transport_archive_bytes"], 1523)
        self.assertIsNone(profile["generic_runtime_bytes"])
        self.assertIsNone(profile["cold_sampling_closure_bytes"])
        self.assertEqual(profile["content_hash"], "blake3_state_plus_dpk")
        self.assertIs(profile["formal_dp"], False)
        self.assertIs(profile["hipaa_deidentification"], False)
        self.assertIs(profile["release_authorized"], False)
        self.assertEqual(profile["confirmation"], "unqualified")
        self.assertEqual(profile["certification"], "unqualified_development_artifact")
        rendered = repr(dict(profile))
        state = json.loads(artifact[56:56 + 1085])
        self.assertNotIn(state["format"], rendered)
        self.assertNotIn(state["affine"]["format"], rendered)
        self.assertNotIn(state["affine"]["visibility"], rendered)
        for column in state["affine"]["columns"]:
            self.assertNotIn(column["id"], rendered)
        for feature in state["features"]:
            self.assertNotIn(feature["id"], rendered)
            for category in feature["categories"]:
                self.assertNotIn(category, rendered)
        public_numbers = {"0", "1", "56", "1085", "382", "1523"}
        for token in _private_tokens({
            "affine": state["affine"],
            "features": state["features"],
        }):
            if not re.fullmatch(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", token):
                continue
            if token in public_numbers:
                continue
            self.assertNotIn(token, rendered)
        for key in (
            "affine", "categories", "probabilities", "completion", "split", "threshold",
            "offset", "scale", "fit_rows", "columns", "constant_target", "visibility",
            "max_field_bytes", "id",
        ):
            self.assertNotIn(key, profile)
        with self.assertRaises(TypeError):
            profile["mathematical_parameter_count"] = 0
        with self.assertRaises(BundleError) as raised:
            account_raw_artifact(artifact)
        self.assertEqual(raised.exception.code, "raw_artifact_invalid")
        self.assertEqual(str(raised.exception), "raw_artifact_invalid")

    def test_raw_magic_and_broken_framing_stay_outside_this_charge(self):
        mixed = RETAINED_MIXED.read_bytes()
        raw = RETAINED_RAW.read_bytes()
        with self.assertRaises(BundleError) as raised:
            account_mixed_artifact(raw)
        self.assertEqual(raised.exception.code, "mixed_artifact_invalid")
        self.assertNotIn(ROW_MARKER, str(raised.exception))
        flipped = bytearray(mixed)
        flipped[80] ^= 1
        with self.assertRaises(BundleError) as raised:
            account_mixed_artifact(bytes(flipped))
        self.assertEqual(str(raised.exception), "mixed_artifact_integrity_failed")
        replaced = bytearray(mixed)
        replaced[56 + 1085:] = b"\x00" * 382
        replaced[24:56] = blake3_hash(bytes(replaced[56:]))
        with self.assertRaises(BundleError) as raised:
            account_mixed_artifact(bytes(replaced))
        self.assertEqual(str(raised.exception), "mixed_payload_limits_invalid")
        state = json.loads(mixed[56:56 + 1085])
        state["profile"] = ROW_MARKER
        refused = framed(json.dumps(state).encode(), dpk(), magic=b"DOPMIX01")
        with self.assertRaises(BundleError) as raised:
            account_mixed_artifact(refused)
        self.assertEqual(str(raised.exception), "mixed_state_invalid")
        self.assertNotIn(ROW_MARKER, str(raised.exception))

    def test_verified_bundle_does_not_charge_mixed_through_the_raw_handler(self):
        artifact = RETAINED_MIXED.read_bytes()
        manifest = {
            "amortization_count": 1,
            "bundle_id": "synthetic-mixed",
            "capabilities": ["byte-inventory", "native-raw-sample"],
            "components": [
                component(
                    "payload",
                    "sampling_payload",
                    "doperaw1",
                    "model.raw",
                    artifact,
                    required=True,
                    consumed=True,
                ),
            ],
            "format": "dope.generator-bundle/v1",
            "sampling_handler": "native-raw-complete-v1",
            "transport_framing": "included",
            "version": 1,
        }
        directory, root, digest = stage(manifest, {"model.raw": artifact})
        with directory:
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "raw_artifact_invalid")
        self.assertNotIn(b"DOPMIX01".decode(), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
