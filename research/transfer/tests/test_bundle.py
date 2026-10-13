"""Synthetic bundle inventory only. No source rows, network, or sampler."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from research.transfer.bundle import (
    CAPABILITIES,
    FORMAT,
    HANDLERS,
    ROLES,
    BundleError,
    BundleLimits,
    VerifiedBundle,
    _scan_open_directory,
    account_inventory,
    charged_sampling_bytes,
    checked_sum,
    preflight_dpk,
    verify_bundle,
    within_budget,
)
from research.transfer.evidence import canonical_json

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = json.loads((ROOT / "schemas/bundle-golden-v1.json").read_bytes())
SCHEMA = json.loads((ROOT / "schemas/generator-bundle-v1.schema.json").read_text())


def stage(case, mutate=None):
    manifest = copy.deepcopy(case["manifest"])
    if mutate is not None:
        mutate(manifest)
    raw = canonical_json(manifest)
    directory = tempfile.TemporaryDirectory(dir=ROOT / "target")
    root = Path(directory.name)
    (root / "manifest.json").write_bytes(raw)
    for name, encoded in case["files_hex"].items():
        (root / name).write_bytes(bytes.fromhex(encoded))
    return directory, root, hashlib.sha256(raw).hexdigest()


class BundleGoldenTests(unittest.TestCase):
    def test_schema_is_closed_and_matches_the_inventory_vocabulary(self):
        self.assertIs(SCHEMA["additionalProperties"], False)
        self.assertEqual(SCHEMA["properties"]["format"]["const"], FORMAT)
        self.assertEqual(SCHEMA["properties"]["version"]["type"], "integer")
        self.assertEqual(SCHEMA["properties"]["version"]["const"], 1)
        self.assertEqual(set(SCHEMA["properties"]["sampling_handler"]["enum"]), HANDLERS)
        self.assertEqual(set(SCHEMA["properties"]["capabilities"]["items"]["enum"]), CAPABILITIES)
        self.assertEqual(set(SCHEMA["$defs"]["component"]["properties"]["role"]["enum"]), ROLES)
        self.assertIs(SCHEMA["$defs"]["component"]["additionalProperties"], False)
        self.assertIs(SCHEMA["$defs"]["component"]["properties"]["executable"]["const"], False)
        self.assertEqual(SCHEMA["$defs"]["component"]["properties"]["handler"]["const"], "none")

    def test_golden_accounting_and_byte_caps(self):
        self.assertEqual(canonical_json(GOLDEN), (ROOT / "schemas/bundle-golden-v1.json").read_bytes())
        charged = []
        for case in GOLDEN["cases"]:
            directory, root, digest = stage(case)
            with directory:
                self.assertEqual(digest, case["manifest_sha256"])
                verified = verify_bundle(root, digest)
            self.assertEqual(verified.accounting, case["accounting"])
            budget = verified.accounting["charged_sampling_bytes"]
            charged.append(budget)
            self.assertTrue(within_budget(budget, budget))
            self.assertFalse(within_budget(budget, budget - 1))
            self.assertTrue(within_budget(budget, budget + 1))
            self.assertGreaterEqual(
                verified.accounting["standalone_learned_state_bytes"],
                verified.accounting["amortized_learned_state_bytes"],
            )
        unmeasured, included = GOLDEN["cases"]
        self.assertNotIn("complete_delivery_bytes", unmeasured["accounting"])
        self.assertEqual(unmeasured["accounting"]["transport_framing"], "unmeasured")
        self.assertEqual(
            unmeasured["accounting"]["charged_sampling_bytes"],
            unmeasured["accounting"]["standalone_learned_state_bytes"]
            + unmeasured["accounting"]["runtime_package_bytes"],
        )
        self.assertEqual(
            included["accounting"]["charged_sampling_bytes"],
            included["accounting"]["standalone_learned_state_bytes"]
            + included["accounting"]["runtime_package_bytes"]
            + included["accounting"]["evidence_authorization_bytes"],
        )
        self.assertEqual(included["accounting"]["amortized_learned_state_bytes"], included["accounting"]["standalone_learned_state_bytes"])
        self.assertLess(
            unmeasured["accounting"]["amortized_learned_state_bytes"],
            unmeasured["accounting"]["standalone_learned_state_bytes"],
        )
        self.assertNotEqual(charged[0], charged[1])

    def test_false_declared_total_is_rejected(self):
        case = GOLDEN["cases"][0]
        directory, root, digest = stage(case, lambda manifest: manifest.__setitem__("charged_sampling_bytes", 1))
        with directory:
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "unknown_field")
        self.assertNotIn("charged_sampling_bytes", str(raised.exception))

    def test_missing_shared_state_stays_unresolved_at_any_amortization(self):
        case = GOLDEN["cases"][0]
        directory, root, digest = stage(case)
        with directory:
            (root / "shared.bin").unlink()
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "unresolved_dependency")

    def test_substituted_bytes_do_not_match_the_declared_identity(self):
        case = GOLDEN["cases"][0]
        directory, root, digest = stage(case)
        with directory:
            (root / "dictionary.bin").write_bytes(b"\x00\x00\x00")
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "dependency_identity_mismatch")

    def test_paths_symlinks_case_and_handlers_fail_closed(self):
        case = GOLDEN["cases"][0]

        def path(manifest):
            manifest["components"][0]["path"] = "../zz-secret-path"

        directory, root, digest = stage(case, path)
        with directory:
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "unsafe_path")
        self.assertNotIn("zz-secret-path", str(raised.exception))

        directory, root, digest = stage(case)
        with directory:
            target = root / "dictionary.bin"
            payload = target.read_bytes()
            target.unlink()
            target.symlink_to(root / "payload.bin")
            (root / "payload.bin").write_bytes(payload)
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "symlink_component")
        self.assertNotIn("dictionary", str(raised.exception))

        def collide(manifest):
            extra = copy.deepcopy(manifest["components"][0])
            extra["id"] = "dictionary-b"
            extra["path"] = "Dictionary.bin"
            manifest["components"].append(extra)
            manifest["components"].sort(key=lambda item: item["id"])

        directory, root, digest = stage(case, collide)
        with directory:
            (root / "Dictionary.bin").write_bytes(bytes.fromhex(case["files_hex"]["dictionary.bin"]))
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "case_collision")

        def executable(manifest):
            manifest["components"][0]["executable"] = True

        directory, root, digest = stage(case, executable)
        with directory:
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "executable_rejected")

        def handler(manifest):
            manifest["components"][0]["handler"] = "pickle"

        directory, root, digest = stage(case, handler)
        with directory:
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "unsupported_external_handler")

    def test_cycles_unbound_dependencies_and_bad_scalars(self):
        case = GOLDEN["cases"][0]

        def cycle(manifest):
            manifest["components"][0]["depends_on"] = ["payload"]
            manifest["components"][2]["depends_on"] = ["dictionary"]

        directory, root, digest = stage(case, cycle)
        with directory:
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "cyclic_dependency")

        def unbound(manifest):
            manifest["components"][2]["depends_on"] = ["dictionary", "missing", "runtime", "shared", "transform"]

        directory, root, digest = stage(case, unbound)
        with directory:
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "unbound_dependency")

        def boolean_count(manifest):
            manifest["components"][0]["bytes"] = True

        directory, root, digest = stage(case, boolean_count)
        with directory:
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, digest)
        self.assertEqual(raised.exception.code, "schema_mismatch")

        directory, root, _digest = stage(case)
        with directory:
            raw = (root / "manifest.json").read_bytes()
            broken = raw.replace(b'"version":1', b'"version":1,"version":1', 1)
            (root / "manifest.json").write_bytes(broken)
            with self.assertRaises(BundleError) as raised:
                verify_bundle(root, hashlib.sha256(broken).hexdigest())
        self.assertEqual(raised.exception.code, "duplicate_json_key")

    def test_checked_sum_overflow_and_module_has_no_sampler_runtime(self):
        with self.assertRaises(BundleError) as raised:
            checked_sum([2**64 - 1, 1])
        self.assertEqual(raised.exception.code, "integer_overflow")
        for name in ("bundle.py", "bundle_io.py"):
            source = (ROOT / "research/transfer" / name).read_text()
            for token in ("subprocess", "socket", "urllib", "pickle", "requests"):
                self.assertNotIn(token, source)
            self.assertLessEqual(source.count("\n"), 500)

    def test_dpk_preflight_rejects_bombs_flags_and_bad_dimensions(self):
        def header(features=1, flags=0, payload=b"", minor=2):
            artifact = bytearray(56 + len(payload))
            artifact[:4] = b"DPK3"
            artifact[4] = 3
            artifact[5] = minor
            artifact[6] = 1
            artifact[8] = 8
            artifact[10:12] = flags.to_bytes(2, "little")
            artifact[12:16] = features.to_bytes(4, "little")
            artifact[32:40] = len(payload).to_bytes(8, "little")
            artifact[56:] = payload
            return bytes(artifact)

        bomb = header(payload=bytes([0x81, 0x01, 0x80, 0x80, 0x80, 0x80, 0x08, 0x00]))
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(bomb)
        self.assertEqual(raised.exception.code, "expansion_cap_exceeded")
        self.assertLess(len(bomb), 128)
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(header(flags=2))
        self.assertEqual(raised.exception.code, "invalid_flags")
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(header(features=0))
        self.assertEqual(raised.exception.code, "invalid_dimensions")
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(header(payload=bytes([0x07, 0x00])))
        self.assertEqual(raised.exception.code, "unknown_section")
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(header()[:-1])
        self.assertEqual(raised.exception.code, "truncated_artifact")
        nan = b'{"value":NaN}'
        payload = bytes([0x06, len(nan)]) + nan
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(header(payload=payload))
        self.assertEqual(raised.exception.code, "nonfinite_value")
        self.assertNotIn("NaN", str(raised.exception))
        for minor in (0, 1, 2, 3):
            summary = preflight_dpk(header(minor=minor))
            self.assertEqual(summary["minor"], minor)
            self.assertIs(summary["v3"], True)
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(header(minor=4))
        self.assertEqual(raised.exception.code, "unsupported_dpk_version")

        def v2(minor):
            artifact = bytearray(header(minor=minor))
            artifact[:4] = b"DPK2"
            artifact[4] = 2
            return bytes(artifact)

        self.assertIs(preflight_dpk(v2(0))["v3"], False)
        self.assertEqual(preflight_dpk(v2(1))["minor"], 1)
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(v2(2))
        self.assertEqual(raised.exception.code, "unsupported_dpk_version")
        with self.assertRaises(BundleError) as raised:
            preflight_dpk(header(minor=1, payload=bytes([0x06, 0x00])))
        self.assertEqual(raised.exception.code, "unknown_section")
        self.assertEqual(preflight_dpk(header(minor=2, payload=bytes([0x06, 0x00])))["minor"], 2)

    def test_native_payload_is_retained_after_replacement(self):
        payload = bytearray(56)
        payload[:4] = b"DPK3"
        payload[4] = 3
        payload[5] = 2
        payload[6] = 1
        payload[8] = 8
        payload[12] = 1
        payload = bytes(payload)
        manifest = {
            "amortization_count": 1,
            "bundle_id": "synthetic-kernel",
            "capabilities": ["byte-inventory", "native-dpk-sample"],
            "components": [{
                "bytes": len(payload),
                "consumed_by_sampler": True,
                "depends_on": [],
                "executable": False,
                "handler": "none",
                "id": "payload",
                "kind": "dpk3",
                "path": "payload.dpk",
                "required": True,
                "role": "sampling_payload",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }],
            "format": FORMAT,
            "sampling_handler": "native-inline-dpk",
            "transport_framing": "unmeasured",
            "version": 1,
        }
        raw = canonical_json(manifest)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            (root / "manifest.json").write_bytes(raw)
            (root / "payload.dpk").write_bytes(payload)
            verified = verify_bundle(root, hashlib.sha256(raw).hexdigest())
            (root / "payload.dpk").write_bytes(b"\x00" * 56)
            self.assertEqual(verified.retained_payload, payload)
            self.assertEqual(verified.features, 1)
        measured = []
        for component in manifest["components"]:
            item = dict(component)
            item["measured_bytes"] = component["bytes"]
            measured.append(item)
        accounting = account_inventory(measured, len(raw), 1, "unmeasured")
        self.assertEqual(accounting["sampling_payload_bytes"], len(payload))
        self.assertNotIn("complete_delivery_bytes", accounting)
        limits = BundleLimits()
        self.assertEqual(limits.max_expanded_section_bytes, 16 * 1024 * 1024)
        self.assertEqual(limits.max_features, 4096)

    def test_noninteger_version_is_rejected(self):
        case = GOLDEN["cases"][0]
        for value, code in ((True, "unknown_version"), (1.0, "invalid_manifest")):
            with self.subTest(version_type=type(value).__name__):
                directory, root, digest = stage(
                    case, lambda manifest, value=value: manifest.__setitem__("version", value)
                )
                with directory, self.assertRaises(BundleError) as raised:
                    verify_bundle(root, digest)
                self.assertEqual(raised.exception.code, code)

    def test_verified_type_requires_verification(self):
        with self.assertRaises(BundleError) as raised:
            VerifiedBundle({"charged_sampling_bytes": 0}, None, None)
        self.assertEqual(raised.exception.code, "verification_required")

        class Forged(VerifiedBundle):
            def __init__(self):
                self._accounting = {"charged_sampling_bytes": 0}

        with self.assertRaises(BundleError) as raised:
            charged_sampling_bytes(Forged())
        self.assertEqual(raised.exception.code, "verification_required")

    def test_report_accounting_cannot_change_verified_charge(self):
        directory, root, digest = stage(GOLDEN["cases"][0])
        with directory:
            value = verify_bundle(root, digest)
            before = charged_sampling_bytes(value)
            try:
                value.accounting["charged_sampling_bytes"] = 0
            except (TypeError, AttributeError):
                pass
            self.assertEqual(before, 2121)
            self.assertEqual(charged_sampling_bytes(value), before)

    def test_directory_stream_stops_on_the_first_undeclared_or_cap(self):
        directory = tempfile.TemporaryDirectory(dir=ROOT / "target")
        root = Path(directory.name)
        (root / "extra.bin").write_bytes(b"x")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        with directory:
            fd = os.open(root, flags)
            try:
                with self.assertRaises(BundleError) as raised:
                    _scan_open_directory(fd, set(), 8, 8 * 64)
                self.assertEqual(raised.exception.code, "undeclared_component")
                self.assertNotIn("extra.bin", str(raised.exception))
            finally:
                os.close(fd)
            fd = os.open(root, flags)
            try:
                with self.assertRaises(BundleError) as raised:
                    _scan_open_directory(fd, set(), 0, 64)
                self.assertEqual(raised.exception.code, "too_many_components")
            finally:
                os.close(fd)
            fd = os.open(root, flags)
            try:
                with self.assertRaises(BundleError) as raised:
                    _scan_open_directory(fd, set(), 8, 1)
                self.assertEqual(raised.exception.code, "too_many_components")
            finally:
                os.close(fd)

    def test_replaced_directory_path_does_not_change_the_fd_listing(self):
        directory = tempfile.TemporaryDirectory(dir=ROOT / "target")
        base = Path(directory.name)
        real = base / "real"
        decoy = base / "decoy"
        real.mkdir()
        decoy.mkdir()
        (real / "manifest.json").write_bytes(b"{}")
        (real / "keep.bin").write_bytes(b"ab")
        (decoy / "manifest.json").write_bytes(b"{}")
        (decoy / "keep.bin").write_bytes(b"ab")
        (decoy / "secret.bin").write_bytes(b"hidden")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        with directory:
            fd = os.open(real, flags)
            try:
                os.rename(real, base / "real.moved")
                os.symlink(decoy, real)
                self.assertTrue(real.is_symlink())
                found = _scan_open_directory(
                    fd, {"manifest.json", "keep.bin"}, 33, 33 * 64
                )
            finally:
                os.close(fd)
        self.assertEqual(found, {"manifest.json", "keep.bin"})

    def test_optional_payload_dependency_stays_in_the_sampling_budget(self):
        def measured(identity, role, size, required, depends_on):
            return {
                "id": identity,
                "role": role,
                "measured_bytes": size,
                "required": required,
                "depends_on": depends_on,
            }

        def charge(components):
            return account_inventory(components, 10, 1, "unmeasured")["charged_sampling_bytes"]

        evidence = measured("evidence", "evidence_authorization", 7, False, [])
        self.assertEqual(
            charge([measured("payload", "sampling_payload", 4, True, ["evidence"]), evidence]),
            21,
        )
        self.assertEqual(
            charge([measured("payload", "sampling_payload", 4, True, []), evidence]),
            14,
        )
        self.assertEqual(
            charge([
                measured("payload", "sampling_payload", 4, True, ["rows"]),
                measured("rows", "materialized_rows", 9, False, []),
            ]),
            23,
        )
        dictionary = account_inventory(
            [
                measured("payload", "sampling_payload", 4, True, ["dictionary"]),
                measured("dictionary", "dictionary", 5, False, []),
            ],
            10,
            1,
            "unmeasured",
        )
        self.assertEqual(dictionary["charged_sampling_bytes"], 19)
        self.assertEqual(dictionary["dictionary_bytes"], 5)


if __name__ == "__main__":
    unittest.main()
