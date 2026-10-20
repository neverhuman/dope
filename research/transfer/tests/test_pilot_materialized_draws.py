"""Binary handoff fixtures and real native DOPE n/4n integration; no science grant."""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from research.transfer.campaign_artifacts import create_private, file_hash
from research.transfer.campaign_runner import CampaignRunnerError, StageJournal
from research.transfer.evidence import canonical_json, digest
from research.transfer.pilot_materialized_draws import (
    DrawSpec, MaterializedDraw, decode_draw, load_pair, seed_hex, store_pair,
)
from research.transfer import pilot_runner
from research.transfer.tests.test_pilot_runner import fixture, simulated_8stage_workers


def draw_fixture(scale="primary_n", rows=2, seed=(1 << 64) - 2, method="dope", artifact=b"fixture"):
    raw = struct.pack("<" + "d" * rows * 2, *[float(i) / 2 for i in range(rows * 2)])
    spec = DrawSpec(scale, method, rows, 2, seed_hex(seed), hashlib.sha256(artifact).hexdigest(),
                    "a" * 64, "b" * 64, 32768)
    report = {"state": "sampled", "rows": rows, "seed": seed, "sample_sha256": hashlib.sha256(raw).hexdigest(),
              "filesystem_network_exec": "denied", "inherited_source_fds": "closed",
              "isolation": "linux_x86_64_seccomp_tsync_default_deny",
              "private_sample_stream": {"format": "f64le", "layout": "row_major",
                                        "column_order": "features_then_target", "rows": rows,
                                        "columns": 2, "bytes": len(raw)},
              "inspection": {"schema": [{}, {}], "accounting": {"complete_sampling_bytes": len(artifact)}}}
    return spec, raw, report


def native_json(value):
    # Native uint64 receipt format is deliberately separate from finite-json-v1.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class MaterializedDrawTests(unittest.TestCase):
    def test_exact_u64_pool_conversion_preserves_legacy_identity(self):
        spec, raw, report = draw_fixture()
        draw = decode_draw(raw, native_json(report), spec)
        pool = draw.pool()
        self.assertEqual(pool.native_draw_seed, (1 << 64) - 2)
        self.assertEqual(pool.draw_seed, "0xfffffffffffffffe")
        self.assertEqual(pool.batch.features.rows, ((0.0,), (1.0,)))
        self.assertEqual(pool.batch.targets, (0.5, 1.5))
        self.assertNotIn("0.5", repr(draw))
        legacy = replace(pool, draw_seed=42)
        self.assertEqual(legacy.sha256, digest(json.loads(json.dumps(asdict(legacy)))))
        self.assertEqual(legacy.native_draw_seed, 42)
        for bad in (True, -1, 1 << 64):
            with self.assertRaises(CampaignRunnerError):
                seed_hex(bad)
        for bad in (True, "0X0000000000000001", "0x1", "0x000000000000000A", 1 << 53):
            with self.assertRaises(ValueError):
                replace(pool, draw_seed=bad)
        with self.assertRaisesRegex(CampaignRunnerError, "draw_row_identity_collision"):
            draw.pool(forbidden_row_ids=draw.row_ids)

    def test_constructor_itself_cannot_bypass_receipt_validation(self):
        spec, raw, report = draw_fixture()
        report["seed"] -= 1
        with self.assertRaisesRegex(CampaignRunnerError, "draw_receipt_mismatch"):
            MaterializedDraw(spec, raw, native_json(report))

    def test_malformed_receipts_shapes_and_nonfinite_streams_refuse(self):
        spec, raw, report = draw_fixture()
        for malformed in (b'{"state":1,"state":2}', b'{"SECRET":NaN}', b'{"SECRET":1e999}', b'[]', b'\xff'):
            with self.subTest(malformed=malformed), self.assertRaises(CampaignRunnerError) as error:
                decode_draw(raw, malformed, spec)
            self.assertNotIn("SECRET", str(error.exception))
        for bad in (raw[:-1], raw + b"\0", struct.pack("<dddd", math.nan, 1, 2, 3)):
            with self.assertRaises(CampaignRunnerError):
                decode_draw(bad, native_json(report), spec)
        mutations = [("rows", True), ("seed", True), ("sample_sha256", "c" * 64),
                     ("isolation", "claimed"), ("inspection", {"schema": [{}]}),
                     ("private_sample_stream", {**report["private_sample_stream"], "bytes": 1})]
        for key, value in mutations:
            with self.subTest(key=key), self.assertRaises(CampaignRunnerError):
                decode_draw(raw, native_json({**report, key: value}), spec)

    def test_two_distinct_draws_private_roundtrip_and_tamper_refusal(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifact = root / "artifact"
            create_private(artifact, b"fixture")
            left = decode_draw(*(lambda s, r, p: (r, native_json(p), s))(*draw_fixture()))
            s, r, p = draw_fixture("sensitivity_4n", 8, (1 << 64) - 1)
            right = decode_draw(r, native_json(p), s)
            manifest = store_pair(root, left, right, artifact=artifact)
            pin = file_hash(root / "materialized-draws/manifest.json")
            loaded = load_pair(root, pin, artifact=artifact)
            self.assertEqual(loaded, (left, right))
            self.assertEqual([x["spec"]["rows"] for x in manifest["draws"]], [2, 8])
            self.assertTrue(set(left.row_ids).isdisjoint(right.row_ids))
            self.assertFalse(manifest["execution_or_scientific_admission"])
            for path in (root / "materialized-draws").iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o400)
            with self.assertRaisesRegex(CampaignRunnerError, "draw_replay_refused"):
                store_pair(root, left, right, artifact=artifact)
            data = root / "materialized-draws/primary_n.f64le"
            data.chmod(0o600)
            data.write_bytes(left.raw[::-1])
            with self.assertRaisesRegex(CampaignRunnerError, "draw_payload_changed"):
                load_pair(root, pin, artifact=artifact)
            data.unlink()  # disposable test fixture only
            data.symlink_to(artifact)
            with self.assertRaisesRegex(CampaignRunnerError, "native_file_unavailable"):
                load_pair(root, pin, artifact=artifact)

    def test_pair_refuses_same_seed_wrong_size_artifact_change_before_persist(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifact = root / "artifact"
            create_private(artifact, b"fixture")
            s, r, p = draw_fixture()
            left = decode_draw(r, native_json(p), s)
            for rows, seed in ((8, (1 << 64) - 2), (7, (1 << 64) - 1)):
                s, r, p = draw_fixture("sensitivity_4n", rows, seed)
                right = decode_draw(r, native_json(p), s)
                with self.assertRaisesRegex(CampaignRunnerError, "draw_pair_mismatch"):
                    store_pair(root, left, right, artifact=artifact)
            artifact.write_bytes(b"changed")
            s, r, p = draw_fixture("sensitivity_4n", 8, (1 << 64) - 1)
            with self.assertRaisesRegex(CampaignRunnerError, "draw_artifact_changed"):
                store_pair(root, left, decode_draw(r, native_json(p), s), artifact=artifact)
            self.assertFalse((root / "materialized-draws").exists())

    def test_undercharged_native_artifact_is_rejected_before_persist(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifact = root / "artifact"
            create_private(artifact, b"fixture")
            draws = []
            for scale, rows, seed in (("primary_n", 2, 5), ("sensitivity_4n", 8, 6)):
                s, r, p = draw_fixture(scale, rows, seed)
                p["inspection"]["accounting"]["complete_sampling_bytes"] = 1
                draws.append(decode_draw(r, native_json(p), s))
            with self.assertRaisesRegex(CampaignRunnerError, "draw_artifact_charge_mismatch"):
                store_pair(root, *draws, artifact=artifact)
            self.assertFalse((root / "materialized-draws").exists())

    def test_real_control_abi_and_complete_byte_charge_are_mandatory(self):
        from research.transfer.tdl.control_boundary import canonical
        for method in ("gaussian_copula", "compressed_real_rows"):
            spec, raw, _ = draw_fixture(method=method)
            request = {"method": method, "rows": spec.rows, "seed": spec.seed,
                       "artifact_sha256": spec.artifact_sha256, "library_sha256": "c" * 64,
                       "abi": 1, "source_sha256": "d" * 64, "runtime_closure_sha256": "e" * 64,
                       "artifact_bytes": 7, "runtime_bytes": 1000}
            report = {**request, "columns": 2, "sample_bytes": len(raw), "scientific_admission": False,
                      "citable": False, "sample_sha256": hashlib.sha256(raw).hexdigest(), "state": "sampled",
                      "format": "f64le", "layout": "row_major", "column_order": "features_then_target",
                      "isolation": "existing_rust_landlock_seccomp_tsync", "pipeline_complete": False,
                      "invocation_bytes": len(canonical(request)),
                      "sampling_closure_bytes": 1007 + len(canonical(request))}
            result = decode_draw(raw, native_json(report), spec, control_request=request)
            request["seed"] = 0
            self.assertEqual(json.loads(result.control_request)["seed"], spec.seed)
            with self.assertRaises(CampaignRunnerError):
                decode_draw(raw, native_json(report), spec, control_request=request)
            with self.assertRaises(CampaignRunnerError):
                decode_draw(raw, native_json(report), replace(spec, byte_budget=10),
                            control_request=json.loads(result.control_request))


class PilotV2IntegrationTests(unittest.TestCase):
    def test_two_calls_share_one_reservation_and_manifest_is_handed_to_worker(self):
        self._sample_fixture_test(fail_second=False)

    def test_second_draw_failure_preserves_first_and_never_retries_or_trains(self):
        self._sample_fixture_test(fail_second=True)

    def _sample_fixture_test(self, *, fail_second):
        with tempfile.TemporaryDirectory() as name:
            spec, plan, contract, cell, root, auth = fixture(name, version=2)
            observed, draw_walls = [], []
            rest = simulated_8stage_workers(cell, observed)

            def invoke(argv, directory, label, wall, ram):
                if not label.startswith("sample_"):
                    return rest(argv, directory, label, wall, ram)
                observed.append((label, argv))
                draw_walls.append(wall)
                self.assertIn("--private-samples-f64le", argv)
                self.assertTrue((root / "development-stages.jsonl").exists())
                rows = int(argv[argv.index("--rows") + 1])
                seed = int(argv[argv.index("--seed") + 1])
                _, raw, receipt = draw_fixture(label[7:], rows, seed,
                                               artifact=(root / "model.raw").read_bytes())
                time.sleep(0.02)
                create_private(root / (label + ".stdout"), raw)
                create_private(root / (label + ".stderr"), native_json(receipt))
                return {"status": "timeout" if fail_second and len(draw_walls) == 2 else "completed",
                        "wall_ms": 20, "peak_ram_bytes": 1024, "returncode": 1 if fail_second and len(draw_walls) == 2 else 0}

            with mock.patch.object(pilot_runner, "invoke", side_effect=invoke):
                report = pilot_runner.execute_pilot_cell(spec, plan, contract, cell, root, **auth)
            self.assertEqual(len(draw_walls), 2)
            self.assertLess(draw_walls[1], draw_walls[0])
            self.assertEqual(report["status"], "failed" if fail_second else "ok")
            self.assertFalse(report["pipeline_complete"])
            with StageJournal(root, spec) as journal:
                begins = [r for r in journal.records if r["type"] == "begin" and r["data"]["stage"] == "sample"]
                self.assertEqual(len(begins), 1)
                self.assertEqual(begins[0]["data"]["reserved_wall_ms"], 10000)
            if fail_second:
                self.assertFalse((root / "materialized-draws").exists())
                self.assertTrue((root / "sample_primary_n.stdout").exists())
                self.assertNotIn("learners", [label for label, _ in observed])
                with mock.patch.object(pilot_runner, "invoke") as forbidden:
                    with self.assertRaises(CampaignRunnerError):
                        pilot_runner.execute_pilot_cell(spec, plan, contract, cell, root, **auth)
                    forbidden.assert_not_called()
            else:
                argv = next(argv for label, argv in observed if label == "learners")
                self.assertNotIn("--sample-stdout", argv)
                self.assertEqual(argv[argv.index("--materialized-draw-manifest-sha256") + 1],
                                 report["materialized_draw_manifest_sha256"])

    def test_real_native_binary_draws_then_explicit_missing_learner_worker(self):
        checkout = Path(__file__).resolve().parents[3]
        binary = checkout / "target/debug/dope-kernel"
        if not binary.is_file():
            subprocess.run(["cargo", "build", "--locked", "--bin", "dope-kernel"], cwd=checkout, check=True)
        with tempfile.TemporaryDirectory() as name:
            spec, plan, contract, cell, root, auth = fixture(name, version=2, binary_source=binary, stage_wall_ms=30000)
            report = pilot_runner.execute_pilot_cell(spec, plan, contract, cell, root, **auth)
            self.assertTrue(all(r["status"] == "completed" for r in report["stages"][:4]))
            self.assertEqual(report["stages"][4]["status"], "failed")
            left, right = load_pair(root, report["materialized_draw_manifest_sha256"], artifact=root / "model.raw")
            self.assertEqual((left.spec.rows, right.spec.rows), (16, 64))
            self.assertEqual((left.spec.seed, right.spec.seed), ((1 << 64) - 2, (1 << 64) - 1))
            self.assertEqual(len(left.pool().batch.targets), 16)
            self.assertEqual(len(right.pool().batch.targets), 64)
            self.assertNotIn("materialized_primary_and_4n_draw_handoff", report["pipeline_prerequisites"])
            self.assertFalse(report["pipeline_complete"])
            self.assertFalse(report["scientific_admission"])
            with StageJournal(root, spec) as journal:
                records = [r for r in journal.records if r.get("data", {}).get("stage") == "sample" and r.get("type") == "begin"]
                self.assertEqual(len(records), 1)
                self.assertEqual(journal.state[(cell.document()["cell_id"], "sample")]["status"], "completed")

    def test_version2_seed_and_control_scope_refusals(self):
        with tempfile.TemporaryDirectory() as name:
            _, _, _, cell, _, _ = fixture(name, version=2)
            doc = cell.document()
            for change in ({"draw_4n_seed": doc["draw_seed"]}, {"draw_seed": 2**63},
                           {"draw_seed": "0x1"}, {"comparator": "gaussian_copula"}):
                with self.assertRaises(CampaignRunnerError):
                    pilot_runner.PilotCellExecutionPlan.from_document({**doc, **change})


if __name__ == "__main__":
    unittest.main()
