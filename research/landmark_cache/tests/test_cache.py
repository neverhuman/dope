import copy
import hashlib
import io
import json
import tempfile
import threading
import time
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from research.landmark_cache.dashboard import summarize
from research.landmark_cache.plan import OPTIONS, canonical, digest, make_plan, validate_plan
from research.landmark_cache.runner import canonical_train, run, run_cell, validate_report, wait_native, write_private
from research.landmark_cache.storage import LocalStore, LocalTrainSource, S3Store


class ClientError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class MemoryS3:
    def __init__(self):
        self.objects, self.calls = {}, []

    def get_object(self, **args):
        self.calls.append(("get", args))
        key = (args["Bucket"], args["Key"])
        if key not in self.objects:
            raise ClientError("NoSuchKey")
        raw = self.objects[key]
        return {"Body": io.BytesIO(raw), "ContentLength": len(raw)}

    def put_object(self, **args):
        self.calls.append(("put", args))
        assert args["IfNoneMatch"] == "*"
        assert args["ServerSideEncryption"] == "AES256"
        assert "ACL" not in args
        key = (args["Bucket"], args["Key"])
        if key in self.objects:
            raise ClientError("PreconditionFailed")
        self.objects[key] = args["Body"]


def entry(task="regression"):
    return dict(dataset_hash="a" * 16, canonical_sha256="b" * 64, task=task,
                files={"train": dict(sha256="c" * 64, bytes=20, rows=20, columns=3, has_header=False),
                       "test": dict(sha256="d" * 64)},
                target=dict(index=0, name="target", type="numeric"), provenance={})


def report(binding):
    return dict(panel=OPTIONS["panel"], options=binding["options"], task=binding["task"],
                kind="real", privacy="private", rows=binding["rows"],
                cells=[dict(id=i, state="complete", gain=0.2, stable=True) for i in range(1, 24)],
                contrasts=[None] * 18)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        raw = canonical(entry()) + b"\n"
        self.catalog = self.root / "catalog.jsonl"
        self.catalog.write_bytes(raw)
        with patch("research.landmark_cache.plan.CATALOG_SHA256", hashlib.sha256(raw).hexdigest()):
            self.plan = make_plan(self.catalog, bucket="source-bucket", engine_sha256="e" * 64,
                                  source_commit="f" * 40)
        self.cell = self.plan["cells"][0]
        self.binding = self.cell["binding"]
        self.client = MemoryS3()
        self.store = S3Store(self.client, "private-results", "derived/landmarks/v1")

    def tearDown(self):
        self.temp.cleanup()

    def test_catalog_digest_is_required(self):
        with self.assertRaisesRegex(ValueError, "catalog_digest"):
            make_plan(self.catalog, bucket="source-bucket", engine_sha256="e" * 64, source_commit="f" * 40)

    def test_full_plan_is_bound_and_train_only(self):
        validate_plan(self.plan)
        self.assertNotIn("test", self.binding["source_key"])
        self.assertEqual(self.binding["source_role"], "official_train")
        self.assertFalse(self.plan["official_test_access"])
        self.assertFalse(self.plan["model_training"])
        self.assertFalse(self.binding["source_family_resolved"])
        self.assertEqual(self.binding["source_license"], {"status": "unknown"})

    def test_tampered_cell_and_plan_refuse(self):
        broken = copy.deepcopy(self.plan)
        broken["cells"][0]["binding"]["options"]["folds"] = 3
        with self.assertRaisesRegex(ValueError, "plan_digest"):
            validate_plan(broken)

    def test_cache_key_changes_with_recipe_or_binary(self):
        for field, value in (("engine_sha256", "a" * 64), ("source_commit", "a" * 40)):
            changed = dict(self.binding, **{field: value})
            self.assertNotEqual(self.cell["key"], digest(changed))
        changed = copy.deepcopy(self.binding)
        changed["options"]["seed"] += 1
        self.assertNotEqual(self.cell["key"], digest(changed))

    def test_target_first_is_moved_without_value_or_order_changes(self):
        source = self.root / "source.csv"
        source.write_text("label0,1,2\nlabel1,3,4\n")
        binding = dict(self.binding, rows=2)
        destination = self.root / "train.csv"
        receipt = canonical_train(source, destination, binding)
        self.assertEqual(destination.read_text(), "feature_0000,feature_0001,target\n1,2,label0\n3,4,label1\n")
        self.assertEqual(receipt["source_column_permutation"], [1, 2, 0])
        self.assertTrue(receipt["physical_row_order_preserved"])
        self.assertEqual(source.read_text(), "label0,1,2\nlabel1,3,4\n")
        self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

    def test_header_disagreement_refuses(self):
        source = self.root / "source.csv"
        source.write_text("wrong,f1,f2\n1,2,3\n")
        with self.assertRaisesRegex(ValueError, "header_declaration"):
            canonical_train(source, self.root / "out.csv", dict(self.binding, rows=1, has_header=True))

    def test_ragged_rows_refuse_without_echoing_values(self):
        source = self.root / "source.csv"
        source.write_text("SECRET_TARGET,SECRET_HEADER\n")
        with self.assertRaises(ValueError) as error:
            canonical_train(source, self.root / "out.csv", self.binding)
        self.assertNotIn("SECRET", str(error.exception))

    def test_report_identity_nonfinite_and_slot_mutations_refuse(self):
        original = report(self.binding)
        validate_report(original, self.binding)
        for field, value in (("kind", "synthetic"), ("privacy", "public"), ("rows", 999)):
            with self.assertRaises(ValueError):
                validate_report(dict(original, **{field: value}), self.binding)
        broken = copy.deepcopy(original)
        broken["cells"][0]["id"] = 2
        with self.assertRaisesRegex(ValueError, "slot_mismatch"):
            validate_report(broken, self.binding)
        broken = copy.deepcopy(original)
        broken["cells"][0]["gain"] = float("nan")
        with self.assertRaises(ValueError):
            validate_report(broken, self.binding)

    def test_publication_manifest_last_and_verified_reuse(self):
        value = report(self.binding)
        self.assertIsNone(self.store.lookup(self.cell))
        self.store.publish(self.cell, value)
        puts = [args for method, args in self.client.calls if method == "put"]
        self.assertIn("objects/sha256/", puts[0]["Key"])
        self.assertTrue(puts[1]["Key"].endswith("manifest.json"))
        self.assertEqual(json.loads(self.store.lookup(self.cell)), value)
        self.store.publish(self.cell, value)

    def test_immutable_collision_and_corruption_refuse(self):
        self.store.publish(self.cell, report(self.binding))
        with self.assertRaisesRegex(ValueError, "immutable_cache_collision"):
            self.store.put_once("cells/" + self.cell["key"] + "/manifest.json", b"changed")
        report_key = next(k for k in self.client.objects if "/objects/" in k[1])
        self.client.objects[report_key] = b"corrupted"
        with self.assertRaisesRegex(ValueError, "cached_report_corrupt"):
            self.store.lookup(self.cell)

    def test_wrong_binding_refuses_cache_reuse(self):
        self.store.publish(self.cell, report(self.binding))
        wrong = copy.deepcopy(self.cell)
        wrong["binding"]["source_role"] = "official_test"
        with self.assertRaisesRegex(ValueError, "cache_binding_mismatch"):
            self.store.lookup(wrong)

    def test_local_store_has_same_integrity_and_collision_contract(self):
        store = LocalStore(self.root / "local")
        store.publish(self.cell, report(self.binding))
        self.assertEqual(json.loads(store.lookup(self.cell)), report(self.binding))
        with self.assertRaisesRegex(ValueError, "immutable_cache_collision"):
            store.put_once("cells/" + self.cell["key"] + "/manifest.json", b"changed")
        with self.assertRaisesRegex(ValueError, "local_cache_key"):
            store.get("../escape")
        paths = list((self.root / "local").rglob("*.json"))
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in paths))

    def test_offline_source_rejects_TEST_and_verifies_role(self):
        source = LocalTrainSource(self.root, self.plan)
        with self.assertRaisesRegex(ValueError, "not_in_TRAIN_schedule"):
            source.get_object(Bucket="source-bucket", Key="blobs/sha256/" + "d" * 64)
        path = self.root / (self.binding["train_sha256"] + ".csv")
        path.write_bytes(b"train-only")
        response = source.get_object(Bucket="source-bucket", Key=self.binding["source_key"])
        with response["Body"] as stream:
            self.assertEqual(stream.read(), b"train-only")

    def test_bounded_scheduler_retains_every_terminal_state(self):
        binary = self.root / "fixture-binary"
        binary.write_bytes(b"not executed: scheduler fixture")
        plan = copy.deepcopy(self.plan)
        cells = []
        for i in range(8):
            binding = dict(self.binding, dataset_id=f"{i:016x}", engine_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            cells.append(dict(self.cell, key=digest(binding), binding=binding))
        plan["cells"] = cells
        plan.pop("plan_sha256")
        plan["plan_sha256"] = digest(plan)
        lock, state = threading.Lock(), {"active": 0, "peak": 0, "calls": []}

        def fixture_worker(cell, **_kwargs):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["calls"].append(cell["key"])
            time.sleep(0.005)
            with lock:
                state["active"] -= 1
            return dict(key=cell["key"], state="failed" if cell is cells[2] else "complete")

        with patch("research.landmark_cache.runner.run_cell", fixture_worker):
            result = run(plan, binary=binary, root=self.root / "run", client=self.client,
                         store=self.store, workers=2, threads=1, timeout=5, wall_seconds=600)
        self.assertEqual(len(state["calls"]), 8)
        self.assertEqual(len(set(state["calls"])), 8)
        self.assertLessEqual(state["peak"], 2)
        self.assertEqual(result["pending"], 0)
        self.assertEqual(result["counts"]["failed"], 1)

    def test_unsupported_job_never_fetches_source(self):
        cell = dict(self.cell, initial_state="unsupported", reason="unsupported_task")
        receipt = run_cell(cell, binary=Path("unused"), root=self.root, client=self.client,
                           store=self.store, threads=1, timeout=5, memory_bytes=2**30)
        self.assertEqual(receipt["state"], "unsupported")
        self.assertEqual(self.client.calls, [])

    def test_interrupted_attempt_requires_explicit_repair(self):
        path = self.root / "cells" / self.cell["key"] / "attempt-0000"
        path.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "requires_repair"):
            run_cell(self.cell, binary=Path("unused"), root=self.root, client=self.client,
                     store=self.store, threads=1, timeout=5, memory_bytes=2**30)

    def test_terminal_failure_does_not_automatically_retry(self):
        path = self.root / "cells" / self.cell["key"] / "attempt-0000" / "receipt.json"
        write_private(path, {"state": "failed"})
        with self.assertRaisesRegex(ValueError, "requires_repair"):
            run_cell(self.cell, binary=Path("unused"), root=self.root, client=self.client,
                     store=self.store, threads=1, timeout=5, memory_bytes=2**30)
        self.assertEqual(len(list(path.parent.parent.glob("attempt-*"))), 1)

    def test_dashboard_allowlist_excludes_private_measurements(self):
        row = dict(key="a", dataset_id="opaque", state="complete", host="fixture", elapsed_seconds=1,
                   target="SECRET_HEADER", gains=[999], raw_values=["SECRET_VALUE"])
        run = dict(host="fixture", shard=0, shards=1, scheduled=1, terminal=1, pending=0,
                   total_catalog=1, counts={"complete": 1}, elapsed_seconds=1, workers=1,
                   threads_per_worker=1, updated_unix=0, plan_sha256="a", cells=[row])
        path = self.root / "progress.json"
        write_private(path, run)
        dashboard = summarize([path])
        self.assertNotIn("SECRET", json.dumps(dashboard))
        self.assertNotIn("gains", json.dumps(dashboard))
        self.assertEqual(dashboard["counts"]["complete"], 1)

    def test_wall_timeout_kills_the_owned_process_group(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
        code, _, reason = wait_native(process, 0.05, 2**30)
        self.assertIsNone(code)
        self.assertEqual(reason, "native_wall_cap")
        self.assertIsNotNone(process.poll())

    def test_sampled_rss_cap_kills_an_overbudget_process(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
        code, peak, reason = wait_native(process, 3, 1024, watch_rss=True)
        self.assertIsNone(code)
        self.assertEqual(reason, "native_rss_cap")
        self.assertGreater(peak, 1024)
        self.assertIsNotNone(process.poll())


if __name__ == "__main__":
    unittest.main()
