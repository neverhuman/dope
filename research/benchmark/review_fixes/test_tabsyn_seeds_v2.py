"""Block B (TabSyn scaled arm, fit seeds 23-71) runner, worker, ledger, and scoring custody.

Fixtures only: no benchmark data, GPU, or runtime python. The captured adapter
is the repository copy, whose digest the amendment pins.
"""

import ast
import collections
import datetime
import errno
import hashlib
import json
import os
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes import bind_v2_cells as B
from research.benchmark.review_fixes import rescore_adapters as A
from research.benchmark.review_fixes import rescore_ledger as L
from research.benchmark.review_fixes import tabsyn_seeds_v2 as T
from research.benchmark.review_fixes import tabsyn_seeds_worker as W
from research.benchmark.review_fixes import v2_panel as P

HERE = Path(__file__).resolve().parent
ADAPTER = HERE.parent / "tabsyn_adapter.py"
SEEDS = [11, 23, 37, 53, 71]
CONFIGURATION = "scaled_200_vae_1000_diffusion"
DATASET = "0123456789abcdef"
GRID = {"sizes": [1, 4], "sample_seeds": [101, 211, 307]}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def seed_compares(tree: ast.Module, function: str) -> list[str]:
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    return [ast.unparse(node) for node in ast.walk(body)
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "seed"]


def plan_document(count: int = 100, config_sha: str | None = None) -> dict:
    config_sha = config_sha or T.load_amendment()["config_sha256"]
    lineages = []
    for index in range(count):
        dataset = f"{index:016x}"
        lineages.append({"dataset": dataset, "train_rows": 10,
                         "learning_identity": {"dataset": dataset, "fit_seed": 11, "joint_train_sha256": "t" * 64,
                                               "joint_validation_sha256": "v" * 64, "projection_sha256": "p" * 64},
                         "operation_template": {"worker": {"files": {"train.csv": "a" * 64}}}})
    return {"configs": [{"config_sha256": config_sha}], "lineages": lineages}


class AmendmentTests(unittest.TestCase):
    def test_amendment_matches_predeclare_and_the_seed_eleven_arm(self):
        amendment = T.load_amendment()
        self.assertEqual(amendment["fit_seeds"], [23, 37, 53, 71])
        self.assertEqual(amendment["configuration"], CONFIGURATION)
        canonical = {key: value for key, value in amendment["config"].items() if key != "config_sha256"}
        digest = sha(json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode())
        self.assertEqual(digest, amendment["config_sha256"])
        self.assertEqual(amendment["config"]["config_sha256"], amendment["config_sha256"])
        self.assertEqual(sha(ADAPTER.read_bytes()), amendment["adapter"]["sha256"])
        self.assertEqual(amendment["environment"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertIs(amendment["official_tests_opened"], False)

    def test_off_contract_amendments_refuse(self):
        original = json.loads(T.AMENDMENT.read_text())
        cases = {
            "another predeclare_v2": {"amends": {**original["amends"], "sha256": "0" * 64}},
            "arm differs": {"config_sha256": "e" * 64},
            "seeds differ": {"fit_seeds": [23, 37, 53]},
            "sample grid differs": {"sample_seeds": [101, 211]},
            "caps differ": {"caps": {**original["caps"], "fit_operation_seconds": 1800}},
            "close differs": {"close_utc": "2026-10-30T00:00:00Z"},
            "not the block B": {"block": "C"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            for message, change in cases.items():
                path = Path(tmp) / "amendment.json"
                path.write_text(json.dumps({**original, **change}))
                with self.subTest(message), self.assertRaisesRegex(SystemExit, message):
                    T.load_amendment(path)

    def test_plan_and_run_refuse_other_hosts_before_reading_any_pin(self):
        for action in ("plan", "run"):
            with self.subTest(action), patch.object(T.os, "uname") as uname, \
                    patch.object(T.W, "checked", side_effect=AssertionError("read a pin")), \
                    patch("sys.argv", ["tabsyn_seeds_v2", action]):
                uname.return_value.nodename = "xbabe2"
                with self.assertRaisesRegex(SystemExit, "refusing to run"):
                    T.main()


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.raw = ADAPTER.read_bytes()

    def test_only_the_fit_seed_admission_changes_in_the_captured_adapter(self):
        widened = W.transform(self.raw, SEEDS)
        stripped = W._StripAdmission().visit(ast.parse(self.raw))
        self.assertEqual(seed_compares(widened, "fit"), ["seed in [11, 23, 37, 53, 71]"])
        for before, after in zip(stripped.body, widened.body):
            if isinstance(after, ast.FunctionDef) and after.name == "fit":
                for node in ast.walk(after):
                    if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.In) \
                            and isinstance(node.left, ast.Name) and node.left.id == "seed":
                        node.ops, node.comparators = [ast.Eq()], [ast.Constant(value=11)]
            self.assertEqual(ast.dump(before), ast.dump(after))

    def test_sample_and_native_seed_rules_stay_as_captured(self):
        widened = W.transform(self.raw, SEEDS)
        self.assertEqual(seed_compares(widened, "sample"), ["seed in [101, 211, 307]"])
        self.assertEqual(seed_compares(widened, "native_objective"), ["seed == 11"])

    def test_off_contract_sources_and_seed_lists_refuse(self):
        text = self.raw.decode()
        guard = "from research.benchmark.tabsyn_runtime_guard import verify as verify_runtime\n"
        self.assertIn(guard, text)
        cases = {
            "not exactly one": text.replace("seed == 11 and type(timeout_seconds)", "seed in [11, 23] and type(timeout_seconds)"),
            "strip count is 3": text.replace(guard, ""),
            "import is outside": text.replace(guard, guard + "import subprocess\n"),
            "call is outside": text.replace("def seed_all(seed):\n", "def seed_all(seed):\n    exec('pass')\n"),
        }
        for message, source in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                W.transform(source.encode(), SEEDS)
        for seeds in ([23, 37], [71, 11]):
            with self.subTest(seeds), self.assertRaisesRegex(ValueError, "sorted and keep"):
                W.transform(self.raw, seeds)

    def test_worker_imports_only_the_standard_library(self):
        tree = ast.parse((HERE / "tabsyn_seeds_worker.py").read_text())
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        # numpy and torch load only after the runtime path switch, through the inventory guard.
        self.assertEqual(imported - {"numpy", "torch", "subprocess"}, {
            "ast", "csv", "errno", "hashlib", "importlib", "json", "os", "struct", "sys", "time", "traceback",
            "types", "pathlib"})
        self.assertNotIn("exec(", (HERE / "tabsyn_seeds_worker.py").read_text())

    def test_the_loaded_adapter_admits_the_declared_seeds_and_no_other(self):
        class Guards:
            def verify_runtime(self, *_args):
                raise AssertionError("no runtime check is reached before the admission guards")

            verify_operation = verify_runtime

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "adapter.allowlisted.py"
            adapter = W.load_adapter(W.transform(self.raw, SEEDS), staging, Guards())
            with self.assertRaises(FileExistsError):
                W.load_adapter(W.transform(self.raw, SEEDS), staging, Guards())
        common = {"runtime_path": "r", "runtime_sha256": "r", "code_path": "c", "code_sha256": "c"}
        for seed, message in ((23, "needs independent request and GPU admission"), (71, "GPU admission"),
                              (99, "fit seed or deadline differs"), (11, "GPU admission")):
            with self.subTest(seed), self.assertRaisesRegex(ValueError, message):
                adapter["fit"](None, None, b"", "unused", {}, seed=seed, device="cpu", timeout_seconds=600,
                               generated_fixture=False, **common)
        with self.assertRaisesRegex(ValueError, "sample identity differs"):
            adapter["sample"]("unused", 10, 23, {}, **common)


class InputTests(unittest.TestCase):
    def test_joint_rows_put_the_target_first_and_hash_like_numpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            path.write_text("f0,f1,target\n1,2,3\n4.5,-6,7e-3\n")
            rows = W.joint_rows(path)
        self.assertEqual(rows, [[3.0, 1.0, 2.0], [0.007, 4.5, -6.0]])
        self.assertEqual(W.joint_sha256(rows), sha(struct.pack("<6d", 3.0, 1.0, 2.0, 0.007, 4.5, -6.0)))
        try:
            import numpy as np
        except ImportError:
            return
        self.assertEqual(W.joint_sha256(rows), sha(np.asarray(rows, dtype=np.float64).tobytes(order="C")))

    def test_sample_csv_moves_the_target_last_under_the_seed_eleven_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample-101-1n.csv"
            W.write_sample(path, [[3.0, 1.0, 2.0], [0.5, -1.0, 4.0]])
            self.assertEqual(path.read_bytes(), b"f0,f1,target\r\n1.0,2.0,3.0\r\n-1.0,4.0,0.5\r\n")
            self.assertEqual(sorted(item.name for item in Path(tmp).iterdir()), ["sample-101-1n.csv"])

    def test_pinned_reads_refuse_symlinks_test_partitions_and_digest_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "train.csv"
            real.write_bytes(b"x")
            self.assertEqual(W.checked(real, sha(b"x")), b"x")
            with self.assertRaisesRegex(ValueError, "pinned_digest_changed"):
                W.checked(real, sha(b"y"))
            (Path(tmp) / "test.csv").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "refused_path"):
                W.checked(Path(tmp) / "test.csv", sha(b"x"))
            os.symlink(real, Path(tmp) / "link.csv")
            with self.assertRaisesRegex(ValueError, "refused_path"):
                W.checked(Path(tmp) / "link.csv", sha(b"x"))

    def test_preflight_records_an_input_mismatch_without_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            workers, scoring = Path(tmp) / "views", Path(tmp) / "scoring"
            (workers / DATASET).mkdir(parents=True)
            (scoring / DATASET).mkdir(parents=True)
            for name, body in (("train.csv", "1,2\n3,4\n"), ("validation.csv", "5,6\n")):
                (workers / DATASET / name).write_text("f0,target\n" + body)
                (scoring / DATASET / name).write_text(body)
            files = {name: sha((workers / DATASET / name).read_bytes()) for name in ("train.csv", "validation.csv")}
            row = {"dataset": DATASET, "train_rows": 2, "files": files,
                   "joint_train_sha256": W.joint_sha256([[2.0, 1.0], [4.0, 3.0]]),
                   "joint_validation_sha256": W.joint_sha256([[6.0, 5.0]])}
            self.assertEqual(T.preflight(row, workers, scoring)["status"], "ok")
            self.assertEqual(T.preflight({**row, "joint_train_sha256": "0" * 64}, workers, scoring)["reason"],
                             "joint_input_differs")
            self.assertEqual(T.preflight({**row, "files": {**files, "projection.json": "0" * 64}},
                                         workers, scoring)["status"], "unavailable")
            (scoring / DATASET / "validation.csv").write_text("5,7\n")
            self.assertEqual(T.preflight(row, workers, scoring)["reason"], "scoring_input_differs")

    def test_scoring_workers_are_the_k0_scorer_workers(self):
        self.assertEqual(Path(T.load_amendment()["scoring"]["workers"]), L.WORKERS)


class PlanTests(unittest.TestCase):
    def test_plan_is_seed_major_over_the_hundred_seed_eleven_lineages(self):
        rows = T.lineages(plan_document(), T.load_amendment()["config_sha256"])
        plan = T.seed_major(rows, [23, 37, 53, 71])
        self.assertEqual(len(plan), 400)
        self.assertEqual([seed for seed, _row in plan[:100]], [23] * 100)
        self.assertEqual([seed for seed, _row in plan[-100:]], [71] * 100)

    def test_plan_refuses_other_counts_duplicates_test_pins_and_configurations(self):
        config_sha = T.load_amendment()["config_sha256"]
        duplicate = plan_document()
        duplicate["lineages"][1] = duplicate["lineages"][0]
        test_pin = plan_document()
        test_pin["lineages"][0]["operation_template"]["worker"]["files"]["test.csv"] = "a" * 64
        cases = {"does not hold 100": plan_document(99), "identity differs": duplicate,
                 "pins a test partition": test_pin, "configuration differs": plan_document(config_sha="e" * 64)}
        for message, document in cases.items():
            with self.subTest(message), self.assertRaisesRegex(SystemExit, message):
                T.lineages(document, config_sha)

    def test_child_environment_is_the_seed_eleven_queue_environment(self):
        declared = dict(T.load_amendment()["environment"])
        declared.pop("python_flags")
        declared["LD_LIBRARY_PATH"] = "/runtime/lib"
        self.assertEqual(T.child_env({"LD_LIBRARY_PATH": "/runtime/lib"}), declared)


class ResumeTests(unittest.TestCase):
    def receipt(self, attempt: Path, name: str, **fields) -> None:
        attempt.mkdir(parents=True, exist_ok=True)
        (attempt / name).write_text(json.dumps(fields))

    def test_outcome_retries_only_a_confirmed_listed_cause(self):
        confirmed = {"foreign_gpu_owner"}

        def confirm(cause, _attempt):
            return cause in confirmed

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.receipt(root / "ok", "fit.complete.receipt.json", status="ok")
            self.receipt(root / "deadline", "fit.failure.receipt.json", status="operation_failed",
                         exception_type="TimeoutError", cause=None)
            self.receipt(root / "gpu", "fit.failure.receipt.json", status="infrastructure_failure",
                         exception_type="RuntimeError", cause="foreign_gpu_owner")
            self.receipt(root / "claimed", "fit.failure.receipt.json", status="infrastructure_failure",
                         exception_type="OSError", cause="storage_refusal")
            self.receipt(root / "invented", "fit.failure.receipt.json", status="infrastructure_failure",
                         exception_type="RuntimeError", cause="cosmic_ray")
            self.receipt(root / "nan", "fit.failure.receipt.json", status="operation_failed",
                         exception_type="ValueError", cause=None)
            self.receipt(root / "request", "request.failure.receipt.json", status="infrastructure_failure",
                         exception_type="RuntimeError", cause="foreign_gpu_owner")
            self.receipt(root / "post-cuda", "worker.CUDA-initialized.receipt.json", pid=1)
            cases = {("ok", 0): ("ok", None), ("ok", 124): ("timeout", "wall"),
                     ("deadline", 2): ("timeout", "deadline"),
                     ("gpu", 2): ("infrastructure_failure", "foreign_gpu_owner"),
                     ("claimed", 2): ("failed", "OSError"), ("invented", 2): ("failed", "RuntimeError"),
                     ("nan", 2): ("failed", "ValueError"),
                     ("request", 2): ("infrastructure_failure", "foreign_gpu_owner"),
                     ("post-cuda", -9): ("unknown", "no_final_receipt"),
                     ("post-cuda", 0): ("unknown", "no_final_receipt")}
            for (name, code), expected in cases.items():
                with self.subTest(name=name, code=code):
                    self.assertEqual(T.outcome(root / name, "fit", code, confirm), expected)
            confirmed.add("host_reboot")
            self.assertEqual(T.outcome(root / "post-cuda", "fit", None, confirm), ("infrastructure_failure", "host_reboot"))

    def test_charged_bytes_leave_out_intermediate_diffusion_checkpoints(self):
        inventory = {"files": {"diffusion.safetensors": {"bytes": 50}, "diffusion-epoch-0.safetensors": {"bytes": 50},
                               "decoder-final.safetensors": {"bytes": 7}, "projection.json": {"bytes": 3}}}
        self.assertEqual(T.charged_bytes(inventory), 60)

    def test_only_an_infrastructure_failure_retries_and_only_once(self):
        self.assertTrue(T.needs_attempt(None))
        self.assertTrue(T.needs_attempt({"status": "infrastructure_failure", "attempts": 1}))
        self.assertFalse(T.needs_attempt({"status": "infrastructure_failure", "attempts": 2}))
        for status in ("ok", "failed", "timeout", "done", "unknown"):
            self.assertFalse(T.needs_attempt({"status": status, "attempts": 1}))

    def test_a_killed_sample_operation_keeps_the_cells_it_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt, cell = Path(tmp) / "attempt", Path(tmp) / "fit-23"
            cells = T.sample_cells({"train_rows": 10}, cell, GRID["sizes"], GRID["sample_seeds"])
            self.assertEqual([(item["seed"], item["size"], item["rows"]) for item in cells][:4],
                             [(101, 1, 10), (211, 1, 10), (307, 1, 10), (101, 4, 40)])
            self.receipt(attempt, W.cell_receipt_name(cells[0]), **cells[0], status="ok", reason=None,
                         synthetic_sha256="a" * 64)
            statuses = T.sample_status(attempt, cells, 124)
            silent = T.sample_status(attempt, cells, -9)
            self.receipt(attempt, "sample.failure.receipt.json", status="operation_failed", cause=None)
            failed = T.sample_status(attempt, cells, 2)
        self.assertEqual([item["status"] for item in statuses], ["ok"] + ["timeout"] * 5)
        self.assertEqual([item["status"] for item in silent], ["ok"] + ["unknown"] * 5)
        self.assertEqual([item["status"] for item in failed], ["ok"] + ["failed"] * 5)
        self.assertTrue(all("path" not in item for item in statuses))


class FakeProcess:
    """A worker that exits 0 at once without writing a receipt."""

    pid = 0

    def __init__(self, argv, **_kwargs):
        self.argv = argv

    def wait(self, timeout=None):
        return 0


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.amendment = T.load_amendment()
        self.close = datetime.datetime.fromisoformat(self.amendment["close_utc"].replace("Z", "+00:00")).timestamp()
        self.now = [self.close - 100.0]
        self.controller = T.Controller(self.amendment, self.root / "out", "c" * 40, [],
                                       runtime={"python": "/bin/false", "LD_LIBRARY_PATH": ""},
                                       clock=lambda: self.now[0], boot=lambda: "boot-a")
        self.row = {"dataset": DATASET, "train_rows": 2, "files": {}, "projection_sha256": "p" * 64,
                    "joint_train_sha256": "t" * 64, "joint_validation_sha256": "v" * 64}
        self.cell = self.root / "out" / DATASET / "fit-23"
        self.lock = self.root / "gpu.lock"
        quiet = patch.object(T, "gpu_name", lambda: None)
        quiet.start()
        self.addCleanup(quiet.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_gpu_lock_released_after_close_starts_no_worker(self):
        def wait(_units):
            self.now[0] = self.close + 5

        with patch.object(T.v1, "_wait_to_fit", wait), \
                patch.object(T.subprocess, "Popen", side_effect=AssertionError("spawned after close")):
            self.controller.run([(23, self.row)], self.lock)
        self.assertEqual(list(self.cell.glob("attempt-*")), [])

    def test_a_fit_that_crosses_close_starts_no_sample(self):
        def fit(*_args):
            self.now[0] = self.close + 5
            return {"status": "ok", "attempts": 1}

        with patch.object(T.v1, "_wait_to_fit", lambda _units: None), patch.object(self.controller, "fit", fit), \
                patch.object(self.controller, "sample", side_effect=AssertionError("sampled after close")):
            self.controller.run([(23, self.row)], self.lock)

    def test_every_operation_deadline_is_cut_at_close_and_a_silent_exit_is_unknown(self):
        with patch.object(T.subprocess, "Popen", FakeProcess):
            record = self.controller.fit(23, self.row, self.cell, None)
        request = json.loads((self.cell / "attempt-1" / "fit" / "operation.request.json").read_text())
        self.assertEqual((request["cap_seconds"], request["operation_seconds"]), (600, 100))
        self.assertEqual((request["boot_id"], request["close_utc"]), ("boot-a", self.amendment["close_utc"]))
        self.assertEqual((record["status"], record["reason"]), ("unknown", "no_final_receipt"))
        self.assertFalse(T.needs_attempt(record))

    def test_a_request_published_after_close_spawns_no_worker(self):
        real_put = T.W.put

        def delayed_put(root, name, value):
            if name == "operation.request.json":
                self.now[0] = self.close + 1
            return real_put(root, name, value)

        with patch.object(T.W, "put", delayed_put), patch.object(T.v1, "_wait_to_fit", lambda _units: None), \
                patch.object(T.subprocess, "Popen", side_effect=AssertionError("spawned after close")), \
                patch.object(self.controller, "sample", side_effect=AssertionError("sampled after close")):
            self.controller.run([(23, self.row), (37, self.row)], self.lock)
        attempt = self.cell / "attempt-1" / "fit"
        self.assertTrue((attempt / T.REFUSED).is_file())
        record = json.loads((self.cell / T.FIT_RECORD).read_text())
        self.assertEqual((record["status"], record["reason"]), ("not_started", "campaign_close"))
        self.assertFalse(T.needs_attempt(record))
        self.assertFalse((self.root / "out" / DATASET / "fit-37" / "attempt-1").exists())

    def test_a_stopped_controllers_attempt_is_adopted_and_its_key_never_reused(self):
        attempt = self.cell / "attempt-1" / "fit"
        attempt.mkdir(parents=True)
        (attempt / "operation.request.json").write_text(json.dumps({"boot_id": "boot-a"}))
        same_boot = self.controller.adopt(23, self.row, self.cell, "fit")
        self.assertEqual((same_boot["status"], same_boot["attempts"]), ("unknown", 1))
        self.controller.boot = lambda: "boot-b"
        rebooted = self.controller.adopt(23, self.row, self.cell, "fit")
        self.assertEqual((rebooted["status"], rebooted["reason"]), ("infrastructure_failure", "host_reboot"))
        self.assertTrue(T.needs_attempt(rebooted))
        with patch.object(T.subprocess, "Popen", FakeProcess):
            retried = self.controller.fit(23, self.row, self.cell, rebooted)
        self.assertEqual(retried["attempts"], 2)
        self.assertFalse(T.needs_attempt(retried))
        self.assertIsNone(self.controller.adopt(37, self.row, self.root / "out" / DATASET / "fit-37", "fit"))

    def test_a_listed_cause_is_confirmed_by_the_controllers_own_observation(self):
        attempt = self.root / "attempt"
        attempt.mkdir()
        usage = collections.namedtuple("usage", "total used free")
        for pids, expected in (({1234}, True), (set(), False), (None, False)):
            with self.subTest(pids=pids), patch.object(T.v1, "_compute_pids", lambda pids=pids: pids):
                self.assertIs(self.controller.confirm("foreign_gpu_owner", attempt), expected)
        for free, expected in ((0, True), (1 << 40, False)):
            with self.subTest(free=free), patch.object(T.shutil, "disk_usage", lambda _p, free=free: usage(1, 1, free)):
                self.assertIs(self.controller.confirm("storage_refusal", attempt), expected)
        lock = self.root / "runtime.lock.json"
        lock.write_bytes(b"x")
        self.controller.amendment = {**self.amendment, "runtime": {"path": str(lock), "sha256": sha(b"x")}}
        self.assertFalse(self.controller.confirm("runtime_lock_mismatch", attempt))
        lock.write_bytes(b"y")
        self.assertTrue(self.controller.confirm("runtime_lock_mismatch", attempt))
        self.assertFalse(self.controller.confirm("cosmic_ray", attempt))

    def test_the_worker_names_only_predeclared_causes(self):
        self.assertEqual(W.listed_cause(OSError(errno.ENOSPC, "full")), "storage_refusal")
        self.assertEqual(W.listed_cause(RuntimeError("foreign_gpu_owner")), "foreign_gpu_owner")
        self.assertEqual(W.listed_cause(RuntimeError("runtime_lock_mismatch")), "runtime_lock_mismatch")
        for error in (RuntimeError("operation_identity_differs"), ValueError("x"), OSError(errno.EIO, "io")):
            self.assertIsNone(W.listed_cause(error))
        receipt = W.failure_receipt(RuntimeError("joint_input_differs"), time.monotonic(), "r", True)
        self.assertEqual((receipt["status"], receipt["cause"], receipt["cuda_initialized"]),
                         ("operation_failed", None, True))
        lock = self.root / "lock.json"
        lock.write_bytes(b"x")
        with self.assertRaisesRegex(RuntimeError, "runtime_lock_mismatch"):
            W.runtime_lock({"runtime": {"path": str(lock), "sha256": "0" * 64}})


class LedgerTests(unittest.TestCase):
    def build(self, out: Path) -> list:
        rows = [{"dataset": DATASET, "train_rows": 2}, {"dataset": "fedcba9876543210", "train_rows": 2}]
        plan = T.seed_major(rows, [23, 37])
        good = out / DATASET / "fit-23"
        good.mkdir(parents=True)
        (good / T.FIT_RECORD).write_text(json.dumps({"status": "ok", "reason": None, "attempts": 1,
                                                     "artifact_bytes": 84_000_000}))
        cells = []
        for item in T.sample_cells(rows[0], good, GRID["sizes"], GRID["sample_seeds"]):
            body = "f0,target\n" + "1.0,2.0\n" * item["rows"]
            Path(item["path"]).write_text(body)
            cells.append({**{k: v for k, v in item.items() if k != "path"}, "status": "ok", "reason": None,
                          "synthetic_sha256": sha(body.encode())})
        (good / T.SAMPLE_RECORD).write_text(json.dumps({"status": "done", "attempts": 1, "cells": cells}))
        failed = out / DATASET / "fit-37"
        failed.mkdir(parents=True)
        (failed / T.FIT_RECORD).write_text(json.dumps({"status": "failed", "reason": "ValueError", "attempts": 1}))
        return plan

    def test_every_planned_cell_is_in_the_ledger_and_none_is_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build(Path(tmp))
            fits, cells = T.ledger(plan, Path(tmp), GRID)
        self.assertEqual([item["status"] for item in fits], ["ok", "not_run", "failed", "not_run"])
        self.assertEqual(len(cells), 24)
        statuses = [item["status"] for item in cells]
        self.assertEqual(statuses[:6], ["ok"] * 6)
        self.assertEqual(set(statuses[6:12]), {"fit_not_run"})
        self.assertEqual(set(statuses[12:18]), {"fit_failed"})
        self.assertTrue(all(item["synthetic_sha256"] is None for item in cells[6:]))

    def test_a_sample_file_that_differs_from_its_receipt_stops_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build(Path(tmp))
            (Path(tmp) / DATASET / "fit-23" / "sample-211-4n.csv").write_text("f0,target\n9,9\n")
            with self.assertRaisesRegex(SystemExit, "differs from its receipt"):
                T.ledger(plan, Path(tmp), GRID)

    def test_binding_pins_the_run_and_computes_no_median(self):
        amendment = T.load_amendment()
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build(Path(tmp) / "out")
            fits, cells = T.ledger(plan, Path(tmp) / "out", GRID)
            binding = T.write_ledger(fits, cells, Path(tmp) / "results", amendment, "c" * 40)
            ledger_text = (Path(tmp) / "results" / "sample-cells.jsonl").read_bytes()
        self.assertEqual(binding["cells_sha256"], sha(ledger_text))
        self.assertEqual(binding["cells"], {"fit_failed": 6, "fit_not_run": 12, "ok": 6})
        self.assertEqual(binding["amendment_sha256"], amendment["sha256"])
        self.assertIs(binding["medians_not_computed"], True)
        self.assertNotIn("retention", ledger_text.decode())


class ScoringTests(unittest.TestCase):
    def ledger_file(self, tmp: Path, rows: list[dict]) -> Path:
        results = tmp / "results"
        results.mkdir()
        (results / "binding.json").write_text(json.dumps({"configuration": CONFIGURATION}))
        path = results / "sample-cells.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def row(self, **fields) -> dict:
        return {"dataset": DATASET, "fit_seed": 23, "sample_seed": 101, "size": 4, "status": "ok",
                "synthetic_sha256": "a" * 64, **fields}

    def test_k0_adapter_reads_headed_block_b_samples_with_no_stored_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.ledger_file(Path(tmp), [self.row(), self.row(sample_seed=211, status="failed",
                                                                     synthetic_sha256=None)])
            jobs = A.tabsyn_seeds(path, {"tabsyn-seeds": "/scratch/tabsyn-seeds"})
        ok, blocked = jobs
        self.assertEqual(ok["csv_path"], f"/scratch/tabsyn-seeds/{DATASET}/fit-23/sample-101-4n.csv")
        self.assertEqual((ok["method"], ok["configuration"], ok["fit_seed"]), ("TabSyn", CONFIGURATION, 23))
        self.assertIs(ok["has_header"], True)
        self.assertIsNone(ok["stored_retention"])
        self.assertEqual(blocked["blocked"], ["sample_absent", "ledger_failed"])

    def test_k0_adapter_refuses_seed_eleven_or_undeclared_seeds(self):
        for seed in (11, 99):
            with self.subTest(seed), tempfile.TemporaryDirectory() as tmp:
                path = self.ledger_file(Path(tmp), [self.row(fit_seed=seed)])
                with self.assertRaisesRegex(ValueError, "not at fit seed 23"):
                    A.tabsyn_seeds(path, {})

    def test_a_scored_cell_binds_into_the_v2_ledger_inside_the_roster(self):
        contract = B._contract()
        report = {"implementation_sha256": contract["implementation_sha256"],
                  "dependencies": dict(contract["dependencies"]), "null_loss": 2.0,
                  "utility": {auditor: {"trtr_loss": 1.0, "tstr_loss": 1.2, "informative": True, "retention": 0.8}
                              for auditor in B.AUDITORS}}
        with tempfile.TemporaryDirectory() as tmp:
            job = A.tabsyn_seeds(self.ledger_file(Path(tmp), [self.row()]), {})[0]
            record = L.cell_record(job, "ok", None, {"file": "sample-cells.jsonl", "sha256": "b" * 64},
                                   {"predeclare_sha256": contract["sha256"]}, report, "a" * 64)
            root = Path(tmp) / "k0"
            path = L.cell_path(root, job)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record))
            (root / "TabSyn" / T.FIT_RECORD).write_text(json.dumps({"status": "ok"}))
            cells = B.collect([(root, "block_b", None)], contract)
            fits = B.collect_fits([(root, "block_b", None)])
        self.assertEqual([(c["method"], c["configuration"], c["fit_seed"]) for c in cells],
                         [("TabSyn", CONFIGURATION, 23)])
        self.assertEqual(fits, [])
        P.check_roster(cells)

    def test_tabsyn_stays_a_one_fit_arm_until_block_b_is_bound(self):
        # The commit that binds the block B cells adds the arm to FIVE_SEED and flips this test.
        self.assertNotIn(("TabSyn", CONFIGURATION), P.FIVE_SEED)


if __name__ == "__main__":
    unittest.main()
