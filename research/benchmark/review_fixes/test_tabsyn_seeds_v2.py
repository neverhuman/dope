"""Block B (TabSyn scaled arm, fit seeds 23-71) runner, worker, ledger, and scoring custody.

Fixtures only: no benchmark data, GPU, or runtime python. The captured adapter
is the repository copy, whose digest the amendment pins.
"""

import ast
import hashlib
import json
import os
import struct
import tempfile
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
            "ast", "csv", "hashlib", "importlib", "json", "os", "struct", "sys", "time", "traceback", "types",
            "pathlib"})
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

    def test_fit_status_reads_exit_codes_and_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.receipt(root / "ok", "fit.complete.receipt.json", status="ok")
            self.receipt(root / "deadline", "fit.failure.receipt.json", status="operation_failed",
                         exception_type="TimeoutError")
            self.receipt(root / "cuda", "fit.failure.receipt.json", status="infrastructure_failure",
                         exception_type="RuntimeError")
            self.receipt(root / "nan", "fit.failure.receipt.json", status="operation_failed",
                         exception_type="ValueError")
            (root / "silent").mkdir()
            self.assertEqual(T.fit_status(root / "ok", 0), ("ok", None))
            self.assertEqual(T.fit_status(root / "ok", 124), ("timeout", "wall"))
            self.assertEqual(T.fit_status(root / "deadline", 2), ("timeout", "deadline"))
            self.assertEqual(T.fit_status(root / "cuda", 2), ("infrastructure_failure", "RuntimeError"))
            self.assertEqual(T.fit_status(root / "nan", 2), ("failed", "ValueError"))
            self.assertEqual(T.fit_status(root / "silent", -9), ("infrastructure_failure", "no_receipt"))

    def test_charged_bytes_leave_out_intermediate_diffusion_checkpoints(self):
        inventory = {"files": {"diffusion.safetensors": {"bytes": 50}, "diffusion-epoch-0.safetensors": {"bytes": 50},
                               "decoder-final.safetensors": {"bytes": 7}, "projection.json": {"bytes": 3}}}
        self.assertEqual(T.charged_bytes(inventory), 60)

    def test_only_an_infrastructure_failure_retries_and_only_once(self):
        self.assertTrue(T.needs_attempt(None))
        self.assertTrue(T.needs_attempt({"status": "infrastructure_failure", "attempts": 1}))
        self.assertFalse(T.needs_attempt({"status": "infrastructure_failure", "attempts": 2}))
        for status in ("ok", "failed", "timeout", "done"):
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
        self.assertEqual([item["status"] for item in statuses], ["ok"] + ["timeout"] * 5)
        self.assertTrue(all("path" not in item for item in statuses))


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
