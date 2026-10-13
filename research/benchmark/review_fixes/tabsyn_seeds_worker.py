"""Fit or sample one TabSyn scaled lineage at a v2 fit seed (predeclare_v2 block B).

tabsyn_seeds_v2.py starts this file under the pinned TabSyn runtime python
with -I -S -B, the way the seed-11 owner queue started its entry. The
scientific code is the captured adapter, authenticated by its sha256. Two AST
edits are made and counted: the four runtime and admission imports become
in-process guards, and the fit seed admission widens from 11 to the five
predeclared seeds. No model, loss, epoch, optimizer, or sampler statement
changes. The process reads train.csv, validation.csv and projection.json of
one worker and never opens a file named test.csv. Standard library only until
the runtime path is switched.
"""

import ast
import csv
import errno
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import struct
import sys
import time
import traceback
import types
from pathlib import Path

GUARDED_MODULES = ("research.benchmark.tabsyn_runtime_guard", "research.benchmark.tabsyn_admission")
ADMISSION_IMPORTS = 4
SAMPLE_SEEDS = [101, 211, 307]
EPOCH_BUDGET = {"vae_epochs": 200, "diffusion_epochs": 1000}
SCIENTIFIC = ("numpy", "torch", "scipy", "sklearn", "xgboost")
BINDINGS = "block_b_tabsyn_bindings"
REJECTED_MODULES = frozenset({"subprocess", "socket", "ctypes", "pickle", "shutil", "pty", "multiprocessing"})
REJECTED_CALLS = frozenset({"eval", "exec", "compile", "__import__", "breakpoint", "input"})


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checked(path, pin: str) -> bytes:
    """Read a pinned file; refuse a symlink on the path or a digest change."""
    target = Path(path)
    if target.name == "test.csv" or any(item.is_symlink() for item in (target, *target.parents)):
        raise ValueError("refused_path")
    raw = target.read_bytes()
    if digest(raw) != pin:
        raise ValueError("pinned_digest_changed")
    return raw


def put(root: Path, name: str, value: dict) -> dict:
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with (root / name).open("xb") as handle:
        handle.write(raw)
    return {"path": str(root / name), "sha256": digest(raw)}


def start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


class _StripAdmission(ast.NodeTransformer):
    def __init__(self):
        self.stripped = []

    def visit_ImportFrom(self, node):
        if node.module in GUARDED_MODULES:
            self.stripped.append(node.module)
            return ast.Pass()
        return node


def _is_seed_eleven(node) -> bool:
    return (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "seed"
            and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq) and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 11
            and type(node.comparators[0].value) is int)


def widen_fit_seed(tree: ast.Module, seeds: list[int]) -> int:
    """Replace fit's `seed == 11` admission with `seed in [seeds]`; return the count of edits."""
    edits = 0
    for function in tree.body:
        if not (isinstance(function, ast.FunctionDef) and function.name == "fit"):
            continue
        for node in ast.walk(function):
            if _is_seed_eleven(node):
                node.ops = [ast.In()]
                node.comparators = [ast.List(elts=[ast.Constant(value=seed) for seed in seeds], ctx=ast.Load())]
                edits += 1
    return edits


def reject_sinks(tree: ast.Module) -> None:
    """Refuse process, socket, serialization, or dynamic-code sinks, as tabsyn_sample.py does."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            names += [alias.name for alias in node.names]
            if any(name == "*" or name.split(".", 1)[0] in REJECTED_MODULES | {"builtins"} for name in names):
                raise ValueError("adapter import is outside the authenticated boundary")
        elif isinstance(node, ast.Name) and node.id in REJECTED_CALLS | {"__builtins__"}:
            # A bare name, called or aliased; torch's module.eval() is an attribute and stays allowed.
            raise ValueError("adapter call is outside the authenticated boundary")


def transform(raw: bytes, seeds: list[int]) -> ast.Module:
    """The captured adapter with exactly four imports stripped and one seed admission widened."""
    if sorted(seeds) != seeds or 11 not in seeds:
        raise ValueError("fit seeds must be sorted and keep the historical seed")
    tree = ast.parse(raw)
    strip = _StripAdmission()
    tree = strip.visit(tree)
    if len(strip.stripped) != ADMISSION_IMPORTS:
        raise ValueError(f"admission import strip count is {len(strip.stripped)}")
    if widen_fit_seed(tree, seeds) != 1:
        raise ValueError("fit seed admission is not exactly one `seed == 11` comparison")
    reject_sinks(tree)
    return ast.fix_missing_locations(tree)


def joint_rows(path: Path) -> list[list[float]]:
    """Worker CSV rows in TabSyn joint order: target first, then the projected inputs."""
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return [[float(row[-1]), *[float(value) for value in row[:-1]]] for row in reader]


def joint_sha256(rows: list[list[float]]) -> str:
    """sha256 of the C-order float64 bytes, equal to numpy's tobytes on a little-endian host."""
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(struct.pack(f"<{len(row)}d", *row))
    return hasher.hexdigest()


def cell_receipt_name(cell: dict) -> str:
    """Per-cell receipt beside the operation, so a killed operation keeps the cells it finished."""
    return f"sample-{cell['seed']}-{cell['size']}n.receipt.json"


def write_sample(path: Path, matrix) -> None:
    """Write target-first rows with the target moved last, as the seed-11 sampler did."""
    staging = path.with_suffix(".csv.tmp")
    with staging.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"f{index}" for index in range(len(matrix[0]) - 1)] + ["target"])
        writer.writerows([[*row[1:], row[0]] for row in matrix])
    staging.replace(path)


def runtime_lock(q: dict) -> bytes:
    """The pinned runtime lock; a changed or missing lock is the listed runtime-lock mismatch."""
    try:
        return checked(q["runtime"]["path"], q["runtime"]["sha256"])
    except (ValueError, OSError):
        raise RuntimeError("runtime_lock_mismatch") from None


def listed_cause(error: BaseException) -> str | None:
    """The predeclared infrastructure cause an exception names, if any; everything else is the operation's own."""
    if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT):
        return "storage_refusal"
    if isinstance(error, RuntimeError) and str(error) in ("foreign_gpu_owner", "runtime_lock_mismatch"):
        return str(error)
    return None


def failure_receipt(error: BaseException, started: float, request_sha: str, cuda_ready: bool) -> dict:
    """A worker claim only: the controller retries a listed cause after confirming it on its own."""
    cause = listed_cause(error)
    return {"status": "infrastructure_failure" if cause else "operation_failed", "cause": cause,
            "exception_type": type(error).__name__, "cuda_initialized": cuda_ready,
            "operation_seconds": time.monotonic() - started, "request_sha256": request_sha,
            "official_tests_opened": False}


def _gpu_owned_only() -> None:
    import subprocess
    result = subprocess.run(["/usr/bin/nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, check=True, timeout=10)
    pids = [int(line) for line in result.stdout.decode().splitlines() if line.strip()]
    if any(pid != os.getpid() for pid in pids):
        raise RuntimeError("foreign_gpu_owner")


class Operation:
    """One authenticated request and the custody checks that every adapter guard call repeats."""

    def __init__(self, argv: list[str]):
        self.root, self.request_path, self.request_sha = Path(argv[1]), Path(argv[2]), argv[3]
        self.q = json.loads(checked(self.request_path, self.request_sha))
        checked(Path(__file__).resolve(), self.q["worker_code_sha256"])
        self.started, self.end = self.q["operation_started_monotonic"], self.q["deadline_monotonic"]
        self.worker = Path(self.q["worker"]["path"])
        self.grant = {"operation_path": str(self.request_path), "operation_sha256": self.request_sha}
        self.runtime = json.loads(runtime_lock(self.q))

    def own(self) -> None:
        q = self.q
        if os.getuid() != 1000 or os.uname().nodename != q["host"]:
            raise RuntimeError("host_or_user_differs")
        if sorted(os.sched_getaffinity(0)) != q["cpu_slot"]:
            raise RuntimeError("cpu_slot_differs")
        if os.getppid() != q["controller_pid"] or start_ticks(os.getppid()) != q["controller_start_ticks"]:
            raise RuntimeError("controller_differs")
        if time.monotonic() >= self.end:
            raise TimeoutError("whole_operation_deadline")

    def check_inputs(self) -> None:
        for name, pin in self.q["worker"]["files"].items():
            checked(self.worker / name, pin)

    def check_runtime(self) -> None:
        """The pinned interpreter, author source, and worker files, before any scientific import."""
        runtime = self.runtime
        if digest(Path(runtime["python"]).read_bytes()) != runtime["python_sha256"]:
            raise RuntimeError("runtime_python_differs")
        if os.path.realpath(sys.executable) != runtime["python"]:
            raise RuntimeError("not_the_runtime_python")
        self.check_inputs()
        for name, pin in runtime["roots"][2]["files"].items():
            checked(Path(runtime["source"]) / name, pin["sha256"])
        if set(SCIENTIFIC).intersection(sys.modules):
            raise RuntimeError("scientific_module_loaded_early")

    def verify_runtime(self, *_args) -> dict:
        self.own()
        runtime_lock(self.q)
        checked(self.q["adapter"]["path"], self.q["adapter"]["sha256"])
        return self.runtime

    def verify_operation(self, given, mode, config_sha, runtime_sha, _code_path, _code_sha,
                         phase="before_dependencies") -> dict:
        q = self.q
        self.own()
        _gpu_owned_only()
        if (mode != q["mode"] or config_sha != q["config"]["config_sha256"]
                or runtime_sha != q["runtime"]["sha256"] or given != self.grant):
            raise RuntimeError("operation_identity_differs")
        self.check_inputs()
        return {"deadline_monotonic": self.end, "epoch_budget": dict(EPOCH_BUDGET),
                "projection_sha256": q["projection_sha256"], "joint_train_sha256": q["joint_train_sha256"],
                "joint_validation_sha256": q["joint_validation_sha256"], "samples": q.get("samples", [])}


class InventoryGuard(importlib.abc.MetaPathFinder):
    """Every import under the runtime base must be inventoried; its source compiles from the checked bytes."""

    def __init__(self, runtime: dict):
        self.base = runtime["base"] + "/"
        self.inventory = {str(Path(row["root"]) / name): pin["sha256"]
                          for row in runtime["roots"] for name, pin in row["files"].items()}

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and spec.origin.startswith(self.base):
            pin = self.inventory.get(spec.origin)
            if pin is None:
                raise ImportError("uninventoried_runtime_import")
            raw = checked(spec.origin, pin)
            if isinstance(spec.loader, importlib.machinery.SourceFileLoader):

                class CheckedLoader(importlib.machinery.SourceFileLoader):
                    def get_code(self, name):
                        return self.source_to_code(raw, self.path)

                spec.loader = CheckedLoader(fullname, spec.origin)
        return spec


def load_adapter(tree: ast.Module, staging: Path, operation: Operation) -> dict:
    """Load the edited adapter from one exclusive payload file through a frozen loader, as tabsyn_sample.py does."""
    bindings = types.ModuleType(BINDINGS)
    bindings.verify_runtime, bindings.verify_operation = operation.verify_runtime, operation.verify_operation
    sys.modules[BINDINGS] = bindings
    payload = (f"from {BINDINGS} import verify_runtime, verify_operation\n" + ast.unparse(tree) + "\n").encode()
    with staging.open("xb") as handle:
        handle.write(payload)

    class FrozenPayloadLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            # The authenticated payload is the only code; a later read of the path is never consulted.
            return self.source_to_code(payload, self.path)

    spec = importlib.util.spec_from_file_location(
        "block_b_tabsyn_adapter", staging, loader=FrozenPayloadLoader("block_b_tabsyn_adapter", str(staging)))
    if spec is None or staging.read_bytes() != payload:
        raise RuntimeError("adapter_payload_changed_before_load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


def fit_mode(adapter: dict, operation: Operation, np) -> dict:
    q, worker = operation.q, operation.worker
    train = np.asarray(joint_rows(worker / "train.csv"), dtype=np.float64)
    validation = np.asarray(joint_rows(worker / "validation.csv"), dtype=np.float64)
    if (train.shape[0] != q["train_rows"] or digest(train.tobytes(order="C")) != q["joint_train_sha256"]
            or digest(validation.tobytes(order="C")) != q["joint_validation_sha256"]):
        raise RuntimeError("joint_input_differs")
    projection = checked(worker / "projection.json", q["projection_sha256"])
    receipt = adapter["fit"](
        train, validation, projection, operation.root / "artifact", q["config"], seed=q["fit_seed"],
        device="cuda:0", timeout_seconds=min(600, int(operation.end - time.monotonic())), generated_fixture=False,
        runtime_path=q["runtime"]["path"], runtime_sha256=q["runtime"]["sha256"],
        code_path=str(operation.request_path), code_sha256=operation.request_sha, admission=operation.grant)
    return {"result": receipt, "inventory": adapter["artifact_inventory"](operation.root / "artifact")}


def sample_cell(adapter: dict, operation: Operation, artifact: Path, cell: dict) -> dict:
    """One sample cell; a refused or failed cell is recorded and the next cell still runs, as for seed 11.

    A listed infrastructure cause (a foreign GPU owner, a runtime-lock mismatch, a
    storage refusal) is not the cell's own failure: it ends the operation, so its
    failure receipt carries the cause for the controller to confirm.
    """
    q, path, begun = operation.q, Path(cell["path"]), time.monotonic()
    if path.name == "test.csv":
        raise ValueError("refused_path")
    status, reason, sha = "timeout", None, None
    if begun < operation.end:
        try:
            matrix = adapter["sample"](
                artifact, cell["rows"], cell["seed"], q["artifact_inventory"], device="cuda:0",
                runtime_path=q["runtime"]["path"], runtime_sha256=q["runtime"]["sha256"],
                code_path=q["fit_ref"]["path"], code_sha256=q["fit_ref"]["sha256"], admission=operation.grant)
            write_sample(path, matrix)
            status, sha = "ok", digest(path.read_bytes())
        except (ValueError, RuntimeError, OSError) as error:
            if listed_cause(error):
                raise
            status, reason = "failed", type(error).__name__
    entry = {**cell, "status": status, "reason": reason, "synthetic_sha256": sha,
             "seconds": round(time.monotonic() - begun, 3), "official_tests_opened": False}
    put(operation.root, cell_receipt_name(cell), entry)
    return entry


def sample_mode(adapter: dict, operation: Operation) -> list[dict]:
    artifact = Path(operation.q["artifact_path"])
    if adapter["artifact_inventory"](artifact) != operation.q["artifact_inventory"]:
        raise RuntimeError("artifact_custody_differs")
    return [sample_cell(adapter, operation, artifact, cell) for cell in operation.q["samples"]]


def main(argv: list[str]) -> int:
    root, started = Path(argv[1]), time.monotonic()
    try:
        operation = Operation(argv)
    except BaseException as error:
        put(root, "request.failure.receipt.json", failure_receipt(error, started, argv[3], False))
        return 2
    return run(operation)


def run(operation: Operation) -> int:
    q, root = operation.q, operation.root
    staging = root / "adapter.allowlisted.py"
    cuda_ready = False
    try:
        operation.check_runtime()
        runtime = operation.runtime
        sys.path[:] = [runtime["source"], runtime["package_site"], runtime["stdlib"], runtime["dynload"]]
        sys.meta_path.insert(0, InventoryGuard(runtime))
        operation.own()
        _gpu_owned_only()
        put(root, "worker.before-imports.receipt.json", {
            "pid": os.getpid(), "start_ticks": start_ticks(os.getpid()), "request_sha256": operation.request_sha,
            "mode": q["mode"], "official_tests_opened": False})
        tree = transform(checked(q["adapter"]["path"], q["adapter"]["sha256"]), q["admitted_fit_seeds"])
        adapter = load_adapter(tree, staging, operation)
        import numpy as np
        import torch
        torch.cuda.set_per_process_memory_fraction(16 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
        cuda_ready = True
        put(root, "worker.CUDA-initialized.receipt.json", {
            "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0), "request_sha256": operation.request_sha,
            "adapter_sha256": q["adapter"]["sha256"], "official_tests_opened": False})
        if q["mode"] not in ("fit", "sample"):
            raise ValueError("mode")
        result = fit_mode(adapter, operation, np) if q["mode"] == "fit" else sample_mode(adapter, operation)
        operation.own()
        put(root, f"{q['mode']}.complete.receipt.json", {
            "status": "ok", "mode": q["mode"], "result": result, "method": "TabSyn",
            "config_sha256": q["config"]["config_sha256"], "fit_seed": q["fit_seed"],
            "operation_seconds": time.monotonic() - operation.started, "official_tests_opened": False})
        return 0
    except BaseException as error:
        put(root, f"{q['mode']}.failure.receipt.json",
            failure_receipt(error, operation.started, operation.request_sha, cuda_ready))
        frames = [{"function": frame.name, "line": frame.lineno} for frame in traceback.extract_tb(error.__traceback__)
                  if frame.filename == str(staging)]
        print(json.dumps({"exception_type": type(error).__name__, "adapter_frames": frames}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
