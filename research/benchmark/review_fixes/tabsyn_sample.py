"""Sample the already fitted scaled TabSyn models. Does not refit them.

The parser check scores the published sample-101-4n.csv for dataset
02a45777900441ce and exits 2 on a mismatch. New cells are written only after
that check. The captured adapter is executed with the four admission imports
removed and with seeds left at 101, 211, and 307. The expired owner window,
parent-pid lock, and 12 hour budget are not reused. Artifacts are read only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import signal
import subprocess
import time
from pathlib import Path

RUNTIME_LOCK = Path("/mnt/fast-scratch/dope-benchmark/tabsyn-x2-runtime-v1/runtime.lock.json")
QUEUE = Path("/home/ubuntu/dope-scratch-x1/ts100-owner-queue-v2/operations")
DIRECT = Path("/home/ubuntu/dope-scratch-x1/ts100-owner-direct-v1")
OUTPUT = Path("/home/ubuntu/dope-scratch-x1/dope-rf-tabsyn-sample-v1")
PANEL_DATASET = "02a45777900441ce"
PUBLISHED_CSV = Path(
    "/home/ubuntu/dope-scratch-x1/ts-GPU-default-samples-v3/operations/"
    "0001-02a45777900441ce-sample/sample-101-4n.csv"
)
WORKERS = Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker")
CONFIG_SHA = "a34729383eac6783c7e754a791094054d54f1062655db9a306cd4cca296c5ebc"
EPOCHS = {"vae_epochs": 200, "diffusion_epochs": 1000}
SEEDS = (101, 211, 307)
TIMEOUT_SECONDS = 180


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _memory_fraction() -> float:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        info[key] = int(rest.split()[0])
    return info["MemAvailable"] / info["MemTotal"]


def _gpu_pids() -> list[int]:
    result = subprocess.run(
        ["/usr/bin/nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, check=True, timeout=15, text=True,
    )
    return [int(line) for line in result.stdout.splitlines() if line.strip()]


def _belongs_to_us(pid: int) -> bool:
    current = pid
    for _ in range(8):
        if current == os.getpid():
            return True
        stat_path = Path(f"/proc/{current}/stat")
        if not stat_path.exists():
            return False
        rest = stat_path.read_text().rsplit(")", 1)[-1].split()
        if len(rest) < 2:
            return False
        parent = int(rest[1])
        if parent == current:
            return False
        current = parent
    return False


def _host_gate() -> None:
    if os.uname().nodename != "xbabe1" or os.getuid() != 1000:
        raise SystemExit("TabSyn sampling is pinned to ubuntu on xbabe1")
    if _memory_fraction() < 0.15:
        raise SystemExit("refusing: less than 15 percent of RAM is free")
    foreign = [pid for pid in _gpu_pids() if not _belongs_to_us(pid)]
    if foreign:
        raise SystemExit(f"refusing: GPU is owned by {foreign}")


def _request() -> dict:
    path = next(QUEUE.glob("*/operation.request.json"))
    return json.loads(path.read_text())


def _dataset_from_dirname(name: str) -> str | None:
    if not name.endswith("-fit"):
        return None
    body = name[:-4]
    _index, separator, dataset = body.partition("-")
    if not separator or not dataset:
        return None
    return dataset


def _artifacts() -> list[tuple[str, Path]]:
    found = []
    for path in sorted(QUEUE.glob("*-fit/artifact/model.json")):
        dataset = _dataset_from_dirname(path.parents[1].name)
        if dataset:
            found.append((dataset, path.parent))
    direct_model = DIRECT / "artifact" / "model.json"
    if direct_model.is_file():
        request = json.loads((DIRECT / "real-fit.request.json").read_text())
        dataset = Path(request["worker"]["path"]).name
        found.append((dataset, direct_model.parent))
    unique = {}
    for dataset, artifact in found:
        unique.setdefault(dataset, artifact)
    return sorted(unique.items())


def _ledger(row: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "ledger.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parser_check(panel: Path, pilot_path: Path) -> int:
    _host_gate()
    document = json.loads(panel.read_text())
    expected = None
    for cell in document["cells"]:
        if (cell.get("dataset") == PANEL_DATASET and cell.get("sample_seed") == 101
                and cell.get("row_multiplier") == 4):
            expected = cell["metrics"]["utility"]["auditors"]["catboost"]["retention"]
            break
    if not isinstance(expected, float):
        print("parser check could not read the published retention")
        return 2
    if not PUBLISHED_CSV.is_file():
        print("parser check published csv is absent")
        return 2
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("pilot_metrics", pilot_path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    raw = np.loadtxt(PUBLISHED_CSV, delimiter=",", skiprows=1, ndmin=2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    headerless = OUTPUT / "parser-check-headerless.csv"
    np.savetxt(headerless, raw, delimiter=",")
    root = WORKERS / PANEL_DATASET
    task = json.loads((root / "projection.json").read_text())["task"]
    report = pilot.measure(root / "train.csv", root / "validation.csv", headerless, task)
    got = (report.get("utility") or {}).get("catboost", {}).get("retention")
    match = isinstance(got, float) and abs(got - expected) <= 1e-9
    payload = {
        "dataset": PANEL_DATASET,
        "sample_seed": 101,
        "row_multiplier": 4,
        "expected_retention": expected,
        "scored_retention": got,
        "match": match,
        "official_tests_opened": False,
    }
    (OUTPUT / "parser-check.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"parser check match={match}")
    return 0 if match else 2


def _load_namespace(request: dict):
    runtime_path = Path(request["runtime_ref"]["path"])
    runtime_pin = request["runtime_ref"]["sha256"]
    if _sha(runtime_path) != runtime_pin:
        raise SystemExit("runtime lock hash changed")
    runtime = json.loads(runtime_path.read_text())
    if os.path.realpath(sys_executable()) != os.path.realpath(runtime["python"]):
        raise SystemExit("sample lineage is not running under the pinned runtime python")
    adapter_path = Path(request["adapter_ref"]["path"])
    adapter_pin = request["adapter_ref"]["sha256"]
    raw = adapter_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != adapter_pin:
        raise SystemExit("captured adapter hash changed")
    if b"[11, 101, 211, 307]" in raw or b"[11,101,211,307]" in raw:
        raise SystemExit("captured adapter already contains the seed-11 patch")
    tree = ast.parse(raw)
    changes = []

    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module in ("research.benchmark.tabsyn_runtime_guard", "research.benchmark.tabsyn_admission"):
                changes.append(node.module)
                return ast.Pass()
            return node

    tree = ast.fix_missing_locations(Strip().visit(tree))
    if len(changes) != 4:
        raise SystemExit(f"adapter import strip count is {len(changes)}")
    inventory = {
        str(Path(row["root"]) / name): pin["sha256"]
        for row in runtime["roots"]
        for name, pin in row["files"].items()
    }

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.startswith(runtime["base"] + "/"):
                pin = inventory.get(spec.origin)
                if pin is None or _sha(Path(spec.origin)) != pin:
                    raise ImportError("uninventoried TabSyn runtime import")
            return spec

    sys_modules_path(runtime)
    import sys
    sys.meta_path.insert(0, Guard())
    holder = {}

    def verify_runtime(*_args):
        _host_gate()
        if _sha(runtime_path) != runtime_pin or _sha(adapter_path) != adapter_pin:
            raise RuntimeError("runtime pin changed during sampling")
        return runtime

    def verify_operation(grant, mode, config_sha, runtime_sha, _code_path, code_sha, phase="before_dependencies"):
        _host_gate()
        if mode != "sample" or config_sha != holder["config_sha"] or runtime_sha != holder["runtime_sha"]:
            raise RuntimeError("sample admission does not match the artifact")
        if code_sha != holder["code_sha"] or grant != holder["grant"]:
            raise RuntimeError("sample code grant does not match the artifact")
        return holder["operation"]

    namespace = {
        "__file__": str(adapter_path),
        "verify_runtime": verify_runtime,
        "verify_operation": verify_operation,
    }
    exec(compile(tree, "captured_review_fix_tabsyn_adapter", "exec"), namespace)
    namespace["_holder"] = holder
    return namespace, runtime


def sys_executable() -> str:
    import sys
    return sys.executable


def sys_modules_path(runtime: dict) -> None:
    import sys
    sys.path[:] = [runtime["source"], runtime["package_site"], runtime["stdlib"], runtime["dynload"]]


def sample_lineage(dataset: str, artifact: Path, git_sha: str) -> int:
    import sys
    request = _request()
    runtime = json.loads(Path(request["runtime_ref"]["path"]).read_text())
    if os.path.realpath(sys.executable) != os.path.realpath(runtime["python"]):
        os.environ["LD_LIBRARY_PATH"] = runtime.get("LD_LIBRARY_PATH") or ""
        preload = runtime.get("LD_PRELOAD") or ""
        if preload:
            os.environ["LD_PRELOAD"] = preload
        os.environ["OMP_NUM_THREADS"] = "1"
        os.execv(runtime["python"], [runtime["python"], os.path.abspath(__file__), *sys.argv[1:]])
    _host_gate()
    info = json.loads((artifact / "model.json").read_text())
    if info.get("declared_epoch_budget") != EPOCHS or info.get("generated_fixture") is not False:
        _ledger({"dataset": dataset, "status": "skipped", "reason": "epoch_budget_differs"})
        return 0
    config_sha = (info.get("config") or {}).get("config_sha256")
    if config_sha != CONFIG_SHA or info.get("execution_admitted") is not True:
        _ledger({"dataset": dataset, "status": "skipped", "reason": "config_or_admission_differs"})
        return 0
    rows = info.get("rows")
    if not isinstance(rows, int) or rows <= 0:
        _ledger({"dataset": dataset, "status": "skipped", "reason": "rows_missing"})
        return 0
    namespace, _runtime = _load_namespace(request)
    holder = namespace["_holder"]
    holder.update({
        "config_sha": config_sha,
        "runtime_sha": info["runtime_sha256"],
        "code_sha": info["code_sha256"],
        "grant": {"operation_path": "review-fixes", "operation_sha256": "review-fixes"},
    })
    import torch
    torch.cuda.set_per_process_memory_fraction(
        (16 * 2**30) / torch.cuda.get_device_properties(0).total_memory, 0)
    inventory = namespace["artifact_inventory"](artifact)
    destination = OUTPUT / dataset
    destination.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + TIMEOUT_SECONDS
    for size in (1, 4):
        for seed in SEEDS:
            final = destination / f"sample-{seed}-{size}n.csv"
            if final.exists():
                continue
            if time.monotonic() >= deadline:
                _ledger({"dataset": dataset, "seed": seed, "size": size, "status": "timeout", "git_sha": git_sha})
                continue
            remaining = max(1, int(deadline - time.monotonic()))
            holder["operation"] = {
                "deadline_monotonic": time.monotonic() + remaining,
                "epoch_budget": dict(EPOCHS),
                "samples": [{"rows": rows * size, "seed": seed}],
            }
            started = time.monotonic()
            try:
                matrix = namespace["sample"](
                    artifact, rows * size, seed, inventory, device="cuda:0",
                    runtime_path=request["runtime_ref"]["path"],
                    runtime_sha256=info["runtime_sha256"],
                    code_path=request["adapter_ref"]["path"],
                    code_sha256=info["code_sha256"],
                    admission=holder["grant"],
                )
                temporary = final.with_suffix(".csv.tmp")
                with temporary.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([f"f{i}" for i in range(matrix.shape[1] - 1)] + ["target"])
                    writer.writerows([[*row[1:], row[0]] for row in matrix])
                temporary.replace(final)
                status = "ok"
                reason = None
            except Exception as error:
                status = "failed"
                reason = type(error).__name__
            _ledger({
                "dataset": dataset, "seed": seed, "size": size, "status": status, "reason": reason,
                "seconds": round(time.monotonic() - started, 3), "rows": rows * size,
                "git_sha": git_sha, "official_tests_opened": False,
            })
    return 0


def sample_all(git_sha: str) -> int:
    _host_gate()
    check = OUTPUT / "parser-check.json"
    if not check.is_file() or not json.loads(check.read_text()).get("match"):
        raise SystemExit("refusing to sample before a matching parser check")
    for dataset, artifact in _artifacts():
        missing = [
            size for size in (1, 4) for seed in SEEDS
            if not (OUTPUT / dataset / f"sample-{seed}-{size}n.csv").exists()
        ]
        if not missing:
            continue
        while _memory_fraction() < 0.15:
            time.sleep(20)
        command = [
            sys_executable(), os.path.abspath(__file__),
            "--sample-lineage", dataset, "--artifact", str(artifact), "--git-sha", git_sha,
        ]
        process = subprocess.Popen(command, start_new_session=True)
        try:
            code = process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            _ledger({"dataset": dataset, "status": "timeout", "git_sha": git_sha, "scope": "lineage"})
            continue
        if code != 0:
            _ledger({"dataset": dataset, "status": "failed", "exit_code": code, "git_sha": git_sha})
    return 0


def score_samples(panel_workers_pilot: Path) -> int:
    import importlib.util
    import numpy as np
    _host_gate()
    spec = importlib.util.spec_from_file_location("pilot_metrics", panel_workers_pilot)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    for dataset, _artifact in _artifacts():
        projection = WORKERS / dataset / "projection.json"
        if not projection.is_file():
            _ledger({"dataset": dataset, "status": "unavailable", "reason": "worker_missing"})
            continue
        task = json.loads(projection.read_text())["task"]
        for size in (1, 4):
            for seed in SEEDS:
                csv_path = OUTPUT / dataset / f"sample-{seed}-{size}n.csv"
                metric = OUTPUT / dataset / f"sample-{seed}-{size}n.metric.json"
                if not csv_path.is_file() or metric.exists():
                    continue
                while _memory_fraction() < 0.15:
                    time.sleep(20)
                headerless = OUTPUT / dataset / f"sample-{seed}-{size}n.headerless.csv"
                try:
                    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
                    np.savetxt(headerless, raw, delimiter=",")
                    report = pilot.measure(
                        WORKERS / dataset / "train.csv",
                        WORKERS / dataset / "validation.csv",
                        headerless,
                        task,
                    )
                    payload = {
                        "dataset": dataset, "sample_seed": seed, "size": size, "status": "ok",
                        "synthetic_sha256": _sha(csv_path),
                        "null_loss": report["null_loss"],
                        "utility": report["utility"],
                        "rows": report["rows"],
                        "official_tests_opened": False,
                        "formal_dp": False,
                    }
                except (ValueError, RuntimeError, OSError) as error:
                    payload = {
                        "dataset": dataset, "sample_seed": seed, "size": size,
                        "status": "failed", "reason": type(error).__name__,
                        "official_tests_opened": False,
                    }
                temporary = metric.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
                temporary.replace(metric)
                print(dataset, size, seed, payload["status"], flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser-check", action="store_true")
    parser.add_argument("--sample-all", action="store_true")
    parser.add_argument("--sample-lineage", default=None)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--git-sha", default="unknown")
    args = parser.parse_args()
    if args.parser_check:
        raise SystemExit(parser_check(args.panel, args.pilot))
    if args.sample_lineage:
        raise SystemExit(sample_lineage(args.sample_lineage, args.artifact, args.git_sha))
    if args.sample_all:
        raise SystemExit(sample_all(args.git_sha))
    if args.score:
        raise SystemExit(score_samples(args.pilot))
    raise SystemExit("choose --parser-check, --sample-all, --sample-lineage, or --score")


if __name__ == "__main__":
    main()
