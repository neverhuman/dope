"""Fit DOPE seeds 23, 37, 53, and 71 on published train-only workers.

The compile invocation is the historical features12_steps2048 command.
Only the fit seed and the output path change. Seed 11 is never written.
A finished cell is skipped. A recorded failure stays recorded.
This process does not open a file named test.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT_PATH = HERE.parent / "pilot_metrics.py"
FIT_SEEDS = (23, 37, 53, 71)
SAMPLE_SEEDS = (101, 211, 307)
SIZES = (1, 4)
PROFILE = "features12_steps2048"
CANDIDATE = "compact_neural_residual_symbolic"
TIER = "l3"
NEURAL_TARGET_WEIGHT = "2.0"
NEURAL_STRUCTURAL_PENALTY = "0.0"
DEADLINE_SECONDS = "600"
FIT_WALL_SECONDS = 660
SAMPLE_WALL_SECONDS = 180


def _refuse_test(path: Path) -> None:
    if path.name == "test.csv" or "test.csv" in path.parts:
        raise ValueError("refusing a path named test.csv")


def _sha256(path: Path) -> str:
    _refuse_test(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    _refuse_test(path)
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count


def _atomic_json(path: Path, payload: dict) -> None:
    _refuse_test(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(staging, path)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def partition(ids: list[str], slot: int, slots: int) -> list[str]:
    if slots < 1 or slot < 0 or slot >= slots:
        raise ValueError("bad slot")
    return [item for index, item in enumerate(ids) if index % slots == slot]


def resume_fit(record: dict | None, model_exists: bool) -> str:
    """skip_ok, skip_failed, or run. A crash before fit.json returns run."""
    if record is None:
        return "run"
    status = record.get("status")
    if status == "ok" and model_exists:
        return "skip_ok"
    if status in {"failed", "timeout"}:
        return "skip_failed"
    return "run"


def resume_sample(csv_path: Path, expected_rows: int, metric: dict | None) -> str:
    if metric is not None and metric.get("status") == "ok":
        return "skip_ok"
    if metric is not None and metric.get("status") in {"failed", "timeout"}:
        return "skip_failed"
    if csv_path.is_file() and _line_count(csv_path) == expected_rows:
        return "score"
    return "sample"


def _memory_fraction() -> float:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        info[key] = int(rest.split()[0])
    return info["MemAvailable"] / info["MemTotal"]


def _unit_active(name: str) -> bool:
    proc = subprocess.run(
        ["/usr/bin/systemctl", "--user", "is-active", name],
        capture_output=True, text=True,
    )
    return proc.stdout.strip() == "active"


def _compute_pids() -> set[int] | None:
    """None means the GPU query failed. An empty set means no compute process."""
    try:
        proc = subprocess.run(
            ["/usr/bin/nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    pids = set()
    for line in proc.stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            pids.add(int(text))
    return pids


def _wait_to_fit(yield_units: list[str]) -> None:
    announced = False
    while True:
        reason = None
        if _memory_fraction() < 0.15:
            reason = "ram"
        elif any(_unit_active(name) for name in yield_units):
            reason = "yield"
        else:
            pids = _compute_pids()
            if pids is None:
                reason = "gpu-query"
            elif pids:
                reason = "gpu-busy"
        if reason is None:
            if announced:
                print("resume", flush=True)
            return
        if not announced:
            print(f"wait {reason}", flush=True)
            announced = True
        time.sleep(20)


def _child_env(torch_lib: str) -> dict[str, str]:
    env = {}
    for key in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR"):
        if key in os.environ:
            env[key] = os.environ[key]
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = torch_lib if not previous else torch_lib + ":" + previous
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "RAYON_NUM_THREADS": "16",
        "DOPE_RESEARCH_TARGET_PROFILE": PROFILE,
        "DOPE_RESEARCH_TARGET_DEVICE": "cuda",
        "LIBTORCH_USE_PYTORCH": "1",
    })
    return env


def _run_bounded(command: list[str], log_path: Path, env: dict[str, str], wall: int) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w") as stream:
        proc = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=wall)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = proc.wait()
            returncode = 124 if returncode == 0 else returncode
    return returncode, time.monotonic() - started


def _parse_compile_log(log_path: Path) -> tuple[int | None, bool | None]:
    artifact_bytes = None
    compliant = None
    if not log_path.is_file():
        return artifact_bytes, compliant
    for line in log_path.read_text(errors="replace").splitlines():
        text = line.strip()
        if not text.startswith("{") or "artifact_bytes" not in text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get("artifact_bytes"), int):
            artifact_bytes = payload["artifact_bytes"]
        if isinstance(payload.get("compliant"), bool):
            compliant = payload["compliant"]
    return artifact_bytes, compliant


def _load_pilot():
    spec = importlib.util.spec_from_file_location("pilot_metrics", PILOT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _datasets(record_path: Path) -> list[tuple[str, str]]:
    record = json.loads(record_path.read_text())
    rows = []
    for row in record["rows"]:
        dataset = row["dataset"]
        if not isinstance(dataset, str) or len(dataset) != 16 or any(ch not in "0123456789abcdef" for ch in dataset):
            raise ValueError("lineage id is not a 16-hex dataset id")
        name = row.get("display_name")
        if not isinstance(name, str) or not name:
            raise ValueError("lineage display name is missing")
        rows.append((dataset, name))
    rows.sort(key=lambda item: item[0])
    return rows


def _authenticate(worker: Path, dataset: str) -> dict:
    _refuse_test(worker)
    manifest_path = worker / "worker-manifest.json"
    projection_path = worker / "projection.json"
    train_path = worker / "train.csv"
    valid_path = worker / "validation.csv"
    for path in (manifest_path, projection_path, train_path, valid_path):
        _refuse_test(path)
        if not path.is_file():
            raise FileNotFoundError(path.name)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_id") != dataset:
        raise ValueError("manifest dataset id does not match the worker")
    hashes = {
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(valid_path),
        "projection_sha256": _sha256(projection_path),
    }
    projected = manifest.get("projected_hashes")
    if not isinstance(projected, dict):
        raise ValueError("manifest projected hashes are missing")
    if hashes["train_sha256"] != projected.get("train") or hashes["validation_sha256"] != projected.get("validation"):
        raise ValueError("worker csv digest mismatch")
    if hashes["projection_sha256"] != manifest.get("projection_sha256"):
        raise ValueError("projection digest mismatch")
    projection = json.loads(projection_path.read_text())
    task = projection.get("task")
    if task not in {"binary", "regression"}:
        raise ValueError("projection task is not binary or regression")
    hashes["task"] = task
    hashes["train_rows"] = _line_count(train_path)
    if hashes["train_rows"] < 1:
        raise ValueError("train csv has no rows")
    return hashes


def _fit_one(cell: Path, worker: Path, dataset: str, fit_seed: int, binary: Path,
             env: dict[str, str], identity: dict, authenticated: dict) -> dict:
    model = cell / "model.dpk"
    record_path = cell / "fit.json"
    decision = resume_fit(_read_json(record_path), model.is_file())
    if decision == "skip_ok":
        print(f"fit skip {dataset} {fit_seed}", flush=True)
        return _read_json(record_path)
    if decision == "skip_failed":
        print(f"fit kept {dataset} {fit_seed}", flush=True)
        return _read_json(record_path)
    cell.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary), "compile",
        "--dataset-dir", str(worker),
        "--task", authenticated["task"],
        "--out", str(model),
        "--candidate", CANDIDATE,
        "--tier", TIER,
        "--fit-seed", str(fit_seed),
        "--neural-target-weight", NEURAL_TARGET_WEIGHT,
        "--neural-structural-penalty", NEURAL_STRUCTURAL_PENALTY,
        "--deadline-seconds", DEADLINE_SECONDS,
    ]
    log_path = cell / "fit.log"
    returncode, elapsed = _run_bounded(command, log_path, env, FIT_WALL_SECONDS)
    artifact_bytes, compliant = _parse_compile_log(log_path)
    if returncode == 124:
        status = "timeout"
        reason = "wall"
    elif returncode == 0 and model.is_file():
        status = "ok"
        reason = None
    else:
        status = "failed"
        reason = "nonzero" if returncode != 0 else "model_missing"
    payload = {
        "status": status,
        "reason": reason,
        "dataset": dataset,
        "display_name": identity["display_name"],
        "fit_seed": fit_seed,
        "profile": PROFILE,
        "device": "cuda",
        "candidate": CANDIDATE,
        "tier": TIER,
        "neural_target_weight": float(NEURAL_TARGET_WEIGHT),
        "neural_structural_penalty": float(NEURAL_STRUCTURAL_PENALTY),
        "deadline_seconds": int(DEADLINE_SECONDS),
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 3),
        "artifact_bytes": artifact_bytes,
        "compliant": compliant,
        "model_sha256": _sha256(model) if model.is_file() else None,
        "binary_sha256": identity["binary_sha256"],
        "source_git_sha": identity["source_git_sha"],
        "runner_sha256": identity["runner_sha256"],
        "train_sha256": authenticated["train_sha256"],
        "validation_sha256": authenticated["validation_sha256"],
        "projection_sha256": authenticated["projection_sha256"],
        "task": authenticated["task"],
        "official_tests_opened": False,
    }
    _atomic_json(record_path, payload)
    print(f"fit {status} {dataset} {fit_seed}", flush=True)
    return payload


def _score_one(pilot, cell: Path, worker: Path, dataset: str, fit_seed: int, sample_seed: int,
               size: int, rows: int, binary: Path, env: dict[str, str], identity: dict,
               authenticated: dict) -> None:
    csv_path = cell / f"sample-{sample_seed}-{size}n.csv"
    metric_path = cell / f"sample-{sample_seed}-{size}n.metric.json"
    _refuse_test(csv_path)
    decision = resume_sample(csv_path, rows, _read_json(metric_path))
    if decision == "skip_ok":
        print(f"score skip {dataset} {fit_seed} {sample_seed} {size}", flush=True)
        return
    if decision == "skip_failed":
        print(f"score kept {dataset} {fit_seed} {sample_seed} {size}", flush=True)
        return
    if decision == "sample":
        if csv_path.exists():
            csv_path.unlink()
        command = [
            str(binary), "sample",
            "--kernel", str(cell / "model.dpk"),
            "--rows", str(rows),
            "--out", str(csv_path),
            "--seed", str(sample_seed),
        ]
        returncode, elapsed = _run_bounded(command, cell / f"sample-{sample_seed}-{size}n.log", env, SAMPLE_WALL_SECONDS)
        if returncode != 0 or not csv_path.is_file() or _line_count(csv_path) != rows:
            _atomic_json(metric_path, {
                "status": "failed" if returncode != 124 else "timeout",
                "reason": "sample",
                "dataset": dataset,
                "method": "DOPE",
                "configuration": PROFILE,
                "fit_seed": fit_seed,
                "sample_seed": sample_seed,
                "size": size,
                "returncode": returncode,
                "elapsed_seconds": round(elapsed, 3),
                "official_tests_opened": False,
            })
            print(f"sample failed {dataset} {fit_seed} {sample_seed} {size}", flush=True)
            return
    if _memory_fraction() < 0.15:
        print("wait ram", flush=True)
        while _memory_fraction() < 0.15:
            time.sleep(20)
        print("resume", flush=True)
    train_path = worker / "train.csv"
    valid_path = worker / "validation.csv"
    try:
        report = pilot.measure(train_path, valid_path, csv_path, authenticated["task"], seed=1729)
        payload = {
            "status": "ok",
            "reason": None,
            "dataset": dataset,
            "display_name": identity["display_name"],
            "method": "DOPE",
            "configuration": PROFILE,
            "fit_seed": fit_seed,
            "sample_seed": sample_seed,
            "size": size,
            "synthetic_sha256": _sha256(csv_path),
            "model_sha256": _sha256(cell / "model.dpk"),
            "binary_sha256": identity["binary_sha256"],
            "source_git_sha": identity["source_git_sha"],
            "runner_sha256": identity["runner_sha256"],
            "train_sha256": authenticated["train_sha256"],
            "validation_sha256": authenticated["validation_sha256"],
            "projection_sha256": authenticated["projection_sha256"],
            "task": authenticated["task"],
            "metrics": {
                "null_loss": report["null_loss"],
                "utility": report["utility"],
                "rows": report["rows"],
            },
            "official_tests_opened": False,
        }
        status = "ok"
    except (ValueError, RuntimeError, OSError) as error:
        payload = {
            "status": "failed",
            "reason": type(error).__name__,
            "dataset": dataset,
            "method": "DOPE",
            "configuration": PROFILE,
            "fit_seed": fit_seed,
            "sample_seed": sample_seed,
            "size": size,
            "official_tests_opened": False,
        }
        status = "failed"
    _atomic_json(metric_path, payload)
    print(f"score {status} {dataset} {fit_seed} {sample_seed} {size}", flush=True)


def _self_check() -> None:
    ids = [f"{index:016x}" for index in range(5)]
    if partition(ids, 0, 2) != [ids[0], ids[2], ids[4]]:
        raise SystemExit("even partition failed")
    if partition(ids, 1, 2) != [ids[1], ids[3]]:
        raise SystemExit("odd partition failed")
    if partition(ids, 0, 1) != ids:
        raise SystemExit("full partition failed")
    if resume_fit(None, False) != "run":
        raise SystemExit("missing fit record must run")
    if resume_fit({"status": "ok"}, True) != "skip_ok":
        raise SystemExit("ok fit must skip")
    if resume_fit({"status": "ok"}, False) != "run":
        raise SystemExit("ok fit without a model must run")
    if resume_fit({"status": "failed"}, True) != "skip_failed":
        raise SystemExit("failed fit must stay failed")
    if resume_fit({"status": "timeout"}, False) != "skip_failed":
        raise SystemExit("timeout fit must stay failed")
    try:
        _refuse_test(Path("/tmp/test.csv"))
    except ValueError:
        pass
    else:
        raise SystemExit("test.csv was accepted")
    if 11 in FIT_SEEDS or tuple(FIT_SEEDS) != (23, 37, 53, 71):
        raise SystemExit("fit seed list changed")
    print("fit-seeds self-check ok", flush=True)


def _run(args) -> None:
    if os.uname().nodename != args.host:
        raise SystemExit(f"refusing to run on {os.uname().nodename}")
    if tuple(args.fit_seed) != FIT_SEEDS:
        raise SystemExit("fit seeds are fixed at 23 37 53 71")
    binary = args.binary.resolve()
    workers = args.workers.resolve()
    out = args.out.resolve()
    if out == workers or out.is_relative_to(workers) or workers.is_relative_to(out):
        raise SystemExit("output and workers must be separate trees")
    _refuse_test(out)
    selected = partition(_datasets(args.lineage_record), args.slot, args.slots)
    if args.plan:
        print(f"host {args.host}", flush=True)
        print(f"selected {len(selected)}", flush=True)
        print(f"slot {args.slot} slots {args.slots}", flush=True)
        print("official_tests_opened false", flush=True)
        return
    if not binary.is_file():
        raise SystemExit("release binary is missing")
    pilot = _load_pilot()
    identity = {
        "binary_sha256": _sha256(binary),
        "source_git_sha": args.git_sha,
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    env = _child_env(args.torch_lib)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "campaign.json", {
        "format": "dope-review-fix-fit-seeds-campaign-v1",
        "host": args.host,
        "slot": args.slot,
        "slots": args.slots,
        "selected": len(selected),
        "fit_seeds": list(FIT_SEEDS),
        "sample_seeds": list(SAMPLE_SEEDS),
        "sizes": list(SIZES),
        "profile": PROFILE,
        "yield_units": list(args.yield_unit),
        "source_git_sha": args.git_sha,
        "runner_sha256": identity["runner_sha256"],
        "binary_sha256": identity["binary_sha256"],
        "official_tests_opened": False,
        "medians_not_computed": True,
    })
    print(f"selected {len(selected)}", flush=True)
    for index, (dataset, display_name) in enumerate(selected, start=1):
        identity["display_name"] = display_name
        worker = workers / dataset
        cell_root = out / dataset
        print(f"dataset {dataset} {index} {len(selected)}", flush=True)
        try:
            authenticated = _authenticate(worker, dataset)
        except (OSError, ValueError, FileNotFoundError, json.JSONDecodeError) as error:
            _atomic_json(cell_root / "unavailable.json", {
                "status": "unavailable",
                "reason": type(error).__name__,
                "dataset": dataset,
                "official_tests_opened": False,
            })
            print(f"unavailable {dataset}", flush=True)
            continue
        for fit_seed in FIT_SEEDS:
            _wait_to_fit(list(args.yield_unit))
            cell = cell_root / f"fit-{fit_seed}"
            if cell.name != f"fit-{fit_seed}":
                raise SystemExit("refusing to write outside the seed cell")
            record = _fit_one(cell, worker, dataset, fit_seed, binary, env, identity, authenticated)
            if record.get("status") != "ok":
                continue
            for size in SIZES:
                rows = authenticated["train_rows"] * size
                for sample_seed in SAMPLE_SEEDS:
                    _score_one(
                        pilot, cell, worker, dataset, fit_seed, sample_seed, size, rows,
                        binary, env, identity, authenticated,
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=Path)
    parser.add_argument("--lineage-record", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--git-sha", default="")
    parser.add_argument("--torch-lib", default="/usr/lib/python3/dist-packages/torch/lib")
    parser.add_argument("--yield-unit", action="append", default=[])
    parser.add_argument("--fit-seed", nargs=4, type=int, default=list(FIT_SEEDS))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    required = (args.workers, args.lineage_record, args.out, args.binary, args.host, args.git_sha)
    if any(item in (None, "") for item in required):
        raise SystemExit("missing required arguments")
    if len(args.git_sha) != 40:
        raise SystemExit("git sha must be the 40-character source commit")
    _run(args)


if __name__ == "__main__":
    main()
