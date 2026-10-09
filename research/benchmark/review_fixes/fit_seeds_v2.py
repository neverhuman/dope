"""Refit DOPE at fit seeds 11, 23, 37, 53, and 71 with one pinned binary.

predeclare_v2.json fixes the binary, the evaluator, the seeds, and the two
arms. The headline arm is the historical features12_steps2048 command. The
product_default arm drops the research profile, so the trainer uses its
untuned 96-step default. Fits run seed-major, so a cutoff drops whole seeds.
Every metric record keeps the evaluator digest and library versions. The v1
runner (fit_seeds.py) stays byte-identical because 400 receipts pin its hash.
This process never opens a file named test.csv.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

from research.benchmark.review_fixes import fit_seeds as v1

HERE = Path(__file__).resolve().parent
PREDECLARE = HERE / "predeclare_v2.json"
ARMS = ("headline", "product_default")
CANDIDATE = v1.CANDIDATE
TIER = v1.TIER


def load_contract(path: Path = PREDECLARE) -> dict:
    """The fixed part of predeclare_v2 that this runner obeys."""
    contract = json.loads(path.read_text())
    if contract.get("format") != "dope-review-fix-predeclaration" or contract.get("version") != 2:
        raise SystemExit("predeclare_v2 has the wrong format")
    dope = contract["dope"]
    seeds = tuple(dope["fit_seeds"])
    if seeds != (11, 23, 37, 53, 71):
        raise SystemExit("predeclare_v2 fit seeds changed")
    if tuple(dope["sample_seeds"]) != v1.SAMPLE_SEEDS or tuple(dope["sizes"]) != v1.SIZES:
        raise SystemExit("predeclare_v2 sample grid changed")
    arms = dope["arms"]
    if set(arms) != set(ARMS):
        raise SystemExit("predeclare_v2 arms changed")
    if arms["headline"]["research_profile"] != "features12_steps2048":
        raise SystemExit("headline profile changed")
    if arms["product_default"]["research_profile"] is not None:
        raise SystemExit("product default must not set a research profile")
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fit_seeds": seeds,
        "arms": arms,
        "diagnostic_prefix": contract["binaries"]["diagnostic"]["sha256_prefix"],
        "binary_sha256": contract["binaries"]["headline"]["sha256"],
        "seed_flag": contract["binaries"]["headline"]["seed_flag"],
        "evaluator_sha256": contract["evaluator"]["utility_sha256"],
        "auditor_seed": contract["evaluator"]["auditor_seed"],
        "compile": dope["compile"],
    }


def compile_command(binary: Path, worker: Path, task: str, model: Path, seed: int, contract: dict) -> list[str]:
    spec = contract["compile"]
    return [
        str(binary), "compile",
        "--dataset-dir", str(worker),
        "--task", task,
        "--out", str(model),
        "--candidate", spec["candidate"],
        "--tier", spec["tier"],
        contract["seed_flag"], str(seed),
        "--neural-target-weight", spec["neural_target_weight"],
        "--neural-structural-penalty", spec["neural_structural_penalty"],
        "--deadline-seconds", spec["deadline_seconds"],
    ]


def child_env(torch_lib: str, arm: dict) -> dict[str, str]:
    env = v1._child_env(torch_lib)
    env.pop("DOPE_RESEARCH_TARGET_PROFILE", None)
    if arm["research_profile"] is not None:
        env["DOPE_RESEARCH_TARGET_PROFILE"] = arm["research_profile"]
    return env


def seed_major(datasets: list[tuple[str, str]], seeds: tuple[int, ...]) -> list[tuple[int, str, str]]:
    return [(seed, dataset, name) for seed in seeds for dataset, name in datasets]


def gpu_name() -> str | None:
    try:
        proc = subprocess.run(
            ["/usr/bin/nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return names[0] if proc.returncode == 0 and names else None


def runtime_drift(lock_path: Path) -> dict:
    """Compare the torch and CUDA libraries against the historical runtime lock."""
    lock = json.loads(lock_path.read_text())
    checked = mismatched = missing = 0
    for meta in lock.get("files", {}).values():
        resolved = Path(meta["resolved_path"])
        checked += 1
        if not resolved.is_file():
            missing += 1
        elif v1._sha256(resolved) != meta["sha256"]:
            mismatched += 1
    return {"lock_sha256": v1._sha256(lock_path), "checked": checked,
            "mismatched": mismatched, "missing": missing}


def _fit(cell: Path, worker: Path, dataset: str, seed: int, arm_name: str, binary: Path,
         env: dict, identity: dict, auth: dict, contract: dict) -> dict:
    model = cell / "model.dpk"
    record_path = cell / "fit.json"
    decision = v1.resume_fit(v1._read_json(record_path), model.is_file())
    if decision in {"skip_ok", "skip_failed"}:
        print(f"fit {decision} {arm_name} {dataset} {seed}", flush=True)
        return v1._read_json(record_path)
    cell.mkdir(parents=True, exist_ok=True)
    command = compile_command(binary, worker, auth["task"], model, seed, contract)
    # One GPU fit per host: every slot shares this lock under the output root.
    with (cell.parents[2] / ".gpu-fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        returncode, elapsed = v1._run_bounded(command, cell / "fit.log", env, v1.FIT_WALL_SECONDS)
    artifact_bytes, compliant = v1._parse_compile_log(cell / "fit.log")
    if returncode == 124:
        status, reason = "timeout", "wall"
    elif returncode == 0 and model.is_file():
        status, reason = "ok", None
    else:
        status, reason = "failed", "nonzero" if returncode != 0 else "model_missing"
    payload = {
        "status": status, "reason": reason, "dataset": dataset, "fit_seed": seed,
        "arm": arm_name, "research_profile": contract["arms"][arm_name]["research_profile"],
        "returncode": returncode, "elapsed_seconds": round(elapsed, 3),
        "artifact_bytes": artifact_bytes, "compliant": compliant,
        "model_sha256": v1._sha256(model) if model.is_file() else None,
        "projection_bytes": (worker / "projection.json").stat().st_size,
        **identity, **{key: auth[key] for key in ("train_sha256", "validation_sha256", "projection_sha256", "task")},
        "official_tests_opened": False,
    }
    v1._atomic_json(record_path, payload)
    print(f"fit {status} {arm_name} {dataset} {seed}", flush=True)
    return payload


def _score(pilot, cell: Path, worker: Path, dataset: str, seed: int, sample_seed: int, size: int,
           rows: int, arm_name: str, binary: Path, env: dict, identity: dict, auth: dict,
           contract: dict) -> None:
    csv_path = cell / f"sample-{sample_seed}-{size}n.csv"
    metric_path = cell / f"sample-{sample_seed}-{size}n.metric.json"
    v1._refuse_test(csv_path)
    base = {"dataset": dataset, "method": "DOPE", "arm": arm_name, "fit_seed": seed,
            "sample_seed": sample_seed, "size": size, "official_tests_opened": False}
    decision = v1.resume_sample(csv_path, rows, v1._read_json(metric_path))
    if decision in {"skip_ok", "skip_failed"}:
        return
    if decision == "sample":
        if csv_path.exists():
            csv_path.unlink()
        command = [str(binary), "sample", "--kernel", str(cell / "model.dpk"), "--rows", str(rows),
                   "--out", str(csv_path), "--seed", str(sample_seed)]
        returncode, elapsed = v1._run_bounded(
            command, cell / f"sample-{sample_seed}-{size}n.log", env, v1.SAMPLE_WALL_SECONDS)
        if returncode != 0 or not csv_path.is_file() or v1._line_count(csv_path) != rows:
            v1._atomic_json(metric_path, {**base, "status": "timeout" if returncode == 124 else "failed",
                                          "reason": "sample", "returncode": returncode,
                                          "elapsed_seconds": round(elapsed, 3)})
            print(f"sample failed {arm_name} {dataset} {seed} {sample_seed} {size}", flush=True)
            return
    try:
        report = pilot.measure(worker / "train.csv", worker / "validation.csv", csv_path, auth["task"],
                               seed=contract["auditor_seed"])
        if report.get("implementation_sha256") != contract["evaluator_sha256"]:
            raise RuntimeError("evaluator digest changed during the run")
        payload = {**base, "status": "ok", "reason": None, "report": report,
                   "synthetic_sha256": v1._sha256(csv_path), "model_sha256": v1._sha256(cell / "model.dpk"),
                   **identity, **{key: auth[key] for key in ("train_sha256", "validation_sha256", "projection_sha256")}}
    except (ValueError, RuntimeError, OSError) as error:
        payload = {**base, "status": "failed", "reason": type(error).__name__}
    v1._atomic_json(metric_path, payload)
    print(f"score {payload['status']} {arm_name} {dataset} {seed} {sample_seed} {size}", flush=True)


def run(args) -> None:
    if os.uname().nodename != args.host:
        raise SystemExit(f"refusing to run on {os.uname().nodename}")
    contract = load_contract()
    binary, workers, out = args.binary.resolve(), args.workers.resolve(), args.out.resolve()
    if out == workers or out.is_relative_to(workers) or workers.is_relative_to(out):
        raise SystemExit("output and workers must be separate trees")
    if "evaluator" in workers.parts:
        raise SystemExit("refusing the evaluator tree")
    v1._refuse_test(out)
    arm = contract["arms"][args.arm]
    if args.role == "diagnostic":
        # The 2x2 diagnostic refits seed 11 with B_new only; it never feeds a headline.
        if args.arm != "headline":
            raise SystemExit("the diagnostic role runs the headline profile only")
        contract = {**contract, "fit_seeds": (11,)}
    datasets = v1.partition([row[0] for row in v1._datasets(args.lineage_record)], args.slot, args.slots)
    names = dict(v1._datasets(args.lineage_record))
    plan = seed_major([(item, names[item]) for item in datasets], contract["fit_seeds"])
    if args.plan:
        print(f"host {args.host} arm {args.arm} fits {len(plan)} official_tests_opened false", flush=True)
        return
    digest = v1._sha256(binary)
    expected_ok = (digest.startswith(contract["diagnostic_prefix"]) if args.role == "diagnostic"
                   else digest == contract["binary_sha256"])
    if not expected_ok:
        raise SystemExit("binary digest does not match predeclare_v2")
    if v1._sha256(v1.PILOT_PATH) != contract["evaluator_sha256"]:
        raise SystemExit("evaluator digest does not match predeclare_v2")
    pilot = v1._load_pilot()
    identity = {"binary_sha256": digest, "binary_role": args.role, "source_git_sha": args.git_sha,
                "runner_sha256": v1._sha256(Path(__file__).resolve()), "predeclare_sha256": contract["sha256"],
                "host": args.host, "gpu": gpu_name(), "device": "cuda"}
    env = child_env(args.torch_lib, arm)
    out.mkdir(parents=True, exist_ok=True)
    campaign = out / f"campaign-{args.role}-{args.arm}-slot{args.slot}.json"
    v1._atomic_json(campaign, {
        "format": "dope-review-fix-fit-seeds-campaign-v2", "arm": args.arm, "slot": args.slot,
        "slots": args.slots, "fits_planned": len(plan), "fit_seeds": list(contract["fit_seeds"]),
        "runtime": runtime_drift(args.runtime_lock) if args.runtime_lock else None,
        **identity, "official_tests_opened": False, "medians_not_computed": True,
    })
    authenticated: dict[str, dict] = {}
    for seed, dataset, name in plan:
        worker = workers / dataset
        if dataset not in authenticated:
            try:
                authenticated[dataset] = v1._authenticate(worker, dataset)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                v1._atomic_json(out / dataset / "unavailable.json", {
                    "status": "unavailable", "reason": type(error).__name__, "dataset": dataset,
                    "official_tests_opened": False})
                authenticated[dataset] = {}
        auth = authenticated[dataset]
        if not auth:
            continue
        v1._wait_to_fit(list(args.yield_unit))
        cell = out / dataset / args.arm / f"fit-{seed}"
        record = _fit(cell, worker, dataset, seed, args.arm, binary, env, {**identity, "display_name": name},
                      auth, contract)
        if record.get("status") != "ok":
            continue
        for size in v1.SIZES:
            for sample_seed in v1.SAMPLE_SEEDS:
                _score(pilot, cell, worker, dataset, seed, sample_seed, size, auth["train_rows"] * size,
                       args.arm, binary, env, {**identity, "display_name": name}, auth, contract)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("--workers", "--lineage-record", "--out", "--binary", "--runtime-lock"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--host")
    parser.add_argument("--arm", choices=ARMS, default="headline")
    parser.add_argument("--role", choices=("headline", "diagnostic"), default="headline")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--git-sha", default="")
    parser.add_argument("--torch-lib", default="/usr/lib/python3/dist-packages/torch/lib")
    parser.add_argument("--yield-unit", action="append", default=[])
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    required = (args.workers, args.lineage_record, args.out, args.binary, args.host)
    if any(item in (None, "") for item in required) or len(args.git_sha) != 40:
        raise SystemExit("missing required arguments or a 40-character source commit")
    run(args)


if __name__ == "__main__":
    main()
