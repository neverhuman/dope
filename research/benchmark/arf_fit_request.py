"""Frozen ARF fit request boundary; the outer dispatcher owns admission."""
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import time


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    require(type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "invalid frozen digest")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "duplicate request key")
        result[key] = value
    return result


def finite(value):
    number = float(value)
    require(math.isfinite(number), "nonfinite request number")
    return number


def decode(data):
    return json.loads(data, object_pairs_hook=pairs, parse_float=finite,
        parse_constant=lambda _: require(False, "nonfinite request number"))


def checked_file(path, expected, base):
    digest(expected)
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(strict=True)
            and path.is_relative_to(base) and stat.S_ISREG(path.lstat().st_mode),
            "unowned worker file")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "worker file drift")
    return data


def request_job(request, lock, expected_round):
    digest(expected_round)
    require(set(request) == {"job", "round_sha256", "deadline_epoch"}
            and request["round_sha256"] == expected_round, "request round mismatch")
    epoch = request["deadline_epoch"]
    require(type(epoch) in (float, int) and math.isfinite(epoch)
            and time.time() < epoch <= time.time() + 600,
            "request fit deadline invalid")
    round_deadline = datetime.fromisoformat(lock["deadline_utc"])
    require(round_deadline.tzinfo is not None, "round deadline timezone required")
    require(epoch <= round_deadline.timestamp(),
            "request exceeds round deadline")
    job_id = identity(request["job"])
    matches = [r for r in lock["jobs"] if r["job_sha256"] == job_id]
    require(len(matches) == 1 and identity(matches[0]["job"]) == job_id,
            "request not in frozen matrix")
    require(matches[0]["adapter_applicability"] == "supported_width", "ARF adapter width unavailable")
    job = matches[0]["job"]
    require(job["method"] == "ARF" and job["track"] == "common_numeric"
            and job["dp_budget"] is None and type(job["fit_seed"]) is int
            and 0 <= job["fit_seed"] < 2**32
            and identity(job["configuration"]) == job["configuration_sha256"],
            "invalid ARF fit identity")
    return job, job_id


def run(request_path, request_sha256, round_path, round_sha256, base, output):
    # Before source-provider import or dependency initialization.
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU environment required")
    lock = decode(checked_file(round_path, round_sha256, base))
    require(lock["format"] == "dope-arf-S3-native-research-round"
            and type(lock["version"]) is int and lock["version"] == 1 and lock["frozen"] is True
            and lock["official_tests_opened"] is False,
            "unfrozen ARF research round")
    request = decode(checked_file(request_path, request_sha256, base))
    job, job_id = request_job(request, lock, round_sha256)
    require(set(os.sched_getaffinity(0)) == set(lock["fit_cpu_slot"]),
            "worker CPU reservation differs")
    require(checked_file(__file__, lock["operation_source_files"][__file__], base)
            is not None, "executing worker source differs")
    source_root = Path(lock["adapter_source_root"])
    actual = set()
    for path in source_root.rglob("*"):
        require(stat.S_ISDIR(path.lstat().st_mode) or stat.S_ISREG(path.lstat().st_mode),
                "adapter source alias")
        if path.is_file(): actual.add(str(path))
    require(actual == set(lock["adapter_source_files"]), "adapter source inventory differs")
    for path, expected in lock["adapter_source_files"].items():
        checked_file(path, expected, base)
    require(job["source_hashes"] == {name: lock["adapter_source_files"][str(source_root/path)]
        for name,path in {"adapter":"research/benchmark/arf_adapter.py",
            "guard":"research/benchmark/arf_runtime_guard.py",
            "native":"research/benchmark/arf_native.py"}.items()}, "fit source identity differs")
    guard_path = source_root / "research/benchmark/arf_runtime_guard.py"
    spec = importlib.util.spec_from_file_location("arf_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.verify(lock["runtime_path"], job["runtime_sha256"], base)
    worker = Path(job["worker"]["path"])
    require(set(job["worker"]["files"]) == {"train.csv", "validation.csv",
        "projection.json", "worker-manifest.json", "row-group-assignments.json"}
        and set(p.name for p in worker.iterdir()) == set(job["worker"]["files"]),
        "worker partition inventory differs")
    for name, expected in job["worker"]["files"].items():
        checked_file(worker/name, expected, base)
    output = Path(output)
    require(output == Path(request_path).parent and output == output.resolve(strict=True)
            and output.parent == Path(round_path).parent/"attempts"/job_id
            and len(output.name)==12 and output.name.startswith("attempt-")
            and all(c in "0123456789" for c in output.name[8:]) and int(output.name[8:])>=1
            and not (output/"fit.json").exists(), "fit output not reserved")
    from research.benchmark import arf_adapter as adapter
    result = adapter.fit(runtime_path=lock["runtime_path"], runtime_sha256=job["runtime_sha256"],
        base=base, train=worker/"train.csv", train_sha256=job["worker"]["files"]["train.csv"],
        validation=worker/"validation.csv", validation_sha256=job["worker"]["files"]["validation.csv"],
        projection=worker/"projection.json", projection_sha256=job["worker"]["files"]["projection.json"],
        artifact=output/"artifact", configuration=job["configuration"],
        fit_seed=job["fit_seed"], deadline_epoch=request["deadline_epoch"])
    require(time.time() < request["deadline_epoch"], "fit deadline exceeded")
    for name, expected in job["worker"]["files"].items():
        checked_file(worker/name, expected, base)
    checked_file(__file__, lock["operation_source_files"][__file__], base)
    for path, expected in lock["adapter_source_files"].items():
        checked_file(path, expected, base)
    checked_file(round_path, round_sha256, base)
    checked_file(request_path, request_sha256, base)
    require(time.time() < request["deadline_epoch"], "final fit deadline exceeded")
    receipt = dict(job_sha256=job_id, round_sha256=round_sha256, request_sha256=request_sha256,
        status="ok", result=result, official_tests_opened=False, mfs_v2=None,
        ptf_v1=None, release_safe=None, superiority=None, requires_gpu=False)
    with (output/"fit.json").open("x") as stream:
        stream.write(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return receipt
