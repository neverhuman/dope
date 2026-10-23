"""Bounded native workers. Source values and native stderr never enter status files."""
from __future__ import annotations

import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .plan import canonical, digest, file_sha, preparation_recipe, require, validate_plan


def write_private(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp")
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def canonical_train(source, destination, binding):
    """Resolve the explicitly declared target; never guess it from cell values."""
    csv.field_size_limit(16_000_000)
    count = binding["columns"]
    declaration = binding["target"]
    require(type(declaration.get("index")) is int, "target_declaration_missing")
    index = declaration["index"]
    if index < 0:
        index += count
    require(0 <= index < count, "target_index_invalid")
    order = [j for j in range(count) if j != index] + [index]
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    rows = 0
    with os.fdopen(descriptor, "w", newline="") as dst, Path(source).open(newline="", encoding="utf-8-sig") as src:
        reader, writer = csv.reader(src), csv.writer(dst, lineterminator="\n")
        if binding["has_header"]:
            headers = next(reader)
            require(len(headers) == count and len(set(headers)) == count
                    and headers[index] == declaration["name"], "source_header_declaration_mismatch")
        writer.writerow([f"feature_{j:04d}" for j in range(count - 1)] + ["target"])
        for row in reader:
            require(len(row) == count, "source_row_width_mismatch")
            writer.writerow([row[j] for j in order])
            rows += 1
        dst.flush()
        os.fsync(dst.fileno())
    require(rows == binding["rows"], "source_row_count_mismatch")
    return dict(format="dope-landmark-target-last-v1", train_sha256=binding["train_sha256"],
                canonical_sha256=file_sha(destination), rows=rows, source_target_index=index,
                source_column_permutation=order, target_index=count - 1,
                physical_row_order_preserved=True, official_test_access=False)


def validate_report(report, binding):
    require(report["panel"] == binding["options"]["panel"]
            and report["options"] == binding["options"]
            and report["kind"] == "real" and report["privacy"] == "private"
            and report["task"] == binding["task"], "native_report_identity_mismatch")
    require(len(report["cells"]) == 23 and len(report["contrasts"]) == 18,
            "native_report_layout_mismatch")
    require([c["id"] for c in report["cells"]] == list(range(1, 24)), "native_report_slot_mismatch")
    require(report["rows"] == binding["rows"], "native_report_row_mismatch")
    canonical(report)  # Reject nonfinite JSON, including permissive parser NaNs.


def wait_native(process, timeout, memory_bytes, *, watch_rss=False):
    deadline, peak = time.monotonic() + timeout, None

    def stop(reason):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return None, peak, reason

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return stop("native_wall_cap")
        try:
            code = process.wait(timeout=min(0.25, remaining) if watch_rss else remaining)
            return code, peak, None
        except subprocess.TimeoutExpired:
            if not watch_rss:
                return stop("native_wall_cap")
        try:
            observed = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(process.pid)],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True, timeout=2, check=False)
            if process.poll() is not None:
                return process.returncode, peak, None
            require(observed.returncode == 0 and observed.stdout.strip().isdigit(),
                    "memory_observation_unavailable")
            rss = int(observed.stdout.strip()) * 1024
            peak = max(peak or 0, rss)
            if rss > memory_bytes:
                return stop("native_rss_cap")
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return stop("memory_observation_unavailable")


def fetch_train(client, cell, cache):
    binding = cell["binding"]
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = cache / (binding["train_sha256"] + ".csv")
    # Distinct catalog records can share a content-addressed TRAIN blob.
    with (cache / (binding["train_sha256"] + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            require(destination.stat().st_size == binding["train_bytes"]
                    and file_sha(destination) == binding["train_sha256"], "cached_TRAIN_corrupt")
            return destination
        response = client.get_object(Bucket=binding["source_bucket"], Key=binding["source_key"])
        stream = response["Body"]
        tmp = destination.with_suffix(".partial")
        require(not tmp.exists(), "interrupted_source_download_requires_repair")
        checksum, size = hashlib.sha256(), 0
        try:
            require(response["ContentLength"] == binding["train_bytes"], "source_transport_size_mismatch")
            descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as dst:
                while block := stream.read(1024 * 1024):
                    size += len(block)
                    require(size <= binding["train_bytes"], "source_transport_overflow")
                    checksum.update(block)
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
            require(size == binding["train_bytes"] and checksum.hexdigest() == binding["train_sha256"],
                    "source_TRAIN_hash_mismatch")
            os.rename(tmp, destination)
        finally:
            stream.close()
        return destination


def run_cell(cell, *, binary, root, client, store, threads, timeout, memory_bytes, repair_reason=None):
    directory = root / "cells" / cell["key"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = sorted(directory.glob("attempt-*/receipt.json"))
        if prior:
            receipt = json.loads(prior[-1].read_bytes())
            if receipt["state"] in ("complete", "cached", "unsupported"):
                if receipt["state"] in ("complete", "cached"):
                    body = store.lookup(cell)
                    require(body is not None, "completed_cell_cache_missing")
                    validate_report(json.loads(body), cell["binding"])
                return receipt
            if receipt["state"] == "running" and receipt.get("native_pid"):
                require(receipt.get("host") == os.uname().nodename, "remote_interruption_requires_reconciliation")
                try:
                    os.kill(receipt["native_pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError("prior_native_process_may_still_be_alive")
            require(bool(repair_reason), "terminal_or_interrupted_cell_requires_repair_reason")
        attempts = list(directory.glob("attempt-*"))
        require(not attempts or prior or bool(repair_reason), "interrupted_cell_requires_repair_reason")
        attempt = directory / f"attempt-{len(attempts):04d}"
        attempt.mkdir(mode=0o700)
        start = time.monotonic()
        receipt = dict(key=cell["key"], dataset_id=cell["binding"]["dataset_id"],
                       binding_sha256=digest(cell["binding"]), state="running", host=os.uname().nodename,
                       attempt=attempt.name, started_unix=time.time(), repair_reason=repair_reason,
                       threads=threads, wall_cap_seconds=timeout, memory_cap_bytes=memory_bytes,
                       memory_policy="sampled_rss_250ms" if sys.platform == "darwin" else "rlimit_as",
                       official_test_access=False, model_training=False)
        write_private(attempt / "receipt.json", receipt)
        try:
            if cell["initial_state"] == "unsupported":
                receipt.update(state="unsupported", code=cell["reason"])
            else:
                cached = store.lookup(cell)
                if cached is not None:
                    validate_report(json.loads(cached), cell["binding"])
                    receipt.update(state="cached", report_sha256=hashlib.sha256(cached).hexdigest())
                else:
                    source = fetch_train(client, cell, root / "train-cache")
                    canonical_path = attempt / "train.csv"
                    normalization = canonical_train(source, canonical_path, cell["binding"])
                    write_private(attempt / "normalization.private.json", normalization)
                    output = attempt / "signature.private.json"
                    command = [str(binary), "landmarks", "real", str(canonical_path),
                               "--target", "target", "--task", cell["binding"]["task"], "--json", "-o", str(output)]
                    for key, value in cell["binding"]["options"].items():
                        command += ["--" + key.replace("_", "-"), str(value)]
                    environment = dict(PATH="/usr/bin:/bin", LANG="C.UTF-8", RAYON_NUM_THREADS=str(threads),
                                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                    with (attempt / "native.private.log").open("xb") as log:
                        os.chmod(log.name, 0o600)
                        wrapper = [sys.executable, str(Path(__file__).with_name("exec_native.py")),
                                   str(memory_bytes), str(timeout * threads + 30)]
                        process = subprocess.Popen(wrapper + command, stdin=subprocess.DEVNULL, stdout=log,
                                                   stderr=log, env=environment, start_new_session=True)
                        receipt["native_pid"] = process.pid
                        write_private(attempt / "receipt.json", receipt)
                        code, peak, disposition = wait_native(process, timeout, memory_bytes,
                                                              watch_rss=sys.platform == "darwin")
                        receipt["sampled_native_rss_peak_bytes"] = peak
                        if disposition:
                            receipt.update(state="overbudget" if disposition != "memory_observation_unavailable"
                                           else "failed", code=disposition)
                    if code == 0:
                        report = json.loads(output.read_bytes())
                        validate_report(report, cell["binding"])
                        manifest = store.publish(cell, report)
                        receipt.update(state="complete", report_sha256=manifest["report_sha256"],
                                       normalization_sha256=digest(normalization),
                                       cell_states={state: sum(c["state"] == state for c in report["cells"])
                                                    for state in ("complete", "uninformative", "unsupported", "over_budget", "failed")})
                    elif code is not None:
                        receipt.update(state="failed", code="native_exit", native_exit=code)
                    canonical_path.unlink(missing_ok=True)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            # Static codes only. Native/transport messages can include private paths or values.
            allowed = {"source_row_count_mismatch", "source_row_width_mismatch", "target_index_invalid",
                       "source_header_declaration_mismatch", "source_TRAIN_hash_mismatch", "cached_TRAIN_corrupt"}
            receipt.update(state="failed", code=str(error) if str(error) in allowed else "worker_exception",
                           error_type=type(error).__name__)
        receipt.update(elapsed_seconds=round(time.monotonic() - start, 6), finished_unix=time.time())
        write_private(attempt / "receipt.json", receipt)
        return receipt


def run(plan, *, binary, root, client, store, workers=4, threads=2, timeout=600,
        memory_bytes=8 * 2**30, shard=0, shards=1, wall_seconds=43_200, repair_reason=None):
    validate_plan(plan)
    binary, root = Path(binary).resolve(), Path(root).resolve()
    engine_sha = file_sha(binary)
    preparation_sha = digest(preparation_recipe())
    require(bool(plan["cells"]) and all(c["binding"]["engine_sha256"] == engine_sha
                                      for c in plan["cells"]), "engine_digest_mismatch")
    require(all(c["binding"]["preparation_recipe_sha256"] == preparation_sha
                for c in plan["cells"]), "preparation_recipe_mismatch")
    require(1 <= workers <= 32 and 1 <= threads <= 16 and 0 <= shard < shards <= 128
            and 1 <= timeout <= 3600 and 1 <= wall_seconds <= 86_400
            and 256 * 2**20 <= memory_bytes <= 512 * 2**30, "invalid_resource_allocation")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_private(root / "plan.private.json", plan)
        assigned = [c for i, c in enumerate(plan["cells"]) if i % shards == shard]
        started, receipts = time.monotonic(), []
        status = dict(format="dope-landmark-cache-progress-v1", plan_sha256=plan["plan_sha256"],
                      host=os.uname().nodename, shard=shard, shards=shards, scheduled=len(assigned),
                      total_catalog=len(plan["cells"]), started_unix=time.time(), workers=workers,
                      threads_per_worker=threads, wall_cap_seconds=wall_seconds,
                      official_test_access=False, model_training=False)

        def record():
            counts = {state: sum(r["state"] == state for r in receipts)
                      for state in ("complete", "cached", "unsupported", "failed", "overbudget", "interrupted")}
            update = dict(status, updated_unix=time.time(), counts=counts, terminal=len(receipts),
                          pending=len(assigned) - len(receipts), cells=receipts,
                          elapsed_seconds=round(time.monotonic() - started, 3))
            write_private(root / "progress.json", update)
            return update

        record()
        # Keep only the admitted worker count in flight, replenishing completed slots.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            remaining, active = iter(assigned), {}

            def submit():
                if time.monotonic() - started + timeout + 240 > wall_seconds:
                    return
                cell = next(remaining, None)
                if cell is not None:
                    future = pool.submit(run_cell, cell, binary=binary, root=root, client=client,
                                         store=store, threads=threads, timeout=timeout,
                                         memory_bytes=memory_bytes, repair_reason=repair_reason)
                    active[future] = cell

            for _ in range(workers):
                submit()
            while active:
                done, _ = concurrent.futures.wait(active, timeout=15, return_when=concurrent.futures.FIRST_COMPLETED)
                if not done:
                    record()
                for future in done:
                    cell = active.pop(future)
                    try:
                        receipts.append(future.result())
                    except (OSError, ValueError, subprocess.SubprocessError):
                        receipts.append(dict(key=cell["key"], state="interrupted",
                                             code="prior_attempt_or_worker_requires_reconciliation"))
                    record()
                    submit()
        result = record()
        result["state"] = "terminal" if result["pending"] == 0 else "budget_exhausted"
        write_private(root / "progress.json", result)
        store.put_once("runs/" + plan["plan_sha256"] + "/" + root.name + "/manifest.json", canonical(result))
        return result
