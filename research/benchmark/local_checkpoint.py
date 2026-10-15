"""Optional public-development hook around the existing offline learner ports.

Expected pins come from an independently supplied custodian decision. Registry
metadata and local byte checks confer no scientific or release admission. This
trusted CPU child bounds time/address space; the custodian still owns immutable
snapshot and filesystem/network isolation. No checkpoint acquisition occurs here.
"""
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing
import os
import re
import resource
import signal
import sys
import time

from research.transfer.adapters.base import (
    AdapterAdmission, FeatureBatch, LearnerAdapter, PredictionResult, TrainingBatch,
)
from research.transfer.adapters.auditors.conventional import ObservedLearner, OfflineRuntime


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pin(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def worker(pipe, adapter, training, query, seed, wall, memory):
    """Only this owned child is terminated on timeout; streams stay private."""
    try:
        os.setsid()
        descriptor = os.open(os.devnull, os.O_WRONLY)
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
        if descriptor > 2:
            os.close(descriptor)
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu = max(1, math.ceil(wall))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        adapter.prepare(training, seed)
        result = adapter.predict(query, seed)
        require(type(result) is PredictionResult and result.kind == "point", "public_checkpoint_output_invalid")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        message = {"state": "completed", "row_ids": result.row_ids, "values": result.values,
                   "consumed_rows": result.consumed_rows,
                   "memory_bytes": max(peak, result.memory_bytes), "gpu_bytes": result.gpu_bytes}
        encoded = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
        require(len(encoded) <= 1024 * 1024, "public_checkpoint_output_cap_exceeded")
        pipe.send_bytes(encoded)
    except BaseException:
        # Never forward backend exception text, headers, paths or source values.
        pipe.send_bytes(b'{"state":"failed"}')
    finally:
        pipe.close()


def stop_owned_child(child):
    if child.is_alive():
        try:
            if os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except ProcessLookupError:
            pass
    child.join(timeout=1)


@dataclass(frozen=True, repr=False)
class LocalCheckpointHook:
    adapter: LearnerAdapter
    admission: AdapterAdmission
    expected_identity_sha256: str
    expected_eligibility_sha256: str
    expected_native_reference_sha256: str
    fixture_only: bool = False
    _spent_seconds: float = field(default=0.0, init=False, compare=False)

    def __post_init__(self):
        require(isinstance(self.adapter, LearnerAdapter) and type(self.admission) is AdapterAdmission,
                "public_checkpoint_interface_invalid")
        require(type(self.fixture_only) is bool, "public_checkpoint_scope_invalid")
        for value in (self.expected_identity_sha256, self.expected_eligibility_sha256,
                      self.expected_native_reference_sha256):
            require(pin(value), "public_checkpoint_expected_pin_invalid")
        identity = self.adapter.identity
        require(identity.sha256 == self.expected_identity_sha256 == self.admission.identity_sha256 and
                self.admission.eligibility_sha256 == self.expected_eligibility_sha256 and
                self.admission.native_reference_sha256 == self.expected_native_reference_sha256 and
                self.admission.kind == "learner", "public_checkpoint_pin_mismatch")
        require(identity.mode == "frozen_context" and identity.weights_sha256 is not None,
                "public_checkpoint_frozen_local_weights_required")
        require(0 < identity.resources.wall_seconds <= 180 and
                0 < identity.resources.memory_bytes <= 4 * 1024**3 and identity.resources.gpu_bytes == 0,
                "public_checkpoint_cpu_budget_invalid")
        if not self.fixture_only:
            require(isinstance(self.adapter, ObservedLearner) and
                    type(self.adapter._runtime) is OfflineRuntime and
                    self.adapter._runtime.factory is None and self.adapter._runtime.version_reader is version and
                    self.adapter.recipe.task == "regression" and self.adapter.recipe.device == "cpu",
                    "public_checkpoint_offline_native_port_required")

    def predict(self, fit_x, fit_y, query_x, seed, arm, schema_sha256):
        started = time.monotonic()
        result = None
        remaining = self.adapter.identity.resources.wall_seconds - self._spent_seconds
        require(remaining > 0, "public_checkpoint_wall_budget_exhausted")
        try:
            result = self._predict(fit_x, fit_y, query_x, seed, arm, schema_sha256, started, remaining)
            return result
        except (TypeError, OverflowError, IndexError, AttributeError):
            raise ValueError("public_checkpoint_input_invalid") from None
        finally:
            elapsed = time.monotonic() - started
            object.__setattr__(self, "_spent_seconds", self._spent_seconds + elapsed)
            if result is not None:
                result[1].update(elapsed_seconds=elapsed, cumulative_hook_seconds=self._spent_seconds,
                                 declared_total_wall_seconds=self.adapter.identity.resources.wall_seconds)
                require(self._spent_seconds <= self.adapter.identity.resources.wall_seconds,
                        "public_checkpoint_wall_budget_exhausted")

    def _predict(self, fit_x, fit_y, query_x, seed, arm, schema_sha256, started, remaining):
        """No query-label argument. Actual consumed rows remain adapter-owned."""
        require(sys.platform == "linux", "public_checkpoint_cpu_worker_unavailable")
        require(type(seed) is int and 0 <= seed <= 2**53 - 1 and
                arm in {"real", "synthetic_1n", "synthetic_4n"} and pin(schema_sha256),
                "public_checkpoint_request_invalid")
        identity = self.adapter.identity
        require(identity.sha256 == self.expected_identity_sha256, "public_checkpoint_pin_mismatch")
        count, query_count = len(fit_x), len(query_x)
        width = len(fit_x[0]) if count else 0
        require(0 < count <= 16384 and 0 < query_count <= min(4096, identity.query_batch_size) and
                len(fit_y) == count and 0 < width <= 128 and (count + query_count) * width <= 1_000_000,
                "public_checkpoint_input_cap_exceeded")
        require(identity.context_policy != "all" or count <= identity.context_cap,
                "public_checkpoint_context_cap_exceeded")
        def ids(role, n):
            return tuple(digest(json.dumps([seed, arm if role == "fit" else "shared", role, i]).encode()) for i in range(n))
        try:
            context_rows = tuple(tuple(float(value) for value in row) for row in fit_x)
            query_rows = tuple(tuple(float(value) for value in row) for row in query_x)
            targets = tuple(float(value) for value in fit_y)
            require(all(len(row) == width and all(math.isfinite(v) for v in row)
                        for row in (*context_rows, *query_rows)) and all(math.isfinite(v) for v in targets),
                    "public_checkpoint_input_invalid")
            training = TrainingBatch(FeatureBatch(ids("fit", count), context_rows, schema_sha256),
                                     targets, "development_train" if arm == "real" else "synthetic_train")
            query = FeatureBatch(ids("query", query_count), query_rows, schema_sha256)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("public_checkpoint_input_invalid") from None
        require(time.monotonic() - started < remaining, "public_checkpoint_wall_budget_exhausted")
        # A fresh interpreter receives only the declared fit/query feature
        # arguments. Fork would inherit the evaluator's query-label memory.
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        child = context.Process(target=worker, args=(writer, self.adapter, training, query, seed,
                                                     remaining, identity.resources.memory_bytes))
        try:
            child.start()
            writer.close()
            if not reader.poll(max(0.0, remaining - (time.monotonic() - started))):
                raise ValueError("public_checkpoint_wall_budget_exhausted")
            message = json.loads(reader.recv_bytes(1024 * 1024))
            child.join(timeout=max(0.0, remaining - (time.monotonic() - started)))
            require(not child.is_alive() and child.exitcode == 0 and message.get("state") == "completed",
                    "public_checkpoint_backend_failed")
            expected = min(count, identity.context_cap) if identity.context_policy == "head" else count
            require(set(message) == {"state", "row_ids", "values", "consumed_rows", "memory_bytes", "gpu_bytes"} and
                    message["row_ids"] == list(query.row_ids) and message["consumed_rows"] == expected and
                    type(message["memory_bytes"]) is int and 0 <= message["memory_bytes"] <= identity.resources.memory_bytes and
                    message["gpu_bytes"] == 0, "public_checkpoint_output_invalid")
            values = tuple(tuple(row) for row in message["values"])
            require(len(values) == query_count and all(len(row) == 1 and type(row[0]) in {int, float} and
                    math.isfinite(row[0]) for row in values), "public_checkpoint_output_invalid")
            require(time.monotonic() - started <= remaining, "public_checkpoint_wall_budget_exhausted")
            receipt = {"state": "interface_fixture_completed" if self.fixture_only else "supplied_offline_observation",
                       "scope": "development_only", "scientific_admission": False, "release_qualified": False,
                       "identity_sha256": identity.sha256, "weights_sha256": identity.weights_sha256,
                       "eligibility_sha256": self.admission.eligibility_sha256,
                       "native_reference_sha256": self.admission.native_reference_sha256,
                       "recipe_sha256": identity.recipe_sha256,
                       "preprocessing_sha256": identity.preprocessing_sha256,
                       "output_aggregation_sha256": identity.output_aggregation_sha256,
                       "provided_rows": count, "consumed_rows": expected, "query_rows": query_count,
                       "context_policy": identity.context_policy, "query_policy": identity.query_policy,
                       "memory_bytes": message["memory_bytes"], "gpu_bytes": 0,
                       "worker": "linux_fresh_spawned_cpu_child_wall_and_address_space_limits",
                       "snapshot_network_filesystem_isolation": "custodian_required_not_established_by_hook",
                       "prediction_sha256": digest(json.dumps(values, allow_nan=False, separators=(",", ":")).encode())}
            return tuple(row[0] for row in values), receipt
        except (OSError, EOFError, json.JSONDecodeError, TypeError, AttributeError):
            raise ValueError("public_checkpoint_backend_failed") from None
        finally:
            if child.pid is not None:
                stop_owned_child(child)
            writer.close()
            reader.close()
