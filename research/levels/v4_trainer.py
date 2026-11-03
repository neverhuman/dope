"""Bounded CPU DML4/V4 scheduling, padding, role checks and resume; no model or science defaults."""
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


class TrainerError(RuntimeError):
    """Base class for fail-closed trainer errors."""


class RoleViolation(TrainerError):
    pass


class IdentityMismatch(TrainerError):
    pass


class NonFiniteUpdate(TrainerError):
    pass


class UnsupportedDeviceError(TrainerError):
    pass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ROLES = {"fit", "gradient_query"}


@dataclass(frozen=True)
class TrainingIdentity:
    source_sha256: str
    wire_sha256: str
    normalization_sha256: str
    fit_role_sha256: str
    objective_sha256: str
    software_sha256: str

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"invalid_{name}_sha256")

    def as_dict(self) -> dict[str, str]:
        return {
            "source_sha256": self.source_sha256,
            "wire_sha256": self.wire_sha256,
            "normalization_sha256": self.normalization_sha256,
            "fit_role_sha256": self.fit_role_sha256,
            "objective_sha256": self.objective_sha256,
            "software_sha256": self.software_sha256,
        }


@dataclass(frozen=True)
class BatchCell:
    """One source-free training cell with explicitly disjoint row memberships."""
    cell_id: str
    x_fit: Any
    y_fit: Any
    x_query: Any
    y_query: Any
    schema: Any
    native_report: Any
    fit_row_ids: tuple[str, ...]
    query_row_ids: tuple[str, ...]
    fit_role: str = "fit"
    query_role: str = "gradient_query"
    keep_mask: Any = None
    family_id: str = ""
    weight: float = 1.0
    fit_scope_sha256: str = ""

    def validate(self) -> tuple[int, int, int, int]:
        if self.fit_scope_sha256 and not _SHA256.fullmatch(self.fit_scope_sha256):
            raise IdentityMismatch("invalid_cell_fit_scope")
        if self.fit_role != "fit" or self.query_role != "gradient_query":
            raise RoleViolation("only_fit_to_gradient_query_cells_are_trainable")
        if not self.cell_id or not self.fit_row_ids or not self.query_row_ids:
            raise RoleViolation("missing_cell_or_row_membership")
        if len(set(self.fit_row_ids)) != len(self.fit_row_ids) or len(set(self.query_row_ids)) != len(self.query_row_ids):
            raise RoleViolation("duplicate_row_membership")
        if set(self.fit_row_ids) & set(self.query_row_ids):
            raise RoleViolation("fit_query_membership_overlap")
        xf, yf, xq, yq = map(torch.as_tensor, (self.x_fit, self.y_fit, self.x_query, self.y_query))
        if xf.ndim != 2 or xq.ndim != 2 or yf.ndim != 1 or yq.ndim != 1:
            raise ValueError("expected_2d_features_and_1d_targets")
        nf, pf = xf.shape
        nq, pq = xq.shape
        if nf != len(self.fit_row_ids) or nq != len(self.query_row_ids) or yf.shape[0] != nf or yq.shape[0] != nq:
            raise ValueError("membership_tensor_length_mismatch")
        if pf != pq or pf < 1 or not math.isfinite(float(self.weight)) or self.weight <= 0:
            raise ValueError("invalid_feature_width_or_weight")
        if self.keep_mask is not None:
            keep = torch.as_tensor(self.keep_mask)
            if keep.dtype != torch.bool or keep.shape != (23,):
                raise RoleViolation("native_keep_mask_must_be_bool23")
        return int(nf), int(pf), int(nq), int(pq)


@dataclass(frozen=True)
class V4TrainerConfig:
    batch_size: int = 4
    row_bucket: int = 8
    column_bucket: int = 8
    max_padded_cells: int = 4_000_000
    seed: int = 0
    report_interval: int = 100
    device: str = "cpu"
    accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if min(self.batch_size, self.row_bucket, self.column_bucket, self.max_padded_cells, self.report_interval) < 1:
            raise ValueError("trainer_limits_must_be_positive")
        if self.device != "cpu":
            raise UnsupportedDeviceError("trainer_updates_currently_support_cpu_only")
        if type(self.accumulation_steps) is not int or self.accumulation_steps != 1:
            raise UnsupportedDeviceError("gradient_accumulation_steps_other_than_one_not_supported")


@dataclass
class PaddedBatch:
    cell_ids: tuple[str, ...]
    x_fit: torch.Tensor
    y_fit: torch.Tensor
    fit_row_mask: torch.Tensor
    fit_column_mask: torch.Tensor
    x_query: torch.Tensor
    y_query: torch.Tensor
    query_row_mask: torch.Tensor
    query_column_mask: torch.Tensor
    keep_masks: torch.Tensor | None
    schemas: list[Any]
    native_reports: list[Any]
    fit_row_ids: list[tuple[str, ...]]
    query_row_ids: list[tuple[str, ...]]
    family_ids: tuple[str, ...]
    weights: torch.Tensor
    fit_scopes: list[str]

from . import v4_training_helpers as _training_helpers
(_finite_tree, _fsync_dir) = (_training_helpers._finite_tree, _training_helpers._fsync_dir)
(_restore_rng, _rng_state, _state_shapes, _to_tuple) = (_training_helpers._restore_rng, _training_helpers._rng_state, _training_helpers._state_shapes, _training_helpers._to_tuple)

def _dataset_fingerprint(cells: Sequence[BatchCell]) -> str:
    return _training_helpers._dataset_fingerprint(cells, IdentityMismatch)


def build_shape_batches(cells: Sequence[BatchCell], config: V4TrainerConfig, seed: int | None = None) -> list[tuple[str, ...]]:
    return _training_helpers.build_shape_batches(cells, config, seed)


def collate_cells(cells: Sequence[BatchCell], config: V4TrainerConfig) -> PaddedBatch:
    return _training_helpers.collate_cells(cells, config, PaddedBatch, TrainerError, RoleViolation)


class V4Trainer:
    """CPU-deterministic batched callback trainer with strict, atomic resume."""
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 cells: Sequence[BatchCell], loss_fn: Callable[[torch.nn.Module, PaddedBatch], Any],
                 identity: TrainingIdentity, config: V4TrainerConfig = V4TrainerConfig(),
                 scheduler: Any = None):
        setup_started = time.perf_counter()
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.cells = list(cells)
        if not self.cells:
            raise ValueError("empty_training_set")
        self.cell_by_id = {cell.cell_id: cell for cell in self.cells}
        if len(self.cell_by_id) != len(self.cells):
            raise ValueError("duplicate_cell_id")
        for cell in self.cells:
            cell.validate()
        self.loss_fn, self.identity, self.config = loss_fn, identity, config
        self.update_count = 0
        self.attempt_count = 0
        self.failed_update_count = 0
        self.setup_seconds = 0.0
        self.compute_seconds = 0.0
        self.cursor = 0
        self.permutation = build_shape_batches(self.cells, config, config.seed)
        self.dataset_fingerprint = _dataset_fingerprint(self.cells)
        self.cell_fingerprints = {c.cell_id: _dataset_fingerprint([c]) for c in self.cells}
        self.last_report: dict[str, Any] = {}
        self.setup_seconds = time.perf_counter() - setup_started

    def _next_batch(self) -> PaddedBatch:
        if self.cursor >= len(self.permutation):
            raise TrainerError("epoch_exhausted_call_new_epoch_explicitly")
        ids = self.permutation[self.cursor]
        for cell_id in ids:
            if _dataset_fingerprint([self.cell_by_id[cell_id]]) != self.cell_fingerprints[cell_id]:
                raise IdentityMismatch("training_cell_content_changed")
        return collate_cells([self.cell_by_id[i] for i in ids], self.config)

    def next_epoch(self) -> None:
        self.permutation = build_shape_batches(self.cells, self.config, self.config.seed + self.update_count + 1)
        self.cursor = 0

    def counters(self) -> dict[str, int | float]:
        """Return explicit attempt, failure, committed-update, setup, and compute totals."""
        return {
            "attempt_count": self.attempt_count,
            "failed_update_count": self.failed_update_count,
            "update_count": self.update_count,
            "setup_seconds": self.setup_seconds,
            "compute_seconds": self.compute_seconds,
            "accumulation_steps": 1,
        }

    def fit_step(self) -> dict[str, Any]:
        batch = self._next_batch()
        self.attempt_count += 1
        compute_started = time.perf_counter()
        snapshot: dict[str, Any] | None = None
        with _training_helpers.UpdateRollbackTransaction(
            rollback=lambda: _training_helpers.restore_update_snapshot(
                self.model, self.optimizer, self.scheduler, snapshot
            ),
            record_failure=lambda: _training_helpers.record_update_failure(self, compute_started),
            rollback_error_factory=lambda _: TrainerError("failed_update_rollback_failed"),
        ):
            snapshot = {
                "model": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "scheduler": copy.deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None,
                "rng": _rng_state(),
                "grads": [None if p.grad is None else p.grad.detach().clone() for p in self.model.parameters()],
            }
            self.optimizer.zero_grad(set_to_none=True)
            output = self.loss_fn(self.model, batch)
            loss = output["loss"] if isinstance(output, Mapping) else output
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("loss_fn_must_return_scalar_tensor_or_loss_mapping")
            if not bool(torch.isfinite(loss.detach()).all()):
                raise NonFiniteUpdate("nonfinite_loss_update_refused")
            loss.backward()
            for parameter in self.model.parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise NonFiniteUpdate("nonfinite_gradient_update_refused")
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if not _finite_tree(self.model.state_dict()) or not _finite_tree(self.optimizer.state_dict()):
                raise NonFiniteUpdate("nonfinite_optimizer_result_refused")
            result = dict(output) if isinstance(output, Mapping) else {"loss": loss.detach()}
            result["loss"] = float(loss.detach().cpu())
            self.cursor += 1
            self.update_count += 1
            self.compute_seconds += time.perf_counter() - compute_started
            result["cell_ids"] = batch.cell_ids
            result.update(self.counters())
            self.last_report = result
            return result

    def fit(self, updates: int) -> list[dict[str, Any]]:
        if updates < 0:
            raise ValueError("updates_must_be_nonnegative")
        return [self.fit_step() for _ in range(updates)]

    def save_checkpoint(self, path: str | Path) -> Path:
        """Write immutable payloads, then atomically publish CURRENT as commit point."""
        from safetensors.torch import save_file

        if _dataset_fingerprint(self.cells) != self.dataset_fingerprint:
            raise IdentityMismatch("training_dataset_content_changed")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        model_name, state_name = f"model-{generation}.safetensors", f"state-{generation}.pt"
        model_tmp, state_tmp = root / (model_name + ".tmp"), root / (state_name + ".tmp")
        model_state = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()}
        save_file(model_state, str(model_tmp))
        state = {
            "format": "dope.v4-training-resume/v2",
            "identity": self.identity.as_dict(),
            "model_shapes": {k: list(v) for k, v in _state_shapes(self.model).items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "update_count": self.update_count,
            "counters": self.counters(),
            "cursor": self.cursor,
            "permutation": [list(x) for x in self.permutation],
            "dataset_fingerprint": self.dataset_fingerprint,
            "config": asdict(self.config),
            "cell_ids": sorted(self.cell_by_id),
            "rng": _rng_state(),
        }
        torch.save(state, state_tmp)
        for tmp in (model_tmp, state_tmp):
            with open(tmp, "rb") as stream:
                os.fsync(stream.fileno())
        os.replace(model_tmp, root / model_name)
        os.replace(state_tmp, root / state_name)
        _fsync_dir(root)
        pointer = {"generation": generation, "model": model_name, "state": state_name}
        fd, tmp_name = tempfile.mkstemp(prefix="CURRENT.", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(pointer, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, root / "CURRENT")
            _fsync_dir(root)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return root / "CURRENT"

    def load_checkpoint(self, path: str | Path) -> None:
        from safetensors.torch import load_file

        root = Path(path)
        with _training_helpers.checkpoint_read_boundary(TrainerError):
            pointer = json.loads((root / "CURRENT").read_text(encoding="utf-8"))
            generation = pointer["generation"]
            if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
                raise ValueError
            if pointer["model"] != f"model-{generation}.safetensors" or pointer["state"] != f"state-{generation}.pt":
                raise ValueError
            model_file = root / pointer["model"]
            state_file = root / pointer["state"]
            state = torch.load(state_file, map_location="cpu", weights_only=True)
        if state.get("format") != "dope.v4-training-resume/v2":
            raise TrainerError("checkpoint_format_mismatch")
        if state.get("identity") != self.identity.as_dict():
            raise IdentityMismatch("checkpoint_training_identity_mismatch")
        if state.get("config") != asdict(self.config):
            raise IdentityMismatch("checkpoint_trainer_configuration_mismatch")
        if state.get("dataset_fingerprint") != _dataset_fingerprint(self.cells):
            raise IdentityMismatch("checkpoint_dataset_content_mismatch")
        if state.get("dataset_fingerprint") != self.dataset_fingerprint or state.get("cell_ids") != sorted(self.cell_by_id):
            raise IdentityMismatch("checkpoint_dataset_membership_mismatch")
        if state.get("model_shapes") != {k: list(v) for k, v in _state_shapes(self.model).items()}:
            raise IdentityMismatch("checkpoint_model_shape_mismatch")
        if (self.scheduler is None) != (state.get("scheduler") is None):
            raise IdentityMismatch("checkpoint_scheduler_configuration_mismatch")
        permutation = [tuple(batch) for batch in state["permutation"]]
        flattened = [cell_id for batch in permutation for cell_id in batch]
        if sorted(flattened) != sorted(self.cell_by_id) or len(flattened) != len(set(flattened)):
            raise IdentityMismatch("checkpoint_cursor_membership_mismatch")
        cursor = state.get("cursor")
        if type(cursor) is not int or not 0 <= cursor <= len(permutation):
            raise TrainerError("checkpoint_cursor_invalid")
        model_state = load_file(str(model_file), device="cpu")
        if set(model_state) != set(self.model.state_dict()) or any(tuple(model_state[k].shape) != tuple(self.model.state_dict()[k].shape) for k in model_state):
            raise IdentityMismatch("checkpoint_model_tensor_shape_mismatch")
        if any(not bool(torch.isfinite(v).all()) for v in model_state.values()):
            raise TrainerError("nonfinite_checkpoint_model")
        update_count = state.get("update_count")
        if type(update_count) is not int or update_count < 0:
            raise TrainerError("checkpoint_update_count_invalid")
        counters = state.get("counters")
        if not isinstance(counters, dict) or set(counters) != {
            "attempt_count", "failed_update_count", "update_count", "setup_seconds", "compute_seconds", "accumulation_steps"
        }:
            raise TrainerError("checkpoint_counters_invalid")
        attempts = counters["attempt_count"]
        failed = counters["failed_update_count"]
        if (type(attempts) is not int or type(failed) is not int or attempts < 0 or failed < 0
                or attempts != update_count + failed or type(counters["update_count"]) is not int
                or counters["update_count"] != update_count
                or type(counters["accumulation_steps"]) is not int or counters["accumulation_steps"] != 1
                or type(counters["setup_seconds"]) not in (float, int)
                or type(counters["compute_seconds"]) not in (float, int)
                or not math.isfinite(float(counters["setup_seconds"])) or float(counters["setup_seconds"]) < 0
                or not math.isfinite(float(counters["compute_seconds"])) or float(counters["compute_seconds"]) < 0):
            raise TrainerError("checkpoint_counters_invalid")
        # Validate mutable libraries on detached copies before committing state.
        try:
            staged_optimizer = copy.deepcopy(self.optimizer)
            staged_optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None:
                copy.deepcopy(self.scheduler).load_state_dict(state["scheduler"])
            random.Random().setstate(_to_tuple(state["rng"]["python"]))
            n = state["rng"]["numpy"]
            np.random.RandomState().set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), int(n["pos"]), n["gauss"], float(n["cached"])))
            torch.Generator(device="cpu").set_state(state["rng"]["torch_cpu"])
        except Exception as exc:
            raise TrainerError("checkpoint_state_invalid") from exc
        previous_model = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        previous_optimizer = copy.deepcopy(self.optimizer.state_dict())
        previous_scheduler = copy.deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None
        previous_rng = _rng_state()
        try:
            self.model.load_state_dict(model_state, strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            _restore_rng(state["rng"])
        except Exception as exc:
            self.model.load_state_dict(previous_model, strict=True)
            self.optimizer.load_state_dict(previous_optimizer)
            if self.scheduler is not None:
                self.scheduler.load_state_dict(previous_scheduler)
            _restore_rng(previous_rng)
            raise TrainerError("checkpoint_restore_refused") from exc
        self.permutation, self.cursor = permutation, cursor
        self.update_count = update_count
        self.attempt_count = attempts
        self.failed_update_count = failed
        self.setup_seconds = float(counters["setup_seconds"])
        self.compute_seconds = float(counters["compute_seconds"])


class BoundedPrefetcher:
    """Small owned host ring; optional CUDA copies complete through per-slot events.

    The trainer update path currently accepts CPU only. This helper is an
    isolated transfer primitive and does not qualify or claim CUDA training.
    """
    def __init__(self, capacity: int = 2, device: str = "cpu"):
        if capacity < 1:
            raise ValueError("capacity_must_be_positive")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise UnsupportedDeviceError("unsupported_prefetch_device")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise UnsupportedDeviceError("cuda_unavailable")
        self.capacity = capacity
        self._slots: list[dict[str, Any] | None] = [None] * capacity
        self._events = [torch.cuda.Event(blocking=False) for _ in range(capacity)] if self.device.type == "cuda" else []
        self._next = 0
        self._closed = False

    def transfer(self, tensors: Mapping[str, torch.Tensor]) -> tuple[int, dict[str, torch.Tensor]]:
        if self._closed:
            raise TrainerError("prefetcher_closed")
        slot = self._next
        if self._slots[slot] is not None:
            raise TrainerError("prefetch_slot_still_owned_by_consumer")
        self._next = (self._next + 1) % self.capacity
        if self.device.type == "cpu":
            result = {k: v.to("cpu", copy=True) for k, v in tensors.items()}
            self._slots[slot] = result
            return slot, result
        host = {k: v.detach().to("cpu").pin_memory() for k, v in tensors.items()}
        stream = torch.cuda.current_stream(self.device)
        result = {k: v.to(self.device, non_blocking=True) for k, v in host.items()}
        self._events[slot].record(stream)
        self._slots[slot] = {"host": host, "device": result}
        return slot, result

    def consume(self, slot: int) -> dict[str, torch.Tensor]:
        if not 0 <= slot < self.capacity or self._slots[slot] is None:
            raise TrainerError("prefetch_slot_not_owned")
        if self.device.type == "cuda":
            self._events[slot].synchronize()
            record = self._slots[slot]
            assert isinstance(record, dict) and "device" in record
            return record["device"]
        return self._slots[slot]  # type: ignore[return-value]

    def release(self, slot: int) -> None:
        if not 0 <= slot < self.capacity:
            raise TrainerError("prefetch_slot_invalid")
        if self.device.type == "cuda" and self._slots[slot] is not None:
            self._events[slot].synchronize()
        self._slots[slot] = None

    def close(self) -> None:
        if self._closed:
            return
        for slot in range(self.capacity):
            self.release(slot)
        self._closed = True

    def __enter__(self) -> "BoundedPrefetcher":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
