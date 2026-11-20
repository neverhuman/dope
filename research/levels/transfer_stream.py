"""Indexed single-device training stream.

Records are read by index. The stream retains a bounded window, not the
manifest. Distributed training stays closed until this single-device
contract holds.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

SCHEMA = "dope-transfer-stream/v1"
MAX_PREFETCH = 8
MAX_BATCH = 32


class StreamError(ValueError):
    """Static refusal. The message is a code, not a source value."""


@dataclass(frozen=True)
class StreamRecord:
    index: int
    shape: tuple[int, ...]
    payload: object


def support() -> dict[str, bool | str]:
    return {
        "schema": SCHEMA,
        "cpu": True,
        "cuda": bool(torch.cuda.is_available()),
        "ddp": False,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise StreamError("stream_device")


class ExposureLedger:
    """Committed indexes. A second commit of the same index refuses."""

    def __init__(self, count: int, committed: tuple[int, ...] = ()) -> None:
        if type(count) is not int or count < 1:
            raise StreamError("stream_limit")
        self.count = count
        self._order: list[int] = []
        self._seen: set[int] = set()
        for index in committed:
            self.commit(index)

    def commit(self, index: int) -> None:
        if type(index) is not int or index < 0 or index >= self.count or index in self._seen:
            raise StreamError("stream_coverage")
        self._seen.add(index)
        self._order.append(index)

    def committed(self) -> tuple[int, ...]:
        return tuple(self._order)

    def missing(self) -> int:
        return self.count - len(self._seen)

    def complete(self) -> bool:
        return self.missing() == 0


class IndexedStream:
    """Pulls records by index and groups a batch by shape."""

    def __init__(self, count: int, read, *, prefetch: int, batch: int) -> None:
        if (
            type(count) is not int
            or type(prefetch) is not int
            or type(batch) is not int
            or count < 1
            or not 1 <= prefetch <= MAX_PREFETCH
            or not 1 <= batch <= MAX_BATCH
            or not callable(read)
        ):
            raise StreamError("stream_limit")
        self.count = count
        self._read = read
        self.prefetch = prefetch
        self.batch = batch
        self._next = 0
        self._window: list[StreamRecord] = []

    def resident(self) -> int:
        return len(self._window)

    def __iter__(self):
        pending: list[StreamRecord] = []
        while True:
            nxt = self._peek()
            if nxt is None:
                if pending:
                    yield tuple(pending)
                return
            if pending and (nxt.shape != pending[0].shape or len(pending) >= self.batch):
                yield tuple(pending)
                pending = []
            pending.append(self._pop())

    def _peek(self) -> StreamRecord | None:
        self._fill()
        if not self._window:
            return None
        return self._window[0]

    def _pop(self) -> StreamRecord:
        self._fill()
        if not self._window:
            raise StreamError("stream_state")
        return self._window.pop(0)

    def _fill(self) -> None:
        while len(self._window) < self.prefetch and self._next < self.count:
            index = self._next
            record = self._read(index)
            if not isinstance(record, StreamRecord) or record.index != index:
                raise StreamError("stream_state")
            shape = record.shape
            if (
                not isinstance(shape, tuple)
                or not shape
                or any(type(part) is not int or part < 1 for part in shape)
            ):
                raise StreamError("stream_shape")
            self._window.append(record)
            self._next += 1


def accumulated_gradient(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    microbatches: int,
) -> tuple[torch.Tensor, ...]:
    """Sum of per-row losses, divided once by the full row count."""
    if type(microbatches) is not int or microbatches < 1:
        raise StreamError("stream_limit")
    if (
        not torch.is_tensor(features)
        or not torch.is_tensor(target)
        or features.ndim != 2
        or target.ndim != 2
        or features.shape[0] < 1
        or target.shape[0] != features.shape[0]
        or features.dtype != torch.float32
        or target.dtype != torch.float32
        or features.shape[0] % microbatches != 0
    ):
        raise StreamError("stream_shape")
    if bool((~torch.isfinite(features)).any()) or bool((~torch.isfinite(target)).any()):
        raise StreamError("stream_nonfinite")
    rows = features.shape[0]
    width = rows // microbatches
    model.zero_grad(set_to_none=True)
    for start in range(0, rows, width):
        stop = start + width
        prediction = model(features[start:stop])
        loss = torch.nn.functional.mse_loss(prediction, target[start:stop], reduction="sum") / rows
        if not bool(torch.isfinite(loss).all()):
            raise StreamError("stream_nonfinite")
        loss.backward()
    grads = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise StreamError("stream_state")
        grads.append(parameter.grad.detach().clone())
    return tuple(grads)


def apply_update(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    lr: float,
) -> None:
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise StreamError("stream_device")
    if type(lr) is not float or not 0.0 < lr <= 1.0:
        raise StreamError("stream_limit")
    grads = accumulated_gradient(model.to(device), features.to(device), target.to(device), microbatches=1)
    if all(bool(torch.count_nonzero(grad) == 0) for grad in grads):
        raise StreamError("stream_state")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.step()


def capture(model: nn.Module, ledger: ExposureLedger, step: int) -> dict[str, object]:
    if type(step) is not int or step != len(ledger.committed()):
        raise StreamError("stream_state")
    parameters = {}
    shapes = {}
    for name, parameter in model.named_parameters():
        values = parameter.detach().cpu()
        if not bool(torch.isfinite(values).all()):
            raise StreamError("stream_nonfinite")
        parameters[name] = values.tolist()
        shapes[name] = list(values.shape)
    return {
        "schema": SCHEMA,
        "count": ledger.count,
        "step": step,
        "committed": list(ledger.committed()),
        "parameters": parameters,
        "shapes": shapes,
    }


def restore(model: nn.Module, payload: dict) -> ExposureLedger:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise StreamError("stream_state")
    parameters = payload.get("parameters")
    shapes = payload.get("shapes")
    committed = payload.get("committed")
    count = payload.get("count")
    step = payload.get("step")
    names = {name for name, _ in model.named_parameters()}
    if (
        not isinstance(parameters, dict)
        or not isinstance(shapes, dict)
        or not isinstance(committed, list)
        or set(parameters) != names
        or set(shapes) != names
        or type(step) is not int
        or step != len(committed)
    ):
        raise StreamError("stream_state")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            shape = shapes[name]
            if not isinstance(shape, list) or parameter.shape != torch.Size(shape):
                raise StreamError("stream_shape")
            try:
                values = torch.tensor(parameters[name], dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError):
                raise StreamError("stream_state") from None
            if values.shape != parameter.shape or not bool(torch.isfinite(values).all()):
                raise StreamError("stream_state")
            parameter.copy_(values)
    return ExposureLedger(count, tuple(committed))


def write_state(path: Path, payload: dict) -> None:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise StreamError("stream_state")
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except (OSError, TypeError, ValueError):
        raise StreamError("stream_state") from None
    finally:
        if temporary.exists():
            temporary.unlink()


def read_state(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise StreamError("stream_state") from None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise StreamError("stream_state")
    return payload
