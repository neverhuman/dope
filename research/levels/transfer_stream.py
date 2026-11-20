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
_TENSOR_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


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

    def __init__(self, count: int, read, *, prefetch: int, batch: int, start: int = 0) -> None:
        if (
            type(count) is not int
            or type(prefetch) is not int
            or type(batch) is not int
            or type(start) is not int
            or count < 1
            or start < 0
            or start > count
            or not 1 <= prefetch <= MAX_PREFETCH
            or not 1 <= batch <= MAX_BATCH
            or not callable(read)
        ):
            raise StreamError("stream_limit")
        self.count = count
        self._read = read
        self.prefetch = prefetch
        self.batch = batch
        self._next = start
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
    rows = features.shape[0]
    width = rows // microbatches
    model.zero_grad(set_to_none=True)
    try:
        for start in range(0, rows, width):
            stop = start + width
            chunk_features = features[start:stop]
            chunk_target = target[start:stop]
            if bool((~torch.isfinite(chunk_features)).any()) or bool((~torch.isfinite(chunk_target)).any()):
                raise StreamError("stream_nonfinite")
            prediction = model(chunk_features)
            loss = torch.nn.functional.mse_loss(prediction, chunk_target, reduction="sum") / rows
            if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(loss).all()):
                raise StreamError("stream_nonfinite")
            loss.backward()
        grads = []
        for parameter in model.parameters():
            grad = parameter.grad
            if grad is None:
                raise StreamError("stream_state")
            if not bool(torch.isfinite(grad).all()):
                raise StreamError("stream_nonfinite")
            grads.append(grad.detach().clone())
        return tuple(grads)
    except StreamError:
        model.zero_grad(set_to_none=True)
        raise


def apply_update(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    microbatches: int = 1,
) -> None:
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise StreamError("stream_device")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise StreamError("stream_state")
    model.to(device)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if [id(parameter) for parameter in owned] != [id(parameter) for parameter in model.parameters()]:
        raise StreamError("stream_state")
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.device != device:
                    state[key] = value.to(device)
    accumulated_gradient(
        model,
        features.to(device),
        target.to(device),
        microbatches=microbatches,
    )
    optimizer.step()


def capture(
    model: nn.Module,
    ledger: ExposureLedger,
    step: int,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    committed_items = ledger.committed()
    committed_count = len(committed_items)
    if (
        type(step) is not int
        or step < 0
        or (step == 0) != (committed_count == 0)
        or committed_count < step
    ):
        raise StreamError("stream_state")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise StreamError("stream_state")
    parameters = {}
    shapes = {}
    for name, parameter in model.named_parameters():
        values = parameter.detach().cpu()
        if not bool(torch.isfinite(values).all()):
            raise StreamError("stream_nonfinite")
        parameters[name] = values.tolist()
        shapes[name] = list(values.shape)
    cuda_rng = (
        [_export_tree(state) for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else []
    )
    return {
        "schema": SCHEMA,
        "count": ledger.count,
        "step": step,
        "committed_count": committed_count,
        "committed": list(committed_items),
        "parameters": parameters,
        "shapes": shapes,
        "optimizer": _export_tree(optimizer.state_dict()),
        "rng": _export_tree(torch.get_rng_state()),
        "cuda_rng": cuda_rng,
    }


def restore(model: nn.Module, optimizer: torch.optim.Optimizer, payload: dict) -> ExposureLedger:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise StreamError("stream_state")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise StreamError("stream_state")
    parameters = payload.get("parameters")
    shapes = payload.get("shapes")
    committed = payload.get("committed")
    count = payload.get("count")
    step = payload.get("step")
    committed_count = payload.get(
        "committed_count", len(committed) if isinstance(committed, list) else -1
    )
    cuda_rng_raw = payload.get("cuda_rng", [])
    names = {name for name, _ in model.named_parameters()}
    if (
        not isinstance(parameters, dict)
        or not isinstance(shapes, dict)
        or not isinstance(committed, list)
        or set(parameters) != names
        or set(shapes) != names
        or type(step) is not int
        or type(committed_count) is not int
        or step < 0
        or committed_count != len(committed)
        or (step == 0) != (committed_count == 0)
        or committed_count < step
        or "optimizer" not in payload
        or "rng" not in payload
        or not isinstance(cuda_rng_raw, list)
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
    try:
        optimizer.load_state_dict(_import_tree(payload["optimizer"]))
        rng = _import_tree(payload["rng"])
        if not torch.is_tensor(rng) or rng.dtype != torch.uint8:
            raise StreamError("stream_state")
        torch.set_rng_state(rng)
        if cuda_rng_raw:
            cuda_states = [_import_tree(item) for item in cuda_rng_raw]
            if any(not torch.is_tensor(s) or s.dtype != torch.uint8 for s in cuda_states):
                raise StreamError("stream_state")
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                if len(cuda_states) != len(torch.cuda.get_rng_state_all()):
                    raise StreamError("stream_state")
                torch.cuda.set_rng_state_all(cuda_states)
    except (RuntimeError, ValueError, TypeError):
        raise StreamError("stream_state") from None
    return ExposureLedger(count, tuple(committed))


def _export_tree(value):
    if torch.is_tensor(value):
        cpu = value.detach().cpu().contiguous()
        kind = str(cpu.dtype).replace("torch.", "")
        if kind not in _TENSOR_DTYPES:
            raise StreamError("stream_state")
        raw = cpu.reshape(-1).view(torch.uint8).reshape(-1)
        return {"tensor": kind, "shape": list(cpu.shape), "values": raw.tolist()}
    if isinstance(value, dict):
        return {str(key): _export_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"tuple": [_export_tree(item) for item in value]}
    if isinstance(value, list):
        return [_export_tree(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise StreamError("stream_state")


def _import_tree(value):
    if isinstance(value, dict) and set(value) == {"tensor", "shape", "values"}:
        dtype = _TENSOR_DTYPES.get(value["tensor"])
        shape = value["shape"]
        if dtype is None or not isinstance(shape, list) or not isinstance(value["values"], list):
            raise StreamError("stream_state")
        try:
            raw = torch.tensor(value["values"], dtype=torch.uint8)
            count = 1
            for dim in shape:
                count *= dim
            if raw.numel() != count * torch.empty((), dtype=dtype).element_size():
                raise StreamError("stream_state")
            return raw.view(dtype).reshape(shape)
        except (TypeError, ValueError, RuntimeError):
            raise StreamError("stream_state") from None
    if isinstance(value, dict) and set(value) == {"tuple"} and isinstance(value["tuple"], list):
        return tuple(_import_tree(item) for item in value["tuple"])
    if isinstance(value, dict):
        return {
            int(key) if isinstance(key, str) and key.isdigit() else key: _import_tree(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_import_tree(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise StreamError("stream_state")


def write_state(path: Path, payload: dict) -> None:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise StreamError("stream_state")
    destination = Path(path)
    staged = destination.with_name(destination.name + ".part")
    try:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        with staged.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except (OSError, TypeError, ValueError):
        raise StreamError("stream_state") from None
    finally:
        if staged.exists():
            staged.unlink()


def read_state(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise StreamError("stream_state") from None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise StreamError("stream_state")
    return payload
