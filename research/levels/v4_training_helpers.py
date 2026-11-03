"""Pure scheduling, padding and identity helpers for the v4 trainer."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _bucket_size(n: int, width: int) -> int:
    return _round_up(n, width)


def _as_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device="cpu")


def collate_cells(cells: Sequence[Any], config: Any, padded_batch_cls: Any, trainer_error: type[Exception], role_violation: type[Exception]) -> Any:
    if not cells:
        raise ValueError("empty_batch")
    dims = [c.validate() for c in cells]
    max_rows = _bucket_size(max(max(d[0], d[2]) for d in dims), config.row_bucket)
    max_cols = _bucket_size(max(d[1] for d in dims), config.column_bucket)
    allocation = len(cells) * max_rows * max_cols
    if allocation > config.max_padded_cells:
        raise trainer_error("padded_allocation_limit_exceeded")
    b = len(cells)
    x_fit = torch.zeros((b, max_rows, max_cols), dtype=torch.float32)
    y_fit = torch.zeros((b, max_rows), dtype=torch.float32)
    fit_row_mask = torch.zeros((b, max_rows), dtype=torch.bool)
    fit_col_mask = torch.zeros((b, max_cols), dtype=torch.bool)
    x_query = torch.zeros_like(x_fit)
    y_query = torch.zeros_like(y_fit)
    query_row_mask = torch.zeros_like(fit_row_mask)
    query_col_mask = torch.zeros_like(fit_col_mask)
    any_keep = any(c.keep_mask is not None for c in cells)
    # Native landmark dropout is bool[23], separate from row/column padding.
    keep = torch.zeros((b, 23), dtype=torch.bool) if any_keep else None
    for i, cell in enumerate(cells):
        nf, pf, nq, _ = dims[i]
        xf, yf = _as_tensor(cell.x_fit), _as_tensor(cell.y_fit)
        xq, yq = _as_tensor(cell.x_query), _as_tensor(cell.y_query)
        x_fit[i, :nf, :pf], y_fit[i, :nf] = xf, yf
        x_query[i, :nq, :pf], y_query[i, :nq] = xq, yq
        fit_row_mask[i, :nf] = True
        query_row_mask[i, :nq] = True
        fit_col_mask[i, :pf] = True
        query_col_mask[i, :pf] = True
        if keep is not None:
            observed = getattr(cell.native_report, "observed", None)
            if isinstance(cell.native_report, Mapping):
                observed = cell.native_report.get("observed", observed)
            if cell.keep_mask is None:
                if observed is None:
                    raise role_violation("native_observed_mask_required")
                candidate = torch.as_tensor(observed)
            else:
                candidate = torch.as_tensor(cell.keep_mask)
            if candidate.dtype != torch.bool or candidate.shape != (23,):
                raise role_violation("native_keep_mask_must_be_bool23")
            if observed is not None:
                available = torch.as_tensor(observed)
                if available.dtype != torch.bool or available.shape != (23,) or bool((candidate & ~available).any()):
                    raise role_violation("native_keep_mask_exceeds_observations")
            keep[i] = candidate
    return padded_batch_cls(
        cell_ids=tuple(c.cell_id for c in cells), x_fit=x_fit, y_fit=y_fit,
        fit_row_mask=fit_row_mask, fit_column_mask=fit_col_mask,
        x_query=x_query, y_query=y_query, query_row_mask=query_row_mask,
        query_column_mask=query_col_mask, keep_masks=keep,
        schemas=[c.schema for c in cells], native_reports=[c.native_report for c in cells],
        fit_row_ids=[c.fit_row_ids for c in cells], query_row_ids=[c.query_row_ids for c in cells],
        family_ids=tuple(c.family_id for c in cells),
        weights=torch.tensor([c.weight for c in cells], dtype=torch.float32),
        fit_scopes=[c.fit_scope_sha256 for c in cells],
    )


def build_shape_batches(cells: Sequence[Any], config: Any, seed: int | None = None) -> list[tuple[str, ...]]:
    """Return a deterministic epoch permutation, bucketing rows and expanded columns."""
    buckets: dict[tuple[int, int], list[Any]] = {}
    for cell in cells:
        nf, pf, nq, _ = cell.validate()
        key = (_bucket_size(max(nf, nq), config.row_bucket), _bucket_size(pf, config.column_bucket))
        buckets.setdefault(key, []).append(cell)
    rng = random.Random(config.seed if seed is None else seed)
    ordered: list[tuple[str, ...]] = []
    for key in sorted(buckets):
        group = buckets[key]
        rng.shuffle(group)
        size = config.batch_size
        for offset in range(0, len(group), size):
            ordered.append(tuple(c.cell_id for c in group[offset:offset + size]))
    rng.shuffle(ordered)
    cell_ids = [c.cell_id for c in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("duplicate_cell_id")
    return ordered



def _content_identity(value: Any, identity_error: type[Exception]) -> Any:
    """Private values contribute only typed content hashes, never repr text."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {"tensor_dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, np.ndarray):
        return _content_identity(torch.as_tensor(value), identity_error)
    if is_dataclass(value):
        return {field.name: _content_identity(getattr(value, field.name), identity_error) for field in fields(value)}
    if isinstance(value, Mapping):
        if not all(type(k) is str for k in value):
            raise identity_error("unsupported_training_identity_value")
        return {k: _content_identity(v, identity_error) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_content_identity(v, identity_error) for v in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise identity_error("unsupported_training_identity_value")


def _dataset_fingerprint(cells: Sequence[Any], identity_error: type[Exception]) -> str:
    content = [_content_identity(c, identity_error) for c in sorted(cells, key=lambda c: c.cell_id)]
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _jsonable_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_tuple(x) for x in value]
    if isinstance(value, list):
        return [_jsonable_tuple(x) for x in value]
    return value


def _to_tuple(value: Any) -> Any:
    return tuple(_to_tuple(x) for x in value) if isinstance(value, list) else value


def _rng_state() -> dict[str, Any]:
    py = random.getstate()
    np_name, np_keys, np_pos, np_gauss, np_cached = np.random.get_state()
    return {
        "python": _jsonable_tuple(py),
        "numpy": {"name": np_name, "keys": np_keys.astype(np.uint32).tolist(), "pos": int(np_pos), "gauss": np_gauss, "cached": float(np_cached)},
        "torch_cpu": torch.get_rng_state(),
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(_to_tuple(state["python"]))
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), int(n["pos"]), n["gauss"], float(n["cached"])))
    torch.set_rng_state(state["torch_cpu"])


def _state_shapes(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if type(value) is float:
        return math.isfinite(value)
    return True


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
