"""Native episode stages: ``dope-ffi`` (ABI 3) DSE7 and V8 tensors from an episode's FIT rows.

Training-coupled batch code for the episode factory's producers. Query rows are only mapped through
what the FIT rows fitted; nothing row-derived outlives the call. ``s3_episode_factory`` re-exports
every name here, so ``research.levels.s3_episode_factory:native_v8`` stays a valid stage path.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from .s3_episode_ring import FactoryError, StageError


# -- native DSE7 stage ----------------------------------------------------------------------
_TASK_ID = {"regression": 0, "binary": 1, "multiclass": 2}
_CODES = {1: "null_pointer", 2: "shape", 3: "fit", 4: "capacity", 5: "panic", 6: "unpack", 7: "target"}
_BLOCKS = ("col_feats", "q_nl", "glob", "knots", "loadings", "w_base", "b_base", "misc", "col_mask",
           "is_const", "col_index", "col_meta", "head")
ABI_VERSION = 3
META_FIELDS = 12  # Nc2ColumnMetaV2::to_f32s
_LIB: Any = None  # the loaded library handle (code, not data: nothing row-derived is kept)


def ffi_path() -> Path:
    """``DOPE_FFI_LIB`` or the workspace build (release before debug)."""
    env = os.environ.get("DOPE_FFI_LIB")
    if env:
        return Path(env)
    target = Path(__file__).resolve().parents[2] / "engine" / "target"
    for profile in ("release", "debug"):
        candidate = target / profile / "libdope_ffi.so"
        if candidate.exists():
            return candidate
    raise FactoryError("ffi_library_missing")


def load_ffi(path: str | Path | None = None) -> Any:
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(path or ffi_path()))
        lib.dope_ffi_abi_version.restype = ctypes.c_uint32
        if lib.dope_ffi_abi_version() != ABI_VERSION:
            raise FactoryError("ffi_abi_version")
        lib.dope_ffi_dse7_len.restype = ctypes.c_uint64
        lib.dope_ffi_dse7_len.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.dope_ffi_dse7_offset.restype = ctypes.c_uint64
        lib.dope_ffi_dse7_offset.argtypes = [ctypes.c_uint32]
        f32p, u8p = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)
        lib.dope_ffi_column_meta.restype = ctypes.c_int32
        lib.dope_ffi_column_meta.argtypes = [f32p, ctypes.c_uint32, ctypes.c_uint32, f32p, ctypes.c_uint64]
        lib.dope_ffi_dse7_episode.restype = ctypes.c_int32
        lib.dope_ffi_dse7_episode.argtypes = [
            f32p, ctypes.c_uint32, ctypes.c_uint32, f32p, f32p, ctypes.c_uint32, f32p, ctypes.c_uint8,
            ctypes.c_uint32, f32p, ctypes.c_uint64, u8p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        lib.offsets = {name: int(lib.dope_ffi_dse7_offset(i)) for i, name in enumerate(_BLOCKS)}
        lib.dope_ffi_v8_len.restype = ctypes.c_uint64
        lib.dope_ffi_v8_len.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        lib.dope_ffi_v8_offset.restype = ctypes.c_uint64
        lib.dope_ffi_v8_offset.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.dope_ffi_v8_episode.restype = ctypes.c_int32
        lib.dope_ffi_v8_episode.argtypes = [
            f32p, ctypes.c_uint32, ctypes.c_uint32, f32p, f32p, ctypes.c_uint32, f32p, ctypes.c_uint8,
            ctypes.c_uint32, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint64, f32p, ctypes.c_uint64, u8p,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        _LIB = lib
    return _LIB


def _ptr(a: np.ndarray, kind: Any) -> Any:
    return a.ctypes.data_as(ctypes.POINTER(kind))


def dse7_arrays(x_fit: np.ndarray, y_fit: np.ndarray, x_qry: np.ndarray, y_qry: np.ndarray,
                task: str, budget: int = 2048) -> tuple[bytes, np.ndarray]:
    """(DSE7 packet bytes, flat float32 tensors) from FIT rows; query rows are only mapped."""
    lib = load_ffi()
    if task not in _TASK_ID:
        raise StageError("task_unsupported")
    xf = np.ascontiguousarray(x_fit, dtype=np.float32)
    xq = np.ascontiguousarray(x_qry, dtype=np.float32)
    yf = np.ascontiguousarray(y_fit, dtype=np.float32)
    yq = np.ascontiguousarray(y_qry, dtype=np.float32)
    n_fit, p = xf.shape
    n_qry = xq.shape[0]
    if xq.shape[1] != p or yf.shape != (n_fit,) or yq.shape != (n_qry,):
        raise StageError("shape")
    out = np.empty(int(lib.dope_ffi_dse7_len(n_fit, n_qry)), dtype=np.float32)
    cap = max(budget, 64)
    pkt = np.empty(cap, dtype=np.uint8)
    pkt_len = ctypes.c_uint32(0)
    code = lib.dope_ffi_dse7_episode(
        _ptr(xf, ctypes.c_float), n_fit, p, _ptr(yf, ctypes.c_float), _ptr(xq, ctypes.c_float), n_qry,
        _ptr(yq, ctypes.c_float), _TASK_ID[task], budget, _ptr(out, ctypes.c_float), out.size,
        _ptr(pkt, ctypes.c_uint8), cap, ctypes.byref(pkt_len))
    if code != 0:
        raise StageError("ffi_" + _CODES.get(int(code), "unknown"))
    return pkt[:pkt_len.value].tobytes(), out


# -- V8 (ABI 3): DSE8 packet + receiver-decoder inputs + evidence bank, all FIT-only ------------
_V8_BLOCKS = ("ev_col", "ev_edge", "ev_glob", "dec_col", "dec_edge", "dec_glob", "dec_code", "knots",
              "loadings", "W_base", "b_base", "is_cont", "col_index", "col_meta", "misc", "head")
_V8_SHAPES = {"ev_col": (-1, 128), "ev_edge": (-1, -1, 16), "dec_col": (-1, 48), "dec_edge": (-1, -1, 4),
              "knots": (-1, 17), "loadings": (-1, 12), "W_base": (8, -1), "col_meta": (-1, META_FIELDS)}
V8_MAX_COLS = 64


def v8_arrays(x_fit: np.ndarray, y_fit: np.ndarray, x_qry: np.ndarray, y_qry: np.ndarray, task: str,
              budget: int = 2048, normalized_only: bool = False, tier: int = 2,
              seed: int = 0) -> tuple[bytes, np.ndarray, int]:
    """(DSE8 packet bytes, flat float32 V8 tensors, layout width) from FIT rows; query rows are only mapped."""
    lib = load_ffi()
    if task not in _TASK_ID:
        raise StageError("task_unsupported")
    xf = np.ascontiguousarray(x_fit, dtype=np.float32)
    xq = np.ascontiguousarray(x_qry, dtype=np.float32)
    yf = np.ascontiguousarray(y_fit, dtype=np.float32)
    yq = np.ascontiguousarray(y_qry, dtype=np.float32)
    n_fit, p = xf.shape
    n_qry = xq.shape[0]
    if xq.shape[1] != p or yf.shape != (n_fit,) or yq.shape != (n_qry,):
        raise StageError("shape")
    out = np.empty(int(lib.dope_ffi_v8_len(p, n_fit, n_qry)), dtype=np.float32)
    cap = max(budget, 64)
    pkt = np.empty(cap, dtype=np.uint8)
    pkt_len = ctypes.c_uint32(0)
    code = lib.dope_ffi_v8_episode(
        _ptr(xf, ctypes.c_float), n_fit, p, _ptr(yf, ctypes.c_float), _ptr(xq, ctypes.c_float), n_qry,
        _ptr(yq, ctypes.c_float), _TASK_ID[task], budget, int(bool(normalized_only)), tier,
        seed & 0xFFFFFFFFFFFFFFFF, _ptr(out, ctypes.c_float), out.size, _ptr(pkt, ctypes.c_uint8), cap,
        ctypes.byref(pkt_len))
    if code != 0:
        raise StageError("ffi_" + _CODES.get(int(code), "unknown"))
    return pkt[:pkt_len.value].tobytes(), out, min(p, V8_MAX_COLS)


def split_v8(floats: np.ndarray, p: int, n_fit: int, n_qry: int) -> dict[str, np.ndarray]:
    """Named float32 views (width ``min(p, 64)``, unsliced) of a ``v8_arrays`` result."""
    lib = load_ffi()
    w = min(p, V8_MAX_COLS)
    off = [int(lib.dope_ffi_v8_offset(i, p)) for i in range(len(_V8_BLOCKS))]
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(_V8_BLOCKS[:-1]):
        block = floats[off[i]:off[i + 1]]
        shape = _V8_SHAPES.get(name)
        if shape is not None:
            shape = tuple(w if d == -1 else d for d in shape)
            shape = tuple(block.size // int(np.prod([d for d in shape[1:]])) if j == 0 and d == -1 else d
                          for j, d in enumerate(shape))
        out[name] = block.reshape(shape) if shape else block
    at = off[-1]
    for name, n in (("fit", n_fit), ("qry", n_qry)):
        out["x_" + name] = floats[at:at + n * w].reshape(n, w)
        at += n * w
        out["y_" + name] = floats[at:at + n]
        at += n
    if at != floats.size:
        raise FactoryError("native_v8_layout")
    return out


def native_v8(episode: dict[str, Any], budget: int = 2048, normalized_only: bool = False,
              tier: int = 2) -> dict[str, Any]:
    """Stage: one ``SingleEncV8`` training episode, every tensor from this episode's FIT rows.

    Wants a view built with ``ViewConfig(emit="raw")``. Returns exactly the precomputed dict
    ``s3_v8_trainer.consume_episode_tensors`` uploads (sliced to the packet's ``p_eff`` columns),
    in the packet's unit space, plus the packet bytes and the episode identity. Classification
    rows whose class the packet does not carry (code -1: under the k >= 10 class floor in FIT,
    or unseen in FIT for query rows) are dropped from the loss tensors and counted. Missing cells
    are 0 (the unit-space median) in ``X_fit``/``X_qry``, with masks ``M_fit``/``M_qry``.
    """
    # A pure function of this episode's identity (its split and view), so payloads stay identical
    # for any producer count.
    seed = int(hashlib.sha256(f"{episode['split_sig']}|{episode['view_sig']}".encode()).hexdigest()[:16], 16)
    packet, floats, w = v8_arrays(episode["X_fit"], episode["y_fit"], episode["X_qry"], episode["y_qry"],
                                  episode["task"], budget, normalized_only, tier, seed)
    n_fit, n_qry = int(episode["y_fit"].shape[0]), int(episode["y_qry"].shape[0])
    v = split_v8(floats, int(episode["X_fit"].shape[1]), n_fit, n_qry)
    misc = v["misc"]
    p_eff = int(misc[2])
    keep_f = v["y_fit"] >= 0 if episode["task"] != "regression" else np.ones(n_fit, dtype=bool)
    keep_q = v["y_qry"] >= 0 if episode["task"] != "regression" else np.ones(n_qry, dtype=bool)
    if int(keep_f.sum()) < 4 or int(keep_q.sum()) < 1:
        raise StageError("v8_too_few_rows")
    task_id = _TASK_ID[episode["task"]]
    n_classes = 1 if task_id == 0 else (2 if task_id == 1 else int(misc[12]))
    # Missing cells: 0 is the unit-space median; the masks travel with the rows.
    x_fit, x_qry = v["x_fit"][keep_f, :p_eff], v["x_qry"][keep_q, :p_eff]
    out = {key: episode[key] for key in ("dataset_hash", "family_stem", "source_kind", "task", "split_sig",
                                         "view_sig", "kind", "table_id", "step", "slot", "attempt")
           if key in episode}
    out.update(
        ev_col=v["ev_col"][:p_eff].copy(), ev_edge=v["ev_edge"][:p_eff, :p_eff].copy(), ev_glob=v["ev_glob"].copy(),
        dec_col=v["dec_col"][:p_eff].copy(), dec_edge=v["dec_edge"][:p_eff, :p_eff].copy(),
        dec_glob=v["dec_glob"].copy(), knots=v["knots"][:p_eff].copy(), loadings=v["loadings"][:p_eff].copy(),
        W_base=v["W_base"][:, :p_eff].copy(), b_base=v["b_base"].copy(), resid_sd=np.float32(misc[0]),
        z_max=np.float32(misc[1]), is_cont=v["is_cont"][:p_eff].copy(),
        X_fit=np.nan_to_num(x_fit, nan=0.0), y_fit=v["y_fit"][keep_f].copy(),
        X_qry=np.nan_to_num(x_qry, nan=0.0), y_qry=v["y_qry"][keep_q].copy(),
        M_fit=~np.isfinite(x_fit), M_qry=~np.isfinite(x_qry),
        task_id=task_id, n_classes=n_classes, packet=packet, packet_bytes=len(packet),
        col_index=v["col_index"][:p_eff].astype(np.int32), col_meta=v["col_meta"][:p_eff].copy(),
        fit_masked=int(n_fit - keep_f.sum()), qry_masked=int(n_qry - keep_q.sum()),
    )
    if w < p_eff:
        raise FactoryError("native_v8_layout")
    return out


def v8_capacity(max_fit: int, max_qry: int, p: int = 48, budget: int = 2048) -> int:
    """Bytes of the largest pickled ``native_v8`` payload, with headroom for metadata."""
    w = min(p, V8_MAX_COLS)
    head = w * 128 + w * w * 16 + 160 + w * 48 + w * w * 4 + 32 + w * 17 + w * 12 + 8 * w + 8 + w + w * META_FIELDS
    return 4 * (head + (max_fit + max_qry) * (w + 1)) + budget + 8 * w + (256 << 10)


def column_meta(x_fit: np.ndarray) -> np.ndarray:
    """``Nc2ColumnMetaV2`` of every column of the FIT rows ``(n_fit, p)``: ``(p, 12)`` float32."""
    lib = load_ffi()
    xf = np.ascontiguousarray(x_fit, dtype=np.float32)
    n_fit, p = xf.shape
    out = np.empty((p, META_FIELDS), dtype=np.float32)
    code = lib.dope_ffi_column_meta(_ptr(xf, ctypes.c_float), n_fit, p, _ptr(out, ctypes.c_float), out.size)
    if code != 0:
        raise StageError("ffi_" + _CODES.get(int(code), "unknown"))
    return out


def native_dse7(episode: dict[str, Any], budget: int = 2048) -> dict[str, Any]:
    """Stage: the DSE7 packet of this episode's FIT rows plus FIT and query rows in its coordinates.

    Wants a view built with ``ViewConfig(emit="raw")`` so the Rust scale card sees raw values, as
    serving does. Multiclass FIT classes beyond the packet's eight most frequent get code -1.
    ``split_native(...)["col_meta"]`` holds each packet column's ``Nc2ColumnMetaV2`` from FIT raw
    values in packet units: group id, kind, width, parent, support class, skew guard, lattice q0
    and phase, lower / upper bound flags, tail asymmetry, distinct-count bucket.
    """
    packet, floats = dse7_arrays(episode["X_fit"], episode["y_fit"], episode["X_qry"], episode["y_qry"],
                                 episode["task"], budget)
    keep = ("dataset_hash", "family_stem", "source_kind", "task", "n_classes", "split_sig", "view_sig",
            "kind", "table_id", "step", "slot", "attempt", "target_col", "aug_stats")
    out = {key: episode[key] for key in keep}
    out.update(packet=packet, floats=floats, n_fit=int(episode["y_fit"].shape[0]),
               n_qry=int(episode["y_qry"].shape[0]), col_source=np.asarray(episode["col_source"]),
               distractor_mask=np.asarray(episode["distractor_mask"]))
    return out


def split_native(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    """Named float32 views of a ``native_dse7`` payload (no copy)."""
    offsets = load_ffi().offsets
    f, n_fit, n_qry = payload["floats"], payload["n_fit"], payload["n_qry"]
    cols = offsets["is_const"] - offsets["col_mask"]  # MAX_COLS
    out = {}
    for name, nxt in zip(_BLOCKS[:-1], _BLOCKS[1:]):
        at, size = offsets[name], offsets[nxt] - offsets[name]
        out[name] = f[at:at + size].reshape(cols, -1) if size % cols == 0 and size > cols else f[at:at + size]
    out["w_base"] = out["w_base"].reshape(-1, cols)  # [class][column]
    at = offsets["head"]
    for name, n in (("fit", n_fit), ("qry", n_qry)):
        out["x_" + name] = f[at:at + n * cols].reshape(n, cols)
        at += n * cols
        out["y_" + name] = f[at:at + n]
        at += n
    if at != f.size:
        raise FactoryError("native_layout")
    return out


def native_capacity(max_fit: int, max_qry: int, budget: int = 2048) -> int:
    """Bytes of the largest pickled ``native_dse7`` payload, with headroom for metadata."""
    head = 48 * 42 + 48 * 4 + 16 + 48 * 17 + 48 * 12 + 8 * 48 + 8 + 8 + 3 * 48 + 48 * META_FIELDS
    return 4 * (head + (max_fit + max_qry) * 49) + budget + 16 * (max_fit + max_qry) + (256 << 10)


# -- bench ----------------------------------------------------------------------------------
