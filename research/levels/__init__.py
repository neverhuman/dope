"""Offline multi-level table-model research; never a product scorer or runtime."""
from __future__ import annotations

from typing import Any

__all__ = [
    "CANONICAL_GATE_KEYS_14",
    "DOPE_ACTIVE_ENCODER_PARAMETERS",
    "DOPE_ACTIVE_ENCODER_TENSORS",
    "DOPE_FEATURES_629_DIM",
    "DopeEncoder629",
    "PINNED_MULTITASK_SAFETENSORS_SHA256",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import encoder

        return getattr(encoder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
