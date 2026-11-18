"""Research view beside the pinned high encoder.

``research-encoder-view/v0`` is not a product recipe and not a weight release.
``DopeEncoder629.forward`` stays level ``high``. Ultra is a separate call.
``dope.packet-global/v0`` is the global fit-row token only.
"""
from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence

import torch

from .encoder import CANONICAL_GATE_KEYS_14, DopeEncoder629

RESEARCH_ENCODER_VIEW_RECIPE = "research-encoder-view/v0"
PACKET_GLOBAL_SCHEMA = "dope.packet-global/v0"
PACKET_GLOBAL_WIDTH = 24
PACKET_GLOBAL_FIELDS: tuple[str, ...] = (
    *CANONICAL_GATE_KEYS_14,
    "residual_std",
    "noise_scale",
    "class_prior",
    "y_min",
    "y_max",
    "ln_n_train",
    "p_use",
    "n_trees",
    "n_pairs",
    "level_code",
)
LEVEL_CODE = {"high": 0.0, "ultra": 1.0}
QUERY_KEYS = ("X_query", "y_query", "query_rows", "query_features")
_BUNDLE_FIELDS = ("residual_std", "noise_scale", "class_prior", "y_min", "y_max", "trees", "pairs")


def encode_research_view(
    encoder: DopeEncoder629,
    dataset_features: torch.Tensor,
    col_landmarks: torch.Tensor,
    *,
    level: str,
) -> torch.Tensor:
    """Return ``[B, 629]``. ``high`` calls ``DopeEncoder629.forward`` itself."""
    if level == "high":
        return encoder.forward(dataset_features, col_landmarks)
    if level != "ultra":
        raise ValueError("research encoder view accepts only high or ultra")
    _validate_encoder_inputs(encoder, dataset_features, col_landmarks)
    routed = encoder.router(dataset_features, level="ultra")
    task = routed["task_latent"]
    columns = torch.cat([task.unsqueeze(1).expand(-1, 64, -1), col_landmarks], dim=-1)
    col_gates = 1.0 + 0.25 * torch.tanh(encoder.col_gate_head(columns).squeeze(-1))
    out = torch.cat([
        dataset_features,
        task,
        routed["dist_latent"],
        torch.stack([routed["gates"][k] for k in CANONICAL_GATE_KEYS_14], dim=-1),
        routed["landmark_pred"],
        routed["bb_pred"],
        col_gates,
    ], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def packet_global_v0(encoding: Mapping[str, object]) -> tuple[float, ...]:
    """24 fit-row globals. Column arrays are ignored. Query keys are refused.

    Error text names the field and does not repeat source values or extrema.
    """
    if not isinstance(encoding, Mapping):
        raise ValueError("packet_global_v0 requires an encode_table mapping")
    for key in QUERY_KEYS:
        if key in encoding:
            raise ValueError(f"packet_global_v0 refuses {key}")
    level = encoding.get("level")
    if level not in LEVEL_CODE:
        raise ValueError("packet_global_v0 requires level high or ultra")
    n_train = _count("n_train", encoding.get("n_train"))
    p_use = _count("p_use", encoding.get("p_use"))
    gates = encoding.get("gates")
    bundle = encoding.get("target_bundle")
    if not isinstance(gates, Mapping) or set(gates) != set(CANONICAL_GATE_KEYS_14):
        raise ValueError("packet_global_v0 requires the 14 canonical gates")
    if not isinstance(bundle, Mapping):
        raise ValueError("packet_global_v0 requires target_bundle")
    for name in _BUNDLE_FIELDS:
        if name not in bundle:
            raise ValueError(f"packet_global_v0 requires target_bundle.{name}")
    trees = _sequence("target_bundle.trees", bundle["trees"])
    pairs = _sequence("target_bundle.pairs", bundle["pairs"])
    values = [_finite(f"gates.{key}", gates[key]) for key in CANONICAL_GATE_KEYS_14]
    values.extend(
        _finite(f"target_bundle.{name}", bundle[name])
        for name in ("residual_std", "noise_scale", "class_prior", "y_min", "y_max")
    )
    values.extend((
        math.log(n_train),
        float(p_use),
        float(len(trees)),
        float(len(pairs)),
        LEVEL_CODE[level],
    ))
    if len(values) != PACKET_GLOBAL_WIDTH or len(PACKET_GLOBAL_FIELDS) != PACKET_GLOBAL_WIDTH:
        raise ValueError("packet_global_v0 width drifted")
    return tuple(values)


def _validate_encoder_inputs(encoder, dataset_features, col_landmarks) -> None:
    # Same refusals as DopeEncoder629.forward. Kept here so ultra does not add a level argument.
    if not isinstance(dataset_features, torch.Tensor) or not isinstance(col_landmarks, torch.Tensor):
        raise ValueError("encoder inputs must be tensors")
    if dataset_features.ndim != 2 or dataset_features.shape[1] != 256 or dataset_features.shape[0] < 1:
        raise ValueError("dataset_features requires shape [B, 256] with B >= 1")
    if (col_landmarks.ndim != 3 or col_landmarks.shape[1:] != (64, 17)
            or col_landmarks.shape[0] != dataset_features.shape[0]):
        raise ValueError("col_landmarks requires matching shape [B, 64, 17]")
    if (dataset_features.dtype != torch.float32 or col_landmarks.dtype != torch.float32
            or dataset_features.layout != torch.strided or col_landmarks.layout != torch.strided):
        raise ValueError("encoder inputs require dense FP32 tensors")
    device = dataset_features.device
    if device.type not in ("cpu", "cuda") or col_landmarks.device != device:
        raise ValueError("encoder inputs require the same CPU or CUDA device")
    if any(p.dtype != torch.float32 or p.device != device for p in encoder.parameters()):
        raise ValueError("encoder parameters require FP32 on the input device")
    if torch.is_autocast_enabled(device.type):
        raise ValueError("encoder requires FP32 with autocast disabled")


def _count(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"packet_global_v0: {name} must be an integer count >= 1")
    try:
        n = operator.index(value)
    except TypeError:
        raise ValueError(f"packet_global_v0: {name} must be an integer count >= 1") from None
    if n < 1:
        raise ValueError(f"packet_global_v0: {name} must be an integer count >= 1")
    return n


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"packet_global_v0: {name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"packet_global_v0: {name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"packet_global_v0: {name} must be a finite number")
    return number


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"packet_global_v0 requires {name}")
    return value
