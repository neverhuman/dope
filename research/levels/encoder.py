"""Pinned FP32/High encoder: [B, 256] + [B, 64, 17] -> [B, 629]."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch
from torch import nn

from .multilevel_model import LandmarkHyperRouter, MultiLevelLandmarkModel

DOPE_FEATURES_629_DIM = 629
DOPE_ACTIVE_ENCODER_TENSORS = 40
DOPE_ACTIVE_ENCODER_PARAMETERS = 353_095
PINNED_MULTITASK_SAFETENSORS_SHA256 = "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c"
CANONICAL_GATE_KEYS_14 = (
    "asinh_atan_w", "copula_retention", "het_scale_w", "hinge_w",
    "monotone_w", "neural_w", "noise_scale", "pair_w", "periodic_fract_w",
    "regime_sharpness", "rff_w", "ridge_alpha", "row_summary_w", "tree_w",
)


class DopeEncoder629(nn.Module):
    """Router and column-gate head only; load verified weights with from_pretrained.

    Inputs must already be padded, FP32, on the model's CPU or CUDA device.
    Nonfinite results become zero, exactly as in Micro's Stage 1A computation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.router = LandmarkHyperRouter(in_dim=256)
        self.col_gate_head = nn.Linear(145, 1)

    def forward(self, dataset_features: torch.Tensor, col_landmarks: torch.Tensor) -> torch.Tensor:
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
        if any(p.dtype != torch.float32 or p.device != device for p in self.parameters()):
            raise ValueError("encoder parameters require FP32 on the input device")
        if torch.is_autocast_enabled(device.type):
            raise ValueError("encoder requires FP32 with autocast disabled")

        routed = self.router(dataset_features, level="high")
        task = routed["task_latent"]
        columns = torch.cat([task.unsqueeze(1).expand(-1, 64, -1), col_landmarks], dim=-1)
        col_gates = 1.0 + 0.25 * torch.tanh(self.col_gate_head(columns).squeeze(-1))
        out = torch.cat([
            dataset_features, task, routed["dist_latent"],
            torch.stack([routed["gates"][k] for k in CANONICAL_GATE_KEYS_14], dim=-1),
            routed["landmark_pred"], routed["bb_pred"], col_gates,
        ], dim=-1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        expected_sha256: str = PINNED_MULTITASK_SAFETENSORS_SHA256,
        device: str | torch.device = "cpu",
    ) -> "DopeEncoder629":
        """Verify bytes before SafeTensors parsing; validate the full checkpoint strictly.

        Defaults to the released multitask digest. An alternate full checkpoint
        needs an explicit digest and separate downstream qualification. No network
        access or pickle loading is performed, regardless of the file extension.
        """
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("expected_sha256 requires a SHA-256 hex digest")
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
            raise ValueError("checkpoint SHA-256 mismatch")
        from safetensors.torch import load

        try:
            state = load(raw)
            full = MultiLevelLandmarkModel()
            schema = full.state_dict()
            if state.keys() != schema.keys() or any(
                t.shape != schema[k].shape or t.dtype != torch.float32 or not torch.isfinite(t).all()
                for k, t in state.items()
            ):
                raise ValueError("incompatible checkpoint")
            full.load_state_dict(state, strict=True)
        except Exception:
            raise ValueError("checkpoint must match the full FP32 model schema") from None
        enc = cls()
        enc.load_state_dict({
            k: state[k if k.startswith("router.") else "target_net." + k]
            for k in enc.state_dict()
        }, strict=True)
        requested_device = torch.device(device)
        if requested_device.type not in ("cpu", "cuda"):
            raise ValueError("encoder requires a CPU or CUDA device")
        return enc.to(device=requested_device, dtype=torch.float32).requires_grad_(False).eval()
