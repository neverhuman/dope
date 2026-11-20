"""Versioned transfer-encoder candidates.

Candidate A consumes a frozen full629 vector. Candidate B keeps that width
and receives a new identity. Candidate C reads the raw-table hidden width.
None of them loads or rewrites the released encoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .contracts import ModelConfig
from .encoder import DOPE_ACTIVE_ENCODER_PARAMETERS, DOPE_FEATURES_629_DIM, DopeEncoder629

SCHEMA = "dope-transfer-encoding/v1"
GLOBAL_WIDTH = 512
COLUMN_WIDTH = 128
TASK_SLOTS = 8
RESIDUAL_ALPHA = 0.1
_DENIED_IDENTITIES = ("Dope512", "DopeEncoder629", "dope-encoder-629", "released-v1")


class TransferEncoderError(ValueError):
    """Static refusal. The message is a code, not a source value."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    identity: str
    graph: str
    input_name: str
    input_width: int
    readout_name: str
    readout_width: int
    continues_released_graph: bool
    loads_released_weights: bool


def _candidates() -> dict[str, Candidate]:
    raw_width = ModelConfig().validate().width
    return {
        "A": Candidate(
            "A",
            "dope.transfer.candidate-a.consumer/v1",
            "released-consumer",
            "full629",
            DOPE_FEATURES_629_DIM,
            "transfer512",
            GLOBAL_WIDTH,
            False,
            False,
        ),
        "B": Candidate(
            "B",
            "dope.transfer.candidate-b.continued-mlc/v1",
            "continued-mlc",
            "full629",
            DOPE_FEATURES_629_DIM,
            "transfer512",
            GLOBAL_WIDTH,
            True,
            False,
        ),
        "C": Candidate(
            "C",
            "dope.transfer.candidate-c.raw-table/v1",
            "raw-table-transfer",
            "raw-table-hidden",
            raw_width,
            "transfer512",
            GLOBAL_WIDTH,
            False,
            False,
        ),
    }


def candidate(candidate_id: str) -> Candidate:
    try:
        return _candidates()[candidate_id]
    except KeyError:
        raise TransferEncoderError("transfer_encoder_candidate") from None


def same_abi(left_identity: str, right_identity: str, left_width: int, right_width: int) -> bool:
    """Equal width is not an ABI match."""
    if type(left_width) is not int or type(right_width) is not int:
        raise TransferEncoderError("transfer_encoder_shape")
    return left_identity == right_identity and left_width == right_width


def task_slot(task_id: str) -> int:
    if not isinstance(task_id, str) or not task_id or len(task_id) > 64:
        raise TransferEncoderError("transfer_encoder_identity")
    total = 0
    for character in task_id:
        total = (total * 33 + ord(character)) % TASK_SLOTS
    return total


def limits() -> dict[str, int | str | bool]:
    """Counts and caps taken from the live modules, not from a comment."""
    config = ModelConfig().validate()
    released = DopeEncoder629()
    parameter_count = sum(parameter.numel() for parameter in released.parameters())
    tensor_count = len(released.state_dict())
    if parameter_count != DOPE_ACTIVE_ENCODER_PARAMETERS or tensor_count != 40:
        raise TransferEncoderError("transfer_encoder_limit")
    return {
        "schema": SCHEMA,
        "raw_table_width": config.width,
        "raw_table_heads": config.heads,
        "feature_blocks": config.feature_blocks,
        "inducing_blocks": config.inducing_blocks,
        "target_blocks": config.target_blocks,
        "max_features": config.max_features,
        "max_context": config.max_context,
        "cell_cap": config.cell_cap,
        "global_width": GLOBAL_WIDTH,
        "column_token_width": COLUMN_WIDTH,
        "full629_width": DOPE_FEATURES_629_DIM,
        "released_architecture_parameters": parameter_count,
        "released_architecture_tensors": tensor_count,
        "released_weights_loaded": False,
    }


class TransferReadout(nn.Module):
    """transfer512 head. Column tokens are an optional ablation."""

    def __init__(self, input_width: int, *, column_tokens: bool) -> None:
        super().__init__()
        if type(input_width) is not int or input_width < 1:
            raise TransferEncoderError("transfer_encoder_shape")
        self.input_width = input_width
        self.hidden = nn.Linear(input_width, GLOBAL_WIDTH)
        self.task = nn.Embedding(TASK_SLOTS, GLOBAL_WIDTH)
        self.column = nn.Linear(COLUMN_WIDTH, GLOBAL_WIDTH) if column_tokens else None
        with torch.no_grad():
            slots = torch.linspace(-1.0, 1.0, TASK_SLOTS, dtype=self.task.weight.dtype)
            self.task.weight.copy_(slots.unsqueeze(1).expand_as(self.task.weight))

    def forward(
        self,
        features: torch.Tensor,
        task_index: torch.Tensor,
        column_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        # Detach the incoming representation so a released encoder cannot receive this gradient.
        projected = self.hidden(features.detach()) + self.task(task_index)
        if self.column is None:
            if column_tokens is not None:
                raise TransferEncoderError("transfer_encoder_shape")
            return projected
        if column_tokens is None:
            raise TransferEncoderError("transfer_encoder_shape")
        return projected + self.column(column_tokens.detach()).mean(dim=1)


class ResidualControl(nn.Module):
    """s = s0 + alpha * r(h), with r initialized at zero and alpha fixed and nonzero."""

    def __init__(self) -> None:
        super().__init__()
        self.correction = nn.Linear(GLOBAL_WIDTH, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, incumbent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if (
            not torch.is_tensor(incumbent)
            or not torch.is_tensor(hidden)
            or incumbent.ndim != 1
            or hidden.ndim != 2
            or hidden.shape[1] != GLOBAL_WIDTH
            or incumbent.shape[0] != hidden.shape[0]
        ):
            raise TransferEncoderError("transfer_encoder_shape")
        if bool((~torch.isfinite(incumbent)).any()) or bool((~torch.isfinite(hidden)).any()):
            raise TransferEncoderError("transfer_encoder_nonfinite")
        correction = self.correction(hidden).squeeze(-1)
        return incumbent.detach() + RESIDUAL_ALPHA * correction


def realize(candidate_id: str, *, column_tokens: bool = False) -> TransferReadout:
    record = candidate(candidate_id)
    if type(column_tokens) is not bool:
        raise TransferEncoderError("transfer_encoder_shape")
    return TransferReadout(record.input_width, column_tokens=column_tokens)


def inventory(module: nn.Module) -> dict[str, object]:
    if not isinstance(module, nn.Module):
        raise TransferEncoderError("transfer_encoder_shape")
    tensors = []
    for name, parameter in module.named_parameters():
        if "correction" in name:
            origin = "residual-zero"
        elif "task" in name:
            origin = "task-conditioning"
        elif "column" in name:
            origin = "column-token"
        else:
            origin = "readout-init"
        tensors.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "count": int(parameter.numel()),
                "origin": origin,
                "trainable": bool(parameter.requires_grad),
            }
        )
    parameter_count = sum(row["count"] for row in tensors)
    return {
        "schema": SCHEMA,
        "parameter_count": parameter_count,
        "trainable_count": sum(row["count"] for row in tensors if row["trainable"]),
        "transferred_from_released": 0,
        "residual_alpha": RESIDUAL_ALPHA if isinstance(module, ResidualControl) else None,
        "residual_alpha_trainable": False,
        "tensors": tensors,
        "limits": limits(),
    }


def encode(
    readout: TransferReadout,
    candidate_id: str,
    features: torch.Tensor,
    *,
    task_id: str,
    claimed_identity: str,
    column_ids: tuple[str, ...] = (),
    column_tokens: torch.Tensor | None = None,
    observed: torch.Tensor | None = None,
) -> dict[str, object]:
    record = candidate(candidate_id)
    _bind_identity(record, claimed_identity)
    _check_features(readout, features)
    ids = _check_columns(readout, features.shape[0], column_ids, column_tokens)
    observed_mask = _check_observed(features.shape[0], ids, observed)
    slot = task_slot(task_id)
    task_index = torch.full((features.shape[0],), slot, dtype=torch.long, device=features.device)
    global_features = readout(features, task_index, column_tokens)
    if global_features.shape != (features.shape[0], GLOBAL_WIDTH):
        raise TransferEncoderError("transfer_encoder_shape")
    if not bool(torch.isfinite(global_features).all()):
        raise TransferEncoderError("transfer_encoder_nonfinite")
    return {
        "schema": SCHEMA,
        "recipe": SCHEMA,
        "identity": record.identity,
        "graph": record.graph,
        "readout": record.readout_name,
        "global": global_features,
        "column_tokens": None if column_tokens is None else column_tokens.detach(),
        "column_ids": ids,
        "observed": observed_mask,
        "task_id": task_id,
    }


def _bind_identity(record: Candidate, claimed_identity: str) -> None:
    if (
        not isinstance(claimed_identity, str)
        or claimed_identity in _DENIED_IDENTITIES
        or claimed_identity != record.identity
    ):
        raise TransferEncoderError("transfer_encoder_alias")


def _check_features(readout: TransferReadout, features: torch.Tensor) -> None:
    if (
        not torch.is_tensor(features)
        or features.ndim != 2
        or features.dtype != torch.float32
        or features.shape[0] < 1
        or features.shape[1] != readout.input_width
        or features.device.type not in ("cpu", "cuda")
        or any(parameter.device != features.device for parameter in readout.parameters())
    ):
        raise TransferEncoderError("transfer_encoder_shape")
    if bool((~torch.isfinite(features)).any()):
        raise TransferEncoderError("transfer_encoder_nonfinite")


def _check_columns(
    readout: TransferReadout,
    batch: int,
    column_ids: tuple[str, ...],
    column_tokens: torch.Tensor | None,
) -> tuple[str, ...]:
    if not isinstance(column_ids, tuple) or any(not isinstance(item, str) or not item for item in column_ids):
        raise TransferEncoderError("transfer_encoder_identity")
    if len(set(column_ids)) != len(column_ids):
        raise TransferEncoderError("transfer_encoder_identity")
    enabled = readout.column is not None
    if not enabled:
        if column_ids or column_tokens is not None:
            raise TransferEncoderError("transfer_encoder_shape")
        return ()
    if len(column_ids) < 1:
        raise TransferEncoderError("transfer_encoder_identity")
    if (
        column_tokens is None
        or not torch.is_tensor(column_tokens)
        or column_tokens.ndim != 3
        or column_tokens.dtype != torch.float32
        or column_tokens.shape[0] != batch
        or column_tokens.shape[2] != COLUMN_WIDTH
    ):
        raise TransferEncoderError("transfer_encoder_shape")
    if column_tokens.shape[1] != len(column_ids):
        raise TransferEncoderError("transfer_encoder_identity")
    if bool((~torch.isfinite(column_tokens)).any()):
        raise TransferEncoderError("transfer_encoder_nonfinite")
    return column_ids


def _check_observed(
    batch: int,
    column_ids: tuple[str, ...],
    observed: torch.Tensor | None,
) -> torch.Tensor | None:
    if observed is None:
        if column_ids:
            raise TransferEncoderError("transfer_encoder_shape")
        return None
    groups = len(column_ids)
    if (
        not torch.is_tensor(observed)
        or observed.dtype != torch.bool
        or observed.shape != (batch, groups)
        or groups < 1
    ):
        raise TransferEncoderError("transfer_encoder_shape")
    return observed.detach()
