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
# TableEncoder feature-attention element cap: rows * features^2 * heads.
ATTENTION_ELEMENT_CAP = 33_554_432
_DENIED_IDENTITIES = ("Dope512", "DopeEncoder629", "dope-encoder-629", "released-v1")
TASK_VOCABULARY = (
    "rank",
    "generate",
    "fit",
    "select",
    "calibrate",
    "distill",
    "retrieve",
    "validate",
)


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
    frozen_backbone: bool


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
            True,
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
    if len(TASK_VOCABULARY) != TASK_SLOTS or type(task_id) is not str:
        raise TransferEncoderError("transfer_encoder_identity")
    try:
        return TASK_VOCABULARY.index(task_id)
    except ValueError:
        raise TransferEncoderError("transfer_encoder_identity") from None


def attention_budget(rows: int, features: int) -> None:
    """Refuse a table outside the live raw-table encoder envelope."""
    if type(rows) is not int or type(features) is not int or rows < 1 or features < 1:
        raise TransferEncoderError("transfer_encoder_shape")
    config = ModelConfig(max_features=256).validate()
    cells = rows * features
    quadratic = cells * features * config.heads
    width_term = cells * config.width
    if (
        features > config.max_features
        or rows > config.max_context
        or cells > config.cell_cap
        or quadratic > ATTENTION_ELEMENT_CAP
        or width_term > ATTENTION_ELEMENT_CAP
    ):
        raise TransferEncoderError("transfer_encoder_limit")


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
        "attention_element_cap": ATTENTION_ELEMENT_CAP,
        "global_width": GLOBAL_WIDTH,
        "column_token_width": COLUMN_WIDTH,
        "full629_width": DOPE_FEATURES_629_DIM,
        "released_architecture_parameters": parameter_count,
        "released_architecture_tensors": tensor_count,
        "released_weights_loaded": False,
    }


class TransferReadout(nn.Module):
    """transfer512 head. Column tokens are an optional ablation."""

    def __init__(self, input_width: int, *, column_tokens: bool, frozen_backbone: bool) -> None:
        super().__init__()
        if type(input_width) is not int or input_width < 1 or type(frozen_backbone) is not bool:
            raise TransferEncoderError("transfer_encoder_shape")
        self.input_width = input_width
        self.frozen_backbone = frozen_backbone
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
        observed: torch.Tensor | None,
    ) -> torch.Tensor:
        incoming = features.detach() if self.frozen_backbone else features
        projected = self.hidden(incoming) + self.task(task_index)
        if self.column is None:
            if column_tokens is not None or observed is not None:
                raise TransferEncoderError("transfer_encoder_shape")
            return projected
        if column_tokens is None or observed is None:
            raise TransferEncoderError("transfer_encoder_shape")
        tokens = column_tokens.detach() if self.frozen_backbone else column_tokens
        return projected + self._pooled_columns(tokens, observed)

    def _pooled_columns(self, column_tokens: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        mask = observed.unsqueeze(-1)
        safe = torch.where(mask, column_tokens, torch.zeros_like(column_tokens))
        projected = self.column(safe)
        masked = torch.where(mask, projected, torch.zeros_like(projected))
        count = observed.sum(dim=1, keepdim=True).to(dtype=masked.dtype)
        numerator = masked.sum(dim=1)
        alive = count > 0
        divided = numerator / torch.where(alive, count, torch.ones_like(count))
        connected = torch.where(torch.isfinite(numerator), numerator, torch.zeros_like(numerator))
        zero = connected.sum(dim=1, keepdim=True) * 0
        return torch.where(alive, divided, zero)


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
        if (
            not incumbent.is_floating_point()
            or not hidden.is_floating_point()
            or incumbent.dtype != hidden.dtype
            or hidden.dtype != self.correction.weight.dtype
        ):
            raise TransferEncoderError("transfer_encoder_dtype")
        if (
            incumbent.device != hidden.device
            or hidden.device.type not in ("cpu", "cuda")
            or any(param.device != hidden.device for param in self.parameters())
        ):
            raise TransferEncoderError("transfer_encoder_device")
        if bool((~torch.isfinite(incumbent)).any()) or bool((~torch.isfinite(hidden)).any()):
            raise TransferEncoderError("transfer_encoder_nonfinite")
        correction = self.correction(hidden).squeeze(-1)
        return incumbent.detach() + RESIDUAL_ALPHA * correction


def realize(candidate_id: str, *, column_tokens: bool = False) -> TransferReadout:
    record = candidate(candidate_id)
    if type(column_tokens) is not bool:
        raise TransferEncoderError("transfer_encoder_shape")
    return TransferReadout(
        record.input_width,
        column_tokens=column_tokens,
        frozen_backbone=record.frozen_backbone,
    )


class FrozenBackboneConsumer(nn.Module):
    """Candidate A wrapper. Outer train() cannot reopen the released encoder."""

    def __init__(self, encoder: DopeEncoder629, readout: TransferReadout) -> None:
        super().__init__()
        if not isinstance(encoder, DopeEncoder629) or not isinstance(readout, TransferReadout):
            raise TransferEncoderError("transfer_encoder_shape")
        if not readout.frozen_backbone:
            raise TransferEncoderError("transfer_encoder_candidate")
        self.encoder = encoder
        self.readout = readout
        self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_encoder()
        return self

    def forward(
        self,
        dataset_features: torch.Tensor,
        col_landmarks: torch.Tensor,
        *,
        task_id: str,
    ) -> torch.Tensor:
        self._freeze_encoder()
        with torch.no_grad():
            released = self.encoder(dataset_features, col_landmarks)
        slot = task_slot(task_id)
        index = torch.full((released.shape[0],), slot, dtype=torch.long, device=released.device)
        return self.readout(released.detach(), index, None, None)


def inventory(module: nn.Module) -> dict[str, object]:
    if not isinstance(module, nn.Module):
        raise TransferEncoderError("transfer_encoder_shape")
    is_consumer = isinstance(module, FrozenBackboneConsumer)
    tensors = []
    for name, parameter in module.named_parameters():
        if is_consumer and name.startswith("encoder."):
            origin = "released-frozen"
        elif "correction" in name:
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
    transferred_params = sum(p.numel() for p in module.encoder.parameters()) if is_consumer else 0
    transferred_tensors = sum(1 for _ in module.encoder.state_dict()) if is_consumer else 0
    return {
        "schema": SCHEMA,
        "parameter_count": parameter_count,
        "trainable_count": sum(row["count"] for row in tensors if row["trainable"]),
        "transferred_from_released": transferred_params,
        "transferred_tensors": transferred_tensors,
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
    table_rows: int | None = None,
) -> dict[str, object]:
    record = candidate(candidate_id)
    _bind_identity(record, claimed_identity)
    _check_features(readout, features)
    ids = _check_columns(readout, features, column_ids, column_tokens)
    observed_mask = _check_observed(features, ids, observed)
    _check_observed_finite(column_tokens, observed_mask)
    if ids:
        if table_rows is None:
            raise TransferEncoderError("transfer_encoder_shape")
        attention_budget(table_rows, len(ids))
    slot = task_slot(task_id)
    task_index = torch.full((features.shape[0],), slot, dtype=torch.long, device=features.device)
    global_features = readout(features, task_index, column_tokens, observed_mask)
    if global_features.shape != (features.shape[0], GLOBAL_WIDTH):
        raise TransferEncoderError("transfer_encoder_shape")
    if not bool(torch.isfinite(global_features).all()):
        raise TransferEncoderError("transfer_encoder_nonfinite")
    stored_tokens = None
    if column_tokens is not None:
        stored_tokens = column_tokens.masked_fill(~observed_mask.unsqueeze(-1), 0).detach()
    return {
        "schema": SCHEMA,
        "recipe": SCHEMA,
        "identity": record.identity,
        "graph": record.graph,
        "readout": record.readout_name,
        "global": global_features,
        "column_tokens": stored_tokens,
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
        or features.shape[0] < 1
        or features.shape[1] != readout.input_width
    ):
        raise TransferEncoderError("transfer_encoder_shape")
    if features.dtype != torch.float32:
        raise TransferEncoderError("transfer_encoder_dtype")
    if features.device.type not in ("cpu", "cuda") or any(
        parameter.device != features.device for parameter in readout.parameters()
    ):
        raise TransferEncoderError("transfer_encoder_device")
    if bool((~torch.isfinite(features)).any()):
        raise TransferEncoderError("transfer_encoder_nonfinite")


def _check_columns(
    readout: TransferReadout,
    features: torch.Tensor,
    column_ids: tuple[str, ...],
    column_tokens: torch.Tensor | None,
) -> tuple[str, ...]:
    if not isinstance(column_ids, tuple) or any(not isinstance(item, str) or not item for item in column_ids):
        raise TransferEncoderError("transfer_encoder_identity")
    if len(set(column_ids)) != len(column_ids):
        raise TransferEncoderError("transfer_encoder_identity")
    if readout.column is None:
        if column_ids or column_tokens is not None:
            raise TransferEncoderError("transfer_encoder_shape")
        return ()
    if len(column_ids) < 1:
        raise TransferEncoderError("transfer_encoder_identity")
    if len(column_ids) > ModelConfig(max_features=256).validate().max_features:
        raise TransferEncoderError("transfer_encoder_limit")
    if (
        column_tokens is None
        or not torch.is_tensor(column_tokens)
        or column_tokens.ndim != 3
        or column_tokens.shape[0] != features.shape[0]
        or column_tokens.shape[2] != COLUMN_WIDTH
    ):
        raise TransferEncoderError("transfer_encoder_shape")
    if not column_tokens.is_floating_point() or column_tokens.dtype != features.dtype:
        raise TransferEncoderError("transfer_encoder_dtype")
    if column_tokens.device != features.device:
        raise TransferEncoderError("transfer_encoder_device")
    if column_tokens.shape[1] != len(column_ids):
        raise TransferEncoderError("transfer_encoder_identity")
    return column_ids


def _check_observed_finite(
    column_tokens: torch.Tensor | None,
    observed: torch.Tensor | None,
) -> None:
    if column_tokens is None:
        return
    active = column_tokens[observed]
    if active.numel() > 0 and bool((~torch.isfinite(active)).any()):
        raise TransferEncoderError("transfer_encoder_nonfinite")


def _check_observed(
    features: torch.Tensor,
    column_ids: tuple[str, ...],
    observed: torch.Tensor | None,
) -> torch.Tensor | None:
    if observed is None:
        if column_ids:
            raise TransferEncoderError("transfer_encoder_shape")
        return None
    groups = len(column_ids)
    if not torch.is_tensor(observed) or observed.shape != (features.shape[0], groups) or groups < 1:
        raise TransferEncoderError("transfer_encoder_shape")
    if observed.dtype != torch.bool:
        raise TransferEncoderError("transfer_encoder_dtype")
    if observed.device != features.device:
        raise TransferEncoderError("transfer_encoder_device")
    return observed.detach()
