"""Research encoder/decoder composition. No product imports or release authority."""
import torch
from torch import nn
from .contracts import LEVELS, require
from .table_encoder import TableEncoder
from .decoder import Decoder


class SharedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.validate()
        self.encoder, self.decoder = TableEncoder(config), Decoder(config)
        self.profiles = nn.ModuleDict({level: nn.Linear(width, 23) for level, width in LEVELS.items()})

    def phase(self, phase):
        require(phase in ("ordinary", "ultra-residual"), "unknown_training_phase")
        for parameter in self.parameters(): parameter.requires_grad_(phase == "ordinary")
        if phase == "ultra-residual":
            for module in (self.encoder.distribution_heads[3], self.encoder.task_heads[3],
                           self.decoder.conditions[3], self.profiles["ultra"]):
                for parameter in module.parameters(): parameter.requires_grad_(True)
        return self


def tensors(prepared, device):
    return (torch.tensor(prepared.values, dtype=torch.float32, device=device),
            torch.tensor(prepared.observed, dtype=torch.bool, device=device))
