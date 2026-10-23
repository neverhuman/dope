"""Restricted attention/SiLU graph shared with the native reference implementation."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.width = heads, width
        self.q, self.k, self.v, self.out = (nn.Linear(width, width) for _ in range(4))

    def forward(self, query, context):
        def project(layer, value):
            return layer(value).reshape(*value.shape[:-1], self.heads,
                                        self.width//self.heads).transpose(-3, -2)
        q, k, v = project(self.q, query), project(self.k, context), project(self.v, context)
        probabilities = torch.softmax((q @ k.transpose(-1, -2)).float()
                                      / math.sqrt(self.width//self.heads), dim=-1).to(v.dtype)
        mixed = (probabilities @ v).transpose(-3, -2).contiguous().reshape(*query.shape[:-1], self.width)
        return self.out(mixed)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_ff = (nn.LayerNorm(width, eps=1e-5) for _ in range(3))
        self.attention = Attention(width, heads)
        self.up, self.down = nn.Linear(width, width*4), nn.Linear(width*4, width)

    def forward(self, query, context):
        value = query + self.attention(self.norm_q(query), self.norm_kv(context))
        return value + self.down(F.silu(self.up(self.norm_ff(value))))


class Induced(nn.Module):
    def __init__(self, width, heads, slots):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(slots, width)*.02)
        self.read, self.write = Block(width, heads), Block(width, heads)

    def forward(self, rows):
        latent = self.read(self.slots, rows)
        return self.write(rows, latent)


def binding_vectors(slots, device):
    slots = torch.as_tensor(slots, dtype=torch.float32, device=device)
    frequencies = torch.tensor([1., .1, .01, .001], device=device)
    angles = slots[:, None]*frequencies[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Tokenizer(nn.Module):
    def __init__(self, width, categories):
        super().__init__()
        self.values = nn.Linear(8, width)
        self.types = nn.Embedding(2, width)
        self.categories = nn.Embedding(categories, width)
        self.bindings = nn.Linear(8, width)

    def forward(self, values, observed, columns):
        device = values.device
        kinds = torch.tensor([c["kind"] == "categorical" for c in columns], device=device)
        numeric = values.clamp(-8, 8)
        tail = (values.sign()*torch.log1p(values.abs())).clamp(-8, 8)
        channels = torch.stack((numeric, tail, observed.float(), (~observed).float(),
                                (values == 0).float(), (values == values.round()).float(),
                                kinds.expand_as(values).float(), (values == 1).float()), dim=-1)
        token = self.values(channels)
        token = token + self.types(kinds.long()) + self.bindings(binding_vectors(
            [c["slot"] for c in columns], device))
        indices = torch.where(kinds, values, torch.zeros_like(values)).long()
        return token + self.categories(indices)*kinds[:, None]
