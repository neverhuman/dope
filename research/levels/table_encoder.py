"""One shared inference path with separate target-free and task-conditioned readouts."""
import torch
from torch import nn
from .blocks import Block, Induced, Tokenizer
from .contracts import BLOCKS, LEVELS, require


class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.validate()
        w, h = config.width, config.heads
        self.tokenizer = Tokenizer(w, config.max_categories)
        self.features = nn.ModuleList(Block(w, h) for _ in range(config.feature_blocks))
        self.rows = nn.ModuleList(Induced(w, h, config.inducing_slots) for _ in range(config.inducing_blocks))
        self.summaries = nn.Parameter(torch.randn(config.summary_slots, w)*.02)
        self.pool = nn.ModuleList(Block(w, h) for _ in range(2))
        self.target = nn.ModuleList(Block(w, h) for _ in range(config.target_blocks))
        self.distribution_norm, self.task_norm = nn.LayerNorm(w), nn.LayerNorm(w)
        self.distribution_heads = nn.ModuleList(nn.Linear(w, b//2) for b in BLOCKS)
        self.task_heads = nn.ModuleList(nn.Linear(w, b//2) for b in BLOCKS)

    def forward(self, values, observed, schema, *, target_included=True, through="high"):
        require(through in LEVELS and values.ndim == 2 and values.shape == observed.shape,
                "encoder_input_shape")
        columns = schema["columns"]
        require(values.shape[0]*(len(columns)-1)**2*self.config.heads<=33554432
                and values.numel()*self.config.width<=33554432, 'encoder_attention_allocation_unsupported')
        require(len(columns) == values.shape[1] and values.shape[0] <= self.config.max_context
                and len(columns)-1 <= self.config.max_features
                and values.shape[0]*(len(columns)-1) <= self.config.cell_cap, "encoder_shape_unsupported")
        feature = self.tokenizer(values[:, :-1], observed[:, :-1], columns[:-1])
        # Feature interactions alternate with bounded row-set aggregation.
        next_rows = 0
        for i, block in enumerate(self.features):
            feature = block(feature, feature)
            if i % 2 == 1 and next_rows < len(self.rows):
                previous = feature.mean(dim=1)
                mixed = self.rows[next_rows](previous)
                feature = feature + (mixed-previous)[:, None, :]
                next_rows += 1
        rows = feature.mean(dim=1)
        for block in self.rows[next_rows:]:
            rows = block(rows)
        summary = self.summaries
        for block in self.pool:
            summary = block(summary, rows)
        distribution = self.distribution_norm(summary.mean(dim=0))
        if target_included:
            target = rows + self.tokenizer(values[:, -1:], observed[:, -1:], columns[-1:]).squeeze(1)
            for block in self.target:
                target = block(target, rows)
            task = self.task_norm(target.mean(dim=0))
        else:
            task = None
        blocks = []
        for i, width in enumerate(BLOCKS):
            d = self.distribution_heads[i](distribution)
            t = self.task_heads[i](task) if task is not None else torch.zeros_like(d)
            blocks.append(torch.cat((d, t)))
            if sum(BLOCKS[:i+1]) == LEVELS[through]:
                break
        return torch.cat(blocks)


def quantized_code(code):
    """Fixed FP16 conversion, no high-derived scales; straight-through gradients for training."""
    quantized = code.to(torch.float16).to(torch.float32)
    require(torch.isfinite(quantized).all().item(), "code_quantization_nonfinite")
    return code + (quantized-code).detach()
