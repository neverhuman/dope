"""One shared inference path with separate target-free and task-conditioned readouts."""
import torch
from torch import nn
from .blocks import Block, Induced, Tokenizer
from .contracts import block_widths, level_widths, MATCHED_GRAPH, require


class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.validate()
        w, h = config.width, config.heads
        self.tokenizer = Tokenizer(w, config.max_categories)
        self.features = nn.ModuleList(Block(w, h) for _ in range(config.feature_blocks))
        self.rows = nn.ModuleList(Induced(w, h, config.inducing_slots) for _ in range(config.inducing_blocks))
        slots = config.readout_slots() if config.slot_readout() else config.summary_slots
        self.summaries = nn.Parameter(torch.randn(slots, w)*.02)
        if config.slot_readout():
            self.residual_summaries = nn.Parameter(torch.randn(slots, w)*.02)
        self.pool = nn.ModuleList(Block(w, h) for _ in range(2))
        self.target = nn.ModuleList(Block(w, h) for _ in range(config.target_blocks))
        self.distribution_norm, self.task_norm = nn.LayerNorm(w), nn.LayerNorm(w)
        self.distribution_heads = nn.ModuleList(nn.Linear(w*config.readout_slots(), b//2)
                                              for b in block_widths(config.layout))
        self.task_heads = nn.ModuleList(nn.Linear(w*config.readout_slots(), b//2)
                                      for b in block_widths(config.layout))

    def forward(self, values, observed, schema, *, target_included=True, through="high"):
        widths = level_widths(self.config.layout)
        require(through in widths and values.ndim == 2 and values.shape == observed.shape,
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
        distribution = self.readout(summary, self.distribution_norm)
        if target_included:
            target = rows + self.tokenizer(values[:, -1:], observed[:, -1:], columns[-1:]).squeeze(1)
            for block in self.target:
                target = block(target, rows)
            if self.config.slot_readout():
                task_summary = self.summaries
                for block in self.pool: task_summary = block(task_summary, target)
                task = self.readout(task_summary, self.task_norm)
            else:
                task = self.task_norm(target.mean(dim=0))
        else:
            task = None
        blocks = []
        blocks_widths = block_widths(self.config.layout)
        for i, width in enumerate(blocks_widths):
            if i == 3 and self.config.slot_readout():
                residual = self.residual_summaries
                for block in self.pool: residual = block(residual, rows)
                distribution = self.readout(residual, self.distribution_norm)
                if target_included:
                    residual = self.residual_summaries
                    for block in self.pool: residual = block(residual, target)
                    task = self.readout(residual, self.task_norm)
            d = self.distribution_heads[i](distribution)
            t = self.task_heads[i](task) if task is not None else torch.zeros_like(d)
            blocks.append(torch.cat((d, t)))
            if sum(blocks_widths[:i+1]) == widths[through]:
                break
        return torch.cat(blocks)

    def readout(self, summary, norm):
        if not self.config.slot_readout(): return norm(summary.mean(dim=0))
        result = norm(summary)
        if self.config.graph == MATCHED_GRAPH:
            result = result.mean(dim=0, keepdim=True).expand_as(result)
        return result.reshape(-1)


def quantized_code(code):
    """Fixed FP16 conversion, no high-derived scales; straight-through gradients for training."""
    quantized = code.to(torch.float16).to(torch.float32)
    require(torch.isfinite(quantized).all().item(), "code_quantization_nonfinite")
    return code + (quantized-code).detach()
