"""Mixed diffusion from selected code + charged positional schema; no context argument."""
import math
import torch
from torch import nn
from .blocks import Block, Tokenizer, binding_vectors
from .contracts import BLOCKS, LEVELS, require


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.validate()
        w, h = config.width, config.heads
        self.tokenizer = Tokenizer(w, config.max_categories)
        self.conditions = nn.ModuleList(nn.Linear(b, config.condition_slots*w) for b in BLOCKS)
        self.time = nn.Linear(8, w)
        self.self_blocks = nn.ModuleList(Block(w, h) for _ in range(config.decoder_blocks))
        self.cross_blocks = nn.ModuleList(Block(w, h) for _ in range(config.decoder_blocks))
        self.norm = nn.LayerNorm(w)
        self.numeric, self.categorical, self.missing = nn.Linear(w, 1), nn.Linear(w, config.max_categories), nn.Linear(w, 1)

    def forward(self, code, schema, noisy_values, noisy_observed, time):
        require(code.ndim == 1 and code.numel() in LEVELS.values(), "decoder_code_layout")
        require(noisy_values.ndim == 2 and noisy_values.shape == noisy_observed.shape
                and noisy_values.shape[1] == len(schema["columns"])
                and len(schema["columns"]) <= self.config.max_features+1
                and noisy_values.shape[0]*len(schema['columns'])**2*self.config.heads<=33554432
                and noisy_values.numel()*self.config.width<=33554432,
                "decoder_input_shape")
        conditions, start = [], 0
        for width, projection in zip(BLOCKS, self.conditions):
            if start + width > code.numel(): break
            conditions.append(projection(code[start:start+width]).reshape(self.config.condition_slots, self.config.width))
            start += width
        condition = torch.cat(conditions)
        token = self.tokenizer(noisy_values, noisy_observed, schema["columns"])
        time = torch.as_tensor(time, dtype=torch.float32, device=code.device).reshape(1)
        phase = time[:, None]*torch.tensor([1., 10., 100., 1000.], device=code.device)[None, :]
        token = token + self.time(torch.cat((phase.sin(), phase.cos()), dim=-1))
        for own, cross in zip(self.self_blocks, self.cross_blocks):
            token = own(token, token)
            token = cross(token, condition)
        token = self.norm(token)
        logits = self.categorical(token).float()
        allowed = torch.arange(self.config.max_categories, device=code.device)
        for j, column in enumerate(schema["columns"]):
            if column["kind"] == "categorical":
                mask = (allowed >= 2) & (allowed < column["categories"])
                logits[:, j] = logits[:, j].masked_fill(~mask, -1e9)
        return self.numeric(token).squeeze(-1).float(), logits, self.missing(token).squeeze(-1).float()


def alpha(time):
    return torch.cos((time+.008)/1.008*math.pi/2).square().clamp(.001, .999)


@torch.no_grad()
def sample(decoder, code, schema, rows, seed, *, steps=None, categorical_schedule='reveal-v1'):
    steps = decoder.config.diffusion_steps if steps is None else steps
    require(type(rows) is int and 1 <= rows <= 1000000 and steps in (8, 16, 32, 64)
            and rows*len(schema["columns"]) <= 4000000, "sampler_allocation_unsupported")
    require(categorical_schedule in ('reveal-v1', 'redraw-v1'), 'unknown_categorical_schedule')
    chunk=min(256,33554432//(len(schema['columns'])**2*decoder.config.heads),
              33554432//(len(schema['columns'])*decoder.config.width))
    require(chunk>=1,'sampler_graph_allocation_unsupported')
    outputs=[_sample_chunk(decoder,code,schema,min(chunk,rows-start),seed+start,steps,categorical_schedule)
             for start in range(0,rows,chunk)]
    return torch.cat([v for v,_ in outputs]),torch.cat([o for _,o in outputs])


@torch.no_grad()
def _sample_chunk(decoder,code,schema,rows,seed,steps,categorical_schedule='reveal-v1'):
    generator = torch.Generator(device=code.device).manual_seed(seed)
    count = len(schema["columns"])
    values = torch.randn((rows, count), generator=generator, device=code.device)
    kinds = torch.tensor([c["kind"] == "categorical" for c in schema["columns"]], device=code.device)
    values[:, kinds] = 0
    observed = torch.zeros_like(values, dtype=torch.bool)
    revealed = torch.zeros_like(observed)
    # Fixed DDIM numerical path and masked categorical reveal; all RNG comes from the seed.
    for i in range(steps, 0, -1):
        time = torch.tensor(i/steps, device=code.device)
        a = alpha(time)
        visible = observed & (~kinds | revealed) if categorical_schedule == 'reveal-v1' else observed
        epsilon, logits, missing = decoder(code, schema, values, visible, time)
        categorical_values = values.clone()
        x0 = ((values-(1-a).sqrt()*epsilon)/a.sqrt()).clamp(-16, 16)
        previous = alpha(torch.tensor((i-1)/steps, device=code.device))
        values = previous.sqrt()*x0 + (1-previous).sqrt()*epsilon if i > 1 else x0
        probabilities = logits.softmax(-1)
        categories = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1,
                                       generator=generator).reshape(rows, count).float()
        if categorical_schedule == 'reveal-v1' and kinds.any():
            newly = (torch.rand(values.shape, generator=generator, device=code.device) < 1/i) & ~revealed & kinds
            revealed |= newly
            values[:, kinds] = torch.where(newly[:, kinds], categories[:, kinds], categorical_values[:, kinds])
        else:
            values[:, kinds] = categories[:, kinds]
        observed = torch.rand(values.shape, generator=generator, device=code.device) > torch.sigmoid(missing)
    target = schema.get("target_index")
    if target is not None: observed[:, target] = True
    return values, observed
