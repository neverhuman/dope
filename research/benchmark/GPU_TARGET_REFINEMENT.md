# Bounded DOPE GPU target refinement

The `gpu-research-training` Cargo feature enables four explicit research
profiles for the existing compact neural residual target. It includes
`gpu-training` and requires the same CUDA 12.8 / libtorch 2.7 environment.
Set `DOPE_RESEARCH_TARGET_PROFILE` to exactly one of:

| Profile | Maximum selected features | Hidden units | AdamW steps |
|---|---:|---:|---:|
| `features12_steps512` | 12 | 16 | 512 |
| `features12_steps2048` | 12 | 16 | 2048 |
| `features24_steps512` | 24 | 16 | 512 |
| `features24_steps2048` | 24 | 16 | 2048 |

The learning rate remains 0.002. The trained hidden basis still uses the
existing regularized readout, artifact operator and decoder dimensions.
Unknown or non-Unicode profiles fail with a generic error. Without the
explicit feature and profile, the production training settings remain
12 selected features, a width clamped to 4–12, and 96 steps.

The trigger is the measured California validation utility gap in the complete
matched neural and TabPC pilot panels. This is architecture research using
only the existing official-training-derived Adult, California and News
workers. It cannot select the final global family or certify a product while
required PTF/MFS gates remain incomplete.

Before any research fit, freeze the exact source patch and binary, dependency
identities, worker/projection hashes, four-profile grid, fit seeds 11 and 23,
three sample seeds 101/211/307, and `n`/`4n` validation schedule. The proposed
24-fit ceiling is 14,400 GPU seconds, with 600 seconds per fit and 16 GiB of
GPU memory. This spend is reported separately from final per-cell tuning
parity. Fresh GPU ownership, free VRAM, CPU, RAM and scratch admission is
required for each fit; only xbabe1/2/3 may admit, without foreign preemption.

Charge the sampler plus projection to the 10,240-byte L3 cap. Decode every
successful artifact to verify that the trained compact neural target was
actually retained. Record every failed launch, fit, byte overrun and metric
cell. Source/runtime checks precede dependency initialization. Frozen round
sources and previous receipts remain immutable on benchmark scratch.
Official tests remain sealed; MFS-v2, PTF-v1 and release-safe claims stay null
until the complete protocol admits them.

The feature's pure profile test checks its finite parameter envelope and
unknown-request rejection. Actual GPU verification additionally requires the
existing deterministic conditional-readout fixture on an admitted GPU host;
CPU checks alone cannot establish GPU training evidence.
