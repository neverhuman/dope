# CPU learned-target research backend

The `gpu-research-training` build can train the compact neural target on CPU
when both `DOPE_RESEARCH_TARGET_DEVICE=cpu` and an explicitly empty
`CUDA_VISIBLE_DEVICES` are set. CPU training requires the frozen
`DOPE_RESEARCH_TARGET_PROFILE=features12_steps2048`: at most 12 selected inputs,
16 hidden units, and 2,048 AdamW steps. The loss, learning rate, clipping,
readout refit and artifact codec are the existing implementation.

An unset device retains the CUDA path. Other device values fail closed.
Ordinary builds reject an explicit research device instead of silently using
the fixed-hidden native implementation.

CPU and GPU fits share the existing process-wide seed lock. The CPU helper
uses `tch::manual_seed` and CPU tensors; it makes no explicit CUDA calls.
Torch can query accelerator availability internally, and the research binary
still links the CUDA-capable Torch runtime. This is not a claim of a runtime
that never queries accelerators. GPU peak-memory evidence is absent for CPU
fits.

Every CPU campaign needs its own source, binary, runtime, backend and job
identities, fresh CPU/RAM/scratch admission and the original per-cell trial
and wall ceilings. Keep previous GPU receipts and costs. Combining a GPU
seed with CPU seeds does not establish within-GPU fit variation. Sealed tests
remain evaluator-only; this backend does not establish production scores.

Verification uses the default CLI rejection control and a separately admitted
research-runtime qualification: CPU seed replay, different-seed dispersion,
generated-data training, artifact inspection and sampling. The source-only
PR does not admit a real dataset matrix or certify a release.
