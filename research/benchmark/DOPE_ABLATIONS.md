# Bounded target ablations

The `gpu-research-training` build accepts the research profile
`grid_features{6|12|24}_width{linear|8|16}_steps{512|2048|8192}_readout{on|off}`.
The Cartesian grid has 54 configurations. It preserves the eight older
research profiles. The CPU research device continues to require
`features12_steps2048`; this grid does not expand CPU admission.

Linear configurations fit a sparse linear or logistic target with AdamW.
Widths 8 and 16 fit the existing neural residual architecture. `readouton`
uses the regularized readout solver; `readoutoff` exports the optimizer's
weights. These targets use the existing artifact codec, with no width-zero
neural target. Every configuration remains subject to the existing per-fit
deadline, memory, seed, artifact and host admission rules.

When validation logging is enabled, each optimizer step records the training
loss before the update and validation loss after the update. Both precede
the optional readout refit. Therefore these losses do not evaluate the
refitted exported artifact.

`research_ablation_binding.py` checks frozen inspection and log byte hashes
before parsing either payload. It checks the requested architecture against
the emitted target opcode and requires the complete finite step log. Its
output omits field names, expressions, coefficients and loss values.
Source, binary, runtime and artifact hashes are supplied custody inputs;
the helper does not re-read those bodies. Width, seed and readout mode are
requested settings, not independently attested artifact properties.

The source and codec controls establish preparation only. Actual generated
training, tensor/export parity, repeated-seed checks and legacy-profile
parity need separately admitted runtime evidence. No matrix, native fit,
GPU result or production eligibility is established by this source change.
Official tests remain sealed and gated scores remain null.
