# Baseline completion and canonical timeout checkpoint

This frozen checkpoint recounts small published scalar receipts and 343
original Forest-Flow job/fit/closure triples. It is not a new scientific fit
or evaluation. The current CPU round has 268 successful fits and 75 timeouts;
13 disjoint five-fit default lineages have published common metrics.

Run `python3 -B research/benchmark/publish_baseline_completion.py` from the
repository root to reproduce `panel.json`, `completion.md` and all 75 entries
in `timeout-dispositions.json`. The publisher verifies the input lock before
decoding inputs, and authenticates original job, fit, closure and frozen-round
bytes before recounting. The archive contains only small metadata; it contains
no sample rows, weights, headers or source extrema. Original runtime and source
identities were rehashed during the read-only capture; current bulk models
were not rehashed. Capture did not execute a scientific model or auditor.

The 6,000-cell denominator is default/native × 100 lineages × five fits ×
n/4n × three sample seeds. Forest's default-only denominator is 3,000 cells.
Fit dispositions, successful fits, measured metric cells, and declared aliases
are separate quantities. Historical failures not recounted in the pinned
publications remain unknown here rather than reported as zero.

The live queue can progress after this checkpoint. `rate.json` preserves the
two original read-only observations used for a conditional disposition ETA.
Six hours of metric evaluation is a planning allowance, not measured compute.
GPU cap-envelope ETAs assume an eligible continuous slot and do not admit work.
No native tuning winner is inferred from TabSyn's inapplicable audit values.

See `docs/whitepaper/BASELINE_PROTOCOL.md` for the prospective disposition
policy. Keep the frozen 600-second canonical cap. No automatic higher-cap
retry, no erased original cost, and no DOPE win from missing cells. Official
tests stay sealed; MFS-v2/PTF-v1/release/superiority remain null.
