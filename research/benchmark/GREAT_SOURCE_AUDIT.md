# GReaT author-source admission audit

GReaT remains `source_audit_pending`. This audit verifies source rights and
contracts identified in code; it admits no model, runtime, tuning or campaign
job. Official tests remain sealed. MFS-v2, PTF-v1, release-safe and superiority
results remain null, and this method contributes no DOPE win.

## Source and rights

The current author repository is [tabularis-ai/be_great](https://github.com/tabularis-ai/be_great),
commit `616b2e74b396fa1f9c7ab4412b9928eec849159a`, with MIT code and package
version 0.0.14. The original paper is cited as `borisov2023great`.
[great-source.audit.json](great-source.audit.json) binds the archive, license,
untruncated Git tree, verification receipt and all 29 declared source/docs
files. Every declared file matches its archive member and Git blob. Archive
links and special members were rejected. Author modules were parsed without
importing them; datasets, notebooks, images and weights were not read.

Code rights do not establish the rights or provenance of a chosen pretrained
checkpoint. The constructor requires an `llm` argument. A README example is
not a complete author default for every study dataset. Model/tokenizer
revisions, licenses, provenance and complete required bytes remain unfrozen.

## Released defaults

The [constructor and fit/sample APIs](https://github.com/tabularis-ai/be_great/blob/616b2e74b396fa1f9c7ab4412b9928eec849159a/be_great/great.py)
provide these defaults:

| Area | Source defaults |
| --- | --- |
| Constructor | Required pretrained `llm`; 100 epochs; batch 8; full fine-tuning; no decimal rounding; no reporting integration |
| Fit | Random conditional column each epoch; no evaluation data; no checkpoint resume |
| Sample | Temperature 0.7; generation batch 100; token limit 100; legacy sampling; keep missing values; automatic device choice |

The optional live evaluation callback displays column-distribution similarity.
It does not define a frozen ML utility checkpoint selector. Learned row text
uses shuffled feature order. NumPy arrays and DataFrames are accepted in the
source, but no study representation or seeded fit/sample contract has been
executed here.

## Native objective

The released [MLEfficiency implementation](https://github.com/tabularis-ai/be_great/blob/616b2e74b396fa1f9c7ab4412b9928eec849159a/be_great/metrics/utility.py)
requires the caller to provide both an auditor model and scoring function.
It reports their mean and population standard deviation over default seeds
512, 13, 23, 28 and 21. The source labels its direction `maximize` regardless
of the scoring function. This is not an author default regression scalar.

The [paper's Table 1](https://arxiv.org/pdf/2210.06280v2) reports regression MSE
separately for linear regression, a decision tree and a random forest. Lower
MSE is better. Those results do not supply a scalar combination for selecting
the generator, and the generic released direction must not reverse an error
metric. No scalar, generator search space or tie-break has been frozen.

The author metric fits encoders and an optional scaler to its `real_data`
argument. If `real_test_data` is absent, it uses the last 20% of that argument
as evaluation rows. Study tuning must explicitly pass the frozen validation
partition derived from official training and use fit rows for preprocessing.
Missing values are replaced with zero in real and synthetic frames. Native
values from this method must never be ranked against another method's KPI.

Freeze a defensible author objective before tuning. If none can be established,
run the frozen author default and mark tuning inapplicable. The shared DOPE
KPI cannot fill this gap.

## Execution gates still required

1. Pin an isolated dependency/runtime inventory and verify it before imports.
   Python >=3.9 is declared; dependencies have lower bounds rather than exact
   versions. `fsspec` appears in requirements but not project dependencies.
   The released Trainer API compatibility has not been verified.
2. Freeze pretrained model/tokenizer identity and a complete default config.
   Constructor and loader use `from_pretrained` without a source-bound revision.
   Offline, artifact-only loading and safe serialization need a verified wrapper.
3. Charge all required state: weights, any base needed by LoRA, tokenizer,
   model config, conditional distribution, column statistics and projection.
   [Conditional state](https://github.com/tabularis-ai/be_great/blob/616b2e74b396fa1f9c7ab4412b9928eec849159a/be_great/great_utils.py)
   stores all observed values when the conditioning dtype matches `float`. The
   saved config includes that state; it must remain on restricted scratch.
4. Establish common-numeric and separate author-faithful representations, fit
   seeds and sample seeds. `sample` has no seed argument. Its legacy path can
   return partial or empty tables after errors; default missing values also need
   explicit generator-validity rejection. Verify all requested sizes and replay.
5. Capture detailed author errors/logs only on scratch and emit typed public
   failures. Run row-copy, leakage, privacy and real-vs-real controls before
   gated claims. The released privacy metric names establish no formal DP claim.
6. Admit each operation on xbabe1/2/3 with disjoint CPU/RAM/scratch reservations,
   the unchanged 600-second neural-fit and 16-GiB GPU limits, and full failure
   accounting. Runtime or sample failure creates an explicit failed cell.

This audit contains no dataset rows, generated samples, weights or benchmark scores.
