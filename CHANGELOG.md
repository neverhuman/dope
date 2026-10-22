# Changelog

## Unreleased

- Add global CPU thread control, fit JSON stdout, runtime identity/digest diagnostics,
  and a documented complete Rust CLI workflow. Exercise commands without helper tools.
- Add a Linux x86-64 CPU release builder, dependency/ELF boundary checks, upstream
  license collection and tag-only GitHub/Sigstore signed provenance with verification
  before draft publication. Python training tools remain outside deployment.
- Add whole-generator parameter caps, explicit decimal/binary byte units and
  small/medium/large engine presets (2/4/16 KiB). Forced plans obey both caps;
  fit and inspect report the versioned scalar-state breakdown.
- Add `dope fit-tiers` for dataset-specific pretrained artifacts with source,
  training and binary digests, checksums and a model card for release review;
  `--anonymize` replaces names and labels with positional codes.
- Add descriptive BTR-AUC/Elo reports, per-dataset rankings and plots, paired
  dataset bootstrap intervals, predeclared schedule/dataset support, and CI
  replay checks. Current reports reuse development evidence; no new sealed-test,
  pretrained foundation-model or certification claim is made.

## 1.0.0 - 2026-10-22 (uncertified research release)

This release publishes the DOPE 1.0 engine (`engine/`, version 1.0.0), its
sixteen-table benchmark and its manuscript, together with every research-kernel
change listed below. It is an uncertified research release: it claims no tier
certification, sealed-holdout result, formal differential privacy, HIPAA
de-identification or other privacy guarantee. The research kernel (root crate,
`dope-kernel`) keeps version 0.3.0-alpha.1, and its `1.0.0-rc.1` identity stays
reserved for a gate-passing `build-rc`. Canonical tasks E03-E06 (development
pilot, signed precommit, sealed confirmation and independent reproduction)
remain open and gate a future certified release.

- Bound engine artifact decoding. Before allocating model arrays, the decoder
  enforces admission limits (64 MiB stored and inflated, a 1 MiB header and
  metadata, at most 64 sections and 4096 features). It checks every size with
  overflow-safe arithmetic and refuses cyclic copula trees. Format version 1
  and the encoder's bytes are unchanged.
- Add the compression frontier: worst-auditor retention (smallest of
  CatBoost, linear and MLP per lineage; a historical diagnostic) against charged
  bytes, and retained lift per KiB, recomputed from the pinned v2 ledgers with
  the panel's nested bootstrap. It is the README hero and Figure 1 of the TDL
  manuscript; its README block and TeX are authenticated by the claims
  firewall. Evidence remains selection-exposed validation.
- Add `examples/quickstart/`: a fit-only table encoder with lossy numeric preprocessing (target-ordered or
  one-hot categoricals, robust scaling, dates, missing values) with a compact
  charged state, transfer-in-the-loop backend selection inside the fit rows,
  a retention helper, an end-to-end quick start on UCI Adult and a five-seed
  ablation on UCI Adult and Abalone. Local engineering evidence only.
- Expand `paper/main.tex` into a full manuscript with a transfer theory
  (retention identities for squared and log loss, two corollaries, a size
  bound) and add `docs/math.md` (theory, byte anatomy, review findings and
  roadmap) and `docs/REPO_MAP.md`.
- Fix `dope-kernel inspect` for mixed artifacts (`DOPMIX01` magic).
- Retain root Python governance authorities and add navigation forwards in
  docs/governance; move the launcher to scripts and add linguist-only attributes.
- Transfer per byte, made measurable. A packed DPK3 value codec (header flag
  bit 1) stores quantization-grid codes, so bits set bytes: programs of the
  headline shape are about half the size and sample bit-identically (shapes with
  copula edges agree to the codec's parameter tolerance). Packed artifacts are
  not yet accepted by the TDL1 qualification bundle. The headline compact
  residual now trains natively in Rust (`--native-target-profile`), matching the
  published GPU profile's retention distribution without libtorch. `fit-table`, `sample-table` and `inspect-table` compile raw
  mixed-type CSVs with an invertible plan (log-scaled skewed numbers, target-
  ordered category codes, dates, missing cells) and decode samples to the
  source schema. An opt-in exact Gaussian copula (dependence family 7) is
  available for research. Defaults and historical artifacts are unchanged.
- Add the equal-byte study (`research/benchmark/equal_bytes`): DOPE programs
  against the most real rows that fit exactly the same bytes (best of three
  lossless codecs, MLP trained on step-matched rows) on the 26 real-world PMLB
  lineages and the full 97-lineage panel, protocol sensitivities, an exact
  Gaussian copula ablation, a scaling study, and a renderer for the new
  first-page figure, paper tables and the README block, authenticated by the
  claims firewall.
- Rewrite the README around transfer per byte with a UCI Adult quick start;
  rewrite `paper/main.tex` as a full manuscript in the ICLR layout with
  propositions and proofs; add `docs/math.md`; move internal planning records
  under `docs/internal/`.
- **DOPE engine 1.0 (`engine/`).** A standalone pure-Rust workspace (`dope-core`, `dope`
  CLI) that compiles a CSV into a byte-budgeted program stored as a standard safetensors
  file and samples it on CPU, CUDA or Metal (candle), with a native row-parallel CPU
  sampler. The program: an ML-ready intake (target-ordered levels, rare-level pooling,
  `log1p` for skewed counts, robust unit scaling; one-hot and `[0, 1]` output on request);
  marginals on a normal-score grid; a regime mixture of Gaussian copulas that keeps every
  marginal exactly; a residual tanh network trained with closed-form readouts plus boosted,
  byte-coded oblivious trees quantized with error feedback; calibrated noise; measured
  rate–distortion bit allocation; and a compile-time self-audit that chooses the byte split
  between dependence and target by retention on held-out training rows. The artifact is a
  standard safetensors file whose one tensor is a brotli stream of the schema and every quantized
  section, so column names, category labels and the tensor table no longer dominate a
  few-kilobyte file. Built-in auditors
  (`dope eval`, `dope frontier`) measure retention and TDL. Python is not needed to fit or
  sample. CI gate: `ops/ci/engine.sh`.
- **Real-world benchmark (`bench/real-world/`).** Sixteen public UCI and PMLB tables, three
  auditor families, the information-compression control (real rows `xz`-compressed to the
  same bytes), SDV and ARF generators, ablations, resumable records, a wall-clock limit per
  fit, and generated README, figures and paper macros authenticated by
  `tools/claims/dope1.py`. The research kernel (`dope-kernel fit-table`) is run on the same panel
  as a comparator.
- **DOPE 1.0 manuscript.** `paper/dope1.tex` (preprint), `paper/dope1-venue.tex` (double-blind
  ICLR layout) and `paper/dope1-supplement.tex` describe the engine and benchmark (theory of
  retention, method, results; every number a generated macro, every setting read from the Rust
  source). `paper/main.tex` and `paper/supplement.tex` stay the TDL manuscript.
- **README.** The landing page leads with the DOPE 1.0 engine (generated hero figure, results
  table, `dope` quick start and a UCI Adult walkthrough); the research kernel's equal-byte study,
  its `dope-kernel` quick start and every historical block are kept verbatim below.

- Add native raw-unit compilation, inspection, sampling, and a restricted
  artifact-only CPU worker. The complete raw-artifact charge includes retained
  fitted transforms and schema, framing, and native payload; historical DPK
  accounting and decoding remain separate. These fixtures do not certify
  learner transfer, tail fidelity, privacy, or a release.
- Rename the Python compatibility console to `dope-kernel-v1`; the native Rust
  executable remains `dope-kernel`. Native installation verifies the supplied
  checksum and identity and refuses to overwrite an existing command.
- Preserve missing fit/sample cells as incomplete estimands, with any
  complete-case summary confined to unqualified sensitivity output. Local and
  hosted CPU gates run the complete offline transfer fixture suite.
- Add a standalone native merge guard with isolated locked dependencies,
  exact-source review and check provenance, required GitHub app bindings, and
  a sole-open-PR constraint. Canonical paper admission retains all four PDF
  builds, independent reducer authentication, and output-drift checks.
- Frame the paper as *DOPE: Generative Compression of Predictive Structure*,
  bind abstract claims to their evidence registry, and show charged bytes
  beside paired auditor contrasts in the first-page figure.

- Block B: TabSyn's scaled arm (200 VAE and at most 1,000 diffusion
  epochs, which the authors' early stop can end sooner) is fit at seeds 23,
  37, 53, and 71 under its seed-11 caps. It is scored by the pinned evaluator,
  which matches bit for bit across fit-machine-1 and fit-machine-3 on 200 seeded cells, and
  it enters the v2 panel under the five-seed design, with missing fits and
  unscored sample cells disclosed. The paper's TabSyn rows,
  intervals, bytes, and seed text follow the panel.
- The pre-declared all-five-fits lineage sensitivity (`v2_all_five.py`) is
  printed for every five-seed arm and DOPE-minus contrast, outside every
  Holm family; CI recomputes it from the pinned ledger.

- Paper v2 (wave 2): a pre-declared v2 analysis contract
  (`research/benchmark/review_fixes/predeclare_v2.json`) committed before any v2
  result; DOPE refit at five fit seeds with the historical binary plus an
  untuned-default arm; a binary-by-seed diagnostic; nested cluster, lineage,
  and fit-seed bootstrap with Hodges-Lehmann contrasts and a robust
  equivalence rule; the full-panel TabSyn comparison, both controls, and the
  privacy panel in the main paper; a utility-per-byte figure; a rewritten
  manuscript and supplement; a README benchmark generated from the same
  ledgers; and guards against orphan tables, figure-text drift, and
  denominator mismatches. On the validation panel the five-seed CatBoost
  retention at 4n is 0.898, against 0.949 for the earlier fit-seed-11
  population: both binaries write identical artifacts, and seed 11, the seed at
  which the profile was chosen, ranks first of five on 47 of 99 lineages.
  The v2 and retained-evidence reducers refuse any undeclared method or
  configuration before grouping, and the statistics text states that the
  Wilcoxon p-values treat lineages as independent. The supplement prints the
  first version's headline contrasts as a labeled traceability table.
  Captions and text now state the loss space, every interval type, and each
  inclusion rule where its count is printed, and the abstract prints the
  interval of every contrast it reports.
  The supplement RMSE table states original target units, as its producer
  computes them, and the compute-cost and publication-row reducers read every
  input through a declared length and digest before decoding it.
  The supplement repeats the source-family sensitivity on the five-seed
  panel (`v2_family.py`): excluding the four simulated families and within
  each family, with fit seeds resampled inside each lineage, as rows outside
  every Holm family; CI recomputes the family file from the pinned ledger.

- Add tiered compact kernels, measured release gates, and versioned MFS-v2
  evidence on the single PR branch. No production release is declared.
