# Changelog

## Unreleased

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
