# Historical Paper Errata (`docs/whitepaper/dope-mfs.tex` and v2 PDFs)

This errata note accompanies the frozen historical v2 validation manuscript
([`docs/whitepaper/dope-mfs.tex`](../whitepaper/dope-mfs.tex),
[`docs/whitepaper/supplement.tex`](../whitepaper/supplement.tex)) and its four
committed historical PDF builds (`dope-mfs.pdf`, `supplement.pdf`,
`dope-mfs-anonymous.pdf`, and `supplement-anonymous.pdf`). Those files are
frozen reproducibility snapshots (`T-111`, 2026-10-15) and are not modified in
place; active public surfaces and the TDL manuscript live in
[`README.md`](../../README.md), [`STATUS.md`](../../STATUS.md), and
[`paper/`](../../paper/README.md).

## WC01 — Historical "Driver Fidelity" Proxy vs. Cross-Model Feature-Driver Preservation (`D80`)

### Affected Frozen Passage

In [`docs/whitepaper/dope-mfs.tex`](../whitepaper/dope-mfs.tex#L233) (Section 2,
*Evaluation Protocol*, paragraph *Structure and efficiency diagnostics*), the
frozen historical text states:

> *Driver fidelity* compares the permutation importance of each input feature on
> the real-trained and synthetic-trained predictors on holdout rows; a value
> near 1 means the synthetic table leads the auditor to rely on the same columns
> as the real table. On the 99-lineage CatBoost panel, DOPE reaches median
> driver fidelity `0.905 [0.844, 0.922]` (`\VDriverMed` `[\VDriverLo,
> \VDriverHi]`).

### Producer Audit and Exact Mathematical Quantity

Direct inspection of the Rust producer
[`rust/certification/privacy_attacks.rs`](../../rust/certification/privacy_attacks.rs#L170)
(`driver_fidelity`, lines `170` and `193–255`) establishes that the historical
`driver_fidelity` field in the v2 and MFS-v3 validation ledgers does **not**
compute holdout permutation-importance agreement between a real-trained and a
synthetic-trained predictor. Instead, it computes the mean across input columns
$j \in \{1, \dots, d\}$ of the absolute feature-target Pearson correlation
agreement between the fit rows and the synthetic rows:

$$\text{proxy}_{\text{corr}} = \frac{1}{d} \sum_{j=1}^{d} \operatorname{clamp}_{[0, 1]}\!\left(1 - \bigl|\,|\operatorname{corr}_{\text{fit}}(X_j, y)| - |\operatorname{corr}_{\text{syn}}(X_j, y)|\,\bigr|\right).$$

Specifically:

1. **99-lineage classical CatBoost panel (`\VDriverMed = 0.905 [0.844, 0.922]`):**
   This value is the lineage median (and 95% bootstrap interval) of the
   fit-vs-synthetic absolute feature-target correlation proxy
   (`absolute_target_correlation`) produced by
   `rust/certification/privacy_attacks.rs::driver_fidelity`.
2. **94/97-lineage historical foundation-model block (`0.975` vs. `0.923` for Gaussian copula):**
   Both values are reductions of the same `driver_fidelity`
   (`absolute_target_correlation`) field across the 94 lineages with non-empty
   feature matrices in `research/benchmark/results/mfs-v3-density-panel.json`:
   - `0.975` is the 94-lineage median of per-lineage medians across the three
     sample seeds (`101`, `211`, `307`).
   - `0.923` (`0.923106`) is the 94-lineage median of per-lineage minimum seed
     values under the legacy `assembler::_min` reducer (the pooled 282-seed
     median is `0.973214`).
3. **Cross-model feature-driver preservation (`D80`):**
   Holdout permutation-importance / attribution agreement (`D80`) across the
   tabular foundation-model auditors (Kumo Large, TabICLv2, Mitra-v2) and the
   TDL driver gate ($\ge 0.80$) remains **`unmeasured`** across all public claim
   surfaces ([`README.md`](../../README.md), [`STATUS.md`](../../STATUS.md),
   [`paper/main.tex`](../../paper/main.tex), and
   [`tools/claims/foundation.py`](../../tools/claims/foundation.py)).
