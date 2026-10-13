# Benchmarking section plan: block B and the five-seed results

- Local draft by paper_claude, 2026-10-13. No PR is opened until the queue clears.
- Inputs, all on main `d620f0c`:
  - `predeclare_v2.json` and `predeclare_v2.amendments/B-tabsyn-seeds.json`;
  - `research/benchmark/results/review-fixes-v2/panel.json` (sha256 prefix `4b3f1307a22d2c2e`);
  - `research/benchmark/review_fixes/v2_panel.py`;
  - `docs/whitepaper/dope-mfs.tex` §4–§6.
- Block B is still running. This plan uses only what is already published, and the run's status, timing and counts. No block B retention exists yet. Every block B value below is a slot.

## 1. Where the section stands on main

All numbers below are CatBoost at size 4n, with 95% nested-bootstrap intervals, from the v2 panel.

| Row | Value | Source |
|---|---|---|
| DOPE headline, five seeds | level 0.898 [0.804, 0.972], n = 99 lineages, 495 fits | `\VDopeCbFour` |
| Strongest comparator (computed) | TabSyn, scaled schedule, **fit seed 11 only**: level 0.897 [0.816, 0.979], n = 98 | `\VStrongestName` |
| DOPE − TabSyn | HL −0.008 [−0.057, 0.042], n = 98, W/T/L 47/0/51, Wilcoxon p 0.64, Holm 1.000 (m = 12) | `\VStrongDiff`, Table 1 |
| DOPE − ARF (author default, five seeds) | 0.048 [−0.034, 0.149], n = 99, Holm 0.258 | Table 1 |
| Equivalence (±0.02, 90% HL interval) | none; DOPE − TabSyn is also not non-inferior (5th percentile below −0.02) | `\VEquivResults` |
| DOPE seeds 23–71 alone | 0.862 [0.771, 0.958] | `\VLaterCbFour` |
| Seed-11 rank test | DOPE seed 11 is the top fit on 47 of 99 lineages, against 19.8 expected | `\VSeedTop` |
| TabSyn charged bytes | median 42.3 MB over 100 seed-11 fits; DOPE 1.9 KB | `\VStrongBytes` |

Primary Holm family slots on main:
- ARF author-default: present (five seeds).
- TabSyn: present, **one seed**.
- TabDDPM: **absent**.
- Forest-Flow under B1: **absent**.

Untested slots keep their place: `v2_panel._holm` uses count = 12 and skips the empty slots. Every primary Holm p printed today is therefore a fixed-m value that changes when the empty slots fill.

## 2. What block B adds, and how it enters the panel

**Block B** is TabSyn's scaled arm (200 VAE and 1,000 diffusion epochs) at fit seeds 23, 37, 53 and 71:
- 100 lineages, 400 fits, three sample seeds at n and 4n;
- caps of 600 s per fit and 180 s per lineage's six samples;
- run on xbabe1 from 41a47ab, with a close of 2026-10-22T05:59Z.

**Status at 15:16Z:**
- Fits: 275 of 400 ok, 4 in infrastructure_failure (seed 53), 121 not run. Seeds 23 and 37 are complete.
- Sample cells: 1,614 ok and 6 in timeout. Another 30 cells are in infrastructure_failure (5 sample operations), and 24 are blocked behind infrastructure-failed fits.
- **The 6 timeouts** are one seed-23 lineage, whose six samples ran 185 s against the 180 s cap. That fit therefore has no complete sample seed. The lineage still qualifies if at least 3 of its other 4 fits complete.
- **The infrastructure failures** are foreign GPU processes (recurring Blender renders on xbabe1). The controller confirmed each one on its own side, so each is non-final at attempt 1. `Controller.run` makes one pass over the plan, so each failure gets its one predeclared retry only from a second `run` invocation after this unit exits.
  - That decision belongs to the watcher and the owner.
  - The ledger and the coverage table must show the first attempt and the retry outcome for each one.

**Binding path.** Each step is pinned and replayable; no number is hand-copied.
1. When the unit exits, record the final status counts. Then run `tabsyn_seeds_v2 ledger` and commit `research/benchmark/results/review-fixes-tabsyn-seeds-v2/sample-cells.jsonl`. It holds status, paths and sha256 only, with no retention.
   - Label from receipts, without editing any receipt: a cell whose receipt says `failed`/`TimeoutError` is a deadline cut and is counted as **timeout**. Put that rule in the ledger code, not only in a display script.
   - Run the ledger only after any retry pass has ended. Each infrastructure_failure record then shows its final attempt.
2. Rescore on xbabe1 CPU, after the GPU block closes, as the amendment's `scoring` says:
   - `rescore_ledger.py` with adapter `rescore_adapters.tabsyn_seeds`;
   - auditor seed 1729;
   - evaluator `pilot_metrics` `6c038926…` and the v2 ledger's dependency set (catboost 1.2.10, numpy 1.26.4, scikit-learn 1.7.2, scipy 1.16.3).
   - Refuse any cell whose evaluator pin differs from the seed-11 TabSyn cells already in the v2 ledger.
3. Extend the v2 scalar ledger:
   - add the new cells to `scalar-cells.jsonl` and the 400 fits to `fits.jsonl`;
   - add a `block_b` source to `provenance.json` (commit 41a47ab, host, runner sha, unit, campaign receipt);
   - recompute `scalar-cells.meta.json`.
4. Change `v2_panel.py`: add `("TabSyn", "scaled_200_vae_1000_diffusion")` to `FIVE_SEED`.
   - This **implements** the predeclaration: methods.primary.TabSyn has seeds 11–71, and estimands.lineage_value requires at least 3 complete fits. It is not a new choice.
   - Without it, `min_fits` stays 1, and a TabSyn lineage with a single complete fit would count.
5. Add the sensitivity rows: a generalized `seeds_23_71` and an all-five-fits row. See §3.
6. Change `v2_bytes.tabsyn()`: read the charged bytes of all 500 fits (seed 11 plus block B) under the **same charged-bytes rule**. That rule takes each saved file except intermediate diffusion-epoch checkpoints, and is not `artifact_bytes`.
   - The seed-23 artifact_bytes median, 84.4 MB, is not comparable with the published charged median of 42.3 MB.
7. Rebuild and pin:
   - `v2_panel.py` → `panel.json`;
   - `review_v2_emit.py`, then `v2_figures.py`, `readme_benchmark.py` and `paper_consistency.py`;
   - update the pins in `ops/ci/paper-review-inputs.json` and `ops/ci/paper-inputs.d`, and read the files through `pinned_inputs`;
   - CI deletes `generated/` and rebuilds with zero diff.
8. Disclose, in the supplement and the ledger PR:
   - one outcome-blind partial run at 5ad517e (6 fits) was stopped for a controller defect. It was kept, not scored and never pooled.
   - The relaunch refit those lineages under new attempt keys in a fresh output root.

## 3. Tables and figures

All data-dependent words come from verdict macros ("is above / is below / cannot be separated from"). No sentence names TabSyn as the strongest comparator; `\VStrongestName` is computed.

| Item | Change |
|---|---|
| **Table 1** (DOPE − comparator, 4n) | The TabSyn row becomes the five-seed arm, labelled "TabSyn (scaled schedule, five seeds)". It keeps n, HL [95% nested CI] for all three auditors, W/T/L and Holm p. The caption keeps "untested members keep their slot" and says the primary Holm p is final only when the TabDDPM and Forest-Flow-B1 slots fill. |
| **Table 2** (levels, n and 4n) | The TabSyn level becomes five-seed (≥3 complete fits per lineage), and its fit count is shown. |
| **Fig. 1** (forest plot) | The TabSyn row moves to the five-seed estimate. Add a light sensitivity marker: TabSyn at seed 11 alone, today's published value, for continuity. |
| **Pareto figure** | The TabSyn point uses the five-seed level and the 500-fit charged-byte median and IQR. |
| **§5.6 Seeds** (text) | Add two sentences. (a) TabSyn's seed-11 level against its seeds-23–71 level, as a descriptive row. (b) **DOPE seeds 23–71 minus TabSyn seeds 23–71**: neither arm uses fit seed 11, the seed at which DOPE's profile was chosen. This row is descriptive, has no Holm family, and its interval is printed. |
| **Supplement E** | Per-seed TabSyn levels; the all-five-fits sensitivity for DOPE and TabSyn; original-unit RMSE beside retention (estimands.original_units). |
| **Supplement C** (coverage) | One row per seed with one column per controller status (ok, timeout, failed, unknown, infrastructure_failure, not_started), plus retries and planned against observed fits and cells (100 and 600 per seed). Give counts only; the lineage identities stay in the ledger, not the paper. Add the 5ad517e disclosure. |
| **Limitations** | "Seeds": TabSyn now contributes five fit seeds, so remove it from the one-seed list. "Baselines": remove "five-seed TabSyn" from the pending list; the TabSyn author-default schedule stays pending. |
| **Abstract and conclusion** | No new words; the macros carry the change. |

Not in the paper: per-seed retention grids and fit-progress bars. Those are monitoring views.

## 4. What block B and the five-seed results can and cannot claim

**Can claim**, reported whatever the values are:
1. The five-seed DOPE minus five-seed TabSyn (scaled) paired difference at n and 4n, under three auditors.
   - Its nested interval includes fitting variance on **both** sides; today it omits TabSyn's.
   - Its Holm p sits in the 12-test primary family. The verdict wording is "is above", "is below" or "cannot be separated from".
2. Which comparator is the computed strongest. TabSyn stays strongest only if its five-seed CatBoost 4n median stays highest among comparators paired on 90 or more lineages; ARF author-default is next at 0.822.
3. Whether TabSyn's seed 11 was typical for TabSyn: the seed-11 level against the seeds-23–71 level. This is descriptive and has no test.
4. DOPE against TabSyn with fit seed 11 removed from both arms. This takes out the seed-specific part of DOPE's selection, and nothing more.
5. The byte ratio over 500 TabSyn fits, with the same charged-bytes rule.

**Cannot claim:**
1. **Superiority.** `claims.superiority` is null. Even an interval above zero is a validation-row estimate.
2. **Equivalence or "on par" from a non-separation.** "Cannot be separated" means the interval contains zero.
   - Equivalence needs the 90% HL interval inside ±0.02. Today the 95% interval spans 0.099.
   - Never write "matches", "TabSyn-level" or "on par" unless the rule is met.
3. **Anything about TabSyn's author-default schedule** (4,000/10,001 epochs). Block B is the scaled arm only, and B′ has not run. Every mention says "scaled schedule".
4. **Removal of the profile selection.** DOPE's profile was chosen at seed 11 on validation rows and is used at every seed. Seeds 23–71 remove the seed effect, not the profile choice. Only the sealed test is free of it.
5. **A test-set result.** Every number is validation-only until the one-shot sealed run.
6. **A TabSyn privacy or fidelity result from block B.** It adds samples but no distance, MIA or detector scoring; TabSyn's distance cells stay pending (block L).
7. **A cost or budget-matched comparison.** The 600 s and 180 s caps are seed-11 operational caps, not the B1 common budget, and block B timings are operational.
8. **A failure rate for TabSyn from the 6 timeout cells.** They are final under the predeclaration, are counted as timeouts, and are never imputed or retried.
9. **Anything outside the panel:** classification, more than 800 fit rows, non-PMLB data, or an MFS score (the score is null).

## 5. Still unmeasured after block B

The arms below are those predeclare_v2 declares that are absent from the v2 panel on main d620f0c.

| Family | Missing | Consequence |
|---|---|---|
| Primary | TabDDPM author-default (v6); Forest-Flow under B1 | The primary Holm p stays provisional (fixed m = 12, 6 slots empty). Forest-Flow stays a 6-lineage historical block. |
| Deep secondary | TabDiff; GReaT | CTGAN and TVAE author-default cover only 17–18 lineages. |
| Other secondary | synthpop-CART; SMOTE | — |
| Secondary | TabSyn author-default (B′) | The "scaled schedule" caveat stays. |
| Seeds | Five seeds for the density baselines (Gaussian copula, Chow–Liu, independent) | Their side of each interval still omits fitting variance. |
| Block K | Common-evaluator re-audit | — |
| Block L | Privacy v2: TabSyn distance and MIA cells, shadow-model and artifact attacks | — |
| Blocks I, J, M | Ablation grid, stress suite, external panel | — |
| Stage 3 | Sealed official test | — |

## 6. Order of work

1. **Now until the last fit (about 17:20Z):** outcome-blind monitoring only. Draft code paths and tests against fixtures (§2, steps 3–7), with no block B values.
2. **After the unit exits:** final status counts; the ledger commit with the timeout labels; the rescore; then the bind and rebuild.
3. **The PR**, opened when the queue reaches it. Its review checklist:
   - the evaluator pins are equal;
   - `FIVE_SEED` includes TabSyn;
   - the charged-bytes rule is unchanged;
   - the Holm family is still 12 slots;
   - zero-diff rebuild and the paper gates pass;
   - the 5ad517e disclosure is present;
   - no claim from §4 "cannot" appears in the text.
