# Benchmarking section plan: block B and the five-seed results

- Local plan. Updated 2026-10-13 against main `d47bd2b`. No PR. The prior tip is `b6dc72d` (the 15:16Z refresh of `66e86b6`). That refresh still described an open run.
- Checked on `d47bd2b`:
  - `predeclare_v2.json` and `predeclare_v2.amendments/B-tabsyn-seeds.json`;
  - `research/benchmark/results/review-fixes-v2/panel.json` (sha256 prefix `4b3f1307a22d2c2e`, unchanged);
  - `research/benchmark/review_fixes/v2_panel.py` (`FIVE_SEED` still omits TabSyn);
  - `research/benchmark/results/review-fixes-tabsyn-seeds-v2/` from #230 (`98120d7`);
  - the scalar ledger is still 27,756 lines, with TabSyn scores only at fit seed 11 (594 cells).
- The Block B run is closed. The P-BB1 ledger is on main. The five-seed score rebind is not. This plan does not copy unpublished panel numbers.

## 1. Where the section stands on main

All numbers below are CatBoost at size 4n, with 95% nested-bootstrap intervals, from the v2 panel on `d47bd2b`. They are the published seed-11 TabSyn contrast, not the five-seed rebind.

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

Five-seed source-family sensitivity is already in the supplement on main (#227). It is not the TabSyn five-seed rebind.

Primary Holm family slots on main:
- ARF author-default: present (five seeds).
- TabSyn: present, **one seed**.
- TabDDPM: **absent**.
- Forest-Flow under B1: **absent**.

Untested slots keep their place: `v2_panel._holm` uses count = 12 and skips the empty slots. Every primary Holm p printed today is a fixed-m value that changes when the empty slots fill.

## 2. What block B adds, and how it enters the panel

**Block B** is TabSyn's scaled arm (200 VAE and at most 1,000 diffusion epochs) at fit seeds 23, 37, 53 and 71:
- 100 lineages, 400 fits, three sample seeds at n and 4n;
- caps of 600 s per fit and 180 s per lineage's six samples;
- source git `41a47ab`, host `xbabe1`, recorded in `binding.json`.

**Ledger on main (`98120d7`).** `binding.json` reports fits ok 400 and cells ok 2,394 / timeout 6. `medians_not_computed` is true.
- Fits: 100 ok at each seed. The 15:16Z infrastructure failures were attempt 1. In the committed ledger those 4 seed-53 fits are status ok at attempt 2, and every other fit is attempt 1. The fit rows record that final attempt number. They do not carry the attempt-1 failure receipt. There is no fit timeout in the final ledger.
- Completed diffusion epochs: 399 fits at the 1,000-epoch budget (`declared_epoch_budget_complete`) and 1 fit at 785 (`author_patience_500`). The manuscript says "at most 1,000". It does not say every fit reached 1,000.
- Sample cells: 2,400 rows, ok 594/600/600/600 and timeout 6/0/0/0 by seed 23, 37, 53, 71. All 6 timeouts are one seed-23 lineage. They are final and were not retried. Do not name that lineage.
- Cell rows have no attempt field. Sample-retry custody is unrecorded, not measured zero.
- The ledger does not contain the cross-host comparison, and it does not contain retention scores. `official_tests_opened` is false.
- Do not quote an `artifact_bytes` median as the charged-byte median. The published charged median remains the seed-11 figure, 42.3 MB, until `v2_bytes.tabsyn()` is rebuilt on all 500 fits under the charged-bytes rule. This update does not compute a Block B byte or epoch median.

**Binding path.** Each step is pinned and replayable; no number is hand-copied.
1. Done on main: the ledger commit. A cell whose receipt says `failed`/`TimeoutError` is a deadline cut and is counted as **timeout**.
2. Not in this tree: the K0 rescore. The amendment still requires `rescore_ledger.py`, auditor seed 1729, and the same evaluator and dependency set as the seed-11 cells. Refuse a cell whose evaluator pin differs.
3. Not on main: extending `scalar-cells.jsonl` and joining the 400 fits into the v2 `fits.jsonl`, the `block_b` provenance source, and a recomputed `scalar-cells.meta.json`.
4. Not on main: add `("TabSyn", "scaled_200_vae_1000_diffusion")` to `FIVE_SEED`.
   - This **implements** the predeclaration: methods.primary.TabSyn has seeds 11–71, and estimands.lineage_value requires at least 3 complete fits. It is not a new choice.
   - Without it, `min_fits` stays 1, and a TabSyn lineage with a single complete fit would count.
5. Add the sensitivity rows: a generalized `seeds_23_71` and an all-five-fits row. See §3. The all-five reducer is local prep on `2321aec` and is not on main.
6. Change `v2_bytes.tabsyn()`: read the charged bytes of all 500 fits (seed 11 plus block B) under the **same charged-bytes rule**. That rule takes each saved file except intermediate diffusion-epoch checkpoints, and is not `artifact_bytes`.
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
| **Supplement C** (coverage) | One row per seed with one column per controller status (ok, timeout, failed, unknown, infrastructure_failure, not_started), plus retries and planned against observed fits and cells (100 and 600 per seed). Do not name the timeout lineage. Sample retries stay unrecorded where the cell row has no attempt. Add the 5ad517e disclosure. |
| **Limitations** | "Seeds": TabSyn now contributes five fit seeds, so remove it from the one-seed list. "Baselines": remove "five-seed TabSyn" from the pending list; the TabSyn author-default schedule stays pending. |
| **Abstract and conclusion** | No new words; the macros carry the change. |

Not in the paper: per-seed retention grids and fit-progress bars. Those are monitoring views. The local Block B draft on this lane renders coverage from the ledger and leaves levels, contrasts, bytes, epochs, and retention unpinned until the rebind is on main.

## 4. What block B and the five-seed results can and cannot claim

**Can claim**, reported whatever the values are, once the rebind is on main:
1. The five-seed DOPE minus five-seed TabSyn (scaled) paired difference at n and 4n, under three auditors.
   - Its nested interval includes fitting variance on **both** sides; the published panel omits TabSyn's.
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

The arms below are those predeclare_v2 declares that are absent from the v2 panel on main `d47bd2b`.

| Family | Missing | Consequence |
|---|---|---|
| Primary | TabDDPM author-default (v6); Forest-Flow under B1 | The primary Holm p stays provisional (fixed m = 12, empty slots remain). Forest-Flow stays a historical block. |
| Deep secondary | TabDiff; GReaT | CTGAN and TVAE author-default cover only 17–18 lineages. |
| Other secondary | synthpop-CART; SMOTE | — |
| Secondary | TabSyn author-default (B′) | The "scaled schedule" caveat stays. |
| Seeds | Five seeds for the density baselines (Gaussian copula, Chow–Liu, independent) | Their side of each interval still omits fitting variance. |
| Block K | Common-evaluator re-audit | — |
| Block L | Privacy v2: TabSyn distance and MIA cells, shadow-model and artifact attacks | — |
| Blocks I, J, M | Ablation grid, stress suite, external panel | — |
| Stage 3 | Sealed official test | — |

## 6. Order of work

1. **Done:** the run has exited and the P-BB1 ledger is on main. The local Block B draft tracks that ledger. Scores stay unpinned.
2. **Not this lane:** P-BB2 is paper_claude's rebind (`adb9a62`), queue slot 2 after codex_2 PR13. It is the step that extends the scalar ledger and adds TabSyn to `FIVE_SEED`.
3. **The PR**, opened by that lane when its slot is open and the open-PR count is 0. Review checklist:
   - the evaluator pins are equal;
   - `FIVE_SEED` includes TabSyn;
   - the charged-bytes rule is unchanged;
   - the Holm family is still 12 slots;
   - zero-diff rebuild and the paper gates pass;
   - the 5ad517e disclosure is present;
   - the text says "at most 1,000" diffusion epochs;
   - the timeout lineage is not named, and missing sample attempts are unrecorded;
   - no claim from §4 "cannot" appears in the text.
