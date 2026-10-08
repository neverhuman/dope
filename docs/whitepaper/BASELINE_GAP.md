# Baseline coverage and remaining evidence

Updated 2026-10-08 from authenticated validation publications. Official
tests remain sealed. This memo grants no fit, sampling, GPU, or test access.
Native objectives select configurations only within their own method;
common fidelity, utility, detection, and privacy metrics select nothing.

## Published common measurements

| Method | Receipt-backed measured coverage | Remaining scope |
| --- | --- | --- |
| ARF | 100 lineages, default/native-selected, five fit seeds 11/23/37/53/71, n/4n, three sample seeds: 6,000 logical cells / 5,892 physical metric receipts | Official-test/release certification; other protocol sizes are not inferred |
| TabSyn | 143 scaled-default cells on 49 lineages, fit11: the earlier eight complete n/4n lineages (48 cells), plus 95 n cells on 41 other lineages | 22 partial n groups; complete default population, native tuning and five-fit replication |
| TabDDPM | 12 retained lineages, 21 models, 126 physical / 132 logical n/4n cells, fit11; default cohort 10, native-selected cohort 12 | Full population, completed native search, five-fit replication |
| Forest-Flow | 13 disjoint default lineages at five fit seeds, 65 models / 390 n/4n cells; earlier six-lineage native/default panel remains separate | Rest of current default fit/metric round, timeouts, native-selected replication |
| Forest-Diffusion | Pilot mode receipts only | Separate diffusion-mode population; Flow does not establish this result |
| GaussianCopula / Chow-Liu | Each 100 lineages, fit11, default/native, 1,200 logical cells; 972 / 1,050 distinct physical receipts | Five-fit common coverage; official-test certification |

Sources, including every original metric receipt path and digest:

- `research/benchmark/results/arf-tabsyn-followon-validation-v1/manifest.json`
  and `panel.json`: complete ARF follow-on and the 95 retained TabSyn cells.
- `research/benchmark/results/tabsyn-eight-lineage-validation/panel.json`:
  the earlier 48 TabSyn measurements. The old and new cohorts do not overlap.
- `research/benchmark/results/tabddpm-twelve-lineage-retained-validation/panel.json`.
- `research/benchmark/results/forest-first-two-fivefit-v1/panel.json`,
  `forest-next-six-fivefit-v1/panel.json`, and
  `forest-third-five-fivefit-v1/panel.json`: default five-fit cohorts.
- `research/benchmark/results/retained-classical-validation/panel.json`:
  the original ARF/copula/Chow-Liu fit11 panel.

`generated/baseline-coverage.json` and `.tex` now use these publications.
`generated/published-baseline-kpis.json` and `.csv` trace each figure value
to a repository source digest and JSON pointer. The new TabSyn n cohort is
separate from the older 4n cohort. Its 22 incomplete groups retain individual
measurements but have no group mean or figure point. There are 35 complete
three-sample groups overall; this is not 49 complete n/4n lineages.

ARF has 1,200 logical cells at each fit seed. The 108 aliases are not extra
independent measurements. Three sample seeds are averaged within each fit;
means and unbiased sample SD are then computed across five independent fits.
Missing auditor support keeps both mean and SD null. SD is not a confidence
interval or a paired superiority result. The earlier 796 physical additional
ARF fit receipts remain in `arf-complete-additional-fits-v1/panel.json`;
the follow-on completes their common-metric coverage rather than fitting again.

## Numeric ETA checkpoint (MDT, UTC-06:00)

The newer receipt-backed checkpoint is
`research/benchmark/results/baseline-completion-20261008-v1/completion.md`.
It separates fit dispositions from metric cells and retains all 75 observed
Forest-Flow timeouts. [The protocol addendum](BASELINE_PROTOCOL.md) keeps the
original 600-second canonical fit cap and forbids selective higher-limit
replacement of those timeouts. This decision was recorded after observing
them; it is not retroactive preregistration.

Forecasts are operational planning, never estimated KPI values. All GPU
capacity checked at this checkpoint is occupied or reserved; no neural continuation
is admitted by this table. No existing worker is restarted or pre-empted.

| Baseline row | Next measurable/publication milestone | Numeric ETA / condition |
| --- | --- | --- |
| ARF | Complete five-fit common measurements closed; publish this frozen follow-on | Actual scalar close Oct 7 16:46 MDT; PR187 merged Oct 7 20:36 MDT |
| TabSyn | Publish all 95 already measured retained cells and partial-group accounting | PR187 merged Oct 7 20:36 MDT; no new compute needed |
| TabSyn | Full native-selection and five-fit population | Conservative 272.67 eligible GPU-slot hours: up to 800 native trials x 600 s plus 800 replication fits (400 default and 400 native-selected) x 600 s, plus 6 h sampling/evaluation. If a slot starts Oct 8 08:00 MDT: Oct 19 16:40 MDT; no slot granted, native-objective readiness still required |
| TabDDPM | Full default/native-selection and five-fit population | Conservative 287.67 eligible GPU-slot hours: up to 90 missing default fits, 800 native trials, 800 default/native replication fits, all at 600 s, plus 6 h sampling/evaluation. If a slot starts Oct 8 08:00 MDT: Oct 20 07:40 MDT; no slot granted, retained-OK jobs must first be reconciled |
| Forest-Flow | Close dispositions for the existing 400-fit CPU queue, then evaluate successful retained models | Frozen Oct8 02:02 MDT checkpoint: 343/400 = 268 OK +75 timeout. Fit-disposition forecast Oct8 10:11 MDT at 6.99 dispositions/hour; conditional metric allowance to Oct8 16:11 MDT. This includes missing timeout models, not 400 successful fits |
| Forest-Flow | Separate native-selected five-fit population / possible budget sensitivity | Not admitted. The former 289.33 CPU-slot-hour allowance (Oct20 09:20 MDT for Oct8 08:00 start) is a planning envelope, not permission to refill timeout cells or replace the canonical comparison. Native selection, reuse and any separate sensitivity matrix must be frozen first |
| Forest-Diffusion | Separate native/default population | Conservative 156 GPU-slot hours (900 capped fits plus 6 h evaluation); Oct 14 20:00 MDT if a slot starts Oct 8 08:00 MDT; no slot granted |

The newer frozen `research/benchmark/results/baseline-completion-20261008-v1/panel.json`
records 343/400 fits closed, 268 successful and 75 timeouts at 08:02:05Z
on Oct8. Its `rate.json` records seven dispositions over the preceding
60.05-minute observation window (6.99/hour); 57 remain at the newer snapshot.
The earlier 284-fit checkpoint remains unchanged in the follow-on publication.
The extrapolation is
`remaining / recent rate`; throughput changes and RAM/storage admission
move the ETA. Queue closure means every attempt has a disposition, not that
all models succeeded or that all lineages have five successful fits. A
timeout is a missing method cell; it contributes no DOPE win. Retries are
not admitted or automatically launched by this memo. The original timed-out
attempt receipts and costs remain retained.

GPU forecasts are cap-based allowances, not observed runtimes or guaranteed
dates. Each final method/dataset tuning cell remains capped at eight trials
and 12 hours, counting failures and timeouts. Native trials use the method's
own frozen KPI. Earlier architecture research, default fitting, and final
replication remain separately accounted; equal per-cell caps do not imply
equal total research spend. The six-hour evaluation allowances are planning
assumptions and are not summed into measured scientific compute costs.

## TabDDPM matched seeds and exact missing cells

`research/benchmark/results/tabddpm-matched-fit-coverage-v1/panel.json`
and `cell-coverage.csv` account for every lineage/selection/fit/sample/size
slot on the same 100 validation lineages as ARF. There are 132 measured
logical slots at fit11 and zero at each of fit23/37/53/71; 5,868 of the
6,000 target slots remain pending. Pending is not a method failure.
The joined comparison has 243 receipt-linked sample pairs. It averages
three sample seeds inside each matched lineage, and preserves native
selection, identical training/validation/projection hashes and real controls.
Detection uses the same frozen five-fold grouped protocol and splitter seed
1729; identical synthetic-row fold membership is not certified.

The CSV gives a numeric conditional ETA for **each** missing slot, relative
to the hypothetical Oct8 08:00 MDT admitted-GPU start used above. The current
order is CPU-only; no neural continuation is granted or launched. The final
cap-envelope date is Oct20 07:40 MDT, including a provisional six-hour metric
allowance. Historical trials count toward the eight-trial/12-hour cap; the
ledger neither permits another eight trials nor freezes unknown configurations.
The retained native-selection pools remain incomplete where originally recorded.
Scalar comparison outcomes contain no forecasts, new confidence intervals,
or superiority claims. The original broader research costs remain separate.

## Scientific limits

All measurements use training-derived validation. MFS-v2, PTF-v1,
release-safe L3, production certification, and superiority remain null.
Different dataset cohorts, fit counts and sample sizes do not support an
across-cohort ranking. The existing matched-eight paired analysis is preserved;
this publication does not replace it with a larger unmatched TabSyn cohort.
Empirical privacy attacks establish no formal DP or HIPAA claim. Original
sample rows, model weights and detailed logs remain restricted. This
publication authenticates scalar exports and original receipt identities;
it does not rehash current bulk weights or rerun scientific auditors.
