# `dope_kpi` statistic commands (`dope-kpi/v2`)

`primary`, `compare`, `null` and `render-readme` turn the rows that
`dope_kpi real-ref` and `dope_kpi score` write into the scoreboard's primary
statistic, paired comparisons, a sign-flip null and the README block;
`guards` adds the realism and privacy gates and the exact-copy excess that
the promotion rule's guards read, and `render-hero` draws the README hook
graphic from the receipts. The
protocol lives in [`contracts/dope-kpi-v2.json`](../../../contracts/dope-kpi-v2.json)
(`dope.kpi-contract/v2`), is compiled into the binary, and its SHA-256 is
written into every receipt. The contract is a **candidate** with
`protocol_frozen: false`; freezing it is a separate, owner-approved step.

Code: `src/kpi/{rundir,trg,boot,spec,rule,pairing,primary,compare,null,receipt,render,panelsel}.rs`
and, for `guards`, `src/kpi/{guards,guard_eval,guard_merge}.rs`; for
`render-hero`, `src/kpi/hero.rs` (receipts and flags) over the standard-library
drawing in `src/kpi/{hero_model,hero_layout,hero_labels,hero_svg}.rs`.
Tests: unit tests in each module plus `src/kpi/stat_tests.rs`,
`src/kpi/guard_tests.rs` (hand-made ledgers in `src/kpi/guard_fixture.rs`) and
`src/kpi/hero_tests.rs`, the black-box tests `tests/kpi_stat_cli.rs` and
`tests/kpi_hero_cli.rs` on the static run directory `tests/fixtures/kpi-stat/`,
and `tests/kpi_guards_cli.rs`, which writes its run directory from the same
hand-made ledgers.

## Commands

```text
dope_kpi primary --run-dir D --arm A [--arm B ...] [--panel all|dev|confirm --dev-tables F] --out R.json
dope_kpi compare --run-dir D --candidate C [--candidate C2 ...] --baseline B
                 [--seed-group NAME=C1,C2,C3] [--panel ... --dev-tables F] [--noise-sd X] --out R.json
dope_kpi null    --run-dir D --candidate C --baseline B [--replicates 2000] [--null-seed S]
                 [--panel ... --dev-tables F] [--noise-sd X] [--replicates-out N.jsonl] --out N.json
dope_kpi render-readme --receipt R.json [--receipt R2.json ...]
dope_kpi render-hero --receipt R.json [--receipt R2.json ...] --theme light|dark --out hero.svg
                 [--label ARM=Text ...] [--highlight ARM] [--previous ARM ...]
                 [--reference ARM ...] [--title TEXT] [--note TEXT]
dope_kpi guards  --run-dir D --arm A [--arm B ...] [--threads N]
```

Every statistic command (`primary`, `compare`, `null`) also takes `--panel`,
`--dev-tables`, `--schedule FILE` (default: the run directory's only
`schedule*.jsonl`; required when it holds several), `--contract FILE`
(default: the compiled contract) and `--seed S` (default: the contract's
bootstrap seed). A `--contract` file may change numbers (thresholds, draws,
seeds, quantiles, the byte grid, the auditor roster) but not the texts that
describe the method, which must equal the compiled contract's; only a contract
whose SHA-256 is compiled into `dope_kpi` as frozen (an owner decision; none
yet) may declare `protocol_frozen: true` or a status other than `candidate`.
Each command prints a one-line JSON summary with the receipt SHA-256;
`render-readme` prints the markdown block and `render-hero` a summary with the
point and reference counts, the unplotted arms and the SVG's SHA-256. Errors
name files, lines, cells, arms and flags, never a value.

## Inputs

A run directory `D` holds `realref.jsonl`, the schedule and
`score/<arm>/rows.jsonl` per arm (formats.md section 7). Before any number is
computed, each rows file must hold exactly one entry per scheduled (cell,
sample seed, multiplier) and real-reference auditor, carry the schedule's cell
metadata, and repeat the real reference's real-only columns (`l0`, `l_a`,
`l_b`, `r_b`, `c2st_real_ref`, `informative`; relative tolerance 1e-12 for the
JSON round trip). Arms read from one run directory are therefore paired on
identical cells, splits and informative flags. A rows file that lacks an entry
(an arm still being scored, a crashed shard) is refused, never read in part,
and a real reference without an entry for one of the contract's statistic
auditors is refused (the receipt records the reference's auditors as
`inputs.realref_auditors`).

The guard fields `realism_pass`, `privacy_pass` (per entry) and
`exact_copy_excess` come from `dope_kpi guards` (next section): when
`score/<arm>/guards.jsonl` exists, loading the arm merges its flags into the
rows by `ledger_id` and sets every row's `exact_copy_excess` to that of its
synthetic file in `guards-summary.json`. Without that file the rows are read
as written (no rows file carries the fields), and the realism, privacy and
exact-copy guards are `not_evaluated`, so nothing can be promoted.

## Guards (`dope.kpi-guards/v2`)

`guards` reads `score/<arm>/ledger.jsonl` (the `FidelityCellRequest` of every
entry, as `score` wrote it) and writes next to it, without touching the
ledger or `rows.jsonl`:

- `guards.jsonl`: one line per ledger entry in ledger order: `ledger_id`,
  `cell_id`, `status`, `realism_pass`, `privacy_pass`, the exact-copy counts
  (`synthetic_rows`, `exact_copies`, `twin_rows`, `twin_exact_copies`),
  dope-core's two assessments and, for an input dope-core refused, its
  refusal code;
- `guards-summary.json`: entries by status, per gate the passes, assessed,
  absent and refused entries, distinct assessments and the pass share over
  every entry and over the complete ones, `file_exact_copy_excess` (one value
  per synthetic file `<cell>:s<seed>:m<multiplier>`), the exact-copy counts,
  rates and `pooled_exact_copy_excess` of the whole arm, the rule text, the
  tool, and the SHA-256 of the ledger, the rows file and `guards.jsonl`.

The gates are dope-core's, applied as `dvs::v3::evaluate_v3` applies them
when partial evidence is not allowed; no threshold is set here:

| Gate | Passes iff | Source |
| --- | --- | --- |
| realism | `assess_realism(..).pass_standard`: no illegal row, C2ST AUC <= 0.70, max marginal AUC (folded, `max(a, 1-a)`) <= 0.60, synthetic/real rule-violation ratio <= 2.0 when both rates are present | `dope-core/src/dvs/v3/realism.rs` |
| privacy (empirical) | `assess_empirical_privacy(..).pass`: share of synthetic rows nearer the first `m` rows of A than the first `m` of B, `m` the smaller of the two (ties count half), <= 0.55, distance MIA AUC <= 0.55, MIA TPR at 1% FPR <= 0.02, this file's exact-copy excess <= 0.005, no canary hit, emitted support >= 10 | `dope-core/src/dvs/v3/privacy.rs` |

These are the realism and privacy gates of
`contracts/dope-fidelity-v3-candidate.json`. `c2st_real_ref` enters no gate.
A complete entry without a realism (privacy) input, or with one dope-core
refuses, fails that gate; a failed, over-budget or unsupported entry fails
both and holds no counts. The privacy input is what `score` wrote: the first
`--privacy-rows` (default 4,096) rows of A, B and S in unit space, rounded to
6 decimals, no canaries, and the emitter's declared support, else the floor
10 (`score.rs`). The privacy result is an empirical audit, never formal
differential privacy or HIPAA de-identification.

An exact copy is a synthetic row within Euclidean distance 1e-12 of one of
those `m` rows of A; the twin's are the `m` rows of B within 1e-12 of them.
A synthetic file's

```text
exact_copy_excess = max(0, sum exact_copies / sum synthetic_rows
                           - sum twin_exact_copies / sum twin_rows)
```

sums over the file's complete entries with an assessed privacy input (its
auditors share one input, so this is dope-core's per-file
`exact_copy_excess`); a file without such an entry has no value, and its rows
none. Every row of the file carries the file's value, so the promotion guard
is the contract's `promotion.exact_copy`: the mean over the candidate's scored
headline files at multiplier 4 on the comparison's common cells. The summary's
`pooled_exact_copy_excess` applies the formula to every complete entry of the
arm (all strata, multipliers and auditors); it is descriptive and no guard
reads it.

The three auditors of one synthetic file (cell, sample seed, multiplier)
carry the same realism and privacy inputs. The ledger is sorted by entry id,
so a file's entries are consecutive: `guards` streams them one file at a time,
`--threads` files in parallel, and assesses each distinct input of a file once
(bit-for-bit equality; the summary's `assessments`). The bytes do not depend
on `--threads`, and a rerun writes the same bytes.

Before writing, every ledger entry must find the row with its `ledger_id`,
and that row the same cell and status, and every row an entry. When the arm is
loaded, every row's `ledger_id` must name its own cell, sample seed,
multiplier and auditor, every row must find its line, every line its row
(same cell and status), the summary must name the arm and hold the SHA-256
of both `guards.jsonl` and the current `rows.jsonl`, and the rows must not
carry guard fields of their own. Any mismatch stops the command with the
file and line numbers, never a value; a changed rows file asks for `guards`
to be rerun. Receipts then record the SHA-256 of each arm's two guard files.

On `lite-prelim-20261122` (seven arms of 2,022 entries; 4.5 minutes at
`--threads 4` under nice 19 on a loaded host, 96 MB peak memory) `copy_a`
and `bootstrap_a` fail privacy on every complete entry with excess 1, and
`twin_c` passes on 93% with no exact copy: two of its cells miss the TPR
bound by one row (21 of 1,024) and one has a 34-row twin. `indep_marginals`,
a column-wise permutation of A, passes on only 44%: its rows are made of A's
values, which the distance tests detect on small tables (94% of its files
with fewer than 64 matched rows fail, 14% of those with 1,024 or more), and
on one table of a single feature and a binary target 329 of its 347
permuted rows are rows of A.

## Statistic (TRG)

For each informative (cell, auditor) unit of the headline stratum at one
multiplier (4 primary; 1 reported as `matched`):

```text
x = mean over sample seeds of v
v = clip(R_S, -1, 1.25) - R_B   for a scored entry (status ok)
v = -1                          for failed, over-budget, single-class or a generator's unsupported entry
```

A cell takes the minimum of `x` over `linear-seeded-v1` and `gbdt`;
`mlp-seeded-v1` is aggregated the same way, held out, and read only by the
direction guard. Cells average within a family, families within a task, and
the tasks present average equally. Units that are uninformative on real data
leave every arm.

**Applicability.** An informative entry with status `unsupported` and class
`control` from a native control arm (`copy_a`, `bootstrap_a`,
`indep_marginals`, `label_shuffle`, `twin_c`: generated by `dope_kpi score`
itself) marks a control that cannot exist for that cell (for example `twin_c`
on a table with too few C rows). The cell leaves that arm's statistic
(`not_applicable_cells` counts it), and every comparison uses only cells
applicable to all the arms it reads. Any other arm that leaves an entry
unsupported has failed it (-1), whatever class its emit log claims.

**Cell accounting.** Per arm and multiplier `scheduled_cells =
uninformative_cells + cells + not_applicable_cells`; `uninformative_cells`
have no informative statistic auditor, and `mlp_only_cells` of them have an
informative held-out auditor (read by the MLP guard only). On the prelim
panel's headline stratum a generator has 167 = 42 + 125 + 0 (one of the 42
informative for the held-out auditor only) and `twin_c` 167 = 42 + 38 + 87.

**Failed entries.** A scored entry can fall to `-1 - R_B` (about -2), below the
-1 of an entry that was not scored, so an arm that withholds its worst files
gains. The protocol of record keeps the constant -1 (the contract's
`failed_value`, the reference implementation and the pre-registration use
it); every arm summary reports `trg_failed_at_floor`, its TRG with a failed
entry at `clip_low - R_B`, as a diagnostic for the owner's decision.

**Byte budgets.** The statistic reads the rows as scored. An arm scored with a
cap above the headline budget (the prelim run scored `dope-1.3.0-medium` at
4,096 B and `-large` at 16,384 B; third-party baselines are not byte-capped)
reports `bytes.scored_above_budget` and `trg_at_budget`, its TRG with those
packets over budget; the README marks it with `†`. Its byte guard fails as a
candidate; as a baseline the comparison reports `baseline_above_budget`.

**Reference match.** On `lite-prelim-20261122` the point estimates equal those
of `.agent/tools/kpi/prelim_summary.py` to better than 1e-15 for every arm
except `twin_c`, which the reference scores with its 87 inapplicable cells of
the statistic as -1; with those cells removed the reference gives the Rust
value.

## Uncertainty

A stratified family bootstrap: each of 10,000 draws resamples the families of
every task with replacement and recomputes the task-macro mean. The draws
depend only on the family counts and the recorded seed, so both arms of a
comparison and every null replicate read the same draws. With `n` sorted
draws the 95% interval is the draw at rank `floor(0.025 (n-1))` to the draw at
rank `ceil(0.975 (n-1))` (249 and 9750), and the one-sided p for `delta <= 0`
is `(#{draw <= 0} + 1) / (n + 1)` (the `dvs::v3::qualify` conventions).

## Comparisons and the promotion rule

`compare` pairs each candidate with the baseline on their common cells and
bootstraps the per-family differences (candidate minus baseline). Holm's
step-down adjustment runs across every comparison of one receipt. A
`--seed-group NAME=A,B,C` pools training seeds: per cell the candidate value
is the mean over the seeds on cells common to all seeds and the baseline; each
seed's own delta is reported.

PROMOTE iff all hold:

1. the 2.5th percentile of the delta is above 0;
2. the Holm-adjusted p is below 0.05;
3. the delta is at least `max(0.01, noise_sd)` (`noise_sd` from `--noise-sd`,
   else the contract default 0.01; the receipt records which);
4. every guard passes: each task's lower bound above -0.02; held-out MLP
   delta at least 0; realism and privacy coverage deltas above -0.03;
   exact-copy excess at most 0.005; every produced headline file within
   2,048 bytes;
5. for a seed group, at least 3 seeds, each with the pooled delta's sign.

REJECT iff the 97.5th percentile is below 0; HOLD otherwise. A guard whose
input is absent is `not_evaluated` and blocks PROMOTE. The byte guard counts a
file as produced when it was scored or carries a measured packet: it fails
when any measured packet exceeds the budget, and is not evaluated when a
produced file has no measured packet (a control) or nothing was produced.
`score` logs a native Format 2 artifact over its arm's budget as `over_budget`
with the artifact length (returned by the engine or named in its budget
refusal), so the guard sees it; a file over budget without a measured size is
not produced under the contract text and is reported as
`bytes.over_budget_unmeasured`.

Reported beside each decision and never read by it (the rule is the
contract's and the pre-registration's):

- `null_calibration`: the comparison's own sign-flip null (below) with the
  contract's 2,000 replicates and null seed; `flip_guard_rule.rate` is its
  false-PROMOTE rate, `acceptable` when at most 5%. With `k` families no rate
  is below `2^-k` for a comparison the rule promotes (1/16 for the four
  families of `tests/fixtures/kpi-stat`).
- `exact_copy`: over the candidate's scored headline files on the common
  cells, the mean the guard reads, the maximum and the share of files above
  0.005.
- `holm_m`: the comparisons the Holm adjustment covered.

## Sign-flip null

Under "no difference" each family's paired difference is as likely `+d` as
`-d`. Each of at least 2,000 replicates draws one fair sign per family
(seeded), flips that family's TRG, MLP and coverage differences and each
seed's difference of a seed group, reruns the bootstrap on the same draws and
applies the full rule (Holm over this one comparison; bytes and exact copies
are properties of the candidate's files and do not flip). The receipt
reports, with Wilson 95% intervals:

- `flip_guard_rule`: every condition and every evaluated guard a flip moves,
  with bytes, exact copies and unevaluated guards taken as passing;
  `acceptable` is this rate at most 5%;
- `false_promote`: the full rule; absent when a guard is not evaluated or a
  fixed guard fails (listed in `not_evaluated_guards` and
  `failing_fixed_guards`), since PROMOTE is then blocked in every replicate
  and a rate of 0 says nothing about the statistic; otherwise it equals
  `flip_guard_rule`;
- `statistical_pass`: conditions 1-3 alone, an upper bound on the rule's rate
  for any guard inputs.

`--replicates-out` writes one JSON line per replicate (delta, bounds, p,
outcome) and records its SHA-256 in the receipt.

On the prelim panel (`dope-1.3.0-medium` vs `dope-1.3.0-small`, 117 families:
94 regression, 16 binary, 7 multiclass; 2,000 replicates) conditions 1-3 alone
pass in 6.2% of replicates (Wilson 95% 5.2-7.3%) against a nominal one-sided
2.5%: the percentile bootstrap is too narrow for the seven-family task that
carries a third of the weight. With the per-task and MLP guards the rate is
0.55% (0.31-0.98%), and the full rule is 0% because the byte guard fails and
three guards are not evaluated. An independent numpy re-implementation of the
null gives the same conditions-only rate (6.15%).

## Receipt (`dope.kpi-receipt/v2`)

One JSON document with sorted keys, two-space indent, every float rounded to
12 decimals and no negative zero, so the same inputs and binary give the same
bytes. It records: `command`; `tool` (name, version, `git_sha` from
`DOPE_KPI_GIT_SHA` at build time, `unset` otherwise); `contract` (id,
SHA-256, source, status, frozen flag); `inputs` (schedule file name and
SHA-256, the schedule's panel SHA-256, real-reference SHA-256, SHA-256 of every
rows file read); `panel` (id, dev-table list SHA-256 and match counts,
`families.jsonl` SHA-256, family count and the SHA-256 of the kept family ids);
`bootstrap` (draws, seed, quantiles, ranks); `noise_sd`; `arms` (one summary
per arm: class, primary and matched statistic with per-task values and family
counts, the cell accounting above, MLP value, packet bytes median / max /
share within budget with the over-budget counts, unclipped median retention
per auditor, median C2ST, `trg_at_budget`, `trg_failed_at_floor`);
`comparisons` (with `holm_m`, `exact_copy`, `null_calibration`,
`baseline_above_budget`); `null`. No path, clock or host is written.

When an arm read has guard files, `inputs` also holds `guards_sha256` and
`guards_summary_sha256` per arm, and that arm's summary holds
`realism_coverage` and `privacy_coverage` (the statistic's aggregation of the
per-cell pass shares over the informative `linear-seeded-v1` and `gbdt`
entries at the primary multiplier, a non-ok entry counting as a miss) and
`exact_copy_excess` (the mean of the file excesses over its scored headline
files at the primary multiplier on the cells in the statistic) with
`exact_copy` (that mean, the maximum file and the share of files above 0.005).
Without guard files these keys are absent.

A release binary stamps its commit when built from a clean tree in `engine/`
with `DOPE_KPI_GIT_SHA=$(git rev-parse HEAD) CARGO_INCREMENTAL=0 cargo build
--release --locked -p dope-arena --bin dope_kpi`; cargo rebuilds `dope_kpi`
whenever the variable changes.

## README block

`render-readme` reads receipts of one run (same contract, schedule, real
reference, panel and bootstrap; an arm present in several must agree) and
prints a block between `<!-- BEGIN DOPE1 SCOREBOARD -->` and
`<!-- END DOPE1 SCOREBOARD -->`: one table row per arm (including realism /
privacy coverage and exact-copy excess mean with the maximum file in
parentheses; `†` and a note for arms scored above the headline budget), one
row per comparison (delta and interval, Holm p with the number of comparisons
it covers, every guard with its value, the sign-flip false-PROMOTE rate,
outcome), one line per null, and a footer with the contract (id, source,
status, SHA-256), the panel id and every receipt's SHA-256. Numbers have 3
decimals; the block is a pure function of the receipt bytes.

## README hero graphic

`render-hero` draws the README hook graphic from receipts of one run (the
`render-readme` checks) as a 960 x 540 SVG, one file per `--theme` (`light`
or `dark`; a README picks one with `<picture>` and `prefers-color-scheme`).
x is an arm's `bytes.median` on a log10 axis with ticks from 100 B to 10 MiB,
the needed range padded to the nearest ticks. For a third-party arm that
median is the fitted model file, scored without the headline cap, so the
2,048-byte budget is a dashed vertical reference line, never a filter. y is
the primary TRG with its 95% interval as a whisker, gridlines every 0.25 and
0 always in range. Every arm with a byte median and a TRG is a point with a
direct label (`--label`, default the arm id): the `--highlight` arm wears the
accent and a bold label with its TRG to 3 decimals, `--previous` arms
(default: arms named `dope-X.Y.Z-...` with a lower release than the
highlight) a lighter step of the accent, the rest neutral gray. `--reference`
arms (controls without bytes) are dotted horizontal lines labelled at the
right end; any other arm without a byte median is listed as `unplotted`. A
small key names the four kinds of mark. Labels are placed in a fixed order
(budget, controls, highlight, earlier releases, the rest) on the first
candidate box (right, left, above, below a point, then one to three rows up
or down; above or below a control line, sliding left from its right end)
that stays in the plot, overlaps no earlier label, marker or whisker, and
keeps off the control and budget lines; failing that, the box that overlaps
least. The title defaults to "Predictive structure kept vs. artifact size";
`--note` is the subtitle, with `{cells}`, `{families}` and `{panel}` filled
from the highlight's statistic and the panel id; it wraps at spaces, so a
no-break space keeps words together. The footer holds the SHA-256 prefixes of
the contract and of every receipt. Colors are theme tokens and text wears
only the ink tokens. Coordinates have one decimal and arms keep receipt
order, so the same receipts and flags give the same bytes; the file holds
labels, numbers and digests only, no path, host, source value or header.
Label boxes are sized with DejaVu Sans metrics, the widest common `system-ui`
fallback, so narrower faces leave more room.

## Panels

`--panel dev --dev-tables F` keeps every family holding a listed table
(listed tables map to families through the schedule, else through
`D/families.jsonl`); `--panel confirm` keeps the other scheduled families;
`--panel all` (the default) keeps all and refuses a list.

## Limits

- No comparison can PROMOTE until `guards` has run for its candidate and
  baseline.
- The gates' fixed bounds leave little room for sampling noise on small
  tables: even for synthetic rows exchangeable with the real ones, the DCR
  share over `n` of them has a binomial standard deviation near `0.5 / sqrt(n)`
  (0.06 at `n = 64`, against the 0.55 bound), and on the prelim run the
  real-data controls fail realism on most files with fewer than 64 matched
  rows. The coverage guards compare paired deltas, which cancels part of this.
- `guards` checks the rows file's SHA-256, not the ledger's, when an arm is
  loaded: a ledger rescored with other privacy inputs but identical rows is
  not detected until `guards` is rerun.
- Control arms carry no packet, so their byte guard is not evaluated.
- The contract is unfrozen; the percentile interval is anti-conservative on
  tasks with few families (see the null above). The rule does not read a
  comparison's own sign-flip rate (`null_calibration`, reported beside it):
  making it a guard is an owner decision and a change of the pre-registered
  rule.
- In the hero's dark theme the accent `#3987e5` and its lighter step
  `#5598e7` are close (OKLab difference about 5.5, below the 15 that keeps
  two marks apart at a glance); the highlight is told apart by its larger
  marker and bold label. A control line within a label height of another
  line or gridline can only be labelled on its far side, which may touch
  that neighbour.
