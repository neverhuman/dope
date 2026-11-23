# `dope_kpi` statistic commands (`dope-kpi/v2`)

`primary`, `compare`, `null` and `render-readme` turn the rows that
`dope_kpi real-ref` and `dope_kpi score` write into the scoreboard's primary
statistic, paired comparisons, a sign-flip null and the README block. The
protocol lives in [`contracts/dope-kpi-v2.json`](../../../contracts/dope-kpi-v2.json)
(`dope.kpi-contract/v2`), is compiled into the binary, and its SHA-256 is
written into every receipt. The contract is a **candidate** with
`protocol_frozen: false`; freezing it is a separate, owner-approved step.

Code: `src/kpi/{rundir,trg,boot,spec,rule,pairing,primary,compare,null,receipt,render,panelsel}.rs`.
Tests: unit tests in each module plus `src/kpi/stat_tests.rs`, and the
black-box test `tests/kpi_stat_cli.rs` on the static run directory
`tests/fixtures/kpi-stat/`.

## Commands

```text
dope_kpi primary --run-dir D --arm A [--arm B ...] [--panel all|dev|confirm --dev-tables F] --out R.json
dope_kpi compare --run-dir D --candidate C [--candidate C2 ...] --baseline B
                 [--seed-group NAME=C1,C2,C3] [--panel ... --dev-tables F] [--noise-sd X] --out R.json
dope_kpi null    --run-dir D --candidate C --baseline B [--replicates 2000] [--null-seed S]
                 [--replicates-out N.jsonl] --out N.json
dope_kpi render-readme --receipt R.json [--receipt R2.json ...]
```

Every statistic command also takes `--schedule FILE` (default: the run
directory's one `schedule*.jsonl`), `--contract FILE` (default: the compiled
contract) and `--seed S` (default: the contract's bootstrap seed). Each prints
a one-line JSON summary with the receipt SHA-256; `render-readme` prints the
markdown block. Errors name files, lines, cells and flags, never a value.

## Inputs

A run directory `D` holds `realref.jsonl`, the schedule and
`score/<arm>/rows.jsonl` per arm (formats.md section 7). Before any number is
computed, each rows file must hold exactly one entry per scheduled (cell,
sample seed, multiplier) and real-reference auditor, carry the schedule's cell
metadata, and repeat the real reference's real-only columns (`l0`, `l_a`,
`l_b`, `r_b`, `c2st_real_ref`, `informative`; relative tolerance 1e-12 for the
JSON round trip). Arms read from one run directory are therefore paired on
identical cells, splits and informative flags.

Guard fields a later `score` may add are read when present:
`realism_pass`, `privacy_pass` (per entry) and `exact_copy_excess` (per file).
No rows file written so far carries them, so those guards are "not
evaluated" and nothing can be promoted yet.

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

**Applicability.** An informative entry with status `unsupported` from a
`control` arm marks a control that cannot exist for that cell (for example
`twin_c` on a table with too few C rows). The cell leaves that arm's
statistic (`not_applicable_cells` counts it), and every comparison uses only
cells applicable to all the arms it reads.

**Reference match.** On `lite-prelim-20261122` the point estimates equal those
of `.agent/tools/kpi/prelim_summary.py` to better than 1e-15 for every arm
except `twin_c`, which the reference scores with its 88 inapplicable cells as
-1; with those cells removed the reference gives the Rust value.

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

## Sign-flip null

Under "no difference" each family's paired difference is as likely `+d` as
`-d`. Each of at least 2,000 replicates draws one fair sign per family
(seeded), flips that family's TRG, MLP and coverage differences, reruns the
bootstrap on the same draws and applies the full rule (Holm over this one
comparison; bytes and exact copies are properties of the candidate's files
and do not flip). The receipt reports, with Wilson 95% intervals:

- `false_promote`: the full rule (acceptable at most 5%);
- `flip_guard_rule`: every condition and every evaluated guard a flip moves,
  with bytes, exact copies and unevaluated guards taken as passing;
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
counts, cells, failed and not-applicable cells, MLP value, packet bytes
median / max / share within budget, unclipped median retention per auditor,
median C2ST); `comparisons`; `null`. No path, clock or host is written.

## README block

`render-readme` reads receipts of one run (same contract, schedule, real
reference, panel and bootstrap; an arm present in several must agree) and
prints a block between `<!-- BEGIN DOPE1 SCOREBOARD -->` and
`<!-- END DOPE1 SCOREBOARD -->`: one table row per arm, one row per
comparison (delta and interval, Holm p, guards, outcome), one line per null,
and a footer with the contract SHA-256, the panel id and every receipt's
SHA-256. Numbers have 3 decimals; the block is a pure function of the receipt
bytes.

## Panels

`--panel dev --dev-tables F` keeps every family holding a listed table
(listed tables map to families through the schedule, else through
`D/families.jsonl`); `--panel confirm` keeps the other scheduled families;
`--panel all` (the default) keeps all and refuses a list.

## Limits

- No rows file carries realism, privacy or exact-copy fields yet, so no
  comparison can PROMOTE until `score` writes them.
- Control arms carry no packet, so their byte guard is not evaluated.
- The contract is unfrozen; the percentile interval is anti-conservative on
  tasks with few families (see the null above).
