# DOPE v3.0.0 scoreboard

The README block `DOPE1 SCOREBOARD` and the v3.0.0 hero figures are generated from the
`dope_kpi` receipts in this directory. `python3 tools/claims/dope1.py --check`, which
`python3 tools/claims/check.py` also runs, refuses any drift between them. The v2.0.0
scoreboard and its receipts stay in [`../v2.0.0/`](../v2.0.0/README.md).

| File | Written by |
| --- | --- |
| `primary.json` | `dope_kpi primary` (`dope.kpi-receipt/v2`): TRG of every arm with its family-bootstrap interval |
| `compare.json` | `dope_kpi compare`: the one pre-registered comparison, its Holm-adjusted p, guards and outcome |
| `readme-block.md` | `dope_kpi render-readme --ranked` from the receipts above. `README.md` carries it byte for byte. |
| `../../../figures/hero-v3-light.svg`, `../../../figures/hero-v3-dark.svg` | `dope_kpi render-hero` from the same receipts |
| `render-manifest.json` | the SHA-256 of every receipt, of the block and of both hero files |

## What this look is

**A DEV estimate, not a sealed test.** Under the owner's pause on CONFIRM, HOLDOUT and SEALED
panels, every v3 number comes from the DEV-SELECT tables. The scoreboard is one pre-registered
look at fresh row splits (data seeds 5 to 8) of the same DEV tables whose other splits (data
seeds 1 to 4) selected the v3 encoder. Those fresh splits were planned before selection and
scored once, after it was final. No guard, threshold or parameter was tuned on them.

The selection, all on DEV and in this order, was written down before each step was scored:

1. Screen on the 12 informative DEV-lite cells.
2. Decide on the 108-cell wide DEV schedule: `--format auto-v8` replaces `--format auto` as
   the release encoder only if its TRG minus v2.0.0 `--format auto` is at least `+0.010` with
   the paired family-bootstrap 2.5 % bound above zero. It passed.
3. Report once on the fresh splits. An upgrade is claimed only if the 95 % interval of that
   difference excludes zero.

The arm `dope-v3b-auto-v8` is `dope fit --format auto-v8 --budget 2048` built from the v3
integration branch. A local static-pie build of `dope 3.0.0` from `3fb802fd` reproduced
all 108 scheduled packet SHA-256 digests. The signed draft binary must reproduce
the same digests before publication.

**Read the guards with the headline.** The comparison's outcome under the `dope-kpi/v2`
promotion rule is HOLD, not PROMOTE. TRG improves, but four guards fail: the regression lower
bound, the held-out MLP auditor, realism coverage and privacy coverage. The block reports each
one. The release ships `--format auto-v8` as the encoder with the best TRG at 2 KiB, with those
trade-offs stated. `--format 1` remains the CLI default.

## TRG, the twin-referenced gap

The scoreboard's primary statistic. Each scheduled cell splits one table into disjoint
row roles with no duplicate group straddling two roles: A (the rows a generator sees),
B (a real twin with as many rows as A), C (calibration) and T (held-out test rows).
An auditor fitted on rows `x` and scored on T has retention

```text
R_x = (L0 - L_x) / (L0 - L_A)      L0: null loss, L_A: auditor fitted on A
v   = clip(R_S, -1, 1.25) - R_B    S: rows sampled from the arm's artifact
v   = -1                           failed, over budget, single class or unsupported cell
```

A cell takes the smaller value of the linear and boosted-tree auditors (the MLP auditor
is held out for a guard). Cells average within a lineage family, families within a task,
and the tasks present average equally. Zero means the sampled rows train the auditors as
well as a second real sample of the same size; higher is better. Intervals come from a
stratified family bootstrap, and comparisons are Holm-adjusted. The full protocol is the
`dope_kpi` documentation in `engine/dope-arena/docs/kpi-v2.md` and the contract
`contracts/dope-kpi.json`.

The README presents the same statistic as **Transfer Score = 100 + 100 × TRG**,
in points. The 4× score sets 100 at parity with the real twin and preserves the
TRG ordering. Paired gains and interval bounds are in points on the same scale.

## Regenerate from the receipts

Run from the repository root, with `KPI` set to a `dope_kpi` binary. It re-renders the
block and both hero files from the committed receipts into a staging directory under
`target/` and compares them with the committed files.

```bash
KPI=/path/to/dope_kpi bash -euo pipefail <<'CHECK'
D=bench/real-world/results/scoreboard/v3.0.0; F=bench/real-world/figures
mkdir -p target; S=$(mktemp -d target/scoreboard-v3.XXXXXX)
args=(--receipt "$D/primary.json" --receipt "$D/compare.json")
labels=(--label "dope-v3b-auto-v8=DOPE 3.0.0 auto-v8"
  --label "dope-2.0.0-auto=DOPE 2.0.0 auto" --label "dope-2.0.0-small=DOPE 2.0.0 small"
  --label "dope-1.3.0-small=DOPE 1.3.0 small" --label "pr390-v7-step0=V7 step 0"
  --label "dse8-p0-v3b=DSE8 analytic")
"$KPI" render-readme --ranked "${args[@]}" "${labels[@]}" < /dev/null > "$S/readme-block.md"
hero=(--highlight dope-v3b-auto-v8 "${labels[@]}" --previous dope-2.0.0-auto --previous dope-2.0.0-small
  --previous dope-1.3.0-small --reference copy_a --reference bootstrap_a
  --note "DEV estimate: fresh row splits of the 22 DEV tables, {cells} cells, {families} families (no CONFIRM look)")
for t in light dark; do "$KPI" render-hero "${args[@]}" --theme "$t" --out "$S/hero-v3-$t.svg" "${hero[@]}" < /dev/null; done
cmp "$S/readme-block.md" "$D/readme-block.md"
for t in light dark; do cmp "$S/hero-v3-$t.svg" "$F/hero-v3-$t.svg"; done
python3 tools/claims/dope1.py --check
CHECK
```
