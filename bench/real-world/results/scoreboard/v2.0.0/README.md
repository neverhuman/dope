# DOPE v2.0.0 scoreboard

The README block `DOPE1 SCOREBOARD` and the v2.0.0 hero figures are generated from the
`dope_kpi` receipts in this directory. `python3 tools/claims/dope1.py --check`, which
`python3 tools/claims/check.py` also runs, refuses any drift between them.

| File | Written by |
| --- | --- |
| `primary.json` | `dope_kpi primary` (`dope.kpi-receipt/v2`): TRG of every arm with its family-bootstrap interval |
| `compare.json` | `dope_kpi compare`: paired comparisons, Holm-adjusted p, guards and the PROMOTE / HOLD / REJECT outcome |
| `null.json` (when scored) | `dope_kpi null`: the sign-flip null of a comparison |
| `readme-block.md` | `dope_kpi render-readme` from the receipts above; `README.md` carries it byte for byte |
| `../../../figures/hero-v2-light.svg`, `../../../figures/hero-v2-dark.svg` | `dope_kpi render-hero` from the same receipts |
| `render-manifest.json` | the SHA-256 of every receipt, of the block and of both hero files |

`dope1.py` checks that every receipt listed in the manifest lies in this directory,
hashes to its manifest entry and is named by its SHA-256 in the block footer, and that
the block and both hero files hash to their manifest entries. Receipts record no path,
clock or host.

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
`contracts/dope-kpi-v2.json`, which land with the `dope_kpi` change.

## Regenerate from the confirmation receipts

Run this one command from the repository root, with `FINAL` set to the directory that
holds the receipts of the one confirmation look and `KPI` to the `dope_kpi` binary named
in the pre-registration. It stops at the first failing step, and it refuses every
receipt set unless each receipt was scored on the CONFIRM families (`panel.id` is
`confirm`) before it copies anything.

```bash
FINAL=/path/to/confirm/receipts KPI=/path/to/dope_kpi bash -euo pipefail <<'SWAP'
D=bench/real-world/results/scoreboard/v2.0.0
NAMES="primary.json compare.json null.json"   # every receipt of the look, in render order
for n in $NAMES; do jq -e '.panel.id == "confirm"' "$FINAL/$n" > /dev/null; done
for n in $NAMES; do cp "$FINAL/$n" "$D/$n"; done
ARGS=$(for n in $NAMES; do printf -- '--receipt %s ' "$D/$n"; done)
"$KPI" render-readme $ARGS < /dev/null > "$D/readme-block.md"
# render-hero is not merged yet: keep these two outputs and adjust the flags to its --help.
"$KPI" render-hero $ARGS --theme light < /dev/null > bench/real-world/figures/hero-v2-light.svg
"$KPI" render-hero $ARGS --theme dark < /dev/null > bench/real-world/figures/hero-v2-dark.svg
sums() { sha256sum "$@" | jq -R 'capture("^(?<d>[0-9a-f]{64})  (?<p>.+)$") | {(.p): .d}' | jq -s add; }
jq -n -S --argjson receipts "$(sums $(for n in $NAMES; do echo "$D/$n"; done))" \
  --argjson files "$(sums "$D/readme-block.md" bench/real-world/figures/hero-v2-dark.svg bench/real-world/figures/hero-v2-light.svg)" \
  '{schema: "dope.scoreboard-render-manifest/v1", renderer: "dope_kpi render-readme", receipts: $receipts, files: $files}' \
  > "$D/render-manifest.json"
python3 tools/claims/dope1.py --write
python3 tools/claims/dope1.py --check
python3 -B tools/claims/check.py
SWAP
```

The script itself is read from standard input, so the `dope_kpi` calls read `/dev/null`
and cannot swallow its later lines. Leave `null.json` out of `NAMES` if the look has no
null receipt, and add any other receipt of the same run (for example a guards receipt). Remove receipts of an earlier
render that the new `NAMES` no longer lists, with `git rm`.
