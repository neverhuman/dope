# Fixed-budget transfer and learned-tier training

These descriptive results use the same development report schedule and real reference as the v3.0.0 scoreboard. Each fitted method sees the same fitting rows, and each evaluator tests on the same real rows. Both methods are evaluated at 1× and 4× sampling. Each interval is a separate 95% lineage-family bootstrap interval. The experiment does not define a paired superiority test.

## Fixed-budget transfer

The signed v3.0.0 executable ran `fit --format auto-v8 --budget B` at each whole-file cap. The 2,048-byte run reproduced the release replay's 216 scheduled emission statuses, byte counts and packet SHA-256s. The retained-rows reference stores uniformly sampled fitting rows without replacement, including the target in each record. At sampling it draws from those records with replacement. The reference uses SplitMix64 with the scheduled fit and sample seeds and saves row values losslessly in `DOPEMAT1`. The private file audit includes its 24-byte header and every stored row. A schema too wide for one complete row is a failed fit. Scored failures count at the contract floor for both methods.

- `served.csv` and `rows.csv` contain Transfer Score points and family intervals at 4× (`score`, `low`, `high`) and 1× (`matched`, `matched_low`, `matched_high`). Actual byte statistics and artifact counts cover the 64 scheduled headline fits at each cap. Failed and informative counts use the respective score denominators. `retained_rows_median` applies to the row reference. Each point cites its bound evidence and primary receipt by SHA-256.
- The cap grid in bytes is 256, 512, 1,024, 2,048, 4,096, 8,192 and 16,384. The cap counts the complete fitted artifact. No scorer-derived bytes are used for the stored-rows artifacts. Their sizes come from the retained-file audit. The score calculation uses the same frozen real reference at every cap.
- `rows_at_budget.rs` is the native retained-rows reference used in this experiment. Build it with `rustc --edition=2021 -O rows_at_budget.rs -o rows-at-budget`. The two commands are `rows-at-budget fit FIT_MATRIX CAP_BYTES FIT_SEED ARTIFACT` and `rows-at-budget sample ARTIFACT SAMPLE_ROWS SAMPLE_SEED OUTPUT_MATRIX`.

The 4 KiB and 8 KiB DOPE runs each recorded four informative headline fit timeouts. The 1,800-second timeout is wall-clock time. Both runs were paused and then resumed by the host CPU guard. The affected cells retain their recorded failures at both multipliers. The 2 KiB case completed every fit on the schedule and matches the signed replay. Runtime-sensitive comparisons across caps should be read in light of these timeouts.

| Bound input | SHA-256 |
| --- | --- |
| Addendum K protocol | `dede108ef1438353af250bc8dd6f54286eaf1c0b94a37146c92c4216355a52f7` |
| Budget experiment freeze | `1a7f13345d57c43926f6b159e946f6b79904e35b2ded11b834080ca273062845` |
| Report schedule | `14238f1d5b9ef7182f862d4aae63d06f3d512a29e6dfebfdb02896735bd20570` |
| Frozen real reference | `7d377b8ab47f7fb550c4a76491b9ed58ce4e8b704302524d34d2ae037657da16` |
| Scoring executable | `c01fd919e189a79cd9a913cbfbfdaf099ba56381f55ab49e163e6f1fff78d153` |
| Signed DOPE executable | `8723955b1fcd59cab8996842a39b3a579c596c957ac1f89f291b0a9d1ebfaea0` |
| Native row-reference source | `a31954415c2c58f76cd8a9f79ad31da4fe7c67ffdb8cf57f4be450c8bcb2f081` |
| Native row-reference executable | `aa79b1f6f8fdfc1a7d24ecc32ec0537fc1e3688f4db2d678fce1bcecd987221c` |
| 2,048-byte replay status, size and digest tuples | `396be7b11702271a845f0c3549a8be967fda9eb9cfcbca0c2c17db5e10a68e48` |
| Independent byte-for-byte reduction of the first eleven results | `baf52abb7f21864a892c38590edd9e4980f72a7bbd3cca564945a2ed9a77dda2` |
| Independent byte-for-byte reduction of the remaining three results | `b3b0b70136f1602c7e2bfda42e9882e0502295e12f375aceb7147c797d0d7a36` |

The score reducer requires private per-cell artifacts and ledgers. Public inputs here contain summary values and digests. Report results apply to the stated benchmark and raw-row retention policy.

`panel-report.csv` summarizes the scheduled headline fits from the pinned report schedule. It uses each cell's projected feature count and fitting and test row counts. The raw fitting-matrix size is the 24-byte `DOPEMAT1` header plus four bytes per feature or target entry. The size ratio is the median of the per-cell raw-matrix-to-artifact ratios on the verified 2,048-byte replay. It uses this particular numeric representation of the input, including the whole-file header.

## Codec attribution

`k3-format1.csv` records the signed engine forced to Format 1 on the report schedule and the same cap. Its file framing is the direct Format 1 output. The primary receipt, result and evidence bind digests are in the CSV. An independent run matched all 216 emission keys against the schedule and reproduced the native primary byte for byte from bound score and guard files. The two other configurations in the paper's table come from the release report receipt. The comparison is descriptive.

`k3-census-public.json` counts the selected codecs by task and stratum across all 108 signed report fits. An independent census matched every packet digest to the original emission ledger before reading the container tag and weights fingerprint. `k3-group.json` binds the full and partitioned reductions of the original score files for all cells, the Format 1 selections and the analytic DSE8 selections. The independent check regenerated the four filtered inputs, both guard files and the native primary receipt for each group byte for byte. The full replay reproduces the released score. `k3-group.csv` projects the public aggregates used by the paper's grouped table. Differences reflect the cells the selector assigned to each codec, including differences in task mix.

`k33-result.json` records a separate partition based on the two scheduled emissions of forced Format 1. It has 85 successful fits and 23 fit exits. The independent check classified all 216 keys, re-filtered the original schedules and ledgers, reproduced all 35 bound input and receipt files byte for byte and checked the results arithmetic. `k33-summary.csv` projects the public aggregates used in the paper. The pairwise differences are conditional on Format 1 fit status, and they compare the complete configurations, including their framing.

## Learned refinement pilot

- `k5-plot-inputs.json` contains the plotted medians over non-overlapping windows of 200 applied updates and the paired Transfer Score differences by schedule. The telemetry spans three seeds at 8,000 corrected-objective updates and three controls at 2,000 updates, with no skipped nonfinite updates.
- `k5-training-summary.csv` counts contiguous applied updates and reports median elapsed training hours per arm from the bound telemetry. Its paired result uses the unrounded fresh-split point and interval from the plot inputs.
- `k5-public.json` binds telemetry, native score receipts, plotted inputs and render outputs by SHA-256. `training-k5r2.pdf` is the bound figure used in the paper. The final fresh-seed DEV result did not promote the learned refinement. The signed v3.0.0 engine remains the release.
