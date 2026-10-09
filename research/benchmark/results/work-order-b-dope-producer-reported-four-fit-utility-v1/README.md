# DOPE producer-reported four-fit validation utility

This leaf describes 100 lineages with four additional fit seeds (23, 37, 53, 71), three sample seeds (101, 211, 307), and two validation sizes (n and 4n): 400 fit receipts and 2400 metric receipts. All 600 lineage/size/auditor cells remain in `result.json`, including incomplete cells. Metric values are producer-reported; historical evaluator and runtime parity are unverified.

| Auditor | Size | Complete four-fit lineages / all lineages | Median retention summary | Median four-fit sample SD |
|---|---:|---:|---:|---:|
| CatBoost | n | 99 / 100 | 0.7913409564 | 0.0416058583 |
| CatBoost | 4n | 99 / 100 | 0.8621311650 | 0.0431994291 |
| Linear | n | 94 / 100 | 0.9838033511 | 0.0300125376 |
| Linear | 4n | 94 / 100 | 0.9889167562 | 0.0289646334 |
| MLP | n | 88 / 100 | 0.9577763473 | 0.0464215164 |
| MLP | 4n | 88 / 100 | 1.1249498918 | 0.0529628502 |

Each fit median requires three successful, informative, finite reported retentions and finite reported losses. Missing, failed, low-signal, or invalid samples leave that fit median unavailable. The sample SD uses four complete fit medians and denominator 4−1. Within-fit ranges remain separate. The table's retention summary is the median across complete lineages of each lineage's median of four fit medians; its SD summary is the median across those same complete lineages. This conditional summary keeps the denominator of all 100 lineages visible. Retention is not clipped: negative and greater-than-one values remain in the result. Greater-than-one retention is not a superiority claim.

These summary rules are post hoc descriptive rules, not preregistered inferential gates. The producer dynamically imported an evaluator without recording its loaded bytes or runtime versions in these receipts. Configuration or source Git labels do not establish common-auditor parity. Verified common-auditor variation, five-fit stability, paired comparisons, MFS/PTF scores, privacy certification, and a quality winner remain null.

The separate seed-11 metadata comparison records 98 successful fits and two charged-cap outcomes. Data/configuration labels match 400 pairwise comparisons, but all 400 generator binary hashes differ; compiler/product-source equivalence and additional historical evaluator/runtime binding are missing. Binary differences alone do not prove algorithmic differences. Seed 11 remains unjoined. No GPU-training backend, production L3 charged bytes, or retained-storage claim follows from this leaf. Native and paper counts are unchanged.

`source-proof.json` binds canonical receipt-set digests from the additional-four-seed disposition publication, verified byte acquisition, independent helper reviews, real successful controls/reduction waits, and the metadata-only seed-11 comparison. Private captures, source rows, models, and receipt display names are excluded. No evaluation or fit is rerun by the renderer.

Reproduce `summary.csv`, this README table, and `summary.svg` from committed `result.json` with `python3 render.py --output-dir <new-directory>`. Verify existing bytes with `python3 render.py --check`. Matplotlib 3.6.3 and NumPy 1.26.4 are used for the standalone SVG with Agg, DejaVu Sans, fixed layout, path-embedded glyphs, fixed `svg.hashsalt`, and no metadata date. The figure shows only descriptive median four-fit SD with complete/all-lineage counts; it contains no retention or score axis.
