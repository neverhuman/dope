# Measured matched validation utility

This descriptive comparison uses committed utility receipts from the same 100
rights-cleared S3 regression lineages and their official-training-derived
validation splits. Official tests remain sealed.

DOPE uses the fixed `features12_steps2048` configuration. ARF, Gaussian copula,
Chow-Liu, and independent marginals use their source panels' native-selected
configurations. This table does not select a DOPE family, change native tuning,
or rank different methods' native KPIs.

For each auditor and sample size, the exporter takes each lineage's median over
sample seeds 101, 211, and 307, then the median over lineages informative and
available for **all five methods**. Fit seed is 11. The corresponding real-vs-real
MSE controls must agree between methods. Source coverage includes two DOPE
lineages with unavailable fits; their missing values remain null and they do not
contribute comparison wins. The complete source coverage and excluded lineages
are retained in `publication.json`.

The 4n descriptive retention medians are:

| Method | CatBoost (97 lineages) | Linear (92) | MLP (75) |
|---|---:|---:|---:|
| DOPE | 0.939876 | 0.994046 | 1.061123 |
| ARF | 0.825507 | 0.936043 | 0.929301 |
| Gaussian copula | 0.655681 | 0.989300 | 0.927569 |
| Chow-Liu | 0.585986 | 0.661218 | 0.691645 |
| Independent marginals | -0.011971 | 0.002056 | -0.027881 |

Retention is unclipped. A value above 1 does not imply a production score pass.
Artifact bytes are recorded, but this quality comparison is unconstrained by the
L3 byte cap. No confidence interval or paired significance is claimed. Full
neural comparison and five-fit coverage remain incomplete. MFS-v2, PTF-v1,
release-safe, and superiority claims remain null.

`figure-kpis.csv` supplies 30 method/auditor/sample-size summaries for paper
figures. `measured-validation-utility-4n.svg` and `.pdf` are standalone exports;
the paper lane chooses whether and how to use them. No paper sources are edited.

Regenerate the table and schemas from the frozen committed CSV:

```sh
python3 -B -m research.benchmark.publish_validation_figure --output-dir target/validation-figure-replay
python3 -B -m unittest research.benchmark.tests.test_validation_figure -v
```

Add `--figures` to render the SVG/PDF. Rendering requires Matplotlib 3.10.8,
FreeType 2.6.1, and the declared DejaVu Sans font hash. The figure receipt records
these versions, publisher/source hashes, and output hashes. Exact regeneration
is verified in this declared rendering environment. No model, sample, real-row,
or test payload is opened by this exporter.
