# Quick start: DOPE on a real, open table

This folder takes a raw CSV with numeric and categorical columns all the way to a
compiled DOPE program, synthetic rows in the original schema, and a retention score
on rows the compiler never saw. It needs the native binary and a few Python packages.

```bash
cargo build --release --bin dope-kernel
pip install numpy pandas scikit-learn catboost      # catboost optional; falls back to sklearn GBDT
python3 examples/quickstart/quickstart.py --dataset adult
```

The UCI Adult table (CC BY 4.0, DOI 10.24432/C5XW20) is downloaded once into
`target/quickstart/cache/` and its SHA-256 is checked. Outputs land in
`target/quickstart/adult/`: `train.csv` (the encoded fit rows), `encoder.bin`,
`model.dpk`, `inspect.json`, `samples.csv`, `synthetic.csv` (decoded rows) and
`report.json`. The committed reference run is
[`results/adult-reference.json`](results/adult-reference.json); the README's quick-start
table is rendered from it.

Your own table:

```bash
python3 examples/quickstart/quickstart.py --csv my_table.csv --target churned --drop customer_id
python3 examples/quickstart/quickstart.py --csv train.csv --holdout test.csv --target price --multiplier 4
```

## What each step does

| Step | Code | Notes |
| --- | --- | --- |
| encode | `dope_tables.TableEncoder` | fit rows only; reversible; writes the compiler's headerless `train.csv` |
| select | `dope_select.select_backend` | 3-fold transfer inside the fit rows over `chow_liu`, `compact_neural_residual_symbolic`, `sparse_gaussian_copula` and the built-in tournament (`--backend auto`, the default) |
| compile | `dope-kernel compile --dataset-dir` | reads only the encoded fit rows |
| inspect | `dope-kernel inspect` | prints the program as an S-expression |
| sample | `dope-kernel sample` | reads only the artifact; `TableEncoder.inverse_transform` decodes |
| evaluate | `dope_eval.retention_report` | CatBoost (or sklearn GBDT), logistic/ridge regression and an MLP, trained on real and on synthetic rows, scored on the holdout |

## Encoding choices

`TableEncoder(target, numeric="robust", categorical="auto")`:

- **Categorical columns.** `"auto"` uses one-hot columns with the most frequent level
  dropped (like `pandas.get_dummies(drop_first=True)`) when a column has at most four
  levels, and one target-ordered code otherwise: levels are sorted by their smoothed mean
  target and placed in [0, 1] in proportion to it, so the code carries the main effect
  linearly, the compiler keeps the codes as an exact categorical grid, and decoding is
  lossless. Rare levels are pooled into `__other__`; missing values get their own level.
- **Numeric columns.** `"robust"` takes `log1p` of skewed non-negative columns, clips at
  Tukey's far-out fences and rescales to [0, 1]. `"minmax"` and a 256-knot `"quantile"`
  map are available. Integer columns decode to integers and dates decode to dates.
- **Target.** A regression target is min-max scaled (an affine map, so retention is
  unchanged); a two-class target becomes 0/1. Multi-class targets are refused.
- **State.** `to_bytes()` writes everything the decoder needs as a compact binary blob
  (`DOPEENC1`), with names and labels zlib-compressed; `labels=False` drops them.

The defaults come from [`ablation.py`](ablation.py), five seeds on UCI Adult and UCI
Abalone (CC BY 4.0, DOI 10.24432/C55C7W); results are in
[`results/ablation.md`](results/ablation.md).

## Tests

```bash
python3 -B -m unittest discover -s examples/quickstart -p 'test_*.py' -v
```

Evidence status: a local engineering demonstration on public tables, not the paper's
benchmark. Empirical copy or membership checks are not run here, and nothing in this
folder makes a privacy claim.

## Python encoder around the native binary

The following is a local development workflow. Encoding fits only the supplied fitting rows. Retained labels and schema types decode; clipping, rare-level grouping and finite precision can lose detail. Use fresh private output locations and evaluate on separate rows.

```python
import pandas as pd
from dope_tables import TableEncoder, read_dope_samples

table = pd.read_csv("my.csv")
encoder = TableEncoder(target="y").fit(table)
encoder.write_dataset_dir(table, "work")   # train.csv and encoder.bin
# compile and sample (below), then decode:
synthetic = encoder.inverse_transform(read_dope_samples("work/samples.csv"))
```

```bash
target/release/dope-kernel compile --dataset-dir work --task regression --out work/model.dpk --candidate chow_liu
target/release/dope-kernel sample --kernel work/model.dpk --rows 10000 --out work/samples.csv
target/release/dope-kernel inspect --kernel work/model.dpk
```

