# External learner evidence

This is an offline, Python 3.12 validation lane. It is separate from the Rust
certifier and release decision. It measures claims about the named library and
model versions; a missing or failed model produces no claim for that model.

`pyproject.toml` pins the direct dependencies and `uv.lock` pins all resolved
versions and wheel hashes. From this directory, set
`UV_PROJECT_ENVIRONMENT=../../target/external-venv` to keep the environment in
the lane's reusable `target/`. On a connected provisioning host, fill the uv
cache once with `uv sync --locked --extra xgboost --extra lightgbm`. Transfer
the cache to the offline evaluator, then run `uv sync --offline --locked --extra
xgboost --extra lightgbm`. The optional extras may be omitted; their model
results will be `missing_library`. The runner itself has no network calls.

Inputs are numeric, headerless CSVs with the target last. Features may have
explicit missing values; all other values must already be in `[0,1]`. The
`dope-external-split` v1 manifest must contain the SHA-256 digests of the real
training and sealed holdout CSVs:

```json
{"format":"dope-external-split","version":1,"real_train_sha256":"...","real_holdout_sha256":"..."}
```

The corpus splitter should create and seal this manifest before candidate
training. The runner verifies both digests and uses the same holdout for TRTR,
TSTR, and permutation importance. It never tunes on that holdout. Synthetic
training rows may differ in count, but feature count and task must match.

```bash
python3.12 run.py --real-train real-train.csv --real-holdout sealed-holdout.csv \
  --synthetic-train synthetic.csv --artifact kernel.dpk \
  --split-manifest split.json --task binary --seed 57721 --out evidence.json
```

Each model uses one thread and a fixed seed, 64 trees where applicable, and
three permutation repeats on the same deterministic subset of at most 256
holdout rows. Utility metrics use the full sealed holdout. Inputs are capped at
32,768 rows and 2,000 features. Binary tasks use log loss and AUC; regression uses
RMSE. The report stores null/TRTR/TSTR losses, retention when informative,
regret, FI Spearman, deterministic top-k Jaccard, normalized importance shares,
and mean ratio error. Its hashes bind each result to the artifact bytes,
training and holdout split, synthetic CSV, model configuration, library version,
and seed. Feature indices remain anonymous. Ratio error has no release
threshold or preservation claim. A library claim requires every selected model
from that library to finish successfully. A feature-importance claim also
requires at least three informative features and a defined rank correlation.
No external result changes Rust certification,
constitutes formal DP, or establishes HIPAA de-identification.

Run the fixture smoke checks with `PYTHONPATH=validation/external
target/external-venv/bin/python -m unittest discover -s
validation/external/tests -v` from the repository root.
