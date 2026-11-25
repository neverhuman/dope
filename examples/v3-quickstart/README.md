# Real-Dataset Quick Start (DOPE v3 Lifecycle)

Run the complete DOPE v3 lifecycle on a real tabular dataset (**UCI Airfoil Self-Noise**, CC BY 4.0, 1,503 aerodynamic test measurements across 106 physical airfoil configurations) in under 10 seconds — **100% offline, with zero Python required**:

1. **Fit** a `<= 2,048`-byte generative program (`model.dope`) on real training rows (`airfoil_train.csv`, 1,190 rows).
2. **Sample** 4,760 synthetic training rows (`synthetic.csv`) from `model.dope` alone in an isolated directory without touching the real data.
3. **Train & Test** downstream models (`linear`, `gbdt` — the v3 primary auditor pair; pass `--auditors linear,gbdt,mlp` to also include the slower CPU MLP auditor) on `synthetic.csv` vs. real `airfoil_train.csv`, scoring both on the untouched real holdout (`airfoil_test.csv`, 313 rows across disjoint physical configurations).

## Run in one command

Works with either the [signed `dope-engine-v3.0.0` binary](../../README.md#quick-start) or a local source build (`cargo build --manifest-path engine/Cargo.toml --release -p dope-cli`):

```bash
bash examples/v3-quickstart/run.sh
```

Or pass an explicit binary and output directory:

```bash
bash examples/v3-quickstart/run.sh \
  --binary "$HOME/.local/opt/dope/3.0.0/bin/dope" \
  --out target/v3-quickstart-airfoil
```

## Run step-by-step (CLI commands)

```bash
dope="$HOME/.local/opt/dope/3.0.0/bin/dope"   # or `dope` if on PATH / built from source
out="target/v3-quickstart-airfoil"
mkdir -p "$out"

# 1. Compile the real training split into a <= 2,048-byte program
"$dope" fit examples/v3-quickstart/data/airfoil_train.csv \
  --target sound_pressure_db --format auto-v8 --budget 2048 --seed 42 \
  -o "$out/model.dope"
wc -c < "$out/model.dope"

# 2. Sample synthetic training rows anywhere from model.dope alone
"$dope" sample "$out/model.dope" -n 4760 --seed 42 -o "$out/synthetic.csv"

# 3. Train and score downstream models on real held-out rows
#    (Direct `dope eval -m model.dope` works on source builds from main; on the published v3.0.0
#     binary, `run.sh` automatically uses a separate `--format 1` schema carrier.)
"$dope" eval -m "$out/model.dope" \
  --train examples/v3-quickstart/data/airfoil_train.csv \
  --test examples/v3-quickstart/data/airfoil_test.csv \
  --synth "$out/synthetic.csv" \
  --auditors linear,gbdt --seed 42 --json "$out/evaluation.json"
```

## Train your own ML model on `synthetic.csv` and test on real data

Because `synthetic.csv` uses the exact original columns and units, you can drop it into any external ML library (e.g., `scikit-learn`, LightGBM, XGBoost, PyTorch) and score on real held-out rows:

```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

target = "sound_pressure_db"
train_real = pd.read_csv("examples/v3-quickstart/data/airfoil_train.csv")
train_syn  = pd.read_csv("target/v3-quickstart-airfoil/synthetic.csv")
test_real  = pd.read_csv("examples/v3-quickstart/data/airfoil_test.csv")

X_te, y_te = test_real.drop(columns=[target]), test_real[target]
null_mse = mean_squared_error(y_te, [train_real[target].mean()] * len(y_te))

real_model = HistGradientBoostingRegressor(random_state=42).fit(train_real.drop(columns=[target]), train_real[target])
syn_model  = HistGradientBoostingRegressor(random_state=42).fit(train_syn.drop(columns=[target]), train_syn[target])

real_mse = mean_squared_error(y_te, real_model.predict(X_te))
syn_mse  = mean_squared_error(y_te, syn_model.predict(X_te))
retention = (null_mse - syn_mse) / (null_mse - real_mse)
print(f"Null MSE: {null_mse:.2f} | Real-trained MSE: {real_mse:.2f} | Synth-trained MSE: {syn_mse:.2f} | Retention: {retention:.3f}")
```

## Scope & attribution

- Dataset: UCI Airfoil Self-Noise (Brooks, Pope & Marcolini, 1989; DOI `10.24432/C5VW2C`; **CC BY 4.0**). See [`data/LICENSE_AND_PROVENANCE.md`](data/LICENSE_AND_PROVENANCE.md) for SHA-256 verification and the configuration-grouped split rule.
- This is a single-table public walkthrough demonstrating the engineering lifecycle, not the multi-table benchmark scoreboard or a privacy guarantee.
