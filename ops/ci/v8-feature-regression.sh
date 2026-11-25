#!/usr/bin/env bash
# Evidence Bank, DSE8 codec, 12-fixture Rust-vs-Python parity, and CPU SingleEncV8 reload smoke lane.
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1. Pure-Rust Evidence Bank, DSE8, auto-v8, and DPA1 eval unit/CLI tests.
cargo test --manifest-path engine/Cargo.toml --locked -p dope-core evidence
cargo test --manifest-path engine/Cargo.toml --locked -p dope-core single_encoding::v8
cargo test --manifest-path engine/Cargo.toml --locked -p dope-core auto::v8
cargo test --manifest-path engine/Cargo.toml --locked -p dope-cli --test auto_v8_cli

# 2. Build release dope CLI, dope_v8_emit binary, and dope-ffi shared library.
# Build dope-ffi in a separate invocation so --bin target filters never skip its cdylib.
cargo build --manifest-path engine/Cargo.toml --locked --release \
  -p dope-cli --bin dope -p dope-arena --bin dope_v8_emit
cargo build --manifest-path engine/Cargo.toml --locked --release -p dope-ffi

# 3. Run 12-fixture Rust-vs-Python V8 parity, loss, code-dim, surgery, and feed suites.
python3 -m pytest -v \
  research/levels/tests/test_v8_rust_parity.py \
  research/levels/tests/test_v8_losses.py \
  research/levels/tests/test_v8_code_dim.py \
  research/levels/tests/test_v8_surgery.py \
  research/levels/tests/test_s3_v8_feed.py

# 4. Run 4-step CPU synthetic-prior smoke train and pure-Rust SafeTensors reload.
smoke_dir="target/v8-ci-smoke"
rm -rf "$smoke_dir"
mkdir -p "$smoke_dir"
trap 'rm -rf "$smoke_dir"' EXIT

python3 -m research.levels.s3_v8_trainer \
  --out-dir "$smoke_dir" --device cpu --steps 4 --episodes-per-step 1 \
  --d-model 32 --n-heads 4 --n-enc-layers 1 --n-dec-layers 1 \
  --use-rust-packet --budget-bytes 2048 --save-every 4 --seed 42

engine/target/release/dope --threads 1 fit examples/v3-quickstart/data/airfoil_train.csv \
  --target sound_pressure_db --format v8 --budget 2048 \
  --v8-weights "$smoke_dir/model_v8_ema.safetensors" \
  --seed 42 -o "$smoke_dir/model.dope"
test "$(wc -c < "$smoke_dir/model.dope")" -le 2048

engine/target/release/dope --threads 1 sample "$smoke_dir/model.dope" \
  -n 256 --seed 42 --v8-weights "$smoke_dir/model_v8_ema.safetensors" \
  -o "$smoke_dir/synth.csv"

engine/target/release/dope --threads 1 eval -m "$smoke_dir/model.dope" \
  --train examples/v3-quickstart/data/airfoil_train.csv \
  --test examples/v3-quickstart/data/airfoil_test.csv \
  --synth "$smoke_dir/synth.csv" \
  --v8-weights "$smoke_dir/model_v8_ema.safetensors" \
  --auditors linear --seed 42 --json "$smoke_dir/eval.json"
