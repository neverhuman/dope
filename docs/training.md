# Training the DOPE `SingleEncV8` Neural Model

DOPE v3.0.0 separates **per-table analytic fitting** from **offline neural pretraining**:

- **Using DOPE on a table (`dope fit --format auto-v8`)** needs **no training, no GPU, and no Python**. The signed pure-Rust CLI fits Format 1 and analytic `DSE8` on your table in seconds.
- **Training `SingleEncV8` (`research/levels/s3_v8_trainer.py`)** is an optional offline PyTorch meta-training loop that learns a shared owner-side encoder (reading the FIT-only Evidence Bank) and receiver-side decoder (emitting zero-initialized residual refinements over analytic `DSE8`). Trained weights export as canonical SafeTensors (`<= 1 GiB`, `active-rank-v1`) and plug directly into the pure-Rust `dope` CLI via `--v8-weights`.

> **Honest baseline status (v3.0.0):** In the v3.0.0 white paper pilot (Section 6), trained `SingleEncV8` checkpoints reduced training loss and screening error, but landed within noise of the weight-free analytic release on fresh development splits (`-0.37` Transfer Score points, 95% CI `[-2.71, +1.72]`). The signed v3.0.0 release therefore ships weight-free. Pushing learned `SingleEncV8` decisively past analytic `DSE8` — primarily by enriching the FIT-only Evidence Bank with better data-specific features — is the central open frontier in [ROADMAP.md](../ROADMAP.md).

---

## 1. Prerequisites & Environment Setup

You need the pinned Rust toolchain (`rust-toolchain.toml`, Rust `1.88.0`) and Python 3.10–3.12 with PyTorch.

```bash
# 1. Build the pure-Rust DOPE CLI, V8 packet emitter, and C FFI library
cargo build --manifest-path engine/Cargo.toml --locked --release \
  -p dope-cli --bin dope \
  -p dope-arena --bin dope_v8_emit \
  -p dope-ffi

# 2. Create an isolated Python training environment
python3 -m venv target/v8-train-env
target/v8-train-env/bin/python -m pip install --upgrade pip
target/v8-train-env/bin/python -m pip install \
  torch safetensors numpy scipy scikit-learn pytest
```

*For CUDA GPUs, install the matching CUDA wheel of PyTorch (e.g., `pip install --index-url https://download.pytorch.org/whl/cu128 torch`).*

---

## 2. Quick Start: 5-Minute CPU Training on Synthetic Priors

You do **not** need private datasets to train, debug, or test new Evidence Bank features. Without an external arena, `s3_v8_trainer.py` generates fresh FIT/QRY tabular episodes on the fly from synthetic structural priors (Gaussian/Student-t/Clayton copulas, mixtures, XOR, oblivious trees, Friedman nonlinearities, heteroscedasticity, lattices, zero inflation, missingness, and nominal one-hot groups) and fits exact Rust `DSE8` Step-0 packets (`--use-rust-packet`):

```bash
source target/v8-train-env/bin/activate
export PATH="$PWD/engine/target/release:$PATH"

python3 -m research.levels.s3_v8_trainer \
  --out-dir target/v8-quickstart \
  --run-name v8_cpu_smoke \
  --device cpu \
  --steps 50 \
  --episodes-per-step 2 \
  --n-fit 256 --n-qry 192 --n-syn 256 \
  --d-model 64 --n-heads 4 --n-enc-layers 2 --n-dec-layers 2 \
  --use-rust-packet \
  --budget-bytes 2048 \
  --save-every 25 \
  --seed 42
```

Outputs written to `target/v8-quickstart/`:
- `train.jsonl` — per-step loss terms (`l_ret`, direct conditional NLL, marginal/copula/tree/driver/DCR/anchor terms)
- `latest.pt` — full resumable training state (model, optimizer, scheduler, 3 EMAs, and Torch/NumPy RNG states; resume with `--resume`)
- `model_v8.safetensors` — canonical `F32` SafeTensors export ready for the pure-Rust `dope` CLI (periodic EMA snapshots are also saved alongside `latest.pt`)

---

## 3. Test Your Trained Weights Inside the Pure-Rust `dope` CLI

Every exported `.safetensors` checkpoint plugs straight into the native Rust engine with zero Python at fit or sample time:

```bash
dope="$PWD/engine/target/release/dope"
weights="$PWD/target/v8-quickstart/model_v8_ema.safetensors"

# Fit directly with your neural V8 weights (--format v8) — or use --format auto-v8 to run the 3-way selector
"$dope" fit examples/v3-quickstart/data/airfoil_train.csv \
  --target sound_pressure_db \
  --format v8 --budget 2048 --seed 42 \
  --v8-weights "$weights" \
  -o target/v8-quickstart/model.dope

wc -c < target/v8-quickstart/model.dope

# Sample synthetic rows from the V8 artifact + weights
"$dope" sample target/v8-quickstart/model.dope \
  -n 4760 --seed 42 \
  --v8-weights "$weights" \
  -o target/v8-quickstart/synthetic.csv

# Evaluate on held-out real data (linear, GBDT; add ,mlp for optional neural auditor)
"$dope" eval -m target/v8-quickstart/model.dope \
  --train examples/v3-quickstart/data/airfoil_train.csv \
  --test examples/v3-quickstart/data/airfoil_test.csv \
  --synth target/v8-quickstart/synthetic.csv \
  --v8-weights "$weights" \
  --auditors linear,gbdt --seed 42 \
  --json target/v8-quickstart/evaluation.json
```

When `--format auto-v8 --v8-weights` is passed, `dope fit` fits Format 1, analytic `DSE8`, and neural `DSE8` independently on duplicate-grouped 2-fold internal checks and selects the neural packet only if it clears the acceptance margin without triggering the single-fold veto.

---

## 4. Full-Scale GPU Training on Real + Synthetic Table Arenas

To train on a mixture of real tables (stored in a flat DOPE arena directory) and synthetic priors, use `research.levels.s3_v8_feed`. It launches parallel CPU episode producers backed by Rust `dope-ffi` so every episode gets a fresh duplicate-grouped FIT/QRY split and zero per-table feature caching:

```bash
source target/v8-train-env/bin/activate
export PATH="$PWD/engine/target/release:$PATH"

python3 -m research.levels.s3_v8_feed \
  --arena /path/to/your/flat-arena \
  --producers 4 \
  -- \
  --out-dir target/v8-gpu-run \
  --run-name v8_d160_gpu \
  --device cuda:0 --amp \
  --steps 8000 \
  --episodes-per-step 4 \
  --grad-accum-steps 2 \
  --d-model 160 --n-heads 4 --n-enc-layers 4 --n-dec-layers 2 \
  --lr 3e-4 --lr-schedule wsd --warmup-steps 500 \
  --prior-mix-prob 0.25 \
  --contrastive-prob 0.15 \
  --use-rust-packet \
  --budget-bytes 2048 \
  --save-every 500 \
  --seed 42
```

### Key training options (`s3_v8_trainer.py`)

| Flag | Purpose |
| --- | --- |
| `--resume` | Exact mid-run resume (restores optimizer, scheduler, 3 EMAs, RNG states, and verifies run flags match) |
| `--init-from FILE.safetensors` | Warm-start from an earlier SafeTensors checkpoint |
| `--no-delta-w` | Zero and freeze the linear `delta_W` head while training nonlinear/feature heads |
| `--freeze-target-heads` / `--freeze-x-heads` | Isolate covariate vs. conditional-target refinement |
| `--export-dtype f32\|bf16` | Export weights in F32 (default) or compact BF16 SafeTensors |
| `--code-dim 64\|128\|256` | FSQ code width (wire Tiers 1/2/3 = 32/64/128 bytes in packet) |
| `--use-pcgrad` / `--use-asam` | Optional multi-objective gradient projection / sharpness-aware minimization |
| `--sync-dir DIR` | Multi-worker outer-momentum DiLoCo synchronization (pairs with `engine/dope-arena/src/bin/dope_diloco_hub/`) |
| `--deterministic --no-tf32` | Exact deterministic GPU replay for debugging and resume-parity tests |
| `--max-wall-s SECONDS` | Graceful wall-clock stop saving `latest.pt` before exit |

---

## 5. Quick Regression & Parity Loop (For Contributors)

Before opening a pull request that adds Evidence Bank features or modifies `SingleEncV8` / `DSE8`, run the fast local regression loop (~2–4 minutes on CPU):

```bash
# 1. Pure-Rust Evidence Bank, DSE8 codec, and auto-v8 unit tests
cargo test --manifest-path engine/Cargo.toml --locked -p dope-core evidence single_encoding::v8 auto::v8

# 2. Rebuild native emitter + FFI library and run the 12-fixture Rust-vs-Python parity suite
cargo build --manifest-path engine/Cargo.toml --locked --release -p dope-arena --bin dope_v8_emit -p dope-ffi
target/v8-train-env/bin/pytest \
  research/levels/tests/test_v8_rust_parity.py \
  research/levels/tests/test_v8_losses.py \
  research/levels/tests/test_v8_code_dim.py \
  research/levels/tests/test_v8_surgery.py -v

# 3. Evaluate candidate weights vs Step-0 analytic baseline across linear, GBDT, and held-out MLP
#    using research/levels/s3_v8_eval.py (`evaluate_dse8_cell`)
```

`test_v8_rust_parity.py` checks 12 diverse tabular stress cases (linear/nonlinear regression, binary interactions, 5-class multiclass, 25% MAR missingness, wide one-hot nominal groups, zero inflation, heavy-tailed Student-t, decimal lattice heaping, pure XOR, tiny `n=24`, constant/sparse columns, and wide `p=48` mixed tables) and enforces `max |py - rs| <= 1e-4`, zero illegal one-hot rows, and strict `<= 2,048 B` CRC-32 packets.

See [ROADMAP.md](../ROADMAP.md) for high-leverage Evidence Bank feature ideas and our contributor co-authorship program.
