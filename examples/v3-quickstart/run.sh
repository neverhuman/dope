#!/usr/bin/env bash
# End-to-end DOPE v3 real-dataset lifecycle (UCI Airfoil Self-Noise, CC BY 4.0).
#
# Runs completely offline with zero Python required:
#   1. Fit a <= 2,048-byte generative program on real training rows (--format auto-v8)
#   2. Copy only model.dope to a clean directory and sample 4x synthetic rows (source-free)
#   3. Train downstream models (linear, gbdt) on synthetic vs real rows and score on held-out real test rows
#
# Usage:
#   bash examples/v3-quickstart/run.sh [--binary /path/to/dope] [--out /path/to/outdir] [--auditors linear,gbdt]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_DIR="$SCRIPT_DIR/data"
TRAIN_CSV="$DATA_DIR/airfoil_train.csv"
TEST_CSV="$DATA_DIR/airfoil_test.csv"
TARGET="sound_pressure_db"

DOPE_BIN=""
OUT_DIR=""
AUDITORS="linear,gbdt"
BUDGET="2048"
SYNTH_ROWS="4760"
SEED="42"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) DOPE_BIN="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --auditors) AUDITORS="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --rows) SYNTH_ROWS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DOPE_BIN" ]]; then
  if command -v dope >/dev/null 2>&1; then
    DOPE_BIN=$(command -v dope)
  elif [[ -x "$HOME/.local/opt/dope/3.0.0/bin/dope" ]]; then
    DOPE_BIN="$HOME/.local/opt/dope/3.0.0/bin/dope"
  elif [[ -x "$SCRIPT_DIR/../../engine/target/release/dope" ]]; then
    DOPE_BIN="$SCRIPT_DIR/../../engine/target/release/dope"
  elif [[ -x "$SCRIPT_DIR/../../engine/target/debug/dope" ]]; then
    DOPE_BIN="$SCRIPT_DIR/../../engine/target/debug/dope"
  else
    echo "error: could not find 'dope' binary. Install it via README Quick start, run 'cargo build --manifest-path engine/Cargo.toml --release -p dope-cli', or pass --binary PATH." >&2
    exit 1
  fi
fi

if [[ ! -x "$DOPE_BIN" ]]; then
  echo "error: dope binary is not executable: $DOPE_BIN" >&2
  exit 1
fi
DOPE_BIN="$(cd "$(dirname "$DOPE_BIN")" && pwd)/$(basename "$DOPE_BIN")"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/dope-v3-quickstart.XXXXXX")
else
  mkdir -p "$OUT_DIR"
  OUT_DIR="$(cd "$OUT_DIR" && pwd)"
fi

echo "==> Binary: $("$DOPE_BIN" --version) ($DOPE_BIN)"
echo "==> Dataset: UCI Airfoil Self-Noise (1,190 train / 313 held-out test rows, group-disjoint)"
echo "==> Output directory: $OUT_DIR"

TRAIN_BYTES=$(wc -c < "$TRAIN_CSV")

echo ""
echo "[1/3] Compiling real training table into <= ${BUDGET}-byte DOPE program (--format auto-v8)..."
"$DOPE_BIN" --threads 1 fit "$TRAIN_CSV" \
  --target "$TARGET" \
  --format auto-v8 \
  --budget "$BUDGET" \
  --seed "$SEED" \
  --report "$OUT_DIR/fit-report.json" \
  -o "$OUT_DIR/model.dope"

MODEL_BYTES=$(wc -c < "$OUT_DIR/model.dope")
if [[ "$MODEL_BYTES" -gt "$BUDGET" ]]; then
  echo "error: artifact exceeded budget ($MODEL_BYTES > $BUDGET bytes)" >&2
  exit 1
fi
echo "    Real train CSV: ${TRAIN_BYTES} bytes -> DOPE artifact: ${MODEL_BYTES} bytes (strictly <= ${BUDGET} B)"

echo ""
echo "[2/3] Sampling ${SYNTH_ROWS} synthetic training rows from model.dope alone (source-free)..."
mkdir -p "$OUT_DIR/recipient"
cp "$OUT_DIR/model.dope" "$OUT_DIR/recipient/model.dope"
(
  cd "$OUT_DIR/recipient"
  "$DOPE_BIN" --threads 1 sample model.dope -n "$SYNTH_ROWS" --seed "$SEED" -o synthetic.csv
)
cp "$OUT_DIR/recipient/synthetic.csv" "$OUT_DIR/synthetic.csv"
SYNTH_LINES=$(( $(wc -l < "$OUT_DIR/synthetic.csv") - 1 ))
echo "    Generated ${SYNTH_LINES} synthetic rows -> $OUT_DIR/synthetic.csv"

echo ""
echo "[3/3] Training downstream models (${AUDITORS}) on synthetic vs. real rows, testing on held-out real data..."
# Use native DPA1 eval when supported; fall back to a Format 1 schema carrier on v3.0.0 binaries.
if "$DOPE_BIN" --threads 1 eval \
  -m "$OUT_DIR/model.dope" \
  --train "$TRAIN_CSV" \
  --test "$TEST_CSV" \
  --synth "$OUT_DIR/synthetic.csv" \
  --auditors "$AUDITORS" \
  --seed "$SEED" \
  --json "$OUT_DIR/evaluation.json" 2>/dev/null; then
  :
else
  "$DOPE_BIN" --threads 1 fit "$TRAIN_CSV" \
    --target "$TARGET" \
    --format 1 \
    --budget "$BUDGET" \
    --seed "$SEED" \
    --quiet \
    -o "$OUT_DIR/audit-schema.dope"
  "$DOPE_BIN" --threads 1 eval \
    -m "$OUT_DIR/audit-schema.dope" \
    --train "$TRAIN_CSV" \
    --test "$TEST_CSV" \
    --synth "$OUT_DIR/synthetic.csv" \
    --auditors "$AUDITORS" \
    --seed "$SEED" \
    --json "$OUT_DIR/evaluation.json"
fi

echo ""
echo "Done! All artifacts saved under: $OUT_DIR"
echo "  - $OUT_DIR/model.dope        (${MODEL_BYTES} bytes)"
echo "  - $OUT_DIR/synthetic.csv     (${SYNTH_LINES} rows)"
echo "  - $OUT_DIR/fit-report.json"
echo "  - $OUT_DIR/evaluation.json"
