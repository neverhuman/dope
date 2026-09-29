#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUT="${OUT:-runs/kernel_tuner_long_random_synth}"
CORPUS="${CORPUS:-/home/ubuntu/remote_super/quant}"
DEVICE="${DEVICE:-cuda}"

if ! grep -q -- "--synthetic-mode" src/dope_kernel/cli.py 2>/dev/null || ! grep -q "train eval regret" src/dope_kernel/tuner.py 2>/dev/null; then
  echo "ERROR: this checkout lacks the expected tuner CLI/objective: $ROOT" >&2
  exit 1
fi

if [ -e "$OUT" ]; then
  echo "ERROR: output already exists: $OUT. Choose a new OUT path." >&2
  exit 1
fi

echo "Starting tuner TUI..."
echo "Source: $ROOT"
echo "Output: $ROOT/$OUT"
exec env PYTHONPATH=src python3 -m dope_kernel train-tuner \
  --preset long \
  --corpus "$CORPUS" \
  --device "$DEVICE" \
  --out "$OUT" \
  --synthetic-mode synthetic-heavy \
  --full-kpi-every 0 \
  --tui
