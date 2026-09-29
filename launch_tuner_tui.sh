#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SOURCE_OF_TRUTH="${SOURCE_OF_TRUTH:-/home/ubuntu/dope}"
OUT="${OUT:-runs/kernel_tuner_long_random_synth}"
CORPUS="${CORPUS:-/home/ubuntu/remote_super/quant}"
DEVICE="${DEVICE:-cuda}"

if ! grep -q -- "--synthetic-mode" src/dope_kernel/cli.py 2>/dev/null; then
  if [ "$ROOT" != "$SOURCE_OF_TRUTH" ] && [ -f "$SOURCE_OF_TRUTH/src/dope_kernel/cli.py" ] && grep -q -- "--synthetic-mode" "$SOURCE_OF_TRUTH/src/dope_kernel/cli.py"; then
    echo "Updating stale source in $ROOT from $SOURCE_OF_TRUTH"
    rm -rf src tests pyproject.toml README.md
    cp -a "$SOURCE_OF_TRUTH/src" "$ROOT/src"
    cp -a "$SOURCE_OF_TRUTH/tests" "$ROOT/tests"
    cp -a "$SOURCE_OF_TRUTH/pyproject.toml" "$ROOT/pyproject.toml"
    [ -f "$SOURCE_OF_TRUTH/README.md" ] && cp -a "$SOURCE_OF_TRUTH/README.md" "$ROOT/README.md"
  fi
fi

if ! grep -q -- "--synthetic-mode" src/dope_kernel/cli.py 2>/dev/null || ! grep -q "train eval regret" src/dope_kernel/tuner.py 2>/dev/null; then
  echo "ERROR: stale source tree: $ROOT" >&2
  echo "This tree does not contain the fixed tuner CLI/objective. Do not run from it." >&2
  exit 1
fi

echo "Killing old dope tuner runs..."
tmux kill-session -t dope_kernel_long 2>/dev/null || true
tmux kill-session -t dope_kernel_long_random 2>/dev/null || true
pkill -f "python[0-9.]* -m dope_kernel train-tuner" 2>/dev/null || true
sleep 1

if pgrep -af "python[0-9.]* -m dope_kernel train-tuner" >/dev/null 2>&1; then
  echo "ERROR: a dope tuner process is still running:" >&2
  pgrep -af "python[0-9.]* -m dope_kernel train-tuner" >&2
  exit 1
fi

echo "Clearing $OUT"
rm -rf "$OUT"

echo "Starting fixed tuner TUI..."
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
