#!/usr/bin/env bash
# Pre-wired DOPE v3.0 Phase II Campaign Launcher (E0-E9).
# Fail-closed guard: refuses fleet compute execution unless --confirm-fleet-unlocked is explicitly passed.
set -euo pipefail

STAGE="${1:-}"
CONFIRM="${2:-}"
OUT_ROOT="${DOPE_V3_OUT_ROOT:-target/v3-campaign}"

if [[ -z "${STAGE}" ]]; then
  echo "Usage: $0 <e0|e1|e2|e3|e4|e5|e6|e7|e8> --confirm-fleet-unlocked" >&2
  exit 2
fi

if [[ "${CONFIRM}" != "--confirm-fleet-unlocked" ]]; then
  echo "REFUSED: 2026-11-23T07:23Z owner JSR compute pause active. Pass --confirm-fleet-unlocked ONLY after owner lifts the pause." >&2
  exit 1
fi

EMIT_BIN="${DOPE_V8_EMIT_BIN:-engine/target/release/dope_v8_emit}"
if [[ ! -x "${EMIT_BIN}" ]]; then
  echo "Building release dope_v8_emit..."
  cargo build --manifest-path engine/Cargo.toml --locked --release -p dope-arena --bin dope_v8_emit
  EMIT_BIN="engine/target/release/dope_v8_emit"
fi

mkdir -p "${OUT_ROOT}/${STAGE}"

case "${STAGE}" in
  e0|e2|e6)
    echo "[${STAGE}] Native DSE8 P0 analytic / budget-ladder scoring ready via ${EMIT_BIN} (pass to dope_kpi emit-cli)."
    "${EMIT_BIN}" --help
    ;;
  e1)
    echo "[e1] Launching FP-0 repaired V7 baseline..."
    PYTHONPATH=. python3 -m research.levels.s3_v7_fast_trainer \
      --out-dir "${OUT_ROOT}/e1_fp0_v7" \
      --n-fit-max 1024 --n-qry-cap 256 --steps 5000
    ;;
  e3)
    echo "[e3] Launching 1k-step SingleEncV8 screening (owner encoder + receiver decoder)..."
    PYTHONPATH=. python3 -m research.levels.s3_v8_trainer \
      --out-dir "${OUT_ROOT}/e3_v8_screen" \
      --device cuda --steps 1000 --episodes-per-step 8 \
      --n-fit 384 --n-qry 256 --n-syn 384 --d-model 128 --n-heads 4 \
      --n-enc-layers 4 --n-dec-layers 2 --save-every 250 --seed 42
    ;;
  e4)
    echo "[e4] Launching 1k-step SingleEncV8 PCGrad-Lite + ASAM ablation..."
    PYTHONPATH=. python3 -m research.levels.s3_v8_trainer \
      --out-dir "${OUT_ROOT}/e4_v8_pcgrad_asam" \
      --device cuda --steps 1000 --episodes-per-step 8 \
      --n-fit 384 --n-qry 256 --n-syn 384 --d-model 128 --n-heads 4 \
      --use-pcgrad --use-asam --asam-every 10 --save-every 250 --seed 42
    ;;
  e5)
    echo "[e5] Launching full 3-phase SingleEncV8 training..."
    PYTHONPATH=. python3 -m research.levels.s3_v8_trainer \
      --out-dir "${OUT_ROOT}/e5_v8_full" \
      --device cuda --steps 20000 --episodes-per-step 8 \
      --n-fit 512 --n-qry 256 --n-syn 512 --d-model 128 --n-heads 4 \
      --n-enc-layers 4 --n-dec-layers 2 --save-every 500 --resume --seed 42
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
