#!/usr/bin/env bash
# Offline research evidence only, deliberately outside the Rust application gate.
set -euo pipefail
cd "$(dirname "$0")/../.."
python_bin=${1:-python3}
"$python_bin" -B -m unittest discover -s bench/real-world -p 'test_*.py' -v
"$python_bin" -B bench/real-world/elo.py --check
"$python_bin" -B bench/real-world/elo.py --methods dope,rows,kernel,copula,arf --out bench/real-world/results/elo-cpu --check
