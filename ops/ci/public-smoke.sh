#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m pip install --constraint validation/requirements-v1.txt numpy scipy pandas 'matplotlib==3.10.8' 'jsonschema==4.26.0'
cargo build --locked --release -p dope-kernel --bin dope
mkdir -p target/public-smoke
target/release/dope compile --dataset-dir examples/readme-regression --task regression --out target/public-smoke/toy.dpk
target/release/dope inspect --kernel target/public-smoke/toy.dpk
target/release/dope sample --kernel target/public-smoke/toy.dpk --rows 200 --out target/public-smoke/synth.csv
cargo build --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin audit-proof
target/debug/audit-proof focus metrics::golden
python3 -m unittest discover -s tools/claims -p 'test_*.py'
python3 tools/claims/check.py
python3 tools/claims/render.py --check
