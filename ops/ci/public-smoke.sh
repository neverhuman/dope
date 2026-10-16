#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m pip install --constraint validation/requirements-v1.txt numpy scipy pandas 'matplotlib==3.10.8' 'jsonschema==4.26.0'
cargo build --locked --release -p dope-kernel --bin dope
mkdir -p target/public-smoke
target/release/dope compile --dataset-dir examples/readme-regression --task regression --out target/public-smoke/toy.dpk
target/release/dope inspect --kernel target/public-smoke/toy.dpk
target/release/dope sample --kernel target/public-smoke/toy.dpk --rows 200 --out target/public-smoke/synth.csv
cargo test --locked --lib metrics::golden -- --list > target/public-smoke/golden-tests.txt
python3 -c "from pathlib import Path; assert any(line.startswith('metrics::golden::') and line.endswith(': test') for line in Path('target/public-smoke/golden-tests.txt').read_text().splitlines()), 'Rust golden consumer is missing'"
cargo test --locked --lib metrics::golden
python3 -m research.transfer.tdl.golden --check
python3 -m unittest discover -s tools/claims -p 'test_*.py'
python3 tools/claims/check.py
python3 tools/claims/render.py --check
