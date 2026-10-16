#!/usr/bin/env bash
# Build the TDL manuscript and supplement into target/paper/ (T-110).
# PDFs are build output, not committed. Generated inputs are read from paper/generated/
# when present; otherwise visible placeholders are typeset (see macros.tex).
set -euo pipefail
cd "$(dirname "$0")"
out="../target/paper"
mkdir -p "$out"
export BIBINPUTS="$PWD:${BIBINPUTS:-}"
# The contract macros and every public surface must agree with contracts/tdl-contract-v1.json.
python3 -B scripts/contract_macros.py --check
python3 -B -m unittest discover -s scripts -p 'test_*.py' >"$out/contract-macros.test.log" 2>&1 || {
  tail -40 "$out/contract-macros.test.log" >&2
  exit 1
}
for doc in main supplement; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$out" "$doc.tex" >"$out/$doc.build.log" 2>&1 || {
    tail -40 "$out/$doc.build.log" >&2
    exit 1
  }
  if grep -E "LaTeX Warning: (Reference|Citation) .* undefined|There were undefined references|Citation .* undefined" "$out/$doc.log" >&2; then
    echo "unresolved references in $doc" >&2
    exit 1
  fi
done
echo "built $out/main.pdf and $out/supplement.pdf"
