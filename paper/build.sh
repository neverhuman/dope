#!/usr/bin/env bash
# Build the manuscripts into target/paper/: the TDL manuscript and supplement (main, supplement)
# and the DOPE 1.0 engine manuscript (dope1 preprint, dope1-venue double-blind, dope1-supplement).
# PDFs are build output, not committed. Generated inputs are read from paper/generated/
# when present; otherwise visible placeholders are typeset (see macros.tex).
set -euo pipefail
cd "$(dirname "$0")"
out="../target/paper"
mkdir -p "$out"
export BIBINPUTS="$PWD:${BIBINPUTS:-}"
# The contract macros and every public surface must agree with contracts/tdl-contract-v1.json.
python3 -B scripts/contract_macros.py --check
# The DOPE 1.0 manuscript's generated tables and macros must be exactly what the benchmark records imply.
python3 -B ../bench/real-world/render.py --check
python3 -B -m unittest discover -s scripts -p 'test_*.py' >"$out/contract-macros.test.log" 2>&1 || {
  tail -40 "$out/contract-macros.test.log" >&2
  exit 1
}
# Each preprint builds before its supplement: the DOPE 1.0 supplement reads dope1's labels (xr-hyper).
for doc in main supplement dope1 dope1-supplement dope1-venue; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$out" "$doc.tex" >"$out/$doc.build.log" 2>&1 || {
    tail -40 "$out/$doc.build.log" >&2
    exit 1
  }
  if grep -E "LaTeX Warning: (Reference|Citation) .* undefined|There were undefined references|Citation .* undefined" "$out/$doc.log" >&2; then
    echo "unresolved references in $doc" >&2
    exit 1
  fi
done
# Figure 1 of each preprint must be a generated hero figure on page 1, not a pending placeholder
# (the double-blind venue layout puts it at the top of page 2, after the abstract).
python3 -B scripts/figure_one.py "$out" main dope1
echo "built $out/main.pdf, $out/supplement.pdf, $out/dope1.pdf, $out/dope1-supplement.pdf and $out/dope1-venue.pdf"
