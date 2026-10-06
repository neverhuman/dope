#!/bin/bash
# Rebuild the journal PDF and the supplement from the tex already in this directory.
# Auxiliaries are removed after a clean run. The PDFs stay.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1760000000}"
build_one() {
  local stem="$1"
  # latexmk reruns pdflatex and bibtex until the citation and reference labels settle.
  # A document with no bibliography does not invoke bibtex.
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$stem"
  cp "$stem".log /tmp/"$stem"-3.log
  rm -f "$stem".aux "$stem".bbl "$stem".blg "$stem".log "$stem".out "$stem".toc "$stem".fls "$stem".fdb_latexmk
}
build_one dope-mfs
build_one supplement
# Author decision is unpicked. This second PDF changes only the byline and
# the running header. The source dope-mfs.tex stays on The DOPE Project.
python3 - << 'PY'
from pathlib import Path
src = Path("dope-mfs.tex").read_text()
pairs = (
    ("\\author{The DOPE Project}", "\\author{Anonymous}"),
    ("\\markboth{The DOPE Project}%", "\\markboth{Anonymous}%"),
)
for old, new in pairs:
    if src.count(old) != 1:
        raise SystemExit("author marker is not unique: " + old)
    src = src.replace(old, new, 1)
anchor = "\\input{generated/numbers.tex}\n"
if src.count(anchor) != 1:
    raise SystemExit("numbers input is not unique")
renew = "".join(
    f"\\renewcommand{{\\{name}}}{{\\{name}Anon}}\n"
    for name in (
        "DopeHardware",
        "GaussHardware",
        "ChowHardware",
        "IndHardware",
        "CtganHardware",
        "TvaeHardware",
        "ForestHardware",
        "ArfHardware",
    )
)
src = src.replace(anchor, anchor + renew, 1)
Path("dope-mfs-anonymous.tex").write_text(src)
PY
build_one dope-mfs-anonymous
rm -f dope-mfs-anonymous.tex
pdfinfo dope-mfs.pdf | awk '/Pages|Page size/'
pdfinfo dope-mfs-anonymous.pdf | awk '/Pages|Page size/'
pdfinfo supplement.pdf | awk '/Pages|Page size/'
