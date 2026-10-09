#!/bin/bash
# Rebuild the journal PDF and the supplement from the tex already in this directory.
# Auxiliaries and logs stay in the checkout target directory; PDFs are copied here.
set -euo pipefail
cd "$(dirname "$0")/.."
# A direct PDF build has the same pinned, validation-only admission as just paper.
nice -n 10 ionice -c 3 python3 ../../ops/ci/paper-review-outputs.py --check-inputs
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1760000000}"
build_root="$(realpath ../..)/target/paper-build"
mkdir -p "$build_root"
exec 9>"$(realpath ../..)/target/paper-latex.lock"
flock 9
# The supplement embeds this cropped figure. Rebuild from the tracked TikZ
# source and generated architecture values under the same serial TeX lock.
nice -n 10 ionice -c 3 pdflatex -output-directory="$build_root" -interaction=nonstopmode -halt-on-error -file-line-error ../../ops/ci/paper-architecture.tex
cp "$build_root/paper-architecture.pdf" figures/architecture.pdf
python3 - << 'PY'
import hashlib
import json
from pathlib import Path
sources = ("figures/architecture.tex", "generated/numbers.tex", "../../ops/ci/paper-architecture.tex")
payload = {
    "format": "dope-paper-architecture-hashes-v1",
    "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
    "pdf_sha256": {"architecture.pdf": hashlib.sha256(Path("figures/architecture.pdf").read_bytes()).hexdigest()},
}
Path("generated/architecture-figure-hashes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
build_one() {
  local stem="$1"
  # latexmk reruns pdflatex and bibtex until the citation and reference labels settle.
  # A document with no bibliography does not invoke bibtex.
  # Force a pass: latexmk can retain a PDF after only engine options change.
  nice -n 10 ionice -c 3 latexmk -g -pdf '-usepretex=\pdftrailerid{}\pdfsuppressptexinfo=-1' -outdir="$build_root" -interaction=nonstopmode -halt-on-error -file-line-error "$stem"
  cp "$build_root/$stem.log" "$build_root/$stem-3.log"
  cp "$build_root/$stem.pdf" "$stem.pdf"
}
build_one dope-mfs
build_one supplement
# The anonymous PDF changes the byline and drops the journal-only repository
# URL. The source dope-mfs.tex keeps the byline. Both copies clear the class
# running header, so neither one names a venue.
python3 - << 'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from public_hardware import strip_journal_availability
src = Path("dope-mfs.tex").read_text()
pairs = (
    (
        "\\author{Jepson Taylor$^1$, Alton Alexander$^1$\\\\[0.15cm]\n"
        "$^1$NEVERHUMAN Research}",
        "\\author{Anonymous\\\\[0.15cm]\n"
        "Affiliation withheld for review.}",
    ),
)
for old, new in pairs:
    if src.count(old) != 1:
        raise SystemExit("author marker is not unique: " + old)
    src = src.replace(old, new, 1)
src = strip_journal_availability(src)
if "neverhuman" in src:
    raise SystemExit("anonymous manuscript still names the repository")
Path("dope-mfs-anonymous.tex").write_text(src)
supplement = Path("supplement.tex").read_text()
for old, new in pairs:
    if supplement.count(old) != 1:
        raise SystemExit("supplement author marker is not unique")
    supplement = supplement.replace(old, new, 1)
if "neverhuman" in supplement.lower():
    raise SystemExit("anonymous supplement still names the repository")
Path("supplement-anonymous.tex").write_text(supplement)
PY
build_one dope-mfs-anonymous
build_one supplement-anonymous
rm -f dope-mfs-anonymous.tex supplement-anonymous.tex
pdfinfo dope-mfs.pdf | awk '/Pages|Page size/'
pdfinfo dope-mfs-anonymous.pdf | awk '/Pages|Page size/'
pdfinfo supplement.pdf | awk '/Pages|Page size/'
pdfinfo supplement-anonymous.pdf | awk '/Pages|Page size/'
