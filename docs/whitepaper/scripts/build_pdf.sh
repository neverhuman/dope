#!/bin/bash
# Direct PDF builds retain full admission through the canonical in-process driver.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec nice -n 10 ionice -c 3 python3 -B docs/whitepaper/scripts/paper_build.py --pdf-only
