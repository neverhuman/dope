#!/usr/bin/env bash
# A missing guest tool must fail before spending the CPU proof budget.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
mkdir -p target/cpu-tool-readiness
fixture=$(mktemp -d "$root/target/cpu-tool-readiness/attempt-XXXXXXXX")
bash_bin=$(command -v bash)
dirname_bin=$(command -v dirname)
mkdir_bin=$(command -v mkdir)
tools=(cargo python3 uv just latexmk pdffonts pdfinfo pdftotext rg tar gzip)

for missing in "${tools[@]}"; do
  bin="$fixture/$missing/bin"
  mkdir -p "$bin"
  ln -s "$dirname_bin" "$bin/dirname"
  ln -s "$mkdir_bin" "$bin/mkdir"
  for tool in "${tools[@]}"; do
    if [[ "$tool" != "$missing" ]]; then
      printf '#!%s\nprintf unexpected-tool-execution > "$CPU_READINESS_SENTINEL"\nexit 88\n' "$bash_bin" > "$bin/$tool"
      chmod +x "$bin/$tool"
    fi
  done
  output="$fixture/$missing/output.log"
  sentinel="$fixture/$missing/executed"
  if PATH="$bin" CPU_READINESS_SENTINEL="$sentinel" "$bash_bin" ops/ci/cpu.sh > "$output" 2>&1; then
    echo "CPU accepted missing tool: $missing" >&2
    exit 1
  else
    status=$?
  fi
  if [[ "$status" != 1 || -e "$sentinel" ]] || ! rg -Fxq "Required CI tool is unavailable: $missing" "$output"; then
    cat "$output" >&2
    echo "CPU did not refuse missing tool before execution: $missing" >&2
    exit 1
  fi
done
printf 'CPU tool readiness: %s real-entry refusals passed\n' "${#tools[@]}"
