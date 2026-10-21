#!/usr/bin/env bash
# Exercise the real entry's refusals/routing; this is not native ARM proof.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
mkdir -p target/arm64-core-readiness
fixture=$(mktemp -d "$root/target/arm64-core-readiness/attempt-XXXXXXXX")
bash_bin=$(command -v bash)
dirname_bin=$(command -v dirname)
mkdir_bin=$(command -v mkdir)
tools=(uname cargo rustc cc ar python3)

assert_line() {
  local expected=$1 file=$2 line
  while IFS= read -r line; do
    [[ "$line" == "$expected" ]] && return 0
  done < "$file"
  return 1
}

make_bin() {
  local directory=$1 missing=$2
  mkdir -p "$directory"
  ln -s "$dirname_bin" "$directory/dirname"
  ln -s "$mkdir_bin" "$directory/mkdir"
  for tool in "${tools[@]}"; do
    [[ "$tool" == "$missing" ]] && continue
    {
      printf '#!%s\n' "$bash_bin"
      cat <<'MOCK'
if [[ "${0##*/}" == uname ]]; then
  printf '%s\n' "$ARM_GATE_FIXTURE_ARCH"
else
  printf '%s %s\n' "${0##*/}" "$*" >> "$ARM_GATE_FIXTURE_LOG"
fi
MOCK
    } > "$directory/$tool"
    chmod +x "$directory/$tool"
  done
}

for missing in "${tools[@]}"; do
  bin="$fixture/missing-$missing/bin"
  make_bin "$bin" "$missing"
  fixture_home="$fixture/missing-$missing/home"
  mkdir -p "$fixture_home"
  log="$fixture/missing-$missing/executed"
  output="$fixture/missing-$missing/output"
  if HOME="$fixture_home" PATH="$bin" ARM_GATE_FIXTURE_ARCH=aarch64 ARM_GATE_FIXTURE_LOG="$log" \
     "$bash_bin" ops/ci/arm64-core.sh > "$output" 2>&1; then
    echo "ARM core gate accepted missing tool: $missing" >&2
    exit 1
  else
    status=$?
  fi
  [[ "$status" == 1 && ! -e "$log" ]]
  assert_line "Required CI tool is unavailable: $missing" "$output"
done

bin="$fixture/routing/bin"
make_bin "$bin" none
fixture_home="$fixture/routing/home"
mkdir -p "$fixture_home"
log="$fixture/routing/commands"
if HOME="$fixture_home" PATH="$bin" ARM_GATE_FIXTURE_ARCH=x86_64 ARM_GATE_FIXTURE_LOG="$log" \
   "$bash_bin" ops/ci/arm64-core.sh > "$fixture/routing/wrong-arch" 2>&1; then
  echo "ARM core gate accepted x86_64 as native ARM proof" >&2
  exit 1
fi
[[ ! -e "$log" ]]
assert_line "ARM64 core gate requires a native aarch64 host" "$fixture/routing/wrong-arch"

HOME="$fixture_home" PATH="$bin" ARM_GATE_FIXTURE_ARCH=aarch64 ARM_GATE_FIXTURE_LOG="$log" \
  "$bash_bin" ops/ci/arm64-core.sh > "$fixture/routing/mock-success" 2>&1
assert_line 'cargo test --locked --lib' "$log"
assert_line 'cargo test --locked --test codec_golden --test rust_cli --test vnext_raw_schema --test vnext_raw_workflow --test vnext_artifact_only --test vnext_bundle --test vnext_bundle_isolation --test vnext_geometry_binding' "$log"
commands=0
while IFS= read -r line; do
  [[ "$line" != 'cargo test '* ]] || ((commands+=1))
done < "$log"
[[ "$commands" == 2 ]]
printf 'ARM core entry: six missing-tool refusals, wrong-architecture refusal and complete routing passed (fixtures only)\n'
