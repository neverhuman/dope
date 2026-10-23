#!/usr/bin/env bash
# Collect upstream license bytes for the actual normal/build package graph.
set -euo pipefail
[[ $# == 3 && -s "$1" && -s "$2" && ! -e "$3" ]] || exit 2
graph=$1
metadata=$2
out=$3
printf '%s\n' 'DOPE engine: locked CPU dependency notices (normal/build graph).' > "$out"
while IFS=$'\t' read -r name version manifest license; do
  grep -Fxq "$name v$version" "$graph" || continue
  [[ "$license" != NOASSERTION ]] || { echo "engine_license_expression_missing: $name" >&2; exit 1; }
  printf '\n===== %s v%s (%s) =====\n' "$name" "$version" "$license" >> "$out"
  if [[ "$name" == dope-cli || "$name" == dope-core ]]; then
    cat LICENSE >> "$out"
    continue
  fi
  dir=$(dirname -- "$manifest")
  found=false
  while IFS= read -r -d '' file; do
    found=true
    printf '\n--- %s ---\n' "$(basename -- "$file")" >> "$out"
    cat -- "$file" >> "$out"
  done < <(find "$dir" -maxdepth 1 -type f \( -iname 'LICENSE*' -o -iname 'LICENCE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' -o -iname 'COPYRIGHT*' \) -print0 | sort -z)
  if [[ "$found" != true ]]; then
    # Three locked crate archives omit their upstream root license. These
    # version-specific copies retain the exact source commit URL and license.
    notice="ops/licenses/engine/$name-$version.txt"
    [[ -f "$notice" && ! -L "$notice" ]] || { echo "engine_upstream_license_file_missing: $name" >&2; exit 1; }
    cat -- "$notice" >> "$out"
  fi
done < <(jq -r '.packages[] | [.name, .version, .manifest_path, (.license // "NOASSERTION")] | @tsv' "$metadata" | sort)
sysroot=$(rustc --print sysroot)
[[ -d "$sysroot/share/doc/rust/licenses" ]] || { echo 'engine_rust_library_notices_missing' >&2; exit 1; }
for file in "$sysroot"/share/doc/rust/licenses/*; do
  [[ -f "$file" && ! -L "$file" ]] || exit 1
  printf '\n===== Rust library: %s =====\n' "$(basename -- "$file")" >> "$out"
  cat -- "$file" >> "$out"
done
