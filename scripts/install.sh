#!/usr/bin/env bash
# Verified installer for the standalone Rust `dope` CLI and pinned DOPE-MLC-v3 SafeTensors weights.
#
# Engine and model releases have independent concrete tags, staging directories and
# checksum inventories. Signed engine provenance never implies model attestation.
#
# Distinguishes SHA-256 content-integrity verification from cryptographic Sigstore OIDC
# provenance verification (--require-attestation, mandatory for dope-engine-v* tags).
set -euo pipefail

REPO_SLUG="neverhuman/dope"
SIGNER_WORKFLOW="neverhuman/dope/.github/workflows/engine-release.yml"
PUBLIC_TAG="dope-engine-v3.0.0"
PUBLIC_SOURCE="e15c415bad9554f7c419dd240aa670d1a79a27fe"
ENGINE_CORE=(BUILDINFO.json Cargo.lock DEPENDENCIES.txt LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64)
ENGINE_V8=(BUILDINFO.json Cargo.lock DEPENDENCIES.txt LICENSE THIRD_PARTY_NOTICES V8_WEIGHTS_MANIFEST.json dope-linux-x86_64 dope-v8-weights.safetensors)

die() { echo "error: $*" >&2; exit 1; }
require_tools() {
  local tool
  for tool in "$@"; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
}

gh_upgrade_error() {
  die "upgrade gh to 2.102.0 or newer from https://github.com/cli/cli/releases/tag/v2.102.0"
}
gh_policy_supported() {
  local verifier="$1" version help major minor flag
  version=$("$verifier" --version 2>/dev/null) || return 1
  [[ "$version" =~ ^gh[[:space:]]version[[:space:]]([0-9]+)\.([0-9]+)\.([0-9]+)([[:space:]]|$) ]] || return 1
  major=$((10#${BASH_REMATCH[1]})); minor=$((10#${BASH_REMATCH[2]}))
  (( major > 2 || (major == 2 && minor >= 102) )) || return 1
  help=$("$verifier" attestation verify --help 2>/dev/null) || return 1
  for flag in --bundle --hostname --repo --signer-workflow --source-ref --source-digest --format; do
    [[ "$help" =~ (^|[[:space:]])${flag}([[:space:]=,]|$) ]] || return 1
  done
}

check_public_source() {
  [[ "$1" != "$PUBLIC_TAG" || "$2" == "$PUBLIC_SOURCE" ]] ||
    die "source digest differs from the pinned $PUBLIC_TAG release commit"
}

verify_attestation() {
  local dir="$1" tag="$2" source="$3" subject="$4"
  [[ -s "$dir/dope.sigstore.json" && -f "$dir/dope.sigstore.json" && ! -L "$dir/dope.sigstore.json" ]] ||
    die "missing, empty or unsafe dope.sigstore.json bundle"
  "$GH_VERIFIER" attestation verify "$dir/$subject" --bundle "$dir/dope.sigstore.json" \
    --hostname github.com --repo "$REPO_SLUG" --signer-workflow "$SIGNER_WORKFLOW" \
    --source-ref "refs/tags/$tag" --source-digest "$source" --format json >/dev/null ||
    die "attestation verification failed for $subject"
}

engine_inventory() {
  local dir="$1" i digest
  local -a lines
  mapfile -t lines < "$dir/SHA256SUMS"
  case "${#lines[@]}" in
    6) EXPECTED_ENGINE=("${ENGINE_CORE[@]}") ;;
    8) EXPECTED_ENGINE=("${ENGINE_V8[@]}") ;;
    *) die "engine_release_inventory_invalid: expected fixed inventory without partial or duplicate SingleEncV8 members" ;;
  esac
  for i in "${!EXPECTED_ENGINE[@]}"; do
    digest=${lines[$i]:0:64}
    [[ "$digest" =~ ^[0-9a-f]{64}$ && "${lines[$i]}" == "$digest  ${EXPECTED_ENGINE[$i]}" ]] ||
      die "engine_release_inventory_invalid: unexpected checksum entry"
  done
}

destination_dir() {
  local path="$1" require_write="${2:-1}" existing=""
  while [[ "$path" != / ]]; do
    [[ ! -L "$path" && ( ! -e "$path" || -d "$path" ) ]] ||
      die "refusing to install through symlink or non-directory path: $path"
    if [[ -d "$path" && -z "$existing" ]]; then existing=$path; fi
    path=${path%/*}
    path=${path:-/}
  done
  existing=${existing:-/}
  [[ $require_write -eq 0 || ( -w "$existing" && -x "$existing" ) ]] ||
    die "destination is not writable: $existing"
}

# The manual and release-workflow verifier sources the same signed policy.
[[ "${BASH_SOURCE[0]:-$0}" == "$0" ]] || return 0

TAG="${DOPE_RELEASE_TAG:-dope-engine-v3.0.0}"
SOURCE_COMMIT="${DOPE_SOURCE_COMMIT:-}"
FROM_DIR=""
STANDALONE_PREFIX=""
CUSTOM_DIRS=0
WITH_MODELS=0
MODEL_TAG="${DOPE_MODEL_TAG:-}"
MODEL_FROM_DIR=""
MODEL_EXPLICIT=0
[[ -z "$MODEL_TAG" ]] || MODEL_EXPLICIT=1
INSTALL_DIR="${DOPE_INSTALL_DIR:-$HOME/.local/bin}"
WEIGHTS_DIR="${DOPE_WEIGHTS_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/dope/weights}"
SHARE_DIR="${DOPE_SHARE_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/dope-engine}"
REQUIRE_ATTESTATION=0
SKIP_WEIGHTS=0
UNINSTALL=0
V8_WEIGHTS_MODE="${DOPE_INSTALL_V8_WEIGHTS:-auto}"
V8_WEIGHTS_EXPLICIT=0
if [[ "$V8_WEIGHTS_MODE" == "1" || "$V8_WEIGHTS_MODE" == "require" ]]; then
  V8_WEIGHTS_MODE=require
  V8_WEIGHTS_EXPLICIT=1
elif [[ "$V8_WEIGHTS_MODE" == "0" || "$V8_WEIGHTS_MODE" == "skip" ]]; then
  V8_WEIGHTS_MODE=skip
  V8_WEIGHTS_EXPLICIT=1
elif [[ "$V8_WEIGHTS_MODE" != "auto" ]]; then
  echo "error: DOPE_INSTALL_V8_WEIGHTS must be auto, require/1, or skip/0" >&2
  exit 2
fi
COLOR_MODE="${DOPE_COLOR:-auto}"

usage() {
  cat <<'EOF'
Usage: install.sh [OPTIONS]

Options:
  --tag <TAG>               Engine/binary release tag (default: dope-engine-v3.0.0)
  --model-tag <TAG>         Independent model release (engine tag default: v1.0.0; otherwise same as --tag)
  --model-from-dir <DIR>    Independent offline model release, including its SHA256SUMS
  --source-digest <SHA40>   Expected 40-char git commit SHA for OIDC attestation verification
  --require-attestation     Require cryptographic Sigstore OIDC attestation (mandatory for dope-engine-v* tags)
  --from-dir <DIR>          Offline engine directory; never falls back to an HTTPS model download
  --standalone-prefix <DIR> Install a signed engine without weights into DIR in one rename
  --install-dir <DIR>       Destination directory for the `dope` binary (default: ~/.local/bin)
  --weights-dir <DIR>       Destination directory for verified .safetensors weights (default: ~/.cache/dope/weights)
  --share-dir <DIR>         Destination directory for LICENSE and THIRD_PARTY_NOTICES (default: ~/.local/share/dope-engine)
  --v8-weights              Require and install attested SingleEncV8 weights from the engine release
  --with-weights            Install and smoke-test the independent model release
  --no-v8-weights           Skip SingleEncV8 weights even if bundled with the engine release
  --no-weights              Install only the standalone `dope` binary and legal notices (skip .safetensors weights)
  --uninstall               Remove owned files from the custom-directory install
  --color <auto|always|never>
  -h, --help                Show this help

Signed engine tags require gh >= 2.102.0. Online standalone installs bootstrap
the pinned verifier when needed. Signed engine tags install the analytic binary
by default, with independent model downloads enabled by an explicit model flag.
The engine tag selects the binary; --model-tag selects models independently:
  ./scripts/install.sh --tag dope-engine-v3.0.0 --model-tag v1.0.0
  ./scripts/install.sh --tag dope-engine-v3.0.0 --no-weights
For an offline split, supply --from-dir ENGINE_DIR --model-from-dir MODEL_DIR.
A one-directory offline release with no model members remains engine-only unless
models were explicitly requested. Partial models are refused, never relabeled.
EOF
}

main() {
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag|--version) TAG="${2:-}"; shift 2 ;;
    --source-digest) SOURCE_COMMIT="${2:-}"; shift 2 ;;
    --require-attestation) REQUIRE_ATTESTATION=1; shift ;;
    --from-dir) FROM_DIR="${2:-}"; shift 2 ;;
    --standalone-prefix) [[ -n "${2:-}" ]] || die "--standalone-prefix requires a directory"; STANDALONE_PREFIX=$2; shift 2 ;;
    --model-tag) MODEL_TAG="${2:-}"; [[ -n "$MODEL_TAG" ]] || { echo "error: --model-tag requires a concrete tag" >&2; exit 2; }; MODEL_EXPLICIT=1; shift 2 ;;
    --model-from-dir) MODEL_FROM_DIR="${2:-}"; [[ -n "$MODEL_FROM_DIR" ]] || { echo "error: --model-from-dir requires a nonempty directory" >&2; exit 2; }; MODEL_EXPLICIT=1; shift 2 ;;
    --install-dir) INSTALL_DIR="${2:-}"; CUSTOM_DIRS=1; shift 2 ;;
    --prefix) INSTALL_DIR="${2:-}/bin"; SHARE_DIR="${2:-}/share/dope-engine"; CUSTOM_DIRS=1; shift 2 ;;
    --weights-dir|--cache-dir) WEIGHTS_DIR="${2:-}"; CUSTOM_DIRS=1; shift 2 ;;
    --share-dir) SHARE_DIR="${2:-}"; CUSTOM_DIRS=1; shift 2 ;;
    --with-weights|--verify-sample) SKIP_WEIGHTS=0; MODEL_EXPLICIT=1; WITH_MODELS=1; shift ;;
    --v8-weights) V8_WEIGHTS_MODE=require; V8_WEIGHTS_EXPLICIT=1; shift ;;
    --no-v8-weights) V8_WEIGHTS_MODE=skip; V8_WEIGHTS_EXPLICIT=1; shift ;;
    --no-weights) SKIP_WEIGHTS=1; [[ $V8_WEIGHTS_EXPLICIT -eq 1 ]] || V8_WEIGHTS_MODE=skip; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    --color) COLOR_MODE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ $UNINSTALL -eq 1 ]]; then
  [[ -z "$STANDALONE_PREFIX" ]] || die "--uninstall acts on the custom-directory install"
  require_tools realpath rm
  owned=(
    "$INSTALL_DIR/dope"
    "$SHARE_DIR/LICENSE"
    "$SHARE_DIR/THIRD_PARTY_NOTICES"
    "$SHARE_DIR/BUILDINFO.json"
    "$WEIGHTS_DIR/MODEL_MANIFEST.json"
    "$WEIGHTS_DIR/dope_mlc_v3_multitask_best.safetensors"
    "$WEIGHTS_DIR/packet_high_regression.dml3"
    "$WEIGHTS_DIR/single-enc-v8/manifest.json"
    "$WEIGHTS_DIR/single-enc-v8/model.safetensors"
  )
  for dir in "$INSTALL_DIR" "$SHARE_DIR" "$WEIGHTS_DIR" "$WEIGHTS_DIR/single-enc-v8"; do
    [[ -n "$dir" ]] || die "destination directory is empty"
    destination_dir "$(realpath -ms -- "$dir")" 0
  done
  for path in "${owned[@]}"; do
    [[ ! -L "$path" ]] || die "refusing to remove symlink: $path"
    [[ ! -e "$path" || -f "$path" ]] || die "refusing to remove non-regular file: $path"
  done
  for path in "${owned[@]}"; do
    if [[ -f "$path" ]]; then rm -f -- "$path"; fi
  done
  echo "uninstalled owned dope files"
  exit 0
fi

USE_COLOR=0
if [[ "$COLOR_MODE" == "always" ]]; then
  USE_COLOR=1
elif [[ "$COLOR_MODE" == "auto" && -z "${NO_COLOR:-}" && -t 1 ]]; then
  USE_COLOR=1
fi

c_bold()  { if [[ $USE_COLOR -eq 1 ]]; then printf '\033[1m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_cyan()  { if [[ $USE_COLOR -eq 1 ]]; then printf '\033[1;36m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_green() { if [[ $USE_COLOR -eq 1 ]]; then printf '\033[1;32m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_yellow(){ if [[ $USE_COLOR -eq 1 ]]; then printf '\033[1;33m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_dim()   { if [[ $USE_COLOR -eq 1 ]]; then printf '\033[2m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }

# 1. Resolve independent concrete tags (refuse floating names and traversal).
if [[ -z "$MODEL_TAG" ]]; then
  if [[ "$TAG" == dope-engine-v* ]]; then MODEL_TAG=v1.0.0; else MODEL_TAG=$TAG; fi
fi
for selected_tag in "$TAG" "$MODEL_TAG"; do
  if [[ -z "$selected_tag" || "${selected_tag,,}" == latest || "$selected_tag" == *"/"* || "$selected_tag" == *".."* || ! "$selected_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "error: release_tag_invalid: must be a concrete release tag, not '$selected_tag'" >&2
    exit 2
  fi
done
if [[ "$TAG" == dope-engine-v* ]]; then
  REQUIRE_ATTESTATION=1
  if [[ $MODEL_EXPLICIT -eq 0 ]]; then SKIP_WEIGHTS=1; fi
fi
if [[ "$TAG" == "$PUBLIC_TAG" ]]; then SOURCE_COMMIT=${SOURCE_COMMIT:-$PUBLIC_SOURCE}; fi
check_public_source "$TAG" "$SOURCE_COMMIT"
if [[ $REQUIRE_ATTESTATION -eq 1 && -n "$SOURCE_COMMIT" && ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  die "--source-digest must be a 40-character lowercase commit SHA"
fi
if [[ -n "$STANDALONE_PREFIX" ]]; then
  [[ "$TAG" == dope-engine-v* && $CUSTOM_DIRS -eq 0 && $WITH_MODELS -eq 0 &&
     $MODEL_EXPLICIT -eq 0 && "$V8_WEIGHTS_MODE" != require ]] ||
    die "--standalone-prefix requires a signed engine tag and is incompatible with custom directory or model-install flags"
  SKIP_WEIGHTS=1
  V8_WEIGHTS_MODE=skip
fi
if [[ "$V8_WEIGHTS_MODE" == "require" && ( "$TAG" != dope-engine-v* || $REQUIRE_ATTESTATION -ne 1 ) ]]; then
  echo "error: --v8-weights requires an attested dope-engine-v* release tag (got '$TAG')" >&2
  exit 1
fi

# 2. Platform & tool check.
require_tools uname getconf
if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "error: prebuilt binary is qualified for Linux x86_64 only (detected $(uname -s):$(uname -m))." >&2
  echo "  hint: build from source with: cargo build --manifest-path engine/Cargo.toml --locked --release --bin dope" >&2
  exit 1
fi
if [[ "$TAG" == dope-engine-v* ]]; then
  libc=$(getconf GNU_LIBC_VERSION 2>/dev/null) || libc=unknown
  if [[ ! "$libc" =~ ^glibc\ ([0-9]+)\.([0-9]+)(\.[0-9]+)?$ ]] ||
     ! (( 10#${BASH_REMATCH[1]} > 2 || (10#${BASH_REMATCH[1]} == 2 && 10#${BASH_REMATCH[2]} >= 39) )); then
    die "this signed engine requires glibc 2.39 or newer (detected $libc)"
  fi
fi
require_tools sha256sum jq awk install mktemp mkdir chmod mv rm cp realpath wc tr

MODEL_ASSETS=(MODEL_MANIFEST.json dope_mlc_v3_multitask_best.safetensors packet_high_regression.dml3)
if [[ $SKIP_WEIGHTS -eq 0 && -n "$FROM_DIR" && -z "$MODEL_FROM_DIR" ]]; then
  local_models=0
  for name in "${MODEL_ASSETS[@]}"; do
    if [[ -e "$FROM_DIR/$name" || -L "$FROM_DIR/$name" ]]; then local_models=1; fi
  done
  if [[ $MODEL_EXPLICIT -eq 0 && $local_models -eq 0 ]]; then
    SKIP_WEIGHTS=1
  elif [[ "$MODEL_TAG" != "$TAG" ]]; then
    die "offline split model request requires --model-from-dir for model tag $MODEL_TAG"
  else
    MODEL_FROM_DIR=$FROM_DIR
  fi
fi
if [[ -z "$FROM_DIR" || ( $SKIP_WEIGHTS -eq 0 && -z "$MODEL_FROM_DIR" ) ]]; then
  require_tools curl
fi

destination_files() {
  local dir="$1" name
  shift
  [[ -n "$dir" ]] || die "destination directory is empty"
  destination_dir "$(realpath -ms -- "$dir")"
  for name in "$@"; do
    [[ ! -L "$dir/$name" && ( ! -e "$dir/$name" || -f "$dir/$name" ) ]] ||
      die "refusing to install over symlink or non-regular path: $dir/$name"
  done
}

notices=(LICENSE THIRD_PARTY_NOTICES BUILDINFO.json)
standalone_layout() (
  shopt -s nullglob dotglob
  local entries name dir
  for dir in "$STANDALONE_PREFIX" "$INSTALL_DIR" "$STANDALONE_PREFIX/share" "$SHARE_DIR"; do
    [[ -d "$dir" && ! -L "$dir" ]] || exit 1
  done
  entries=("$STANDALONE_PREFIX"/*); [[ ${#entries[@]} -eq 2 ]] || exit 1
  entries=("$INSTALL_DIR"/*); [[ ${#entries[@]} -eq 1 ]] || exit 1
  entries=("$STANDALONE_PREFIX/share"/*); [[ ${#entries[@]} -eq 1 ]] || exit 1
  entries=("$SHARE_DIR"/*); [[ ${#entries[@]} -eq ${#notices[@]} ]] || exit 1
  [[ -f "$INSTALL_DIR/dope" && -x "$INSTALL_DIR/dope" && ! -L "$INSTALL_DIR/dope" ]] || exit 1
  for name in "${notices[@]}"; do
    [[ -f "$SHARE_DIR/$name" && ! -L "$SHARE_DIR/$name" ]] || exit 1
  done
)

V8_DST_DIR="$WEIGHTS_DIR/single-enc-v8"
if [[ -n "$STANDALONE_PREFIX" ]]; then
  require_tools cmp
  STANDALONE_PREFIX=$(realpath -ms -- "$STANDALONE_PREFIX")
  [[ "$STANDALONE_PREFIX" != / ]] || die "refusing to install at /"
  PREFIX_PARENT=${STANDALONE_PREFIX%/*}
  PREFIX_PARENT=${PREFIX_PARENT:-/}
  INSTALL_DIR="$STANDALONE_PREFIX/bin"
  SHARE_DIR="$STANDALONE_PREFIX/share/dope-engine"
  if [[ -e "$STANDALONE_PREFIX" || -L "$STANDALONE_PREFIX" ]]; then
    destination_dir "$PREFIX_PARENT" 0
    standalone_layout || die "conflicting or symlinked standalone destination: $STANDALONE_PREFIX"
  else
    destination_dir "$PREFIX_PARENT"
  fi
else
  destination_files "$INSTALL_DIR" dope
  destination_files "$SHARE_DIR" "${notices[@]}"
  if [[ $SKIP_WEIGHTS -eq 0 ]]; then destination_files "$WEIGHTS_DIR" "${MODEL_ASSETS[@]}"; fi
  if [[ "$V8_WEIGHTS_MODE" == require ]]; then destination_files "$V8_DST_DIR" manifest.json model.safetensors; fi
fi

GH_VERIFIER=""
if [[ $REQUIRE_ATTESTATION -eq 1 ]]; then
  GH_VERIFIER=$(command -v gh || true)
  if [[ -z "$GH_VERIFIER" ]] || ! gh_policy_supported "$GH_VERIFIER"; then
    [[ -n "$STANDALONE_PREFIX" && -z "$FROM_DIR" ]] || gh_upgrade_error
    require_tools tar gzip
    GH_VERIFIER=""
  fi
fi

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/dope-install.XXXXXX")
PREFIX_STAGE=""
LEGACY_TMP=""
cleanup_install() {
  rm -rf -- "$STAGE"
  if [[ -n "$PREFIX_STAGE" ]]; then rm -rf -- "$PREFIX_STAGE"; fi
  if [[ -n "$LEGACY_TMP" ]]; then rm -f -- "$LEGACY_TMP"; fi
}
trap cleanup_install EXIT
ENGINE_STAGE="$STAGE/engine"
MODEL_STAGE="$STAGE/models"
mkdir -- "$ENGINE_STAGE" "$MODEL_STAGE"

download() {
  curl -q -fsSL --proto '=https' --tlsv1.2 --retry 2 --connect-timeout 15 --max-time 600 -o "$2" "$1"
}
if [[ $REQUIRE_ATTESTATION -eq 1 && -z "$GH_VERIFIER" ]]; then
  gh_dist=gh_2.102.0_linux_amd64
  archive="$STAGE/$gh_dist.tar.gz"
  GH_VERIFIER="$STAGE/$gh_dist/bin/gh"
  download "https://github.com/cli/cli/releases/download/v2.102.0/$gh_dist.tar.gz" "$archive" ||
    die "failed to download the pinned GitHub CLI"
  printf '%s  %s\n' bb766f710eef8ede859c18578c72c327597cd4c8a85b06001b1f3843c6019386 "$gh_dist.tar.gz" |
    (cd -- "$STAGE" && sha256sum --check --status) || die "GitHub CLI archive SHA-256 mismatch"
  tar -xzf "$archive" -C "$STAGE" "$gh_dist/bin/gh"
  printf '%s  %s\n' 7469124f706944133d6a169691dd1c6c3511b12e85878d255e044e2948df4c9b "$gh_dist/bin/gh" |
    (cd -- "$STAGE" && sha256sum --check --status) || die "GitHub CLI executable SHA-256 mismatch"
  gh_policy_supported "$GH_VERIFIER" || gh_upgrade_error
fi

echo "$(c_cyan "● DOPE Installer") · concrete release tag $(c_bold "$TAG") · repo $(c_bold "$REPO_SLUG")"
REQUIRED_CORE=(BUILDINFO.json LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64 SHA256SUMS)

fetch_asset() {
  local stage="$1" from_dir="$2" tag="$3" name="$4"
  local dst="$stage/$name"
  if [[ -n "$from_dir" ]]; then
    local src="$from_dir/$name"
    if [[ -L "$src" || ! -f "$src" ]]; then
      echo "error: missing, symlinked or non-regular release asset: $src" >&2
      exit 1
    fi
    cp -- "$src" "$dst"
  else
    local url="https://github.com/$REPO_SLUG/releases/download/$tag/$name"
    if ! download "$url" "$dst"; then
      echo "error: failed to download required asset: $url" >&2
      exit 1
    fi
  fi
}
if [[ "$TAG" == dope-engine-v* ]]; then
  REQUIRED_CORE=("${ENGINE_CORE[@]}" SHA256SUMS dope.sigstore.json)
elif [[ $REQUIRE_ATTESTATION -eq 1 ]]; then
  REQUIRED_CORE+=(dope.sigstore.json)
fi
for name in "${REQUIRED_CORE[@]}"; do fetch_asset "$ENGINE_STAGE" "$FROM_DIR" "$TAG" "$name"; done

HAVE_MODEL_ASSETS=0
if [[ $SKIP_WEIGHTS -eq 0 ]]; then
  for name in SHA256SUMS "${MODEL_ASSETS[@]}"; do fetch_asset "$MODEL_STAGE" "$MODEL_FROM_DIR" "$MODEL_TAG" "$name"; done
  HAVE_MODEL_ASSETS=1
fi

# 3. Validate each checksum inventory, then verify staged bytes before execution.
validate_sha256sums() {
  local stage="$1" line
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" ]] || continue
    if [[ ! "$line" =~ ^([0-9a-f]{64})\ \ ([A-Za-z0-9._-]+)$ ]]; then
      echo "error: malformed or unsafe entry in SHA256SUMS: $line" >&2
      exit 1
    fi
  done < "$stage/SHA256SUMS"
}
if [[ "$TAG" == dope-engine-v* ]]; then
  engine_inventory "$ENGINE_STAGE"
  if [[ -n "$FROM_DIR" && ${#EXPECTED_ENGINE[@]} -eq 6 ]]; then
    for name in V8_WEIGHTS_MANIFEST.json dope-v8-weights.safetensors; do
      [[ ! -e "$FROM_DIR/$name" && ! -L "$FROM_DIR/$name" ]] || die "engine_release_stray_v8_weights_forbidden"
    done
  fi
else
  validate_sha256sums "$ENGINE_STAGE"
fi
if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then validate_sha256sums "$MODEL_STAGE"; fi

verify_in_sha256sums() {
  local stage="$1" name="$2"
  local expected
  expected=$(awk -v f="$name" '$2 == f {print $1}' "$stage/SHA256SUMS")
  if [[ -z "$expected" || ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    echo "error: $name missing from SHA256SUMS" >&2
    exit 1
  fi
  local actual
  actual=$(sha256sum < "$stage/$name"); actual=${actual%% *}
  if [[ "$actual" != "$expected" ]]; then
    echo "error: SHA-256 mismatch for $name: expected $expected, got $actual" >&2
    exit 1
  fi
}

core_checks=(BUILDINFO.json LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64)
if [[ "$TAG" == dope-engine-v* ]]; then core_checks=("${ENGINE_CORE[@]}"); fi
for name in "${core_checks[@]}"; do
  verify_in_sha256sums "$ENGINE_STAGE" "$name"
done

# Detect and validate optional SingleEncV8 weights bundled with the engine release.
HAS_V8_MANIFEST_SUM=$(awk '$2 == "V8_WEIGHTS_MANIFEST.json" {n++} END {print n+0}' "$ENGINE_STAGE/SHA256SUMS")
HAS_V8_MODEL_SUM=$(awk '$2 == "dope-v8-weights.safetensors" {n++} END {print n+0}' "$ENGINE_STAGE/SHA256SUMS")
HAVE_V8_WEIGHTS=0
V8_ACTUAL_SHA=""
V8_ACTUAL_BYTES=0
if [[ $HAS_V8_MANIFEST_SUM -ne $HAS_V8_MODEL_SUM || $HAS_V8_MANIFEST_SUM -gt 1 ]]; then
  echo "error: partial or duplicate SingleEncV8 weights inventory in engine SHA256SUMS" >&2
  exit 1
fi
if [[ $HAS_V8_MANIFEST_SUM -eq 1 ]]; then
  if [[ "$V8_WEIGHTS_MODE" != "skip" ]]; then
    if [[ "$TAG" != dope-engine-v* || $REQUIRE_ATTESTATION -ne 1 ]]; then
      echo "error: bundled SingleEncV8 weights require an attested dope-engine-v* release tag (pass --no-v8-weights to skip on legacy tags)" >&2
      exit 1
    fi
    destination_files "$V8_DST_DIR" manifest.json model.safetensors
    for name in V8_WEIGHTS_MANIFEST.json dope-v8-weights.safetensors; do
      fetch_asset "$ENGINE_STAGE" "$FROM_DIR" "$TAG" "$name"
      verify_in_sha256sums "$ENGINE_STAGE" "$name"
    done
    HAVE_V8_WEIGHTS=1
  fi
elif [[ "$V8_WEIGHTS_MODE" == "require" ]]; then
  echo "error: release $TAG does not ship SingleEncV8 weights in SHA256SUMS" >&2
  exit 1
fi

# Cross-check BUILDINFO.json binary_sha256 against dope-linux-x86_64.
BIN_SHA=$(sha256sum "$ENGINE_STAGE/dope-linux-x86_64" | awk '{print $1}')
BUILDINFO_BIN_SHA=$(jq -r '.binary_sha256 // empty' "$ENGINE_STAGE/BUILDINFO.json")
if [[ -z "$BUILDINFO_BIN_SHA" || "$BIN_SHA" != "$BUILDINFO_BIN_SHA" ]]; then
  echo "error: BUILDINFO.json binary_sha256 ($BUILDINFO_BIN_SHA) does not match dope-linux-x86_64 ($BIN_SHA)" >&2
  exit 1
fi

if [[ $HAVE_V8_WEIGHTS -eq 1 ]]; then
  V8_ACTUAL_SHA=$(sha256sum "$ENGINE_STAGE/dope-v8-weights.safetensors" | awk '{print $1}')
  V8_ACTUAL_BYTES=$(wc -c < "$ENGINE_STAGE/dope-v8-weights.safetensors" | tr -d ' ')
  jq -e --arg sha "$V8_ACTUAL_SHA" --argjson bytes "$V8_ACTUAL_BYTES" \
    '.schema == "dope.single-enc-v8-weights/v1" and .file == "model.safetensors" and .arch == "single_enc_v8" and .sha256 == $sha and .bytes == $bytes and .bytes > 0 and .bytes <= 1073741824' \
    "$ENGINE_STAGE/V8_WEIGHTS_MANIFEST.json" >/dev/null || {
    echo "error: SingleEncV8 weights failed V8_WEIGHTS_MANIFEST.json verification" >&2
    exit 1
  }
  jq -e --arg sha "$V8_ACTUAL_SHA" --argjson bytes "$V8_ACTUAL_BYTES" \
    '.v8_weights_sha256 == $sha and .v8_weights_bytes == $bytes' \
    "$ENGINE_STAGE/BUILDINFO.json" >/dev/null || {
    echo "error: BUILDINFO.json v8_weights metadata does not match dope-v8-weights.safetensors" >&2
    exit 1
  }
fi

if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then
  for name in MODEL_MANIFEST.json dope_mlc_v3_multitask_best.safetensors packet_high_regression.dml3; do
    verify_in_sha256sums "$MODEL_STAGE" "$name"
  done
  MANIFEST_TAG=$(jq -r '.release_tag // empty' "$MODEL_STAGE/MODEL_MANIFEST.json")
  if [[ "$MANIFEST_TAG" != "$MODEL_TAG" ]]; then
    echo "error: MODEL_MANIFEST.json release_tag ($MANIFEST_TAG) does not match requested model tag ($MODEL_TAG)" >&2
    exit 1
  fi
  ST_EXPECTED_SHA=$(jq -r '.models[] | select(.filename == "dope_mlc_v3_multitask_best.safetensors") | .sha256' "$MODEL_STAGE/MODEL_MANIFEST.json")
  ST_EXPECTED_BYTES=$(jq -r '.models[] | select(.filename == "dope_mlc_v3_multitask_best.safetensors") | .bytes' "$MODEL_STAGE/MODEL_MANIFEST.json")
  ST_ACTUAL_SHA=$(sha256sum "$MODEL_STAGE/dope_mlc_v3_multitask_best.safetensors" | awk '{print $1}')
  ST_ACTUAL_BYTES=$(wc -c < "$MODEL_STAGE/dope_mlc_v3_multitask_best.safetensors" | tr -d ' ')
  if [[ "$ST_ACTUAL_SHA" != "$ST_EXPECTED_SHA" || "$ST_ACTUAL_BYTES" != "$ST_EXPECTED_BYTES" ]]; then
    echo "error: SafeTensors weights failed MODEL_MANIFEST.json verification" >&2
    exit 1
  fi
  WIRE_EXPECTED_SHA=$(jq -r '.sample_wire_fixture.sha256' "$MODEL_STAGE/MODEL_MANIFEST.json")
  WIRE_EXPECTED_BYTES=$(jq -r '.sample_wire_fixture.bytes' "$MODEL_STAGE/MODEL_MANIFEST.json")
  WIRE_ACTUAL_SHA=$(sha256sum "$MODEL_STAGE/packet_high_regression.dml3" | awk '{print $1}')
  WIRE_ACTUAL_BYTES=$(wc -c < "$MODEL_STAGE/packet_high_regression.dml3" | tr -d ' ')
  if [[ "$WIRE_ACTUAL_SHA" != "$WIRE_EXPECTED_SHA" || "$WIRE_ACTUAL_BYTES" != "$WIRE_EXPECTED_BYTES" ]]; then
    echo "error: sample DML3 wire packet failed MODEL_MANIFEST.json verification" >&2
    exit 1
  fi
fi

# 4. Provenance / Sigstore OIDC attestation verification against repository identity policy.
VERIFICATION_LEVEL="sha256-content-integrity-only (pinned SHA256SUMS + BUILDINFO.json verified; not a cryptographic signature or OIDC attestation)"
ATTESTATION_VERIFIED=0

if [[ $REQUIRE_ATTESTATION -eq 1 ]]; then
  if [[ -z "$SOURCE_COMMIT" ]]; then
    SOURCE_COMMIT=$("$GH_VERIFIER" api "repos/$REPO_SLUG/commits/$TAG" --jq .sha 2>/dev/null || true)
  fi
  if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: could not resolve 40-char source commit SHA for tag $TAG; pass --source-digest <SHA40>" >&2
    exit 1
  fi
  if [[ "$(jq -r '.source_commit // empty' "$ENGINE_STAGE/BUILDINFO.json")" != "$SOURCE_COMMIT" ]]; then
    echo "error: BUILDINFO.json source_commit differs from expected engine source digest" >&2
    exit 1
  fi
  attest_subjects=(dope-linux-x86_64 SHA256SUMS)
  if [[ $HAVE_V8_WEIGHTS -eq 1 ]]; then
    attest_subjects+=(dope-v8-weights.safetensors V8_WEIGHTS_MANIFEST.json)
  fi
  for subject in "${attest_subjects[@]}"; do
    verify_attestation "$ENGINE_STAGE" "$TAG" "$SOURCE_COMMIT" "$subject"
  done
  ATTESTATION_VERIFIED=1
  VERIFICATION_LEVEL="sigstore-oidc-attestation + sha256-content-integrity (verified against $SIGNER_WORKFLOW @ refs/tags/$TAG [$SOURCE_COMMIT])"

fi

standalone_matches() {
  local name
  standalone_layout && cmp -s -- "$ENGINE_STAGE/dope-linux-x86_64" "$INSTALL_DIR/dope" || return 1
  for name in "${notices[@]}"; do
    cmp -s -- "$ENGINE_STAGE/$name" "$SHARE_DIR/$name" || return 1
  done
}

print_standalone() {
  printf '%s %s\n' "$1" "$INSTALL_DIR/dope"
  printf '  Verified %s\n' "$VERIFICATION_LEVEL"
  printf '  Optional: export PATH=%q:"%s"\n' "$INSTALL_DIR" "\$PATH"
}

if [[ -n "$STANDALONE_PREFIX" && ( -e "$STANDALONE_PREFIX" || -L "$STANDALONE_PREFIX" ) ]]; then
  standalone_matches || die "conflicting or symlinked standalone destination: $STANDALONE_PREFIX"
  print_standalone "Already installed"
  exit 0
fi

chmod u+x -- "$ENGINE_STAGE/dope-linux-x86_64"
# 5. Staged self-checks; every verification finishes before installation writes.
"$ENGINE_STAGE/dope-linux-x86_64" doctor --json >/dev/null
if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then
  DOPE_WEIGHTS_DIR="$MODEL_STAGE" "$ENGINE_STAGE/dope-linux-x86_64" mlc-v3 inspect-weights >/dev/null
  "$ENGINE_STAGE/dope-linux-x86_64" mlc-v3 inspect-wire --wire "$MODEL_STAGE/packet_high_regression.dml3" >/dev/null
  DOPE_WEIGHTS_DIR="$MODEL_STAGE" "$ENGINE_STAGE/dope-linux-x86_64" mlc-v3 sample \
    --wire "$MODEL_STAGE/packet_high_regression.dml3" -n 16 --seed 42 -o "$STAGE/smoke.csv" >/dev/null
fi

# 6. Activate a complete standalone installation with one no-clobber rename.
if [[ -n "$STANDALONE_PREFIX" ]]; then
  destination_dir "$PREFIX_PARENT"
  mkdir -p -- "$PREFIX_PARENT"
  PREFIX_STAGE=$(mktemp -d "$PREFIX_PARENT/.dope-install.XXXXXX")
  install -Dm755 -- "$ENGINE_STAGE/dope-linux-x86_64" "$PREFIX_STAGE/bin/dope"
  for name in "${notices[@]}"; do
    install -Dm644 -- "$ENGINE_STAGE/$name" "$PREFIX_STAGE/share/dope-engine/$name"
  done
  if mv -nT -- "$PREFIX_STAGE" "$STANDALONE_PREFIX" && [[ ! -e "$PREFIX_STAGE" ]]; then
    PREFIX_STAGE=""
    print_standalone "Installed"
  else
    standalone_matches || die "conflicting or symlinked standalone destination: $STANDALONE_PREFIX"
    print_standalone "Already installed"
  fi
  exit 0
fi

# The legacy custom-directory route replaces each destination file atomically.
atomic_install() {
  local src="$1"
  local dst_dir="$2"
  local dst_name="$3"
  local mode="$4"
  mkdir -p -- "$dst_dir"
  local dst="$dst_dir/$dst_name"
  if [[ -L "$dst" || ( -e "$dst" && ! -f "$dst" ) ]]; then
    echo "error: refusing to overwrite symlink or non-regular destination at $dst" >&2
    exit 1
  fi
  LEGACY_TMP=$(mktemp "$dst_dir/.${dst_name}.tmp.XXXXXX")
  if ! install -m "$mode" "$src" "$LEGACY_TMP" || ! mv -fT -- "$LEGACY_TMP" "$dst"; then
    die "could not install $dst"
  fi
  LEGACY_TMP=""
}

atomic_install "$ENGINE_STAGE/dope-linux-x86_64" "$INSTALL_DIR" "dope" 0755
atomic_install "$ENGINE_STAGE/LICENSE" "$SHARE_DIR" "LICENSE" 0644
atomic_install "$ENGINE_STAGE/THIRD_PARTY_NOTICES" "$SHARE_DIR" "THIRD_PARTY_NOTICES" 0644
atomic_install "$ENGINE_STAGE/BUILDINFO.json" "$SHARE_DIR" "BUILDINFO.json" 0644

if [[ $HAVE_V8_WEIGHTS -eq 1 ]]; then
  atomic_install "$ENGINE_STAGE/V8_WEIGHTS_MANIFEST.json" "$V8_DST_DIR" "manifest.json" 0644
  atomic_install "$ENGINE_STAGE/dope-v8-weights.safetensors" "$V8_DST_DIR" "model.safetensors" 0644
fi

if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then
  atomic_install "$MODEL_STAGE/MODEL_MANIFEST.json" "$WEIGHTS_DIR" "MODEL_MANIFEST.json" 0644
  atomic_install "$MODEL_STAGE/dope_mlc_v3_multitask_best.safetensors" "$WEIGHTS_DIR" "dope_mlc_v3_multitask_best.safetensors" 0644
  atomic_install "$MODEL_STAGE/packet_high_regression.dml3" "$WEIGHTS_DIR" "packet_high_regression.dml3" 0644
fi

echo "$(c_green "✓ Installed") $(c_bold "$INSTALL_DIR/dope") (sha256: ${BIN_SHA:0:16}…)"
echo "  $(c_dim "notices")       $SHARE_DIR/{LICENSE,THIRD_PARTY_NOTICES,BUILDINFO.json}"
if [[ $HAVE_V8_WEIGHTS -eq 1 ]]; then
  echo "  $(c_dim "v8 weights")    $WEIGHTS_DIR/single-enc-v8/model.safetensors (${V8_ACTUAL_BYTES} B, sha256: ${V8_ACTUAL_SHA:0:16}…)"
  echo "  $(c_dim "v8 verification") tag $TAG · sigstore-oidc-attestation + sha256-content-integrity (build/packaging identity only; not quality or privacy certification)"
fi
if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then
  echo "  $(c_dim "weights")       $WEIGHTS_DIR/dope_mlc_v3_multitask_best.safetensors (${ST_ACTUAL_BYTES} B, sha256: ${ST_ACTUAL_SHA:0:16}…)"
  echo "  $(c_dim "sample wire")   $WEIGHTS_DIR/packet_high_regression.dml3 (${WIRE_ACTUAL_BYTES} B, sha256: ${WIRE_ACTUAL_SHA:0:16}…)"
  echo "  $(c_dim "model verification") tag $MODEL_TAG · sha256-content-integrity-only (independent model inventory; no model attestation claimed)"
fi
if [[ $ATTESTATION_VERIFIED -eq 1 ]]; then
  echo "  $(c_dim "verification")  $(c_green "$VERIFICATION_LEVEL")"
else
  echo "  $(c_dim "verification")  $(c_yellow "$VERIFICATION_LEVEL")"
fi
echo ""
DOPE_CMD="$INSTALL_DIR/dope"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) DOPE_CMD=dope ;;
  *) printf '  Optional: export PATH=%q:"%s"\n' "$INSTALL_DIR" "\$PATH" ;;
esac
other_dope=$(command -v dope 2>/dev/null || true)
if [[ -n "$other_dope" && "$other_dope" != "$INSTALL_DIR/dope" ]]; then
  printf '  Another dope is first on PATH: %s\n' "$other_dope"
  DOPE_CMD="$INSTALL_DIR/dope"
fi
echo "  $DOPE_CMD --version"
echo "  $DOPE_CMD doctor"
echo "  $DOPE_CMD fit train.csv --target y --format auto-v8 --budget 2048 --seed 1 -o model.dope"
echo "  $DOPE_CMD sample model.dope -n 1000 --seed 1 -o synthetic.csv"
if [[ $HAVE_V8_WEIGHTS -eq 1 ]]; then
  echo "  $DOPE_CMD fit data.csv --target y --format auto-v8 --v8-weights $WEIGHTS_DIR/single-enc-v8/model.safetensors -o model.dope"
fi
if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then
  echo "  $DOPE_CMD mlc-v3 sample --wire $WEIGHTS_DIR/packet_high_regression.dml3 -n 10 --seed 42"
fi
}

main "$@"
