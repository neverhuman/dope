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
TAG="${DOPE_RELEASE_TAG:-dope-engine-v3.0.0}"
SOURCE_COMMIT="${DOPE_SOURCE_COMMIT:-}"
FROM_DIR=""
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
  --install-dir <DIR>       Destination directory for the `dope` binary (default: ~/.local/bin)
  --weights-dir <DIR>       Destination directory for verified .safetensors weights (default: ~/.cache/dope/weights)
  --share-dir <DIR>         Destination directory for LICENSE and THIRD_PARTY_NOTICES (default: ~/.local/share/dope-engine)
  --v8-weights              Require and install attested SingleEncV8 weights from the engine release
  --no-v8-weights           Skip SingleEncV8 weights even if bundled with the engine release
  --no-weights              Install only the standalone `dope` binary and legal notices (skip .safetensors weights)
  --uninstall               Remove only files this installer owns, then exit
  --color <auto|always|never>
  -h, --help                Show this help

Signed engine tags install the standalone binary. They require gh >= 2.102.0 and
verify the published dope.sigstore.json bundle, so a GitHub login is not required.
Pass --with-weights to also install the independent model release.
  ./scripts/install.sh --tag dope-engine-v3.0.0 --source-digest <SHA40>
  ./scripts/install.sh --tag dope-engine-v3.0.0 --with-weights --model-tag v1.0.0
  ./scripts/install.sh --uninstall
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
    --model-tag) MODEL_TAG="${2:-}"; [[ -n "$MODEL_TAG" ]] || { echo "error: --model-tag requires a concrete tag" >&2; exit 2; }; MODEL_EXPLICIT=1; shift 2 ;;
    --model-from-dir) MODEL_FROM_DIR="${2:-}"; [[ -n "$MODEL_FROM_DIR" ]] || { echo "error: --model-from-dir requires a nonempty directory" >&2; exit 2; }; MODEL_EXPLICIT=1; shift 2 ;;
    --install-dir) INSTALL_DIR="${2:-}"; shift 2 ;;
    --prefix) INSTALL_DIR="${2:-}/bin"; SHARE_DIR="${2:-}/share/dope-engine"; shift 2 ;;
    --weights-dir|--cache-dir) WEIGHTS_DIR="${2:-}"; shift 2 ;;
    --share-dir) SHARE_DIR="${2:-}"; shift 2 ;;
    --with-weights|--verify-sample) SKIP_WEIGHTS=0; MODEL_EXPLICIT=1; shift ;;
    --v8-weights) V8_WEIGHTS_MODE=require; V8_WEIGHTS_EXPLICIT=1; shift ;;
    --no-v8-weights) V8_WEIGHTS_MODE=skip; V8_WEIGHTS_EXPLICIT=1; shift ;;
    --no-weights) SKIP_WEIGHTS=1; [[ $V8_WEIGHTS_EXPLICIT -eq 1 ]] || V8_WEIGHTS_MODE=skip; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    --color) COLOR_MODE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Remove owned files before platform, gh, or glibc checks so a host that cannot
# install can still uninstall. Refuse symlinks and non-regular paths.
uninstall_dope() {
  local path
  local owned=(
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
  for path in "${owned[@]}"; do
    if [[ -L "$path" ]]; then
      echo "error: refusing to remove symlink: $path" >&2
      exit 1
    fi
    if [[ -e "$path" && ! -f "$path" ]]; then
      echo "error: refusing to remove non-regular file: $path" >&2
      exit 1
    fi
  done
  for path in "${owned[@]}"; do
    if [[ -f "$path" ]]; then
      rm -f -- "$path"
    fi
  done
  echo "uninstalled owned dope files"
  exit 0
}
if [[ $UNINSTALL -eq 1 ]]; then
  uninstall_dope
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
if [[ "$TAG" == dope-engine-v* ]]; then REQUIRE_ATTESTATION=1; fi
# Analytic engine tags do not download model files unless a model flag asks for them.
if [[ "$TAG" == dope-engine-v* && $MODEL_EXPLICIT -eq 0 ]]; then SKIP_WEIGHTS=1; fi
if [[ "$V8_WEIGHTS_MODE" == "require" && ( "$TAG" != dope-engine-v* || $REQUIRE_ATTESTATION -ne 1 ) ]]; then
  echo "error: --v8-weights requires an attested dope-engine-v* release tag (got '$TAG')" >&2
  exit 1
fi

# 2. Platform & tool check.
if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "error: prebuilt binary is qualified for Linux x86_64 only (detected $(uname -s):$(uname -m))." >&2
  echo "  hint: build from source with: cargo build --manifest-path engine/Cargo.toml --locked --release --bin dope" >&2
  exit 1
fi

# Fail before any asset fetch or destination creation. Older gh releases lack policy
# flags or the patched attestation enforcement; flag presence alone is insufficient.
gh_upgrade_error() {
  echo "error: upgrade gh to 2.102.0 or newer with attestation verify --repo, --signer-workflow, --source-ref, --source-digest, --bundle and --format support" >&2
  echo "  https://github.com/cli/cli/releases/tag/v2.102.0" >&2
  echo "  note: Ubuntu 24.04's packaged gh is 2.45.0; use the release above or GitHub's apt repository" >&2
  exit 1
}
require_gh_policy() {
  local gh_version gh_help major minor flag
  command -v gh >/dev/null 2>&1 || gh_upgrade_error
  gh_version=$(gh --version 2>/dev/null) || gh_upgrade_error
  [[ "$gh_version" =~ ^gh[[:space:]]version[[:space:]]([0-9]+)\.([0-9]+)\.([0-9]+)([[:space:]]|$) ]] || gh_upgrade_error
  major=$((10#${BASH_REMATCH[1]})); minor=$((10#${BASH_REMATCH[2]}))
  (( major > 2 || (major == 2 && minor >= 102) )) || gh_upgrade_error
  gh_help=$(gh attestation verify --help 2>/dev/null) || gh_upgrade_error
  for flag in --repo --signer-workflow --source-ref --source-digest --format --bundle; do
    [[ "$gh_help" =~ (^|[[:space:]])${flag}([[:space:]=,]|$) ]] || gh_upgrade_error
  done
}
preflight() {
  local missing=() tool glibc
  for tool in sha256sum jq awk install mktemp sort getconf; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  if [[ -z "$FROM_DIR" ]]; then
    command -v curl >/dev/null 2>&1 || missing+=(curl)
  fi
  if (( ${#missing[@]} )); then
    echo "error: required tools missing: ${missing[*]}" >&2
    echo "  fix (Debian/Ubuntu): sudo apt-get install -y curl jq coreutils" >&2
    exit 1
  fi
  if [[ "${DOPE_SKIP_GLIBC_CHECK:-0}" != 1 ]]; then
    glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
    if [[ -z "$glibc" ]] && command -v ldd >/dev/null 2>&1; then
      glibc=$(ldd --version 2>/dev/null | awk 'NR==1 { print $NF }')
    fi
    if [[ -z "$glibc" || "$(printf '%s\n' "2.39" "$glibc" | sort -V | head -n 1)" != "2.39" ]]; then
      echo "error: dope-linux-x86_64 needs glibc 2.39 or newer (found: ${glibc:-no glibc})." >&2
      echo "  fix: use Ubuntu 24.04+, Debian 13+ or a container such as ubuntu:24.04, or build from source:" >&2
      echo "       cargo build --manifest-path engine/Cargo.toml --locked --release --bin dope" >&2
      exit 1
    fi
  fi
}
preflight
if [[ $REQUIRE_ATTESTATION -eq 1 ]]; then require_gh_policy; fi

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/dope-install.XXXXXX")
TMP_PATHS=()
cleanup_install() {
  rm -rf -- "$STAGE"
  local path
  if ((${#TMP_PATHS[@]})); then
    for path in "${TMP_PATHS[@]}"; do
      rm -f -- "$path"
    done
  fi
}
trap cleanup_install EXIT
ENGINE_STAGE="$STAGE/engine"
MODEL_STAGE="$STAGE/models"
mkdir -- "$ENGINE_STAGE" "$MODEL_STAGE"

echo "$(c_cyan "● DOPE Installer") · concrete release tag $(c_bold "$TAG") · repo $(c_bold "$REPO_SLUG")"
# Runtime members only. Cargo.lock and DEPENDENCIES.txt stay in the signed
# release inventory checked by scripts/verify-engine-release.sh. The attestation
# bundle dope.sigstore.json is fetched separately and is not a SHA256SUMS member.
REQUIRED_CORE=(BUILDINFO.json LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64 SHA256SUMS)
MODEL_ASSETS=(MODEL_MANIFEST.json dope_mlc_v3_multitask_best.safetensors packet_high_regression.dml3)

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
    command -v curl >/dev/null 2>&1 || { echo "error: curl is required for HTTPS download" >&2; exit 1; }
    local url="https://github.com/$REPO_SLUG/releases/download/$tag/$name"
    if ! curl -fsSL --proto '=https' --tlsv1.2 --retry 2 --connect-timeout 15 --max-time 600 -o "$dst" "$url"; then
      echo "error: failed to download required asset: $url" >&2
      exit 1
    fi
  fi
}
for name in "${REQUIRED_CORE[@]}"; do fetch_asset "$ENGINE_STAGE" "$FROM_DIR" "$TAG" "$name"; done
if [[ $REQUIRE_ATTESTATION -eq 1 ]]; then
  fetch_asset "$ENGINE_STAGE" "$FROM_DIR" "$TAG" "dope.sigstore.json"
fi

HAVE_MODEL_ASSETS=0
if [[ $SKIP_WEIGHTS -eq 0 ]]; then
  if [[ -n "$FROM_DIR" && -z "$MODEL_FROM_DIR" ]]; then
    local_models=0
    for name in "${MODEL_ASSETS[@]}"; do
      if [[ -e "$FROM_DIR/$name" || -L "$FROM_DIR/$name" ]]; then local_models=1; fi
    done
    if [[ $MODEL_EXPLICIT -eq 0 && $local_models -eq 0 ]]; then
      SKIP_WEIGHTS=1 # Preserve one-directory offline engine-only installation.
    elif [[ "$MODEL_TAG" != "$TAG" ]]; then
      echo "error: offline split model request requires --model-from-dir for model tag $MODEL_TAG" >&2
      exit 1
    else
      MODEL_FROM_DIR=$FROM_DIR
    fi
  fi
  if [[ $SKIP_WEIGHTS -eq 0 ]]; then
    for name in SHA256SUMS "${MODEL_ASSETS[@]}"; do fetch_asset "$MODEL_STAGE" "$MODEL_FROM_DIR" "$MODEL_TAG" "$name"; done
    HAVE_MODEL_ASSETS=1
  fi
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
validate_sha256sums "$ENGINE_STAGE"
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
  actual=$(sha256sum "$stage/$name" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "error: SHA-256 mismatch for $name: expected $expected, got $actual" >&2
    exit 1
  fi
}

for name in BUILDINFO.json LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64; do
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
    SOURCE_COMMIT=$(gh api "repos/$REPO_SLUG/commits/$TAG" --jq .sha 2>/dev/null || true)
  fi
  if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: pass --source-digest <SHA40> (shown in the README). Resolving it with gh api needs a GitHub login." >&2
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
  if [[ ! -f "$ENGINE_STAGE/dope.sigstore.json" || -L "$ENGINE_STAGE/dope.sigstore.json" ]]; then
    echo "error: attestation bundle missing: dope.sigstore.json" >&2
    echo "  Nothing was installed." >&2
    exit 1
  fi
  for subject in "${attest_subjects[@]}"; do
    if ! gh attestation verify "$ENGINE_STAGE/$subject" \
      --bundle "$ENGINE_STAGE/dope.sigstore.json" \
      --repo "$REPO_SLUG" \
      --signer-workflow "$SIGNER_WORKFLOW" \
      --source-ref "refs/tags/$TAG" \
      --source-digest "$SOURCE_COMMIT" \
      --format json >/dev/null; then
      echo "error: attestation verification failed for $subject (tag $TAG, commit $SOURCE_COMMIT)." >&2
      echo "  Nothing was installed. Either Sigstore's trust root could not be reached, or this file" >&2
      echo "  is not the one the release workflow signed. Do not work around this check." >&2
      exit 1
    fi
  done
  ATTESTATION_VERIFIED=1
  VERIFICATION_LEVEL="sigstore-oidc-attestation + sha256-content-integrity (verified against $SIGNER_WORKFLOW @ refs/tags/$TAG [$SOURCE_COMMIT])"

fi

if [[ $HAVE_V8_WEIGHTS -eq 1 ]]; then
  [[ $ATTESTATION_VERIFIED -eq 1 ]] || { echo "error: SingleEncV8 weights require verified Sigstore OIDC attestation" >&2; exit 1; }
  V8_DST_DIR="$WEIGHTS_DIR/single-enc-v8"
  if [[ -L "$WEIGHTS_DIR" || ( -e "$WEIGHTS_DIR" && ! -d "$WEIGHTS_DIR" ) || -L "$V8_DST_DIR" || ( -e "$V8_DST_DIR" && ! -d "$V8_DST_DIR" ) ]]; then
    echo "error: refusing to install SingleEncV8 weights through symlink or non-directory path" >&2
    exit 1
  fi
  for v8_name in manifest.json model.safetensors; do
    v8_dst="$V8_DST_DIR/$v8_name"
    if [[ -L "$v8_dst" || ( -e "$v8_dst" && ! -f "$v8_dst" ) ]]; then
      echo "error: refusing to install SingleEncV8 weights over symlink or non-regular path: $v8_dst" >&2
      exit 1
    fi
  done
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

# 6. Atomic installation (refusing symlinks and non-regular destination files).
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
  local tmp="$dst_dir/.${dst_name}.tmp.$$"
  TMP_PATHS+=("$tmp")
  rm -f -- "$tmp"
  install -m "$mode" "$src" "$tmp"
  mv -fT -- "$tmp" "$dst"
}

probe_dir_writable() {
  local dir="$1"
  local probe="$dir/.dope-write-probe.$$"
  if ! touch -- "$probe" 2>/dev/null; then
    echo "error: destination is not writable: $dir" >&2
    exit 1
  fi
  rm -f -- "$probe"
}

prepare_install_dirs() {
  local dir name dst
  for dir in "$INSTALL_DIR" "$SHARE_DIR"; do
    if [[ -L "$dir" || ( -e "$dir" && ! -d "$dir" ) ]]; then
      echo "error: refusing to install through symlink or non-directory path: $dir" >&2
      exit 1
    fi
    mkdir -p -- "$dir"
  done
  for dst in "$INSTALL_DIR/dope" "$SHARE_DIR/LICENSE" "$SHARE_DIR/THIRD_PARTY_NOTICES" "$SHARE_DIR/BUILDINFO.json"; do
    if [[ -L "$dst" || ( -e "$dst" && ! -f "$dst" ) ]]; then
      echo "error: refusing to overwrite symlink or non-regular destination at $dst" >&2
      exit 1
    fi
  done
  if [[ $HAVE_MODEL_ASSETS -eq 1 || $HAVE_V8_WEIGHTS -eq 1 ]]; then
    if [[ -L "$WEIGHTS_DIR" || ( -e "$WEIGHTS_DIR" && ! -d "$WEIGHTS_DIR" ) ]]; then
      echo "error: refusing to install through symlink or non-directory path: $WEIGHTS_DIR" >&2
      exit 1
    fi
    mkdir -p -- "$WEIGHTS_DIR"
  fi
  if [[ $HAVE_V8_WEIGHTS -eq 1 ]]; then
    if [[ -L "$V8_DST_DIR" || ( -e "$V8_DST_DIR" && ! -d "$V8_DST_DIR" ) ]]; then
      echo "error: refusing to install through symlink or non-directory path: $V8_DST_DIR" >&2
      exit 1
    fi
    mkdir -p -- "$V8_DST_DIR"
    for name in manifest.json model.safetensors; do
      dst="$V8_DST_DIR/$name"
      if [[ -L "$dst" || ( -e "$dst" && ! -f "$dst" ) ]]; then
        echo "error: refusing to overwrite symlink or non-regular destination at $dst" >&2
        exit 1
      fi
    done
  fi
  if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then
    for name in MODEL_MANIFEST.json dope_mlc_v3_multitask_best.safetensors packet_high_regression.dml3; do
      dst="$WEIGHTS_DIR/$name"
      if [[ -L "$dst" || ( -e "$dst" && ! -f "$dst" ) ]]; then
        echo "error: refusing to overwrite symlink or non-regular destination at $dst" >&2
        exit 1
      fi
    done
  fi
  probe_dir_writable "$INSTALL_DIR"
  probe_dir_writable "$SHARE_DIR"
  if [[ $HAVE_MODEL_ASSETS -eq 1 || $HAVE_V8_WEIGHTS -eq 1 ]]; then
    probe_dir_writable "$WEIGHTS_DIR"
  fi
  if [[ $HAVE_V8_WEIGHTS -eq 1 ]]; then
    probe_dir_writable "$V8_DST_DIR"
  fi
}
prepare_install_dirs

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
DOPE_CMD="$INSTALL_DIR/dope"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) DOPE_CMD=dope ;;
  *)
    echo "$(c_yellow "! $INSTALL_DIR is not on your PATH.") Run this now:"
    echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
    echo "    Add that line to ~/.bashrc yourself if you want it in new shells."
    ;;
esac
other=$(command -v dope 2>/dev/null || true)
if [[ -n "$other" && "$other" != "$INSTALL_DIR/dope" ]]; then
  echo "$(c_yellow "! Another dope is first on PATH:") $other   (this installer wrote $INSTALL_DIR/dope)"
fi
echo ""
c_bold "Next:"
echo
echo "  $DOPE_CMD --version"
echo "  $DOPE_CMD doctor"
echo "  $DOPE_CMD fit train.csv --target <column> --format auto-v8 --budget 2048 --seed 1 -o model.dope"
echo "  $DOPE_CMD sample model.dope -n 1000 --seed 1 -o synthetic.csv"
if [[ $HAVE_V8_WEIGHTS -eq 1 ]]; then
  echo "  $DOPE_CMD fit data.csv --target y --format auto-v8 --v8-weights $WEIGHTS_DIR/single-enc-v8/model.safetensors -o model.dope"
fi
if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then
  echo "  $DOPE_CMD mlc-v3 sample --wire $WEIGHTS_DIR/packet_high_regression.dml3 -n 10 --seed 42"
fi
}

main "$@"
