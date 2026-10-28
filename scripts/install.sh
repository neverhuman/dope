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
TAG="${DOPE_RELEASE_TAG:-dope-engine-v1.1.0}"
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
COLOR_MODE="${DOPE_COLOR:-auto}"

usage() {
  cat <<'EOF'
Usage: install.sh [OPTIONS]

Options:
  --tag <TAG>               Engine/binary release tag (default: dope-engine-v1.1.0)
  --model-tag <TAG>         Independent model release (engine tag default: v1.0.0; otherwise same as --tag)
  --model-from-dir <DIR>    Independent offline model release, including its SHA256SUMS
  --source-digest <SHA40>   Expected 40-char git commit SHA for OIDC attestation verification
  --require-attestation     Require cryptographic Sigstore OIDC attestation (mandatory for dope-engine-v* tags)
  --from-dir <DIR>          Offline engine directory; never falls back to an HTTPS model download
  --install-dir <DIR>       Destination directory for the `dope` binary (default: ~/.local/bin)
  --weights-dir <DIR>       Destination directory for verified .safetensors weights (default: ~/.cache/dope/weights)
  --share-dir <DIR>         Destination directory for LICENSE and THIRD_PARTY_NOTICES (default: ~/.local/share/dope-engine)
  --no-weights              Install only the standalone `dope` binary and legal notices (skip .safetensors weights)
  --color <auto|always|never>
  -h, --help                Show this help

Signed engine tags require gh >= 2.102.0 and all attestation policy flags before fetching.
The engine tag selects the binary; --model-tag selects models independently:
  ./scripts/install.sh --tag dope-engine-v1.1.0 --model-tag v1.0.0
  ./scripts/install.sh --tag dope-engine-v1.1.0 --no-weights
For an offline split, supply --from-dir ENGINE_DIR --model-from-dir MODEL_DIR.
A one-directory offline release with no model members remains engine-only unless
models were explicitly requested. Partial models are refused, never relabeled.
EOF
}

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
    --no-weights) SKIP_WEIGHTS=1; shift ;;
    --color) COLOR_MODE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

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

# 2. Platform & tool check.
if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "error: prebuilt binary is qualified for Linux x86_64 only (detected $(uname -s):$(uname -m))." >&2
  echo "  hint: build from source with: cargo build --manifest-path engine/Cargo.toml --locked --release --bin dope" >&2
  exit 1
fi

for tool in sha256sum jq awk install mktemp; do
  command -v "$tool" >/dev/null 2>&1 || { echo "error: required tool missing: $tool" >&2; exit 1; }
done

# Fail before any asset fetch or destination creation. Older gh releases lack policy
# flags or the patched attestation enforcement; flag presence alone is insufficient.
gh_upgrade_error() {
  echo "error: upgrade gh to 2.102.0 or newer with attestation verify --repo, --signer-workflow, --source-ref, --source-digest and --format support" >&2
  echo "  https://github.com/cli/cli/releases/tag/v2.102.0" >&2
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
  for flag in --repo --signer-workflow --source-ref --source-digest --format; do
    [[ "$gh_help" =~ (^|[[:space:]])${flag}([[:space:]=,]|$) ]] || gh_upgrade_error
  done
}
if [[ $REQUIRE_ATTESTATION -eq 1 ]]; then require_gh_policy; fi

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/dope-install.XXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT
ENGINE_STAGE="$STAGE/engine"
MODEL_STAGE="$STAGE/models"
mkdir -- "$ENGINE_STAGE" "$MODEL_STAGE"

echo "$(c_cyan "● DOPE Installer") · concrete release tag $(c_bold "$TAG") · repo $(c_bold "$REPO_SLUG")"
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
    if ! curl -fsSL --proto '=https' --tlsv1.2 --retry 2 -o "$dst" "$url"; then
      echo "error: failed to download required asset: $url" >&2
      exit 1
    fi
  fi
}
for name in "${REQUIRED_CORE[@]}"; do fetch_asset "$ENGINE_STAGE" "$FROM_DIR" "$TAG" "$name"; done

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

# Cross-check BUILDINFO.json binary_sha256 against dope-linux-x86_64.
BIN_SHA=$(sha256sum "$ENGINE_STAGE/dope-linux-x86_64" | awk '{print $1}')
BUILDINFO_BIN_SHA=$(jq -r '.binary_sha256 // empty' "$ENGINE_STAGE/BUILDINFO.json")
if [[ -z "$BUILDINFO_BIN_SHA" || "$BIN_SHA" != "$BUILDINFO_BIN_SHA" ]]; then
  echo "error: BUILDINFO.json binary_sha256 ($BUILDINFO_BIN_SHA) does not match dope-linux-x86_64 ($BIN_SHA)" >&2
  exit 1
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
    echo "error: could not resolve 40-char source commit SHA for tag $TAG; pass --source-digest <SHA40>" >&2
    exit 1
  fi
  if [[ "$(jq -r '.source_commit // empty' "$ENGINE_STAGE/BUILDINFO.json")" != "$SOURCE_COMMIT" ]]; then
    echo "error: BUILDINFO.json source_commit differs from expected engine source digest" >&2
    exit 1
  fi
  for subject in dope-linux-x86_64 SHA256SUMS; do
    gh attestation verify "$ENGINE_STAGE/$subject" --repo "$REPO_SLUG" \
      --signer-workflow "$SIGNER_WORKFLOW" \
      --source-ref "refs/tags/$TAG" \
      --source-digest "$SOURCE_COMMIT" \
      --format json >/dev/null
  done
  ATTESTATION_VERIFIED=1
  VERIFICATION_LEVEL="sigstore-oidc-attestation + sha256-content-integrity (verified against $SIGNER_WORKFLOW @ refs/tags/$TAG [$SOURCE_COMMIT])"

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

# 6. Atomic installation (refusing symlinks at destination paths).
atomic_install() {
  local src="$1"
  local dst_dir="$2"
  local dst_name="$3"
  local mode="$4"
  mkdir -p -- "$dst_dir"
  local dst="$dst_dir/$dst_name"
  if [[ -L "$dst" ]]; then
    echo "error: refusing to overwrite symlink at $dst" >&2
    exit 1
  fi
  local tmp="$dst_dir/.${dst_name}.tmp.$$"
  rm -f -- "$tmp"
  install -m "$mode" "$src" "$tmp"
  mv -f -- "$tmp" "$dst"
}

atomic_install "$ENGINE_STAGE/dope-linux-x86_64" "$INSTALL_DIR" "dope" 0755
atomic_install "$ENGINE_STAGE/LICENSE" "$SHARE_DIR" "LICENSE" 0644
atomic_install "$ENGINE_STAGE/THIRD_PARTY_NOTICES" "$SHARE_DIR" "THIRD_PARTY_NOTICES" 0644
atomic_install "$ENGINE_STAGE/BUILDINFO.json" "$SHARE_DIR" "BUILDINFO.json" 0644

if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then
  atomic_install "$MODEL_STAGE/MODEL_MANIFEST.json" "$WEIGHTS_DIR" "MODEL_MANIFEST.json" 0644
  atomic_install "$MODEL_STAGE/dope_mlc_v3_multitask_best.safetensors" "$WEIGHTS_DIR" "dope_mlc_v3_multitask_best.safetensors" 0644
  atomic_install "$MODEL_STAGE/packet_high_regression.dml3" "$WEIGHTS_DIR" "packet_high_regression.dml3" 0644
fi

echo "$(c_green "✓ Installed") $(c_bold "$INSTALL_DIR/dope") (sha256: ${BIN_SHA:0:16}…)"
echo "  $(c_dim "notices")       $SHARE_DIR/{LICENSE,THIRD_PARTY_NOTICES,BUILDINFO.json}"
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
echo "$(c_bold "Quick Start:")"
echo "  $INSTALL_DIR/dope doctor"
if [[ $HAVE_MODEL_ASSETS -eq 1 ]]; then
  echo "  $INSTALL_DIR/dope mlc-v3 sample --wire $WEIGHTS_DIR/packet_high_regression.dml3 -n 10 --seed 42"
fi
