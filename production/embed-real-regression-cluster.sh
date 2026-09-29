#!/usr/bin/env bash
# shellcheck disable=SC2029 # Remote commands intentionally use locally resolved absolute paths.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${DOPE_BINARY:-$repo_root/target/release/dope-kernel}"
dataset_root="${DOPE_DATASET_ROOT:-/home/ubuntu/remote_super/quant/regression}"
router_dir="${DOPE_ROUTER_DIR:-$repo_root/runs/full-rust-metrics-20260718/router/attention-small}"
method="${DOPE_EMBEDDING_METHOD:-attention-small}"
expected_dimension="${DOPE_EMBEDDING_DIMENSION:-4168}"
output_root="${DOPE_EMBEDDING_OUTPUT:-/home/ubuntu/remote_super/quant/regression-embeddings/${method}_${expected_dimension}}"
jobs="${DOPE_EMBEDDING_JOBS:-32}"
hosts=(xbabe1 xbabe2 xbabe3)
expected_count_total=2326
expected_zero_feature_datasets=7
expected_promotion_failures='["best_fixed_candidate","paired_hypervolume","profile_regret","router_inference_p95","top_four_oracle_recall"]'

if [[ ! -x "$binary" ]]; then
    echo "release binary is absent or not executable: $binary" >&2
    exit 1
fi
if [[ ! -f "$router_dir/router.bundle.json" || ! -f "$router_dir/router-evidence.json" ]]; then
    echo "router bundle/evidence pair is incomplete: $router_dir" >&2
    exit 1
fi

# This is intentionally the first substantive operation. Failed integrity checks
# exits before remote directories are created, artifacts are copied, or jobs run.
qualification="$($binary qualify-embedding-router \
    --router-bundle "$router_dir/router.bundle.json" \
    --router-evidence "$router_dir/router-evidence.json")"
actual_dimension="$(jq -er '.dimension' <<<"$qualification")"
if [[ "$actual_dimension" != "$expected_dimension" ]]; then
    echo "qualified router dimension is $actual_dimension, expected $expected_dimension" >&2
    exit 1
fi
if [[ "$(jq -er '.embedding_eligible' <<<"$qualification")" != "true" ]]; then
    echo "router is not eligible for embedding" >&2
    exit 1
fi
if [[ "$(jq -er '.promotion_qualified' <<<"$qualification")" != "false" ]]; then
    echo "attention-small promotion status unexpectedly changed" >&2
    exit 1
fi
actual_promotion_failures="$(jq -c '.promotion_failed_gates | sort' <<<"$qualification")"
if [[ "$actual_promotion_failures" != "$expected_promotion_failures" ]]; then
    echo "attention-small promotion failures differ from measured evidence" >&2
    exit 1
fi
normalized_qualification="$(jq -cS . <<<"$qualification")"

stage_name=".embedding-worker-${method}_${expected_dimension}"
remote_stage="$output_root/$stage_name"
remote_binary="$remote_stage/dope-kernel"
remote_bundle="$remote_stage/router.bundle.json"
remote_evidence="$remote_stage/router-evidence.json"
mkdir -p "$output_root/manifests"

smoke_root="$(mktemp -d)"
cleanup_smoke() {
    rm -rf -- "$smoke_root"
}
trap cleanup_smoke EXIT

smoke_dataset() {
    local dataset_id="$1"
    local expected_features="$2"
    local corpus="$smoke_root/$dataset_id/corpus"
    local dataset="$corpus/$dataset_id"
    local manifest="$smoke_root/$dataset_id/manifest.json"
    local json_output="$output_root/${dataset_id}_${method}_${expected_dimension}.json"
    local vector_output="$output_root/${dataset_id}_${method}_${expected_dimension}.vs"
    mkdir -p "$dataset"
    cp "$dataset_root/$dataset_id/meta.json" "$dataset/meta.json"
    cp "$dataset_root/$dataset_id/train.csv" "$dataset/train.csv"
    "$binary" embed-regression-corpus \
        --dataset-root "$corpus" \
        --output-dir "$output_root" \
        --router-bundle "$router_dir/router.bundle.json" \
        --router-evidence "$router_dir/router-evidence.json" \
        --method "$method" \
        --expected-dimension "$expected_dimension" \
        --shard-index 0 \
        --shard-count 1 \
        --jobs 1 \
        --determinism-checks 1 \
        --manifest-out "$manifest" >/dev/null
    if [[ "$(jq -er '.success' "$manifest")" != "1" || \
          "$(jq -er '.determinism_checked' "$manifest")" != "1" || \
          "$(jq -er '.records[0].features' "$manifest")" != "$expected_features" ]]; then
        echo "embedding smoke test failed for $dataset_id" >&2
        exit 1
    fi
    if [[ ! -f "$json_output" || ! -f "$vector_output" || \
          "$(jq -er '.features' "$json_output")" != "$expected_features" ]]; then
        echo "embedding smoke outputs are invalid for $dataset_id" >&2
        exit 1
    fi
}

# The first real encode is repeated internally and must be byte-identical.
smoke_dataset "6051037b0820049d" 1
# Exercise the sketch's native empty-feature distributions; no sentinel is added.
smoke_dataset "72235646c52be632" 0

copy_if_changed() {
    local host="$1"
    local source="$2"
    local destination="$3"
    local expected_hash remote_hash
    expected_hash="$(sha256sum "$source" | cut -d ' ' -f 1)"
    remote_hash="$(ssh "$host" "sha256sum '$destination' 2>/dev/null | cut -d ' ' -f 1" || true)"
    if [[ "$remote_hash" != "$expected_hash" ]]; then
        scp -q "$source" "$host:$destination"
    fi
}

reference_inventory_hash=""
expected_count=""
for host in "${hosts[@]}"; do
    ssh "$host" "mkdir -p '$remote_stage'"
    copy_if_changed "$host" "$binary" "$remote_binary"
    copy_if_changed "$host" "$router_dir/router.bundle.json" "$remote_bundle"
    copy_if_changed "$host" "$router_dir/router-evidence.json" "$remote_evidence"
    ssh "$host" "chmod 755 '$remote_binary'"

    remote_qualification="$(ssh "$host" "'$remote_binary' qualify-embedding-router --router-bundle '$remote_bundle' --router-evidence '$remote_evidence'")"
    if [[ "$(jq -cS . <<<"$remote_qualification")" != "$normalized_qualification" ]]; then
        echo "router qualification differs on $host" >&2
        exit 1
    fi
    inventory="$(ssh "$host" "'$remote_binary' inspect-regression-corpus --dataset-root '$dataset_root'")"
    if [[ "$(jq -er '.schema_mismatches' <<<"$inventory")" != "0" ]]; then
        echo "regression corpus has schema mismatches on $host" >&2
        exit 1
    fi
    inventory_hash="$(jq -er '.dataset_ids_sha256' <<<"$inventory")"
    inventory_count="$(jq -er '.discovered' <<<"$inventory")"
    if [[ -z "$reference_inventory_hash" ]]; then
        reference_inventory_hash="$inventory_hash"
        expected_count="$inventory_count"
    elif [[ "$inventory_hash" != "$reference_inventory_hash" || "$inventory_count" != "$expected_count" ]]; then
        echo "regression corpus inventory differs on $host" >&2
        exit 1
    fi
done
if [[ "$expected_count" != "$expected_count_total" ]]; then
    echo "found $expected_count regression datasets, expected $expected_count_total" >&2
    exit 1
fi

pids=()
for index in "${!hosts[@]}"; do
    host="${hosts[$index]}"
    remote_output="$output_root/.shards/$host"
    remote_manifest="$remote_stage/$host.manifest.json"
    ssh "$host" "mkdir -p '$remote_output' && '$remote_binary' embed-regression-corpus \
        --dataset-root '$dataset_root' \
        --output-dir '$remote_output' \
        --router-bundle '$remote_bundle' \
        --router-evidence '$remote_evidence' \
        --method '$method' \
        --expected-dimension '$expected_dimension' \
        --shard-index '$index' \
        --shard-count '${#hosts[@]}' \
        --jobs '$jobs' \
        --determinism-checks 1 \
        --manifest-out '$remote_manifest'" \
        >"$output_root/manifests/$host.log" 2>&1 &
    pids+=("$!")
done

worker_failed=0
for index in "${!hosts[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "embedding worker failed on ${hosts[$index]}; see $output_root/manifests/${hosts[$index]}.log" >&2
        worker_failed=1
    fi
done
if [[ "$worker_failed" != "0" ]]; then
    exit 1
fi

for host in "${hosts[@]}"; do
    remote_output="$output_root/.shards/$host"
    remote_manifest="$remote_stage/$host.manifest.json"
    scp -q "$host:$remote_manifest" "$output_root/manifests/$host.json"
    rsync -a "$host:$remote_output/" "$output_root/"
done

manifest_files=("$output_root"/manifests/xbabe{1,2,3}.json)
manifest_hashes="$(jq -r '.dataset_ids_sha256' "${manifest_files[@]}" | sort -u | wc -l)"
assigned_total="$(jq -s 'map(.assigned) | add' "${manifest_files[@]}")"
success_total="$(jq -s 'map(.success) | add' "${manifest_files[@]}")"
failure_total="$(jq -s 'map(.failed + .schema_mismatch) | add' "${manifest_files[@]}")"
if [[ "$manifest_hashes" != "1" || "$assigned_total" != "$expected_count" || "$success_total" != "$expected_count" || "$failure_total" != "0" ]]; then
    echo "shard manifests do not cover the complete regression inventory" >&2
    exit 1
fi
expected_shard_assignments=(776 775 775)
for index in "${!manifest_files[@]}"; do
    if [[ "$(jq -er '.assigned' "${manifest_files[$index]}")" != "${expected_shard_assignments[$index]}" ]]; then
        echo "unexpected shard size for ${hosts[$index]}" >&2
        exit 1
    fi
    if [[ "$(jq -cS '.router' "${manifest_files[$index]}")" != "$normalized_qualification" ]]; then
        echo "router qualification was not preserved in ${hosts[$index]} manifest" >&2
        exit 1
    fi
done
zero_feature_total="$(jq -s '[.[] | .records[] | select(.status == "success" and .features == 0)] | length' "${manifest_files[@]}")"
if [[ "$zero_feature_total" != "$expected_zero_feature_datasets" ]]; then
    echo "found $zero_feature_total zero-feature datasets, expected $expected_zero_feature_datasets" >&2
    exit 1
fi

json_count="$(find "$output_root" -maxdepth 1 -type f -name "*_${method}_${expected_dimension}.json" | wc -l)"
vector_count="$(find "$output_root" -maxdepth 1 -type f -name "*_${method}_${expected_dimension}.vs" | wc -l)"
if [[ "$json_count" != "$expected_count" || "$vector_count" != "$expected_count" ]]; then
    echo "consolidated embedding file counts do not match the corpus" >&2
    exit 1
fi

verification="$($binary verify-regression-embeddings \
    --output-dir "$output_root" \
    --router-bundle "$router_dir/router.bundle.json" \
    --router-evidence "$router_dir/router-evidence.json" \
    --method "$method" \
    --expected-dimension "$expected_dimension" \
    --expected-count "$expected_count")"
printf '%s\n' "$verification" >"$output_root/verification.json"
if [[ "$(jq -er '.datasets' <<<"$verification")" != "$expected_count" || \
      "$(jq -er '.zero_feature_datasets' <<<"$verification")" != "$expected_zero_feature_datasets" || \
      "$(jq -er '.embedding_eligible' <<<"$verification")" != "true" || \
      "$(jq -er '.promotion_qualified' <<<"$verification")" != "false" || \
      "$(jq -c '.promotion_failed_gates | sort' <<<"$verification")" != "$expected_promotion_failures" ]]; then
    echo "consolidated verification does not preserve embedding and promotion status" >&2
    exit 1
fi

manifest_tmp="$output_root/.manifest.json.tmp-$$"
jq -sS -c \
    --arg output "$output_root" \
    --arg method "$method" \
    --argjson dimension "$expected_dimension" \
    '{
        format: "dope-regression-embedding-cluster-manifest",
        version: 2,
        method: $method,
        dimension: $dimension,
        dataset_ids_sha256: .[0].dataset_ids_sha256,
        discovered: .[0].discovered,
        attempted: (map(.assigned) | add),
        success: (map(.success) | add),
        schema_mismatch: (map(.schema_mismatch) | add),
        failed: (map(.failed) | add),
        determinism_checked: (map(.determinism_checked) | add),
        router: .[0].router,
        records: ([.[].records[] | if .status == "success" then
            .json_output = ($output + "/" + .dataset_id + "_" + $method + "_" + ($dimension | tostring) + ".json") |
            .vector_output = ($output + "/" + .dataset_id + "_" + $method + "_" + ($dimension | tostring) + ".vs")
        else . end] | sort_by(.global_index))
    }' "${manifest_files[@]}" >"$manifest_tmp"
mv "$manifest_tmp" "$output_root/manifest.json"

echo "embedded and verified $expected_count regression datasets in $output_root"
