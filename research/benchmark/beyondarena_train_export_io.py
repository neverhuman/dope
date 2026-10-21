"""Trusted courier CLI: pinned mixed Parquet -> private official TRAIN only.

Parquet row groups can physically decode outer-test values inside this exporter.
They are never returned to an optimizer. This is NOT independent sealed custody.
No preprocessing is fitted here; inner split identities precede later fitting.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
import time

from research.benchmark.beyondarena_train_export import (
    ExportRefusal, authorize, canonical, inner_partition, ordered_time,
    require, scalar, sha,
)


def producer_files():
    root = Path(__file__).parent
    names = ('beyondarena_train_export.py', 'beyondarena_train_export_io.py')
    return {name: sha((root / name).read_bytes()) for name in names}


def prepare_table(table, source, task, train, test):
    """Only role keys from outer TEST are inspected after physical decoding."""
    names = table.column_names
    target = task['target_column_name']
    require(len(names) == source['columns'] and len(set(names)) == len(names)
            and names.count(target) == 1 and table.num_rows == source['rows'],
            'decoded_schema_mismatch')
    group, clock = task['group_on'], task['time_on']
    require(all(role is None or names.count(role) == 1 for role in (group, clock)),
            'decoded_role_column_missing')
    features = [name for name in names if name not in {target, group, clock}]
    require(bool(features), 'no_predictor_columns')
    # Take TRAIN before reading target or feature values into Python objects.
    selected = table.take(train)
    target_values = selected[target].to_pylist()
    require(all(type(v) in (int, float) and math.isfinite(v) for v in target_values),
            'nonfinite_or_nonnumeric_TRAIN_target')
    if group:
        values = table[group].to_pylist()
        require(all(values[i] is not None and not
                    (type(values[i]) is float and not math.isfinite(values[i])) for i in train + test), 'missing_group_identity')
        keys = [sha(canonical(scalar(values[i]))) for i in train]
        forbidden = {sha(canonical(scalar(values[i]))) for i in test}
        require(not set(keys).intersection(forbidden), 'outer_group_overlap')
    elif clock:
        values = table[clock].to_pylist()
        keys = [ordered_time(values[i]) for i in train]
        forbidden = [ordered_time(values[i]) for i in test]
        require(max(keys) < min(forbidden), 'outer_time_order_invalid')
    else:
        # Feature-identical aliases stay together; targets never choose a split.
        columns = [selected[name].to_pylist() for name in features]
        keys = [sha(canonical([scalar(value) for value in row])) for row in zip(*columns)]
    partitions = inner_partition(keys, temporal=clock is not None)
    canonical_table = selected.select([*features, target])
    membership = {
        'official_TRAIN_original_row_indices': train,
        'original_row_ids': [sha(canonical([source['files']['dataset.parquet']['sha256'], i]))
                             for i in train],
        'group_or_time_keys': keys,
        'inner_partitions_in_export_order': partitions,
        'partition_rule': 'ordered_time_blocks_80_20' if clock else 'seed1729_identity_blocks_80_20',
        'source_target_index': names.index(target),
        'source_feature_indices': [names.index(name) for name in features],
        'removed_role_columns': [name for name in (group, clock) if name is not None],
        'target': target, 'target_index': len(features),
    }
    return canonical_table, membership


def export(grant_raw, grant_sha256, source_raw, task_raw, split_raw, *, get_object, now=None):
    """Caller supplies a pinned external grant and exact versioned S3 transport."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    now = time.time() if now is None else now
    grant, source, task, train, test = authorize(
        grant_raw, grant_sha256, source_raw, task_raw, split_raw, uid=os.getuid(), now=now)
    require(grant['producer_files'] == producer_files(), 'producer_source_mismatch')
    output = Path(grant['output_dir'])
    require(output.is_absolute() and output == output.resolve()
            and not any(part.is_symlink() for part in (output, *output.parents))
            and not output.exists(), 'output_path_not_fresh_private')
    ref = grant['raw_object']
    response = get_object(Bucket=grant['bucket'], Key=ref['key'], VersionId=ref['version_id'])
    stream = response['Body']
    try:
        require(response.get('VersionId') == ref['version_id']
                and response.get('ContentLength') == ref['bytes'], 'transport_identity_mismatch')
        raw = stream.read(ref['bytes'] + 1)
    finally:
        stream.close()
    require(len(raw) == ref['bytes'] and sha(raw) == ref['sha256'], 'raw_source_hash_mismatch')
    require(time.time() < grant['expires_unix'], 'export_deadline_exceeded')
    table = pq.read_table(pa.BufferReader(raw))
    selected, membership = prepare_table(table, source, task, train, test)
    del table, raw  # Mixed source is never a materialized optimizer input.
    require(time.time() < grant['expires_unix'], 'export_deadline_exceeded')
    output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    parquet = output / 'official-train.parquet'
    # Open exclusively with private permissions; no credentials or raw input copied.
    with parquet.open('xb') as handle:
        os.chmod(parquet, 0o600)
        pq.write_table(selected, handle)
        handle.flush()
        os.fsync(handle.fileno())
    member_path = output / 'membership.private.json'
    with member_path.open('xb') as handle:
        os.chmod(member_path, 0o600)
        handle.write(canonical(membership) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        'format': 'dope-beyondarena-official-train-export-v1',
        'status': 'TRAIN_EXPORT_READY_NOT_MODEL_ADMISSION',
        'grant_sha256': grant_sha256, 'producer_files': producer_files(),
        'source_metadata_sha256': sha(source_raw), 'task_metadata_sha256': sha(task_raw),
        'split_metadata_sha256': sha(split_raw), 'source_family_id': source['source_family_id'],
        'source_revision': source['revision'], 'view': grant['view'],
        'raw_original': {'bucket': source['bucket'], **ref},
        'raw_original_preserved': 'immutable versioned S3 object, not rewritten or copied to optimizer',
        'license_recorded_verbatim': source['license'],
        'source_metric': source['objective_metric'],
        'metric_policy': 'no metric computed; original-unit MAPE zero convention still needs an explicit caller binding',
        'rows': selected.num_rows, 'columns': selected.column_names,
        'target': source['target'], 'target_index': selected.num_columns - 1,
        'train_membership_sha256': grant['train_membership_sha256'],
        'outer_test_membership_sha256': grant['outer_test_membership_sha256'],
        'files': {path.name: {'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size}
                  for path in (parquet, member_path)},
        'mixed_parquet_physically_decoded_by_trusted_exporter': True,
        'outer_test_values_exported_to_optimizer': False,
        'outer_group_or_time_roles_checked': bool(task['group_on'] or task['time_on']),
        'preprocessing_fitted': False, 'models_fit': 0,
        'independent_sealed_custody_proven': False,
        'consumer_requirement': 'mount only this dedicated export; never grant raw S3/source access to optimizer',
    }
    require(time.time() < grant['expires_unix'], 'export_deadline_exceeded')
    with (output / 'manifest.json').open('xb') as handle:
        os.chmod(output / 'manifest.json', 0o600)
        handle.write(canonical(manifest) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('grant', 'source', 'task', 'split'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--grant-sha256', required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    try:
        raw = args.grant.read_bytes()
        grant, *_ = authorize(raw, args.grant_sha256, args.source.read_bytes(),
                              args.task.read_bytes(), args.split.read_bytes(),
                              uid=os.getuid(), now=time.time())
        require(args.profile == grant['profile'], 'explicit_profile_mismatch')
        import resource
        import signal
        resource.setrlimit(resource.RLIMIT_AS, (16 * 2**30, 16 * 2**30))
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        signal.alarm(min(600, max(1, int(grant['expires_unix'] - time.time()))))
        import boto3
        from botocore.config import Config
        client = boto3.Session(profile_name=args.profile).client(
            's3', config=Config(retries={'total_max_attempts': 1}, connect_timeout=10, read_timeout=60))
        export(raw, args.grant_sha256, args.source.read_bytes(), args.task.read_bytes(),
               args.split.read_bytes(), get_object=client.get_object)
        print(json.dumps({'status': 'TRAIN_EXPORT_READY_NOT_MODEL_ADMISSION'}))
        return 0
    except ExportRefusal as error:
        print(json.dumps({'status': 'REFUSED', 'code': str(error)}))
    return 1


def terminal_error_hook(_kind, _error, _traceback):
    """Uncaught CLI errors terminate normally with exit1 and no source diagnostics.

    This hook does not catch, retry, resume or convert a failed export to success.
    Keep third-party exception values and tracebacks out of public stderr.
    """
    print(json.dumps({'status': 'REFUSED', 'code': 'unhandled_export_error'}))


if __name__ == '__main__':
    sys.excepthook = terminal_error_hook
    raise SystemExit(main())
