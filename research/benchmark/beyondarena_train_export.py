"""Authenticate one official TRAIN view before the trusted exporter opens data.

This is a data-courier boundary, never an optimizer or sealed-evaluator grant.
No values, files, model providers or credentials are opened by these helpers.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math


class ExportRefusal(ValueError):
    """A fixed diagnostic code; source values never enter the message."""


def require(condition, code):
    if not condition:
        raise ExportRefusal(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate_metadata_key')
        result[key] = value
    return result


def bound_json(raw, expected):
    require(type(raw) is bytes and type(expected) is str and sha(raw) == expected,
            'metadata_hash_mismatch')
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: require(False, 'nonfinite_metadata'))
    except (UnicodeError, json.JSONDecodeError):
        raise ExportRefusal('metadata_json_invalid') from None


def memberships(split, view, rows):
    require(view['split_pointer'] == '/splits/0/0'
            and view['repeat_id'] == view['fold_id'] == '0', 'unsupported_fold')
    try:
        pair = split['splits']['0']['0']
    except (KeyError, IndexError, TypeError):
        raise ExportRefusal('split_shape_invalid') from None
    require(type(pair) is list and len(pair) == 2, 'split_shape_invalid')
    train, test = pair
    require(type(rows) is int and rows > 1, 'source_row_count_invalid')
    for indices, count in [(train, view['train_rows']), (test, view['test_rows'])]:
        require(type(indices) is list and type(count) is int and count > 0
                and len(indices) == count and all(type(i) is int and 0 <= i < rows for i in indices)
                and len(set(indices)) == count, 'split_indices_invalid')
    require(not set(train).intersection(test) and len(train) + len(test) == rows,
            'split_not_disjoint_complete')
    return train, test


def authorize(grant_raw, grant_sha256, source_raw, task_raw, split_raw, *, uid, now):
    """External launcher pins the grant hash; this function does not issue it."""
    grant = bound_json(grant_raw, grant_sha256)
    require(grant.get('format') == 'dope-beyondarena-train-export-v1'
            and grant.get('role') == 'trusted_data_exporter'
            and grant.get('optimizer_may_read_mixed_source') is False
            and grant.get('execution_authorized') is True,
            'export_role_required')
    require(type(grant.get('exporter_uid')) is int and grant['exporter_uid'] == uid,
            'exporter_identity_mismatch')
    require(type(grant.get('expires_unix')) is int and now < grant['expires_unix']
            and type(grant.get('started_unix')) is int
            and grant['started_unix'] <= now
            and 0 < grant['expires_unix'] - grant['started_unix'] <= 3600,
            'export_window_invalid')
    source = bound_json(source_raw, grant['source_metadata_sha256'])
    task_ref = source['files']['task_metadata.predictive-ml-task-mold-v1.json']
    split_ref = source['files']['experiment_metadata.predictive-ml-splits-mold-v1.json']
    task = bound_json(task_raw, task_ref['sha256'])
    split = bound_json(split_raw, split_ref['sha256'])
    view = grant['view']
    require(source.get('schema') == 'jope.beyondarena-source/v1'
            and view.get('schema') == 'jope.beyondarena-view/v1'
            and source.get('benchmark') == view.get('benchmark') == 'BeyondArena'
            and source.get('task') == view.get('task') == task.get('problem_type') == 'regression'
            and source.get('sampled') is False and view.get('sampled') is False,
            'source_task_scope_mismatch')
    require(source['source_family_id'] == view['source_family_id'] == 'beyondarena:' + source['name']
            and view['id'] == source['name'] + '/r0/f0'
            and view['source_sha256'] == grant['source_metadata_sha256']
            and view['test_role'] == 'outer_test'
            and source['target'] == view['target'] == task['target_column_name']
            and source['objective_metric'] == view['metric'] == task['objective_metric_name'],
            'source_view_target_mismatch')
    require(grant['source_version'] == source['revision']
            and grant['raw_object'] == source['files']['dataset.parquet']
            and grant['bucket'] == source['bucket'], 'source_version_mismatch')
    require(type(grant['raw_object'].get('version_id')) is str
            and grant['raw_object']['version_id'] not in ('', 'null'),
            'versioned_object_required')
    require(grant['group_on'] == task['group_on'] and grant['time_on'] == task['time_on']
            and task.get('group_time_on') is None
            and task.get('stratify_on') is None,
            'group_time_binding_mismatch')
    group, time = task['group_on'], task['time_on']
    require(all(x is None or (type(x) is str and x and x != source['target'])
                for x in (group, time)), 'group_time_role_invalid')
    # per_sample means the declared group column supplies each physical row's
    # group identity. No aggregate label expansion or metadata rewriting occurs.
    require((group is not None and task.get('group_labels') == 'per_sample')
            or (group is None and task.get('group_labels') is None),
            'unsupported_group_labels')
    regime = source['split_regime']
    require((regime == 'iid' and group is None and time is None)
            or (regime == 'grouped_non_iid' and group is not None and time is None)
            or (regime == 'temporal_non_iid' and time is not None and group is None),
            'unsupported_split_regime')
    require(type(grant.get('maximum_source_bytes')) is int
            and 0 < grant['raw_object']['bytes'] <= grant['maximum_source_bytes'] <= 32 * 2**20,
            'source_byte_cap_invalid')
    require(type(source['columns']) is int and 2 <= source['columns'] <= 256
            and type(source['rows']) is int and 1 < source['rows'] <= 100000,
            'source_shape_cap_invalid')
    train, test = memberships(split, view, source['rows'])
    require(sha(canonical(train)) == grant['train_membership_sha256']
            and sha(canonical(test)) == grant['outer_test_membership_sha256'],
            'membership_binding_mismatch')
    return grant, source, task, train, test


def scalar(value):
    """Stable identity encoding; no target enters the IID grouping key."""
    if isinstance(value, (dt.datetime, dt.date)):
        return ['time', value.isoformat()]
    if value is None:
        return ['missing']
    if type(value) is float:
        if math.isnan(value):
            return ['float_nan']
        require(math.isfinite(value), 'nonfinite_identity_value')
        return ['float', value.hex()]
    require(type(value) in (str, int, bool), 'unsupported_identity_type')
    return [type(value).__name__, value]


def ordered_time(value):
    if isinstance(value, dt.datetime):
        require(value.tzinfo is None, 'unsupported_time_zone')
        return value.isoformat()
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time()).isoformat()
    if type(value) is str:
        try:
            parsed = dt.datetime.fromisoformat(value)
        except ValueError:
            raise ExportRefusal('time_requires_unambiguous_iso_format') from None
        require(parsed.tzinfo is None, 'unsupported_time_zone')
        return parsed.isoformat()
    raise ExportRefusal('unsupported_time_type')


def inner_partition(keys, *, temporal):
    """Keep every duplicate/group/time block in one TRAIN-derived partition."""
    distinct = sorted(set(keys))
    require(len(distinct) >= 2, 'insufficient_inner_groups')
    if not temporal:
        distinct.sort(key=lambda key: sha(canonical([1729, key])))
    cut = min(len(distinct) - 1, max(1, int(len(distinct) * .8)))
    fitting = set(distinct[:cut])
    return {'inner_fit': [i for i, key in enumerate(keys) if key in fitting],
            'inner_validation': [i for i, key in enumerate(keys) if key not in fitting]}
