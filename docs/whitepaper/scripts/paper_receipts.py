"""Authenticate committed original metadata and reduce the historical paper extracts.

Captured in October 2026, not a reconstruction of the original historical
custody. Numeric replay logs are auxiliary fits, not scored model artifacts.
No scratch access, raw rows, weights, or official test contents are required.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import statistics
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
INDEX_PATH = 'research/benchmark/results/paper-original-metadata-v1/index.json'
INDEX_SHA256 = '6d73d5f7b37849526a98a451b41a49e045ef21e911058d920940c5ea989f7fd9'
PROFILES = ('features12_steps2048', 'features12_steps8192')
LOSS_SUMMARY_SHA256 = '0124915497c98e560ef99c1e5ecf7ad876f9174439ff9c763cf1cf906b4aebe1'


def checked_bytes(ref, repo=REPO):
    """Check path, byte count and digest before any JSON/TSV decoding."""
    relative = ref['path']
    path = PurePosixPath(relative) if type(relative) is str else None
    if (path is None or path.is_absolute() or '..' in path.parts
            or path.as_posix() != relative or not relative.startswith('research/benchmark/results/paper-original-metadata-v1/')):
        raise ValueError('invalid committed metadata path')
    current = repo
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('symlinked committed metadata')
    if type(ref['bytes']) is not int or not 0 < ref['bytes'] < 1_000_000:
        raise ValueError('invalid committed metadata length')
    with current.open('rb') as handle:
        raw = handle.read(ref['bytes'] + 1)
    if (type(ref['sha256']) is not str
            or len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']):
        raise ValueError('committed metadata drift')
    return raw


def load_index(repo=REPO):
    raw = checked_bytes({'path': INDEX_PATH, 'bytes': 298306, 'sha256': INDEX_SHA256}, repo)
    document = json.loads(raw)
    if (document['format'] != 'dope-paper-original-metadata-v1'
            or any(document[key] is not False for key in ('official_tests_opened', 'rows_or_weights_read',
                   'original_historical_frozen_hashes_reconstructed', 'replay_replaces_scored_artifact'))):
        raise ValueError('invalid original metadata scope')
    records = document['records']
    identities = [(r['kind'], r['profile'], r['lineage_or_family']) for r in records]
    if len(records) != 210 or len(set(identities)) != 210:
        raise ValueError('incomplete original receipt matrix')
    groups = {}
    for kind, count in [('replay', 100), ('beyond', 5)]:
        for profile in PROFILES:
            group = {r['lineage_or_family'] for r in records if r['kind'] == kind and r['profile'] == profile}
            if len(group) != count:
                raise ValueError('incomplete original receipt matrix')
            groups[kind, profile] = group
        if groups[kind, PROFILES[0]] != groups[kind, PROFILES[1]]:
            raise ValueError('unmatched original receipt profiles')
    return document


def load_fit(record, repo=REPO):
    receipt = json.loads(checked_bytes(record['receipt_ref'], repo))
    report = json.loads(checked_bytes(record['report_ref'], repo))
    flag = 'official_tests_opened' if record['kind'] == 'replay' else 'official_test_opened'
    identity = 'dataset' if record['kind'] == 'replay' else 'name'
    if (receipt[flag] is not False or receipt['status'] != 'ok' or receipt['exit_code'] != 0
            or type(receipt['exit_code']) is not int
            or receipt['profile'] != record['profile'] or receipt[identity] != record['lineage_or_family']
            or type(receipt['artifact_bytes']) is not int or receipt['artifact_bytes'] != report['artifact_bytes']
            or type(report['rows']) is not int or report['contains_row_payloads'] is not False
            or report['contains_row_references'] is not False or report['compliant'] is not False
            or report['release_policy']['maximum_artifact_bytes'] != 10240
            or report['release_policy']['tier'] != 'l3'
            or not math.isfinite(receipt['elapsed_seconds']) or receipt['elapsed_seconds'] < 0):
        raise ValueError('invalid original fit receipt')
    if record['kind'] == 'replay' and receipt['replaces_scored_artifact'] is not False:
        raise ValueError('replay cannot replace scored artifact')
    return receipt, report


def load_loss(record, repo=REPO):
    ref = record['loss_trace_ref']
    compressed = checked_bytes(ref, repo)
    limit = ref['original_bytes']
    if type(limit) is not int or not 0 < limit < 1_000_000:
        raise ValueError('invalid numeric log length')
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as handle:
        raw = handle.read(limit + 1)
    if len(raw) != limit or hashlib.sha256(raw).hexdigest() != ref['original_sha256']:
        raise ValueError('original numeric log drift')
    steps = int(record['profile'].rsplit('steps', 1)[1])
    rows = []
    for step, line in enumerate(raw.decode('ascii').splitlines()):
        fields = line.split('\t')
        if len(fields) != 3 or int(fields[0]) != step:
            raise ValueError('invalid numeric log steps')
        values = [float(x) for x in fields[1:]]
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError('invalid numeric log loss')
        rows.append(values)
    if len(rows) != steps:
        raise ValueError('incomplete numeric log')
    return rows


def loss_matrix(profile, repo=REPO):
    if profile not in PROFILES:
        raise ValueError('unknown replay profile')
    index = load_index(repo)
    records = sorted((r for r in index['records'] if r['kind'] == 'replay' and r['profile'] == profile),
                     key=lambda r: r['lineage_or_family'])
    return [load_loss(r, repo) for r in records]


def prepare(index=None, repo=REPO):
    index = load_index(repo) if index is None else index
    value = json.loads(checked_bytes(index['beyond_prepare_ref'], repo))
    rows = value['results']
    if (value['official_test_opened_by_optimizer'] is not False or len(rows) != value['families']
            or len({r['name'] for r in rows}) != len(rows)
            or sum(r['status'] == 'prepared' for r in rows) != value['prepared']
            or sum(r['status'] == 'prepare_failed' for r in rows) != value['failed']):
        raise ValueError('invalid preparation projection')
    for row in rows:
        if row['status'] == 'prepared':
            if row['official_test_opened_by_optimizer'] is not False or row['test_in_worker'] is not False:
                raise ValueError('invalid preparation split flags')
        elif row.get('reason_category') not in ('projected_feature_range', 'cross_partition_overlap'):
            raise ValueError('invalid preparation category')
    return value


def source(ref, pointer):
    return {k: ref[k] for k in ('path', 'bytes', 'sha256')} | {'pointer': pointer}


def reduce_originals(repo=REPO):
    index = load_index(repo)
    replay, families, trace = {}, {}, []
    for profile in PROFILES:
        values = []
        for record in sorted(index['records'], key=lambda r: (r['kind'], r['profile'], r['lineage_or_family'])):
            if record['profile'] != profile:
                continue
            receipt, report = load_fit(record, repo)
            losses = load_loss(record, repo)
            trace.append({'kind': record['kind'], 'profile': profile, 'identity': record['lineage_or_family'],
                          'elapsed_seconds': source(record['receipt_ref'], '/elapsed_seconds'),
                          'artifact_bytes': source(record['receipt_ref'], '/artifact_bytes'),
                          'fit_rows': source(record['report_ref'], '/rows'),
                          'loss': source(record['loss_trace_ref'], 'TSV fields 2=train and 3=hidden-basis validation; zero-based step in field 1')})
            if record['kind'] == 'replay':
                values.append(receipt['elapsed_seconds'])
            else:
                family = families.setdefault(record['lineage_or_family'], {'name': record['lineage_or_family'], 'fit_rows': report['rows']})
                if family['fit_rows'] != report['rows']:
                    raise ValueError('unmatched family fit rows')
                family['steps' + profile.rsplit('steps', 1)[1]] = {
                    'bytes': receipt['artifact_bytes'], 'elapsed_seconds': receipt['elapsed_seconds'],
                    'official_test_opened': receipt['official_test_opened'], 'status': receipt['status'],
                    'validation_loss': losses[-1][1], 'within_l3_cap': receipt['within_l3_cap']}
        values.sort()
        replay[profile] = {'max': values[-1], 'median': statistics.median(values), 'min': values[0], 'n': len(values), 'sum': sum(values)}
    inventory = json.loads(checked_bytes(index['filename_inventory_ref'], repo))
    if inventory['file_contents_opened'] is not False:
        raise ValueError('filename inventory opened contents')
    counts = {'note': 'File-name counts only. Test file contents were not read.'}
    for group, prefix, columns in [('s3_worker', 's3', ('train','validation','test')),
                                   ('beyond_worker', 'beyond_worker', ('test',)),
                                   ('beyond_evaluator', 'beyond_evaluator', ('test',))]:
        rows = inventory[group]
        if len({r['identity'] for r in rows}) != len(rows):
            raise ValueError('duplicate filename inventory identity')
        for column in columns:
            if any(type(r[column + '_csv']) is not bool for r in rows):
                raise ValueError('invalid filename presence')
            key = 's3_worker_test_csv' if group == 's3_worker' and column == 'test' else prefix + '_' + column + '_csv'
            counts[key] = sum(r[column + '_csv'] for r in rows)
    counts['s3_workers'] = len(inventory['s3_worker']); counts['beyond_workers'] = len(inventory['beyond_worker'])
    if counts['s3_worker_test_csv'] or counts['beyond_worker_test_csv']:
        raise ValueError('test filename present in worker')
    preparation = prepare(index, repo)
    mapping = {'format': 'dope-paper-original-field-map-v1', 'index': {'path':INDEX_PATH,'sha256':INDEX_SHA256},
               'fits': trace, 'filename_counts': source(index['filename_inventory_ref'], '/'),
               'prepare': source(index['beyond_prepare_ref'], '/'),
               'replay_cost_reduction': 'Sort elapsed_seconds within each 100-lineage profile; n/min/max/sum/median.',
               'loss_curve_reduction': 'Per-step lineage median; 10000 lineage bootstrap draws with seed 20261005, train then validation per profile, stride 8 with linear interpolation. render_figures.py.',
               'beyond_reduction': 'Each family/profile artifact bytes and elapsed_seconds from receipt; fit_rows from report; validation_loss from final third TSV field.'}
    return {'replay-cost.json': replay, 'beyond-fit.json': {'families':[families[k] for k in sorted(families)],
            'source': 'beyondarena-prepared-v1/fits receipt.json, report.json, and the final loss.tsv row',
            'validation_column':'third TSV field, the hidden-basis validation loss'},
            'provenance-counts.json':counts, 'original-field-map.json':mapping}, preparation


def checked_loss_summary(repo=REPO):
    raw = (repo / 'docs/whitepaper/generated/loss-curves.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOSS_SUMMARY_SHA256:
        raise ValueError('loss summary drift; regenerate from original numeric logs')
    return json.loads(raw)


def checked_extracts(repo=REPO):
    extracts, _ = reduce_originals(repo)
    for name, expected in extracts.items():
        raw = (repo / 'docs/whitepaper/generated' / name).read_text()
        if raw != json.dumps(expected, indent=2, sort_keys=True) + '\n':
            raise ValueError('original receipt extract drift')
    return extracts, checked_loss_summary(repo)
