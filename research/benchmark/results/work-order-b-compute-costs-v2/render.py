"""Validate the committed cost report and reproduce its CSV and JSON schema."""
import argparse
import csv
import hashlib
import io
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

REPORT_SHA = '9ffb69a46f509fc9c2a0da8efdf81bc6ccf1bbfe97c8491a5439f910cb1f42a4'
PUBLISHER_SHA = 'd90e275318b8eb766fb340334103060de8aeb3c81f65728e28314b2da4f34e01'
FIELDS = ('id', 'method', 'phase', 'receipts', 'started', 'seconds', 'clock',
          'scheduler_seconds', 'peak_rss_bytes', 'peak_device_used_mib', 'hosts',
          'statuses', 'accounting', 'evidence')


def checked(path, pin):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('file alias or unavailable metadata')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin:
        raise ValueError('frozen digest changed')
    return data


def primitives(path):
    """Import the pinned serializer source after rejecting existing caches."""
    expected = Path(__file__).resolve().parents[2] / 'publish_work_order_b_costs.py'
    if path.absolute() != expected:
        raise ValueError('publisher path differs from the serializer owner')
    checked(expected, PUBLISHER_SHA)
    cache = expected.parent / '.cost-report-source-only-cache'
    if cache.exists() or cache.is_symlink() or 'publish_work_order_b_costs' in sys.modules:
        raise ValueError('publisher cache must be absent before initialization')
    old_path, old_prefix, old_write = sys.path[:], sys.pycache_prefix, sys.dont_write_bytecode
    try:
        sys.path.append(str(expected.parent))
        sys.pycache_prefix, sys.dont_write_bytecode = str(cache), True
        spec = importlib.util.find_spec('publish_work_order_b_costs')
        if (spec is None or spec.origin != str(expected)
                or type(spec.loader) is not importlib.machinery.SourceFileLoader
                or not Path(spec.cached).is_relative_to(cache)):
            raise ValueError('publisher import origin differs')
        import publish_work_order_b_costs as publisher
        checked(expected, PUBLISHER_SHA)
        return {name: getattr(publisher, name) for name in
                ('require', 'canonical', 'reject_constant', 'finite_float',
                 'unique_object', 'number', 'count', 'schema')}
    finally:
        sys.modules.pop('publish_work_order_b_costs', None)
        sys.path[:] = old_path
        sys.pycache_prefix, sys.dont_write_bytecode = old_prefix, old_write


def validate(report, base):
    require = base['require']
    for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority',
                'total_gpu_hours', 'total_operation_seconds'):
        require(report[key] is None, 'unmeasured result acquired a claim')
    require(report['status'] == 'partial_not_final' and report['official_tests_opened'] is False
            and type(report['new_fits_samples_or_metric_jobs_started']) is int
            and report['new_fits_samples_or_metric_jobs_started'] == 0, 'research scope changed')
    require(all(report['resources'][k] is None for k in
                ('hardware_models', 'historical_co_tenant', 'shared_gpu_hours', 'energy_joules')),
            'unknown hardware or sharing acquired a claim')
    rows = report['rows']; require(len({r['id'] for r in rows}) == len(rows), 'physical row alias conflict')
    for r in rows:
        require(set(r) == set(FIELDS), 'row scope changed')
        for k in ('seconds', 'scheduler_seconds', 'peak_rss_bytes', 'peak_device_used_mib'):
            base['number'](r[k], nullable=True)
        for k in ('receipts', 'started'):
            if r[k] is not None: base['count'](r[k])
        require(not any(k in ('method_failure', 'method_failed') for k in r['statuses']),
                'scheduling or infrastructure relabeled as method failure')
        if r['receipts'] is not None:
            require(not r['statuses'] or sum(base['count'](n) for n in r['statuses'].values()) == r['receipts'],
                    'receipt count changed')
        cuts = r['statuses'].get('deadline_unstarted', 0)
        if cuts:
            classes = report['native_outcome_classes'][r['method'] + '/' + r['phase']]
            require(r['started'] + cuts == r['receipts'] and classes.get('scheduling_cutoff', 0) >= cuts
                    and not any(k in ('method_failure', 'method_failed') for k in classes), 'scheduling class changed')
    aliases = {r['id']: r for r in rows}
    require(aliases['density_shared_validation_v1']['accounting'] == 'alias_of_density_phase_rows_do_not_add',
            'density alias acquired another charge')
    require(sum(r['method'] == 'ARF' and r['phase'] == 'fit_and_native_wrapper' for r in rows) == 1,
            'ARF native wrapper charged twice')
    td = report['partial_scopes']['tabddpm']
    require(td['scientific_seconds'] is None and td['coordinator_numeric_exit'] is None
            and td['method_complete'] is False and td['method_failure'] is False, 'TD boundary acquired a result')


def project(report, base):
    validate(report, base); rows = report['rows']; schema = base['schema'](report); props = {}
    for key in FIELDS:
        if key in ('receipts', 'started'): props[key] = {'type': ['integer', 'null'], 'minimum': 0}
        elif key in ('seconds', 'scheduler_seconds', 'peak_rss_bytes', 'peak_device_used_mib'):
            props[key] = {'type': ['number', 'null'], 'minimum': 0}
        elif key == 'statuses': props[key] = {'type': 'object', 'additionalProperties': {'type': 'integer', 'minimum': 0}}
        elif key == 'hosts': props[key] = {'type': 'array', 'uniqueItems': True, 'items': {'enum': ['xbabe1', 'xbabe2', 'xbabe3']}}
        else: props[key] = {'enum': sorted({r[key] for r in rows})}
    schema['properties']['rows'] = dict(type='array', minItems=len(rows), maxItems=len(rows),
        items=dict(type='object', additionalProperties=False, required=sorted(props), properties=props))
    schema['properties']['partial_scopes']['properties']['forest']['properties']['negative22_actual_watchdog_exit'] = {'const': -15}
    stream = io.StringIO(); writer = csv.writer(stream, lineterminator='\n'); writer.writerow(FIELDS)
    for r in rows:
        writer.writerow([json.dumps(r[k], sort_keys=True, separators=(',', ':'))
                         if isinstance(r[k], (dict, list)) else r[k] for k in FIELDS])
    return {'costs.csv': stream.getvalue().encode(), 'results.schema.json':
            base['canonical']({'$schema': 'https://json-schema.org/draft/2020-12/schema', **schema}) + b'\n'}


def main():
    parser = argparse.ArgumentParser(); here = Path(__file__).resolve().parent
    parser.add_argument('--publisher', type=Path, default=here.parents[1] / 'publish_work_order_b_costs.py')
    parser.add_argument('--input', type=Path, default=here / 'results.json')
    parser.add_argument('--output', type=Path, default=here); args = parser.parse_args()
    base = primitives(args.publisher)
    report = json.loads(checked(args.input, REPORT_SHA), parse_constant=base['reject_constant'],
                        parse_float=base['finite_float'], object_pairs_hook=base['unique_object'])
    files = project(report, base); args.output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = args.output / name
        base['require'](not path.is_symlink() and not any(p.is_symlink() for p in path.parents)
                        and (not path.exists() or path.read_bytes() == data), 'projection drift or alias')
    for name, data in files.items():
        if not (args.output / name).exists():
            with (args.output / name).open('xb') as handle: handle.write(data)
    print(json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in files.items()}, sort_keys=True))


if __name__ == '__main__':
    main()
