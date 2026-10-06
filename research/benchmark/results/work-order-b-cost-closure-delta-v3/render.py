"""Reproduce rights-safe closure cost tables from the committed, pinned JSON."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

REPORT_SHA = '23d6712824c3553e6012b700d77425b938c0feb43a2a6bd9ee0295083191cecc'
PROOF_SHA = 'cf0cc457555aac38b45fc983069c0e90b53832ebafb92295eac4511819ea8c5e'
FIELDS = ('id', 'method', 'phase', 'receipts', 'started', 'seconds', 'clock',
          'scheduler_seconds', 'peak_rss_bytes', 'peak_device_used_mib', 'hosts',
          'statuses', 'accounting', 'evidence')
ROW_IDS = ('forest_current_v24_custody', 'forest_current_v24_fit',
           'forest_current_v24_metric', 'forest_current_v24_native',
           'forest_current_v24_sample', 'tabsyn_matched_v11_new_sample')


def require(condition):
    if not condition:
        raise ValueError('closure cost evidence or projection changed')


def unique(items):
    result = {}
    for key, value in items:
        require(key not in result)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('nonfinite closure cost metadata')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def checked(path, sha):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)))
    body = path.read_bytes()
    require(len(body) <= 1024 * 1024 and hashlib.sha256(body).hexdigest() == sha)
    return json.loads(body, object_pairs_hook=unique, parse_constant=reject_constant)


def validate(report):
    require(report['format'] == 'work-order-b-current-closure-cost-delta-v3')
    require(report['status'] == 'partial_not_final' and report['campaign_additivity'] is None)
    require(type(report['new_scientific_jobs_started_by_preparation']) is int
            and report['new_scientific_jobs_started_by_preparation'] == 0)
    require(all(value is None for value in report['unmeasured_claims'].values()))
    rows = report['rows']
    require(tuple(row['id'] for row in rows) == ROW_IDS)
    for row in rows:
        require(set(row) == set(FIELDS))
        require(type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0)
        require(type(row['receipts']) is int and row['receipts'] >= 0 and row['started'] == row['receipts'])
        require(type(row['started']) is int)
        require(all(type(n) is int and n >= 0 for n in row['statuses'].values()))
        require(sum(row['statuses'].values()) == row['receipts'])
        require(not {'method_failure', 'method_failed'} & set(row['statuses']))
        require(all(row[k] is None for k in ('scheduler_seconds', 'peak_rss_bytes', 'peak_device_used_mib')))
    forest, ts = report['forest_current_v24'], report['tabsyn_matched_v11']
    require(forest['full_required_cells'] + forest['unavailable_or_incomplete'] + forest['genuinely_unstarted'] == forest['denominator'] == 200)
    require(forest['active_jobs'] == 0 and forest['actual_controller_exit'] == 2 and forest['actual_supervisor_exit'] is None)
    require(forest['job36_actual_child_exit'] == 0 and forest['job36_actual_watchdog_exit'] == 1
            and forest['job36_full_quality_upgraded'] is False)
    require(sum(row['receipts'] for row in rows[:5]) == forest['current_unique_operations'] == 32)
    require(math.isclose(math.fsum(row['seconds'] for row in rows[:5]), forest['current_unique_operation_seconds'], rel_tol=0, abs_tol=1e-9))
    require(rows[0]['seconds'] == forest['custody_operation_seconds'] and rows[0]['receipts'] == forest['custody_operations'])
    require(ts['admitted'] + ts['unavailable'] + ts['unstarted'] == ts['denominator'] == 600)
    require(ts['admitted'] == ts['retained_admitted'] + ts['newly_admitted'])
    require(ts['unavailable'] == ts['prior_unavailable'] + ts['newly_unavailable'])
    require(ts['new_raw_workers'] == ts['newly_admitted'] + ts['newly_unavailable'] == rows[-1]['receipts'])
    require(ts['new_whole_operation_seconds'] == rows[-1]['seconds'])
    require(ts['actual_controller_exit'] is None and ts['parent_exit_status'] == 'UNKNOWN'
            and ts['sampler_closed_receipt_present'] is False and ts['full_panel_complete'] is False)


def schema(value):
    """A finite publication schema binds this exact retained metadata document."""
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'const': value}


def project(report):
    validate(report)
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(FIELDS)
    table = ['# Recorded costs from closed research queues', '',
             '| Method | Phase | Operations | Recorded seconds |',
             '|---|---|---:|---:|']
    for row in report['rows']:
        writer.writerow([json.dumps(row[k], sort_keys=True, separators=(',', ':'))
                         if isinstance(row[k], (dict, list)) else row[k] for k in FIELDS])
        table.append(f"| {row['method']} | {row['phase']} | {row['receipts']} | {row['seconds']:.6f} |")
    table += ['', 'Forest custody is metadata overhead. Phase walls are distinct within this current scope; earlier scopes have unproved additivity.',
              'Raw zero exits do not certify complete cells. Unknown parent exits, peak memory, hardware, energy and gated scores remain unknown.', '']
    return {'costs.csv': stream.getvalue().encode(), 'costs.md': '\n'.join(table).encode(),
            'delta.schema.json': canonical(schema(report))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).absolute().parent
    outputs = project(checked(here / 'delta.json', REPORT_SHA))
    proof = checked(here / 'source-proof.json', PROOF_SHA)
    require(proof['forest_referenced_operation_bodies_independently_rehashed'] is True
            and proof['forest_current_operation_body_elapsed_status_job_worker_exit_joins_verified'] is True
            and proof['forest_unique_operation_count'] == 32
            and proof['forest_current_operation_body_bytes'] == 57102)
    outputs['source-proof.schema.json'] = canonical(schema(proof))
    manifest = {'format': 'work-order-b-closure-cost-delta-publication-v3',
                'scientific_scope': 'TRAIN-derived research receipts; official TEST sealed',
                'private_receipt_custody_ref': proof['private_receipt_custody_ref'],
                'new_scientific_jobs_started': 0, 'production_claims': None,
                'files': {name: {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
                          for name, body in outputs.items()}}
    for name in ('delta.json', 'source-proof.json', 'render.py'):
        body = (here / name).read_bytes()
        manifest['files'][name] = {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    outputs['publication-manifest.json'] = canonical(manifest)
    outputs['publication-manifest.schema.json'] = canonical(schema(manifest))
    for name, body in outputs.items():
        path = here / name
        require(not any(p.is_symlink() for p in (path, *path.parents)))
        if path.exists():
            require(path.read_bytes() == body)
        else:
            require(not args.check)
            with path.open('xb') as stream:
                stream.write(body)
    print(json.dumps({name: hashlib.sha256(body).hexdigest() for name, body in outputs.items()}, sort_keys=True))


if __name__ == '__main__':
    main()
