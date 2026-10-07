"""Reproduce cost sidecars using only this committed, pinned numeric ledger."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

REPORT_SHA = 'bb94cf926bb37e3984a707c049facdf178e40cd61131daa6d369e27e93c7ee64'
PROOF_SHA = 'dddcb70f89a4e7dad0108ce40f87b712798b798bb02fe0df4348a3bea8754a2a'
FIELDS = ('id', 'method', 'phase', 'category', 'receipts', 'started', 'seconds',
          'clock', 'scheduler_seconds', 'peak_rss_bytes', 'peak_device_used_mib',
          'co_tenant', 'shared_gpu_cost', 'statuses', 'accounting', 'evidence')
ROW_IDS = ('tabsyn_original_cpu_v12_new41_sample',
           'A952_bulk_generated12_attempt01_whole',
           'exact16_metadata_mirror_01', 'exact16_metadata_mirror_02')


def require(condition):
    if not condition:
        raise ValueError('cost delta evidence or projection changed')


def unique(items):
    result = {}
    for key, value in items:
        require(key not in result)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('nonfinite cost metadata')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def checked(path, sha):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)))
    body = path.read_bytes()
    require(len(body) <= 1024 * 1024 and hashlib.sha256(body).hexdigest() == sha)
    return json.loads(body, object_pairs_hook=unique, parse_constant=reject_constant)


def validate(report, proof):
    require(report['format'] == 'work-order-b-current-closure-cost-delta-v4')
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
        require(all(row[k] is None for k in ('scheduler_seconds', 'peak_device_used_mib', 'co_tenant', 'shared_gpu_cost')))
    require([r['category'] for r in rows] == ['science', 'generated_qualification', 'engineering', 'engineering'])
    require(rows[0]['peak_rss_bytes'] is None and rows[1]['peak_rss_bytes'] == 255500288
            and rows[2]['peak_rss_bytes'] is None and rows[3]['peak_rss_bytes'] is None)
    ts = report['tabsyn_original_cpu_v12']
    require(ts['admitted'] + ts['unavailable'] + ts['unstarted'] == ts['denominator'] == 600)
    require(ts['admitted'] == ts['retained_admitted'] + ts['newly_admitted'] == 174)
    require(ts['unavailable'] == ts['prior_unavailable'] + ts['newly_unavailable'] == 49)
    require(ts['unstarted'] == ts['eligible_original_unstarted'] + ts['expired_original_unstarted'] == 377)
    require(ts['new_raw_workers'] == ts['new_raw_exit_zero'] == rows[0]['receipts'] == 41)
    require(ts['newly_admitted'] == rows[0]['statuses']['admitted_new'] == 31)
    require(ts['newly_unavailable'] == rows[0]['statuses']['unavailable_new'] == 10)
    require(math.fsum((ts['new_admitted_seconds'], ts['new_unavailable_seconds'])) == ts['new_whole_operation_seconds'] == rows[0]['seconds'])
    require(ts['actual_controller_kernel_exit'] is None and ts['actual_parent_wait_exit'] is None
            and ts['actual_parent_wait_captured'] is False and ts['full_panel_complete'] is False
            and ts['method_completion_claim'] is None and ts['original_clocks_reset'] is False)
    require(ts['captured_controller_and_worker_paths_absent_twice'] is True and ts['active'] == 0)
    bulk = report['bulk_generated12']
    require(bulk['actual_worker_exit'] == bulk['actual_Popen_exit'] == bulk['actual_parent_wait_exit'] == 0)
    require(bulk['positive_qualification'] is True and bulk['actor_absent'] is True)
    require(bulk['whole_operation_seconds'] == rows[1]['seconds'] == 11.803844023030251)
    require(bulk['nested_inner_seconds_not_added'] == 11.180450793821365)
    require(bulk['only_generated_arrays'] is True and bulk['study_data_decoded'] is False
            and bulk['new_generator_fits'] == bulk['GPU_queries'] == 0)
    require(bulk['sampled_resident_measurement_is_allocator_cap'] is False)
    mirrors = report['engineering_mirrors']
    require(mirrors['distinct_transports'] == 2 and mirrors['files_verified_per_transport'] == 16
            and mirrors['metadata_bytes_verified_per_transport'] == 344997)
    require(mirrors['same_readback_digest_does_not_merge_transports'] is True)
    require(math.fsum(r['seconds'] for r in rows[2:]) == mirrors['transport_wall_seconds'] == 0.5042349062860012)
    require(mirrors['scientific_initializers'] == mirrors['probes'] == 0)
    require(proof['format'] == 'work-order-b-cost-closure-delta-input-proof-v4')
    require(proof['tabsyn_exit_receipts_independently_rehashed'] is True
            and proof['tabsyn_small_exit_files'] == 41 and proof['tabsyn_small_exit_bytes'] == 21610
            and proof['tabsyn_elapsed_admission_worker_exit_joins_verified'] is True)
    require(proof['bulk_actual_custody_release_exact_joins_verified'] is True
            and proof['distinct_mirror_transports_verified'] is True)
    require(proof['raw_rows_models_samples_or_official_tests_opened'] is False
            and proof['dependency_modules_imported'] is False and proof['new_scientific_jobs'] == 0)


def schema(value):
    """A finite publication schema binds this exact retained metadata document."""
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'const': value}


def project(report, proof):
    validate(report, proof)
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(FIELDS)
    table = ['# New recorded costs from closed research operations', '',
             '| Category | Method | Phase | Operations | Recorded seconds |',
             '|---|---|---|---:|---:|']
    for row in report['rows']:
        writer.writerow(['null' if row[k] is None else json.dumps(row[k], sort_keys=True, separators=(',', ':'))
                         if isinstance(row[k], (dict, list, bool)) else row[k] for k in FIELDS])
        table.append(f"| {row['category']} | {row['method']} | {row['phase']} | {row['receipts']} | {row['seconds']:.6f} |")
    table += ['', 'The TabSyn sample row contains 31 admitted and 10 unavailable operations; all 41 worker exits were 0. Controller kernel exit and parent wait remain unknown.',
              'Generated qualification and engineering transports are separate categories. The two transports remain distinct despite identical readback digests.',
              'The bulk inner clock overlaps its whole-operation clock and is not added. Previous scopes are not charged again; no campaign total or native winner is claimed.', '']
    return {'costs.csv': stream.getvalue().encode(), 'costs.md': '\n'.join(table).encode(),
            'delta.schema.json': canonical(schema(report)), 'source-proof.schema.json': canonical(schema(proof))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).absolute().parent
    report = checked(here / 'delta.json', REPORT_SHA)
    proof = checked(here / 'source-proof.json', PROOF_SHA)
    outputs = project(report, proof)
    manifest = {'format': 'work-order-b-closure-cost-delta-publication-v4',
                'scientific_scope': 'Recorded TRAIN-derived sample operations; generated qualification and byte-only engineering shown separately; official TEST sealed',
                'private_preparation_input_proof_ref': proof['private_preparation_input_proof_ref'],
                'new_scientific_jobs_started': 0, 'production_claims': None,
                'files': {name: {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
                          for name, body in outputs.items()}}
    for name in ('delta.json', 'source-proof.json', 'render.py', 'README.md'):
        path = here / name
        require(not any(p.is_symlink() for p in (path, *path.parents)))
        body = path.read_bytes()
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
