"""Publish the closed ARF native ledger without executing models or opening tests."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat

from . import arf_native_selection as selection
from . import arf_runtime_guard as guard
from .manifest import digest
from .score import sha256

BASE = Path('/mnt/fast-scratch/dope-benchmark')
ROOT = BASE / 'arf-s3-population-native-v3'
ROUND = '610c1ba0db58fe380350936f55c5e2e754cd09190616d7b56a6f9311d0d5133c'
NAME = 'arf-s3-population-native'
PATH_CONTRACT = 'cb74bea53a5ca97a9ef8e969afd577c6ffaa6455f3a3655d6cc0d09457a0c29e'
GATES = dict(official_tests_opened=False, full_campaign_admitted=False,
    shared_validation_evaluation_complete=False, shared_outcomes_used_for_selection=False,
    native_values_ranked_across_methods=False, native_values_aggregated_across_datasets=False,
    system_dynamic_library_closure_certified=False, counts_as_dope_win=False,
    mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)


def safe(path):
    p = guard.safe(path, BASE)
    guard.require('evaluator' not in p.parts and p.name != 'test.csv', 'sealed input forbidden')
    return p


def bound(path, expected):
    guard.digest(expected)
    safe(path)
    return guard.bound(path, expected, BASE)


def flat(root, files):
    root = safe(root)
    guard.require({p.name for p in root.iterdir()} == set(files), 'flat inventory differs')
    for name, row in files.items():
        guard.require(Path(name).name == name, 'invalid flat member')
        p = safe(root / name)
        expected = row['sha256'] if isinstance(row, dict) else row
        guard.digest(expected)
        guard.require(stat.S_ISREG(p.lstat().st_mode) and sha256(p) == expected,
                      'flat member changed')
        if isinstance(row, dict):
            guard.require(type(row['bytes']) is int and p.stat().st_size == row['bytes'],
                          'flat byte count changed')


def directory_name():
    path = Path(__file__).with_name('arf-native-paths.lock.json')
    declaration = guard.bound(path, PATH_CONTRACT, path.parent)
    name = declaration['runtime_directory']
    guard.require(declaration['origin_round_sha256'] == ROUND
        and declaration['directory_must_be_empty'] is True
        and type(name) is str and Path(name).name == name and name not in ('', '.', '..'),
        'frozen native path declaration changed')
    return name


def attempt_inventory(out, evidence):
    entries = list(out.rglob('*'))
    expected_dirs = {str(parent) for name in evidence
        for parent in (out / name).parents if parent != out and parent.is_relative_to(out)}
    reserved = out / directory_name()
    expected_dirs.add(str(reserved))  # Created by the frozen coordinator.
    guard.require(not any(q.is_symlink() for q in entries)
        and all(stat.S_ISREG(q.lstat().st_mode) or stat.S_ISDIR(q.lstat().st_mode) for q in entries)
        and {str(q) for q in entries if q.is_dir()} == expected_dirs
        and not list(safe(reserved).iterdir())
        and {q.relative_to(out).as_posix() for q in entries if q.is_file()} == set(evidence) | {'receipt.json'},
        'attempt evidence inventory differs')


def verify_inputs(receipt_sha256, report_sha256):
    guard.digest(receipt_sha256); guard.digest(report_sha256)
    anchor = bound(ROOT / 'receipt-lock-v1.json', receipt_sha256)
    guard.require(anchor['complete_native_matrix'] is True
        and anchor['round_sha256'] == ROUND and anchor['reconciliation_sha256'] == report_sha256,
        'complete externally anchored closure required')
    # All receipt/source/model/worker bytes precede any native metric decoding.
    for path, expected in anchor['refs'].items():
        guard.digest(expected)
        p = safe(path)
        guard.require(stat.S_ISREG(p.lstat().st_mode) and sha256(p) == expected,
                      'frozen native evidence changed')
    lock = bound(ROOT / 'round.lock.json', ROUND)
    declaration = guard.bound(Path(__file__).with_name('arf-native-paths.lock.json'),
                              PATH_CONTRACT, Path(__file__).parent)
    guard.require(declaration['origin_coordinator_sha256'] == lock['source_files'][str(ROOT / 'source/coordinator.py')],
                  'frozen path declaration source differs')
    guard.require(anchor['refs'][str(ROOT / 'round.lock.json')] == ROUND,
                  'round absent from receipt inventory')
    flat(ROOT / 'source', {Path(p).name: h for p, h in lock['source_files'].items()})
    runtime = bound(lock['runtime_path'], lock['runtime_sha256'])
    for ref in runtime['inventory_locks']:
        inv = bound(ref['path'], ref['sha256'])
        guard.require(not inv.get('aliases'), 'runtime aliases forbidden')
        for root in inv['roots']:
            members = {p: row for p, row in inv['files'].items() if Path(p).is_relative_to(root)}
            guard.inventory(root, members, BASE, runtime['directory_inventories'][root])
    guard.inventory(runtime['source_root'], runtime['source_files'], BASE,
                    runtime['directory_inventories'][runtime['source_root']])
    guard.require(not os.path.lexists(runtime['absent_zip'])
        and not list(safe(runtime['empty_cache']).rglob('*'))
        and sha256(safe(runtime['python'])) == runtime['python_sha256'],
        'declared interpreter chain changed')
    selected_path = ROOT / 'source/arf_native_selection.py'
    guard.require(sha256(Path(selection.__file__)) == lock['source_files'][str(selected_path)],
                  'native selection implementation changed')
    closed = bound(ROOT / 'completion.json', anchor['refs'][str(ROOT / 'completion.json')])
    actual = bound(ROOT / 'coordinator-exit.json', anchor['refs'][str(ROOT / 'coordinator-exit.json')])
    launch = bound(ROOT / 'supervisor-launch.json', anchor['refs'][str(ROOT / 'supervisor-launch.json')])
    guard.require(type(actual['exit_code']) is int and actual['exit_code'] == 0
        and actual['round_sha256'] == closed['round_sha256'] == launch['round_sha256'] == ROUND
        and type(closed['jobs']) is int and closed['jobs'] == 800
        and actual['coordinator_log_sha256'] == anchor['refs'][str(ROOT / 'coordinator.log')]
        and all(not Path('/proc', str(launch[k])).exists() for k in ('pid', 'coordinator_pid')),
        'actual closed coordinator required')
    report = bound(ROOT / 'reconciliation-v1.json', report_sha256)
    guard.require(report['status_counts'] == closed['counts'], 'closure counts differ')
    return anchor, lock, report


def assemble(lock, report, receipts):
    guard.require(report['complete_native_matrix'] is True
        and report['planned_fit_jobs'] == report['closed_fit_jobs'] == lock['planned_fit_jobs'] == 800
        and report['dataset_lineages'] == 100 and report['actual_coordinator_exit_code'] == 0,
        'complete 800-trial native matrix required')
    for data in (lock, report):
        for key, value in GATES.items():
            if key in data:
                guard.require(data[key] is value, 'native ledger acquired a gated claim')
    jobs = [r['job'] for r in lock['jobs']]
    guard.require(len(jobs) == len(receipts) == 800
        and {digest(j) for j in jobs} == set(receipts), 'trial coverage changed')
    groups = defaultdict(list); trials = []
    for frozen, job in zip(lock['jobs'], jobs):
        key = digest(job); r = receipts[key]
        guard.require(key == frozen['job_sha256'] == r['job_sha256'] == digest(r['job'])
            and type(job['trial_index']) is int and type(job['fit_seed']) is int and job['fit_seed'] == 11
            and job['configuration'] == lock['native_grid'][job['trial_index']]
            and r['round_sha256'] == ROUND and r['shared_outcomes_used_for_selection'] is False,
            'canonical frozen trial differs')
        for name in ('official_tests_opened', 'mfs_v2', 'ptf_v1', 'release_safe', 'superiority'):
            guard.require(r[name] is GATES[name], 'trial acquired a gated claim')
        status = r['status']; result = r['result']
        outcome = ('ok' if status == 'ok' else 'scheduling_cutoff' if status.startswith('deadline_')
                   else 'infrastructure' if status == 'infra_interrupted' else 'method_failure')
        groups[job['dataset']].append((job, r))
        trials.append(dict(dataset=job['dataset'], method='ARF', trial_index=job['trial_index'],
            job_sha256=key, configuration=job['configuration'], configuration_sha256=job['configuration_sha256'],
            fit_seed=11, status=status, outcome_class=outcome, operation_seconds=r['elapsed_seconds'],
            charged_artifact_bytes=result['artifact_bytes'] if result else None,
            projection_bytes=result['artifact_inventory']['projection.json']['bytes'] if result else None,
            native_kpi=result['native_kpi'] if result else None,
            host=r['operation']['host'], peak_resident_bytes_including_coordinator=r['operation'].get('peak_resident_bytes_including_coordinator'),
            energy_attributable_to_job=None, new_operation_started=r['operation']['new_operation_started'],
            receipt_path=str(ROOT / 'attempts' / key / 'attempt-0001/receipt.json'),
            counts_as_dope_win=False))
    guard.require(len(groups) == 100, 'lineage coverage changed')
    selections, cells = [], []
    by_key = {t['job_sha256']: t for t in trials}
    for dataset, group in sorted(groups.items()):
        guard.require(len(group) == 8 and {j['trial_index'] for j, _ in group} == set(range(8)),
                      'eight native trials per lineage required')
        chosen = selection.select([j for j, _ in group], [r for _, r in group])
        default = next((digest(j) for j, r in group if j['trial_index'] == 0 and r['status'] == 'ok'), None)
        selections.append(dict(dataset=dataset, fit_seed=11, author_default_job_sha256=default, **chosen))
        winner = chosen['selected_job_sha256']
        cells.append(dict(dataset=dataset, method='ARF', fit_seed=11,
            author_default_job_sha256=default, selected_job_sha256=winner,
            default_native_kpi=by_key[default]['native_kpi'] if default else None,
            selected_native_kpi=by_key[winner]['native_kpi'] if winner else None,
            default_charged_bytes=by_key[default]['charged_artifact_bytes'] if default else None,
            selected_charged_bytes=by_key[winner]['charged_artifact_bytes'] if winner else None,
            successful_trials=chosen['successful_trials'], operation_seconds=chosen['elapsed_seconds']))
    guard.require(digest(selections) == digest(report['default_native_cells'])
        and dict(Counter(t['status'] for t in trials)) == report['status_counts']
        and abs(sum(t['operation_seconds'] for t in trials) - report['operation_seconds']) <= 1e-6,
        'fresh native selection or accounting differs from sealed evidence')
    return dict(format='dope-arf-population-native-publication', version=1,
        scope='Complete bounded official-training-derived native research; shared validation sampling remains pending.',
        datasets=100, trials=trials, cells=cells, native_grid=lock['native_grid'],
        native_objective=lock['native_objective'], status_counts=report['status_counts'],
        outcome_class_counts=dict(Counter(t['outcome_class'] for t in trials)),
        new_physical_attempts=800, fit_seeds=[11], all_frozen_cells_accounted=True,
        native_selected_cells=sum(c['selected_job_sha256'] is not None for c in cells),
        cost=dict(operation_seconds=report['operation_seconds'], scheduler_wall_seconds=report['scheduler_wall_seconds']),
        author_commit='8b63c1b3999981125b4af2828ff52cba8e29169d', author_code_modified=False,
        density_implementation_role=lock['native_objective']['implementation_role'],
        artifact_projection_bytes_included=True, l3_byte_cap=10240,
        selected_artifacts_within_l3_byte_cap=sum(c['selected_charged_bytes'] is not None and c['selected_charged_bytes'] <= 10240 for c in cells),
        **GATES)


def build(receipt_sha256, report_sha256):
    anchor, lock, report = verify_inputs(receipt_sha256, report_sha256)
    expected = {r['job_sha256'] for r in lock['jobs']}
    guard.require({p.name for p in safe(ROOT / 'attempts').iterdir()} == expected,
                  'attempt inventory differs')
    receipts = {}
    for key in sorted(expected):
        parent = safe(ROOT / 'attempts' / key)
        guard.require({p.name for p in parent.iterdir()} == {'attempt-0001'}, 'retry inventory differs')
        out = safe(parent / 'attempt-0001'); p = out / 'receipt.json'
        r = bound(p, anchor['refs'][str(p)])
        attempt_inventory(out, r['evidence_files'])
        for name, expected_sha in r['evidence_files'].items():
            guard.require(anchor['refs'].get(str(out / name)) == expected_sha,
                          'attempt file differs from frozen receipt lock')
        flat(Path(r['job']['worker']['path']), r['job']['worker']['files'])
        if r['status'] == 'ok':
            fit = bound(out / 'fit.json', anchor['refs'][str(out / 'fit.json')])
            guard.require(fit['status'] == 'ok' and fit['round_sha256'] == ROUND
                and fit['job_sha256'] == key and digest(fit['result']) == digest(r['result']),
                'native fit result differs')
            flat(out / 'artifact', r['result']['artifact_inventory'])
            guard.require(set(r['result']['artifact_inventory']) == {'adapter.json', 'bounds.csv',
                'categories.csv', 'continuous.csv', 'model.json', 'projection.json'}
                and r['result']['artifact_inventory']['projection.json']['sha256']
                    == r['job']['worker']['files']['projection.json'], 'complete projected artifact required')
        receipts[key] = r
    result = assemble(lock, report, receipts)
    for t in result['trials']:
        t['receipt_sha256'] = anchor['refs'][t['receipt_path']]
    workers = {r['job']['dataset']: r['job']['worker'] for r in lock['jobs']}
    result['dataset_lineages'] = [{k: w[k] for k in ('dataset', 'catalog_entry_sha256',
        'source_manifest_sha256', 'preparation_receipt_sha256', 'object_sha256',
        'raw_training_derived_split_hashes', 'files', 'train_rows', 'projected_features')}
        for _, w in sorted(workers.items())]
    result['source_locks'] = dict(round=ROUND, receipts=receipt_sha256, reconciliation=report_sha256)
    for path, expected in anchor['refs'].items():
        guard.require(sha256(safe(path)) == expected, 'evidence changed during publication')
    bound(ROOT / 'receipt-lock-v1.json', receipt_sha256)
    from . import publish_s3_matched
    result['publisher_sources'] = {Path(m.__file__).name: sha256(Path(m.__file__))
        for m in (selection, guard, publish_s3_matched, __import__(__name__, fromlist=['']))}
    result['publisher_sources']['arf-native-paths.lock.json'] = PATH_CONTRACT
    return result


def table(report):
    out = io.StringIO(); writer = csv.writer(out, lineterminator='\n')
    columns = ['dataset', 'default_native_density', 'selected_native_density', 'default_bytes',
               'selected_bytes', 'successful_trials', 'operation_seconds']
    writer.writerow(columns)
    for c in report['cells']:
        writer.writerow([c['dataset'], *(c[k]['value'] if c[k] else '' for k in ('default_native_kpi', 'selected_native_kpi')),
            c['default_charged_bytes'], c['selected_charged_bytes'], c['successful_trials'], c['operation_seconds']])
    return out.getvalue()


def markdown(report):
    return '\n'.join(['# ARF complete native research ledger', '', report['scope'], '',
        f"100 matched S3 lineages; 800 closed trials; status counts: {report['status_counts']}.",
        f"Native winners available: {report['native_selected_cells']}/100.", '',
        'Original author ARF/FORDE code is unchanged. Selection maximizes held-out FORDE',
        'mean log-density computed by the study evaluator of the author leaf mixture.',
        'Ties use charged bytes then configuration SHA. The grid has eight trials per lineage.',
        'Every trial and default/native winner is retained; shared outcomes never enter selection.',
        'Native density values have dataset-dependent units and are not aggregated or cross-ranked.', '',
        f"Charged native-selected artifacts within 10,240 bytes: {report['selected_artifacts_within_l3_byte_cap']}/100.",
        'This is byte feasibility only. Unconstrained ARF quality remains eligible for later measurement;',
        'a successful fit or a small artifact is not a generator gate pass or a DOPE win.', '',
        f"Charged operation time: {report['cost']['operation_seconds']:.6f} seconds.",
        f"Coordinator wall time including admission waits: {report['cost']['scheduler_wall_seconds']:.6f} seconds.",
        'Memory peaks include the coordinator; attributable energy is unavailable.', '',
        'One fit seed (11); bounded training views, not full official training partitions.',
        'Shared n/4n validation sampling, five-fit replication and privacy/release gates remain pending.',
        'Python/provider/source inventories are checked; complete system linker closure is not certified.',
        'Official tests remain sealed. MFS-v2/PTF-v1/release-safe/superiority remain null.', '',
        'Regenerate the table from committed JSON with `--from-json`, or verify scratch using the',
        'external receipt/reconciliation hashes in the publication manifest.', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-lock-sha256')
    parser.add_argument('--reconciliation-sha256')
    parser.add_argument('--from-json', type=Path)
    parser.add_argument('--publication-sha256')
    parser.add_argument('--output-directory', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    if args.from_json:
        guard.digest(args.publication_sha256)
        data = args.from_json.read_bytes()
        guard.require(hashlib.sha256(data).hexdigest() == args.publication_sha256,
                      'committed publication changed')
        result = guard.decode(data)
        guard.require(all(result[k] is v for k, v in GATES.items()), 'publication acquired a claim')
        for name, expected in result['publisher_sources'].items():
            guard.require(Path(name).name == name and sha256(Path(__file__).with_name(name)) == expected,
                          'publisher implementation changed')
    else:
        result = build(args.receipt_lock_sha256, args.reconciliation_sha256)
    from .publish_s3_matched import schema
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for suffix, text in (('json', json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'),
                         ('csv', table(result)), ('md', markdown(result)),
                         ('schema.json', json.dumps(schema(result), sort_keys=True, indent=2) + '\n')):
        (args.output_directory / (NAME + '.' + suffix)).write_text(text)
    manifest = dict(format='dope-arf-native-publication-manifest', version=1,
        source_locks=result['source_locks'], publisher_sources=result['publisher_sources'],
        artifacts={NAME + '.' + s: dict(sha256=sha256(args.output_directory / (NAME + '.' + s)),
            bytes=(args.output_directory / (NAME + '.' + s)).stat().st_size)
            for s in ('json', 'csv', 'md', 'schema.json')}, **GATES)
    for suffix, value in (('manifest.json', manifest), ('manifest.schema.json', schema(manifest))):
        (args.output_directory / (NAME + '.' + suffix)).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
