"""Close the frozen SDV population research ledger without opening test rows."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

from . import sdv_native_accounting as accounting
from .manifest import digest
from .score import sha256

ROOT = Path('/mnt/fast-scratch/dope-benchmark/sdv-s3-population-native-v2')
ROUND_SHA = 'bf93cf99d730403956bd8730b61f456c4d22ae77a4280ba48ef0090238d792ed'


def failed_history(lock, base, refs):
    result = []
    for row in lock['previous_failed_trials']:
        path = accounting.evidence(row['receipt_path'], row['receipt_sha256'], base, refs)
        receipt = accounting.read(path)
        accounting.claims(receipt)
        job = receipt['job']
        accounting.require(receipt['status'] == 'failed' and digest(job) == row['source_job_sha256']
                           and receipt['round_sha256'] == lock['frozen_references'][str(path.parents[3] / 'round.lock.json')]
                           and (job['dataset'], job['method'], job['trial_index'])
                           == (row['dataset'], row['method'], row['trial_index'])
                           and receipt['counts_as_dope_win'] is False,
                           'previous failed native attempt was relabeled')
        for name, expected in receipt['evidence_files'].items():
            accounting.require(Path(name).name == name, 'previous failed evidence path changed')
            accounting.evidence(path.parent / name, expected, base, refs)
        seconds = 0.0
        for op in receipt['operations']:
            accounting.require(type(op['new_operation_started']) is bool
                               and accounting.finite(op['elapsed_seconds']) and op['elapsed_seconds'] >= 0,
                               'previous failed operation cost missing')
            if op['new_operation_started']:
                seconds += op['elapsed_seconds']
        accounting.require(abs(seconds - row['failed_operation_seconds']) <= 1e-9,
                           'previous failed native operation cost changed')
        partial = []
        directory = path.parent / 'artifact'
        if directory.exists():
            accounting.safe(directory, base)
            accounting.require(not any(p.is_symlink() for p in directory.rglob('*')),
                               'previous failed artifact gained an alias')
            for p in sorted(directory.rglob('*')):
                if p.is_file():
                    h = sha256(p)
                    accounting.evidence(p, h, base, refs)
                    partial.append({'path': p.relative_to(directory).as_posix(), 'bytes': p.stat().st_size, 'sha256': h})
        result.append(row | {'status': 'failed', 'partial_artifact_inventory': partial,
                             'partial_artifact_bytes': sum(r['bytes'] for r in partial),
                             'native_selection_eligible': False, 'counts_as_dope_win': False})
    accounting.require(len(result) == 2
                       and abs(sum(r['failed_operation_seconds'] for r in result)
                               - lock['previous_failed_operation_seconds']) <= 1e-9,
                       'previous failed native trials omitted')
    return result


def source_inventory(lock, refs):
    source = ROOT / 'source'
    accounting.safe(source, ROOT.parent)
    accounting.require(not any(p.is_symlink() for p in source.rglob('*'))
                       and {str(p) for p in source.rglob('*') if p.is_file()} == set(lock['source_files']),
                       'frozen native source inventory changed')
    for path, expected in lock['source_files'].items():
        accounting.evidence(path, expected, ROOT.parent, refs)


def once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def declared_runtime(lock, refs):
    accounting.require(sys.dont_write_bytecode is True,
                       'native closure requires bytecode writes disabled')
    source_inventory(lock, refs)
    # The inventory rejects added caches before this source loader executes.
    spec = importlib.util.spec_from_file_location('frozen_native_custody', ROOT / 'source/common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.check(ROUND_SHA)
    inventory = common.verify_runtime(lock)
    for path, row in inventory['files'].items():
        accounting.require(path not in refs or refs[path] == row['sha256'],
                           'native runtime evidence bindings disagree')
        refs[path] = row['sha256']


def reconcile(seal=False):
    refs = {}
    accounting.evidence(ROOT / 'round.lock.json', ROUND_SHA, ROOT.parent, refs)
    lock = accounting.read(ROOT / 'round.lock.json')
    accounting.claims(lock)
    accounting.require(lock['allowed_gpu_hosts'] == ['xbabe1'] and lock['full_campaign_admitted'] is False
                       and lock['shared_kpi_used_for_selection'] is False and lock['native_kpis_cross_ranked'] is False
                       and lock['new_trials_planned'] == len(lock['jobs']) == 704
                       and lock['reused_prior_trials'] == len(lock['prior_trials']) == 96
                       and lock['trial_cap_per_method_dataset'] == 8
                       and lock['total_seconds_cap_per_method_dataset'] == 43200,
                       'native population scope or compute budget changed')
    source_inventory(lock, refs)
    for path, expected in lock['frozen_references'].items():
        accounting.evidence(path, expected, ROOT.parent, refs)
    parent = ROOT.parent / 'dope-s3-population-research-v2' / 'round.lock.json'
    accounting.require(str(parent) in refs, 'matched DOPE workers are not bound')
    workers = {j['dataset']: j['worker'] for j in accounting.read(parent)['jobs']}
    accounting.require(len(workers) == 100, 'matched native dataset coverage changed')
    verified_lock = lock | {'round_sha256': ROUND_SHA}
    closed = [row for job in lock['jobs'] if (row := accounting.trial(job, ROOT, verified_lock, refs)) is not None]
    prior = [accounting.prior_trial(row, verified_lock, workers[row['dataset']], ROOT.parent, refs)
             for row in lock['prior_trials']]
    failures = failed_history(lock, ROOT.parent, refs)
    identities = [(r['dataset'], r['method'], r['trial_index']) for r in lock['logical_trials']]
    accounting.require(len(identities) == len(set(identities)) == 800
                       and set(identities) == {(d, m, t) for d in workers for m in ('CTGAN', 'TVAE') for t in range(4)},
                       'native logical tuning matrix incomplete or duplicated')
    jobs = {digest(j): j for j in lock['jobs']}
    accounting.require(len(jobs) == 704, 'native physical tuning jobs duplicated')
    prior_specs = {(r['dataset'], r['method'], r['trial_index']): r for r in lock['prior_trials']}
    for row in lock['logical_trials']:
        identity = (row['dataset'], row['method'], row['trial_index'])
        if row['kind'] == 'immutable_prior_native_trial_reuse':
            accounting.require(row == prior_specs[identity], 'historical logical trial binding changed')
        else:
            accounting.require(row['kind'] == 'new_native_trial' and row['physical_job_sha256'] in jobs
                               and identity == tuple(jobs[row['physical_job_sha256']][k]
                                                     for k in ('dataset', 'method', 'trial_index')),
                               'new logical trial binding changed')
    pairs = [(d, m) for d in workers for m in ('CTGAN', 'TVAE')]
    selections = accounting.select(prior + closed, lock['native_objectives'], pairs, failures)
    report = {'format': 'dope-sdv-population-native-reconciliation', 'version': 1,
        'utc': datetime.now(timezone.utc).isoformat(), 'round_sha256': ROUND_SHA,
        'datasets': 100, 'logical_tuning_trials': 800, 'new_trials_planned': 704,
        'new_trials_closed': len(closed), 'new_trial_status_counts': dict(Counter(r['status'] for r in closed)),
        'prior_trials_reused': 96, 'previous_failed_trials': failures,
        'complete_native_matrix': len(closed) == 704,
        'trials': prior + closed, 'default_native_cells': selections,
        'cost': {'new_operation_seconds': sum(r['operation_seconds'] for r in closed),
                 'prior_native_trial_wall_seconds': sum(r['operation_seconds'] for r in prior),
                 'previous_failed_operation_seconds': sum(r['failed_operation_seconds'] for r in failures),
                 'energy_attributable_to_job': None},
        'native_kpis_never_ranked_across_methods': True, 'shared_kpi_used_for_selection': False,
        'sampling_success_used_for_selection': False, 'historical_runtime_closure_upgraded': False,
        'shared_validation_evaluation_complete': False, 'full_campaign_admitted': False,
        'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
        'release_safe': None, 'superiority': None, 'counts_as_dope_win': False,
        'accounting_source_sha256': sha256(Path(accounting.__file__)),
        'reconciliation_source_sha256': sha256(Path(__file__))}
    accounting.require(abs(report['cost']['prior_native_trial_wall_seconds']
                           - lock['prior_native_trial_wall_seconds']) <= 1e-9,
                       'historical native trial cost changed')
    if seal:
        accounting.require(not (ROOT / 'receipt-lock-v1.json').exists()
                           and not (ROOT / 'reconciliation-v1.json').exists(), 'native custody already sealed')
        report['cost']['coordinator_wall_seconds'] = accounting.closure(ROOT, verified_lock, closed, refs)
        declared_runtime(lock, refs)
        for name in ('execution.lock.json', 'freeze-proof.json', 'runtime-inventory.lock.json',
                     'transport-execution-v1.lock.json', 'transport-bootstrap-v1.py'):
            path = accounting.safe(ROOT / name, ROOT.parent)
            accounting.evidence(path, sha256(path), ROOT.parent, refs)
        transport = accounting.read(ROOT / 'transport-execution-v1.lock.json')
        supervisor = accounting.read(ROOT / 'supervisor-launch.json')
        accounting.require(transport['round_sha256'] == supervisor['round_sha256'] == ROUND_SHA
                           and transport['permitted_gpu_hosts'] == ['xbabe1']
                           and supervisor['gpu_host'] == 'xbabe1'
                           and supervisor['transport_sha256'] == refs[str(ROOT / 'transport-execution-v1.lock.json')]
                           and supervisor['source_sha256'] == transport['launcher_sha256']
                           and transport['bootstrap_sha256'] == refs[str(ROOT / 'transport-bootstrap-v1.py')]
                           and accounting.read(ROOT / 'execution.lock.json')['round_sha256'] == ROUND_SHA,
                           'native launch transport or interpreter bootstrap changed')
        for module in (accounting, __import__(__name__, fromlist=[''])):
            path = Path(module.__file__)
            dest = ROOT / ('custody-' + path.name)
            with dest.open('xb') as stream:
                stream.write(path.read_bytes())
            accounting.evidence(dest, sha256(path), ROOT.parent, refs)
        once(ROOT / 'reconciliation-v1.json', report)
        once(ROOT / 'receipt-lock-v1.json', {'format': 'dope-sdv-population-native-receipt-lock', 'version': 1,
            'round_sha256': ROUND_SHA, 'refs': refs,
            'reconciliation_sha256': sha256(ROOT / 'reconciliation-v1.json'),
            'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    report = reconcile(args.seal)
    print(json.dumps({k: v for k, v in report.items() if k not in ('trials', 'default_native_cells', 'previous_failed_trials')},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
