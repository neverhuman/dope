"""Prepare native retry/reuse work from complete SDV research custody.

No model is loaded, winner selected, capacity reserved, or job launched.
Use build() for externally anchored disk evidence; plan() accepts already
verified metadata for deterministic planning and hermetic controls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

from .manifest import digest
from . import sdv_native_accounting as accounting
from . import reconcile_sdv_population as predecessor
from .sdv_native_accounting import native_metric
from .score import sha256


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def plan(lock, report, workers):
    require(report['complete_native_matrix'] is True
            and report['new_trials_closed'] == report['new_trials_planned'] == 704
            and report['logical_tuning_trials'] == 800
            and report['prior_trials_reused'] == 96
            and report['datasets'] == 100,
            'continuation requires the complete predecessor ledger')
    require(report['official_tests_opened'] is False
            and report['shared_kpi_used_for_selection'] is False
            and report['sampling_success_used_for_selection'] is False
            and report['mfs_v2'] is None and report['ptf_v1'] is None
            and report['release_safe'] is None and report['superiority'] is None,
            'predecessor acquired a test or selection claim')
    require(lock['trial_cap_per_method_dataset'] == 8
            and lock['total_seconds_cap_per_method_dataset'] == 43200
            and lock['whole_operation_timeout_seconds'] == 600,
            'continuation cannot loosen trial or cost caps')
    identities = {(r['dataset'], r['method'], r['trial_index']): r
                  for r in lock['logical_trials']}
    rows = report['trials']
    require(len(identities) == len(lock['logical_trials']) == len(rows) == 800
            and len({(r['dataset'], r['method'], r['trial_index']) for r in rows}) == 800,
            'predecessor trial inventory incomplete or duplicated')
    jobs = {digest(job): job for job in lock['jobs']}
    require(len(jobs) == len(lock['jobs']) == 704, 'predecessor physical jobs changed')
    require(len(workers) == 100 and all(job['worker'] == workers.get(job['dataset'])
            for job in lock['jobs']), 'matched worker coverage changed')
    require(set(lock['native_objectives']) == {'CTGAN', 'TVAE'}
            and all(spec['default_config'] == spec['configurations'][0]
                    and len(spec['configurations']) == 4
                    for spec in lock['native_objectives'].values()),
            'native default or four-trial search grid changed')
    groups = defaultdict(list)
    for row in rows:
        key = (row['dataset'], row['method'], row['trial_index'])
        require(key in identities and row['kind'] == identities[key]['kind'],
                'predecessor logical trial lineage changed')
        specification = identities[key]
        if row['kind'] == 'new_native_trial':
            source = jobs.get(specification['physical_job_sha256'])
            require(source is not None and row['job_sha256'] == specification['physical_job_sha256']
                    and key == tuple(source[k] for k in ('dataset', 'method', 'trial_index')),
                    'predecessor physical trial identity changed')
            configuration = source['config']
        else:
            require(row['kind'] == 'immutable_prior_native_trial_reuse'
                    and row['native_selection_eligible'] is True,
                    'immutable reused native trial acquired a retry')
            configuration = specification['config']
        require(type(row['trial_index']) is int and 0 <= row['trial_index'] < 4
                and configuration == lock['native_objectives'][row['method']]['configurations'][row['trial_index']]
                and row['config'] == configuration
                and row['status'] in ('ok', 'failed') and row['fit_seed'] == 11
                and finite(row['operation_seconds'])
                and type(row['complete_fitted_artifact']) is bool
                and type(row['native_selection_eligible']) is bool
                and type(row['common_samples_complete']) is bool,
                'predecessor configuration, cost or stage identity changed')
        require(row['native_selection_eligible'] == (row['native_kpi'] is not None)
                and (not row['native_selection_eligible'] or row['complete_fitted_artifact'])
                and (not row['common_samples_complete'] or row['native_selection_eligible']),
                'partial predecessor fit acquired native eligibility')
        if row['native_selection_eligible']:
            native_metric(row['native_kpi'], workers[row['dataset']],
                          lock['native_objectives'][row['method']]['native_objective'], lock)
        groups[key[:2]].append(row)
    require(set(groups) == {(d, m) for d in workers for m in ('CTGAN', 'TVAE')}
            and all({r['trial_index'] for r in group} == {0, 1, 2, 3} for group in groups.values()),
            'predecessor dataset-method groups changed')
    history = report['previous_failed_trials']
    require(len(history) == len(lock['previous_failed_trials']) == 2,
            'older failed native attempts omitted')
    expected_failures = {r['source_job_sha256']: r for r in lock['previous_failed_trials']}
    require(len(expected_failures) == len({r['source_job_sha256'] for r in history}) == 2,
            'older failed attempt identity duplicated')
    failed = defaultdict(list)
    for row in history:
        expected = expected_failures.get(row['source_job_sha256'])
        require(expected is not None and all(row[k] == v for k, v in expected.items())
                and row['status'] == 'failed' and finite(row['failed_operation_seconds'])
                and (row['dataset'], row['method']) in groups,
                'older failed native identity or cost changed')
        failed[(row['dataset'], row['method'])].append(row)
    cells = []
    for (dataset, method), group in sorted(groups.items()):
        group.sort(key=lambda r: r['trial_index'])
        charged_attempts = 4 + len(failed[(dataset, method)])
        seconds = sum(r['operation_seconds'] for r in group) + sum(
            r['failed_operation_seconds'] for r in failed[(dataset, method)])
        require(charged_attempts <= 8 and seconds <= 43200, 'predecessor already exceeds per-cell caps')
        remaining_attempts = min(8 - charged_attempts, int((43200 - seconds) // 600))
        retries, unavailable = [], []
        eligible = []
        for row in group:
            if row['native_selection_eligible']:
                eligible.append({'trial_index': row['trial_index'], 'receipt_path': row['receipt_path'],
                    'receipt_sha256': row['receipt_sha256'], 'native_value': row['native_kpi']['value'],
                    'common_samples_complete': row['common_samples_complete'],
                    'new_fit_needed': False, 'sampling_success_used_for_selection': False,
                    'common_sampling_requires_gpu': False})
                continue
            if remaining_attempts == 0:
                unavailable.append({'trial_index': row['trial_index'], 'reason': 'remaining_attempt_or_cost_budget'})
                continue
            kind = 'native_on_complete_fitted_artifact' if row['complete_fitted_artifact'] else 'fit_then_native'
            specification = identities[(dataset, method, row['trial_index'])]
            identity = specification.get('physical_job_sha256')
            require(identity == row['job_sha256'], 'retry lost the original fit identity')
            retries.append({'trial_index': row['trial_index'], 'fit_job_sha256': identity,
                'predecessor_attempt': 1, 'continuation_attempt': 2, 'operation': kind,
                'new_fit_needed': not row['complete_fitted_artifact'],
                'requires_gpu': not row['complete_fitted_artifact'],
                'native_scoring_requires_gpu': False,
                'prior_receipt_path': row['receipt_path'], 'prior_receipt_sha256': row['receipt_sha256'],
                'maximum_whole_operation_seconds': 600, 'attempt_counts_toward_eight': True})
            remaining_attempts -= 1
        cells.append({'dataset': dataset, 'method': method,
            'previous_attempts_charged': charged_attempts, 'previous_seconds_charged': seconds,
            'native_evidence_reused_without_refit': eligible, 'native_retry_previews': retries,
            'budget_unavailable_trials': unavailable,
            'winner_selection_deferred_until_native_retries_close': True,
            'common_samples_after_native_selection_only': ['default', 'native_tuned'],
            'common_sampling_requires_gpu': False,
            'maximum_additional_seconds_reserved': 600 * len(retries)})
    return {'format': 'dope-sdv-postclosure-continuation-plan-preview', 'version': 1,
        'method_dataset_cells': cells, 'datasets': 100, 'method_dataset_pairs': 200,
        'tuning_grid_changed': False, 'shared_kpi_used_for_selection': False,
        'native_kpis_cross_ranked': False, 'predecessor_failures_relabelled': False,
        'physical_jobs_launched': 0, 'execution_admitted': False,
        'runtime_monitor_or_owner_registry_verified': False,
        'counts_as_dope_win': False,
        'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
        'release_safe': None, 'superiority': None}


def build(receipt_sha256):
    """Recheck closed custody before producing an unadmitted work preview.

    The receipt-lock digest must come from the frozen closure record outside
    these files. The live round has no default or inferred receipt digest.
    """
    require(isinstance(receipt_sha256, str)
            and re.fullmatch(r'[0-9a-f]{64}', receipt_sha256) is not None,
            'externally frozen receipt-lock digest required')
    root, base = predecessor.ROOT, predecessor.ROOT.parent
    refs = {}
    accounting.evidence(root / 'round.lock.json', predecessor.ROUND_SHA, base, refs)
    accounting.evidence(root / 'receipt-lock-v1.json', receipt_sha256, base, refs)
    anchor = accounting.read(root / 'receipt-lock-v1.json')
    accounting.claims(anchor)
    require(anchor['format'] == 'dope-sdv-population-native-receipt-lock'
            and anchor['version'] == 1 and anchor['round_sha256'] == predecessor.ROUND_SHA,
            'predecessor custody scope changed')
    for path, expected in anchor['refs'].items():
        accounting.evidence(path, expected, base, refs)
    report_path = root / 'reconciliation-v1.json'
    accounting.evidence(report_path, anchor['reconciliation_sha256'], base, refs)
    lock = accounting.read(root / 'round.lock.json')
    matched = base / 'dope-s3-population-research-v2' / 'round.lock.json'
    require(str(matched) in anchor['refs']
            and anchor['refs'][str(matched)] == lock['frozen_references'][str(matched)],
            'matched worker manifest is not bound to predecessor custody')
    workers = {job['dataset']: job['worker'] for job in accounting.read(matched)['jobs']}
    report = accounting.read(report_path)
    require(report['round_sha256'] == predecessor.ROUND_SHA, 'predecessor report round changed')
    result = plan(lock, report, workers)
    for row in report['trials']:
        require(anchor['refs'].get(row['receipt_path']) == row['receipt_sha256'],
                'trial receipt differs from frozen custody')
    new_rows = [row for row in report['trials'] if row['kind'] == 'new_native_trial']
    seconds = accounting.closure(root, lock | {'round_sha256': predecessor.ROUND_SHA}, new_rows, refs)
    require(seconds == report['cost']['coordinator_wall_seconds'], 'closed coordinator cost changed')
    # This standard-library verifier rejects added source/runtime aliases and
    # caches before loading its frozen source. It never imports ML libraries.
    predecessor.declared_runtime(lock, refs)
    for worker in workers.values():
        accounting.worker_files(worker, base, refs)
    return result | {'predecessor_round_sha256': predecessor.ROUND_SHA,
                     'predecessor_receipt_lock_sha256': receipt_sha256,
                     'predecessor_reconciliation_sha256': anchor['reconciliation_sha256'],
                     'planner_source_sha256': sha256(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-lock-sha256', required=True)
    args = parser.parse_args()
    result = build(args.receipt_lock_sha256)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
