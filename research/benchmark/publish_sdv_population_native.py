"""Publish the complete closed native ledger; shared quality remains unmeasured."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from statistics import median

from . import sdv_native_accounting as accounting
from . import sdv_native_continuation as continuation
from . import reconcile_sdv_population as predecessor
from .manifest import digest
from .score import sha256

RESULTS = Path(__file__).with_name('results')
NAME = 'sdv-s3-population-native'
GATES = {'official_tests_opened': False, 'production_certified': False,
         'full_campaign_admitted': False, 'shared_validation_evaluation_complete': False,
         'shared_kpi_used_for_selection': False, 'sampling_success_used_for_selection': False,
         'native_kpis_cross_ranked': False, 'historical_runtime_closure_upgraded': False,
         'counts_as_dope_win': False, 'mfs_v2': None, 'ptf_v1': None,
         'release_safe': None, 'superiority': None}


def bound_json(path, expected):
    """Parse the same owned bytes whose externally frozen digest was checked."""
    accounting.require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected),
                       'externally frozen digest required')
    path = accounting.safe(path, predecessor.ROOT.parent)
    accounting.require(path.is_file(), 'native publication reference is not a file')
    data = path.read_bytes()
    accounting.require(hashlib.sha256(data).hexdigest() == expected,
                       'native publication reference changed')
    return json.loads(data)


def native(row):
    metric = row['native_kpi']
    if metric is None:
        return None
    return {k: metric[k] for k in ('objective', 'direction', 'value', 'components',
                                   'implementation_sha256', 'validation_sha256', 'seed')}


def outcome(terminal):
    """Separate scheduler/transport outcomes; never infer a method failure."""
    status, limit = terminal['status'], terminal.get('timeout_seconds')
    if status in ('ok', 'immutable_prior_reuse'):
        return status, status
    if status == 'deadline_unstarted':
        return 'scheduling_cutoff', 'deadline_unstarted'
    # The frozen coordinator uses min(600, remaining(round deadline)); its
    # shortened final transport operation is an infrastructure cut-off.
    if status == 'transport_or_prelaunch_failure' and type(limit) is int and 0 < limit < 600:
        return 'scheduling_cutoff', 'deadline_truncated'
    if status in ('transport_or_prelaunch_failure', 'foreign_gpu_owner_appeared'):
        return 'infrastructure_interruption', status
    return 'unclassified_unavailable', status


def trial(row, terminal):
    classification, reason = outcome(terminal)
    return {k: row[k] for k in ('dataset', 'method', 'trial_index', 'kind', 'fit_seed',
        'config', 'artifact_bytes', 'complete_fitted_artifact',
        'native_selection_eligible', 'common_samples_complete', 'operation_seconds')} | {
        'receipt_status': row['status'], 'outcome_class': classification,
        'outcome_reason': reason, 'method_failure_inferred': False,
        'config_sha256': digest(row['config']), 'terminal_operation_status': terminal['status'],
        'terminal_timeout_seconds': terminal.get('timeout_seconds'),
        'receipt': {'path': row['receipt_path'], 'sha256': row['receipt_sha256']},
        'native_kpi': native(row), 'counts_as_dope_win': False}


def assemble(lock, report, terminals):
    """Accept already verified metadata; recompute selection and all aggregates."""
    accounting.require(report['complete_native_matrix'] is True
        and report['datasets'] == 100 and report['logical_tuning_trials'] == 800
        and report['new_trials_closed'] == report['new_trials_planned'] == 704
        and report['prior_trials_reused'] == 96,
        'publication requires the complete native matrix')
    for key, value in GATES.items():
        if key in report:
            accounting.require(report[key] is value, 'native publication acquired a claim')
    rows = report['trials']
    ids = [(r['dataset'], r['method'], r['trial_index']) for r in rows]
    datasets = sorted({r['dataset'] for r in rows})
    accounting.require(len(ids) == len(set(ids)) == 800 and len(datasets) == 100
        and set(ids) == {(d, m, t) for d in datasets for m in ('CTGAN', 'TVAE') for t in range(4)}
        and set(terminals) == set(ids), 'native publication trial coverage changed')
    selections = accounting.select(rows, lock['native_objectives'],
        [(d, m) for d in datasets for m in ('CTGAN', 'TVAE')], report['previous_failed_trials'])
    accounting.require(digest(selections) == digest(report['default_native_cells']),
                       'native publication selection differs from sealed evidence')
    trials = [trial(r, terminals[(r['dataset'], r['method'], r['trial_index'])])
              for r in sorted(rows, key=lambda r: (r['dataset'], r['method'], r['trial_index']))]
    by_id = {(r['dataset'], r['method'], r['trial_index']): r for r in trials}
    cells = []
    for cell in selections:
        reference = lambda r: None if r is None else by_id[(r['dataset'], r['method'], r['trial_index'])]
        cells.append({k: cell[k] for k in ('dataset', 'method', 'status', 'all_four_trials_closed',
            'tuning_trials_with_previous_failures', 'operation_seconds_with_previous_failures')} | {
            'default_trial_index': 0,
            'default_native_kpi': native(cell['default_trial']) if cell['default_trial'] else None,
            'native_selected_trial_index': cell['native_selected_trial']['trial_index']
                if cell['native_selected_trial'] else None,
            'default_charged_artifact_bytes': reference(cell['default_trial'])['artifact_bytes'],
            'selected_charged_artifact_bytes': reference(cell['native_selected_trial'])['artifact_bytes']
                if cell['native_selected_trial'] else None,
            'selected_native_kpi': native(cell['native_selected_trial']) if cell['native_selected_trial'] else None,
            'counts_as_dope_win': False})
    summaries = []
    for method in ('CTGAN', 'TVAE'):
        group = [c for c in cells if c['method'] == method]
        paired = [c for c in group if c['default_native_kpi'] is not None and c['selected_native_kpi'] is not None]
        summaries.append({'method': method, 'datasets': 100,
            'native_selected_cells': sum(c['status'] == 'selected' for c in group),
            'native_selection_unavailable_cells': sum(c['status'] == 'native_selection_unavailable' for c in group),
            'paired_default_selected_cells': len(paired),
            'paired_default_median_native_r2': median(c['default_native_kpi']['value'] for c in paired) if paired else None,
            'paired_selected_median_native_r2': median(c['selected_native_kpi']['value'] for c in paired) if paired else None,
            'paired_median_native_r2_change': median(c['selected_native_kpi']['value'] - c['default_native_kpi']['value']
                                                    for c in paired) if paired else None})
    new = [r for r in rows if r['kind'] == 'new_native_trial']
    prior = [r for r in rows if r['kind'] == 'immutable_prior_native_trial_reuse']
    counts = dict(Counter(r['status'] for r in new))
    accounting.require(counts == report['new_trial_status_counts'] and len(prior) == 96
        and len(new) == 704 and all(c['all_four_trials_closed'] is True for c in cells),
        'native publication status accounting changed')
    for key, values in (('new_operation_seconds', new), ('prior_native_trial_wall_seconds', prior)):
        accounting.require(abs(sum(r['operation_seconds'] for r in values) - report['cost'][key]) <= 1e-9,
                           'native publication cost differs')
    return {'format': 'dope-sdv-population-native-publication', 'version': 1,
        'scope': 'Complete bounded native research ledger; training-derived validation only; no shared test-quality comparison',
        'datasets': 100, 'method_dataset_cells': 200, 'logical_tuning_trials': 800,
        'new_physical_attempts': 704, 'prior_trials_reused': 96,
        'all_frozen_cells_accounted': True, 'fit_seeds': [11],
        'native_sample_seed': 101, 'native_metric_seed': 1729,
        'raw_new_receipt_status_counts': counts,
        'new_outcome_class_counts': dict(Counter(r['outcome_class'] for r in trials
                                                   if r['kind'] == 'new_native_trial')),
        'new_scheduling_cutoff_reason_counts': dict(Counter(r['outcome_reason'] for r in trials
            if r['kind'] == 'new_native_trial' and r['outcome_class'] == 'scheduling_cutoff')),
        'method_failure_conclusions_drawn': False,
        'new_terminal_operation_status_counts': dict(Counter(r['terminal_operation_status'] for r in trials
                                                               if r['kind'] == 'new_native_trial')),
        'cell_status_counts': dict(Counter(c['status'] for c in cells)),
        'trials': trials, 'cells': cells, 'within_method_summaries': summaries,
        'previous_failed_trials': [{k: r[k] for k in ('dataset', 'method', 'trial_index',
            'source_job_sha256', 'receipt_path', 'receipt_sha256',
            'failed_operation_seconds', 'partial_artifact_bytes')} | {
            'receipt_status': r['status'], 'method_failure_inferred': False}
            for r in report['previous_failed_trials']],
        'cost': report['cost'], 'native_objectives': lock['native_objectives'],
        'limitations': ['One fit seed; no fit uncertainty or paired superiority analysis',
            'Native R2 values are unclipped and never ranked across methods',
            'Scheduling cut-offs and infrastructure interruptions remain in every coverage denominator',
            'Raw failed receipt status is not a method-failure classification',
            'Native eligibility survives a later common-sampling failure',
            'Artifact bytes include projection; unconstrained baselines have no L3 release claim',
            'Historical reused evidence retains its original runtime limitations'], **GATES}


def build(receipt_sha256, report_sha256):
    # Validate both external anchors before the verifier can read any metrics.
    for value in (receipt_sha256, report_sha256):
        accounting.require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value),
                           'externally frozen digest required')
    root = predecessor.ROOT
    anchor = bound_json(root / 'receipt-lock-v1.json', receipt_sha256)
    accounting.require(anchor['reconciliation_sha256'] == report_sha256,
                       'external native reconciliation digest differs')
    continuation.build(receipt_sha256)  # Complete recursive source/runtime/worker custody, no ML imports.
    lock = bound_json(root / 'round.lock.json', predecessor.ROUND_SHA)
    report = bound_json(root / 'reconciliation-v1.json', report_sha256)
    # Also reject added files/aliases and independently recompute artifact charges,
    # phase costs and native eligibility; hashes of existing files alone do not
    # establish a complete directory inventory.
    replay = predecessor.reconcile(seal=False)
    without_clock = lambda r: {k: v for k, v in r.items() if k not in ('utc', 'cost')}
    accounting.require(digest(without_clock(replay)) == digest(without_clock(report))
        and replay['cost'] == {k: v for k, v in report['cost'].items() if k != 'coordinator_wall_seconds'},
        'fresh native accounting differs from the sealed ledger')
    for module, key in ((accounting, 'accounting_source_sha256'), (predecessor, 'reconciliation_source_sha256')):
        accounting.require(sha256(Path(module.__file__)) == report[key], 'native verifier source differs')
    terminals, operations = {}, {}
    for row in report['trials']:
        accounting.require(anchor['refs'].get(row['receipt_path']) == row['receipt_sha256'],
                           'native trial differs from frozen receipt lock')
        receipt = bound_json(row['receipt_path'], row['receipt_sha256'])
        if row['kind'] == 'new_native_trial':
            accounting.require(digest(receipt['job']) == row['job_sha256']
                and receipt['status'] == row['status'], 'native trial receipt identity differs')
            terminal = receipt['operations'][-1]['status']
            phases = []
            for phase, operation in zip(('fit', 'native', 'sample'), receipt['operations']):
                path = str(Path(row['receipt_path']).parent / (phase + '.monitor.json'))
                monitor = bound_json(path, anchor['refs'][path]) if path in anchor['refs'] else {}
                phases.append({'phase': phase, 'status': operation['status'],
                    'new_operation_started': operation['new_operation_started'],
                    'elapsed_seconds': operation['elapsed_seconds'], 'host': operation.get('host'),
                    'peak_resident_bytes': monitor.get('peak_resident_bytes'),
                    'peak_gpu_used_mib': monitor.get('peak_gpu_used_mib'),
                    'gpu_process_observed': monitor.get('gpu_process_observed'),
                    'monitor_sha256': anchor['refs'].get(path), 'energy_attributable_to_job': None})
            if row['native_kpi'] is not None:
                path = str(Path(row['receipt_path']).parent / 'native.json')
                accounting.require(bound_json(path, anchor['refs'][path]) == row['native_kpi'],
                                   'native trial metric differs from receipt')
        else:
            accounting.require(receipt['native_kpi'] == row['native_kpi'] and receipt['status'] == row['status'],
                               'historical native metric differs')
            terminal = 'immutable_prior_reuse'
            phases = [{'phase': 'historical_trial', 'status': receipt['status'],
                'new_operation_started': False, 'elapsed_seconds': receipt['wall_seconds'],
                'host': receipt['host'], 'peak_resident_bytes': None,
                'peak_gpu_used_mib': receipt['fit']['peak_device_used_mib'],
                'gpu_process_observed': None, 'monitor_sha256': None,
                'energy_attributable_to_job': None}]
        terminals[(row['dataset'], row['method'], row['trial_index'])] = {
            'status': terminal, 'timeout_seconds': receipt['operations'][-1].get('timeout_seconds')
            if row['kind'] == 'new_native_trial' else None}
        operations[(row['dataset'], row['method'], row['trial_index'])] = phases
    result = assemble(lock, report, terminals)
    for row in result['trials']:
        row['operation_evidence'] = operations[(row['dataset'], row['method'], row['trial_index'])]
    matched = root.parent / 'dope-s3-population-research-v2' / 'round.lock.json'
    workers = {j['dataset']: j['worker'] for j in bound_json(matched, anchor['refs'][str(matched)])['jobs']}
    result['dataset_lineages'] = [{k: w[k] for k in ('dataset', 'catalog_entry_sha256',
        'source_manifest_sha256', 'preparation_receipt_sha256', 'object_sha256',
        'raw_training_derived_split_hashes', 'files', 'train_rows', 'projected_features')}
        for _, w in sorted(workers.items())]
    result['source_locks'] = {'round': predecessor.ROUND_SHA, 'receipts': receipt_sha256,
                             'reconciliation': report_sha256}
    result['publisher_sources'] = {Path(m.__file__).name: sha256(Path(m.__file__))
        for m in (accounting, continuation, predecessor, __import__(__name__, fromlist=['']))}
    bound_json(root / 'receipt-lock-v1.json', receipt_sha256)
    bound_json(root / 'reconciliation-v1.json', report_sha256)
    return result


def table(report):
    out = io.StringIO(newline='')
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(['dataset', 'method', 'status', 'default_native_r2', 'selected_native_r2',
                     'selected_trial', 'default_bytes', 'selected_bytes', 'attempts', 'operation_seconds'])
    for c in report['cells']:
        writer.writerow([c['dataset'], c['method'], c['status'],
            c['default_native_kpi']['value'] if c['default_native_kpi'] else '',
            c['selected_native_kpi']['value'] if c['selected_native_kpi'] else '',
            c['native_selected_trial_index'] if c['native_selected_trial_index'] is not None else '',
            c['default_charged_artifact_bytes'], c['selected_charged_artifact_bytes'],
            c['tuning_trials_with_previous_failures'], c['operation_seconds_with_previous_failures']])
    return out.getvalue()


def markdown(report):
    lines = ['# CTGAN / TVAE complete native research ledger', '', report['scope'], '',
        '100 matched S3 lineages, 800 logical trials (704 new attempts, 96 historical reuses).',
        'All trials are accounted for; this does not mean all fits succeeded.', '',
        '| Method | Native selected / 100 | Unavailable / 100 | Paired default–selected | Default median native R² | Selected median native R² |',
        '|---|---:|---:|---:|---:|---:|']
    for s in report['within_method_summaries']:
        lines.append(f"| {s['method']} | {s['native_selected_cells']} | {s['native_selection_unavailable_cells']} | "
            f"{s['paired_default_selected_cells']} | {s['paired_default_median_native_r2']:.6g} | "
            f"{s['paired_selected_median_native_r2']:.6g} |")
    lines += ['', 'Medians use the same paired cells within each method; native KPIs are not cross-method ranks.', '',
        'New attempt terminal statuses: ' + ', '.join(f'{k}: {v}' for k, v in
                                                      sorted(report['new_terminal_operation_status_counts'].items())) + '.', '',
        'Scheduling outcomes: ' + ', '.join(f'{k}: {v}' for k, v in
                                             sorted(report['new_outcome_class_counts'].items())) + '.',
        'The 524 unstarted and one deadline-truncated attempt are scheduling cut-offs (infrastructure),',
        'separate from the 154 successful attempts and 25 earlier infrastructure interruptions.',
        'Immutable raw receipt statuses and two previous-round failed attempts remain in cost accounting.',
        'No method-failure or method-quality conclusion is drawn from scheduling or infrastructure outcomes.',
        'Official tests remain sealed. Shared validation quality is pending; MFS-v2, PTF-v1,',
        'release-safe status and superiority are null. No DOPE win or production certification is claimed.', '',
        'Regenerate using `python3 -B -m research.benchmark.publish_sdv_population_native` with',
        'the two external SHA-256 anchors in the committed publication manifest.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-lock-sha256', required=True)
    parser.add_argument('--reconciliation-sha256', required=True)
    parser.add_argument('--output-directory', type=Path, default=RESULTS)
    args = parser.parse_args()
    report = build(args.receipt_lock_sha256, args.reconciliation_sha256)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for suffix, value in (('json', json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n'),
                          ('csv', table(report)), ('md', markdown(report))):
        (args.output_directory / (NAME + '.' + suffix)).write_text(value)


if __name__ == '__main__':
    main()
