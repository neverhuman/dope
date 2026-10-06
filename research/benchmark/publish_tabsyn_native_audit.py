"""Replay the closed TabSyn native audit, preserving inapplicability and failures."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from .publish_tabsyn_population_fits import (
    AUTHOR, CONFIG_SHA, RESULTS, anchored, finite, require, same_cost,
)
from .publish_s3_matched import schema as infer_schema

NAME = 'tabsyn-default100-native-audit'
CACHE = Path(__file__).resolve().parents[2] / 'target/tabsyn-native-ledger-v1/publisher-input-custody'
CACHE_SHA = 'b10dd61c08ae9437fa79fdccf1fe985cdba8a198f99d065e7e93a201df3902de'
PHASE_SHA = '3c6f925a9c015281f5d97ddd8de45674ee6d92b3a8272d4f550a70285eb6e0ce'
FIT_SHA = 'cb2c1c134ccd1af9865e156faee8796143c212d1f76e70ab70f7048f82a08272'
COUNTS = {'native_audit_admitted': 77, 'native_audit_unavailable': 14, 'native_input_unavailable': 9}
INAPPLICABLE = {
    'status': 'tuning_inapplicable_uninformative_author_target',
    'reason': 'constant_float32_author_log_clipped_validation_target',
    'native_value': None, 'native_winner': None, 'default_retained': True,
    'native_track': 'author KPI on TRAIN-projection-clipped source target units',
    'common_projection_units_changed': False, 'outside_fit_extrema_reconstructed': False,
}
NULL_CLAIMS = ('native_value', 'native_winner', 'common_validation', 'privacy',
               'mfs_v2', 'ptf_v1', 'release_safe_l3', 'paired_superiority')
FALSE_CLAIMS = ('generator_tuned_winner_claimed', 'author_gpu_hist_reproduction',
                'native_selection_complete', 'common_validation_complete',
                'production_certified', 'official_tests_opened', 'counts_as_dope_win',
                'current_bulk_model_hashes_reverified', 'full_runtime_hardware_certified')


def native_cell(row, fit, read_ref, sources):
    require(row['dataset'] == fit['dataset_id'] and row['lineage_sha256'] == fit['lineage_sha256']
            and row['config_sha256'] == CONFIG_SHA and fit['fit_seed'] == 11
            and row['fit_request_ref'] == fit['fit_request'], 'native fit identity changed')
    inventory = row['artifact_inventory']
    require(inventory['files'] == fit['artifact_files']
            and inventory['artifact_bytes'] == fit['charged_artifact_bytes']
            and sum(x['bytes'] for x in inventory['files'].values()) == fit['charged_artifact_bytes']
            and inventory['projection_bytes'] == fit['projection_bytes'], 'native model byte charge differs')
    closure = read_ref(row['logical_closure_ref'])
    require(closure['dataset'] == row['dataset'] and closure['lineage_sha256'] == row['lineage_sha256']
            and closure['stage'] == 'native_audit' and closure['official_tests_opened'] is False
            and type(closure['scientific_fit_allocations']) is int and closure['scientific_fit_allocations'] == 0
            and closure['original_lineage_start'] == row['original_lineage_started_unix_seconds'],
            'native logical closure changed')
    same_cost(row['original_lineage_deadline_unix_seconds'] - row['original_lineage_started_unix_seconds'], 43200)
    cell = {'dataset_id': fit['dataset_id'], 'lineage_sha256': fit['lineage_sha256'],
            'fit_seed': 11, 'config_sha256': CONFIG_SHA, 'status': row['status'],
            'native_input_status': row['native_input_status'],
            'charged_artifact_bytes': fit['charged_artifact_bytes'],
            'model_bytes': fit['model_bytes'], 'projection_bytes': fit['projection_bytes'],
            'fit_request': fit['fit_request'], 'logical_closure': row['logical_closure_ref'],
            'native_value': None, 'native_winner': None, 'actual_XGB_fit_calls': 0,
            'actual_worker_exit': None, 'physical_native_attempt': None,
            'measured_whole_seconds': 0, 'prior_failed_native_whole_seconds': 0,
            'prior_failed_native_attempts': 0, 'reason': None,
            'default_retained_by_native_audit': False, 'native_result_admitted': False,
            'original_lineage_started_unix_seconds': row['original_lineage_started_unix_seconds'],
            'original_lineage_deadline_unix_seconds': row['original_lineage_deadline_unix_seconds']}
    if row['status'] == 'native_input_unavailable':
        require(type(row['scientific_native_Popen']) is int and row['scientific_native_Popen'] == 0
                and row['native_input_status'] == 'unavailable_closed_monitor'
                and closure['operations'] == [{'status': 'native_input_unavailable'}],
                'missing input acquired a native result')
        cell['reason'] = 'native_input_monitoring_unavailable'
        return cell
    require(row['status'] in ('native_audit_admitted', 'native_audit_unavailable')
            and type(row['scientific_native_Popen']) is int and row['scientific_native_Popen'] == 1,
            'invalid native audit status')
    request = read_ref(row['request_ref']); popen = read_ref(row['Popen_ref'])
    exit_row = read_ref(row['actual_exit_ref'])
    complete = read_ref(row['observed_physical_native_complete_ref'])
    csv_readback = read_ref(row['physical_csv_readback_ref'])
    calls = read_ref(row['native_auditor_calls_ref'])
    identity = request['learning_identity']
    require(identity['dataset'] == row['dataset'] and identity['source_commit'] == AUTHOR
            and type(identity['fit_seed']) is int and identity['fit_seed'] == 11
            and request['lineage_sha256'] == row['lineage_sha256']
            and request['artifact_code_ref'] == fit['fit_request'] and request['mode'] == 'native'
            and request['config']['config_sha256'] == CONFIG_SHA and request['source_files'] == sources
            and sources[request['entry_ref']['path']] == request['entry_ref']['sha256']
            and request['lineage_started_unix_seconds'] == row['original_lineage_started_unix_seconds']
            and request['sample_backend'] == 'cpu' and request['native_backend'] == 'cpu_hist'
            and request['official_tests_opened'] is False and request['production_certified'] is False
            and request['generated_fixture'] is False and request['scientific_fit_allocations'] == 0,
            'native request source, track or partition changed')
    require(popen['request_ref'] == row['request_ref'] and popen['CUDA_VISIBLE_DEVICES'] == ''
            and popen['scientific_worker_pid'] == exit_row['scientific_worker_pid']
            and popen['start_ticks'] == exit_row['start_ticks']
            and popen['cpu_slot'] == list(range(80, 96)) and popen['nice'] == 19
            and type(exit_row['actual_worker_exit']) is int and exit_row['actual_worker_exit'] == 0
            and exit_row['actual_worker_exit'] == row['actual_worker_exit']
            and exit_row['foreign_signals'] == 0 and exit_row['official_tests_opened'] is False,
            'native process identity or actual exit changed')
    same_cost(exit_row['whole_operation_seconds'], row['measured_whole_operation_seconds'])
    require(complete['status'] == 'ok' and complete['lineage_sha256'] == row['lineage_sha256']
            and complete['config_sha256'] == CONFIG_SHA and complete['mode'] == 'native'
            and complete['native_backend'] == 'cpu_hist' and complete['result'] == INAPPLICABLE
            and complete['author_gpu_hist_reproduction'] is False
            and complete['official_tests_opened'] is False and complete['production_certified'] is False
            and complete['mfs_v2'] is None and complete['ptf_v1'] is None,
            'physical native completion changed')
    require(csv_readback['lineage_sha256'] == row['lineage_sha256']
            and csv_readback['artifact_code_ref'] == fit['fit_request']
            and request['physical_csv_readback_ref'] == row['physical_csv_readback_ref']
            and csv_readback['physical_csv_joint_sha256'] == request['joint_synthetic_sha256']
            and csv_readback['original_tensor_joint_sha256'] == request['original_tensor_joint_sha256']
            and csv_readback['synthetic_ref'] == request['synthetic_ref']
            and csv_readback['dtype'] == 'float64' and csv_readback['order'] == 'C'
            and csv_readback['byteorder'] == 'little' and csv_readback['scientific_values_or_headers_logged'] is False,
            'physical CSV and original tensor identity changed')
    require(calls['request_sha256'] == row['request_ref']['sha256']
            and type(calls['actual_XGB_fit_calls']) is int and calls['actual_XGB_fit_calls'] == 0
            and calls['auditor_configurations'] == 36 and calls['postselection_refits'] == 4
            and calls['native_backend'] == 'cpu_hist' and calls['author_gpu_hist_reproduction'] is False,
            'native author auditor accounting changed')
    admitted = (exit_row['owned_signal_calls'] == 0 and exit_row['owned_termination_reason'] is None
                and exit_row['deadline_rechecked'] is True
                and exit_row['completed_monotonic'] <= exit_row['deadline_monotonic'])
    require(row['owned_signal_calls'] == exit_row['owned_signal_calls']
            and row['owned_termination_reason'] == exit_row['owned_termination_reason']
            and row['deadline_rechecked'] is exit_row['deadline_rechecked']
            and (row['status'] == 'native_audit_admitted') is admitted,
            'worker exit zero cannot replace native monitoring admission')
    if admitted:
        require(row['native_result'] == INAPPLICABLE and row['actual_XGB_fit_calls'] == 0,
                'uninformative objective replaced or tuned')
    else:
        require('native_result' not in row, 'unavailable native result upgraded')
    cell.update(actual_worker_exit=0, physical_native_attempt=row['physical_native_attempt'],
                measured_whole_seconds=row['measured_whole_operation_seconds'],
                native_result_admitted=admitted, default_retained_by_native_audit=admitted,
                reason=INAPPLICABLE['reason'] if admitted else 'native_audit_monitoring_unavailable',
                owned_signal_calls=exit_row['owned_signal_calls'], deadline_rechecked=exit_row['deadline_rechecked'],
                native_request=row['request_ref'], native_Popen=row['Popen_ref'], native_exit=row['actual_exit_ref'],
                physical_completion=row['observed_physical_native_complete_ref'],
                physical_csv_readback=row['physical_csv_readback_ref'], auditor_calls=row['native_auditor_calls_ref'])
    retry = row.get('retry_of')
    if retry is not None:
        require(type(retry['actual_worker_exit']) is int and retry['actual_worker_exit'] == 2
                and retry['lineage_sha256'] == row['lineage_sha256'], 'prior failure identity changed')
        cell.update(prior_failed_native_attempts=1,
                    prior_failed_native_whole_seconds=finite(retry['whole_operation_seconds']),
                    prior_failed_native_receipts=retry)
    return cell


def summarize(cells):
    require(len(cells) == len({x['dataset_id'] for x in cells}) == len({x['lineage_sha256'] for x in cells}) == 100,
            'incomplete or duplicate native audit population')
    counts = dict(Counter(x['status'] for x in cells))
    require(counts == COUNTS and all(x['native_value'] is None and x['native_winner'] is None
                and x['actual_XGB_fit_calls'] == 0 for x in cells), 'native coverage or objective changed')
    for row in cells:
        admitted = row['status'] == 'native_audit_admitted'
        missing = row['status'] == 'native_input_unavailable'
        require(re.fullmatch('[0-9a-f]{16}', row['dataset_id']) is not None
                and row['native_result_admitted'] is admitted
                and row['default_retained_by_native_audit'] is admitted
                and row['config_sha256'] == CONFIG_SHA and type(row['fit_seed']) is int and row['fit_seed'] == 11
                and type(row['charged_artifact_bytes']) is int
                and row['charged_artifact_bytes'] == row['model_bytes'] + row['projection_bytes']
                and (row['actual_worker_exit'] is None if missing else
                     type(row['actual_worker_exit']) is int and row['actual_worker_exit'] == 0),
                'native cell status, byte charge or admission changed')
    return {'logical_lineages_closed': 100, 'status_counts': counts,
            'native_objective_inapplicable': sum(x['native_result_admitted'] for x in cells),
            'physical_workers_exit_zero': sum(type(x['actual_worker_exit']) is int and x['actual_worker_exit'] == 0 for x in cells),
            'informative_native_objectives': 0, 'actual_XGB_fit_calls': 0, 'generator_tuning_trials': 0,
            'measured_current_native_whole_seconds': sum(finite(x['measured_whole_seconds']) for x in cells),
            'prior_failed_native_attempts': sum(x['prior_failed_native_attempts'] for x in cells),
            'prior_failed_native_whole_seconds': sum(finite(x['prior_failed_native_whole_seconds']) for x in cells),
            'charged_model_and_projection_bytes': sum(x['charged_artifact_bytes'] for x in cells)}


def build():
    lock, buffers = anchored(CACHE, CACHE_SHA)
    def read_ref(ref):
        item = lock['metadata'][ref['path']]
        require(item['sha256'] == ref['sha256'], 'native frozen receipt reference changed')
        return json.loads(buffers[item['file']])
    require(lock['phase_ref']['sha256'] == PHASE_SHA and lock['fit_report_ref']['sha256'] == FIT_SHA,
            'native publication anchors changed')
    phase = read_ref(lock['phase_ref']); fit_report = read_ref(lock['fit_report_ref'])
    sources = {path: item['sha256'] for path, item in lock['source_files'].items()}
    require(len(sources) == 18 and sources == phase['source_files'] and phase['phase100_closed'] is True
            and phase['counts'] == COUNTS and phase['native_backend'] == 'cpu_hist'
            and phase['actual_XGB_fit_calls'] == phase['actual_informative_native_objectives'] == 0
            and phase['no_fit_or_sample_Popen'] is True and phase['official_tests_opened'] is False
            and phase['generator_tuned_winner_claimed'] is False and phase['mfs_v2'] is None
            and phase['ptf_v1'] is None and phase['author_gpu_hist_reproduction'] is False,
            'native frozen phase scope changed')
    input_phase = read_ref(phase['native_input_phase_ref'])
    input_rows = {x['dataset']: x for x in input_phase['lineages']}
    fits = {x['dataset_id']: x for x in fit_report['cells']}
    require(len(input_rows) == len(fits) == 100 and set(fits) == set(input_rows), 'native input or fit cohort changed')
    for row in phase['lineages']:
        require(row['native_input_status'] == input_rows[row['dataset']]['status'], 'prior input status upgraded')
    cells = []
    for frozen in phase['lineages']:
        row = dict(frozen)
        if row['scientific_native_Popen'] == 1 and 'native_auditor_calls_ref' not in row:
            path = str(Path(row['request_ref']['path']).with_name('native.CPU-auditor-calls.receipt.json'))
            row['native_auditor_calls_ref'] = {'path': path, 'sha256': lock['metadata'][path]['sha256']}
        cells.append(native_cell(row, fits[row['dataset']], read_ref, sources))
    summary = summarize(cells)
    prior = read_ref(phase['prior_failed_native_closure_ref'])
    retries = [row['retry_of'] for row in phase['lineages'] if row.get('retry_of') is not None]
    require(retries == prior['failed_native_operations'] and len(retries) == prior['scientific_native_Popen_count'] == 68
            and prior['actual_native_successes'] == 0, 'prior failure ledger changed')
    same_cost(summary['prior_failed_native_whole_seconds'], prior['failed_native_whole_seconds'])
    same_cost(summary['measured_current_native_whole_seconds'], phase['measured_current_native_whole_seconds'])
    family = read_ref(lock['family_ref'])
    require(family['all_captured_identities_absent_twice'] is True and family['no_own_GPU_context_twice'] is True
            and family['actual_scientific_Popen_count'] == family['actual_scientific_worker_exit0_count'] == 91
            and family['actual_parent_kernel_exit'] is None and family['actual_parent_kernel_exit_observed'] is False,
            'native family closure or unknown parent exit changed')
    return {'format': 'tabsyn-default100-native-audit-publication', 'version': 1,
            'scope': 'Closed default-only native audit; not a native-tuned winner or common quality panel',
            'partition': 'official_training_derived_validation', 'method': 'TabSyn', 'author_commit': AUTHOR,
            'config_sha256': CONFIG_SHA, 'fit_seed': 11, 'native_backend': 'cpu_hist',
            'native_objective': 'Author best_r2_scores XGBRegressor R2; author target transform preserved',
            'track': 'scaled_200_vae_1000_diffusion', 'summary': summary, 'cells': cells, 'source_files': sources,
            'publisher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'immutable_references': [lock['phase_ref'], lock['fit_report_ref'], lock['family_ref'],
                                     phase['native_input_phase_ref'], phase['prior_failed_native_closure_ref']],
            'input_custody_lock': {'path': str(CACHE / 'input-custody.lock.json'), 'sha256': CACHE_SHA},
            'verified_receipt_buffers': len(lock['metadata']), 'verified_source_buffers': len(sources),
            'actual_parent_kernel_exit': None, 'actual_parent_kernel_exit_observed': False,
            'native_audit_ledger_complete': True,
            **{key: None for key in NULL_CLAIMS}, **{key: False for key in FALSE_CLAIMS}}


def tables(report):
    require(summarize(report['cells']) == report['summary'] and all(report[k] is None for k in NULL_CLAIMS)
            and all(report[k] is False for k in FALSE_CLAIMS), 'native ledger acquired a scientific claim')
    columns = ('dataset_id', 'lineage_sha256', 'fit_seed', 'status', 'reason', 'actual_worker_exit',
               'native_result_admitted', 'native_value', 'native_winner', 'charged_artifact_bytes',
               'measured_whole_seconds', 'prior_failed_native_attempts', 'prior_failed_native_whole_seconds')
    out = io.StringIO(); writer = csv.writer(out, lineterminator='\n'); writer.writerow(columns)
    for cell in report['cells']: writer.writerow([cell[key] for key in columns])
    s = report['summary']
    markdown = '\n'.join(['# TabSyn default native audit ledger', '',
        'All 100 lineages are accounted for: 77 admitted but uninformative author objectives, 14 audit-monitoring unavailable, and nine input-monitoring unavailable.', '',
        'All 91 physical audit workers exited zero. Fourteen remain unavailable because monitoring admission failed; exit zero does not turn them into usable results.', '',
        'The 77 admitted audits found a constant target after the unchanged author log/float32 transform on TRAIN-projection-clipped target units. Native values and winners are null; defaults remain retained. Zero XGB auditor fits and zero generator tuning trials ran.', '',
        'This uses the documented CPU hist auditor variant, not a reproduction of the author GPU hist runtime. The author 36-configuration and four-postselection-refit contract is preserved; none ran for the uninformative targets.', '',
        f"Current native operations: {s['measured_current_native_whole_seconds']:.6f}s measured. Earlier {s['prior_failed_native_attempts']} infrastructure failures: {s['prior_failed_native_whole_seconds']:.6f}s measured, retained separately.", '',
        f"Existing model plus projection charge: {s['charged_model_and_projection_bytes']:,} bytes. No generator fits or new samples ran in this audit phase; earlier CUDA fit costs remain in the separate fit ledger.", '',
        'Sampling and shared validation are separate phases. This ledger makes no common utility, privacy, production or superiority claim. Official tests remain sealed; MFS-v2/PTF-v1 and all gated scores are null.', ''])
    return out.getvalue(), markdown


def schema(report):
    shape = infer_schema(report)
    for key in (*NULL_CLAIMS, *FALSE_CLAIMS, 'method', 'author_commit', 'native_backend', 'config_sha256', 'fit_seed'):
        shape['properties'][key] = {'const': report[key]}
    cell = shape['properties']['cells']['items']
    for item in cell.get('anyOf', [cell]):
        item['properties']['dataset_id'] = {'type': 'string', 'pattern': '^[0-9a-f]{16}$'}
        for key in ('native_value', 'native_winner', 'actual_XGB_fit_calls', 'config_sha256', 'fit_seed'):
            item['properties'][key] = {'const': report['cells'][0][key]}
    return shape


def main():
    report = build(); table, markdown = tables(report); path = RESULTS / (NAME + '.json')
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(report), sort_keys=True, indent=2) + '\n')
    path.with_suffix('.csv').write_text(table); path.with_suffix('.md').write_text(markdown)
    print(path)


if __name__ == '__main__': main()
