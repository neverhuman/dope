"""Bind admitted TabSyn samples to shared validation inputs, without execution.

This standard-library-only preparation accepts immutable metadata bytes. It
does not read numeric tables, import auditors, load models, or grant capacity.
An admitted external worker must rehash the physical input bytes, verify its
runtime and deadline, and call the unchanged shared evaluators. The historical
TabSyn parent exit is unknown and is never replaced with an inferred zero.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

AUTHOR = 'cb5ac0f74ec36ee88e7a974a393dfbef50d42da7'
CONFIG = 'a34729383eac6783c7e754a791094054d54f1062655db9a306cd4cca296c5ebc'
EVALUATOR = 'a952062c0f83805f6442a440a5ae15293a843349d423eaa106a9c56a20d14c0f'
WRITERS = {
    'cpu': '6117591f8e831ef6b72cc82ce411b174bfbe2a7d1b886eebe8d29e78582b96e3',
    'cuda:0': '5c69a932c9125ba59d38ed4b6140973c12c88c3f2f55a66f68cbd76840c1f5a4',
}
ADMITTED = {'admitted_new_CPU', 'admitted_retained_GPU'}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def reference(ref):
    require(type(ref) is dict and type(ref.get('path')) is str
            and type(ref.get('sha256')) is str
            and re.fullmatch('[0-9a-f]{64}', ref['sha256']) is not None,
            'invalid frozen reference')
    path = Path(ref['path'])
    require(path.is_absolute() and '..' not in path.parts and path.name != 'test.csv',
            'invalid validation input path')
    return ref


def read_local(ref):
    path = Path(reference(ref)['path'])
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'metadata alias rejected')
    return path.read_bytes()


def payload(ref, read_bytes):
    reference(ref)
    data = read_bytes(ref)
    require(type(data) is bytes and hashlib.sha256(data).hexdigest() == ref['sha256'],
            'frozen metadata bytes changed')
    if 'bytes' in ref:
        require(type(ref['bytes']) is int and ref['bytes'] == len(data), 'metadata byte count changed')
    return data


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate metadata key')
        result[key] = value
    return result


def decode(ref, read_bytes):
    # Hash the captured buffer before parsing; never reread it for decoding.
    return json.loads(payload(ref, read_bytes), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'nonfinite metadata constant'))


def integer(value, allowed=None):
    require(type(value) is int and value >= 0 and (allowed is None or value in allowed),
            'invalid integer identity')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def positive_exit(exit_row):
    integer(exit_row['actual_worker_exit'], {0})
    integer(exit_row['foreign_signals'], {0})
    if 'owned_signal_calls' in exit_row:
        integer(exit_row['owned_signal_calls'], {0})
    require(exit_row['owned_termination_reason'] is None
            and exit_row['official_tests_opened'] is False, 'sample lifecycle unavailable')


def input_refs(worker):
    train = reference(worker['original_TRAIN_input_ref'])
    validation = reference(worker['original_derived_validation_input_ref'])
    projection = reference(worker['projection_ref'])
    require(Path(train['path']).name == 'train.csv'
            and Path(validation['path']).name == 'validation.csv'
            and Path(projection['path']).name == 'projection.json'
            and train['csv_format'] == validation['csv_format'] == 'headerless_numeric'
            and worker['validation_provenance']['partition'] == 'official_training_derived_validation'
            and worker['validation_provenance']['official_tests_opened'] is False
            and worker['validation_provenance']['projection_sha256'] == projection['sha256'],
            'validation partition or projection changed')
    integer(train['row_count']); integer(validation['row_count'])
    require(train['row_count'] >= 2 and validation['row_count'] >= 2, 'insufficient real rows')
    return train, validation, projection


def real_input_binding(worker, fitted, train, validation, read_bytes):
    source = fitted['worker']
    if 'cohort' in source:
        require(source['cohort']['train_sha256'] == train['sha256']
                and source['cohort']['validation_sha256'] == validation['sha256'],
                'original fit numeric inputs changed')
        return
    proof = decode(worker['header_view_parent_binding_ref'], read_bytes)
    require(proof['format'] == 'dope-retained-first-worker-header-payload-identity-readonly-v1'
            and proof['first_fit_request_ref'] == worker['fit_request_ref'] and len(proof['rows']) == 2,
            'retained header view proof changed')
    for name, ref in (('train.csv', train), ('validation.csv', validation)):
        rows = [r for r in proof['rows'] if r['file'] == name]
        require(len(rows) == 1 and rows[0]['original_payload_byte_identical'] is True
                and rows[0]['original_ref']['path'] == ref['path']
                and rows[0]['original_ref']['sha256'] == rows[0]['payload_sha256'] == ref['sha256']
                and rows[0]['view_ref']['path'] == str(Path(source['path']) / name)
                and rows[0]['view_ref']['sha256'] == source['files'][name],
                'retained fit numeric payload changed')


def prepare_cell(cell, worker, fit, read_bytes):
    require(cell['status'] in ADMITTED and cell['eligible_for_evaluator'] is True,
            'unavailable sample is not an evaluator input')
    integer(cell['seed'], {101, 211, 307}); integer(cell['row_multiplier'], {1, 4})
    integer(cell['rows']); integer(cell['actual_worker_exit'], {0})
    d = integer(worker['dimensions'])
    require(d >= 2 and worker['author_target_index'] == 0 and type(worker['author_target_index']) is int
            and worker['common_target_index'] == d - 1 and type(worker['common_target_index']) is int
            and worker['author_joint_to_common_permutation'] == list(range(1, d)) + [0]
            and worker['common_to_author_joint_permutation'] == [d - 1] + list(range(d - 1))
            and worker['sample_CSV_to_common_permutation'] == list(range(d)),
            'frozen target-last CSV order changed')
    for name in ('author_joint_to_common_permutation', 'common_to_author_joint_permutation',
                 'sample_CSV_to_common_permutation'):
        require(all(type(x) is int for x in worker[name]), 'invalid integer column order')
    train, validation, projection = input_refs(worker)
    require(cell['rows'] == train['row_count'] * cell['row_multiplier']
            and cell['purpose'] == 'matched' and worker['config_sha256'] == CONFIG
            and worker['fit_request_ref'] == fit['fit_request_ref']
            and worker['fit_complete_ref'] == fit['fit_complete_ref']
            and worker['fit_exit_ref'] == fit['fit_exit_ref']
            and fit['lineage_sha256'] == cell['lineage_sha256']
            and fit['dataset'] == worker['dataset'] and fit['status'] == 'fit_ok'
            and type(fit['fit_seed']) is int and fit['fit_seed'] == 11
            and fit['actual_worker_exit'] == 0 and type(fit['actual_worker_exit']) is int
            and worker['artifact_inventory'] == fit['artifact_inventory']
            and worker['artifact_path'] == fit['artifact_path'], 'default fit or sample matrix changed')
    kinds = decode(worker['kind_map_ref'], read_bytes)
    require(kinds['column_kinds'] == worker['column_kinds_common_order']
            and len(kinds['column_kinds']) == d
            and not set(kinds['column_kinds']) - {'continuous', 'categorical'}
            and kinds['train_sha256'] == train['sha256']
            and kinds['worker_key'] == worker['kind_worker_key']
            and kinds['rule_sha256'] == worker['kind_rule_sha256']
            and kinds['producer_source_sha256'] == worker['kind_producer_source_sha256']
            and kinds['validation_or_synthetic_used_for_types'] is False
            and kinds['official_tests_opened'] is False, 'TRAIN-only column kinds changed')
    q = decode(cell['sample_request_ref'], read_bytes)
    fitted = decode(worker['fit_request_ref'], read_bytes)
    popen = decode(cell['sample_Popen_ref'], read_bytes)
    exit_row = decode(cell['sample_exit_ref'], read_bytes)
    complete = decode(cell['sample_complete_ref'], read_bytes)
    identity = q['learning_identity']
    require(type(identity['fit_seed']) is int and identity['fit_seed'] == 11
            and identity['method'] == 'TabSyn' and identity['source_commit'] == AUTHOR
            and identity['dataset'] == worker['dataset']
            and identity['projection_sha256'] == projection['sha256']
            and q['lineage_sha256'] == cell['lineage_sha256']
            and q['config']['config_sha256'] == CONFIG and q['mode'] == 'sample'
            and q['artifact_code_ref'] == worker['fit_request_ref']
            and q['artifact_inventory'] == worker['artifact_inventory']
            and q['artifact_path'] == worker['artifact_path']
            and q['official_tests_opened'] is False and q['production_certified'] is False
            and q['generated_fixture'] is False
            and q['lineage_started_unix_seconds'] == worker['original_lineage_start']
            and canonical(identity) == canonical(fitted['learning_identity'])
            and canonical(q['config']) == canonical(fitted['config'])
            and worker['original_lineage_start'] == fit['original_lineage_started_unix_seconds']
            and worker['original_lineage_deadline'] == fit['original_lineage_deadline_unix_seconds'],
            'sample request differs from the frozen default fit')
    real_input_binding(worker, fitted, train, validation, read_bytes)
    backend = cell['physical_backend']
    require(backend in WRITERS and q['entry_ref']['sha256'] == WRITERS[backend]
            and q['source_files'][q['entry_ref']['path']] == WRITERS[backend]
            and (q.get('sample_backend') == 'cpu') == (backend == 'cpu')
            and (cell['status'] == 'admitted_new_CPU') == (backend == 'cpu'), 'sample writer changed')
    integer(popen['scientific_fit_allocations'], {0})
    integer(exit_row['scientific_fit_allocations'], {0})
    if backend == 'cpu':
        integer(q['scientific_fit_allocations'], {0})
        integer(exit_row['owned_signal_calls'], {0})
    require(popen['request_ref'] == cell['sample_request_ref']
            and popen['scientific_worker_pid'] == exit_row['scientific_worker_pid']
            and popen['start_ticks'] == exit_row['start_ticks']
            and popen['whole_deadline_monotonic'] == q['deadline_monotonic'], 'sample process join changed')
    positive_exit(exit_row)
    require(exit_row['deadline_rechecked'] is True
            and exit_row['completed_monotonic'] <= exit_row['deadline_monotonic'] == q['deadline_monotonic']
            and complete['status'] == 'ok' and complete['mode'] == 'sample'
            and complete['lineage_sha256'] == cell['lineage_sha256']
            and complete['config_sha256'] == CONFIG and complete['official_tests_opened'] is False
            and complete['production_certified'] is False and complete['mfs_v2'] is None
            and complete['ptf_v1'] is None, 'sample completion is unavailable')
    result = cell['sample_result']
    require(all(type(r['seed']) is int and type(r['rows']) is int for r in q['samples']),
            'request sample identity changed')
    require(sum(row == result for row in complete['result']) == 1
            and result['rows'] == cell['rows'] and type(result['rows']) is int
            and result['seed'] == cell['seed'] and type(result['seed']) is int
            and result['purpose'] == 'matched' and Path(result['path']).parent == Path(cell['sample_request_ref']['path']).parent
            and Path(result['path']).name == result['filename']
            and {'filename': result['filename'], 'purpose': 'matched', 'rows': cell['rows'], 'seed': cell['seed']} in q['samples'],
            'physical sample is not the requested output')
    reference(result); integer(result['bytes'])
    start, deadline = worker['original_lineage_start'], worker['original_lineage_deadline']
    require(all(type(x) in (int, float) and math.isfinite(x) for x in (start, deadline))
            and math.isclose(deadline - start, 43200, abs_tol=1e-8, rel_tol=0), 'original lineage clock changed')
    inventory = worker['artifact_inventory']
    require(sum(integer(v['bytes']) for v in inventory['files'].values()) == inventory['artifact_bytes']
            and inventory['projection_bytes'] == inventory['files']['projection.json']['bytes']
            and inventory['files']['projection.json']['sha256'] == projection['sha256'], 'artifact byte charge changed')
    return {'dataset': worker['dataset'], 'lineage_sha256': cell['lineage_sha256'], 'fit_seed': 11,
            'config_sha256': CONFIG, 'selection_binding': 'scaled_author_default',
            'physical_sampling_backend': backend,
            'sample_seed': cell['seed'], 'row_multiplier': cell['row_multiplier'], 'rows': cell['rows'],
            'train_ref': train, 'validation_ref': validation, 'synthetic_ref': result,
            'projection_ref': projection, 'kind_map_ref': worker['kind_map_ref'],
            'column_kinds': kinds['column_kinds'], 'column_order': 'projected_inputs_then_target',
            'sample_CSV_to_common_permutation': list(range(d)), 'synthetic_csv_header_rows': 1,
            'original_lineage_started_unix_seconds': start, 'original_lineage_deadline_unix_seconds': deadline,
            'charged_artifact_bytes': inventory['artifact_bytes'], 'projection_bytes': inventory['projection_bytes'],
            'sample_operation_refs': {k: cell[k] for k in ('sample_request_ref', 'sample_Popen_ref',
                                                         'sample_exit_ref', 'sample_complete_ref')},
            'historical_owned_signal_calls': exit_row.get('owned_signal_calls'),
            'current_numeric_input_bytes_rehashed': False, 'current_bulk_model_hashes_reverified': False,
            'historical_parent_exit': None, 'execution_enabled': False, 'official_tests_opened': False,
            'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None, 'superiority': None}


def prepare(snapshot_ref, read_bytes=read_local):
    snapshot = decode(snapshot_ref, read_bytes)
    require(snapshot['format'] == 'dope-TS-admitted-common-diagnostic-inputs-PREPARATION-ONLY-v1'
            and snapshot['execution_enabled'] is False and snapshot['scientific_Popen_authorized'] is False
            and snapshot['actual_evaluator_jobs_started'] == 0
            and snapshot['claims']['official_tests_opened'] is False
            and snapshot['evaluator_ref']['sha256'] == EVALUATOR,
            'snapshot acquired execution or test authority')
    phase = decode(snapshot['fit_phase100_ref'], read_bytes)
    require(phase['fit_phase_closed'] is True and phase['default_fits_all_ok'] is True
            and phase['source_commit'] == AUTHOR and phase['config_sha256'] == CONFIG
            and phase['official_tests_opened'] is False, 'default fit phase is incomplete')
    fits = {r['lineage_sha256']: r for r in phase['lineages']}
    require(len(fits) == len(phase['lineages']) == len(snapshot['workers']) == 100,
            'fit lineages missing or duplicated')
    counts = Counter(r['status'] for r in snapshot['sample_cells'])
    require(dict(counts) == snapshot['status_counts'] and sum(counts.values()) == 600,
            'sample accounting changed')
    jobs, identities = [], set()
    for cell in snapshot['sample_cells']:
        integer(cell['seed'], {101, 211, 307}); integer(cell['row_multiplier'], {1, 4})
        key = (cell['lineage_sha256'], cell['seed'], cell['row_multiplier'])
        require(key not in identities and key[0] in fits, 'sample identity missing or duplicated')
        identities.add(key)
        require(type(cell['eligible_for_evaluator']) is bool
                and cell['eligible_for_evaluator'] == (cell['status'] in ADMITTED),
                'unavailable outcome acquired eligibility')
        if cell['eligible_for_evaluator']:
            jobs.append(prepare_cell(cell, snapshot['workers'][key[0]], fits[key[0]], read_bytes))
    require(len(jobs) == snapshot['admitted_snapshot_inputs'], 'admitted sample count changed')
    return {'format': 'dope-tabsyn-shared-validation-input-preparation-v1', 'snapshot_ref': snapshot_ref,
            'status_counts': dict(counts), 'jobs': jobs, 'sample_cells': 600,
            'physical_samples_rehashed': False, 'numeric_rows_decoded': False,
            'historical_parent_exit': None, 'execution_enabled': False, 'official_tests_opened': False,
            'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None, 'superiority': None}
