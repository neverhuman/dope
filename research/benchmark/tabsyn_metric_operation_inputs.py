"""Join frozen TabSyn fit and sample operations for shared metric preparation.

This module never reads a numeric/model/projection body or imports an auditor.
It binds a single snapshot operation; future admitted execution must separately
verify input bodies and runtime before calling the unchanged shared evaluator.
The injected readers must return captured metadata/source bytes for exact refs.
"""
from collections import Counter
import hashlib
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import re

ADAPTER_SHA256 = '19111055329f4d05f4366c668bd2cd5562fa10c1957f3fed06645f3700aec628'
EVALUATOR_SHA256 = 'a952062c0f83805f6442a440a5ae15293a843349d423eaa106a9c56a20d14c0f'
KINDS_REF = {
    'path': '/home/ubuntu/dope/.agent/worktrees/integration/target/work-order-b-expanded-metric-round-v1/runtime-final-rehash-repair-v3/source-v10/kinds.py',
    'bytes': 2375,
    'sha256': '576169b3afb4e4b8168b1e2e05a931f6bf98e1ab44fea793d70e9d08b0be7add',
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def checked(ref, reader):
    require(type(ref) is dict and type(ref.get('path')) is str
            and Path(ref['path']).is_absolute() and '..' not in Path(ref['path']).parts
            and type(ref.get('sha256')) is str
            and re.fullmatch('[0-9a-f]{64}', ref['sha256']) is not None, 'invalid_reference')
    expected = dict(ref)
    data = reader(dict(expected))
    require(type(data) is bytes and hashlib.sha256(data).hexdigest() == expected['sha256'],
            'pinned_bytes_changed')
    if 'bytes' in expected:
        require(type(expected['bytes']) is int and expected['bytes'] == len(data), 'pinned_size_changed')
    return data


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate_json_key')
        result[key] = value
    return result


def decode(ref, reader):
    data = checked(ref, reader)  # Authenticate the same buffer before JSON decoding.
    def finite_float(text):
        value = float(text)
        require(math.isfinite(value), 'nonfinite_json_constant')
        return value
    return json.loads(data, object_pairs_hook=pairs,
                      parse_float=finite_float,
                      parse_constant=lambda _: require(False, 'nonfinite_json_constant'))


class FrozenAdapterLoader(importlib.machinery.SourceFileLoader):
    """Compile the authenticated source buffer without consulting bytecode."""
    def __init__(self, path, data):
        super().__init__('frozen_tabsyn_validation_inputs', path)
        self.data = data

    def get_code(self, fullname):
        require(fullname == self.name, 'adapter_module_identity_changed')
        return compile(self.data, self.path, 'exec')


def bind_adapter(adapter_ref, source_reader):
    require(type(adapter_ref) is dict and adapter_ref['sha256'] == ADAPTER_SHA256,
            'unreviewed_validation_adapter')
    data = checked(dict(adapter_ref), source_reader)
    require(hashlib.sha256(data).hexdigest() == ADAPTER_SHA256, 'pinned_adapter_bytes_changed')
    loader = FrozenAdapterLoader(adapter_ref['path'], data)
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module.__dict__


def validate_fit_operation(worker, fit, metadata_reader):
    """Join actual scientific Popen/exit/completion; never invent parent exit."""
    refs = [r for r in fit['actual_metadata_refs']
            if Path(r['path']).name == 'actual-scientific-Popen.receipt.json']
    require(len(refs) == 1, 'fit_Popen_receipt_missing_or_duplicated')
    request = decode(worker['fit_request_ref'], metadata_reader)
    popen = decode(refs[0], metadata_reader)
    complete = decode(worker['fit_complete_ref'], metadata_reader)
    exit_row = decode(worker['fit_exit_ref'], metadata_reader)
    require(type(fit['retained_first']) is bool, 'fit_receipt_variant_unbound')
    retained = fit['retained_first']
    require((retained and 'mode' not in request) or
            (not retained and request.get('mode') == 'fit'), 'fit_receipt_variant_changed')
    require(type(request['lineage_sha256']) is str
            and request['lineage_sha256'] == fit['lineage_sha256'], 'fit_request_lineage_changed')
    require(popen['request_ref'] == worker['fit_request_ref']
            and type(popen['scientific_worker_pid']) is int and popen['scientific_worker_pid'] > 0
            and type(popen['uid']) is int and popen['uid'] == 1000
            and type(popen['start_ticks']) is str and popen['start_ticks'].isdigit()
            and canonical([popen['scientific_worker_pid'], popen['start_ticks']]) ==
                canonical([exit_row['scientific_worker_pid'], exit_row['start_ticks']])
            and popen['whole_deadline_monotonic'] == request['deadline_monotonic']
            and popen['official_tests_opened'] is False, 'fit_process_join_changed')
    require(type(exit_row['actual_worker_exit']) is int and exit_row['actual_worker_exit'] == 0
            and type(exit_row['foreign_signals']) is int and exit_row['foreign_signals'] == 0,
            'fit_scientific_exit_unavailable')
    if retained:
        require(exit_row['fit_complete_receipt_present'] is True
                and exit_row['owned_termination_reason'] is None
                and exit_row['official_tests_opened'] is False
                and exit_row['production_certified'] is False, 'retained_fit_exit_unavailable')
        result, inventory = complete['result'], complete['inventory']
    else:
        require(exit_row['success_receipt_ref'] == worker['fit_complete_ref'],
                'fit_success_receipt_join_changed')
        result, inventory = complete['result']['result'], complete['result']['inventory']
    require(complete['status'] == 'ok' and complete['method'] == 'TabSyn'
            and complete['lineage_sha256'] == fit['lineage_sha256']
            and complete['config_sha256'] == worker['config_sha256']
            and complete['official_tests_opened'] is False and complete['production_certified'] is False
            and complete['mfs_v2'] is None and complete['ptf_v1'] is None
            and canonical(inventory) == canonical(worker['artifact_inventory'])
            and result['complete'] is True and type(result['seed']) is int and result['seed'] == 11
            and type(result['rows']) is int and result['rows'] == worker['original_TRAIN_input_ref']['row_count']
            and type(result['columns']) is int and result['columns'] == worker['dimensions']
            and result['code_sha256'] == result['operation_admission_sha256'] == worker['fit_request_ref']['sha256']
            and canonical(result['config']) == canonical(request['config'])
            and result['source_commit'] == request['learning_identity']['source_commit']
            and result['runtime_sha256'] == request['runtime_ref']['sha256']
            and result['generated_fixture'] is False and result['official_tests_opened'] is False,
            'fit_completion_binding_changed')
    return request, refs[0]


def prepare_operation(snapshot_ref, adapter_ref, lineage_sha256, sample_seed, row_multiplier,
                      *, metadata_reader, source_reader):
    """Return a disabled input plan or preserve the cell's unavailable outcome.

    Missing captured operation metadata returns unavailable. Integrity errors
    raise; they are never converted into an eligible input or silently retried.
    """
    require(type(lineage_sha256) is str and re.fullmatch('[0-9a-f]{64}', lineage_sha256) is not None
            and type(sample_seed) is int and sample_seed in (101, 211, 307)
            and type(row_multiplier) is int and row_multiplier in (1, 4), 'invalid_cell_identity')
    adapter = bind_adapter(adapter_ref, source_reader)
    snapshot = decode(snapshot_ref, metadata_reader)
    require(snapshot['format'] == 'dope-TS-admitted-common-diagnostic-inputs-PREPARATION-ONLY-v1'
            and snapshot['execution_enabled'] is False and snapshot['scientific_Popen_authorized'] is False
            and type(snapshot['actual_evaluator_jobs_started']) is int and snapshot['actual_evaluator_jobs_started'] == 0
            and snapshot['claims']['official_tests_opened'] is False
            and snapshot['evaluator_ref']['sha256'] == EVALUATOR_SHA256, 'snapshot_scope_changed')
    phase = decode(snapshot['fit_phase100_ref'], metadata_reader)
    require(phase['fit_phase_closed'] is True and phase['default_fits_all_ok'] is True
            and phase['source_commit'] == adapter['AUTHOR'] and phase['config_sha256'] == adapter['CONFIG']
            and phase['official_tests_opened'] is False, 'fit_phase_incomplete')
    fits = {r['lineage_sha256']: r for r in phase['lineages']}
    require(len(fits) == len(phase['lineages']) == len(snapshot['workers']) == 100
            and set(fits) == set(snapshot['workers']), 'fit_population_changed')
    identities = []
    for cell in snapshot['sample_cells']:
        require(type(cell['seed']) is int and cell['seed'] in (101, 211, 307)
                and type(cell['row_multiplier']) is int and cell['row_multiplier'] in (1, 4),
                'snapshot_cell_identity_changed')
        identities.append((cell['lineage_sha256'], cell['seed'], cell['row_multiplier']))
    require(len(identities) == len(set(identities)) == 600
            and {k[0] for k in identities} == set(fits)
            and dict(Counter(c['status'] for c in snapshot['sample_cells'])) == snapshot['status_counts'],
            'snapshot_accounting_changed')
    for cell in snapshot['sample_cells']:
        require(type(cell['eligible_for_evaluator']) is bool
                and cell['eligible_for_evaluator'] == (cell['status'] in adapter['ADMITTED']),
                'cell_eligibility_changed')
    require(type(snapshot['admitted_snapshot_inputs']) is int
            and snapshot['admitted_snapshot_inputs'] ==
                sum(c['eligible_for_evaluator'] for c in snapshot['sample_cells']),
            'snapshot_admitted_count_changed')
    cell = snapshot['sample_cells'][identities.index((lineage_sha256, sample_seed, row_multiplier))]
    identity = dict(format='dope-tabsyn-metric-input-job-v1', snapshot_ref=snapshot_ref,
                    adapter_ref=adapter_ref, evaluator_ref=snapshot['evaluator_ref'],
                    lineage_sha256=lineage_sha256, sample_seed=sample_seed, row_multiplier=row_multiplier,
                    evaluator_seed=1729, maximum_rows=800)
    outcome = dict(job_sha256=digest(identity), job_identity=identity,
                   historical_parent_exit=None, execution_enabled=False, official_tests_opened=False,
                   production_certified=False, mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None)
    if not cell['eligible_for_evaluator']:
        return dict(outcome, status='unavailable', reason='snapshot_sample_not_admitted',
                    historical_sample_status=cell['status'], evaluator_input=None)
    worker, fit = snapshot['workers'][lineage_sha256], fits[lineage_sha256]
    try:
        fitted, fit_popen_ref = validate_fit_operation(worker, fit, metadata_reader)
        plan = adapter['prepare_cell'](
            cell, worker, fit, lambda ref: checked(ref, metadata_reader))
        catalog = decode(snapshot['actual100_kind_catalog_ref'], metadata_reader)
        require(catalog['format'] == 'dope100-actual-kind-map-catalog'
                and catalog['official_tests_opened'] is False
                and catalog['numerical_source_changes'] is False and len(catalog['kind_maps']) == 100,
                'kind_catalog_changed')
        kind_ref = catalog['kind_maps'][worker['kind_worker_key']]
        suffix = Path(kind_ref['path']).parts
        require(bool(suffix) and '..' not in suffix and not Path(kind_ref['path']).is_absolute()
                and Path(worker['kind_map_ref']['path']).parts[-len(suffix):] == suffix
                and kind_ref['sha256'] == worker['kind_map_ref']['sha256'], 'kind_catalog_join_changed')
        kinds = decode(worker['kind_map_ref'], metadata_reader)
        require(digest(kinds['rule']) == kinds['rule_sha256']
                and kinds['author_semantic_types'] is None
                and kinds['author_semantic_categorical_tv_available'] is False
                and kinds['producer_source_sha256'] == KINDS_REF['sha256'], 'kind_rule_or_scope_changed')
        checked(KINDS_REF, source_reader)
        sample_request = decode(cell['sample_request_ref'], metadata_reader)
        # Verify scientific source buffers, without importing or executing them.
        sources = dict(fitted['source_files'])
        for path, pin in sample_request['source_files'].items():
            require(path not in sources or sources[path] == pin, 'source_reference_conflict')
            sources[path] = pin
        for request in (fitted, sample_request):
            for name in ('adapter_ref', 'entry_ref'):
                ref = request[name]
                require(request['source_files'].get(ref['path']) == ref['sha256'],
                        'executable_missing_from_source_inventory')
        require(fitted['adapter_ref']['sha256'] == snapshot['deterministic_order_evidence']['adapter_ref']['sha256']
                == sample_request['adapter_ref']['sha256'], 'author_adapter_changed')
        require(canonical(fitted['runtime_ref']) == canonical(sample_request['runtime_ref']),
                'sample_runtime_differs_from_fit')
        for path, pin in sorted(sources.items()):
            checked({'path': path, 'sha256': pin}, source_reader)
        checked(snapshot['evaluator_ref'], source_reader)
    except FileNotFoundError:
        return dict(outcome, status='unavailable', reason='captured_receipt_or_source_unmaterialized', evaluator_input=None)
    body_refs = [plan['train_ref'], plan['validation_ref'], plan['synthetic_ref'], plan['projection_ref']]
    for name, pin in sorted(worker['artifact_inventory']['files'].items()):
        require(Path(name).name == name and name not in ('.', '..'), 'artifact_member_path_changed')
        body_refs.append(dict(path=str(Path(worker['artifact_path']) / name), **pin))
    return dict(outcome, status='prepared_metadata_only', evaluator_input=plan,
                fit_Popen_ref=fit_popen_ref, fit_complete_ref=worker['fit_complete_ref'], fit_exit_ref=worker['fit_exit_ref'],
                scientific_source_refs=[dict(path=p, sha256=s) for p, s in sorted(sources.items())],
                required_body_refs_before_dependency_import=body_refs,
                required_runtime_ref=fitted['runtime_ref'],
                required_runtime_and_capacity_admission=True, current_body_bytes_verified=False,
                common_target_last_already_applied=True, additional_column_permutation=None)
