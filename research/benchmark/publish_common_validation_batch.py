"""Source-only candidate: project authenticated, closed A952 metric receipts.

No CSV/model/sample/provider read, evaluator import, subprocess or admission.
ROOT must bind a literal frozen input and validate the emitted public schema.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from research.benchmark import common_validation_receipt_verifier as receipt_verifier

A952 = 'a952062c0f83805f6442a440a5ae15293a843349d423eaa106a9c56a20d14c0f'
ADAPTER = '27e1d7dda3dbc411418d714b9f8e578d4fe4612701d64533fab29659961adf06'
SCHEMA = '7a22ce930edb578ca67d048ef4f31438cc8f1e9e0bf0de5eddebd47599735f52'
HEX = re.compile(r'^[0-9a-f]{64}$')
CORE_KEYS = {'format', 'version', 'fidelity', 'alpha_beta', 'detection', 'privacy',
             'official_tests_opened', 'mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority'}
READ_REFS = {}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def no_duplicates(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate_json_key')
        result[key] = value
    return result


def checked_bytes(item):
    require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}, 'exact_reference_required')
    path, length, pin = item['path'], item['bytes'], item['sha256']
    require(type(path) is str and type(length) is int and 0 <= length <= 24 * 1024 ** 2
            and type(pin) is str and HEX.fullmatch(pin), 'reference_identity_changed')
    path = Path(path)
    require(str(path) == item['path'], 'metadata_reference_path_not_canonical')
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'metadata_alias')
    raw = path.read_bytes()
    require(len(raw) == length and hashlib.sha256(raw).hexdigest() == pin, 'metadata_pin_changed')
    require(str(path) not in READ_REFS or READ_REFS[str(path)] == item, 'metadata_identity_rebound')
    READ_REFS[str(path)] = dict(item)
    return raw


def checked(item):
    return json.loads(checked_bytes(item), object_pairs_hook=no_duplicates,
                      parse_constant=lambda _: require(False, 'nonfinite_json'))


def number(value, reason, integer=False):
    require(type(value) is int if integer else type(value) in (int, float), reason)
    require(math.isfinite(value) and value >= 0, reason)
    return value


def adapter_from_ref(item):
    require(item['sha256'] == ADAPTER, 'unreviewed_receipt_verifier')
    # The frozen scratch adapter is provenance only. Its bytes are authenticated
    # and tracked for final rehash, but never compiled, imported or executed.
    checked_bytes(item)
    return receipt_verifier


def core_diagnostics(stdout, lock):
    full = stdout['diagnostics']; representation = full['representation']
    rule = lock['kind_rule']
    require(representation['rule'] == rule and representation['rule_sha256'] == digest(rule)
            and representation['target_included'] is True
            and representation['author_semantic_types'] is None
            and representation['author_semantic_categorical_claim'] is False,
            'representation_provenance_changed')
    for name in ('kind_map_canonical_sha256', 'kind_map_file_sha256'):
        require(type(representation[name]) is str and HEX.fullmatch(representation[name]), 'kind_map_identity_changed')
    result = copy.deepcopy(full)
    del result['representation']
    require(set(result) == CORE_KEYS and result['format'] == 'dope-expanded-validation-diagnostics'
            and type(result['version']) is int and result['version'] == 1,
            'evaluator_output_contract_changed')
    for key in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority'):
        require(result[key] is None, 'gated_score_claim')
    require(result['official_tests_opened'] is False and result['privacy']['empirical_only'] is True
            and result['privacy']['formal_dp'] is False and result['privacy']['hipaa_deidentification'] is False,
            'sealed_test_or_certification_claim')
    fidelity = result['fidelity']; marginals = fidelity['marginals']
    for key, kind in (('binary_representation_total_variation', 'categorical'),
                      ('continuous_numeric_representation_ks', 'continuous')):
        require(canonical(fidelity.pop(key)) == canonical([row for row in marginals if row['kind'] == kind]),
                'representation_alias_changed')
    require(fidelity.pop('author_semantic_categorical_total_variation') ==
            {'status': 'unavailable', 'value': None, 'reason': 'author_semantic_column_metadata_absent'},
            'author_semantic_claim_changed')
    require(fidelity['categorical_contingency_tv'].pop('provenance') == rule['provenance'],
            'contingency_provenance_changed')
    return result, {key: representation[key] for key in ('rule_sha256', 'kind_map_canonical_sha256',
            'kind_map_file_sha256', 'target_included', 'author_semantic_types', 'author_semantic_categorical_claim')}


def public_alias(row, worker_dataset):
    require(row['dataset'] == worker_dataset and re.fullmatch(r'[0-9a-f]{16}', row['dataset']), 'alias_lineage_changed')
    for key in ('method', 'configuration'):
        require(type(row[key]) is str and re.fullmatch(r'[A-Za-z0-9_+.-]{1,80}', row[key]), 'unsafe_public_alias')
    for key in ('fit_seed', 'sample_seed', 'size_multiplier'):
        number(row[key], 'alias_numeric_identity_changed', integer=True)
    return {key: row[key] for key in ('method', 'configuration', 'dataset', 'fit_seed', 'sample_seed', 'size_multiplier')}


def project(snapshot):
    require(snapshot['format'] == 'A952-completed-batch-publication-input-v1'
            and type(snapshot['version']) is int and snapshot['version'] == 1,
            'publication_scope_changed')
    route = snapshot['route']; require(route in ('ROOT', 'TS143'), 'unknown_route')
    require(snapshot['evaluator_ref']['sha256'] == A952 and snapshot['evaluator_schema_ref']['sha256'] == SCHEMA,
            'evaluator_or_schema_changed')
    checked_bytes(snapshot['evaluator_ref']); checked(snapshot['evaluator_schema_ref'])  # Never import evaluator.
    adapter = adapter_from_ref(snapshot['queue_adapter_ref'])
    lock = checked(snapshot['round_ref']); adapter.validate_jobs(lock)
    require((route == 'TS143') == (lock.get('scope') == 'tabsyn_closed143_common_validation_v2'), 'route_round_changed')
    cohort = checked(snapshot['cohort_ref'])
    require(cohort['evaluator_ref']['sha256'] == A952 and cohort['historical_results_pooled'] is False,
            'historical_results_pooled')
    cohort_round = cohort['ROOT_round_ref' if route == 'ROOT' else 'TS143_round_ref']
    require(snapshot['round_ref']['sha256'] == cohort_round['sha256'], 'cohort_round_changed')
    records = checked(snapshot['positive_records_ref']); phase = checked(snapshot['actual_phase_ref'])
    require(type(records) is list and phase['positive_record_manifest_ref'] == snapshot['positive_records_ref']
            and phase['owned_custody_complete'] is True and phase['status'] in ('complete', 'stopped')
            and phase['official_tests_opened'] is False and phase['GPU_queries'] == 0,
            'closed_phase_evidence_missing')
    batch = snapshot['batch_job_sha256s']
    require(type(batch) is list and 0 < len(batch) <= len(lock['jobs']) and len(set(batch)) == len(batch)
            and all(type(key) is str and key in lock['jobs'] for key in batch), 'batch_not_frozen_queue_subset')
    matrix = checked(snapshot['matrix_ref']) if route == 'ROOT' else None
    if matrix is not None:
        require(snapshot['matrix_ref']['sha256'] == lock['matrix']['sha256'], 'matrix_identity_changed')
    aliases = {}
    if matrix is not None:
        for row in matrix['logical_aliases']:
            key = row['metric_job_sha256']
            if key not in lock['jobs']:
                continue
            worker = matrix['workers'][lock['jobs'][key]['worker_key']]
            public = public_alias(row, worker['dataset'])
            require(public not in aliases.setdefault(key, []), 'logical_alias_repeated')
            aliases[key].append(public)
    done = {}; cells = []; seen = set()
    for record in records:
        action, key, stdout = adapter.positive(lock, record, checked)
        require((action, key) not in seen, 'positive_receipt_repeated'); seen.add((action, key))
        if action == 'kind-one':
            continue
        require(key not in done, 'physical_result_repeated')
        done[key] = record
        job = lock['jobs'][key]
        require((job.get('metric_source_sha256') or job.get('evaluator_ref', {}).get('sha256')) == A952,
                'non_A952_cell')
        if key not in batch:
            continue
        config = checked(record['config_ref']); request = checked(config['request_ref'])
        require(type(request['job_sha256']) is str and digest(request['job']) == key
                and canonical(request['job']) == canonical(job), 'exact_job_identity_changed')
        actual = checked(record['operation_ref'])
        # adapter.positive already checks actual wait0, complete empty family, source/runtime and original-parent gates.
        require(type(actual['actual_Popen_exit']) is int and actual['actual_Popen_exit'] == 0, 'exact_actual_exit_missing')
        diagnostics, representation = core_diagnostics(stdout, lock)
        operation_seconds = number(actual['elapsed_seconds'], 'operation_clock_invalid')
        worker_seconds = number(stdout['elapsed_seconds'], 'worker_clock_invalid')
        require(operation_seconds < 600 and worker_seconds < 600, 'operation_cap_changed')
        peak = number(actual['peak_summed_process_resident_bytes'], 'group_peak_invalid', integer=True)
        worker_peak = number(stdout['peak_resident_bytes'], 'worker_peak_invalid', integer=True)
        require(peak <= 16 * 1024 ** 3 and worker_peak <= 16 * 1024 ** 3, 'RAM_cap_changed')
        if route == 'ROOT':
            evaluation = matrix['physical_evaluations'][key]
            require(canonical(evaluation['job']) == canonical(job), 'matrix_job_changed')
            logical = aliases.get(key, []); require(logical, 'logical_alias_missing')
            input_hash = job['sample_sha256']
        else:
            logical = [dict(method='TabSyn', configuration=None, dataset=job['lineage_sha256'],
                            fit_seed=None, sample_seed=job['sample_seed'], size_multiplier=job['row_multiplier'])]
            input_hash = job['recipe_sha256']
        cells.append(dict(physical_job_sha256=key, route=route, input_identity_sha256=input_hash,
            logical_aliases=logical, diagnostics=diagnostics, representation=representation,
            evidence_sha256={name: record[name + '_ref']['sha256'] for name in ('config', 'operation', 'stdout')},
            request_sha256=config['request_ref']['sha256'],
            cost=dict(operation_wall_seconds=operation_seconds, evaluator_worker_seconds=worker_seconds,
                peak_summed_process_resident_bytes=peak, worker_peak_resident_bytes=worker_peak,
                core_seconds=None, energy_joules=None, GPU_queries=0)))
    closed = {row['physical_job_sha256'] for row in cells}
    return dict(format='common-A952-completed-validation-batch', version=1, evaluator_sha256=A952,
        route=route, frozen_queue_job_count=len(lock['jobs']), requested_batch_job_count=len(batch),
        positive_frozen_queue_cell_count=len(done), batch_positive_cell_count=len(cells),
        batch_pending_job_sha256s=sorted(set(batch)-closed), batch_complete=len(closed)==len(batch),
        cohort_complete=len(done)==len(lock['jobs']), actual_phase_status=phase['status'],
        cells=sorted(cells, key=lambda row: row['physical_job_sha256']),
        cost_accounting='one cost per physical evaluator operation; logical aliases do not add costs',
        closed_batch_operation_wall_seconds_sum=sum(row['cost']['operation_wall_seconds'] for row in cells),
        elapsed_campaign_wall_seconds=None, fit_or_sample_seconds_included=False,
        historical_ROOT_1ae_results_pooled=False, historical_flat_report_pooled=False,
        official_tests_opened=False, empirical_only=True, formal_dp=False, hipaa_deidentification=False,
        mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input-ref', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    input_ref = json.loads(args.input_ref, object_pairs_hook=no_duplicates)
    result = project(checked(input_ref)); output = Path(args.output)
    for item in list(READ_REFS.values()):
        checked_bytes(item)
    require(output.is_absolute() and not output.exists()
            and not any(p.is_symlink() for p in (output, *output.parents)), 'new_public_output_required')
    with output.open('xb') as stream:
        stream.write(canonical(result) + b'\n')
    print(json.dumps({'status': 'projected_schema_validation_required', 'bytes': output.stat().st_size,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == '__main__':
    main()
