"""Deterministic aggregate publication; standard library, metadata only.

Input constants are frozen before adoption. Every bound JSON is hashed before
decode. Original scratch receipts remain necessary for density and Forest phase
attribution; their anchored receipt sets are hashed, never copied into results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

INPUT_PINS = {'measured-costs.json': '721beb8c5afd36dd74ff6930ba7146d599c4010abd99dcad46363b3fc6524d0a', 'component-gaps.json': '81012524ebff8f752f18238e81386d2ccda30f4d4baed07cb143c2177907e700', 'source-proof.json': 'a569b5b9d8a9b54d736a8944e71bdbe136c4f2daf48a1376176ebf39380de3d3'}
HERE = Path(__file__).resolve().parent
SCRATCH = Path('/mnt/fast-scratch/dope-benchmark')
COMPONENT_NAMES = ('utility_transfer', 'driver_fidelity', 'distribution_fidelity',
                   'structure_fidelity', 'coverage_realism', 'compactness')
HARD_GATE_NAMES = (
    'attribute_inference_advantage_max', 'calibration_degradation_max', 'driver_agreement_min',
    'exact_copies_max', 'feature_importance_min_informative_features',
    'feature_importance_spearman_min', 'feature_importance_top_k_jaccard_min',
    'joint_fidelity_min', 'membership_auc_max', 'near_copies_max',
    'nominal_95_coverage_max', 'nominal_95_coverage_min', 'ptf_v1_min',
    'query_p95_normalized_error_max', 'rare_tail_subgroup_retention_min', 'type_i_error_max')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def reject_constant(_):
    raise ValueError('nonfinite JSON number')


def finite_float(text):
    value = float(text)
    require(math.isfinite(value), 'nonfinite JSON number')
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def bound(path, expected, base):
    path, base = Path(path), Path(base)
    require(type(expected) is str and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected), 'invalid digest')
    require(path.is_file() and path.resolve().is_relative_to(base.resolve())
            and not any(p.is_symlink() for p in (path, *path.parents))
            and path.suffix == '.json' and 'evaluator' not in path.parts,
            'metadata evidence path unavailable or outside scope')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'metadata digest changed')
    return json.loads(data, parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=unique_object)


def number(value, nullable=False):
    if nullable and value is None:
        return None
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'invalid measured nonnegative number')
    return value


def count(value):
    require(type(value) is int and value >= 0, 'invalid measured count')
    return value


def aggregate(rows):
    """One charge per physical receipt and phase, irrespective of logical reuse."""
    unique = {}
    for row in rows:
        require(type(row['identity']) is str, 'invalid physical identity')
        number(row['seconds'])
        require(type(row['new_operation_started']) is bool, 'invalid start flag')
        if row['status'] in ('deadline_unstarted', 'deadline_exhausted', 'capacity_deferred'):
            require(row['native_outcome_class'] in ('scheduling_cutoff', 'infrastructure', 'historical'),
                    'scheduling cut relabeled as method failure')
        if row['status'] in ('deadline_unstarted', 'capacity_deferred'):
            require(row['new_operation_started'] is False and row['seconds'] == 0,
                    'unstarted phase acquired operation or cost')
        if row['identity'] in unique:
            require(canonical(unique[row['identity']]) == canonical(row), 'conflicting physical alias')
        else:
            unique[row['identity']] = row
    grouped = defaultdict(list)
    for row in unique.values():
        grouped[(row['method'], row['phase'])].append(row)
    result = []
    for (method, phase), group in sorted(grouped.items()):
        ram = [number(r['peak_ram_bytes']) for r in group if r['peak_ram_bytes'] is not None]
        vram = [number(r['peak_device_used_mib']) for r in group if r['peak_device_used_mib'] is not None]
        result.append(dict(method=method, phase=phase, distinct_phase_receipts=len(group),
            operation_wall_seconds=math.fsum(r['seconds'] for r in group),
            new_phase_operations_started=sum(r['new_operation_started'] for r in group),
            native_outcome_class_counts=dict(Counter(r['native_outcome_class'] for r in group)),
            status_counts=dict(Counter(r['status'] for r in group)),
            hosts=sorted({r['host'] for r in group if r['host'] is not None}),
            peak_ram_bytes=max(ram, default=None), ram_observed_operations=len(ram),
            peak_device_used_mib=max(vram, default=None), vram_observed_operations=len(vram),
            co_tenant=None, shared_gpu_timing='not_recorded_in_historical_publication',
            gpu_model=None, cpu_model=None, attributed_energy_joules=None))
    return result


def validate_costs(costs):
    require(all(costs[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
            'cost acquired score claim')
    identities = set()
    for row in costs['rows']:
        key = (row['method'], row['phase'])
        require(key not in identities, 'duplicate phase summary')
        identities.add(key)
        number(row['operation_wall_seconds'])
        for name in ('distinct_phase_receipts', 'new_phase_operations_started',
                     'ram_observed_operations', 'vram_observed_operations'):
            count(row[name])
        for name in ('native_outcome_class_counts', 'status_counts'):
            require(sum(count(v) for v in row[name].values()) == row['distinct_phase_receipts'],
                    'phase count inconsistent')
        for name in ('peak_ram_bytes', 'peak_device_used_mib'):
            number(row[name], nullable=True)
        require(row['new_phase_operations_started'] <= row['distinct_phase_receipts']
                and row['ram_observed_operations'] <= row['distinct_phase_receipts']
                and row['vram_observed_operations'] <= row['distinct_phase_receipts'],
                'observed operation counts exceed receipts')
        require(row['co_tenant'] is None and row['gpu_model'] is None and row['cpu_model'] is None
                and row['attributed_energy_joules'] is None, 'unsupported hardware or tenancy claim')
    identities = set()
    for row in costs['runs']:
        require(row['identity'] not in identities, 'duplicate run cost')
        identities.add(row['identity'])
        number(row['operation_wall_seconds'])
        if 'scheduler_wall_seconds' in row:
            number(row['scheduler_wall_seconds'])
        for value in row.get('host_operation_seconds', {}).values():
            number(value)
        for field in ('peak_ram_bytes', 'peak_device_used_mib'):
            if field in row:
                number(row[field], nullable=True)
    for value in costs['historical_prior_costs'].values():
        number(value)


def scratch_operations(reports, proof, scratch):
    rows, receipt_sets = [], {}
    def row(pin, method, phase, seconds, host, status, started, outcome, ram=None, vram=None):
        rows.append(dict(identity=pin + ':' + phase, method=method, phase=phase, seconds=number(seconds),
            host=host, status=status, new_operation_started=started, native_outcome_class=outcome,
            peak_ram_bytes=ram, peak_device_used_mib=vram))
    for name in ('density_native', 'density_shared', 'forest_native'):
        custody = proof['scratch_custody'][name]
        anchor_path = Path(custody['anchor']['path'])
        anchor = bound(anchor_path, custody['anchor']['sha256'], scratch)
        refs = []
        if name.startswith('density'):
            report_ref = custody['reconciliation']
            require(anchor['reconciliation_sha256'] == report_ref['sha256'], 'density reconciliation anchor changed')
            report = bound(report_ref['path'], report_ref['sha256'], scratch)
            if name == 'density_native':
                entries = report['trials']
                for trial in entries:
                    path, pin = trial['attempt_receipt_path'], trial['attempt_receipt_sha256']
                    require(anchor['refs'][path] == pin, 'native receipt anchor changed')
                    receipt = bound(path, pin, scratch)
                    require(receipt['identity']['method'] == trial['method']
                            and receipt['status'] == trial['status']
                            and canonical(receipt['wall_seconds']) == canonical(trial['seconds']),
                            'native receipt cost or identity changed')
                    row(pin, trial['method'], 'historical_fit_native', trial['seconds'], None,
                        trial['status'], False, 'historical')
                    refs.append([path, pin])
            else:
                require(report['complete_matrix'] is True, 'density physical matrix incomplete')
                for batch in report['batches']:
                    path = str(anchor_path.parent / 'attempts' / batch['physical_job_sha256'] / 'attempt-0001/receipt.json')
                    pin = batch['receipt_sha256']
                    require(anchor['refs'][path] == pin, 'density sample receipt anchor changed')
                    receipt = bound(path, pin, scratch)
                    operation = receipt['operation']
                    require(digest(receipt['job']) == batch['physical_job_sha256']
                            and canonical(operation['elapsed_seconds']) == canonical(batch['operation_seconds']),
                            'density physical identity or cost changed')
                    row(pin, receipt['job']['method'], 'sample_and_evaluate_combined', operation['elapsed_seconds'],
                        operation['host'], receipt['status'], operation['new_operation_started'],
                        'ok' if receipt['status'] == 'ok' else 'infrastructure',
                        operation.get('peak_resident_bytes_including_coordinator'))
                    refs.append([path, pin])
        else:
            for trial in reports['s3-matched-forest-confirmation-validation.json']['native_attempts']:
                ref = trial['receipt']; path, pin = ref['path'], ref['sha256']
                require(anchor['refs'][path] == pin, 'Forest original receipt anchor changed')
                receipt = bound(path, pin, scratch)
                require(receipt['status'] == trial['status'] and receipt['job']['dataset'] == trial['dataset']
                        and len(receipt['operations']) == 3, 'Forest phase receipt changed')
                for phase, operation in zip(('fit', 'native', 'sample'), receipt['operations']):
                    row(pin, 'Forest-Flow', phase, operation['elapsed_seconds'], operation['host'],
                        operation['status'], operation['new_operation_started'], 'ok')
                refs.append([path, pin])
        require(len(refs) == custody['original_receipt_count'] and digest(sorted(refs)) == custody['original_receipt_set_sha256'],
                'original receipt set changed')
        receipt_sets[name] = len(refs)
    return rows, receipt_sets


def verify_run_costs(costs, reports):
    population = reports['dope-s3-population-gpu-fits.json']['cost']
    refinement = reports['dope-target-refinement-population-fits.json']['cost']
    specifications = {
        'dope_population_research_v2': ('dope-s3-population-gpu-fits.json',
            population['new_fit_operation_seconds'], {
                'host_operation_seconds': population['host_operation_seconds'],
                'scheduler_wall_seconds': population['coordinator_wall_seconds'],
                'peak_device_used_mib': population['peak_observed_device_used_mib']}),
        'dope_population_validation_v1': ('dope-s3-population-validation.json',
            reports['dope-s3-population-validation.json']['cost']['shared_sampler_and_evaluator_seconds'], {
                'scheduler_wall_seconds': reports['dope-s3-population-validation.json']['cost']['coordinator_wall_seconds']}),
        'dope_refinement_expansion_v2': ('dope-target-refinement-population-fits.json',
            refinement['new_operation_seconds'], {
                'host_operation_seconds': refinement['host_operation_seconds'],
                'scheduler_wall_seconds': refinement['scheduler_wall_seconds']}),
        'dope_refinement_expansion_validation_v2': ('dope-target-refinement-population-validation.json',
            reports['dope-target-refinement-population-validation.json']['validation_operation_seconds'], {
                'scheduler_wall_seconds': reports['dope-target-refinement-population-validation.json']['validation_scheduler_wall_seconds']}),
        'sdv_shared_validation_v1': ('sdv-matched-population-validation.json',
            reports['sdv-matched-population-validation.json']['cost']['shared_metric_operation_seconds'], {
                'scheduler_wall_seconds': reports['sdv-matched-population-validation.json']['cost']['shared_coordinator_wall_seconds']}),
        'density_shared_validation_v1': ('density-matched-population-validation.json',
            reports['density-matched-population-validation.json']['cost']['density']['operation_seconds'], {})
    }
    require({row['identity'] for row in costs['runs']} == set(specifications),
            'measured run coverage changed')
    for row in costs['runs']:
        source, seconds, fields = specifications[row['identity']]
        method = ('CTGAN+TVAE' if row['identity'] == 'sdv_shared_validation_v1'
                  else 'density_methods_combined' if row['identity'] == 'density_shared_validation_v1'
                  else 'DOPE')
        phase = ('fit_wrapper' if row['identity'] in ('dope_population_research_v2',
                 'dope_refinement_expansion_v2') else 'sample_and_evaluate_combined')
        if row['identity'] == 'dope_population_validation_v1':
            fields = dict(fields, host=reports[source]['cost']['host'])
        require(row['source'] == source
                and row['method'] == method and row['phase'] == phase
                and canonical(row['operation_wall_seconds']) == canonical(seconds)
                and all(canonical(row[name]) == canonical(value) for name, value in fields.items()),
                'measured run source or cost changed')
        expected_fields = {'identity', 'method', 'phase', 'operation_wall_seconds', 'source', *fields}
        unavailable_field = {'sdv_shared_validation_v1': 'phase_attribution_unavailable',
                             'density_shared_validation_v1': 'method_attribution_unavailable'}.get(row['identity'])
        if unavailable_field:
            expected_fields.add(unavailable_field)
            require(row.get(unavailable_field) is (unavailable_field == 'phase_attribution_unavailable'),
                    'run phase attribution claim changed')
        require(set(row) == expected_fields, 'unsupported measured run provenance field')
    expected_prior = {
        'dope_closed_fit_seconds': population['prior_closed_fit_seconds'],
        'dope_prior_wrapper_transport_seconds': population['prior_wrapper_new_operation_transport_seconds'],
        'dope_discovery_confirmation_seconds': refinement['prior_whole_discovery_confirmation_operation_seconds']}
    require(canonical(costs['historical_prior_costs']) == canonical(expected_prior),
            'historical prior cost changed')


def verify_sources(repo, costs, gaps, proof, scratch):
    reports = {name: bound(repo / 'research/benchmark/results' / name, pin, repo)
               for name, pin in costs['source_reports_sha256'].items()}
    for ref in proof['source_code_and_contract']:
        path = repo / ref['path']
        require(path.resolve().is_relative_to(repo.resolve()) and path.is_file()
                and not any(p.is_symlink() for p in (path, *path.parents))
                and hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'], 'source contract path or digest changed')
    validate_proof(proof)
    verify_run_costs(costs, reports)
    scratch_rows, counts = scratch_operations(reports, proof, scratch)
    rows = []
    for trial in reports['sdv-s3-population-native.json']['trials']:
        for operation in trial['operation_evidence']:
            rows.append(dict(identity=trial['receipt']['sha256'] + ':' + operation['phase'],
                method=trial['method'], phase=operation['phase'], seconds=float(number(operation['elapsed_seconds'])),
                host=operation['host'], status=operation['status'], new_operation_started=operation['new_operation_started'],
                native_outcome_class=trial['outcome_class'], peak_ram_bytes=operation['peak_resident_bytes'],
                peak_device_used_mib=operation['peak_gpu_used_mib']))
    for trial in reports['arf-s3-population-native.json']['trials']:
        rows.append(dict(identity=trial['receipt_sha256'] + ':fit_and_native_wrapper', method='ARF',
            phase='fit_and_native_wrapper', seconds=float(number(trial['operation_seconds'])), host=trial['host'],
            status=trial['status'], new_operation_started=trial['new_operation_started'],
            native_outcome_class=trial['outcome_class'], peak_ram_bytes=trial['peak_resident_bytes_including_coordinator'],
            peak_device_used_mib=None))
    rows.extend(scratch_rows)
    require(canonical(aggregate(rows)) == canonical(costs['rows']), 'physical phase aggregates changed')
    require(len(rows) == proof['physical_operations'] and digest(rows) == proof['physical_operation_metadata_sha256'],
            'original physical metadata proof changed')
    density = reports['density-matched-population-validation.json']['cost']['density']
    require(math.isclose(math.fsum(r['seconds'] for r in scratch_rows if r['phase'] == 'historical_fit_native'),
                         density['prior_native_trial_operation_seconds'], rel_tol=1e-12, abs_tol=1e-9), 'density native total changed')
    require(math.isclose(math.fsum(r['seconds'] for r in scratch_rows if r['method'] == 'Forest-Flow'),
                         reports['s3-matched-forest-confirmation-validation.json']['cost']['forest_fit_native_sample_operation_seconds'],
                         rel_tol=1e-12, abs_tol=1e-9), 'Forest phase total changed')
    fit = reports['dope-target-refinement-population-fits.json']
    for reference in (gaps['source_result'], gaps['byte_charge_source_result']):
        require(reference['path'] == 'research/benchmark/results/' + Path(reference['path']).name
                and reference['sha256'] == costs['source_reports_sha256'][Path(reference['path']).name],
                'component report provenance changed')
    charges = [r for r in fit['cells'] if r['charged_artifact_bytes'] is not None]
    require(len(fit['cells']) == gaps['fit_cells'] and len(charges) == gaps['measured_charges']
            and sum(r['status'] == 'ok' for r in charges) == gaps['successful_charges']
            and sum(r['status'] == 'charged_artifact_cap' for r in charges) == gaps['cap_failed_charges'],
            'compactness charge provenance changed')
    return counts


def schema(value, name=''):
    """Same strict shape concept as existing publishers; all observed array shapes."""
    if value is None:
        return {'type': 'null'}
    if type(value) is bool:
        return {'const': value}
    if type(value) is dict:
        return dict(type='object', additionalProperties=False, required=sorted(value),
                    properties={k: schema(v, k) for k, v in value.items()})
    if type(value) is list:
        shapes = {canonical(schema(v)): schema(v) for v in value}
        return dict(type='array', minItems=len(value), maxItems=len(value),
                    items=next(iter(shapes.values())) if len(shapes) == 1 else {'anyOf': list(shapes.values())} if shapes else {})
    if type(value) in (int, float):
        return {'type': 'integer' if type(value) is int else 'number', 'minimum': 0}
    if name.endswith('sha256'):
        return {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}
    return {'const': value}


def validate_gaps(gaps):
    for key in ('fit_cells', 'measured_charges', 'successful_charges', 'cap_failed_charges', 'unmeasured_charges'):
        count(gaps[key])
    require(gaps['fit_cells'] == 200 and gaps['measured_charges'] == 138
            and gaps['successful_charges'] == 136 and gaps['cap_failed_charges'] == 2
            and gaps['unmeasured_charges'] == 62, 'component charge counts changed')
    require(gaps['scalar_formula_changed'] is False and gaps['official_tests_opened'] is False
            and gaps['production_certified'] is False, 'component ledger acquired claim')
    require(type(gaps['new_metric_jobs_started']) is int and gaps['new_metric_jobs_started'] == 0,
            'component ledger acquired metric execution claim')
    for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority', 'hard_gate_passed'):
        require(gaps[key] is None, 'component scalar or hard gate claim changed')
    require(tuple(row['name'] for row in gaps['components']) == COMPONENT_NAMES,
            'component coverage or identity changed')
    require(tuple(row['name'] for row in gaps['hard_gates']) == HARD_GATE_NAMES,
            'hard gate coverage or identity changed')
    for component in gaps['components']:
        require(component['score'] is None, 'aggregate component score is not evidenced')
        require(component['known_fit_values'] == (138 if component['name'] == 'compactness' else 0),
                'component known-value count changed')
        count(component['known_fit_values'])
        require(type(component['missing_reason']) is str and bool(component['missing_reason']), 'missing component reason')
    for gate in gaps['hard_gates']:
        require(gate['passed'] is None and gate['status'] == 'not_evidenced', 'hard gate claim changed')


def validate_proof(proof):
    for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority',
                'attributed_energy_joules', 'hardware_models', 'historical_co_tenant'):
        require(key in proof and proof[key] is None, 'source proof acquired unsupported claim')
    for key in ('new_metric_jobs_started', 'new_training_or_samples_started'):
        require(type(proof.get(key)) is int and proof[key] == 0, 'source proof acquired execution claim')
    require(proof.get('official_tests_opened') is False
            and proof.get('logical_aliases_charged_again') is False, 'source proof acquired test or double-charge claim')


def cost_tables(costs):
    fields = ['kind', 'method', 'phase', 'operation_wall_seconds', 'physical_phase_receipts', 'new_phase_operations_started', 'accounting']
    stream = io.StringIO(); writer = csv.writer(stream, lineterminator='\n'); writer.writerow(fields)
    lines = ['# Measured research operation costs', '',
        'Phase receipts are deduplicated physical operations. Wall seconds include wrapper overhead. Combined sample/evaluator timing stays combined. These entries do not establish equal research spend.', '',
        '| Method | Phase | Receipts | Operation seconds |', '|---|---|---:|---:|']
    for row in costs['rows']:
        writer.writerow(['phase', row['method'], row['phase'], row['operation_wall_seconds'], row['distinct_phase_receipts'], row['new_phase_operations_started'], 'distinct_physical_phase'])
        lines.append(f"| {row['method']} | {row['phase']} | {row['distinct_phase_receipts']} | {row['operation_wall_seconds']:.6f} |")
    lines += ['', '| Run | Method | Phase | Operation seconds | Accounting |', '|---|---|---|---:|---|']
    for row in costs['runs']:
        accounting = 'alias_of_density_phase_rows_do_not_add' if row['identity'] == 'density_shared_validation_v1' else 'separate_run_aggregate'
        writer.writerow(['run', row['method'], row['phase'], row['operation_wall_seconds'], '', '', accounting])
        lines.append(f"| {row['identity']} | {row['method']} | {row['phase']} | {row['operation_wall_seconds']:.6f} | {accounting} |")
    lines += ['', 'Prior pilot costs remain separate; no historical grand total is reported. Scheduling cutoffs retain their native outcome class and are not method failures. Whole-device VRAM observations are not process VRAM or attributed energy.', '',
              'Historical co-tenancy is unknown. Future shared GPU claims require clock-bound receipt observations; historical omissions cannot establish exclusive use.', '']
    lines += ['Missing evidence:'] + ['- ' + reason for reason in costs['missing_evidence']] + ['']
    return stream.getvalue().encode(), '\n'.join(lines).encode()


def gap_tables(gaps):
    stream = io.StringIO(); writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['component', 'aggregate_score', 'known_fit_values', 'missing_reason'])
    lines = ['# MFS component evidence gaps', '',
        'Compactness has 138 measured per-fit values: 136 successful charges and two over-cap charges (compactness zero). Another 62 fits lack a measured charge. Model and projection bytes are fully charged.', '',
        '| Component | Known fit values | Aggregate score | Missing evidence |', '|---|---:|---|---|']
    for row in gaps['components']:
        writer.writerow([row['name'], '', row['known_fit_values'], row['missing_reason']])
        lines.append(f"| {row['name']} | {row['known_fit_values']} | null | {row['missing_reason']} |")
    lines += ['', 'All hard gate outcomes, MFS-v2, PTF-v1, release-safe and superiority remain null. Production contract bytes and the scalar formula are unchanged. No official test data was opened and no metric job was started.', '', 'Missing global gates:']
    lines += ['- ' + reason for reason in gaps['missing_global_gates']] + ['']
    return stream.getvalue().encode(), '\n'.join(lines).encode()


def projections(costs, gaps, proof):
    validate_costs(costs); validate_gaps(gaps); validate_proof(proof)
    result = {}
    published_costs = dict(costs, format='dope-measured-cost-aggregates', version=1,
        scope='Closed validation research receipts; no equal total research-spend claim',
        run_accounting={r['identity']: 'alias_of_density_phase_rows_do_not_add' if r['identity'] == 'density_shared_validation_v1'
                        else 'separate_run_aggregate' for r in costs['runs']},
        historical_prior_costs_additive=False, release_safe=None, superiority=None)
    for name, document, tables in (('measured-costs', published_costs, cost_tables), ('component-gaps', gaps, gap_tables)):
        result[name + '.json'] = encode(document)
        result[name + '.schema.json'] = encode({'$schema': 'https://json-schema.org/draft/2020-12/schema', **schema(document)})
        result[name + '.csv'], result[name + '.md'] = tables(document)
    require(len(result['measured-costs.json']) <= 20000, 'cost publication exceeds byte budget')
    result['source-proof.json'] = encode(proof)
    require(len(result['source-proof.json']) <= 20000, 'source proof exceeds byte budget')
    result['source-proof.schema.json'] = encode({'$schema': 'https://json-schema.org/draft/2020-12/schema', **schema(proof)})
    return result


def publish(repo, inputs, output, scratch=SCRATCH):
    costs, gaps, proof = [bound(inputs / name, INPUT_PINS[name], inputs)
                          for name in ('measured-costs.json', 'component-gaps.json', 'source-proof.json')]
    validate_costs(costs); validate_gaps(gaps)
    receipt_counts = verify_sources(repo, costs, gaps, proof, scratch)
    files = projections(costs, gaps, proof)
    manifest = dict(format='dope-cost-component-publication-manifest', version=1,
        input_sha256=INPUT_PINS, publisher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifacts={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)) for name, data in sorted(files.items())},
        original_receipts_verified=receipt_counts, new_training_or_samples_started=0, new_metric_jobs_started=0,
        official_tests_opened=False, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)
    files['publication-manifest.json'] = encode(manifest)
    files['publication-manifest.schema.json'] = encode({'$schema': 'https://json-schema.org/draft/2020-12/schema', **schema(manifest)})
    output.mkdir(parents=True, exist_ok=True)
    require(not any(path.is_symlink() for path in (output, *output.parents)), 'output directory alias rejected')
    # Existing frozen files are checked before any output write.
    for name, data in files.items():
        require(not (output / name).is_symlink() and
                (not (output / name).exists() or (output / name).read_bytes() == data), 'existing publication differs')
    for name, data in files.items():
        if not (output / name).exists():
            with (output / name).open('xb') as stream:
                stream.write(data)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=HERE / 'results' / 'work-order-b-costs-v1')
    parser.add_argument('--output', type=Path, default=HERE / 'results' / 'work-order-b-costs-v1')
    args = parser.parse_args()
    manifest = publish(args.repo, args.inputs, args.output)
    print(json.dumps(dict(artifacts=len(manifest['artifacts']), original_receipts_verified=manifest['original_receipts_verified']), sort_keys=True))


if __name__ == '__main__':
    main()
