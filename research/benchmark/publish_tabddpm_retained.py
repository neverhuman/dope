"""Publish existing TabDDPM comparison samples without changing native selection."""
import argparse
import csv
import datetime
import hashlib
import html
import io
import json
import math
from pathlib import Path
import statistics

from research.benchmark import publish_retained_metrics as retained
from research.benchmark.publish_retained_comparison import summarize

GATES = dict(official_tests_opened=False, mfs_v2=None, ptf_v1=None,
             release_safe_l3=None, superiority=None, counts_as_dope_win=False)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verified_bytes(ref, path=None):
    require(type(ref['sha256']) is str and len(ref['sha256']) == 64
            and all(c in '0123456789abcdef' for c in ref['sha256']), 'invalid_frozen_digest')
    file = Path(path or ref.get('mirror_path', ref['path']))
    require(file.is_absolute() and file == file.resolve(strict=True)
            and not any(p.is_symlink() for p in (file, *file.parents))
            and file.stat().st_size <= 64_000_000, 'invalid_frozen_metadata_path')
    raw = file.read_bytes()
    require(type(ref['bytes']) is int and len(raw) == ref['bytes']
            and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'frozen_metadata_changed')
    return raw


def read(ref, path=None):
    return json.loads(verified_bytes(ref, path))


def native_member_refs(selection, original):
    if 'trial_native_receipts' in selection:
        refs = selection['trial_native_receipts']
    else:
        refs = []
        require(selection['all_five_cells_closed'] is True, 'historical_native_grid_not_closed')
        for ref in selection['trial_receipts']:
            trial = original(ref)
            require(digest(trial['job']) == trial['job_sha256'] and trial['official_tests_opened'] is False,
                    'historical_native_trial_changed')
            if 'native.json' in trial['evidence_files']:
                refs.append(dict(path=str(Path(ref['path']).parent / 'native.json'),
                                 sha256=trial['evidence_files']['native.json']))
    return {(ref['path'], ref['sha256']) for ref in refs}


def native_versions(models):
    """Newest original frozen round selects the reporting snapshot, never common metrics."""
    chosen = {}
    for model in models:
        if 'native_selected' not in model['logical_roles']:
            continue
        timestamp = datetime.datetime.fromisoformat(model['frozen_round_utc'].replace('Z', '+00:00'))
        require(timestamp.utcoffset() is not None, 'native_snapshot_timezone_required')
        prior = chosen.get(model['dataset'])
        if prior is not None:
            prior_time = datetime.datetime.fromisoformat(prior['frozen_round_utc'].replace('Z', '+00:00'))
            require(timestamp != prior_time or model['native_selection_sha256'] == prior['native_selection_sha256'],
                    'conflicting_native_snapshot')
            if timestamp <= prior_time:
                continue
        chosen[model['dataset']] = model
    return {dataset: model['job_sha256'] for dataset, model in chosen.items()}


def join_cell(cell, record, original):
    require(type(cell['sample_seed']) is type(cell['row_multiplier']) is int, 'sample_identity_type_changed')
    for key, expected in [('dataset', record['dataset']), ('fit_seed', 11),
                          ('config_sha256', digest(record['job']['config'])),
                          ('charged_artifact_bytes', record['charged_artifact_bytes'])]:
        require(type(cell[key]) is type(expected) and cell[key] == expected, 'fit_metric_identity_changed')
    require(cell['method'] == 'TabDDPM'
            and original['job_identity']['source_fit_job_sha256'] == record['job_sha256']
            and original['job_identity']['matched_receipt_sha256'] == record['matched_closure']['sha256'],
            'metric_original_operation_changed')
    for name, key in [('train.csv', 'train_ref'), ('validation.csv', 'validation_ref'), ('projection.json', 'projection_ref')]:
        require(cell['input_hashes'][key] == record['job']['worker']['files'][name], 'matched_split_changed')
    matches = [s for s in record['samples'] if (s['sample_seed'], s['row_multiplier'])
               == (cell['sample_seed'], cell['row_multiplier'])]
    require(len(matches) == 1 and matches[0]['csv']['sha256'] == cell['input_hashes']['synthetic_ref'],
            'original_sample_binding_changed')


def build(inputs):
    require(inputs['format'] == 'tabddpm-retained-comparison-input-v1'
            and inputs['official_tests_opened'] is False, 'training_validation_required')
    proof = read(inputs['original_verification'])
    require(proof['common_outcomes_read_for_selection'] is False and proof['new_generator_fits'] == 0
            and proof['new_sample_generation'] is False and proof['official_tests_opened'] is False
            and not proof['exclusions'] and len(proof['records']) == 21, 'original_readset_incomplete')
    mirrors = {x['path']: x for x in inputs['metadata_mirrors']}
    require(len(mirrors) == len(inputs['metadata_mirrors']), 'duplicate_metadata_path')

    def original(ref):
        mirror = mirrors[ref['path']]
        require(mirror['sha256'] == ref['sha256']
                and ('bytes' not in ref or mirror['bytes'] == ref['bytes']), 'original_mirror_changed')
        # Historical selection pointers omit size; the frozen mirror binds it.
        return read(dict(ref, bytes=mirror['bytes']), mirror['mirror_path'])

    models = []
    for record in proof['records']:
        job = record['job']
        require(digest(job) == record['job_sha256'] and job['method'] == 'TabDDPM'
                and type(job['fit_seed']) is int and job['fit_seed'] == 11 and job['final'] is False,
                'original_generator_identity_changed')
        request, closure, operation, summary, round_ = [original(record[k]) for k in
            ('matched_request', 'matched_closure', 'matched_operation', 'matched_summary', 'round')]
        require(digest(request['job']) == digest(job) and operation['request_sha256'] == record['matched_request']['sha256']
                and operation['exit_code'] == 0 and operation['status'] == 'ok'
                and closure['operation'] == operation and closure['status'] == summary['status'] == 'ok'
                and summary['sample_replay_exact'] is closure['sample_replay_exact'] is True
                and closure['job_sha256'] == summary['job_sha256'] == record['job_sha256']
                and closure['round_sha256'] == record['round']['sha256'], 'original_sampling_closure_changed')
        require(closure['official_tests_opened'] is summary['official_tests_opened'] is round_['official_tests_opened'] is False,
                'official_tests_must_be_sealed')
        require(record['logical_roles'] == closure['logical_configurations']
                and set(record['logical_roles']) <= {'default', 'native_selected'}, 'native_roles_changed')
        inventory = request['artifact_inventory']
        actual = {Path(x['path']).name: (x['sha256'], x['bytes']) for x in record['artifact_files']}
        require(len(actual) == len(inventory) == 4
                and all(actual.get(x['path']) == (x['sha256'], x['bytes']) for x in inventory)
                and sum(x['bytes'] for x in inventory) == record['charged_artifact_bytes'], 'artifact_accounting_changed')
        for observation in record['current_source_observation']:
            ref = observation['current']
            mirror = mirrors[ref['path']]
            verified_bytes(ref, mirror['mirror_path'])
            require(ref['sha256'] == observation['original_declared_sha256']
                    and observation['matches_original_declaration'] is True, 'original_source_binding_changed')
        selection = original(record['native_selection'])
        require(record['native_selection']['sha256'] == closure['native_selection_sha256']
                and selection['shared_kpi_used_for_selection'] is False
                and selection['selection_frozen_before_shared_sampling'] is True
                and selection['native_objective']['name'] == 'author_five_synthetic_seed_validation_catboost_r2_mean'
                and selection['native_objective']['direction'] == 'maximize', 'native_selection_changed')
        native = original(record['native_receipt'])
        require(type(native['value']) in (int, float) and math.isfinite(native['value'])
                and all(type(x['r2']) in (int, float) and math.isfinite(x['r2']) for x in native['components']),
                'invalid_native_value')
        require(native['job_sha256'] == record['job_sha256'] and native['value'] == record['native_value']
                and native['validation_sha256'] == job['worker']['files']['validation.csv']
                and native['official_tests_opened'] is False and native['partition'] == 'official_training_derived_validation'
                and len(native['components']) == 5 and {x['sample_seed'] for x in native['components']} == set(range(5))
                and math.isclose(statistics.mean(x['r2'] for x in native['components']), native['value'],
                                 rel_tol=1e-12, abs_tol=1e-12), 'native_objective_value_changed')
        members = native_member_refs(selection, original)
        require(members == {(x['receipt']['path'], x['receipt']['sha256']) for x in record['native_candidates']},
                'native_selection_membership_changed')
        candidates = [original(x['receipt']) for x in record['native_candidates']]
        require(len(candidates) == record['native_successful_pool_size'] and 1 <= len(candidates) <= 5
                and record['native_grid_size'] == 5
                and record['native_full_grid_successful'] is (len(candidates) == 5), 'native_pool_changed')
        require(all(n['name'] == native['name'] and n['validation_sha256'] == native['validation_sha256']
                    and n['official_tests_opened'] is False for n in candidates), 'native_pool_split_changed')
        if 'native_selected' in record['logical_roles']:
            require(selection['selected_job_sha256'] == record['job_sha256']
                    and selection['selected_native_value'] == max(n['value'] for n in candidates), 'native_winner_changed')
        require(len(record['samples']) == 6 and {(x['row_multiplier'], x['sample_seed']) for x in record['samples']}
                == {(n, s) for n in (1, 4) for s in (101, 211, 307)}, 'complete_sample_schedule_required')
        models.append(dict(dataset=record['dataset'], job_sha256=record['job_sha256'], fit_seed=11,
            configuration_name=job['configuration_name'], config_sha256=digest(job['config']),
            logical_roles=record['logical_roles'], frozen_round_utc=round_['frozen_utc'],
            artifact_bytes=record['charged_artifact_bytes'], artifact_inventory=inventory,
            artifact_cap_bytes=10240, artifact_cap_exceeded=record['charged_artifact_bytes'] > 10240,
            native_objective=native['name'], native_value=native['value'], native_receipt=record['native_receipt'],
            native_audit_seconds=native['native_seconds'],
            native_successful_pool_size=len(candidates), native_declared_grid_size=5,
            native_full_grid_successful=record['native_full_grid_successful'], native_selection_sha256=record['native_selection']['sha256'],
            matched_receipt=record['matched_closure'], matched_operation_seconds=operation['elapsed_seconds']))
    frontier = native_versions(models)
    execution = read(inputs['execution'])
    process_exit = read(execution['process_exit_ref'])
    launch = read(execution['process_launch_ref'])
    require(execution['exit_code'] == process_exit['exit_code'] == 0
            and execution['elapsed_seconds'] == process_exit['elapsed_seconds']
            and process_exit['manifest_sha256'] == launch['manifest_sha256'] == inputs['input_manifest']['sha256']
            and process_exit['new_generator_fits'] == launch['new_fits'] == 0
            and process_exit['official_tests_opened'] is launch['official_tests_opened'] is False,
            'actual_metric_closure_changed')
    manifest = read(inputs['input_manifest'])
    jobs = {x['job_sha256']: x for x in manifest['jobs']}
    panel = retained.build(inputs['receipt_lock'], inputs['metric_directory'], inputs['input_manifest'], inputs['execution'])
    require(panel['sample_cells'] == 126 and panel['lineages'] == 12 and len(jobs) == 126, 'complete_metric_cohort_required')
    by_fit = {x['job_sha256']: x for x in proof['records']}
    logical = []
    for cell in panel['cells']:
        job = jobs[cell['job_sha256']]
        record = by_fit[job['job_identity']['source_fit_job_sha256']]
        join_cell(cell, record, job)
        for role in record['logical_roles']:
            if role == 'native_selected' and frontier[record['dataset']] != record['job_sha256']:
                continue
            scalar = retained.flatten(cell)
            scalar.update(selection_binding='author_default' if role == 'default' else 'native_selected',
                          original_fit_job_sha256=record['job_sha256'], metric_receipt=cell['receipt_ref'])
            logical.append(scalar)
    summaries = []
    for role in ('author_default', 'native_selected'):
        group = [x for x in logical if x['selection_binding'] == role]
        datasets = sorted({x['dataset'] for x in group})
        # Extra provenance fields are kept in the scalar export, not averaged.
        flat = [{k: v for k, v in x.items() if k not in ('original_fit_job_sha256', 'metric_receipt')} for x in group]
        summaries.extend(summarize(flat, datasets, 'complete_available_' + role))
    require(len(logical) == 132 and len(frontier) == 12, 'logical_reporting_cohort_changed')
    return dict(format='tabddpm-retained-native-default-validation-v1', inputs=inputs,
                physical_panel=panel, checkpoints=models, native_reporting_frontier=frontier,
                native_reporting_rule='latest_original_frozen_round_per_lineage_before_common_outcome_reads',
                rows=logical, summary=summaries, physical_metric_cells=126, logical_metric_cells=132,
                retained_models=21, lineages=12, default_lineages=10, native_selected_lineages=12,
                costs=dict(sum_common_metric_seconds=sum(c['metric_seconds'] for c in panel['cells']),
                           common_process_wall_seconds=execution['elapsed_seconds'],
                           sum_matched_operation_seconds=sum(m['matched_operation_seconds'] for m in models),
                           sum_native_audit_seconds=sum(m['native_audit_seconds'] for m in models),
                           charged_artifact_bytes=sum(m['artifact_bytes'] for m in models),
                           clocks_are_additive=False, new_generator_fits=0),
                full_population_complete=False, five_fit_coverage=False, native_search_complete=False,
                original_runtime_closure_recertified=False, selection_use=False, **GATES)


def render(panel):
    rows = panel['rows']
    csv_buffer = io.StringIO(newline='')
    identity = ['dataset', 'method', 'fit_seed', 'sample_seed', 'row_multiplier',
                'config_sha256', 'selection_binding', 'charged_artifact_bytes']
    fields = identity + sorted(k for k in rows[0] if k not in (*identity, 'metric_receipt'))
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    checkpoints = io.StringIO(newline='')
    fields = ['dataset', 'job_sha256', 'configuration_name', 'fit_seed', 'artifact_bytes', 'native_objective',
              'native_value', 'native_successful_pool_size', 'native_declared_grid_size', 'native_full_grid_successful',
              'catboost_tstr_mse_n', 'catboost_tstr_mse_4n', 'native_receipt_path', 'native_receipt_sha256']
    writer = csv.DictWriter(checkpoints, fieldnames=fields, extrasaction='ignore', lineterminator='\n'); writer.writeheader()
    for model in panel['checkpoints']:
        row = dict(model)
        for n, name in [(1, 'catboost_tstr_mse_n'), (4, 'catboost_tstr_mse_4n')]:
            cells = [x for x in rows if x['original_fit_job_sha256'] == model['job_sha256'] and x['row_multiplier'] == n]
            values = {x['sample_seed']: x['catboost_tstr_loss'] for x in cells}
            row[name] = statistics.mean(values.values()) if len(values) == 3 and all(v is not None for v in values.values()) else None
        row.update(native_receipt_path=model['native_receipt']['path'], native_receipt_sha256=model['native_receipt']['sha256'])
        writer.writerow(row)
    groups = panel['summary']
    columns = [('catboost_retention', 'CatBoost retention'), ('marginal_error_mean', 'Marginal KS/TV'),
               ('c2st_catboost_auc', 'C2ST AUC'), ('distance_mia_auc', 'Distance MIA AUC')]
    height = 165 + 32 * len(groups)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="{height}" viewBox="0 0 1050 {height}">',
           '<rect width="100%" height="100%" fill="white"/><g font-family="DejaVu Sans,sans-serif" fill="#17212b">',
           '<text x="20" y="30" font-size="20">TabDDPM: retained default and native-selected validation samples</text>',
           '<text x="20" y="55" font-size="12">Fit seed 11; three sample seeds per dataset, then dataset median. Cohorts differ; no cross-cohort ranking.</text>']
    for i, (_, label) in enumerate(columns):
        svg.append(f'<text x="{340 + i * 170}" y="85" font-size="13">{html.escape(label)}</text>')
    for i, group in enumerate(groups):
        y = 112 + i * 32
        label = f'{group["selection_binding"]} / {group["row_multiplier"]}n / {group["datasets"]} lineages'
        svg.append(f'<text x="20" y="{y}" font-size="13">{html.escape(label)}</text>')
        for c, (metric, _) in enumerate(columns):
            result = group[metric]; text = 'NA' if result['median'] is None else f'{result["median"]:.6f}'
            svg.append(f'<text x="{340 + c * 170}" y="{y}" font-size="13">{text} ({result["measured_datasets"]})</text>')
    svg.append(f'<text x="20" y="{height - 24}" font-size="12">Native R² selected configurations. Partial successful pools disclosed. Tests sealed; production scores null.</text></g></svg>\n')
    return {'panel.json': (json.dumps(panel, sort_keys=True, indent=2, allow_nan=False) + '\n').encode(),
            'figure-kpis.csv': csv_buffer.getvalue().encode(), 'checkpoint-kpis.csv': checkpoints.getvalue().encode(),
            'validation.svg': '\n'.join(svg).encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = read(dict(path=str(args.inputs.absolute()), bytes=args.inputs.stat().st_size, sha256=args.sha256))
    panel = build(inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, body in render(panel).items():
        (args.output / name).write_bytes(body)


if __name__ == '__main__':
    main()
