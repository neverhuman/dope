"""Reconcile five independent fit seeds without pooling sample replicates."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from research.benchmark.publish_retained_metrics import flatten

FIT_SEEDS = {11, 23, 37, 53, 71}
SAMPLE_SEEDS = {101, 211, 307}
GATES = dict(official_tests_opened=False, mfs_v2=None, ptf_v1=None,
             release_safe_l3=None, superiority=None, counts_as_dope_win=False)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verified(ref):
    path = Path(ref.get('mirror_path', ref['path']))
    require(type(ref['sha256']) is str and len(ref['sha256']) == 64, 'invalid_frozen_digest')
    require(path.is_absolute() and path == path.resolve(strict=True)
            and not any(p.is_symlink() for p in (path, *path.parents))
            and path.name not in ('test.csv', 'test.tsv') and path.stat().st_size <= 64_000_000,
            'invalid_metadata_path')
    raw = path.read_bytes()
    require(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'],
            'frozen_metadata_changed')
    return json.loads(raw)


def complete_mean(values):
    return statistics.mean(values) if all(v is not None for v in values) else None


def fit_statistic(values):
    require(len(values) == 5, 'five_independent_fits_required')
    require(all(v is None or type(v) in (int, float) and math.isfinite(v) for v in values),
            'invalid_metric_value')
    measured = sum(v is not None for v in values)
    return dict(mean=complete_mean(values),
                between_fit_sample_sd=statistics.stdev(values) if measured == 5 else None,
                measured_fits=measured, unavailable_fits=5 - measured)


def build(inputs):
    if inputs['format'] == 'forest-two-lineage-fivefit-input-v1':
        lineages = 2
    else:
        require(inputs['format'] == 'forest-complete-cohort-fivefit-input-v1', 'input_format_changed')
        lineages = inputs['lineages']
        require(type(lineages) is int and 1 <= lineages <= 100, 'invalid_lineage_count')
    expected_fits, expected_cells = lineages * 5, lineages * 30
    require(len(inputs['fits']) == expected_fits, 'complete_fit_cohort_required')
    require(inputs['official_tests_opened'] is False, 'official_tests_must_be_sealed')
    proof = verified(inputs['original_verification'])
    old_inputs = verified(inputs['original_verification_inputs'])
    require(proof['actual_complete'] is True and proof['fit_checkpoints'] == expected_fits
            and proof['input_sha256'] == inputs['original_verification_inputs']['sha256']
            and proof['original_checkpoint_bytes_verified'] is True
            and proof['official_tests_opened'] is False and proof['pickle_loaded'] is False,
            'original_checkpoint_verification_required')
    originals = {r['path']: (r['sha256'], r['bytes']) for r in proof['refs']}
    require(len(originals) == len(proof['refs']), 'duplicate_original_verification_path')
    require([(e['fit']['path'], e['fit']['sha256']) for e in inputs['fits']]
            == [(e['fit']['path'], e['fit']['sha256']) for e in old_inputs['fits']],
            'original_verification_cohort_changed')
    cells = []
    implementations = set()
    for batch in inputs['metric_batches']:
        lock, manifest, actual = (verified(batch[k]) for k in ('receipt_lock', 'input_manifest', 'process_exit'))
        require(lock['actual_complete'] is True and lock['count'] == len(lock['receipts'])
                and lock['official_tests_opened'] is False and actual['exit_code'] == 0
                and actual['manifest_sha256'] == batch['input_manifest']['sha256']
                and lock['manifest_sha256'] == digest(manifest), 'metric_batch_not_complete')
        jobs = {j['job_sha256']: j for j in manifest['jobs']}
        require(len(jobs) == len(manifest['jobs']) == lock['count'], 'metric_job_matrix_changed')
        for ref in lock['receipts']:
            cell = verified(ref)
            require(cell['status'] == 'ok' and cell['job_sha256'] == digest(cell['job_identity'])
                    and cell['job_identity'] == jobs[cell['job_sha256']]['job_identity'], 'metric_identity_changed')
            original = jobs[cell['job_sha256']]['evaluator_input']
            for key in ('dataset', 'fit_seed', 'sample_seed', 'row_multiplier', 'config_sha256',
                        'selection_binding', 'charged_artifact_bytes'):
                require(type(cell[key]) is type(original[key]) and cell[key] == original[key], 'metric_identity_changed')
            for key, value in cell['input_refs'].items():
                require(value == original[key], 'metric_input_binding_changed')
            require(cell['official_tests_opened'] is False and cell['metrics']['official_tests_opened'] is False
                    and all(cell[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority')),
                    'unsupported_metric_claim')
            implementations.add(tuple(sorted((k, v['sha256']) for k, v in cell['source_refs'].items())))
            cells.append(dict(cell, receipt_ref=ref))
    require(len(cells) == expected_cells and len({c['job_sha256'] for c in cells}) == expected_cells
            and len(implementations) == 1, 'complete_common_metric_cohort_required')
    native_members = {}
    for batch in inputs['native_batches']:
        lock, complete = (verified(batch[k]) for k in ('round', 'completion'))
        require(complete['actual_complete'] is True and complete['count'] == len(complete['receipts'])
                == len(lock['fits']) and complete['native_round_sha256'] == batch['round']['sha256']
                and complete['official_tests_opened'] is False, 'native_batch_not_complete')
        for ref in complete['receipts']:
            require(ref['path'] not in native_members, 'duplicate_native_receipt')
            native_members[ref['path']] = (ref['sha256'], batch['round']['sha256'])
    require(len(native_members) == expected_fits, 'complete_native_objective_receipts_required')
    fits = []
    for entry in inputs['fits']:
        fit, job, close, native = (verified(entry[k]) for k in ('fit', 'job', 'close', 'native'))
        round_lock = verified(entry['round'])
        for key in ('fit', 'job', 'close', 'native', 'round'):
            ref = entry[key]
            require(originals.get(ref['path']) == (ref['sha256'], ref['bytes']),
                    'original_metadata_verification_gap')
        for ref in fit['artifact_inventory']:
            artifact_path = str(Path(entry['fit']['path']).parent / 'artifact' / ref['path'])
            require(originals.get(artifact_path) == (ref['sha256'], ref['bytes']),
                    'original_artifact_verification_gap')
        for path, sha in round_lock['source_files'].items():
            require(path in originals and originals[path][0] == sha, 'original_source_verification_gap')
        require(native_members.get(entry['native']['path']) == (entry['native']['sha256'], native['native_round_sha256'])
                and fit['round_sha256'] == close['round_sha256'] == entry['round']['sha256']
                and any(j['job_sha256'] == fit['job_sha256'] and j['file_sha256'] == entry['job']['sha256']
                        for j in round_lock['jobs']), 'fit_not_in_frozen_round')
        require(fit['status'] == native['status'] == 'ok' and close['worker_exit_code'] == 0
                and close['fit_receipt_sha256'] == native['fit_ref']['sha256'] == entry['fit']['sha256']
                and fit['job_file_sha256'] == native['job_ref']['sha256'] == entry['job']['sha256']
                and fit['job_sha256'] == digest(job), 'fit_native_closure_changed')
        require(type(job['fit_seed']) is type(fit['fit_seed']) is type(native['fit_seed']) is int
                and fit['fit_seed'] == native['fit_seed'] == job['fit_seed']
                and job['fit_seed'] in FIT_SEEDS and job['configuration_name'] == 'author_default'
                and fit['method'] == 'Forest-Flow' and job['method'] == 'ForestDiffusion/Forest-Flow'
                and fit['retained_training_row_containers'] is False and fit['original_sample_parity'] == 'exact'
                and fit['dataset'] == native['dataset'] == job['dataset'], 'native_fit_identity_changed')
        require(fit['official_tests_opened'] is False and native['official_tests_opened'] is False
                and native['native_selection_changed'] is False and native['new_tuning_trials'] == 0
                and native['new_generator_fits'] == 0, 'unsupported_native_claim')
        objective = native['native']
        scores = objective['auditor_seed_scores']
        require(objective['name'] == 'author_mean_ml_r2' and objective['direction'] == 'maximize'
                and objective['implementation'] == 'study_implemented_author_four_model_five_seed_formula'
                and objective['shared_kpi_used_for_selection'] is False
                and objective['auditor_fit_seeds'] == [0, 1, 2, 3, 4]
                and set(scores) == {'adaboost', 'linear', 'random_forest', 'xgboost'}
                and all(len(v) == 5 for v in scores.values())
                and math.isclose(statistics.mean(v for values in scores.values() for v in values),
                                 objective['value'], rel_tol=1e-12, abs_tol=1e-12), 'native_author_objective_changed')
        require(sum(r['bytes'] for r in fit['artifact_inventory']) == fit['artifact_bytes']
                and {r['path'] for r in fit['artifact_inventory']} == {'sampler.pkl', 'projection.json', 'adapter.json'},
                'incomplete_artifact_accounting')
        selected = [c for c in cells if (c['dataset'], c['fit_seed']) == (fit['dataset'], fit['fit_seed'])]
        require(len(selected) == 6 and {(c['row_multiplier'], c['sample_seed']) for c in selected}
                == {(n, s) for n in (1, 4) for s in SAMPLE_SEEDS}, 'complete_fit_sample_schedule_required')
        for cell in selected:
            require(cell['job_identity']['fit_receipt_sha256'] == entry['fit']['sha256']
                    and cell['job_identity']['source_fit_job_sha256'] == fit['job_sha256']
                    and cell['config_sha256'] == digest(job['config'])
                    and cell['charged_artifact_bytes'] == fit['artifact_bytes']
                    and cell['selection_binding'] == 'author_default'
                    and all(cell['input_refs'][k]['sha256'] == job['worker']['files'][name]
                            for k, name in [('train_ref', 'train.csv'), ('validation_ref', 'validation.csv'),
                                            ('projection_ref', 'projection.json')]), 'fit_metric_join_changed')
        fits.append(dict(dataset=fit['dataset'], fit_seed=fit['fit_seed'], job_sha256=fit['job_sha256'],
            configuration_sha256=digest(job['config']), artifact_bytes=fit['artifact_bytes'],
            artifact_inventory=fit['artifact_inventory'], native_objective=objective['name'],
            native_value=objective['value'], native_implementation=objective['implementation'],
            native_auditor_seed_scores=scores,
            fit_seconds=fit['fit_seconds'], whole_fit_seconds=fit['elapsed_seconds'], native_seconds=native['elapsed_seconds'],
            train_sha256=job['worker']['files']['train.csv'], validation_sha256=job['worker']['files']['validation.csv'],
            fit_ref=entry['fit'], close_ref=entry['close'], native_ref=entry['native'], job_ref=entry['job']))
    per_fit, summaries = summarize(fits, cells, lineages)
    public_cells = []
    for cell in cells:
        public = dict(cell)
        identity = dict(public.pop('job_identity'))
        identity['worker_partition_sha256'] = identity.pop('worker_key')
        public['job_identity_projection'] = identity
        public_cells.append(public)
    return dict(format='forest-default-fivefit-variance-panel-v1', inputs=inputs, fits=fits, cells=public_cells, per_fit=per_fit,
        summary=summaries, lineages=lineages, generator_fits=expected_fits, common_sample_cells=expected_cells,
        fit_seeds=sorted(FIT_SEEDS), sample_seeds=sorted(SAMPLE_SEEDS), row_multipliers=[1, 4],
        within_fit_aggregation='mean_of_three_samples', between_fit_aggregation='mean_and_unbiased_sample_standard_deviation',
        standard_deviation_is_not_confidence_interval=True, full_population_complete=False,
        native_selected_population_complete=False, five_fit_default_cohort_complete=True, **GATES)


def summarize(fits, cells, lineages=2):
    datasets = sorted({f['dataset'] for f in fits})
    require(type(lineages) is int and 1 <= lineages <= 100, 'invalid_lineage_count')
    require(len(fits) == lineages * 5 and len(datasets) == lineages
            and len({(f['dataset'], f['fit_seed']) for f in fits}) == lineages * 5,
            'ten_distinct_fits_required' if lineages == 2 else 'complete_distinct_fits_required')
    per_fit, summaries = [], []
    for dataset in datasets:
        selected = sorted((f for f in fits if f['dataset'] == dataset), key=lambda f: f['fit_seed'])
        require({f['fit_seed'] for f in selected} == FIT_SEEDS
                and len({(f['train_sha256'], f['validation_sha256'], f['configuration_sha256']) for f in selected}) == 1,
                'same_split_and_configuration_required')
        for n in (1, 4):
            rows = []
            for fit in selected:
                group = [c for c in cells if (c['dataset'], c['fit_seed'], c['row_multiplier']) == (dataset, fit['fit_seed'], n)]
                require(len(group) == 3 and {c['sample_seed'] for c in group} == SAMPLE_SEEDS,
                        'complete_three_samples_required')
                flat = [flatten(c) for c in group]
                exclude = {'dataset', 'method', 'fit_seed', 'sample_seed', 'row_multiplier', 'config_sha256',
                           'selection_binding', 'charged_artifact_bytes'}
                row = dict(dataset=dataset, fit_seed=fit['fit_seed'], row_multiplier=n,
                           metrics={k: complete_mean([x[k] for x in flat]) for k in flat[0] if k not in exclude},
                           native_value=fit['native_value'], receipt_refs=[c['receipt_ref'] for c in group])
                rows.append(row)
                per_fit.append(row)
            summaries.append(dict(dataset=dataset, row_multiplier=n,
                metrics={k: fit_statistic([r['metrics'][k] for r in rows]) for k in rows[0]['metrics']},
                native_value=fit_statistic([r['native_value'] for r in rows])))
    return per_fit, summaries


def render(panel):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['dataset', 'row_multiplier', 'metric', 'mean',
        'between_fit_sample_sd', 'measured_fits', 'unavailable_fits'], lineterminator='\n')
    writer.writeheader()
    for row in panel['summary']:
        for metric, result in sorted(dict(row['metrics'], native_author_mean_ml_r2=row['native_value']).items()):
            writer.writerow(dict(dataset=row['dataset'], row_multiplier=row['row_multiplier'], metric=metric, **result))
    checkpoints = io.StringIO(newline='')
    fields = ['dataset', 'fit_seed', 'sampler_sha256', 'sampler_bytes', 'charged_model_and_projection_bytes',
              'native_author_mean_ml_r2', 'catboost_tstr_loss_n', 'catboost_retention_n',
              'catboost_tstr_loss_4n', 'catboost_retention_4n', 'fit_receipt_path', 'fit_receipt_sha256',
              'native_receipt_sha256']
    writer = csv.DictWriter(checkpoints, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for fit in sorted(panel['fits'], key=lambda f: (f['dataset'], f['fit_seed'])):
        sampler = next(r for r in fit['artifact_inventory'] if r['path'] == 'sampler.pkl')
        row = dict(dataset=fit['dataset'], fit_seed=fit['fit_seed'], sampler_sha256=sampler['sha256'],
                   sampler_bytes=sampler['bytes'], charged_model_and_projection_bytes=fit['artifact_bytes'],
                   native_author_mean_ml_r2=fit['native_value'], fit_receipt_path=fit['fit_ref']['path'],
                   fit_receipt_sha256=fit['fit_ref']['sha256'], native_receipt_sha256=fit['native_ref']['sha256'])
        for n, label in ((1, 'n'), (4, '4n')):
            values = next(r['metrics'] for r in panel['per_fit']
                          if (r['dataset'], r['fit_seed'], r['row_multiplier']) == (fit['dataset'], fit['fit_seed'], n))
            for metric in ('catboost_tstr_loss', 'catboost_retention'):
                row[metric + '_' + label] = values[metric]
        writer.writerow(row)
    return {'panel.json': (json.dumps(panel, sort_keys=True, indent=2, allow_nan=False) + '\n').encode(),
            'figure-kpis.csv': stream.getvalue().encode(), 'checkpoint-kpis.csv': checkpoints.getvalue().encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.inputs)
    inputs = verified(dict(path=str(source), bytes=source.stat().st_size, sha256=args.inputs_sha256))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name, body in render(build(inputs)).items():
        (output / name).write_bytes(body)


if __name__ == '__main__':
    main()
