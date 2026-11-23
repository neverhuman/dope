"""Compare a frozen shared package against a pinned native generator on owned development data."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .comparison import schedule, ridge_loss, retention, summarize
from .contracts import LEVELS, level_widths, ResearchError, digest, levels, require
from .corpus import file_sha
from .checkpoints import private_json
from .train import append


def native_fit(binary, source, artifact, cap, task, seed, seconds):
    result = subprocess.run([str(binary), '--threads', '4', 'fit', str(source), '-t', 'c8',
        '-o', str(artifact), '--budget', str(cap), '--seed', str(seed), '--task', task,
        '--device', 'cpu', '--quiet'], capture_output=True, timeout=seconds)
    require(result.returncode == 0, 'comparison_native_fit_failed')
    require(artifact.is_file() and not artifact.is_symlink() and artifact.stat().st_size <= cap,
            'comparison_native_budget_failed')


def native_sample(binary, artifact, destination, rows, seed, seconds):
    result = subprocess.run([str(binary), '--threads', '4', 'sample', str(artifact), '-n', str(rows),
        '--seed', str(seed), '-o', str(destination), '--device', 'cpu'], capture_output=True, timeout=seconds)
    require(result.returncode == 0, 'comparison_native_sample_failed')
    with destination.open() as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == [f'c{i}' for i in range(9)], 'comparison_native_header_drift')
        values = np.asarray([[float(x) for x in row] for row in reader], dtype=np.float64)
    require(values.shape == (rows, 9) and np.isfinite(values).all(), 'comparison_native_shape_drift')
    return values, np.ones(values.shape, dtype=bool)


def native_audit(binary, artifact, context, query, synthetic, report, seed, seconds):
    names = ['linear-seeded-v1', 'gbdt', 'mlp-seeded-v1']
    result = subprocess.run([str(binary), '--threads', '4', 'eval', '--model', str(artifact),
        '--train', str(context), '--test', str(query), '--synth', str(synthetic),
        '--auditors', ','.join(names), '--seed', str(seed), '--json', str(report)],
        capture_output=True, timeout=seconds)
    require(result.returncode == 0, 'comparison_native_audit_failed')
    value = json.loads(report.read_bytes())
    require([a['auditor'] for a in value['auditors']] == names
            and all(np.isfinite(a[k]) and a[k] >= 0 for a in value['auditors']
                    for k in ('null_loss', 'real_loss', 'synth_loss')), 'comparison_native_audit_invalid')
    return value['auditors']


def run(model_path, baseline, expected_baseline_sha256, out, *, selected=('low', 'medium', 'high'),
        table_seeds=(20261101, 20261102, 20261103), sample_seeds=(101, 211, 307),
        fit_seed=20261023, categorical_schedule='reveal-v1', device='cpu', wall_seconds=600, resource_check=None,
        auditor_engine=None, expected_auditor_sha256=None, landmark_input_cache=None):
    import torch
    from .packages import load_package, encode_table, export_encoding
    from .decoder import sample
    from .data import episode
    from .priors import table
    from .resources import Guard, snapshot, admissible
    cases = schedule(selected, table_seeds, sample_seeds)
    baseline, out = Path(baseline).resolve(), Path(out)
    require(type(wall_seconds) is int and 1 <= wall_seconds <= 7200
            and device in ('cpu', 'cuda') and type(fit_seed) is int and 0 <= fit_seed < 2**32,
            'comparison_allocation_invalid')
    require(categorical_schedule in ('reveal-v1', 'redraw-v1'), 'unknown_categorical_schedule')
    require(baseline.is_file() and baseline.stat().st_size <= 200000000
            and file_sha(baseline) == expected_baseline_sha256, 'comparison_binary_digest_mismatch')
    require((auditor_engine is None) == (expected_auditor_sha256 is None), 'comparison_auditor_pin_required')
    if auditor_engine is not None:
        auditor_engine = Path(auditor_engine).resolve()
        require(auditor_engine.is_file() and file_sha(auditor_engine) == expected_auditor_sha256,
                'comparison_auditor_digest_mismatch')
    require(out.parent.is_dir() and not out.exists(), 'comparison_output_must_be_new')
    initial = snapshot(out.parent, cuda=device == 'cuda')
    require(admissible(initial, cuda=device == 'cuda')
            and initial['load'][0] < initial['cpu_available']*.85, 'comparison_resource_refused')
    os.umask(0o077)
    out.mkdir(mode=0o700)
    private_json(out/'admission.json', initial)
    if resource_check is None:
        resource_check = Guard(out.parent, out/'resources.jsonl', cuda=device == 'cuda')
    require(resource_check(), 'comparison_resource_refused')
    if hasattr(os, 'nice'):
        os.nice(10)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    if device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(min(1., 16*2**30/torch.cuda.get_device_properties(0).total_memory))
    model, manifest = load_package(model_path, device)
    from .landmark_features import InputCache
    input_cache=InputCache(landmark_input_cache)if landmark_input_cache is not None else None
    require(model.config.landmark_input()==(input_cache is not None), 'comparison_landmark_cache_required_for_graph')
    require(input_cache is None or input_cache.feature_recipe==model.config.landmark_recipe(),
            'comparison_landmark_cache_recipe_mismatch')
    require(all(level in manifest['trained_levels'] for level in selected), 'comparison_level_not_trained')
    fixed_bytes = sum(v['bytes'] for v in manifest['weights'].values()) + (Path(model_path)/'model.json').stat().st_size
    protocol = dict(format='dope-owned-comparison-protocol-v2', cases=cases, fit_seed=fit_seed,
        model_coordinate_space=manifest['coordinate_space_id'], training_source_sha256=manifest['source_sha256'],
        comparison_source_sha256=file_sha(__file__), scoring_source_sha256=file_sha(Path(__file__).with_name('comparison.py')),
        baseline_sha256=expected_baseline_sha256, diffusion_steps=32, categorical_schedule=categorical_schedule, context_role='fit', query_role='tune',
        sample_rows='context_rows', fixed_shared_dependency_bytes=fixed_bytes,
        baseline_byte_cap='complete selected shared-code payload including charged schema',
        auditor='numpy64 ridge relative1e-3; MSE regression/Brier binary; fit-only normalized predictors plus missing flags',
        device=device, wall_seconds=wall_seconds, matched_training_compute=False,
        official_test_access=False, final_family_access=False, qualification='owned_development_exploration_only')
    if auditor_engine is not None:
        protocol.update(format='dope-owned-comparison-protocol-v3', native_auditor=dict(
            engine_sha256=expected_auditor_sha256, names=['linear-seeded-v1', 'gbdt', 'mlp-seeded-v1'],
            representation='common baseline artifact schema; fit context only; supplied synthetic rows',
            loss='existing native regression MSE/classification log loss; auxiliary to unchanged primary ridge',
            driver_qualified=False))
    if input_cache is not None:
        protocol.update(format='dope-owned-comparison-protocol-v4',landmark_inputs=dict(
            cache_sha256=input_cache.identity,feature_recipe=input_cache.feature_recipe,
            scope='innerCV inside source fit rows only; no tune/calibration/confirmation labels',
            serving='private encoder input; code/decoder-only recipient',
            precomputation_cost='excluded from sample timing; report original cache costs separately'))
    private_json(out/'protocol.json', protocol)
    records = []
    for family in (7, 8):
        for seed in table_seeds:
            source = table(family, seed)
            input_features=None;feature_failure=False;feature_provenance={}
            if input_cache is not None:
                source['source_provenance']=dict(kind='owned_prior',physical_table_id=digest(dict(family=family,seed=seed)))
                try:
                    source=input_cache.attach(source);input_features=source['landmark_input']
                    feature_provenance={k:v for k,v in source['source_provenance'].items()
                                        if k in('landmark_input_manifest_sha256','landmark_feature_sha256')}
                except (ResearchError,OSError,KeyError,TypeError,ValueError):feature_failure=True
            context, query, identity = episode(**{k:source[k] for k in
                ('rows', 'kinds', 'slots', 'row_ids', 'task', 'roles')},
                config=model.config, cursor=0, seed=fit_seed)
            require(all(c.kind == 'numeric' for c in context.columns[:-1]), 'comparison_predictor_kind_unsupported')
            classification = source['task'] == 'classification'
            real = ridge_loss(context.values, context.observed, query, classification)
            offset = 2 if classification else 0
            null = float(np.mean((query.values[:, -1]-offset-(context.values[:, -1]-offset).mean())**2))
            directory = out/f'family-{family}-seed-{seed}'
            directory.mkdir()
            csv_path = directory/'context.csv'
            with csv_path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow([f'c{i}' for i in range(9)])
                writer.writerows(context.values.tolist())
            query_path = directory/'query.csv'
            if auditor_engine is not None:
                with query_path.open('w', newline='') as stream:
                    writer = csv.writer(stream); writer.writerow([f'c{i}' for i in range(9)])
                    writer.writerows(query.values.tolist())
            encoded = None
            for level in selected:
                artifact = directory/(level+'.dope')
                native_error = False
                code = None
                for sample_seed in sample_seeds:
                    for method in ('shared', 'baseline'):
                        record = dict(family=family, table_seed=seed, level=level,
                            sample_seed=sample_seed, method=method, episode_sha256=identity,
                            context_rows=len(context.values), query_rows=len(query.values),
                            real_loss=real, null_loss=null, fixed_model_bytes=fixed_bytes if method == 'shared' else 0)
                        if input_cache is not None:record.update(landmark_feature_recipe=input_cache.feature_recipe,**feature_provenance)
                        try:
                            require(not feature_failure,'comparison_landmark_inputs_unavailable')
                            require(time.monotonic()-started < wall_seconds and resource_check(), 'comparison_budget_or_resource_pause')
                            if encoded is None:
                                input_args=dict(landmark_features=input_features)if input_cache is not None else{}
                                encoded = encode_table(model, manifest, context, through=max(selected, key=level_widths(model.config.layout).get),**input_args)
                            if code is None:
                                code = export_encoding(encoded, level)
                                (directory/(level+'.safetensors')).write_bytes(code.bytes)
                            record['code_bytes'] = len(code.bytes)
                            if auditor_engine is not None and not artifact.exists():
                                require(not native_error, 'comparison_native_fit_failed')
                                try:
                                    native_fit(baseline, csv_path, artifact, len(code.bytes), source['task'], fit_seed,
                                               max(.01, min(60, wall_seconds-(time.monotonic()-started))))
                                except (ResearchError, subprocess.SubprocessError):
                                    native_error = True
                                    raise
                            before = time.monotonic()
                            if method == 'shared':
                                values, observed = sample(model.decoder, encoded.code[:level_widths(model.config.layout)[level]].to(device),
                                    encoded.schema, len(context.values), sample_seed, steps=32, categorical_schedule=categorical_schedule)
                                if device == 'cuda':
                                    torch.cuda.synchronize()
                                values, observed = values.cpu().numpy(), observed.cpu().numpy()
                                record['artifact_bytes'] = len(code.bytes)
                            else:
                                require(not native_error, 'comparison_native_fit_failed')
                                if not artifact.exists():
                                    try:
                                        native_fit(baseline, csv_path, artifact, len(code.bytes), source['task'], fit_seed,
                                                   max(.01, min(60, wall_seconds-(time.monotonic()-started))))
                                    except (ResearchError, subprocess.SubprocessError):
                                        native_error = True
                                        raise
                                values, observed = native_sample(baseline, artifact,
                                    directory/f'{level}-seed-{sample_seed}.csv', len(context.values), sample_seed,
                                    max(.01, min(20, wall_seconds-(time.monotonic()-started))))
                                record['artifact_bytes'] = artifact.stat().st_size
                            if auditor_engine is not None:
                                synthetic = directory/f'{level}-{sample_seed}-{method}-audit.csv'
                                with synthetic.open('w', newline='') as stream:
                                    writer = csv.writer(stream); writer.writerow([f'c{i}' for i in range(9)])
                                    writer.writerows([[float(v) if m else '' for v,m in zip(row,mask)]
                                                      for row,mask in zip(values,observed)])
                                report = directory/f'{level}-{sample_seed}-{method}-audit.json'
                                record['native_auditors'] = native_audit(auditor_engine, artifact, csv_path,
                                    query_path, synthetic, report, fit_seed,
                                    max(.01, min(60, wall_seconds-(time.monotonic()-started))))
                                record.update(native_report_sha256=file_sha(report),
                                              native_synthetic_sha256=file_sha(synthetic))
                            metric = ridge_loss(values, observed, query, classification)
                            record.update(state='complete', synthetic_loss=metric,
                                retention=retention(null, real, metric), sample_and_audit_seconds=time.monotonic()-before)
                        except (ResearchError, OSError, RuntimeError, ValueError, subprocess.SubprocessError, np.linalg.LinAlgError) as error:
                            record.update(state='failed', failure_type=type(error).__name__)
                        append(out/'records.jsonl', record)
                        records.append(record)
    summary = summarize(cases, records)
    summary.update(protocol_sha256=file_sha(out/'protocol.json'), records_sha256=file_sha(out/'records.jsonl'),
                   wall_seconds=time.monotonic()-started)
    private_json(out/'summary.json', summary)
    private_json(out/'terminal.json', dict(state='complete' if summary['all_cells_complete'] else 'complete_with_failures',
                                         scheduled=len(cases), failures=summary['failures']))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--expected-baseline-sha256', required=True)
    parser.add_argument('--auditor-engine', type=Path)
    parser.add_argument('--expected-auditor-sha256')
    parser.add_argument('--landmark-input-cache',type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--levels', default='all')
    parser.add_argument('--categorical-schedule', choices=('reveal-v1', 'redraw-v1'), default='reveal-v1')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--wall-seconds', type=int, default=600)
    args = parser.parse_args(argv)
    result = run(args.model, args.baseline, args.expected_baseline_sha256, args.out,
                 selected=levels(args.levels), categorical_schedule=args.categorical_schedule, device=args.device,
                 wall_seconds=args.wall_seconds, auditor_engine=args.auditor_engine,
                 expected_auditor_sha256=args.expected_auditor_sha256,
                 landmark_input_cache=None if args.landmark_input_cache is None else json.loads(args.landmark_input_cache.read_bytes()))
    print(json.dumps({k:result[k] for k in ('scheduled', 'complete', 'failures', 'superiority_qualified')}))
    return 0 if result['all_cells_complete'] else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ResearchError, OSError, ValueError, RuntimeError, KeyError, TypeError):
        print(json.dumps(dict(state='failed', reason=ResearchError.reason, repair_hint=ResearchError.repair_hint,
                              docs=ResearchError.docs_url)))
        raise SystemExit(1)
