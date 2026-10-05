"""Publish a closed refinement progress batch without claiming matrix completion."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

BASE = Path('/mnt/fast-scratch/dope-benchmark')
ROOT = BASE / 'dope-target-refinement-progress-validation-v1'
RECEIPTS = '0b4de737276bcf81536261627bed565c8d0deab6bff21e2c1f02245f5d1f6cda'
ROUND = '7312ed0119e3b9b3bb91afb02491444b1c83c07fafdbeebe21ce87a4e743b1c3'
CONTINUATION = '57baef5e717c9229871b11056c094f12033bfad9d81590755fc1a1b6f7604768'
BASELINE = '5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a'
NAME = 'dope-target-refinement-progress'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def bound(path, expected):
    path = Path(path)
    if (type(expected) is not str or len(expected) != 64
            or not all(c in '0123456789abcdef' for c in expected)
            or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))
            or sha(path) != expected):
        raise ValueError('frozen publication evidence changed')
    return json.loads(path.read_bytes())


def verify_sources(lock):
    groups = defaultdict(dict)
    for name, h in lock['source_files'].items():
        p = Path(name)
        if sha(p) != h: raise ValueError('frozen source changed')
        groups[p.parent][str(p)] = h
    for root, files in groups.items():
        if (any(p.is_symlink() for p in (root, *root.parents))
                or any(p.is_symlink() for p in root.rglob('*'))
                or {str(p) for p in root.rglob('*') if p.is_file()} != set(files)):
            raise ValueError('frozen source inventory changed')


def build():
    anchor = bound(ROOT / 'receipt-lock-v1.json', RECEIPTS)
    # Verify frozen receipt digests, executable sources, model/worker/sample
    # bytes before reading metrics. No package imports or candidate execution.
    for p, h in anchor['refs'].items():
        p = Path(p)
        if not p.resolve(strict=True).is_relative_to(BASE): raise ValueError('outside evidence root')
        if any(q.is_symlink() for q in (p, *p.parents)) or sha(p) != h:
            raise ValueError('frozen receipt evidence changed')
        if p.name == 'round.lock.json': verify_sources(bound(p, h))
    lock = bound(ROOT / 'round.lock.json', ROUND)
    closed = bound(ROOT / 'completion.json', anchor['refs'][str(ROOT / 'completion.json')])
    actual = bound(ROOT / 'coordinator-exit.json', anchor['refs'][str(ROOT / 'coordinator-exit.json')])
    if (closed['counts'] != {'ok': 2} or closed['jobs'] != 2 or actual['exit_code'] != 0
            or actual['coordinator_log_sha256'] != sha(ROOT / 'coordinator.log')
            or len(lock['logical_cells']) != 18 or anchor['full_discovery_complete'] is not False):
        raise ValueError('complete progress batch required')
    gpu_path = BASE / 'dope-target-refinement-continuation-v3' / 'round.lock.json'
    gpu = bound(gpu_path, CONTINUATION)
    if (len(gpu['all_original_jobs']) != 24 or len(gpu['jobs']) != 21
            or len(gpu['original_completed_fit_refs']) != 3
            or len(gpu['infrastructure_retry_job_digests']) != 1
            or gpu['conservative_total_operation_seconds_bound'] > 14400):
        raise ValueError('complete retry accounting required')
    prior = [bound(r['path'], r['sha256']) for r in gpu['original_attempt_refs'].values()]
    attempts = []
    for r in gpu['original_attempt_refs'].values():
        receipt = bound(r['path'], r['sha256']); path = Path(r['path']).parent / 'fit.json'
        fit = bound(path, anchor['refs'][str(path)])
        attempts.append(dict(job_sha256=fit['job_sha256'], configuration=fit['job']['research_profile'],
            fit_seed=fit['job']['seed'], attempt=fit['attempt'], status=fit['status'],
            outcome_class='ok' if fit['status']=='ok' else 'infrastructure_interruption',
            host=fit['host'], elapsed_seconds=fit['elapsed_seconds'],
            operation_seconds=receipt['operations'][0]['elapsed_seconds'],
            charged_artifact_bytes=fit['artifact_bytes'], model_bytes=fit['model_bytes'],
            projection_bytes=fit['projection_bytes'], peak_gpu_used_mib=fit['peak_gpu_used_mib'],
            gpu_target_operator_verified=fit['gpu_target_operator_verified'],
            receipt_sha256=r['sha256'], runtime_sha256=fit['runtime_sha256'],
            energy_attributable_to_job=None, foreign_processes_signaled=False))
    if Counter(r['status'] for r in prior) != {'ok': 3, 'foreign_gpu_owner_appeared': 1}:
        raise ValueError('original infrastructure interruption must remain visible')
    pin = bound(ROOT / 'closed-receipt-pin-v1.json', anchor['closed_receipt_pin_sha256'])
    jobs = {identity(j): j for j in lock['jobs']}
    physical = {}
    for p, h in pin['receipts'].items():
        receipt = bound(p, h); job = receipt['job']; key = identity(job)
        if key not in jobs or receipt['status'] != 'ok': raise ValueError('batch identity changed')
        out = Path(p).parent
        if {q.name for q in out.iterdir() if q.is_file() and q.name != 'receipt.json'} != set(receipt['evidence_files']):
            raise ValueError('batch evidence inventory changed')
        batch = bound(out / 'batch.json', receipt['evidence_files']['batch.json'])
        if batch['job_sha256'] != key or batch['sample_replays_exact'] is not True or len(batch['samples']) != 6:
            raise ValueError('sample schedule or replay changed')
        for sample in batch['samples']:
            for name in ('metric', 'sample'):
                p = out / sample[name + '_file']
                if not p.resolve(strict=True).is_relative_to(out) or sha(p) != sample[name + '_sha256']:
                    raise ValueError('metric or sample evidence changed')
            metric = bound(out / sample['metric_file'], sample['metric_sha256'])
            if metric['implementation_sha256'] != lock['metric_source_sha256'] or metric['mfs_v2'] is not None:
                raise ValueError('metric contract changed')
            physical[key, sample['size_multiplier'], sample['sample_seed']] = metric
    cells = []
    for c in lock['logical_cells']:
        key = c['physical_job_sha256']; j = jobs[key]; metric = physical[key, c['size_multiplier'], c['sample_seed']]
        cells.append(dict(dataset=c['dataset'], method='DOPE', configuration=c['profile'],
            fit_seed=11, sample_seed=c['sample_seed'], size_multiplier=c['size_multiplier'],
            charged_artifact_bytes=j['artifact_bytes'], projection_bytes_included=True,
            fit_receipt_sha256=c['fit_receipt_sha256'], utility=metric['utility'],
            copy_counts=metric['copy_counts'], real_vs_real_control_counts=metric['real_vs_real_control_counts'],
            marginal_ks_mean=metric['marginal_ks_mean'], pair_correlation_fidelity=metric['pair_correlation_fidelity'],
            c2st_auc=metric['c2st_auc'], mfs_v2=None, ptf_v1=None, release_safe=None))
    baseline_path = Path(__file__).parent / 'results/density-matched-population-validation.json'
    baseline = bound(baseline_path, BASELINE)
    datasets = {c['dataset'] for c in cells}
    comparison = [c for c in baseline['cells'] if c['dataset'] in datasets and c['status'] == 'ok']
    density = bound(BASE / 'density-s3-population-shared-validation-v1/round.lock.json',
                    baseline['source_locks']['density_round'])
    density_workers = {j['dataset']: j['worker'] for j in density['jobs']}
    for j in jobs.values():
        if identity(j['worker']) != identity(density_workers[j['dataset']]):
            raise ValueError('baseline training/validation view differs')
    for p, h in anchor['refs'].items():
        if sha(p) != h: raise ValueError('evidence changed during publication')
    return dict(format='dope-target-refinement-progress', version=1,
        scope='One discovery lineage; three completed GPU profiles; eighteen completed validation cells. Full24-fit discovery is pending; this is a progress batch, not a final benchmark.',
        planned_discovery_fits=24, closed_successful_fits_at_snapshot=3,
        original_infrastructure_interrupted_attempts=1, infrastructure_retry_queued=1,
        unstarted_fits_carried_forward=20, allowed_gpu_hosts=['xbabe1', 'xbabe2', 'xbabe3'],
        original_job_identities_preserved=True, original_attempts_preserved=True,
        original_interruption_is_method_failure=False, full_discovery_complete=False,
        fit_compute_ceiling_seconds=14400, conservative_operation_seconds_bound=gpu['conservative_total_operation_seconds_bound'],
        prior_operation_seconds=gpu['prior_operation_seconds'], physical_validation_batches=2,
        validation_seconds=actual['elapsed_seconds'], fit_attempts=attempts, cells=cells, matched_reference_cells=comparison,
        baseline_native_selection_changed=False, baseline_selected_by_common_utility=False,
        source_locks=dict(progress_round=ROUND, progress_receipts=RECEIPTS,
                          continuation_round=CONTINUATION, matched_reference=BASELINE),
        global_family_selected=False, full_campaign_admitted=False, sdv_v3_launched=False,
        official_tests_opened=False, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)


def tables(report):
    rows = report['cells'] + report['matched_reference_cells']
    groups = defaultdict(list)
    for c in rows: groups[c['method'], c['configuration'], c['size_multiplier']].append(c)
    output = io.StringIO(); writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['method', 'configuration', 'size_multiplier', 'sample_cells', 'charged_artifact_bytes',
                     'catboost_median_retention', 'linear_median_retention', 'mlp_median_retention'])
    lines = ['# DOPE target refinement progress', '', report['scope'], '',
             'Comparators retain their frozen native likelihood selection. Retention is descriptive validation; no method selection or production score is inferred.', '',
             '| Method | Configuration | Size | Charged bytes | CatBoost median retention |',
             '| --- | --- | ---: | ---: | ---: |']
    for (method, config, size), cells in sorted(groups.items()):
        values = [statistics.median(c['utility'][a]['retention'] for c in cells) for a in ('catboost', 'linear', 'mlp')]
        charged = cells[0]['charged_artifact_bytes']
        writer.writerow([method, config, size, len(cells), charged, *values])
        lines.append(f'| {method} | {config} | {size}n | {charged} | {values[0]:.6f} |')
    lines += ['', 'The infrastructure-interrupted attempt remains recorded and charged; attempt2 is queued with the original job identity. Twenty unstarted jobs continue under the original limits. All official tests remain sealed; MFS-v2/PTF-v1/release/superiority stay null.', '']
    return output.getvalue(), '\n'.join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args(); report = build(); csv_text, markdown = tables(report)
    args.results_dir.mkdir(exist_ok=True)
    (args.results_dir / (NAME + '.json')).write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    (args.results_dir / (NAME + '.csv')).write_text(csv_text)
    (args.results_dir / (NAME + '.md')).write_text(markdown)


if __name__ == '__main__': main()
