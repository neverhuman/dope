"""Publish disjoint confirmation; preserve native selection and null claims."""
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path

from research.benchmark.publish_dope_refinement_progress import bound, identity, sha, verify_sources
from research.benchmark.publish_s3_forest import shared_inventory
from research.benchmark.publish_dope_refinement_discovery import summaries, verify_refs, paired as paired_profiles

BASE = Path('/mnt/fast-scratch/dope-benchmark')
ROOT = BASE / 'dope-target-refinement-confirmation-validation-v1'
FIT = BASE / 'dope-target-refinement-confirmation-v1'
ROUND = '82bff81c1930d4fd66579d54a0274b39f4f72cd5901f9a6261d3e66f29e12fc8'
RECEIPTS = '4df7037ce3ec91c8abe25359018178bd28e4e583bc6504631255af7634c6f174'
FIT_RECEIPTS = 'd7f5c4d850458b8db55b2c578fa25e89a168b146b69b4e7f0fe2f731c95bdfc5'
REFERENCE = '5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a'
NAME = 'dope-target-refinement-confirmation'
PROFILES = ('features12_steps8192', 'features12_width8_steps8192')
AUDITORS = ('catboost', 'linear', 'mlp')
SEEDS = (101, 211, 307)
SIZES = (1, 4)


def require(condition, reason):
    if not condition: raise ValueError(reason)


def validate_matrix(cells, datasets):
    expected = {(d, p, z, s) for d in datasets for p in PROFILES for z in SIZES for s in SEEDS}
    actual = [(r['dataset'], r['configuration'], r['size_multiplier'], r['sample_seed']) for r in cells]
    require(len(datasets) == 6 and len(actual) == 72 and len(set(actual)) == 72
            and set(actual) == expected, 'complete confirmation matrix required')
    require(Counter(r['status'] for r in cells) == {'ok': 72},
            'confirmation failure accounting changed')
    for row in cells:
        require(row['method'] == 'DOPE' and row['projection_bytes_included'] is True
                and all(row[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
                'research cell claim changed')
        require(row['charged_artifact_bytes'] <= 10240 and row['utility'] is not None,
                'successful fit exceeds artifact cap')


def paired(groups):
    return paired_profiles(groups, PROFILES)

def build():
    anchor = bound(ROOT / 'receipt-lock-v1.json', RECEIPTS)
    verify_refs(anchor['refs'])
    lock = bound(ROOT / 'round.lock.json', ROUND)
    require(anchor['round_sha256'] == ROUND and anchor['fit_receipt_lock_sha256'] == FIT_RECEIPTS,
            'parent receipt binding changed')
    shared_inventory(lock, {})  # Verify complete declared package/stdlib membership without imports.
    reconciliation = bound(ROOT / 'reconciliation-v1.json', anchor['reconciliation_sha256'])
    require(reconciliation['complete_matrix'] is True
            and reconciliation['physical_status_counts'] == {'ok': 12}
            and reconciliation['logical_status_counts'] == {'ok': 72},
            'validation closure changed')
    actual = bound(ROOT / 'coordinator-exit.json', anchor['refs'][str(ROOT / 'coordinator-exit.json')])
    closed = bound(ROOT / 'completion.json', anchor['refs'][str(ROOT / 'completion.json')])
    require(actual['exit_code'] == 0 and actual['round_sha256'] == ROUND
            and closed['jobs'] == 12 and closed['logical_cells'] == 72
            and actual['coordinator_log_sha256'] == sha(ROOT / 'coordinator.log'), 'actual closure required')
    fit_anchor = bound(FIT / 'receipt-lock-v1.json', FIT_RECEIPTS)
    fits = bound(FIT / 'reconciliation-v1.json', fit_anchor['reconciliation_sha256'])
    gpu = bound(FIT / 'round.lock.json', anchor['refs'][str(FIT / 'round.lock.json')])
    require(fits['closed_fit_cells'] == 12 and fits['status_counts'] == {'ok': 12}
            and fits['all_frozen_fit_cells_accounted'] is True
            and fits['new_operation_seconds'] <= 7920 and len(gpu['jobs']) == 12,
            'fit or cost accounting changed')
    require(not set(gpu['confirmation_dataset_ids']) & set(gpu['discovery_disjoint_dataset_ids'])
            and len(set(gpu['confirmation_dataset_ids'])) == 6, 'confirmation overlaps discovery')
    attempts = []
    for row in fits['fit_cells']:
        out = Path(row['model_path']).parent
        receipt = bound(out/'receipt.json', row['receipt_sha256'])
        fit = bound(out/'fit.json', anchor['refs'][str(out/'fit.json')])
        require(identity(receipt['job']) == out.parent.name == row['job_sha256']
                and fit['job_sha256'] == row['job_sha256'] and fit['status'] == receipt['status'] == 'ok'
                and fit['foreign_processes_signaled'] is False, 'fit request or outcome changed')
        operation = receipt['operations'][0]
        attempts.append(dict(dataset=fit['job']['dataset'], configuration=fit['job']['research_profile'],
            fit_job_sha256=fit['job_sha256'], fit_seed=fit['job']['seed'], attempt=fit['attempt'],
            status=fit['status'], outcome_class='ok', host=fit['host'],
            operation_seconds=operation['elapsed_seconds'], charged_artifact_bytes=fit['artifact_bytes'],
            model_bytes=fit['model_bytes'], projection_bytes=fit['projection_bytes'],
            peak_gpu_used_mib=fit['peak_gpu_used_mib'], peak_resident_bytes=fit['peak_resident_bytes'],
            energy_attributable_to_job=fit['energy_attributable_to_job'],
            gpu_target_operator_verified=fit['gpu_target_operator_verified'], runtime_sha256=fit['runtime_sha256'],
            receipt_sha256=row['receipt_sha256'], foreign_processes_signaled=False))
    require(len(attempts) == 12 and len({r['fit_job_sha256'] for r in attempts}) == 12
            and all(r['gpu_target_operator_verified'] is True for r in attempts), 'confirmation attempts incomplete')
    require(math.isclose(sum(a['operation_seconds'] for a in attempts), fits['new_operation_seconds'],
                         abs_tol=1e-9), 'fit operation costs disagree')
    cells = []
    for row in reconciliation['cells']:
        sample = row['sample_evidence']; metric = None
        if sample:
            require(anchor['refs'][sample['metric_path']] == sample['metric_sha256'], 'metric pin changed')
            metric = bound(sample['metric_path'], sample['metric_sha256'])
            require(metric['implementation_sha256'] == lock['metric_source_sha256']
                    and metric['mfs_v2'] is None, 'metric contract changed')
        cells.append(dict(dataset=row['dataset'], method='DOPE', configuration=row['profile'],
            fit_seed=row['fit_seed'], sample_seed=row['sample_seed'], size_multiplier=row['size_multiplier'],
            status=row['status'], unavailable_reason=row['unavailable_reason'],
            charged_artifact_bytes=row['charged_artifact_bytes'], projection_bytes_included=True,
            fit_job_sha256=row['fit_job_sha256'], fit_receipt_sha256=row['fit_receipt_sha256'],
            validation_receipt_sha256=row['validation_receipt_sha256'], sample_evidence=sample,
            utility=metric['utility'] if metric else None, null_loss=metric['null_loss'] if metric else None,
            copy_counts=metric['copy_counts'] if metric else None,
            real_vs_real_control_counts=metric['real_vs_real_control_counts'] if metric else None,
            marginal_ks_mean=metric['marginal_ks_mean'] if metric else None,
            pair_correlation_fidelity=metric['pair_correlation_fidelity'] if metric else None,
            c2st_auc=metric['c2st_auc'] if metric else None, mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None))
    datasets = sorted({r['dataset'] for r in cells}); validate_matrix(cells, datasets)
    reference = bound(Path(__file__).parent / 'results/density-matched-population-validation.json', REFERENCE)
    comparison = [r for r in reference['cells'] if r['dataset'] in datasets]
    require(len(comparison) == 360, 'complete reference matrix required')
    density_path = BASE / 'density-s3-population-shared-validation-v1/round.lock.json'
    density = bound(density_path, reference['source_locks']['density_round'])
    workers = {j['dataset']: j['worker'] for j in density['jobs']}
    for job in lock['jobs']:
        require(identity(job['worker']) == identity(workers[job['dataset']]), 'matched worker views differ')
    groups, panels = summaries(cells + comparison)
    verify_refs(anchor['refs'])
    discovery = bound(Path(__file__).parent/'results/dope-target-refinement-discovery.json',
                      'e0c9cd641ff4975126c0ecd3942315299ba14835b9fdad25df5f10d693b47bc4')
    require(not set(datasets) & set(discovery['datasets'])
            and set(datasets) == set(gpu['confirmation_dataset_ids']), 'confirmed cohort differs')
    require(gpu['closed_discovery_operation_seconds'] == discovery['fit_operation_seconds']
            and fits['new_operation_seconds'] + discovery['fit_operation_seconds'] <= 21600,
            'combined research compute ceiling changed')
    controls = Counter(r['sample_evidence']['sample_replay'] for r in cells)
    metric_controls = Counter(r['sample_evidence']['metric_replay'] for r in cells)
    require(controls == metric_controls == {'exact': 12, 'not_repeated': 60}, 'replay control coverage changed')
    require(all(r['sample_evidence']['sample_replay'] == ('exact' if r['size_multiplier'] == 1 and r['sample_seed'] == 101 else 'not_repeated')
                and r['sample_evidence']['metric_replay'] == ('exact' if r['size_multiplier'] == 4 and r['sample_seed'] == 101 else 'not_repeated')
                for r in cells), 'replay control schedule changed')
    return dict(format=NAME, version=1,
        scope='Six stratified confirmation lineages disjoint from discovery; two bounded GPU refinements; one fit seed and three sample seeds at n/4n. Training-derived validation only; no global family selection.',
        datasets=datasets, discovery_datasets=discovery['datasets'], confirmation_disjoint=True,
        planned_confirmation_fits=12, successful_fit_cells=12, artifact_cap_rejections=0,
        physical_validation_batches=12, logical_validation_cells=72, logical_status_counts={'ok': 72},
        fit_attempts=attempts, fit_operation_seconds=fits['new_operation_seconds'],
        discovery_fit_operation_seconds=discovery['fit_operation_seconds'],
        combined_discovery_confirmation_fit_operation_seconds=fits['new_operation_seconds']+discovery['fit_operation_seconds'],
        confirmation_compute_ceiling_seconds=7920, whole_refinement_compute_ceiling_seconds=21600,
        fit_scheduler_wall_seconds=bound(FIT/'coordinator-exit.json',
            anchor['refs'][str(FIT/'coordinator-exit.json')])['elapsed_seconds'],
        validation_operation_seconds=reconciliation['new_operation_seconds'], validation_wall_seconds=actual['elapsed_seconds'],
        sample_replay_control_counts=dict(controls), metric_replay_control_counts=dict(metric_controls),
        sample_replay_control_schedule=dict(size_multiplier=1, sample_seed=101),
        metric_replay_control_schedule=dict(size_multiplier=4, sample_seed=101),
        replay_control_overlap_cells=0,
        earlier_architecture_research_cost_separate=True, cells=cells, matched_reference_cells=comparison,
        lineage_groups=groups, summary=panels, paired_descriptive=paired(groups),
        source_locks=dict(validation_round=ROUND, validation_receipts=RECEIPTS, fit_receipts=FIT_RECEIPTS,
            matched_reference=REFERENCE, discovery_report='e0c9cd641ff4975126c0ecd3942315299ba14835b9fdad25df5f10d693b47bc4',
            metric_source=lock['metric_source_sha256']),
        allowed_gpu_hosts=['xbabe1', 'xbabe2', 'xbabe3'], baseline_native_selection_changed=False,
        baseline_selected_by_common_utility=False, native_kpis_never_ranked_across_methods=True,
        confirmation_complete=True, global_family_selected=False, full_campaign_admitted=False,
        sdv_v3_launched=False, official_tests_opened=False, mfs_v2=None, ptf_v1=None,
        release_safe=None, superiority=None)


def tables(report):
    out = io.StringIO(); writer = csv.writer(out, lineterminator='\n')
    writer.writerow(['method', 'configuration', 'size_multiplier', 'planned_lineages', 'measured_lineages',
                     'charged_bytes_min', 'charged_bytes_median', 'charged_bytes_max',
                     'catboost_informative_lineages', 'catboost_median_retention',
                     'linear_informative_lineages', 'linear_median_retention',
                     'mlp_informative_lineages', 'mlp_median_retention'])
    lines = ['# Completed disjoint DOPE GPU refinement confirmation', '', report['scope'], '',
        'Baselines retain their own frozen held-out likelihood selections. Sample seeds are reduced within each lineage before taking dataset medians. Negative retention is retained. This small confirmation subset is descriptive; no superiority or production certification is inferred.', '',
        '| Method | Configuration | Size | Measured / planned lineages | Charged bytes, min–max | CatBoost median retention (informative N) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in report['summary']:
        values = [x for a in AUDITORS for x in (row['utility'][a]['informative_complete_lineages'], row['utility'][a]['median_retention'])]
        writer.writerow([row[k] for k in ('method', 'configuration', 'size_multiplier', 'planned_lineages',
            'measured_lineages', 'charged_bytes_min', 'charged_bytes_median', 'charged_bytes_max')] + values)
        value = row['utility']['catboost']; text = 'null' if value['median_retention'] is None else f"{value['median_retention']:.6f}"
        lines.append(f"| {row['method']} | {row['configuration']} | {row['size_multiplier']}n | {row['measured_lineages']}/{row['planned_lineages']} | {row['charged_bytes_min']}–{row['charged_bytes_max']} | {text} ({value['informative_complete_lineages']}) |")
    lines += ['', '12/12 GPU fit cells and 72/72 logical validation cells succeeded. Twelve physical batches closed with actual coordinator exit zero. All charges include model and projection.', '',
        f"Confirmation fit operations: {report['fit_operation_seconds']:.6f}s; validation operations: {report['validation_operation_seconds']:.6f}s. Combined discovery/confirmation fit operations: {report['combined_discovery_confirmation_fit_operation_seconds']:.6f}s within the original 21,600-second ceiling. Scheduler waiting wall and earlier architecture research costs are separate.", '',
        'Sampler replay is exact for n/seed101 (12 cells) and metric replay for 4n/seed101 (12 different cells). Each kind has 60 cells not repeated; 48 cells have neither repetition. The two profiles were frozen after discovery and before confirmation metrics. Prior DOPE and native/default density references are unchanged.', '',
        'Confirmation is disjoint from discovery. This is not the public-core headline, final five-fit matrix, global candidate-family selection, or production certification. Official tests remain sealed; MFS-v2/PTF-v1/release/superiority stay null.', '']
    return out.getvalue(), '\n'.join(lines)


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
