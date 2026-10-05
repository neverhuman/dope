"""Publish the complete bounded discovery round; retain failures and null gates."""
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import statistics

from research.benchmark.publish_dope_refinement_progress import bound, identity, sha, verify_sources
from research.benchmark.publish_s3_forest import shared_inventory

BASE = Path('/mnt/fast-scratch/dope-benchmark')
ROOT = BASE / 'dope-target-refinement-validation-v2'
FIT = BASE / 'dope-target-refinement-continuation-v3'
ROUND = '666849b480c39325266590778c15c9e9865134d45ae4371c56b2186a5c33d181'
RECEIPTS = 'cd1909d34b81225a214361972c1b5ac4e79e7e6da24525b9945e7b37b4debd3b'
FIT_RECEIPTS = '9c2f0272f2691442c25f86d63ef437d1848e4cd78085d5f048f98cf61d5ab36e'
REFERENCE = '5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a'
NAME = 'dope-target-refinement-discovery'
PROFILES = ('features12_steps8192', 'features16_steps8192',
            'features12_width8_steps8192', 'features24_width8_steps8192')
AUDITORS = ('catboost', 'linear', 'mlp')
SEEDS = (101, 211, 307)
SIZES = (1, 4)


def require(condition, reason):
    if not condition: raise ValueError(reason)


def verify_refs(refs):
    """Rehash pinned artifacts, executable sources and flat worker inputs."""
    for name, expected in refs.items():
        path = Path(name)
        require(path.resolve(strict=True).is_relative_to(BASE), 'outside evidence root')
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'linked evidence')
        require(sha(path) == expected, 'frozen evidence changed')
        if path.name == 'round.lock.json': verify_sources(bound(path, expected))


def retention(row, auditor):
    value = (row.get('utility') or {}).get(auditor, {})
    if row['status'] != 'ok' or value.get('informative') is not True: return None
    result = value.get('retention')
    require(type(result) in (int, float) and math.isfinite(result), 'nonfinite retention')
    return result


def summaries(cells):
    """Reduce sample seeds within lineage before descriptive dataset summaries."""
    groups = defaultdict(list)
    seen = set()
    for row in cells:
        key = (row['dataset'], row['method'], row['configuration'], row['size_multiplier'])
        sample = (*key, row['sample_seed'])
        require(sample not in seen and type(row['sample_seed']) is int
                and row['sample_seed'] in SEEDS and type(row['size_multiplier']) is int
                and row['size_multiplier'] in SIZES and type(row['fit_seed']) is int
                and row['fit_seed'] == 11, 'duplicate or changed sample identity')
        seen.add(sample); groups[key].append(row)
    panels = defaultdict(list)
    lineage_groups = []
    for key, rows in sorted(groups.items()):
        require(len(rows) == 3 and {r['sample_seed'] for r in rows} == set(SEEDS),
                'incomplete sample schedule')
        require(len({r['charged_artifact_bytes'] for r in rows}) == 1, 'artifact charges disagree')
        values = {}
        for auditor in AUDITORS:
            measured = [retention(r, auditor) for r in sorted(rows, key=lambda r: r['sample_seed'])]
            values[auditor] = dict(complete_informative=all(v is not None for v in measured),
                median_retention=statistics.median(measured) if all(v is not None for v in measured) else None)
        group = dict(dataset=key[0], method=key[1], configuration=key[2], size_multiplier=key[3],
                     charged_artifact_bytes=rows[0]['charged_artifact_bytes'],
                     statuses=dict(Counter(r['status'] for r in rows)), utility=values)
        lineage_groups.append(group); panels[key[1:]].append(group)
    output = []
    for key, groups in sorted(panels.items()):
        bytes_ = [r['charged_artifact_bytes'] for r in groups if r['charged_artifact_bytes'] is not None]
        utility = {}
        for auditor in AUDITORS:
            values = [r['utility'][auditor]['median_retention'] for r in groups
                      if r['utility'][auditor]['complete_informative']]
            utility[auditor] = dict(informative_complete_lineages=len(values),
                median_retention=statistics.median(values) if values else None)
        output.append(dict(method=key[0], configuration=key[1], size_multiplier=key[2],
            planned_lineages=len(groups), measured_lineages=sum(r['statuses'] == {'ok': 3} for r in groups),
            charged_bytes_min=min(bytes_) if bytes_ else None,
            charged_bytes_median=statistics.median(bytes_) if bytes_ else None,
            charged_bytes_max=max(bytes_) if bytes_ else None, utility=utility))
    return lineage_groups, output


def validate_matrix(cells, datasets):
    expected = {(d, p, z, s) for d in datasets for p in PROFILES for z in SIZES for s in SEEDS}
    actual = [(r['dataset'], r['configuration'], r['size_multiplier'], r['sample_seed']) for r in cells]
    require(len(datasets) == 6 and len(actual) == 144 and len(set(actual)) == 144
            and set(actual) == expected, 'complete discovery matrix required')
    require(Counter(r['status'] for r in cells) == {'ok': 138, 'fit_unavailable': 6},
            'discovery failure accounting changed')
    for row in cells:
        require(row['method'] == 'DOPE' and row['projection_bytes_included'] is True
                and all(row[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
                'research cell claim changed')
        if row['status'] == 'ok':
            require(row['charged_artifact_bytes'] <= 10240 and row['utility'] is not None,
                    'successful fit exceeds artifact cap')
        else:
            require(row['unavailable_reason'] == 'charged_artifact_cap'
                    and row['charged_artifact_bytes'] > 10240 and row['utility'] is None,
                    'artifact rejection hidden')


def paired(groups, profiles=PROFILES):
    """Descriptive paired medians; no tests, ranks, selection or win claims."""
    lookup = {(g['dataset'], g['method'], g['configuration'], g['size_multiplier']): g for g in groups}
    datasets = sorted({g['dataset'] for g in groups})
    references = [('DOPE', 'features12_steps2048'), ('GaussianCopula', 'native_selected'),
                  ('Chow-Liu', 'native_selected'), ('independent_marginals', 'native_selected')]
    rows = []
    for profile in profiles:
        for method, configuration in references:
            for size in SIZES:
                for auditor in AUDITORS:
                    values = []
                    for dataset in datasets:
                        left = lookup[dataset, 'DOPE', profile, size]['utility'][auditor]
                        right = lookup[dataset, method, configuration, size]['utility'][auditor]
                        if left['complete_informative'] and right['complete_informative']:
                            values.append((left['median_retention'], right['median_retention']))
                    rows.append(dict(configuration=profile, reference_method=method,
                        reference_configuration=configuration, size_multiplier=size, auditor=auditor,
                        planned_lineages=len(datasets), paired_complete_informative_lineages=len(values),
                        dope_median_retention=statistics.median(v[0] for v in values) if values else None,
                        reference_median_retention=statistics.median(v[1] for v in values) if values else None,
                        median_paired_difference=statistics.median(v[0]-v[1] for v in values) if values else None,
                        superiority=None))
    return rows


def build():
    anchor = bound(ROOT / 'receipt-lock-v1.json', RECEIPTS)
    verify_refs(anchor['refs'])
    lock = bound(ROOT / 'round.lock.json', ROUND)
    require(anchor['round_sha256'] == ROUND and anchor['fit_receipt_lock_sha256'] == FIT_RECEIPTS,
            'parent receipt binding changed')
    shared_inventory(lock, {})  # Verify complete declared package/stdlib membership without imports.
    reconciliation = bound(ROOT / 'reconciliation-v1.json', anchor['reconciliation_sha256'])
    require(reconciliation['complete_matrix'] is True
            and reconciliation['physical_status_counts'] == {'ok': 15}
            and reconciliation['logical_status_counts'] == {'ok': 138, 'fit_unavailable': 6},
            'validation closure changed')
    actual = bound(ROOT / 'coordinator-exit.json', anchor['refs'][str(ROOT / 'coordinator-exit.json')])
    closed = bound(ROOT / 'completion.json', anchor['refs'][str(ROOT / 'completion.json')])
    require(actual['exit_code'] == 0 and actual['round_sha256'] == ROUND
            and closed['jobs'] == 15 and closed['logical_cells'] == 144
            and actual['coordinator_log_sha256'] == sha(ROOT / 'coordinator.log'), 'actual closure required')
    fit_anchor = bound(FIT / 'receipt-lock-v1.json', FIT_RECEIPTS)
    fits = bound(FIT / 'reconciliation-v1.json', fit_anchor['reconciliation_sha256'])
    gpu = bound(FIT / 'round.lock.json', anchor['refs'][str(FIT / 'round.lock.json')])
    require(fits['closed_fit_cells'] == 24 and fits['status_counts'] == {'ok': 23, 'charged_artifact_cap': 1}
            and fits['new_operation_seconds'] <= 14400 and len(gpu['all_original_jobs']) == 24,
            'fit or cost accounting changed')
    # Include each original receipt and each selected continuation exactly once.
    attempt_refs = dict(gpu['original_attempt_refs'])
    selected = bound(FIT / 'closed-receipt-pin-v1.json', anchor['refs'][str(FIT / 'closed-receipt-pin-v1.json')])
    for key, ref in selected['receipts'].items(): attempt_refs[ref['path']] = ref
    by_path = {ref['path']: ref['sha256'] for ref in attempt_refs.values()}
    attempts = []
    for path, expected in sorted(by_path.items()):
        receipt = bound(path, expected); out = Path(path).parent
        require(identity(receipt['job']) == out.parent.name, 'fit request identity changed')
        fit = bound(out / 'fit.json', anchor['refs'][str(out / 'fit.json')])
        operation = receipt['operations'][0]
        require(fit['status'] == receipt['status'] and fit['foreign_processes_signaled'] is False,
                'fit receipt outcome changed')
        attempts.append(dict(dataset=fit['job']['dataset'], configuration=fit['job']['research_profile'],
            fit_job_sha256=fit['job_sha256'], fit_seed=fit['job']['seed'], attempt=fit['attempt'],
            status=fit['status'], outcome_class='infrastructure_interruption' if fit['status'] == 'foreign_gpu_owner_appeared'
                else 'artifact_cap_rejection' if fit['status'] == 'charged_artifact_cap' else 'ok',
            host=fit['host'], operation_seconds=operation['elapsed_seconds'],
            charged_artifact_bytes=fit['artifact_bytes'], model_bytes=fit['model_bytes'],
            projection_bytes=fit['projection_bytes'], peak_gpu_used_mib=fit['peak_gpu_used_mib'],
            peak_resident_bytes=fit['peak_resident_bytes'], energy_attributable_to_job=fit['energy_attributable_to_job'],
            gpu_target_operator_verified=fit['gpu_target_operator_verified'],
            runtime_sha256=fit['runtime_sha256'], receipt_sha256=expected, foreign_processes_signaled=False))
    require(len(attempts) == 25 and Counter(a['status'] for a in attempts) ==
            {'ok': 23, 'charged_artifact_cap': 1, 'foreign_gpu_owner_appeared': 1}, 'retry accounting incomplete')
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
    return dict(format=NAME, version=1,
        scope='Six stratified discovery lineages; four bounded GPU refinements; one fit seed, three sample seeds at n/4n. Discovery validation only; confirmation and global family selection remain pending.',
        datasets=datasets, planned_discovery_fits=24, successful_fit_cells=23, artifact_cap_rejections=1,
        physical_validation_batches=15, logical_validation_cells=144,
        logical_status_counts=dict(Counter(r['status'] for r in cells)), full_discovery_complete=True,
        fit_attempts=attempts, fit_operation_seconds=fits['new_operation_seconds'],
        prior_operation_seconds=fits['prior_operation_seconds'],
        continuation_operation_seconds=fits['continuation_operation_seconds'],
        fit_compute_ceiling_seconds=14400, fit_scheduler_wall_seconds=bound(FIT / 'coordinator-exit.json',
            anchor['refs'][str(FIT / 'coordinator-exit.json')])['elapsed_seconds'],
        validation_operation_seconds=reconciliation['new_operation_seconds'],
        validation_wall_seconds=actual['elapsed_seconds'], earlier_architecture_research_cost_separate=True,
        cells=cells, matched_reference_cells=comparison, lineage_groups=groups, summary=panels,
        paired_descriptive=paired(groups),
        source_locks=dict(validation_round=ROUND, validation_receipts=RECEIPTS,
            fit_receipts=FIT_RECEIPTS, matched_reference=REFERENCE, metric_source=lock['metric_source_sha256']),
        allowed_gpu_hosts=['xbabe1', 'xbabe2', 'xbabe3'], original_attempts_preserved=True,
        original_interruption_is_method_failure=False, baseline_native_selection_changed=False,
        baseline_selected_by_common_utility=False, native_kpis_never_ranked_across_methods=True,
        confirmation_complete=False, global_family_selected=False, full_campaign_admitted=False,
        sdv_v3_launched=False, official_tests_opened=False, mfs_v2=None, ptf_v1=None,
        release_safe=None, superiority=None)


def tables(report):
    out = io.StringIO(); writer = csv.writer(out, lineterminator='\n')
    writer.writerow(['method', 'configuration', 'size_multiplier', 'planned_lineages', 'measured_lineages',
                     'charged_bytes_min', 'charged_bytes_median', 'charged_bytes_max',
                     'catboost_informative_lineages', 'catboost_median_retention',
                     'linear_informative_lineages', 'linear_median_retention',
                     'mlp_informative_lineages', 'mlp_median_retention'])
    lines = ['# Completed DOPE GPU refinement discovery', '', report['scope'], '',
        'Baselines retain their own frozen held-out likelihood selections. Sample seeds are reduced within each lineage before taking dataset medians. Negative retention is retained. This selected discovery subset is descriptive; no superiority or production certification is inferred.', '',
        '| Method | Configuration | Size | Measured / planned lineages | Charged bytes, min–max | CatBoost median retention (informative N) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in report['summary']:
        values = [x for a in AUDITORS for x in (row['utility'][a]['informative_complete_lineages'], row['utility'][a]['median_retention'])]
        writer.writerow([row[k] for k in ('method', 'configuration', 'size_multiplier', 'planned_lineages',
            'measured_lineages', 'charged_bytes_min', 'charged_bytes_median', 'charged_bytes_max')] + values)
        value = row['utility']['catboost']; text = 'null' if value['median_retention'] is None else f"{value['median_retention']:.6f}"
        lines.append(f"| {row['method']} | {row['configuration']} | {row['size_multiplier']}n | {row['measured_lineages']}/{row['planned_lineages']} | {row['charged_bytes_min']}–{row['charged_bytes_max']} | {text} ({value['informative_complete_lineages']}) |")
    lines += ['', '24/24 fit cells accounted: 23 successful, one charged artifact rejection (10,381 bytes > 10,240). All 25 attempt receipts are retained, including the earlier infrastructure interruption and its successful retry. Validation: 15/15 physical batches, 138 measured / 6 fit-unavailable logical cells.', '',
        f"Fit operations: {report['fit_operation_seconds']:.6f}s; validation operations: {report['validation_operation_seconds']:.6f}s. Scheduler wall includes admission waits and is reported separately. Earlier architecture research costs remain separate.", '',
        'The six disjoint confirmation lineages have not been evaluated in this panel. Official tests remain sealed; MFS-v2/PTF-v1/release/superiority stay null.', '']
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
