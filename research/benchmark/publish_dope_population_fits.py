"""Publish complete, hash-bound DOPE population fit custody; no utility claims."""
from __future__ import annotations

from collections import Counter
import csv
import io
import json
import math
from pathlib import Path

from .manifest import digest
from .publish_s3_forest import check, read, require, safe
from .publish_s3_matched import PROFILES, schema
from .score import sha256

ROOT = Path('/mnt/fast-scratch/dope-benchmark/dope-s3-population-research-v2')
HERE = Path(__file__).parent
RESULTS = HERE / 'results'
NAME = 'dope-s3-population-gpu-fits'
ROUND_SHA = '87d031fe6d1d49c9c86417d6f4e8995b16de507d89a8fefe8f30fc6ed20e9d64'
RECEIPT_SHA = 'b0471a8936042b907c4481fb2b84e2205bedd2fbec7538bfd26d1f8d07d94461'
REPORT_SHA = '76132e62124fd1535c741df86d0e46525c64a315fdda0268e6c278923fdb23f4'


def anchored(root, round_sha, receipt_sha, report_sha):
    refs = {}
    check(root / 'round.lock.json', round_sha, refs)
    check(root / 'receipt-lock-v1.json', receipt_sha, refs)
    anchor = read(root / 'receipt-lock-v1.json')
    require(anchor['round_sha256'] == round_sha and anchor['official_tests_opened'] is False
            and anchor['mfs_v2'] is None and anchor['ptf_v1'] is None,
            'fit custody seal changed')
    # Receipt files cannot supply replacement identities for their own contents.
    for name, expected in anchor['refs'].items(): check(name, expected, refs)
    require(anchor['reconciliation_sha256'] == report_sha, 'fit accounting identity changed')
    check(root / 'reconciliation-v1.json', report_sha, refs)
    return read(root / 'round.lock.json'), read(root / 'reconciliation-v1.json'), refs


def cell(job, row, refs):
    key = digest(job)
    require(key == row['job_sha256'] and row['dataset'] == job['dataset']
            and row['profile'] == job['research_profile'] and row['fit_seed'] == job['seed'] == 11
            and job['final'] is False, 'frozen fit identity changed')
    out = ROOT / 'attempts' / key / 'attempt-0001'
    receipt_path = out / 'receipt.json'
    require(refs[str(receipt_path)] == row['receipt_sha256'], 'fit receipt binding differs')
    receipt = read(receipt_path)
    require(receipt['job'] == job and receipt['round_sha256'] == ROUND_SHA
            and receipt['status'] == row['status'] and receipt['kind'] == row['kind']
            and receipt['official_tests_opened'] is False and receipt['counts_as_dope_win'] is False
            and receipt['mfs_v2'] is None and receipt['ptf_v1'] is None,
            'fit receipt claims or identity changed')
    require({p.name for p in out.iterdir() if p.is_file() and p.name != 'receipt.json'}
            == set(receipt['evidence_files']) and not any(p.is_symlink() for p in out.iterdir()),
            'fit evidence inventory changed')
    for name, h in receipt['evidence_files'].items():
        require(Path(name).name == name and refs[str(out / name)] == h, 'fit evidence identity differs')
    worker = safe(job['worker']['path'])
    require(not (worker / 'test.csv').exists()
            and {p.name for p in worker.iterdir()} == set(job['worker']['files']),
            'training-derived worker inventory changed')
    for name, h in job['worker']['files'].items():
        require(refs[str(worker / name)] == h, 'worker lineage identity differs')
    require(row['status'] in ('ok', 'charged_artifact_cap'), 'population status coverage changed')
    model = safe(row['model_path'])
    require(refs[str(model)] == row['model_sha256']
            and row['charged_artifact_bytes'] == model.stat().st_size + (worker / 'projection.json').stat().st_size
            and row['projection_bytes_included'] is True, 'complete generator byte charge differs')
    require((row['status'] == 'ok') == (row['charged_artifact_bytes'] <= 10240)
            and row['gpu_target_operator_verified'] is (row['status'] == 'ok'),
            'artifact cap failure was relabeled')
    require(math.isfinite(row['new_operation_seconds']) and row['new_operation_seconds'] >= 0
            and row['historical_runtime_closure_upgraded'] is False,
            'fit cost or historical closure claim changed')
    return {'dataset': job['dataset'], 'profile': job['research_profile'], 'fit_seed': 11,
            'status': row['status'], 'kind': row['kind'], 'fit_job_sha256': key,
            'charged_artifact_bytes': row['charged_artifact_bytes'], 'model_bytes': model.stat().st_size,
            'model_sha256': row['model_sha256'],
            'projection_bytes': (worker / 'projection.json').stat().st_size,
            'artifact_within_l3_cap': row['status'] == 'ok',
            'new_operation_seconds': row['new_operation_seconds'], 'host_operation_seconds': row['host_operation_seconds'],
            'receipt': {'path': str(receipt_path), 'sha256': row['receipt_sha256']},
            'gpu_target_operator_verified': row['gpu_target_operator_verified'],
            'historical_runtime_closure_upgraded': False, 'counts_as_dope_win': False,
            'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None, 'paired_superiority': None}


def summarize(cells):
    ids = sorted({r['dataset'] for r in cells})
    identities = [(r['dataset'], r['profile']) for r in cells]
    require(len(ids) == 100 and len(identities) == len(set(identities)) == 400
            and set(identities) == {(d, p) for d in ids for p in PROFILES},
            'complete population fit matrix required')
    require(all(r['fit_seed'] == 11 and r['counts_as_dope_win'] is False
                and all(r[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'paired_superiority'))
                for r in cells), 'fit panel acquired a gated claim')
    return [{'profile': p, 'closed': 100, 'status_counts': dict(Counter(r['status'] for r in cells if r['profile'] == p)),
             'new_operation_seconds': sum(r['new_operation_seconds'] for r in cells if r['profile'] == p)} for p in PROFILES]


def build():
    lock, report, refs = anchored(ROOT, ROUND_SHA, RECEIPT_SHA, REPORT_SHA)
    require(report['all_frozen_fit_cells_accounted'] is True and report['closed_fit_cells'] == 400
            and report['datasets'] == 100 and report['shared_validation_evaluation_complete'] is False
            and report['global_family_selected'] is False and report['full_campaign_admitted'] is False
            and report['official_tests_opened'] is False
            and all(report[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
            'population fit reconciliation scope changed')
    require(read(ROOT / 'completion.json')['jobs'] == 400
            and read(ROOT / 'coordinator-exit.json')['exit_code'] == 0,
            'population coordinator did not close')
    require({str(p) for p in (ROOT / 'source').rglob('*') if p.is_file()} == set(lock['source_files'])
            and not any(p.is_symlink() for p in (ROOT / 'source').rglob('*')),
            'frozen executable source inventory changed')
    jobs = {digest(j): j for j in lock['jobs']}
    require(len(jobs) == 400 and lock['allowed_gpu_hosts'] == ['xbabe1', 'xbabe2']
            and lock['blocked_gpu_hosts'] == ['xbabe0', 'xbabe3'], 'fit host or matrix scope changed')
    cells = [cell(jobs[r['job_sha256']], r, refs) for r in report['fit_cells']]
    summary = summarize(cells)
    counts = dict(Counter(r['status'] for r in cells))
    require(counts == report['fit_status_counts']
            and dict(Counter(r['kind'] for r in cells)) == report['fit_kind_counts']
            and math.isclose(sum(r['new_operation_seconds'] for r in cells), report['new_operation_seconds'], abs_tol=1e-9),
            'fit summary or cost reconciliation differs')
    prior = report['prior_cost']
    return {'format': 'dope-s3-population-gpu-fit-publication', 'version': 1,
            'scope': 'Complete four-profile, single-fit-seed architecture research on 100 S3 training-derived validation lineages; fit custody only',
            'source_sha256': sha256(Path(__file__)), 'source_round_sha256': ROUND_SHA,
            'receipt_lock_sha256': RECEIPT_SHA, 'reconciliation_sha256': REPORT_SHA,
            's3_data_lock_sha256': sha256(RESULTS / 's3-data.lock.json'),
            'dataset_ids': sorted({r['dataset'] for r in cells}), 'closed_fit_cells': 400,
            'status_counts': counts, 'fit_kind_counts': report['fit_kind_counts'], 'summary': summary, 'cells': cells,
            'cost': {'new_fit_operation_seconds': report['new_operation_seconds'], 'host_operation_seconds': report['host_operation_seconds'],
                     'coordinator_wall_seconds': read(ROOT / 'coordinator-exit.json')['elapsed_seconds'],
                     'peak_observed_device_used_mib': report['peak_gpu_used_mib'],
                     'prior_closed_fit_seconds': prior['prior_closed_fit_seconds'],
                     'prior_wrapper_new_operation_transport_seconds': prior['prior_wrapper_new_operation_transport_seconds'],
                     'architecture_research_spend_separate_from_final_cell_parity': True, 'energy_attributable_to_job': None},
            'shared_validation_complete': False, 'global_family_selected': False, 'full_campaign_admitted': False,
            'privacy_attack_coverage_complete': False, 'system_dynamic_library_closure_certified': False,
            'production_certified': False, 'official_tests_opened': False,
            'mfs_v2': None, 'ptf_v1': None, 'release_safe_l3': None, 'paired_superiority': None, 'counts_as_dope_win': False,
            'frozen_reference_files_verified': len(refs),
            # The closed receipt lock retains the complete detailed inventory
            # on scratch; public outputs bind its hash and each cell receipt.
            'immutable_references': [{'path': str(ROOT / name), 'sha256': refs[str(ROOT / name)]}
                                     for name in ('round.lock.json', 'receipt-lock-v1.json', 'reconciliation-v1.json',
                                                  'completion.json', 'coordinator-exit.json', 'prior-fit-evidence.lock.json',
                                                  'execution.lock.json', 'freeze-proof.json', 'transport-execution-v1.lock.json')]}


def tables(report):
    require(summarize(report['cells']) == report['summary'], 'committed fit matrix reconciliation differs')
    out = io.StringIO(); writer = csv.writer(out, lineterminator='\n')
    writer.writerow(['dataset', 'profile', 'fit_seed', 'status', 'kind', 'charged_artifact_bytes', 'model_bytes', 'projection_bytes', 'new_operation_seconds'])
    for r in report['cells']:
        writer.writerow([r[k] for k in ('dataset','profile','fit_seed','status','kind','charged_artifact_bytes','model_bytes','projection_bytes','new_operation_seconds')])
    lines = ['# DOPE all 100 S3 GPU fit custody', '',
             'Training results only. Single fit seed 11; all four declared profiles remain visible. Official tests sealed; MFS-v2/PTF-v1/release-safe/superiority null.', '',
             '| Profile | Closed | Fits within 10,240 charged bytes | Byte-cap failures |',
             '|---|---:|---:|---:|']
    for row in report['summary']:
        lines.append(f"| {row['profile']} | {row['closed']} | {row['status_counts'].get('ok',0)} | {row['status_counts'].get('charged_artifact_cap',0)} |")
    lines += ['', 'Charges include the encoded generator and projection. Failures count as no DOPE win.', '',
              f"New fit operations: {report['cost']['new_fit_operation_seconds']:.6f}s; coordinator wall: {report['cost']['coordinator_wall_seconds']:.6f}s. 354 new fits and 46 immutable prior successes; prior research/repair costs remain separate.", '',
              'Shared validation, final five-fit coverage, full privacy controls, public-core paired analysis and final campaign admission remain incomplete. No production family selected.', '']
    return out.getvalue(), '\n'.join(lines)


def main():
    report = build(); path = RESULTS / (NAME + '.json')
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(report), sort_keys=True, indent=2) + '\n')
    table, markdown = tables(report)
    path.with_suffix('.csv').write_text(table); path.with_suffix('.md').write_text(markdown)
    print(path)


if __name__ == '__main__': main()
