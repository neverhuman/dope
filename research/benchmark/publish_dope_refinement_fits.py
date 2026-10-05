"""Publish the complete refinement fit matrix without quality or release claims."""
import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import stat

from . import arf_runtime_guard as guard
from .manifest import digest
from .publish_s3_matched import schema
from .score import sha256

BASE = Path('/mnt/fast-scratch/dope-benchmark')
ROOT = BASE / 'dope-s3-refinement-expansion-v2'
ROUND = 'fe53444eeb3b485f895473a27101d93fb515c9dfbe3ee72d50d71417fef487d4'
RECEIPTS = '179d005ddbc883099f41bb243026748b3d4508171734745c0bed80b3d4356c88'
REPORT = 'e65806dc58a6f92072307b0821fe4345e22914b968cde563500fb1219f7bc295'
DATA = '857b61324d0a5b7dbbb98e0c147fcc79c5bb8616e7ca61afd6785469c17f038f'
NAME = 'dope-target-refinement-population-fits'
PROFILES = ('features12_steps8192', 'features12_width8_steps8192')
STATUSES = ('ok', 'charged_artifact_cap', 'transport_or_prelaunch_failure', 'foreign_gpu_owner_appeared')
GATES = dict(global_family_selected=False, full_campaign_admitted=False,
    official_tests_opened=False, counts_as_dope_win=False, production_certified=False,
    mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None)


def safe(path):
    p = guard.safe(path, BASE)
    guard.require('evaluator' not in p.parts and p.name != 'test.csv', 'sealed input forbidden')
    return p


def bound(path, expected):
    return guard.bound(safe(path), expected, BASE)


def checked(path, expected, refs):
    guard.digest(expected); p = safe(path)
    guard.require(stat.S_ISREG(p.lstat().st_mode) and sha256(p) == expected
        and (str(p) not in refs or refs[str(p)] == expected), 'immutable evidence changed')
    refs[str(p)] = expected


SCRATCH_DIRECTORIES = frozenset(('catboost_info', 'runtime-temp'))


def flat(root, files):
    root = safe(root)
    names = []
    for entry in root.iterdir():
        info = entry.lstat()
        if stat.S_ISDIR(info.st_mode) and entry.name in SCRATCH_DIRECTORIES:
            continue
        names.append(entry.name)
    guard.require(set(names) == set(files), 'flat inventory differs')
    for name, pin in files.items():
        guard.require(type(name) is str and Path(name).name == name and name not in ('', '.', '..'),
                      'invalid flat member')
        p = safe(root / name)
        guard.digest(pin)
        guard.require(stat.S_ISREG(p.lstat().st_mode) and sha256(p) == pin, 'flat member changed')


def anchored():
    refs = {}
    checked(ROOT / 'receipt-lock-v1.json', RECEIPTS, refs)
    anchor = bound(ROOT / 'receipt-lock-v1.json', RECEIPTS)
    guard.require(anchor['complete_fit_matrix'] is True and anchor['round_sha256'] == ROUND
        and anchor['reconciliation_sha256'] == REPORT, 'complete external fit anchor required')
    # Authenticate all receipts and executable/input bytes before decoding outcomes.
    for path, pin in anchor['refs'].items(): checked(path, pin, refs)
    checked(ROOT / 'reconciliation-v1.json', REPORT, refs)
    guard.require(refs[str(ROOT / 'round.lock.json')] == ROUND
        and refs[str(ROOT / 'reconciliation-v1.json')] == REPORT, 'closure bindings differ')
    lock = bound(ROOT / 'round.lock.json', ROUND)
    # Include predecessor executable inventories; reject newly added aliases/directories.
    for path, pin in list(refs.items()):
        if Path(path).name == 'round.lock.json':
            parent = bound(path, pin)
            if 'source_files' in parent:
                source = Path(path).parent / 'source'
                guard.require(all(Path(p).parent == source for p in parent['source_files']),
                              'source inventory escaped its root')
                flat(source, {Path(p).name: h for p, h in parent['source_files'].items()})
    actual = bound(ROOT / 'coordinator-exit.json', refs[str(ROOT / 'coordinator-exit.json')])
    done = bound(ROOT / 'completion.json', refs[str(ROOT / 'completion.json')])
    launch = bound(ROOT / 'supervisor-launch.json', refs[str(ROOT / 'supervisor-launch.json')])
    guard.require(type(actual['exit_code']) is int and actual['exit_code'] == 0
        and actual['round_sha256'] == done['round_sha256'] == launch['round_sha256'] == ROUND
        and type(done['jobs']) is int and done['jobs'] == 176
        and all(not Path('/proc', str(launch[k])).exists() for k in ('pid', 'supervisor_pid')),
        'actual clean fit closure required')
    return lock, bound(ROOT / 'reconciliation-v1.json', REPORT), refs


def cell(row, refs):
    job = row['original_fit_job']; key = digest(job)
    guard.require(key == row['job_sha256'] and job['final'] is False
        and type(job['seed']) is int and type(row['fit_seed']) is int and job['seed'] == row['fit_seed'] == 11
        and job['method'] == 'DOPE' and job['dataset'] == row['dataset']
        and job['research_profile'] == row['profile'], 'canonical fit identity differs')
    path = safe(row['receipt_path'])
    guard.require(refs[str(path)] == row['receipt_sha256'] and path.parent.parent.name == key,
                  'fit receipt binding differs')
    receipt = bound(path, row['receipt_sha256'])
    guard.require(digest(receipt['job']) == key and receipt['status'] == row['status']
        and receipt['official_tests_opened'] is False and receipt['counts_as_dope_win'] is False
        and receipt['mfs_v2'] is None and receipt['ptf_v1'] is None, 'receipt scope differs')
    evidence = receipt['evidence_files']
    flat(path.parent, {**evidence, 'receipt.json': row['receipt_sha256']})
    for name, pin in evidence.items(): checked(path.parent / name, pin, refs)
    worker = safe(job['worker']['path']); flat(worker, job['worker']['files'])
    for name, pin in job['worker']['files'].items():
        guard.require(refs[str(worker / name)] == pin, 'worker absent from sealed evidence')
    reused = row['immutable_prior_fit_reused']
    guard.require(type(reused) is bool, 'reuse flag must be boolean')
    if not reused:
        guard.require(receipt['kind'] == 'new_gpu_refinement_fit' and receipt['round_sha256'] == ROUND
            and type(receipt['attempt']) is int and receipt['attempt'] == 1, 'new fit attempt differs')
    fit = bound(path.with_name('fit.json'), evidence['fit.json']) if 'fit.json' in evidence else None
    model_bytes = projection_bytes = None
    if row['model_path'] is not None:
        model = safe(row['model_path']); checked(model, row['model_sha256'], refs)
        model_bytes = model.stat().st_size; projection_bytes = (worker / 'projection.json').stat().st_size
        guard.require(row['charged_artifact_bytes'] == model_bytes + projection_bytes
            and row['projection_bytes_included'] is True, 'projection charge omitted')
    status = row['status']
    if status == 'ok':
        guard.require(fit is not None and fit['status'] == 'ok' and fit['exit_code'] == 0
            and fit['gpu_target_operator_verified'] is True and fit['gpu_process_observed'] is True
            and fit['foreign_processes_signaled'] is False and row['gpu_target_operator_verified'] is True,
            'successful GPU fit evidence absent')
    if fit is not None:
        guard.require(digest(fit['job']) == key and fit['job_sha256'] == key
            and fit['official_tests_opened'] is False and fit['foreign_processes_signaled'] is False,
            'fit record identity differs')
    return dict(dataset=row['dataset'], method='DOPE', profile=row['profile'], fit_seed=11,
        fit_job_sha256=key, immutable_prior_fit_reused=reused,
        kind='immutable_prior_reuse' if reused else 'new_gpu_refinement_fit', status=status,
        outcome_class='ok' if status == 'ok' else 'artifact_gate_failure' if status == 'charged_artifact_cap' else 'infrastructure',
        artifact_gate_failed=status == 'charged_artifact_cap', counts_as_method_failure=False,
        fit_receipt_written=fit is not None, charged_artifact_bytes=row['charged_artifact_bytes'],
        model_bytes=model_bytes, projection_bytes=projection_bytes,
        new_operation_seconds=row['new_operation_seconds'], host_operation_seconds=row['host_operation_seconds'],
        gpu_target_operator_verified=row['gpu_target_operator_verified'],
        receipt=dict(path=str(path), sha256=row['receipt_sha256']), **GATES)


def summarize(cells):
    ids = sorted({r['dataset'] for r in cells}); keys = [(r['dataset'], r['profile']) for r in cells]
    guard.require(len(ids) == 100 and len(keys) == len(set(keys)) == 200
        and set(keys) == {(d, p) for d in ids for p in PROFILES}, 'complete 200-cell matrix required')
    for r in cells:
        guard.require(type(r['fit_seed']) is int and r['fit_seed'] == 11 and r['status'] in STATUSES
            and all(r[k] is v for k, v in GATES.items()), 'fit matrix acquired an invalid claim')
        guard.require(type(r['new_operation_seconds']) in (int, float)
            and math.isfinite(r['new_operation_seconds']) and r['new_operation_seconds'] >= 0
            and type(r['immutable_prior_fit_reused']) is bool
            and r['immutable_prior_fit_reused'] is (r['kind'] == 'immutable_prior_reuse')
            and (not r['immutable_prior_fit_reused'] or r['new_operation_seconds'] == 0),
            'reuse or operation cost differs')
        status = r['status']; charged = r['charged_artifact_bytes']
        outcome = 'ok' if status == 'ok' else 'artifact_gate_failure' if status == 'charged_artifact_cap' else 'infrastructure'
        guard.require(r['outcome_class'] == outcome and r['artifact_gate_failed'] is (status == 'charged_artifact_cap')
            and r['counts_as_method_failure'] is False, 'infrastructure or artifact gate reclassified')
        if status in ('ok', 'charged_artifact_cap'):
            guard.require(type(charged) is int and charged == r['model_bytes'] + r['projection_bytes']
                and (charged <= 10240) is (status == 'ok'), 'artifact byte gate differs')
    return [dict(profile=p, closed=100, status_counts=dict(Counter(r['status'] for r in cells if r['profile'] == p)),
        new_operation_seconds=sum(r['new_operation_seconds'] for r in cells if r['profile'] == p)) for p in PROFILES]


def build():
    lock, report, refs = anchored()
    guard.require(report['all_frozen_fit_cells_accounted'] is True and report['closed_fit_cells'] == 200
        and report['datasets'] == 100 and report['shared_validation_evaluation_complete'] is False
        and all(report[k] is v for k, v in GATES.items() if k != 'production_certified')
        and lock['allowed_gpu_hosts'] == ['xbabe1', 'xbabe2', 'xbabe3']
        and lock['blocked_gpu_hosts'] == ['xbabe0'] and lock['artifact_cap_bytes'] == 10240
        and lock['fit_timeout_seconds'] == 600 and lock['gpu_vram_limit_mib'] == 16384
        and report['cumulative_operation_seconds'] <= lock['whole_refinement_operation_ceiling_seconds'],
        'bounded fit scope differs')
    rows = report['fit_cells']; new_keys = {r['job_sha256'] for r in rows if not r['immutable_prior_fit_reused']}
    guard.require(len(lock['jobs']) == len(new_keys) == 176
        and {digest(j) for j in lock['jobs']} == new_keys, 'new job inventory differs')
    cells = [cell(r, refs) for r in rows]; summary = summarize(cells)
    data_path = Path(__file__).with_name('results') / 's3-data.lock.json'
    data = guard.bound(data_path, DATA, data_path.parent)
    guard.require(data['prepared'] == 100 and data['final_evaluation_authorized'] is False
        and {r['dataset'] for r in cells} == {r['id'] for r in data['entries'] if r['status'] == 'prepared'},
        'rights-cleared cohort differs')
    counts = dict(Counter(r['status'] for r in cells))
    guard.require(counts == report['fit_status_counts'] and sum(r['immutable_prior_fit_reused'] for r in cells) == 24
        and math.isclose(sum(r['new_operation_seconds'] for r in cells), report['new_operation_seconds'], abs_tol=1e-9)
        and math.isclose(report['new_operation_seconds'] + report['prior_whole_discovery_confirmation_operation_seconds'],
                         report['cumulative_operation_seconds'], abs_tol=1e-9), 'complete fit cost accounting differs')
    for path, pin in list(refs.items()): checked(path, pin, refs)
    return dict(format='dope-refinement-population-fit-publication', version=1,
        scope='Complete bounded official-training-derived 100-lineage/two-profile research training; no quality selection.',
        source_sha256=sha256(Path(__file__)), source_round_sha256=ROUND,
        receipt_lock_sha256=RECEIPTS, reconciliation_sha256=REPORT,
        s3_data_lock_sha256=DATA,
        dataset_ids=sorted({r['dataset'] for r in cells}), closed_fit_cells=200,
        new_fit_cells=176, immutable_prior_fit_cells=24,
        new_ok_fits=sum(r['status'] == 'ok' and not r['immutable_prior_fit_reused'] for r in cells),
        status_counts=counts, summary=summary, cells=cells,
        cost=dict(new_operation_seconds=report['new_operation_seconds'],
            prior_whole_discovery_confirmation_operation_seconds=report['prior_whole_discovery_confirmation_operation_seconds'],
            cumulative_bounded_refinement_operation_seconds=report['cumulative_operation_seconds'],
            scheduler_wall_seconds=report['scheduler_wall_seconds'], host_operation_seconds=report['host_operation_seconds'],
            earlier_architecture_research_spend_separate_from_final_cell_parity=True, energy_attributable_to_job=None),
        shared_validation_complete=False, privacy_attack_coverage_complete=False,
        system_dynamic_library_closure_certified=False, frozen_reference_files_verified=len(refs),
        immutable_references=[dict(path=str(ROOT / n), sha256=refs[str(ROOT / n)]) for n in
            ('round.lock.json', 'receipt-lock-v1.json', 'reconciliation-v1.json', 'completion.json', 'coordinator-exit.json')],
        **GATES)


def tables(report):
    guard.require(summarize(report['cells']) == report['summary']
        and all(report[k] is v for k, v in GATES.items()), 'committed matrix differs')
    out = io.StringIO(); writer = csv.writer(out, lineterminator='\n')
    fields = ('dataset', 'profile', 'fit_seed', 'kind', 'status', 'outcome_class',
              'charged_artifact_bytes', 'model_bytes', 'projection_bytes', 'new_operation_seconds')
    writer.writerow(fields)
    for row in report['cells']: writer.writerow([row[k] for k in fields])
    lines = ['# DOPE refinement: complete population training ledger', '',
        '100 bounded official-training-derived S3 lineages; fit seed 11. Training custody only. Official tests sealed; MFS-v2/PTF-v1/release/superiority null.', '',
        '| Profile | Closed | OK | Transport/prelaunch | Foreign owner | Byte-cap failure |',
        '|---|---:|---:|---:|---:|---:|']
    for row in report['summary']:
        c = row['status_counts']
        lines.append(f"| {row['profile']} | 100 | {c.get('ok', 0)} | {c.get('transport_or_prelaunch_failure', 0)} | {c.get('foreign_gpu_owner_appeared', 0)} | {c.get('charged_artifact_cap', 0)} |")
    cost = report['cost']
    lines += ['', f"176 new fit cells + 24 immutable prior successes; {report['new_ok_fits']} new OK fits. Charges include model and complete projection. Infrastructure rejections and byte-cap failures remain separate; neither supplies a DOPE win.", '',
        f"New operations: {cost['new_operation_seconds']:.6f}s; whole prior discovery/confirmation: {cost['prior_whole_discovery_confirmation_operation_seconds']:.6f}s; cumulative bounded refinement: {cost['cumulative_bounded_refinement_operation_seconds']:.6f}s. Scheduler wall: {cost['scheduler_wall_seconds']:.6f}s. Prior architecture research remains a separate cost.", '',
        'No shared validation quality, native baseline comparison, global family selection, privacy certification or production score is inferred from fit success. Earlier failures remain hash-bound on scratch.', '']
    return out.getvalue(), '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--results-dir', type=Path, default=Path(__file__).with_name('results'))
    root = parser.parse_args().results_dir; root.mkdir(parents=True, exist_ok=True)
    report = build(); path = root / (NAME + '.json')
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(report), sort_keys=True, indent=2) + '\n')
    table, text = tables(report); path.with_suffix('.csv').write_text(table); path.with_suffix('.md').write_text(text)
    print(path)


if __name__ == '__main__': main()
