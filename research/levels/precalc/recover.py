"""Derive validated cache entries from saved successful native reports, without fitting."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from ..contracts import canonical, digest, require
from ..landmark_features import FEATURE_RECIPE
from .plan import fit_csv, validate_plan
from .worker import allocation_snapshot, cache_key, measured_features, private, producer, publish, sha


def recover(source_run, output, *, catalog, binary, expected_engine, source_commit, store):
    source_run, output = Path(source_run), Path(output)
    parent = validate_plan(json.loads((source_run/'plan.json').read_bytes()), catalog)
    owner = json.loads((source_run/'ownership.json').read_bytes())
    process = subprocess.run(['/bin/ps', '-p', str(owner['pid']), '-o', 'args='],
                             capture_output=True, text=True, timeout=5)
    require(process.returncode != 0 or 'research.levels.precalc' not in process.stdout
            or str(source_run) not in process.stdout, 'precalc_recovery_parent_still_active')
    terminal = json.loads((source_run/'terminal.json').read_bytes())
    require(terminal['state'] in ('complete', 'resource_or_budget_paused')
            and owner['plan_sha256'] == parent['sha256'], 'precalc_recovery_parent_terminal')
    require(len(source_commit) == 40 and all(c in '0123456789abcdef' for c in source_commit),
            'precalc_recovery_source_commit')
    current = producer(binary)
    require(current['engine_sha256'] == expected_engine, 'precalc_recovery_binary_digest')
    admission = allocation_snapshot(source_run.parent)
    require(admission['memory_available'] >= 12*2**30 and admission['disk_available'] >= 5*2**30
            and admission['load'][0] < admission['cpu_available']-8, 'precalc_recovery_resource_refused')
    cells = {c['key']: c for c in parent['cells']}
    recorded = {r['cell_key']: r for r in terminal['records']}
    candidates = []
    for path in sorted(source_run.glob('cells/*/receipt.json')):
        receipt = json.loads(path.read_bytes())
        if receipt['state'] in ('failed', 'publication_failed') and receipt.get('native_exit') == 0:
            require(receipt['cell_key'] in cells, 'precalc_recovery_unknown_case')
            require(receipt['cell_key'] in recorded and recorded[receipt['cell_key']]['key'] == receipt['key'],
                    'precalc_recovery_unrecorded_case')
            candidates.append((path, receipt))
    plan = {k: v for k, v in parent.items() if k not in ('sha256', 'cells', 'reuse_entries')}
    plan.update(cells=[cells[r['cell_key']] for _, r in candidates], reuse_entries={},
                source_commit=source_commit, parent_plan_sha256=parent.get('parent_plan_sha256', parent['sha256']),
                measurement_plan_sha256=parent['sha256'],
                purpose='validate_saved_successful_native_reports_no_new_fits')
    plan['sha256'] = digest(plan)
    output.mkdir(mode=0o700, parents=True)
    os.umask(0o077)
    os.nice(10)
    private(output/'ownership.json', dict(pid=os.getpid(), plan_sha256=plan['sha256']))
    private(output/'plan.json', plan)
    private(output/'producer.json', current)
    private(output/'admission.json', admission)
    results = []
    run_started = time.monotonic()
    for path, original in candidates:
        if time.monotonic()-run_started > 1800 or allocation_snapshot(output)['memory_available'] < 8*2**30:
            break
        started = time.monotonic()
        cell = cells[original['cell_key']]
        original_hash = sha(path)
        report_path = path.parent/'report.json'
        adopted_report_hash = sha(report_path)
        derivation = dict(recipe='validate-saved-fit-cv-report-v1', source_sha256=current['source_sha256'],
                          original_receipt_sha256=original_hash, original_failure_key=original['key'],
                          adopted_report_sha256=adopted_report_hash, measurement_plan_sha256=parent['sha256'])
        who = dict(original['producer'], derivation=derivation)
        key = cache_key(cell, plan, who)
        out = output/'cells'/key
        out.mkdir(parents=True, mode=0o700)
        receipt = dict(key=key, cell_key=cell['key'], state='repair_refused', producer=who,
                       original_failure_receipt_sha256=original_hash, original_failure_key=original['key'],
                       original_failure_receipt_locator='receipts/'+original['key']+'/'+source_run.name+'.json',
                       original_measurement_state=original['state'], report_adopted_unix=time.time(),
                       adopted_report_sha256=adopted_report_hash, new_native_executions=0,
                       new_optimizer_updates=0, model_training=False, official_test_access=False)
        try:
            require(original['producer']['engine_sha256'] == expected_engine
                    and original['key'] == cache_key(cell, parent, original['producer']),
                    'precalc_recovery_original_identity')
            native = json.loads((path.parent/'native-owner.json').read_bytes())
            require(native['key'] == original['key'] and native['binary_sha256'] == expected_engine,
                    'precalc_recovery_native_owner')
            source = source_run/'source-cache'/(cell['scope']['train_sha256']+'.csv')
            csv, scope = fit_csv(source, cell['scope'])
            require(csv == (path.parent/'fit.csv').read_bytes(), 'precalc_recovery_saved_csv_mismatch')
            require(not report_path.is_symlink() and report_path.stat().st_size <= 1048576,
                    'precalc_recovery_report_envelope')
            report_raw = report_path.read_bytes()
            require(hashlib.sha256(report_raw).hexdigest() == adopted_report_hash,
                    'precalc_recovery_report_changed')
            values, mask = measured_features(json.loads(report_raw), scope['rows'], scope['task'], plan['options'])
            manifest = dict(format='dope-fit-cv-landmark-input-v1', feature_recipe=FEATURE_RECIPE,
                            scope=scope, options=plan['options'], engine_sha256=expected_engine,
                            report_sha256=adopted_report_hash, producer=who)
            with (out/'report.json').open('xb') as stream:
                stream.write(report_raw)
            private(out/'manifest.json', manifest)
            private(out/'features.json', dict(feature_recipe=FEATURE_RECIPE, values=values, mask=mask))
            receipt.update(state='complete', valid_features=sum(mask), encoder_input_usable=any(mask),
                           original_native_cost=original, report_sha256=adopted_report_hash,
                           manifest_sha256=sha(out/'manifest.json'), feature_sha256=sha(out/'features.json'))
            store.put(receipt['original_failure_receipt_locator'], path.read_bytes())
            publish(cell, plan, receipt, out, store)
        except (OSError, ValueError, KeyError, TypeError) as error:
            receipt.update(state='repair_refused', error_type=type(error).__name__)
        receipt['elapsed_seconds'] = time.monotonic()-started
        private(out/'receipt.json', receipt)
        store.put('receipts/'+key+'/'+output.name+'.json', canonical(receipt))
        results.append(receipt)
    result = dict(format='dope-fit-cv-report-recovery-v1',
                  state='complete' if len(results) == len(candidates) else 'resource_or_budget_paused',
                  plan_sha256=plan['sha256'],
                  source_plan_sha256=parent['sha256'], scheduled=len(candidates), records=results,
                  recovered=sum(r['state'] == 'complete' for r in results),
                  refused=sum(r['state'] != 'complete' for r in results),
                  new_native_executions=0, new_optimizer_updates=0, original_failures_preserved=True)
    private(output/'terminal.json', result)
    store.put('plans/'+plan['sha256']+'.json', canonical(plan))
    return result
