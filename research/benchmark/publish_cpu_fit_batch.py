"""Publish a frozen prefix of real CPU fit receipts; incomplete matrices stay visible."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path

GATES = dict(official_tests_opened=False, full_campaign_admitted=False,
             common_metrics_complete=False, five_fit_matrix_complete=False,
             native_values_ranked_across_methods=False, mfs_v2=None, ptf_v1=None,
             release_safe=None, superiority=None)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def verified(ref, read_body=True):
    p = Path(ref['path'])
    require(type(ref['sha256']) is str and len(ref['sha256']) == 64,
            'invalid receipt digest')
    require(p.is_absolute() and p == p.resolve(strict=True)
            and not any(q.is_symlink() for q in (p, *p.parents))
            and p.name != 'test.csv' and 'evaluator' not in p.parts,
            'unowned publication input')
    require(not read_body or p.stat().st_size <= 64_000_000, 'oversized publication metadata')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    require(h.hexdigest() == ref['sha256'], 'frozen receipt drift')
    data = p.read_bytes() if read_body else None
    if data is not None:
        require(hashlib.sha256(data).hexdigest() == ref['sha256'], 'frozen receipt drift')
    return data


def build(manifest):
    require(manifest['format'] == 'dope-cpu-fit-batch-input-v1', 'batch manifest differs')
    rows = []
    seen = set()
    for entry in manifest['fits']:
        # Authenticate receipts before decoding any metric values.
        blobs = {key: verified(entry[key]) for key in ('round', 'job', 'fit', 'close')}
        round_lock, job, fit, close = (json.loads(blobs[k]) for k in ('round', 'job', 'fit', 'close'))
        jid = identity(job)
        require(jid not in seen, 'duplicate physical fit')
        seen.add(jid)
        require(any(j['job_sha256'] == jid and j['file_sha256'] == entry['job']['sha256']
                    for j in round_lock['jobs']), 'job not in frozen matrix')
        require(fit['job_sha256'] == jid and fit['round_sha256'] == entry['round']['sha256']
                and close['round_sha256'] == entry['round']['sha256']
                and close['fit_receipt_sha256'] == entry['fit']['sha256'], 'fit closure differs')
        require(round_lock['official_tests_opened'] is False and fit['official_tests_opened'] is False
                and all(fit[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
                'unsupported publication claim')
        for source, expected in round_lock['source_files'].items():
            verified(dict(path=source, sha256=expected))
        runtime = Path(entry['round']['path']).parent/'runtime.lock.json'
        verified(dict(path=str(runtime), sha256=round_lock['runtime_sha256']))
        require(type(job['fit_seed']) is int and job['fit_seed'] in (23, 37, 53, 71),
                'replication seed differs')
        require(fit['status'] in ('ok', 'failed', 'infra', 'timeout'), 'unknown fit outcome')
        value, objective, charged, inventory = None, None, None, None
        if fit['status'] == 'ok':
            require(fit['fit_seed'] == job['fit_seed'] and fit['dataset'] == job['dataset']
                    and fit['cpu_affinity'] == round_lock['cpu_affinity']
                    and close['worker_exit_code'] == 0, 'successful fit identity differs')
            inventory = fit['artifact_inventory']
            inventory = ([dict(path=k, **v) for k, v in inventory.items()]
                         if type(inventory) is dict else inventory)
            require(type(inventory) is list and len({r['path'] for r in inventory}) == len(inventory)
                    and 'projection.json' in {r['path'] for r in inventory}
                    and all(type(r['bytes']) is int and r['bytes'] >= 0
                            and Path(r['path']).name == r['path'] for r in inventory),
                    'complete artifact accounting required')
            charged = sum(r['bytes'] for r in inventory)
            require(type(fit['artifact_bytes']) is int and fit['artifact_bytes'] == charged,
                    'charged artifact bytes differ')
            artifact = Path(entry['fit']['path']).parent/'artifact'
            require({p.name for p in artifact.iterdir()} == {r['path'] for r in inventory},
                    'artifact inventory differs')
            for row in inventory:
                path = artifact/row['path']
                verified(dict(path=str(path), sha256=row['sha256']), read_body=False)
                require(path.stat().st_size == row['bytes'], 'artifact size differs')
            require(type(fit['elapsed_seconds']) in (int, float) and math.isfinite(fit['elapsed_seconds'])
                    and 0 <= fit['elapsed_seconds'] <= round_lock['fit_timeout_seconds'],
                    'fit timer differs')
            if job['method'] == 'ARF':
                kpi = fit['native_kpi']
                require(kpi['objective'] == 'heldout_forde_mean_log_density'
                        and kpi['direction'] == 'maximize' and kpi['partition'] == 'validation'
                        and kpi['fit_seed'] == job['fit_seed']
                        and kpi['validation_sha256'] == job['worker']['files']['validation.csv']
                        and type(kpi['value']) in (float, int) and math.isfinite(kpi['value']),
                        'native density evidence differs')
                value, objective = kpi['value'], kpi['objective']
            else:
                require(job['method'] == 'ForestDiffusion/Forest-Flow'
                        and fit['retained_training_row_containers'] is False,
                        'Forest-Flow fit contract differs')
        rows.append(dict(method='ARF' if job['method'] == 'ARF' else 'Forest-Flow',
                         dataset=job['dataset'], fit_seed=job['fit_seed'],
                         configuration_labels=job.get('configuration_labels', [job.get('configuration_name')]),
                         configuration_sha256=identity(job.get('configuration', job.get('config'))),
                         status=fit['status'], job_sha256=jid,
                         native_objective=objective, native_validation_value=value,
                         artifact_bytes=charged, artifact_inventory=inventory,
                         elapsed_seconds=fit.get('elapsed_seconds'), cpu_affinity=round_lock['cpu_affinity'],
                         train_sha256=job['worker']['files']['train.csv'],
                         validation_sha256=job['worker']['files']['validation.csv'],
                         projection_sha256=job['worker']['files']['projection.json'],
                         receipt=entry['fit'], closure=entry['close'], round=entry['round'], job=entry['job']))
    rows.sort(key=lambda x: (x['method'], x['dataset'], x['fit_seed'], x['job_sha256']))
    return dict(format='dope-cpu-fivefit-progress-v1', version=1,
                snapshot_utc=manifest['snapshot_utc'], fits=rows,
                physical_fit_counts=dict(Counter(r['method'] for r in rows)),
                outcome_counts=dict(Counter(r['status'] for r in rows)),
                planned_new_physical_fits=manifest['planned_new_physical_fits'],
                forest_native_selected_replications='not_admitted_in_this_round',
                **GATES)


def render(panel):
    buffer = io.StringIO()
    fields = ['method', 'dataset', 'fit_seed', 'status', 'native_validation_value',
              'artifact_bytes', 'elapsed_seconds', 'job_sha256']
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in panel['fits']:
        writer.writerow({k: row[k] for k in fields})
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    panel = build(json.loads(verified(dict(path=args.manifest, sha256=args.manifest_sha256))))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'panel.json').write_text(json.dumps(panel, sort_keys=True, indent=2, allow_nan=False)+'\n')
    (out/'fits.csv').write_text(render(panel))


if __name__ == '__main__':
    main()
