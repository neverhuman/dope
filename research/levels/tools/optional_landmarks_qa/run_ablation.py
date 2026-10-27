"""Subsample ablation (spec 2.3/5.4): native landmark bundles at N in SIZES vs full rows.

For each table: `dope landmarks real --emit-bundle` once unsampled and once per N
(`--subsample N --subsample-seed 0`), plus per-arm wall time. Writes
<out>/bundles/<id>.<arm>.json and <out>/results.jsonl. Metrics are computed by
summarize_ablation.py. N >= rows is skipped (identical to full by construction).
"""
import argparse, concurrent.futures as cf, csv, json, os, subprocess, time

SIZES = (100, 200, 300, 500, 1000)
OPTS = ['--folds', '5', '--knn-reference', '2048', '--ridge-rows', '8192', '--seed', '0', '--small-rows', '4096',
        '--spectrum-features', '64', '--work-budget', '2000000000000']


def arm(binary, row, n, threads, out, seed='0'):
    with open(row['csv'], newline='') as f:
        target = len(next(csv.reader(f))) - 1
    name = 'full' if n is None else f'n{n}'
    bundle = os.path.join(out, 'bundles', f"{row['id']}.{name}.json")
    report = os.path.join(out, 'reports', f"{row['id']}.{name}.json")
    cmd = [binary, '--threads', str(threads), 'landmarks', 'real', row['csv'], '--target', str(target),
           '--task', row['task'], '--json', '--emit-bundle', bundle, '-o', report] + [seed if o == '0' and OPTS[k - 1] == '--seed' else o for k, o in enumerate(OPTS)]
    if n is not None:
        cmd += ['--subsample', str(n), '--subsample-seed', '0']
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, RAYON_NUM_THREADS=str(threads)))
    return dict(id=row['id'], arm=name, n_sub=n, rows=row['n'], p=row['p'], task=row['task'],
                exit=p.returncode, wall_s=round(time.perf_counter() - t0, 4),
                stderr=p.stderr.decode(errors='replace')[-300:] if p.returncode else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=48)
    ap.add_argument('--seed', default='0', help='CV seed for every arm (noise-floor control: rerun full with another seed)')
    ap.add_argument('--sizes', default=','.join(map(str, SIZES)))
    a = ap.parse_args()
    for d in ('bundles', 'reports'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    rows = [json.loads(l) for l in open(a.manifest)]
    sizes = [int(x) for x in a.sizes.split(',') if x]
    work = [(r, None) for r in rows] + [(r, n) for r in rows for n in sizes if n < r['n']]
    work.sort(key=lambda w: -(w[0]['n'] if w[1] is None else w[1]) * w[0]['p'])
    with open(os.path.join(a.out, 'meta.json'), 'w') as fh:
        json.dump(dict(bin=a.bin, seed=a.seed, threads=a.threads, jobs=a.jobs, sizes=SIZES, options=OPTS,
                       loadavg_start=os.getloadavg(), started=time.time()), fh, indent=1)
    with cf.ThreadPoolExecutor(a.jobs) as ex, open(os.path.join(a.out, 'results.jsonl'), 'a') as res:
        for f in cf.as_completed([ex.submit(arm, a.bin, r, n, a.threads, a.out, a.seed) for r, n in work]):
            r = f.result()
            res.write(json.dumps(r) + '\n'); res.flush()
            if r['exit']:
                print('FAIL', r['id'], r['arm'], r['stderr'][-150:], flush=True)
    print('done', len(work), 'loadavg_end', os.getloadavg())


if __name__ == '__main__':
    main()
