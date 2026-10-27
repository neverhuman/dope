"""Run `dope landmarks real --json` over the corpus manifest with one binary.

Writes <out>/<threads>/<id>.json (the report bytes) and <out>/results.jsonl
(id, threads, sha256, wall_s, exit). Used for both the byte-identity sweep and
whole-panel timing (H1). Concurrency: --jobs tables at a time, each with
--threads Rayon threads; keep jobs*threads within the bench window budget.
"""
import argparse, concurrent.futures as cf, csv, hashlib, json, os, subprocess, time

OPTS = dict(folds=5, knn_reference=2048, ridge_rows=8192, seed=0, small_rows=4096,
            spectrum_features=64, work_budget=2_000_000_000_000)


def run(binary, row, threads, out, opts):
    with open(row['csv'], newline='') as f:
        target = len(next(csv.reader(f))) - 1
    cmd = [binary, '--threads', str(threads), 'landmarks', 'real', row['csv'], '--target', str(target),
           '--task', row['task'], '--json']
    for k, v in opts.items():
        cmd += ['--' + k.replace('_', '-'), str(v)]
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, env=env)
    wall = time.perf_counter() - t0
    path = os.path.join(out, str(threads), row['id'] + '.json')
    with open(path, 'wb') as fh:
        fh.write(p.stdout)
    return dict(id=row['id'], threads=threads, exit=p.returncode, wall_s=round(wall, 4),
                sha256=hashlib.sha256(p.stdout).hexdigest(), bytes=len(p.stdout),
                stderr=p.stderr.decode(errors='replace')[-300:] if p.returncode else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', default='1,8')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--work-budget', type=int, default=OPTS['work_budget'])
    ap.add_argument('--filter', default='')
    a = ap.parse_args()
    opts = dict(OPTS, work_budget=a.work_budget)
    rows = [json.loads(l) for l in open(a.manifest)]
    rows = [r for r in rows if a.filter in r['id']]
    rows.sort(key=lambda r: -r['n'] * r['p'])          # longest first
    ths = [int(t) for t in a.threads.split(',')]
    for t in ths:
        os.makedirs(os.path.join(a.out, str(t)), exist_ok=True)
    meta = dict(bin=a.bin, bin_sha256=hashlib.sha256(open(a.bin, 'rb').read()).hexdigest(),
                options=opts, threads=ths, jobs=a.jobs, loadavg_start=os.getloadavg(), started=time.time())
    with open(os.path.join(a.out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    with cf.ThreadPoolExecutor(a.jobs) as ex, open(os.path.join(a.out, 'results.jsonl'), 'a') as res:
        futs = [ex.submit(run, a.bin, r, t, a.out, opts) for t in ths for r in rows]
        for i, f in enumerate(cf.as_completed(futs)):
            r = f.result()
            res.write(json.dumps(r) + '\n'); res.flush()
            if r['exit']:
                print('FAIL', r['id'], r['threads'], r['stderr'][-120:], flush=True)
    meta.update(loadavg_end=os.getloadavg(), finished=time.time())
    with open(os.path.join(a.out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('done', len(futs))


if __name__ == '__main__':
    main()
