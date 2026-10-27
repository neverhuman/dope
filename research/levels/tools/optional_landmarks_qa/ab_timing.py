"""Interleaved A/B wall time of `dope landmarks real --json` (A, B, A, B, ...) per table.

Each table's repetitions alternate binaries so host-load drift hits both equally; tables
run --jobs at a time. Reports per-table median wall and the A/B ratio, plus whether the
report bytes matched on every repetition.
"""
import argparse, concurrent.futures as cf, csv, hashlib, json, os, statistics, subprocess, time


def once(binary, row, threads):
    with open(row['csv'], newline='') as f:
        target = len(next(csv.reader(f))) - 1
    cmd = [binary, '--threads', str(threads), 'landmarks', 'real', row['csv'], '--target', str(target), '--task', row['task'],
           '--json', '--work-budget', '2000000000000']
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, RAYON_NUM_THREADS=str(threads)))
    return time.perf_counter() - t, hashlib.sha256(p.stdout).hexdigest(), p.returncode


def table(a, b, row, reps, threads):
    ta, tb, hs = [], [], set()
    for _ in range(reps):
        for binary, acc in ((a, ta), (b, tb)):
            w, h, rc = once(binary, row, threads)
            acc.append(w); hs.add((h, rc))
    ma, mb = statistics.median(ta), statistics.median(tb)
    return dict(id=row['id'], n=row['n'], p=row['p'], task=row['task'], a_med_s=round(ma, 3), b_med_s=round(mb, 3),
                speedup=round(ma / mb, 3), bytes_identical=len(hs) == 1, a_all=[round(x, 3) for x in ta], b_all=[round(x, 3) for x in tb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--manifest', required=True); ap.add_argument('--ids', required=True)
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--out', required=True)
    x = ap.parse_args()
    rows = {r['id']: r for r in map(json.loads, open(x.manifest))}
    want = x.ids.split(',')
    meta = dict(a=x.a, b=x.b, threads=x.threads, reps=x.reps, loadavg_start=os.getloadavg())
    with cf.ThreadPoolExecutor(x.jobs) as ex:
        res = list(ex.map(lambda i: table(x.a, x.b, rows[i], x.reps, x.threads), want))
    meta['loadavg_end'] = os.getloadavg()
    json.dump(dict(meta=meta, tables=res), open(x.out, 'w'), indent=1)
    for r in res:
        print(f"{r['id']:45s} n={r['n']:6d} p={r['p']:4d} {r['a_med_s']:8.2f}s -> {r['b_med_s']:8.2f}s  {r['speedup']:5.2f}x  identical={r['bytes_identical']}")
    print('loadavg', meta['loadavg_start'], '->', meta['loadavg_end'])


if __name__ == '__main__':
    main()
