"""Per-probe wall time via `dope landmarks real --ids <id> --spectrum-features 0`.

Each run prepares folds again, so the cost of an id-14 (`missing`) run is reported
as `prep_s` and subtracted. Also times the spectrum alone (all ids minus the
no-spectrum full run). Single thread; median of --reps runs.
"""
import argparse, csv, json, os, statistics, subprocess, time

NAMES = 'lin log sqrt pow sat abs hinge step periodic bump rows pairs rff missing best1 stump otree3 gbdt knn1 knn16 net lin_quarter gbdt_quarter'.split()


def wall(binary, row, extra, threads):
    with open(row['csv'], newline='') as f:
        target = len(next(csv.reader(f))) - 1
    cmd = [binary, '--threads', str(threads), 'landmarks', 'real', row['csv'], '--target', str(target), '--task', row['task'],
           '--json', '--work-budget', '2000000000000'] + extra
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, RAYON_NUM_THREADS=str(threads)))
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-200:])
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True); ap.add_argument('--manifest', required=True)
    ap.add_argument('--ids', required=True, help='comma list of table ids')
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    rows = {r['id']: r for r in map(json.loads, open(a.manifest))}
    for tid in a.ids.split(','):
        row = rows[tid]
        med = lambda extra: statistics.median(wall(a.bin, row, extra, a.threads) for _ in range(a.reps))
        prep = med(['--ids', '14', '--spectrum-features', '0'])
        cells = {NAMES[i - 1]: round(max(med(['--ids', str(i), '--spectrum-features', '0']) - prep, 0), 3) for i in range(1, 24)}
        nospec = med(['--spectrum-features', '0'])
        full = med([])
        print(json.dumps(dict(id=tid, rows=row['n'], p=row['p'], threads=a.threads, prep_s=round(prep, 3),
                              panel_nospectrum_s=round(nospec, 3), spectrum_s=round(full - nospec, 3), full_s=round(full, 3),
                              cells_minus_prep_s=dict(sorted(cells.items(), key=lambda kv: -kv[1])))), flush=True)


if __name__ == '__main__':
    main()
