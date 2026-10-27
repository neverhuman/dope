"""Q2: cold public-call latency of `dope mlc-v3 encode` per landmark mode and device.

Each repetition is a fresh process (cold: weight load, device init, CSV read, preparation,
native evidence, conditioning, forward, JSON). Cells run sequentially under the shared
qualification lock with a declared CPU affinity. Records per call: wall time, the CLI's
own latency_ms per emitted record (progressive: time to seq 0 and to the last record),
final status/recipe/options digest and exit code. Gate per the shared protocol: per cell
p50(full)/p50(fast) on cold GPU calls; every cell and failure is reported.
"""
import argparse, csv, json, os, statistics, subprocess, time


def call(binary, weights, row, target, mode, device, threads, cpus):
    cmd = ['taskset', '-c', cpus, binary, 'mlc-v3', 'encode', '--weights', weights, '--csv', row['csv'],
           '--target', target, '--task', row['task'], '--landmark-mode', mode, '--device', device,
           '--threads', str(threads), '--json']
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, RAYON_NUM_THREADS=str(threads)))
    wall = time.perf_counter() - t
    recs = []
    for line in p.stdout.decode(errors='replace').splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    last = recs[-1] if recs else {}
    return dict(wall_ms=round(wall * 1e3, 3), exit=p.returncode, records=len(recs),
                first_latency_ms=recs[0].get('latency_ms') if recs else None, last_latency_ms=last.get('latency_ms'),
                status=last.get('status'), recipe=last.get('recipe'), options_digest=last.get('options_digest'),
                work_budget=last.get('work_budget'), device_used=last.get('device'),
                stderr=p.stderr.decode(errors='replace')[-200:] if p.returncode else '')


def q(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round((len(xs) - 1) * p)))], 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True); ap.add_argument('--weights', required=True)
    ap.add_argument('--manifest', required=True); ap.add_argument('--ids', required=True)
    ap.add_argument('--devices', default='cuda:0,cpu'); ap.add_argument('--modes', default='none,progressive,full')
    ap.add_argument('--reps', type=int, default=15); ap.add_argument('--slow-reps', type=int, default=5)
    ap.add_argument('--slow-ms', type=float, default=3000.0, help='cells whose first full call exceeds this use --slow-reps')
    ap.add_argument('--threads', type=int, default=8); ap.add_argument('--cpus', default='64-71')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = {r['id']: r for r in map(json.loads, open(a.manifest))}
    os.makedirs(a.out, exist_ok=True)
    meta = dict(bin=a.bin, devices=a.devices, modes=a.modes, reps=a.reps, slow_reps=a.slow_reps, threads=a.threads,
                cpus=a.cpus, loadavg_start=os.getloadavg(), started=time.time())
    calls = open(os.path.join(a.out, 'calls.jsonl'), 'a')
    cells = []
    for tid in a.ids.split(','):
        row = rows[tid]
        with open(row['csv'], newline='') as f:
            target = next(csv.reader(f))[-1].strip()
        for device in a.devices.split(','):
            first_full = call(a.bin, a.weights, row, target, 'full', device, a.threads, a.cpus)  # warm-up probe, recorded
            reps = a.slow_reps if first_full['wall_ms'] > a.slow_ms else a.reps
            res = {}
            for mode in a.modes.split(','):
                rs = []
                for k in range(reps):
                    r = call(a.bin, a.weights, row, target, mode, device, a.threads, a.cpus)
                    r.update(id=tid, device=device, mode=mode, rep=k, loadavg=os.getloadavg()[0])
                    calls.write(json.dumps(r) + '\n'); calls.flush()
                    rs.append(r)
                ok = [r for r in rs if r['exit'] == 0]
                res[mode] = dict(n=len(rs), failures=len(rs) - len(ok), wall_p50_ms=q([r['wall_ms'] for r in ok], .5),
                                 wall_p95_ms=q([r['wall_ms'] for r in ok], .95),
                                 first_record_p50_ms=q([r['first_latency_ms'] for r in ok if r['first_latency_ms'] is not None], .5),
                                 status=ok[-1]['status'] if ok else None, recipe=ok[-1]['recipe'] if ok else None,
                                 work_budget=ok[-1]['work_budget'] if ok else None)
            fast, full = res.get('none', {}).get('wall_p50_ms'), res.get('full', {}).get('wall_p50_ms')
            cell = dict(id=tid, rows=row['n'], p=row['p'], task=row['task'], device=device, reps=reps, modes=res,
                        full_over_fast_p50=round(full / fast, 2) if fast and full else None, probe_full_ms=first_full['wall_ms'])
            cells.append(cell)
            print(json.dumps({k: cell[k] for k in ('id', 'device', 'reps', 'full_over_fast_p50')}), flush=True)
    meta.update(loadavg_end=os.getloadavg(), finished=time.time())
    json.dump(dict(meta=meta, cells=cells), open(os.path.join(a.out, 'matrix.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
