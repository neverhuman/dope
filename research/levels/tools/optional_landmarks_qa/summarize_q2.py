"""Summarize q2_public_matrix.py calls into the per-cell 5x gate table (shared protocol).

Per (table, device): p50/p95 cold public-call wall time per mode, full/fast ratio of p50,
progressive time to seq 0 (CLI latency_ms of the first record) and to the last record.
Gate: every synthetic 1k-50k cell on cuda must have p50(full)/p50(none) >= 5; all cells
and failures are listed, plus a descriptive geometric mean that decides nothing.
"""
import argparse, collections, json, math


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * p)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--manifest', required=True)
    a = ap.parse_args()
    shape = {r['id']: (r['n'], r['p']) for r in map(json.loads, open(a.manifest))}
    calls = [json.loads(l) for l in open(f'{a.run}/calls.jsonl')]
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in calls:
        cells[(c['id'], c['device'])][c['mode']].append(c)
    rows = []
    for (tid, dev), modes in cells.items():
        def stat(mode, key='wall_ms', p=.5):
            v = [c[key] for c in modes.get(mode, []) if c['exit'] == 0 and c.get(key) is not None]
            return round(q(v, p), 1) if v else None
        fails = sum(1 for m in modes.values() for c in m if c['exit'] != 0)
        fast, full = stat('none'), stat('full')
        rows.append(dict(id=tid, device=dev, rows=shape.get(tid, (None, None))[0], p=shape.get(tid, (None, None))[1],
                         reps={m: len(v) for m, v in modes.items()}, failures=fails,
                         none_p50=fast, none_p95=stat('none', p=.95), full_p50=full, full_p95=stat('full', p=.95),
                         progressive_seq0_p50=stat('progressive', 'first_latency_ms'),
                         progressive_last_p50=stat('progressive'), full_over_fast=round(full / fast, 2) if fast and full else None,
                         max_loadavg=round(max(c['loadavg'] for m in modes.values() for c in m), 1),
                         full_status=next((c['status'] for c in modes.get('full', []) if c['exit'] == 0), None)))
    synth_gpu = [r for r in rows if r['id'].startswith('synth-') and r['device'].startswith('cuda')]
    gate = dict(required_cells=len(synth_gpu), passing=[r['id'] for r in synth_gpu if (r['full_over_fast'] or 0) >= 5],
                failing=[(r['id'], r['full_over_fast']) for r in synth_gpu if (r['full_over_fast'] or 0) < 5])
    gate['blanket_pass'] = not gate['failing'] and bool(synth_gpu)
    ratios = [r['full_over_fast'] for r in synth_gpu if r['full_over_fast']]
    gate['descriptive_geomean'] = round(math.exp(sum(map(math.log, ratios)) / len(ratios)), 2) if ratios else None
    print(json.dumps(dict(schema='dope.optional-landmarks-latency/v1', run=a.run, gate_cuda_cold=gate,
                          cells=sorted(rows, key=lambda r: (r['device'], r['id']))), indent=1))


if __name__ == '__main__':
    main()
