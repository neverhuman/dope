"""Summarize a run_ablation.py output: similarity of sub-N bundles to the full-row bundle.

Per table and N: mask agreement (23 slots and the 18 High-level slots), cosine,
Spearman and max |delta| of squashed values on jointly present High slots, top-1
(best present High landmark) agreement, macro/larger group-max deltas, and
wall-time ratio. These are landmark-vector proxies; downstream retention is Q3.
"""
import json, os, statistics, sys, collections
import numpy as np

MACRO = (0, 1, 2, 3, 4, 5, 13, 14, 21)
LARGER = (6, 7, 8, 9, 10, 11, 12, 15, 16)
HIGH = tuple(sorted(MACRO + LARGER))


def spearman(a, b):
    if len(a) < 3:
        return float('nan')
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def gmax(v, m, idx):
    xs = [v[i] for i in idx if m[i]]
    return max(xs) if xs else 0.0


def compare(full, sub):
    fv, fm, sv, sm = (np.array(full['values']), full['mask'], np.array(sub['values']), sub['mask'])
    both = [i for i in HIGH if fm[i] and sm[i]]
    a, b = fv[both], sv[both]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) if both and np.linalg.norm(a) and np.linalg.norm(b) else float('nan')
    top = lambda v, m: max((i for i in HIGH if m[i]), key=lambda i: (v[i], -i), default=None)
    return dict(mask23=sum(fm[i] == sm[i] for i in range(23)) / 23, mask18=sum(fm[i] == sm[i] for i in HIGH) / 18,
                cos=cos, spearman=spearman(a, b), maxabs=float(np.max(np.abs(a - b))) if both else float('nan'),
                top1=top(fv, fm) == top(sv, sm),
                d_macro=abs(gmax(fv, fm, MACRO) - gmax(sv, sm, MACRO)), d_larger=abs(gmax(fv, fm, LARGER) - gmax(sv, sm, LARGER)))


def q(xs, p):
    xs = sorted(x for x in xs if x == x)
    return round(xs[int(round((len(xs) - 1) * p))], 4) if xs else None


def main():
    out = sys.argv[1]
    res = [json.loads(l) for l in open(os.path.join(out, 'results.jsonl'))]
    ok = {(r['id'], r['arm']): r for r in res if r['exit'] == 0}
    per = collections.defaultdict(list)
    for (tid, arm), r in ok.items():
        if arm == 'full' or (tid, 'full') not in ok:
            continue
        full = json.load(open(os.path.join(out, 'bundles', f'{tid}.full.json')))
        sub = json.load(open(os.path.join(out, 'bundles', f'{tid}.{arm}.json')))
        m = compare(full, sub)
        m.update(id=tid, rows=r['rows'], task=r['task'], speed=ok[(tid, 'full')]['wall_s'] / max(r['wall_s'], 1e-6))
        per[r['n_sub']].append(m)
    summary = {}
    for n in sorted(per):
        ms = per[n]
        summary[n] = dict(tables=len(ms),
                          cos_p10=q([m['cos'] for m in ms], .1), cos_p50=q([m['cos'] for m in ms], .5),
                          frac_cos_ge_0_99=round(np.mean([m['cos'] >= .99 for m in ms if m['cos'] == m['cos']]), 4),
                          spearman_p50=q([m['spearman'] for m in ms], .5), spearman_p10=q([m['spearman'] for m in ms], .1),
                          maxabs_p50=q([m['maxabs'] for m in ms], .5), maxabs_p90=q([m['maxabs'] for m in ms], .9),
                          mask18_mean=round(np.mean([m['mask18'] for m in ms]), 4),
                          top1_agree=round(np.mean([m['top1'] for m in ms]), 4),
                          d_macro_p50=q([m['d_macro'] for m in ms], .5), d_larger_p50=q([m['d_larger'] for m in ms], .5),
                          speedup_p50=q([m['speed'] for m in ms], .5),
                          by_rows={band: round(float(np.median([m['cos'] for m in ms if lo <= m['rows'] < hi and m['cos'] == m['cos']] or [float('nan')])), 4)
                                   for band, lo, hi in (('1k-1.5k', 1000, 1500), ('1.5k-3k', 1500, 3000), ('3k-10k', 3000, 10000), ('10k+', 10000, 10**9))})
    meta = json.load(open(os.path.join(out, 'meta.json')))
    report = dict(schema='dope.subsample-ablation/v1', source=meta, failures=sum(1 for r in res if r['exit']), by_n=summary)
    with open(os.path.join(out, 'summary.json'), 'w') as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(out, 'per_table.jsonl'), 'w') as fh:
        for n, ms in sorted(per.items()):
            for m in ms:
                fh.write(json.dumps(dict(m, n_sub=n)) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
