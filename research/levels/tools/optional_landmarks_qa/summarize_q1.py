"""Aggregate q3_retention.py results per the shared protocol (Codex plan lines 139-158).

Per table: mean over the primary paired seeds of six-auditor clipped R (and unclipped R),
RF feature-importance Spearman, Encoder629 cosine to full (auxiliary only). Across tables:
equal weight on the valid roster common to (arm, full). Primary result = ratio of mean R
(arm / full) with a paired bootstrap 95% CI over tables, plus median, per-table regressions
and absolute deltas. Every progressive transition is paired too.
"""
import argparse, hashlib, json
import numpy as np

ARMS = ['zero', 'macro', 'larger', 'full', 'full_2e9', 'sub100', 'sub200', 'sub300', 'sub500', 'sub1000']
TRANSITIONS = [('zero', 'macro'), ('macro', 'larger'), ('larger', 'full')]


def per_table(r, seeds_idx, key):
    out = {}
    for arm, v in r.get('arms', {}).items():
        rs = [v['retention'][i][key] for i in seeds_idx if i < len(v['retention'])]
        fi = [v['fi_spearman'][i] for i in seeds_idx if i < len(v['fi_spearman']) and v['fi_spearman'][i] is not None]
        out[arm] = dict(R=float(np.mean(rs)) if rs else None, FI=float(np.mean(fi)) if fi else None,
                        enc629_cos=v.get('enc629_cos_full'))
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 4) if xs else None


def boot_ratio(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    num, den = a[idx].mean(1), b[idx].mean(1)
    ok = den > 0
    r = num[ok] / den[ok]
    return [round(float(np.quantile(r, .025)), 4), round(float(np.quantile(r, .975)), 4)] if ok.any() else None


def compare(tabs, arm, base, metric='R'):
    pairs = [(t[arm][metric], t[base][metric]) for t in tabs if arm in t and base in t
             and t[arm][metric] is not None and t[base][metric] is not None]
    if not pairs:
        return dict(valid=0)
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    d = a - b
    res = dict(valid=len(pairs), mean_arm=round(float(a.mean()), 4), mean_base=round(float(b.mean()), 4),
               median_arm=round(float(np.median(a)), 4), median_base=round(float(np.median(b)), 4),
               mean_delta=round(float(d.mean()), 4), median_delta=round(float(np.median(d)), 4),
               worse_by_gt_0_05=int((d < -0.05).sum()), better_by_gt_0_05=int((d > 0.05).sum()))
    if metric == 'R':
        res['ratio_of_means'] = round(float(a.mean() / b.mean()), 4) if b.mean() > 0 else None
        res['ratio_ci95'] = boot_ratio(a, b)
        res['nonpositive_base_tables'] = int((b <= 0).sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--seeds', default='0,1', help='indices of primary seeds in the run')
    ap.add_argument('--key', default='clipped_R')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f'{a.run}/results.jsonl')]
    meta = json.load(open(f'{a.run}/meta.json'))
    ids = sorted(r['dataset_id'] for r in rows)
    roster_sha = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    seeds_idx = [int(s) for s in a.seeds.split(',')]
    tabs = [per_table(r, seeds_idx, a.key) for r in rows if 'arms' in r]
    avail = {arm: sum(1 for t in tabs if arm in t) for arm in ARMS}
    errors = [dict(dataset_id=r['dataset_id'], error=r['error']) for r in rows if 'error' in r]
    report = dict(schema='dope.optional-landmarks-retention/v1', run=a.run, roster_size=len(rows),
                  roster_sha256=roster_sha, primary_seeds=[meta['seeds'][i] for i in seeds_idx], metric=a.key,
                  source_bin_sha256=meta.get('bin_sha256'), arm_availability=avail, table_errors=errors,
                  vs_full={arm: compare(tabs, arm, 'full') for arm in ARMS if arm != 'full'},
                  fi_vs_full={arm: compare(tabs, arm, 'full', 'FI') for arm in ARMS if arm != 'full'},
                  transitions={f'{x}->{y}': compare(tabs, y, x) for x, y in TRANSITIONS},
                  enc629_cos_to_full={arm: med([t[arm]['enc629_cos'] for t in tabs if arm in t]) for arm in ARMS})
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
