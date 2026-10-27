"""Aggregate q3_retention.py results per the shared protocol (Codex plan lines 139-158).

Per table: mean over the primary paired seeds of six-auditor clipped R (and unclipped R),
RF feature-importance Spearman, Encoder629 cosine to full (auxiliary only). Across tables:
equal weight on the valid roster common to (arm, full). Primary result = ratio of mean R
(arm / full) with a paired bootstrap 95% CI over tables, plus median, per-table regressions
and absolute deltas. Every progressive transition is paired too.

Invalid measurements are excluded, never scored as zero. A table wider than the generator
(p > 64) is protocol_unavailable: its synthetic rows cannot be scored on the full-width real
test split. Runs that predate the harness flag get the same rule from the recorded p. An arm
whose per-auditor failure map shows an exception on a primary seed is unavailable for that
table; a single-class synthetic draw is a measured outcome and stays.

--against REF pairs this run with another checkpoint's run on the same roster and seeds:
per table, each arm against REF's same arm and against REF's full arm, with the exclusions
above applied to both runs.
"""
import argparse, hashlib, json
import numpy as np

ARMS = ['zero', 'macro', 'larger', 'random_sparse', 'random_dense', 'full', 'full_2e9', 'sub100', 'sub200', 'sub300', 'sub500', 'sub1000']
TRANSITIONS = [('zero', 'macro'), ('macro', 'larger'), ('larger', 'full')]
GENERATOR_WIDTH = 64
SUB_ARM_PARTITION = ('sub arms pair sampled native gains with building blocks computed on all fit rows in the Python '
                     'evaluator; the served Rust path computes building blocks on the native selection, so these arms '
                     'do not certify served sampled conditioning')
MEASURED_FAILURES = {'single_class_synthetic'}


def protocol_reason(r):
    if 'protocol_unavailable' in r:
        return r['protocol_unavailable']
    if r.get('p', 0) > GENERATOR_WIDTH:
        return f"p={r['p']} exceeds the generator width {GENERATOR_WIDTH}"
    return None


def truly_subsampled(r, arm):
    """The subsample arm selected fewer rows than the full arm (N >= rows selects every row)."""
    sc = r.get('arm_scopes', {})
    sub, full = sc.get(arm, {}).get('effective_rows'), sc.get('full', {}).get('effective_rows')
    return sub is not None and full is not None and sub < full


def failed_auditors(v, seeds_idx):
    maps = v.get('auditor_failures', [])
    return sorted({f'{k}:{f}' for i in seeds_idx if i < len(maps) for k, f in maps[i].items() if f not in MEASURED_FAILURES})


def per_table(r, seeds_idx, key):
    out = {}
    for arm, v in r.get('arms', {}).items():
        if failed_auditors(v, seeds_idx):
            continue
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
    ap.add_argument('--against', help='reference run of another checkpoint (same roster and seeds)')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f'{a.run}/results.jsonl')]
    meta = json.load(open(f'{a.run}/meta.json'))
    ids = sorted(r['dataset_id'] for r in rows)
    roster_sha = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    seeds_idx = [int(s) for s in a.seeds.split(',')]
    valid = [r for r in rows if 'arms' in r and protocol_reason(r) is None]
    tabs = [per_table(r, seeds_idx, a.key) for r in valid]
    by_task = {}
    for task in sorted({r['catalog_task'] for r in rows}):
        t_tabs = [per_table(r, seeds_idx, a.key) for r in valid if r['catalog_task'] == task]
        by_task[task] = {arm: compare(t_tabs, arm, 'full') for arm in ARMS if arm != 'full'}
    truly = {arm: compare([per_table(r, seeds_idx, a.key) for r in valid if truly_subsampled(r, arm)], arm, 'full')
             for arm in ARMS if arm.startswith('sub')}
    avail = {arm: sum(1 for t in tabs if arm in t) for arm in ARMS}
    errors = [dict(dataset_id=r['dataset_id'], error=r['error']) for r in rows if 'error' in r]
    unavailable = [dict(dataset_id=r['dataset_id'], p=r.get('p'), reason=protocol_reason(r)) for r in rows
                   if 'error' not in r and protocol_reason(r) is not None]
    failed = [dict(dataset_id=r['dataset_id'], arm=arm, auditors=failed_auditors(v, seeds_idx)) for r in valid
              for arm, v in r['arms'].items() if failed_auditors(v, seeds_idx)]
    report = dict(schema='dope.optional-landmarks-retention/v2', run=a.run, roster_size=len(rows),
                  roster_sha256=roster_sha, primary_seeds=[meta['seeds'][i] for i in seeds_idx], metric=a.key,
                  source_bin_sha256=meta.get('bin_sha256'), arm_availability=avail, table_errors=errors,
                  protocol_unavailable=unavailable, auditor_failure_arms=failed,
                  vs_full={arm: compare(tabs, arm, 'full') for arm in ARMS if arm != 'full'},
                  fi_vs_full={arm: compare(tabs, arm, 'full', 'FI') for arm in ARMS if arm != 'full'},
                  transitions={f'{x}->{y}': compare(tabs, y, x) for x, y in TRANSITIONS},
                  vs_full_by_task=by_task, vs_full_truly_subsampled=truly, sub_arm_partition=SUB_ARM_PARTITION,
                  zero_vs_full_where_full_positive=compare([t for t in tabs if t.get('full', {}).get('R') is not None
                                                            and t['full']['R'] > 0], 'zero', 'full'),
                  enc629_cos_to_full={arm: med([t[arm]['enc629_cos'] for t in tabs if arm in t]) for arm in ARMS})
    report['ckpt_sha256'] = meta.get('ckpt_sha256')
    if a.against:
        report['vs_reference'] = against(a, valid, meta, seeds_idx)
    print(json.dumps(report, indent=1))


def against(a, valid, meta, seeds_idx):
    ref_rows = {r['dataset_id']: r for r in map(json.loads, open(f'{a.against}/results.jsonl'))}
    ref_meta = json.load(open(f'{a.against}/meta.json'))
    if ref_meta['seeds'] != meta['seeds']:
        raise SystemExit('reference run used different seeds')
    both = [r for r in valid if 'arms' in ref_rows.get(r['dataset_id'], {}) and protocol_reason(ref_rows[r['dataset_id']]) is None]

    def paired(sel, arm, ref_arm):
        tabs = []
        for r in sel:
            c, b = per_table(r, seeds_idx, a.key), per_table(ref_rows[r['dataset_id']], seeds_idx, a.key)
            if arm in c and ref_arm in b:
                tabs.append(dict(cand=c[arm], ref=b[ref_arm]))
        return compare(tabs, 'cand', 'ref')
    tasks = sorted({r['catalog_task'] for r in both})
    return dict(run=a.against, ckpt_sha256=ref_meta.get('ckpt_sha256'), tables=len(both),
                same_arm={arm: paired(both, arm, arm) for arm in ARMS},
                vs_reference_full={arm: paired(both, arm, 'full') for arm in ARMS},
                same_arm_by_task={t: {arm: paired([r for r in both if r['catalog_task'] == t], arm, arm) for arm in ARMS}
                                  for t in tasks})


if __name__ == '__main__':
    main()
