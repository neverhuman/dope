"""Build the landmark QA corpus manifest (JSONL) for identity sweeps and benchmarks.

Sources: precalc fit.csv cells (real, regression/classification, n<=1024),
JopeDime datasets/test train.csv (real regression), and deterministic synthetic
tables (regression/binary/multiclass; n in 1k/10k/50k; p in 10/64/200; with
categorical + missing columns). Target is always the last column.
"""
import csv, glob, hashlib, json, os, random, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
QUAL = os.environ.get('DOPE_QA_DIR', os.path.join(REPO, 'target', 'qual'))
PRE = os.path.expanduser('~/dope-fit-cv-precalc-20261023')
JOPE = os.path.expanduser('~/JopeDime/datasets/test')


def ncols(path):
    with open(path, newline='') as f:
        return len(next(csv.reader(f)))


def precalc(rng):
    real = []
    for root in ['full-v4/cells', 'owned-input-precalc-v8-0001/cells']:
        for d in sorted(glob.glob(os.path.join(PRE, root, '*'))):
            m, f = os.path.join(d, 'manifest.json'), os.path.join(d, 'fit.csv')
            if not (os.path.exists(m) and os.path.exists(f)):
                continue
            s = json.load(open(m)).get('scope', {})
            real.append(dict(id='pre-' + os.path.basename(d)[:16], csv=f, task=s['task'],
                             n=s['rows'], p=len(s['column_kinds']) - 1, source=root.split('/')[0]))
    clf = [r for r in real if r['task'] == 'classification']
    wide = [r for r in real if r['task'] == 'regression' and r['p'] > 64]
    rest = [r for r in real if r['task'] == 'regression' and r['p'] <= 64]
    rng.shuffle(wide); rng.shuffle(rest)
    return clf + wide[:40] + rest[:120]


def jope():
    """JopeDime test train.csv files are headerless; write headed copies (c0..cK)."""
    out = []
    os.makedirs(os.path.join(QUAL, 'corpus', 'jope'), exist_ok=True)
    for d in sorted(glob.glob(os.path.join(JOPE, '*'))):
        src = os.path.join(d, 'train.csv')
        if not os.path.exists(src):
            continue
        k = ncols(src)
        dst = os.path.join(QUAL, 'corpus', 'jope', os.path.basename(d) + '.csv')
        with open(src) as fi, open(dst, 'w') as fo:
            fo.write(','.join(f'c{j}' for j in range(k)) + '\n')
            n = 0
            for line in fi:
                fo.write(line); n += 1
        out.append(dict(id='jope-' + os.path.basename(d), csv=dst, task='regression', n=n,
                        p=k - 1, source='jopedime-test'))
    return out


def synth(task, n, p, seed):
    g = np.random.default_rng(seed)
    x = g.normal(size=(n, p))
    x[:, ::7] = np.exp(x[:, ::7])                      # skewed columns
    w = g.normal(size=p) / np.sqrt(p)
    f = x[:, :min(p, 8)] @ w[:min(p, 8)] + 0.5 * np.sin(2 * x[:, 0]) + 0.3 * x[:, 1] * x[:, 2 % p]
    cat = g.integers(0, 4, size=n)
    f = f + 0.4 * (cat == 2)
    miss = g.random(size=(n, p)) < 0.02
    if task == 'regression':
        y = f + 0.3 * g.normal(size=n)
    elif task == 'binary':
        y = (f + 0.3 * g.normal(size=n) > np.median(f)).astype(int)
    else:
        y = np.digitize(f + 0.3 * g.normal(size=n), np.quantile(f, [0.2, 0.4, 0.6, 0.8]))
    name = f'synth-{task}-n{n}-p{p}'
    path = os.path.join(QUAL, 'corpus', 'synth', name + '.csv')
    if not os.path.exists(path):
        with open(path, 'w', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow([f'x{j}' for j in range(p)] + ['cat', 'y'])
            for i in range(n):
                row = ['' if miss[i, j] else repr(float(x[i, j])) for j in range(p)]
                wr.writerow(row + ['abcd'[cat[i]], str(y[i]) if task != 'regression' else repr(float(y[i]))])
    return dict(id=name, csv=path, task='regression' if task == 'regression' else 'classification',
                n=n, p=p + 1, source='synthetic')


def main():
    rng = random.Random(20261027)
    rows = precalc(rng) + jope()
    for i, (task, n, p) in enumerate([(t, n, p) for t in ('regression', 'binary', 'multiclass')
                                      for n in (1000, 10000, 50000) for p in (10, 64, 200)]):
        rows.append(synth(task, n, p, 1000 + i))
    out = os.path.join(QUAL, 'corpus', 'manifest.jsonl')
    with open(out, 'w') as fh:
        for r in rows:
            r['csv_sha256'] = hashlib.sha256(open(r['csv'], 'rb').read()).hexdigest()
            fh.write(json.dumps(r, sort_keys=True) + '\n')
    print(out, len(rows))


if __name__ == '__main__':
    main()
