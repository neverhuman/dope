"""Ablation corpus: catalog train splits with >=1000 rows (headed copies, target last).

All tables with >=1500 rows plus a seeded sample of 1000-1499-row tables per task.
Writes corpus/ablation.jsonl. Timeseries are excluded (no iid subsample semantics).
"""
import csv, hashlib, json, os, random

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
QUAL = os.environ.get('DOPE_QA_DIR', os.path.join(REPO, 'target', 'qual'))
ROOTS = ['/home/ubuntu/scratch/jopedime-corpus-benchmark-20260928/blobs',
         '/home/ubuntu/scratch/jopedime-corpus-throughput-20260928/blobs', '/home/ubuntu/out/loop-store/blobs']
TASK = {'regression': 'regression', 'binary': 'classification', 'multi_label': 'classification'}
SMALL_PER_TASK = {'regression': 120, 'binary': 80, 'multi_label': 40}


def find(sha):
    for r in ROOTS:
        for c in (f'{r}/{sha}', f'{r}/sha256/{sha[:2]}/{sha}'):
            if os.path.exists(c):
                return c


def main():
    rng = random.Random(20261027)
    big, small = [], {t: [] for t in TASK}
    for line in open('/home/ubuntu/dope-fit-cv-precalc-20261023/catalog.jsonl'):
        e = json.loads(line)
        tr = e['files'].get('train', {})
        if e['task'] not in TASK or tr.get('rows', 0) < 1000 or e['target'].get('index') != -1:
            continue
        src = find(tr.get('sha256', ''))
        if not src:
            continue
        row = dict(id='cat-' + e['id'], src=src, task=TASK[e['task']], catalog_task=e['task'],
                   n=tr['rows'], p=e['shape']['features'], header=bool(tr.get('has_header')))
        (big if tr['rows'] >= 1500 else small[e['task']]).append(row)
    rows = big
    for t, pool in small.items():
        rng.shuffle(pool)
        rows += pool[:SMALL_PER_TASK[t]]
    os.makedirs(os.path.join(QUAL, 'corpus', 'catalog'), exist_ok=True)
    with open(os.path.join(QUAL, 'corpus', 'ablation.jsonl'), 'w') as out:
        for r in rows:
            dst = os.path.join(QUAL, 'corpus', 'catalog', r['id'] + '.csv')
            with open(r['src'], newline='') as fi, open(dst, 'w', newline='') as fo:
                rd = csv.reader(fi)
                first = next(rd)
                wr = csv.writer(fo)
                wr.writerow(first if r['header'] else [f'c{j}' for j in range(len(first))])
                if not r['header']:
                    wr.writerow(first)
                wr.writerows(rd)
            r['csv'] = dst
            r['csv_sha256'] = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
            out.write(json.dumps(r, sort_keys=True) + '\n')
    print(len(rows))


if __name__ == '__main__':
    main()
