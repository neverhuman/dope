"""H7: which landmark slots/blocks influence MLC-v3 High conditioning (legacy full builder)?

For real native reports (run_ablation full arm), perturb one cell at a time (gain +0.05,
or cell made absent) and the spectrum (removed), rebuild
`build_v3_conditioning_bundle(..., level='high')`, and record which vec256 blocks and
whether the encoder's 64x17 column input change. Uses the research/levels reference
code of the checkout given by --repo (read-only).
"""
import argparse, copy, csv, json, os, sys
import numpy as np

BLOCKS = {'vector116': (0, 116), 'table24': (116, 140), 'inter48': (140, 188), 'bb68': (188, 256)}


def load(path):
    rows = list(csv.reader(open(path, newline='')))[1:]
    return [r[:-1] for r in rows], [r[-1] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True); ap.add_argument('--ablation', required=True)
    ap.add_argument('--manifest', required=True); ap.add_argument('--tables', type=int, default=24)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    from research.levels.building_block_landmarks import build_v3_conditioning_bundle
    from research.levels.scaling import FitOnlyTableTransform
    rows = [json.loads(l) for l in open(a.manifest)]
    have = [r for r in rows if os.path.exists(os.path.join(a.ablation, 'reports', r['id'] + '.full.json'))]
    picked = []
    for task in ('regression', 'binary', 'multi_label'):
        picked += [r for r in have if r['catalog_task'] == task][: a.tables // 3]
    tally = {}
    for r in picked:
        X, y = load(r['csv'])
        task = {'regression': 'regression', 'binary': 'classification', 'multi_label': 'multiclass'}[r['catalog_task']]
        tf = FitOnlyTableTransform.fit(X, y, task=task)
        Xs, ys = tf.transform(X, y)
        rep = json.load(open(os.path.join(a.ablation, 'reports', r['id'] + '.full.json')))
        build = lambda rp: build_v3_conditioning_bundle(rp, Xs, ys, level='high')
        base = build(rep)
        v0, c0 = np.asarray(base['vec256']), np.asarray(base['col_landmarks24'])[:64, :17]

        def diff(rp):
            b = build(rp)
            v, c = np.asarray(b['vec256']), np.asarray(b['col_landmarks24'])[:64, :17]
            out = {k: bool(np.any(v[s:e].view(np.uint32) != v0[s:e].view(np.uint32))) for k, (s, e) in BLOCKS.items()}
            out['col17'] = bool(c.shape != c0.shape or np.any(c.view(np.uint32) != c0.view(np.uint32)))
            return out

        cases = {}
        for i, cell in enumerate(rep['cells']):
            if cell.get('gain') is None:
                continue
            bump = copy.deepcopy(rep); bump['cells'][i]['gain'] = cell['gain'] + 0.05
            drop = copy.deepcopy(rep); drop['cells'][i]['state'] = 'failed'; drop['cells'][i]['gain'] = None
            cases[f"slot{i:02d}:{cell.get('name')}:bump"] = diff(bump)
            cases[f"slot{i:02d}:{cell.get('name')}:drop"] = diff(drop)
        nospec = copy.deepcopy(rep); nospec['spectrum'] = []
        cases['spectrum:drop'] = diff(nospec)
        for k, d in cases.items():
            t = tally.setdefault(k, {b: 0 for b in list(BLOCKS) + ['col17']} | {'tables': 0})
            t['tables'] += 1
            for b, changed in d.items():
                t[b] += int(changed)
        print(r['id'], task, 'done', flush=True)
    json.dump(tally, open(a.out, 'w'), indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
