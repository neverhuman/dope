"""Q3 retention arms for optional landmarks (spec 5.4) on non-training catalog tables.

Roster: catalog entries whose v3 role is dev_val_ablation or an eval_gradient_disjoint
panel (never MLC-v3 training), numeric-only, p <= 64, with local train/test blobs.
Native evidence per table comes from the integrated `dope landmarks real` (work_budget
2e12 = MLC-v3 training options, plus the 2e9 product default and --subsample 300).
Arms (level High, the Encoder629 tier):
  zero        landmark_mode none
  macro       progressive, bundle = native full report restricted to MACRO cells
  larger      progressive, MACRO+LARGER cells
  full        legacy full (unchanged historical path) on the 2e12 report
  full_2e9    legacy full on the 2e9 (product default) report
  sub300      sampled full on the --subsample 300 report
Per arm: vec256 / Encoder629 cosine to `full`, then wire -> generate_synthetic (2 seeds)
-> 6-auditor sklearn probe retention R (v3_ledger_replay recipe) on the real test split.
Restricting a full report to MACRO/LARGER cells equals a native `--ids` run (cells are
independent given folds); `--verify-ids K` checks that on K tables.
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = "1"  # one thread per worker process; parallelism is across tables

import argparse, concurrent.futures as cf, copy, csv, hashlib, json, random, subprocess, sys, time

import numpy as np

MACRO = (0, 1, 2, 3, 4, 5, 13, 14, 21)
LARGER = (6, 7, 8, 9, 10, 11, 12, 15, 16)
OPTS = ['--folds', '5', '--knn-reference', '2048', '--ridge-rows', '8192', '--seed', '0', '--small-rows', '4096',
        '--spectrum-features', '64']
ROLES = ('dev_val_ablation',)
SIZES = (100, 200, 300, 500, 1000)
TASK = {'regression': 'regression', 'binary': 'classification', 'multi_label': 'multiclass'}


def restrict(report, keep):
    r = copy.deepcopy(report)
    for i, c in enumerate(r['cells']):
        if i not in keep:
            c.update(state='not_requested', gain=None, se=None)
    r['spectrum'] = []
    return r


def native(binary, csv_path, target, task, extra, out):
    """Native report + emitted bundle; the bundle's source_digest binds the selected physical rows."""
    cmd = [binary, '--threads', '1', 'landmarks', 'real', csv_path, '--target', str(target), '--task',
           'regression' if task == 'regression' else 'classification', '--json', '-o', out,
           '--emit-bundle', out.replace('.json', '.bundle.json')] + OPTS + extra
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, RAYON_NUM_THREADS='1'))
    if p.returncode:
        raise RuntimeError('native: ' + p.stderr.decode(errors='replace').strip().splitlines()[-1][:160])
    return json.load(open(out)), time.perf_counter() - t


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else float('nan')


def run_table(job):
    repo, binary, e, workdir, seeds, verify = job
    sys.path.insert(0, repo)
    import torch
    torch.set_num_threads(1)
    from research.levels.multilevel_model import MultiLevelLandmarkModel, build_conditioning
    from research.levels.encoder import DopeEncoder629
    from research.levels.optional_landmarks import LandmarkBundle
    from research.levels.scaling import FitOnlyTableTransform
    from research.levels.budget_planner import serialize_dope_v3_envelope, deserialize_dope_v3_envelope
    from research.levels.v3_ledger_replay import (evaluate_reg_bin_dev_cell, evaluate_multiclass_auditors,
                                                  score_multiclass_syn_against_real)
    from safetensors.torch import load_file
    ckpt = '/home/ubuntu/dope/target/s3-multilevel-training-20261023/v3/dope_mlc_v3_multitask_best.safetensors'
    out = dict(dataset_id=e['dataset_id'], role=e['v3_role'], catalog_task=e['catalog_task'])
    try:
        task = TASK[e['catalog_task']]
        rows = lambda p, cap: list(csv.reader(open(p, newline='')))[:cap]
        tr, te = rows(e['train_path'], 4000), rows(e['test_path'], 2000)

        def floats(rs):
            try:
                return np.array([[float(v) for v in r] for r in rs], dtype=np.float64)
            except ValueError:
                return None
        ftr, fte = floats(tr), floats(te)
        numeric = ftr is not None and fte is not None and not (np.isnan(ftr).any() or np.isnan(fte).any())
        if numeric:  # all-numeric fast path (identical to the historical ndarray scaler fit)
            Xf, yf, Xt, yt = ftr[:, :-1], ftr[:, -1], fte[:, :-1], fte[:, -1]
        else:        # categorical/missing cells: string rows, scaler infers kinds
            Xf, Xt = [r[:-1] for r in tr], [r[:-1] for r in te]
            yf = np.array([float(r[-1]) for r in tr]); yt = np.array([float(r[-1]) for r in te])
        out.update(n_fit=len(Xf), n_test=len(Xt), p=len(Xf[0]), numeric_only=bool(numeric))
        d = os.path.join(workdir, e['dataset_id']); os.makedirs(d, exist_ok=True)
        fit_csv = os.path.join(d, 'fit.csv')
        with open(fit_csv, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow([f'c{j}' for j in range(len(tr[0]))]); w.writerows(tr)
        tgt = len(tr[0]) - 1
        out['native_s'], unavailable, scopes = {}, {}, {}

        def attempt(name, extra, fname):
            try:
                rep, t_n = native(binary, fit_csv, tgt, task, extra, os.path.join(d, fname))
                out['native_s'][name] = round(t_n, 3)
                b = json.load(open(os.path.join(d, fname.replace('.json', '.bundle.json'))))
                scopes[name] = dict(effective_rows=rep.get('rows'), subsample_n=b.get('subsample_n'),
                                    source_digest=b.get('source_digest'), status=b.get('status'))
                return rep
            except RuntimeError as ex:
                unavailable[name] = str(ex)
                return None
        rep_full = attempt('full', ['--work-budget', '2000000000000'], 'full.json')
        rep_2e9 = attempt('full_2e9', ['--work-budget', '2000000000'], 'full2e9.json')
        subs = {}
        for n_sub in SIZES:  # N >= rows selects every row; kept so effective N is recorded
            rep_sub = attempt(f'sub{n_sub}', ['--work-budget', '2000000000000', '--subsample', str(n_sub),
                                              '--subsample-seed', '0'], f'sub{n_sub}.json')
            if rep_sub is not None:
                subs[n_sub] = rep_sub
        if verify and rep_full is not None:
            rep_ids, _ = native(binary, fit_csv, tgt, task, ['--work-budget', '2000000000000', '--ids', 'macro'], os.path.join(d, 'macro_ids.json'))
            out['verify_macro_ids_equal'] = all(rep_ids['cells'][i].get('gain') == rep_full['cells'][i].get('gain') for i in MACRO)
        out['arm_scopes'], out['arm_unavailable'] = scopes, unavailable
        scaler = FitOnlyTableTransform.fit(Xf, yf, task=task)
        Xs, ys = scaler.transform(Xf, yf)
        Xts, yts = scaler.transform(Xt, yt)
        model = MultiLevelLandmarkModel().eval(); model.load_state_dict(load_file(ckpt), strict=True)
        enc629 = DopeEncoder629.from_pretrained(ckpt)
        mkb = lambda rp, n=None: LandmarkBundle.from_report(rp, n)
        arms = {'zero': dict(mode='none')}
        if rep_full is not None:
            arms.update(macro=dict(mode='progressive', bundle=mkb(restrict(rep_full, MACRO))),
                        larger=dict(mode='progressive', bundle=mkb(restrict(rep_full, MACRO + LARGER))),
                        full=dict(mode='full', report=rep_full))
        if rep_2e9 is not None:
            arms['full_2e9'] = dict(mode='full', report=rep_2e9)
        for n_sub, rep_sub in subs.items():
            arms[f'sub{n_sub}'] = dict(mode='full', report=rep_sub, subsample_n=n_sub, bundle=mkb(rep_sub, n_sub))
        if task == 'multiclass':
            k = int(len(np.unique(np.rint(ys))))
            real_info = evaluate_multiclass_auditors(Xs, ys, Xts, yts, k)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from scipy.stats import spearmanr

        def importances(X, y):
            # Feature-importance dynamics: impurity importances of a fixed RF on real vs synthetic rows.
            rf = (RandomForestRegressor if task == 'regression' else RandomForestClassifier)(
                n_estimators=32, max_depth=8, random_state=42, n_jobs=1)
            yy = y if task == 'regression' else (np.rint(y).astype(int) if task == 'multiclass' else (y >= 0.5).astype(int))
            return rf.fit(X, yy).feature_importances_

        fi_real = importances(Xs, ys)
        res = {}
        feats = {}
        for name, a in arms.items():
            kw = dict(task=task, bundle=a.get('bundle'), mode=a['mode'], native_report=a.get('report'), level='high',
                      subsample_n=a.get('subsample_n'))
            cond = build_conditioning(Xs, ys, **kw)
            with torch.no_grad():
                f = enc629(torch.tensor(cond['vec256'])[None], torch.tensor(cond['col_landmarks'])[None])[0].numpy()
            feats[name] = (cond['vec256'], f)
            if name == 'full':
                enc = model.encode_table(Xs, ys, rep_full, task=task, level='high', adaptive_budget=True, include_v3_primitives=True)
            else:
                enc = model.encode_table(Xs, ys, a.get('report'), task=task, level='high', adaptive_budget=True,
                                         include_v3_primitives=True, landmarks=a.get('bundle'), landmark_mode=a['mode'],
                                         subsample_n=a.get('subsample_n'))
            wire = serialize_dope_v3_envelope(enc, scaler=scaler, compress=True)
            dec, _ = deserialize_dope_v3_envelope(wire)
            rs, fis = [], []
            for s in seeds:
                Xsyn, ysyn = model.generate_synthetic(dec, n_rows=len(Xs), seed=s)
                if task == 'multiclass':
                    ev = score_multiclass_syn_against_real(real_info, Xsyn, ysyn, Xts, yts, k)
                    probe = ev['sklearn_6aud_probe']
                else:
                    ev = evaluate_reg_bin_dev_cell(Xs, ys, Xsyn, ysyn, Xts, yts, task)
                    probe = ev['sklearn_6aud_probe']
                rs.append(probe)
                try:
                    rho = spearmanr(fi_real, importances(Xsyn, ysyn)).correlation if Xs.shape[1] > 1 else float('nan')
                except ValueError:
                    rho = float('nan')
                fis.append(None if rho != rho else float(rho))
            res[name] = dict(retention=rs, fi_spearman=fis, wire_bytes=len(wire), recipe=cond['recipe'], digest=cond['digest'],
                             status=(a['bundle'].status if a.get('bundle') is not None else None))
        for name in res:
            if 'full' not in feats:
                continue
            res[name]['vec256_cos_full'] = cos(feats[name][0], feats['full'][0])
            res[name]['enc629_cos_full'] = cos(feats[name][1], feats['full'][1])
            res[name]['enc629_maxabs_full'] = float(np.max(np.abs(feats[name][1] - feats['full'][1])))
        out['arms'] = res
        out['cells_full_2e12'] = None if rep_full is None else sum(1 for c in rep_full['cells'] if c.get('gain') is not None)
        out['cells_full_2e9'] = None if rep_2e9 is None else sum(1 for c in rep_2e9['cells'] if c.get('gain') is not None)
    except Exception as ex:  # recorded, never silently dropped
        out['error'] = f'{type(ex).__name__}: {str(ex)[:200]}'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True); ap.add_argument('--bin', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--tables', type=int, default=112)
    ap.add_argument('--jobs', type=int, default=48); ap.add_argument('--verify-ids', type=int, default=6)
    ap.add_argument('--seeds', default='20261024,20261025')
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    from pathlib import Path
    from research.levels.v3_ablation_and_eval import build_reconciled_v3_manifest, find_local_blob
    os.makedirs(a.out, exist_ok=True)
    manifest = build_reconciled_v3_manifest(Path(a.out) / 'manifest')
    cat = {json.loads(l)['id']: json.loads(l) for l in open('/home/ubuntu/dope-fit-cv-precalc-20261023/catalog.jsonl')}
    pool, excluded = [], []
    for e in manifest['entries']:
        c = cat.get(e['dataset_id'])
        if e['v3_role'] not in ROLES or c is None:
            continue
        trp, tep = find_local_blob(c['files']['train']['sha256']), find_local_blob(c['files']['test']['sha256'])
        if trp and tep:
            pool.append(dict(e, train_path=str(trp), test_path=str(tep)))
        else:
            excluded.append(dict(dataset_id=e['dataset_id'], reason='missing_local_blob'))
    rng = random.Random(20261027)
    rng.shuffle(pool)
    picked = sorted(pool[: a.tables], key=lambda p: p['dataset_id'])
    seeds = [int(s) for s in a.seeds.split(',')]
    jobs = [(a.repo, a.bin, e, os.path.join(a.out, 'work'), seeds, i < a.verify_ids) for i, e in enumerate(picked)]
    meta = dict(pool=len(pool), picked=len(picked), excluded=excluded, roles=ROLES, seeds=seeds, bin=a.bin,
                bin_sha256=hashlib.sha256(open(a.bin, 'rb').read()).hexdigest(), started=time.time(), loadavg=os.getloadavg())
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    with cf.ProcessPoolExecutor(a.jobs) as ex, open(os.path.join(a.out, 'results.jsonl'), 'a') as fh:
        for r in ex.map(run_table, jobs):
            fh.write(json.dumps(r, default=float) + '\n'); fh.flush()
            print(r['dataset_id'], r.get('error', 'ok'), flush=True)


if __name__ == '__main__':
    main()
