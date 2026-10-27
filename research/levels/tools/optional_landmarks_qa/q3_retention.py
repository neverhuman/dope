"""Q3 retention arms for optional landmarks (spec 5.4) on non-training catalog tables.

Roster: catalog entries of the pinned role manifest (--role-manifest, SHA-256 checked) whose
v3 role is dev_val_ablation (never MLC-v3 training), with local train/test blobs; tables with
categorical or missing cells use the string-row scaler. Tables wider than the generator
(p > GENERATOR_WIDTH = 64 columns) stay in the roster but are recorded as
protocol_unavailable: synthetic rows have 64 columns and cannot be scored on the full-width
real test split.
Native evidence per table comes from the integrated `dope landmarks real` (work_budget
2e12 = MLC-v3 training options, plus the 2e9 product default and --subsample 300).
Arms (level High, the Encoder629 tier):
  zero        landmark_mode none
  macro       progressive, bundle = native full report restricted to MACRO cells
  larger      progressive, MACRO+LARGER cells
  full        unchanged historical full-report conditioning on the 2e12 report
  full_2e9    the same full-report conditioning on the 2e9 (product default) report
  sub300      sampled full on the --subsample 300 report
Per arm: vec256 / Encoder629 cosine to `full`, then wire -> generate_synthetic (2 seeds)
-> 6-auditor sklearn probe retention R (v3_ledger_replay recipe) on the real test split.
Each seed's per-auditor failure map is kept, so a score that fell back to chance is visible.
Every arm except the unsampled full report gets the served Rust inputs: raw-target moments for
regression (y_train_raw), and for sub-N arms the native stratified selection as evidence_positions,
cross-checked against the target_stats the native report computed on the rows it measured.
The zero arm's conditioning digest is compared with the served `dope mlc-v3 encode --landmark-mode none`
on the same fit.csv; a table whose evaluator preparation differs from the served one is recorded.
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
GENERATOR_WIDTH = 64  # MultiLevelLandmarkModel.max_cols: columns emitted by generate_synthetic
RELEASED_CKPT = '/home/ubuntu/dope/target/s3-multilevel-training-20261023/v3/dope_mlc_v3_multitask_best.safetensors'
TASK = {'regression': 'regression', 'binary': 'classification', 'multi_label': 'multiclass'}
# Per-table failures the evaluation records explicitly instead of aborting the roster: native CLI
# refusals (RuntimeError), unreadable or malformed splits (OSError, ValueError, IndexError, KeyError)
# and singular numerics (ArithmeticError, numpy.linalg.LinAlgError is a ValueError subclass).
TABLE_ERRORS = (RuntimeError, OSError, ValueError, IndexError, KeyError, ArithmeticError)


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


def served_none_digest(binary, fit_csv, target, task):
    """Effective-conditioning digest of the served none-mode encoding (weight-independent)."""
    cmd = [binary, 'mlc-v3', 'encode', '--weights', RELEASED_CKPT, '--csv', fit_csv, '--target', target,
           '--task', {'regression': 'regression', 'classification': 'classification'}.get(task, 'multiclass'),
           '--landmark-mode', 'none', '--json', '--threads', '1']
    p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, RAYON_NUM_THREADS='1'))
    recs = [json.loads(l) for l in p.stdout.decode(errors='replace').splitlines() if l.startswith('{')]
    if p.returncode or not recs:
        raise RuntimeError('served encode: ' + p.stderr.decode(errors='replace').strip()[-160:])
    return recs[-1]['conditioning_digest']


def selection_reproduced(y, pos, task, report, table_target_stats):
    """The native report measured exactly len(pos) rows; for regression its raw-target moments on those
    rows also agree (to summation-order rounding) with the moments of y[pos]."""
    if report.get('rows') != len(pos):
        return False
    if task != 'regression':
        return True
    nat, py = report.get('target_stats') or [None, None], table_target_stats(y[pos], task)
    return all((a is None) == (b is None) and (a is None or abs(a - b) <= 1e-9 * max(1.0, abs(a))) for a, b in zip(nat[:2], py[:2]))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else float('nan')


def run_table(job):
    repo, binary, e, workdir, seeds, verify, ckpt, ckpt_sha = job
    sys.path.insert(0, repo)
    import torch
    torch.set_num_threads(1)
    from research.levels.multilevel_model import MultiLevelLandmarkModel, build_conditioning
    from research.levels.encoder import DopeEncoder629
    from research.levels.optional_landmarks import LandmarkBundle, table_target_stats
    from research.levels.observed_conditioning import stratified_subsample
    from research.levels.scaling import FitOnlyTableTransform
    from research.levels.budget_planner import serialize_dope_v3_envelope, deserialize_dope_v3_envelope
    from research.levels.v3_ledger_replay import (evaluate_reg_bin_dev_cell, evaluate_multiclass_auditors,
                                                  score_multiclass_syn_against_real)
    from safetensors.torch import load_file
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
        if out['p'] > GENERATOR_WIDTH:
            out['protocol_unavailable'] = f"p={out['p']} exceeds the generator width {GENERATOR_WIDTH}"
            return out
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
        enc629 = DopeEncoder629.from_pretrained(ckpt, expected_sha256=ckpt_sha)
        mkb = lambda rp, n=None: LandmarkBundle.from_report(rp, n)
        arms = {'zero': dict(mode='none')}
        if rep_full is not None:
            arms.update(macro=dict(mode='progressive', bundle=mkb(restrict(rep_full, MACRO))),
                        larger=dict(mode='progressive', bundle=mkb(restrict(rep_full, MACRO + LARGER))),
                        full=dict(mode='full', report=rep_full))
            # Random partial views of the observed High cells (keep prob .25 / .75), seeded per table
            # so the pretraining baseline and a mask-trained candidate see the same masks.
            observed = [i for i in MACRO + LARGER if rep_full['cells'][i].get('gain') is not None]
            seed = int(hashlib.sha256(e['dataset_id'].encode()).hexdigest()[:8], 16)
            for name, keep in (('random_sparse', 0.25), ('random_dense', 0.75)):
                rng = np.random.default_rng([seed, int(keep * 100)])
                kept = tuple(i for i in observed if rng.random() < keep)
                arms[name] = dict(mode='progressive', bundle=mkb(restrict(rep_full, kept)), kept=len(kept))
        if rep_2e9 is not None:
            arms['full_2e9'] = dict(mode='full', report=rep_2e9)
        for n_sub, rep_sub in subs.items():
            pos = stratified_subsample(yf, task, n_sub, seed=0)
            if not selection_reproduced(yf, pos, task, rep_sub, table_target_stats):
                unavailable[f'sub{n_sub}'] = 'native selection not reproduced'
                continue
            arms[f'sub{n_sub}'] = dict(mode='full', report=rep_sub, subsample_n=n_sub, bundle=mkb(rep_sub, n_sub),
                                       evidence_positions=[int(i) for i in pos])
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
            observed_or_sampled = a['mode'] != 'full' or a.get('evidence_positions') is not None
            served = dict(y_train_raw=yf if task == 'regression' and observed_or_sampled else None,
                          evidence_positions=a.get('evidence_positions'))
            kw = dict(task=task, bundle=a.get('bundle'), mode=a['mode'], native_report=a.get('report'), level='high',
                      subsample_n=a.get('subsample_n'), **served)
            cond = build_conditioning(Xs, ys, **kw)
            with torch.no_grad():
                f = enc629(torch.tensor(cond['vec256'])[None], torch.tensor(cond['col_landmarks'])[None])[0].numpy()
            feats[name] = (cond['vec256'], f)
            if name == 'full':
                enc = model.encode_table(Xs, ys, rep_full, task=task, level='high', adaptive_budget=True, include_v3_primitives=True)
            else:
                enc = model.encode_table(Xs, ys, a.get('report'), task=task, level='high', adaptive_budget=True,
                                         include_v3_primitives=True, landmarks=a.get('bundle'), landmark_mode=a['mode'],
                                         subsample_n=a.get('subsample_n'), **served)
            wire = serialize_dope_v3_envelope(enc, scaler=scaler, compress=True)
            dec, _ = deserialize_dope_v3_envelope(wire)
            rs, fis, fails = [], [], []
            for s in seeds:
                Xsyn, ysyn = model.generate_synthetic(dec, n_rows=len(Xs), seed=s)
                if task == 'multiclass':
                    ev = score_multiclass_syn_against_real(real_info, Xsyn, ysyn, Xts, yts, k)
                    probe = ev['sklearn_6aud_probe']
                else:
                    ev = evaluate_reg_bin_dev_cell(Xs, ys, Xsyn, ysyn, Xts, yts, task)
                    probe = ev['sklearn_6aud_probe']
                rs.append(probe)
                fails.append(ev['failures'])
                try:
                    rho = spearmanr(fi_real, importances(Xsyn, ysyn)).correlation if Xs.shape[1] > 1 else float('nan')
                except ValueError:
                    rho = float('nan')
                fis.append(None if rho != rho else float(rho))
            res[name] = dict(retention=rs, fi_spearman=fis, auditor_failures=fails, kept_cells=a.get('kept'), wire_bytes=len(wire), recipe=cond['recipe'], digest=cond['digest'],
                             status=(a['bundle'].status if a.get('bundle') is not None else None))
        for name in res:
            if 'full' not in feats:
                continue
            res[name]['vec256_cos_full'] = cos(feats[name][0], feats['full'][0])
            res[name]['enc629_cos_full'] = cos(feats[name][1], feats['full'][1])
            res[name]['enc629_maxabs_full'] = float(np.max(np.abs(feats[name][1] - feats['full'][1])))
        out['arms'] = res
        out['served_zero_digest_equal'] = served_none_digest(binary, fit_csv, f'c{tgt}', task) == res['zero']['digest']
        out['cells_full_2e12'] = None if rep_full is None else sum(1 for c in rep_full['cells'] if c.get('gain') is not None)
        out['cells_full_2e9'] = None if rep_2e9 is None else sum(1 for c in rep_2e9['cells'] if c.get('gain') is not None)
    except TABLE_ERRORS as ex:  # a table that cannot be evaluated is recorded with its error type
        out['error'] = f'{type(ex).__name__}: {str(ex)[:200]}'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True); ap.add_argument('--bin', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--tables', type=int, default=112)
    ap.add_argument('--jobs', type=int, default=48); ap.add_argument('--verify-ids', type=int, default=6)
    ap.add_argument('--seeds', default='20261024,20261025')
    ap.add_argument('--ckpt', default='/home/ubuntu/dope/target/s3-multilevel-training-20261023/v3/dope_mlc_v3_multitask_best.safetensors')
    ap.add_argument('--ckpt-sha', default='d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c',
                    help='SHA-256 the checkpoint must match (the released pin by default)')
    ap.add_argument('--role-manifest', required=True,
                    help='pinned v3_corpus_exposure_and_split_manifest.json; read directly, so no mixed-role cache is opened')
    ap.add_argument('--role-manifest-sha', required=True, help='SHA-256 the role manifest must match')
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    from research.levels.v3_ablation_and_eval import find_local_blob
    os.makedirs(a.out, exist_ok=True)
    manifest_bytes = open(a.role_manifest, 'rb').read()
    if hashlib.sha256(manifest_bytes).hexdigest() != a.role_manifest_sha:
        raise SystemExit('role manifest SHA-256 mismatch')
    manifest = json.loads(manifest_bytes)
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
    if hashlib.sha256(open(a.ckpt, 'rb').read()).hexdigest() != a.ckpt_sha:
        raise SystemExit('checkpoint SHA-256 mismatch')
    jobs = [(a.repo, a.bin, e, os.path.join(a.out, 'work'), seeds, i < a.verify_ids, a.ckpt, a.ckpt_sha)
            for i, e in enumerate(picked)]
    meta = dict(pool=len(pool), picked=len(picked), excluded=excluded, role_manifest=a.role_manifest,
                role_manifest_sha256=a.role_manifest_sha, argv=sys.argv,
                harness_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(), ckpt=a.ckpt, ckpt_sha256=a.ckpt_sha, roles=ROLES, seeds=seeds, bin=a.bin,
                bin_sha256=hashlib.sha256(open(a.bin, 'rb').read()).hexdigest(), started=time.time(), loadavg=os.getloadavg())
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    with cf.ProcessPoolExecutor(a.jobs) as ex, open(os.path.join(a.out, 'results.jsonl'), 'a') as fh:
        for r in ex.map(run_table, jobs):
            fh.write(json.dumps(r, default=float) + '\n'); fh.flush()
            print(r['dataset_id'], r.get('error', 'ok'), flush=True)


if __name__ == '__main__':
    main()
