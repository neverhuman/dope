"""Authenticated manifest for native training evidence produced by run_ablation.py.

For every (dataset, arm) it binds: the fit CSV sha256 (re-verified against the caller's
manifest), the native report and emitted LandmarkBundle sha256, the bundle's scope
`source_digest`, `subsample_n` and status, the report's effective `rows`, the CLI binary
sha256 and the native options. Unavailable arms keep their recorded refusal. The output
JSONL plus its own sha256 is what a trainer should pin instead of report filenames.
"""
import argparse, hashlib, json, os


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run_ablation output directory')
    ap.add_argument('--source-manifest', required=True, help="the trainer's fit manifest (fit_csv, fit_csv_sha256)")
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.run, 'meta.json')))
    bin_sha = sha(os.path.join(os.path.dirname(os.path.abspath(a.run)), meta['bin']) if not os.path.isabs(meta['bin']) else meta['bin'])
    src = {d['dataset_id']: d for d in map(json.loads, open(a.source_manifest))}
    results = [json.loads(l) for l in open(os.path.join(a.run, 'results.jsonl'))]
    rows = []
    for r in sorted(results, key=lambda r: (r['id'], r['arm'])):
        s = src[r['id']]
        entry = dict(dataset_id=r['id'], arm=r['arm'], requested_subsample=r['n_sub'], fit_csv_sha256=s['fit_csv_sha256'],
                     fit_csv_verified=sha(s['fit_csv']) == s['fit_csv_sha256'], native_exit=r['exit'])
        if r['exit'] == 0:
            rep = os.path.join(a.run, 'reports', f"{r['id']}.{r['arm']}.json")
            bun = os.path.join(a.run, 'bundles', f"{r['id']}.{r['arm']}.json")
            b = json.load(open(bun))
            entry.update(report_sha256=sha(rep), bundle_sha256=sha(bun), bundle_source_digest=b.get('source_digest'),
                         bundle_subsample_n=b.get('subsample_n'), bundle_status=b.get('status'),
                         effective_rows=json.load(open(rep)).get('rows'))
        else:
            entry['unavailable'] = r['stderr'].strip().splitlines()[-1][:200] if r['stderr'] else 'nonzero exit'
        rows.append(entry)
    with open(a.out, 'w') as fh:
        fh.write(json.dumps(dict(record='header', cli_sha256=bin_sha, native_options=meta['options'], seed=meta.get('seed', '0'),
                                 source_manifest_sha256=sha(a.source_manifest), entries=len(rows)), sort_keys=True) + '\n')
        for e in rows:
            fh.write(json.dumps(e, sort_keys=True) + '\n')
    print(a.out, sha(a.out), len(rows), 'unverified inputs:', sum(1 for e in rows if not e['fit_csv_verified']))


if __name__ == '__main__':
    main()
