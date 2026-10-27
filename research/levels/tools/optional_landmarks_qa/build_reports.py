"""Collect QA runs into committed evidence for the Dope optional-landmarks PR.

Writes <repo>/docs/evidence/optional-landmarks/{retention-dev112,retention-candidate,subsample-ablation,
identity-sweeps}.json
from the run directories given on the command line; each file is written only when its inputs
are given. Numbers are copied from run outputs; nothing is estimated here. The Markdown page
cites these files.
"""
import argparse, hashlib, json, os, subprocess, sys


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--q1', help='q3_retention run dir')
    ap.add_argument('--q1-harness-sha', help='SHA-256 of the q3_retention.py that produced --q1 '
                    '(default: the run meta, else the current file)')
    ap.add_argument('--candidate', help='q3_retention run dir of a candidate checkpoint, paired against --q1')
    ap.add_argument('--candidate-harness-sha', help='SHA-256 of the q3_retention.py that produced --candidate')
    ap.add_argument('--ablation', help='run_ablation dir (seed 0)')
    ap.add_argument('--ablation-seed1')
    ap.add_argument('--identity', help='json list of {name, ref_dir, cand_dir}')
    a = ap.parse_args()
    out = os.path.join(a.repo, 'docs', 'evidence', 'optional-landmarks')
    os.makedirs(out, exist_ok=True)
    tools = os.path.dirname(os.path.abspath(__file__))
    if a.q1:
        write_retention(a, tools, out)
    if a.candidate:
        write_candidate(a, tools, out)
    if a.ablation:
        abl = json.load(open(os.path.join(a.ablation, 'summary.json')))
        abl['seed_noise_floor_run'] = a.ablation_seed1
        abl['harness_sha256'] = sha(os.path.join(tools, 'run_ablation.py'))
        json.dump(abl, open(os.path.join(out, 'subsample-ablation.json'), 'w'), indent=1, sort_keys=True)
    if a.identity:
        write_identity(a, tools, out)
    print('wrote', out)


def write_retention(a, tools, out):
    summarize = os.path.join(tools, 'summarize_q1.py')
    q1 = json.loads(subprocess.check_output([sys.executable, summarize, a.q1, '--seeds', '0,1']))
    q1_all = json.loads(subprocess.check_output([sys.executable, summarize, a.q1, '--seeds', '0,1,2,3']))
    q1['supplement_all_four_seeds'] = {k: q1_all[k] for k in ('vs_full', 'transitions')}
    meta = json.load(open(os.path.join(a.q1, 'meta.json')))
    q1['harness_sha256'] = a.q1_harness_sha or meta.get('harness_sha256') or sha(os.path.join(tools, 'q3_retention.py'))
    q1['summarizer_sha256'] = sha(summarize)
    q1['role_manifest_sha256'] = meta.get('role_manifest_sha256')
    json.dump(q1, open(os.path.join(out, 'retention-dev112.json'), 'w'), indent=1, sort_keys=True)


def write_candidate(a, tools, out):
    summarize = os.path.join(tools, 'summarize_q1.py')
    run = lambda seeds: json.loads(subprocess.check_output([sys.executable, summarize, a.candidate, '--seeds', seeds,
                                                            '--against', a.q1]))
    cand, cand_all = run('0,1'), run('0,1,2,3')
    cand['supplement_all_four_seeds'] = {k: cand_all[k] for k in ('vs_full', 'vs_full_by_task', 'vs_reference')}
    meta = json.load(open(os.path.join(a.candidate, 'meta.json')))
    cand['harness_sha256'] = a.candidate_harness_sha or meta.get('harness_sha256') or sha(os.path.join(tools, 'q3_retention.py'))
    cand['summarizer_sha256'] = sha(summarize)
    json.dump(cand, open(os.path.join(out, 'retention-candidate.json'), 'w'), indent=1, sort_keys=True)


def write_identity(a, tools, out):
    ident = []
    for spec in json.load(open(a.identity)):
        res = subprocess.run([sys.executable, os.path.join(tools, 'compare.py'), spec['ref_dir'], spec['cand_dir']],
                             capture_output=True, text=True)
        ident.append(dict(spec, compare_stdout=res.stdout.strip(), identical=res.returncode == 0,
                          cand_bin_sha256=json.load(open(os.path.join(spec['cand_dir'], 'meta.json')))['bin_sha256']))
    json.dump(ident, open(os.path.join(out, 'identity-sweeps.json'), 'w'), indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
