"""Collect QA runs into committed evidence for the Dope optional-landmarks PR.

Writes <repo>/docs/evidence/optional-landmarks/{retention-dev112,subsample-ablation,identity-sweeps}.json
from the run directories given on the command line. Numbers are copied from run outputs;
nothing is estimated here. The Markdown page cites these files.
"""
import argparse, hashlib, json, os, subprocess, sys


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--q1', required=True, help='q3_retention run dir')
    ap.add_argument('--ablation', required=True, help='run_ablation dir (seed 0)')
    ap.add_argument('--ablation-seed1', required=True)
    ap.add_argument('--identity', required=True, help='json list of {name, ref_dir, cand_dir}')
    a = ap.parse_args()
    out = os.path.join(a.repo, 'docs', 'evidence', 'optional-landmarks')
    os.makedirs(out, exist_ok=True)
    tools = os.path.dirname(os.path.abspath(__file__))
    q1 = json.loads(subprocess.check_output([sys.executable, os.path.join(tools, 'summarize_q1.py'), a.q1, '--seeds', '0,1']))
    q1_all = json.loads(subprocess.check_output([sys.executable, os.path.join(tools, 'summarize_q1.py'), a.q1, '--seeds', '0,1,2,3']))
    q1['supplement_all_four_seeds'] = {k: q1_all[k] for k in ('vs_full', 'transitions')}
    q1['harness_sha256'] = sha(os.path.join(tools, 'q3_retention.py'))
    json.dump(q1, open(os.path.join(out, 'retention-dev112.json'), 'w'), indent=1, sort_keys=True)
    abl = json.load(open(os.path.join(a.ablation, 'summary.json')))
    abl['seed_noise_floor_run'] = a.ablation_seed1
    abl['harness_sha256'] = sha(os.path.join(tools, 'run_ablation.py'))
    json.dump(abl, open(os.path.join(out, 'subsample-ablation.json'), 'w'), indent=1, sort_keys=True)
    ident = []
    for spec in json.load(open(a.identity)):
        res = subprocess.run([sys.executable, os.path.join(tools, 'compare.py'), spec['ref_dir'], spec['cand_dir']],
                             capture_output=True, text=True)
        ident.append(dict(spec, compare_stdout=res.stdout.strip(), identical=res.returncode == 0,
                          cand_bin_sha256=json.load(open(os.path.join(spec['cand_dir'], 'meta.json')))['bin_sha256']))
    json.dump(ident, open(os.path.join(out, 'identity-sweeps.json'), 'w'), indent=1, sort_keys=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
