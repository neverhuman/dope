"""Compare report sha256 between a reference run (threads=1) and a candidate run at every thread count."""
import json, sys, collections
ref_dir, cand_dir = sys.argv[1], sys.argv[2]
ref = {}
for x in map(json.loads, open(ref_dir + '/results.jsonl')):
    if x['threads'] == 1 and x['exit'] == 0:
        ref[x['id']] = x
cand = [json.loads(l) for l in open(cand_dir + '/results.jsonl')]
stats = collections.Counter(); diffs = []
for x in cand:
    if x['exit'] or x['id'] not in ref:
        stats['skip_or_fail'] += 1; continue
    same = x['sha256'] == ref[x['id']]['sha256']
    stats[('same' if same else 'DIFF', x['threads'])] += 1
    if not same: diffs.append((x['id'], x['threads']))
print(dict(stats)); print('diffs:', diffs[:20])
sys.exit(1 if diffs else 0)
