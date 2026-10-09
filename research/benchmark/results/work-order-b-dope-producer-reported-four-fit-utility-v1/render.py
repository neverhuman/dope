#!/usr/bin/env python3
"""Reproduce public descriptive artifacts from this committed result only."""
import argparse, collections, csv, hashlib, io, json, math, pathlib, re, statistics

ROOT = pathlib.Path(__file__).resolve().parent
PROOF_PIN = (5924, 'b424d85d8e3c2ba3c7e6168131282156762105e83237640b4772b7152ec9e703')
PUBLIC_FILES = {'result.json','schema.json','summary.csv','summary.svg','README.md','source-proof.schema.json'}
FIT_SEEDS, SAMPLE_SEEDS, SIZES, AUDITORS = (23,37,53,71), (101,211,307), (1,4), ('catboost','linear','mlp')
GROUP_FIELDS = ['auditor','size_multiplier','lineages','complete_four_fit_lineages','producer_reported_median_of_four_fit_median_retention','producer_reported_median_four_fit_sample_sd','informative_finite_ok_samples','not_informative_samples']

def require(ok):
    if not ok: raise ValueError('public description refused')
def pairs(items):
    result = {}
    for key,value in items: require(key not in result); result[key] = value
    return result
def decode(raw):
    def invalid(_): raise ValueError('public description refused')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
def digest(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def finite(value):
    return type(value) in (int,float) and math.isfinite(value)
def no_private_values(value):
    if type(value) is dict:
        require(not set(value).intersection({'path','argv','display_name','headers','source_values','raw_base64','weights','raw_errors'}))
        for item in value.values(): no_private_values(item)
    elif type(value) is list:
        for item in value: no_private_values(item)
    elif type(value) is str: require('/home/' not in value and '/mnt/' not in value and not value.startswith('/'))
    elif type(value) is float: require(math.isfinite(value))

def validate_fit(fit):
    require(type(fit['fit_seed']) is int and fit['fit_seed'] in FIT_SEEDS and len(fit['samples']) == 3)
    require([s['sample_seed'] for s in fit['samples']] == list(SAMPLE_SEEDS) and all(type(s['sample_seed']) is int for s in fit['samples']))
    values = []
    for sample in fit['samples']:
        if sample['status'] == 'ok': require(finite(sample['retention'])); values.append(sample['retention'])
        else: require(sample['status'] in ('not_informative','missing_auditor','auditor_failed','receipt_not_ok','invalid_utility') and sample['retention'] is None)
    require(type(fit['informative_finite_ok_count']) is int and fit['informative_finite_ok_count'] == len(values))
    require(fit['complete_three_sample_group'] is (len(values) == 3))
    if len(values) == 3:
        require(fit['producer_reported_retention_median'] == statistics.median(values) and fit['producer_reported_within_fit_retention_range'] == [min(values),max(values)])
    else: require(fit['producer_reported_retention_median'] is None and fit['producer_reported_within_fit_retention_range'] is None)

def validate(result):
    require(result['format'] == 'work-order-b-dope-producer-reported-four-fit-utility-v1')
    require(result['panel'] == {'lineages':100,'fit_receipts':400,'metric_receipts':2400,'fit_seeds':[23,37,53,71],'sample_seeds':[101,211,307],'size_multipliers':[1,4],'auditors':['catboost','linear','mlp']})
    require(result['interpretation']['producer_reported'] is True and result['interpretation']['post_hoc_descriptive_rule'] is True and result['interpretation']['preregistered_inferential_rule'] is False and result['interpretation']['retention_clipped'] is False)
    require(result['interpretation']['historical_evaluator_and_runtime_parity_verified'] is False and result['interpretation']['seed11_joined'] is False)
    require(all(value is None for value in result['null_claims'].values()))
    require(set(result['null_claims']) == {'historical_evaluator_source_sha256','historical_auditor_versions','verified_common_auditor_variation','five_fit_variance','five_fit_sample_sd','paired_comparison','MFS_v2','PTF_v1','quality_winner','privacy_certification','actual_training_backend','GPU_training_verified','production_L3_charged_inventory_bytes','retained_storage_bytes'})
    require(result['native_or_paper_count_additions'] == result['new_fits_or_evaluations'] == 0)
    cells = result['cells']; require(len(cells) == 600); seen = set(); lineages = set()
    for cell in cells:
        identity = (cell['dataset_lineage_id'],cell['size_multiplier'],cell['auditor'])
        require(re.fullmatch('[0-9a-f]{16}',identity[0]) is not None and type(identity[1]) is int and identity[1] in SIZES and identity[2] in AUDITORS and identity not in seen)
        seen.add(identity); lineages.add(identity[0]); require([fit['fit_seed'] for fit in cell['fits']] == list(FIT_SEEDS))
        for fit in cell['fits']: validate_fit(fit)
        medians = [fit['producer_reported_retention_median'] for fit in cell['fits'] if fit['complete_three_sample_group']]
        require(type(cell['complete_fit_count']) is int and cell['complete_fit_count'] == len(medians))
        if len(medians) == 4:
            require(finite(cell['producer_reported_four_fit_sample_sd']) and cell['producer_reported_four_fit_sample_sd'] == statistics.stdev(medians) and cell['status'] == 'complete_producer_reported_description')
        else: require(cell['producer_reported_four_fit_sample_sd'] is None and cell['status'] == 'incomplete_four_fit_group')
    require(len(lineages) == 100 and seen == {(lineage,size,auditor) for lineage in lineages for size in SIZES for auditor in AUDITORS})
    require(result['statistics']['sample_sd_denominator'] == 3 and result['statistics']['required_informative_samples_per_fit'] == 3 and result['statistics']['required_complete_fits_for_sd'] == 4)
    binding = result['seed11_metadata_delta']; require(binding['comparison_pairs'] == binding['generator_binary_hash_mismatches'] == 400 and binding['seed11_status_counts'] == {'ok':98,'charged_artifact_cap':2})
    require(binding['compiled_generator_equivalence_verified'] is False and binding['historical_evaluator_runtime_parity_verified'] is False and binding['five_fit_cohort_join'] is None)
    require(binding['seed11_binary_sha256'] != binding['additional_binary_sha256'])
    require(all(binding['configuration_and_data_match_counts'][name] == 400 for name in ('candidate','profile_label','target_weight','structural_penalty','requested_deadline','train_sha256','validation_sha256','projection_sha256')))
    receipt = result['receipt_bindings']; fits,metrics = receipt['fits'],receipt['sample_metrics']
    require(len(fits) == 400 and len(metrics) == 2400)
    require(fits == sorted(fits,key=lambda x:(x['dataset_lineage_id'],x['fit_seed'])) and metrics == sorted(metrics,key=lambda x:(x['dataset_lineage_id'],x['fit_seed'],x['sample_seed'],x['size_multiplier'])))
    require({(x['dataset_lineage_id'],x['fit_seed']) for x in fits} == {(d,f) for d in lineages for f in FIT_SEEDS})
    require({(x['dataset_lineage_id'],x['fit_seed'],x['sample_seed'],x['size_multiplier']) for x in metrics} == {(d,f,s,n) for d in lineages for f in FIT_SEEDS for s in SAMPLE_SEEDS for n in SIZES})
    require(digest(fits) == result['receipt_set_sha256']['fits'] == 'd5be30682d6f544b94ff2c5468b68c5e892cf086f9127368c35eb4bdba712550')
    require(digest(metrics) == result['receipt_set_sha256']['sample_metrics'] == 'e51eee26d1fff82cefbc4b4269d212ef147b807178219473b845836a4dc0bbd2')
    no_private_values(result)

def groups(result):
    rows = []
    for auditor in AUDITORS:
        for size in SIZES:
            cells = [cell for cell in result['cells'] if cell['auditor'] == auditor and cell['size_multiplier'] == size]
            complete = [cell for cell in cells if cell['complete_fit_count'] == 4]
            statuses = collections.Counter(sample['status'] for cell in cells for fit in cell['fits'] for sample in fit['samples'])
            require(set(statuses).issubset({'ok','not_informative'}))
            rows.append({'auditor':auditor,'size_multiplier':size,'lineages':100,'complete_four_fit_lineages':len(complete),'producer_reported_median_of_four_fit_median_retention':statistics.median(statistics.median(fit['producer_reported_retention_median'] for fit in cell['fits']) for cell in complete),'producer_reported_median_four_fit_sample_sd':statistics.median(cell['producer_reported_four_fit_sample_sd'] for cell in complete),'informative_finite_ok_samples':statuses['ok'],'not_informative_samples':statuses['not_informative']})
    return rows

def readme_bytes(rows):
    text = '''# DOPE producer-reported four-fit validation utility

This leaf describes 100 lineages with four additional fit seeds (23, 37, 53, 71), three sample seeds (101, 211, 307), and two validation sizes (n and 4n): 400 fit receipts and 2400 metric receipts. All 600 lineage/size/auditor cells remain in `result.json`, including incomplete cells. Metric values are producer-reported; historical evaluator and runtime parity are unverified.

| Auditor | Size | Complete four-fit lineages / all lineages | Median retention summary | Median four-fit sample SD |
|---|---:|---:|---:|---:|
'''
    labels = {'catboost':'CatBoost','linear':'Linear','mlp':'MLP'}
    for row in rows:
        text += f"| {labels[row['auditor']]} | {'n' if row['size_multiplier'] == 1 else '4n'} | {row['complete_four_fit_lineages']} / 100 | {row['producer_reported_median_of_four_fit_median_retention']:.10f} | {row['producer_reported_median_four_fit_sample_sd']:.10f} |\n"
    text += '''
Each fit median requires three successful, informative, finite reported retentions and finite reported losses. Missing, failed, low-signal, or invalid samples leave that fit median unavailable. The sample SD uses four complete fit medians and denominator 4−1. Within-fit ranges remain separate. The table's retention summary is the median across complete lineages of each lineage's median of four fit medians; its SD summary is the median across those same complete lineages. This conditional summary keeps the denominator of all 100 lineages visible. Retention is not clipped: negative and greater-than-one values remain in the result. Greater-than-one retention is not a superiority claim.

These summary rules are post hoc descriptive rules, not preregistered inferential gates. The producer dynamically imported an evaluator without recording its loaded bytes or runtime versions in these receipts. Configuration or source Git labels do not establish common-auditor parity. Verified common-auditor variation, five-fit stability, paired comparisons, MFS/PTF scores, privacy certification, and a quality winner remain null.

The separate seed-11 metadata comparison records 98 successful fits and two charged-cap outcomes. Data/configuration labels match 400 pairwise comparisons, but all 400 generator binary hashes differ; compiler/product-source equivalence and additional historical evaluator/runtime binding are missing. Binary differences alone do not prove algorithmic differences. Seed 11 remains unjoined. No GPU-training backend, production L3 charged bytes, or retained-storage claim follows from this leaf. Native and paper counts are unchanged.

`source-proof.json` binds canonical receipt-set digests from the additional-four-seed disposition publication, verified byte acquisition, independent helper reviews, real successful controls/reduction waits, and the metadata-only seed-11 comparison. Private captures, source rows, models, and receipt display names are excluded. No evaluation or fit is rerun by the renderer.

Reproduce `summary.csv`, this README table, and `summary.svg` from committed `result.json` with `python3 render.py --output-dir <new-directory>`. Verify existing bytes with `python3 render.py --check`. Matplotlib 3.6.3 and NumPy 1.26.4 are used for the standalone SVG with Agg, DejaVu Sans, fixed layout, path-embedded glyphs, fixed `svg.hashsalt`, and no metadata date. The figure shows only descriptive median four-fit SD with complete/all-lineage counts; it contains no retention or score axis.
'''
    return text.encode()

def figure_bytes(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy
    require(matplotlib.__version__ == '3.6.3' and numpy.__version__ == '1.26.4')
    settings = {'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'path','svg.hashsalt':'DOPE-producer-reported-four-fit-SD-v1','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'figure.facecolor':'white','axes.facecolor':'white'}
    with matplotlib.rc_context(settings):
        figure,axis = plt.subplots(figsize=(8.4,4.8),dpi=100)
        figure.subplots_adjust(left=0.27,right=0.96,top=0.79,bottom=0.23)
        values = [row['producer_reported_median_four_fit_sample_sd'] for row in rows]
        names = {'catboost':'CatBoost','linear':'Linear','mlp':'MLP'}
        labels = [f"{names[row['auditor']]}  {'n' if row['size_multiplier'] == 1 else '4n'}   {row['complete_four_fit_lineages']}/100" for row in rows]
        axis.barh(range(6),values,height=0.64,color=['#355C7D','#355C7D','#537D5B','#537D5B','#875E85','#875E85'])
        axis.set_yticks(range(6),labels); axis.invert_yaxis(); axis.set_xlim(0,max(values)*1.42)
        axis.set_xlabel('Median descriptive four-fit sample SD (complete lineages)'); axis.tick_params(axis='y',length=0)
        for index,value in enumerate(values): axis.text(value+0.001,index,f'{value:.5f}',va='center',fontsize=9)
        figure.text(0.04,0.94,'Producer-reported four-fit spread',fontsize=15,weight='bold')
        figure.text(0.04,0.885,'Validation only; historical evaluator/runtime unverified',fontsize=10)
        figure.text(0.04,0.12,'Labels show complete / all 100 lineages. Each fit median uses 3 informative finite samples.',fontsize=9)
        figure.text(0.04,0.075,'SD uses 4 fit medians (k−1). Post hoc description; no common-auditor or score claim.',fontsize=9)
        output = io.BytesIO(); figure.savefig(output,format='svg',metadata={'Date':None,'Creator':'Dope descriptive renderer v1','Title':'Producer-reported four-fit descriptive spread'}); plt.close(figure)
    return b'\n'.join(line.rstrip() for line in output.getvalue().splitlines()) + b'\n'

def project(result):
    rows = groups(result); output = io.StringIO(newline=''); writer = csv.DictWriter(output,fieldnames=GROUP_FIELDS,lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    return {'summary.csv':output.getvalue().encode(),'README.md':readme_bytes(rows),'summary.svg':figure_bytes(rows)}
def verify_proof_raw(raw):
    require((len(raw),hashlib.sha256(raw).hexdigest()) == PROOF_PIN); return decode(raw)
def load():
    proof = verify_proof_raw((ROOT/'source-proof.json').read_bytes()); refs = proof['public_files']; require(len(refs) == len(PUBLIC_FILES) and {ref['file'] for ref in refs} == PUBLIC_FILES)
    normalized = re.sub(rb'^PROOF_PIN = .*$',b"PROOF_PIN = (0, 'PENDING')",(ROOT/'render.py').read_bytes(),count=1,flags=re.M)
    require(hashlib.sha256(normalized).hexdigest() == proof['renderer_normalized_sha256'])
    for ref in refs:
        path = ROOT/ref['file']; require(not path.is_symlink()); raw = path.read_bytes(); require(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'])
    result = decode((ROOT/'result.json').read_bytes()); validate(result)
    require(proof['canonical_receipt_set_sha256'] == result['receipt_set_sha256'] and proof['historical_evaluator_runtime_parity_verified'] is False and proof['five_fit_cohort_join'] is None)
    return result
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); parser.add_argument('--output-dir',type=pathlib.Path); args = parser.parse_args(); output = project(load())
    if args.check:
        for name,raw in output.items(): require((ROOT/name).read_bytes() == raw)
    else:
        require(args.output_dir is not None and not args.output_dir.exists()); args.output_dir.mkdir(parents=True)
        for name,raw in output.items(): (args.output_dir/name).write_bytes(raw)
    print(json.dumps({'status':'public_descriptive_render_complete','files':sorted(output),'private_inputs_opened':False,'new_fits_or_evaluations':0},sort_keys=True))
if __name__ == '__main__':
    try: main()
    except Exception: raise SystemExit('public description refused')
