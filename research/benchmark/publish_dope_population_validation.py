"""Publish a complete frozen all100 validation panel; all production claims null."""
from collections import Counter,defaultdict
import csv
import io
import json
import math
from pathlib import Path
from statistics import median
from .manifest import digest
from .publish_dope_population_fits import build as fit_build
from .publish_s3_forest import check,read,require,safe,shared_inventory
from .publish_s3_matched import PROFILES, schema

RESULTS = Path(__file__).parent/'results'
NAME = 'dope-s3-population-validation'
from .score import sha256

ROOT = Path('/mnt/fast-scratch/dope-benchmark/dope-s3-population-validation-v1')
ROUND_SHA = '8d6fb1b5a5a028acf45673feadd40e8a1dc29f9e27bb52f0bf246a20a0f009ac'
FIT_RECEIPT_SHA = 'b0471a8936042b907c4481fb2b84e2205bedd2fbec7538bfd26d1f8d07d94461'
SEEDS = (101,211,307)
SIZES = (1,4)
AUDITORS = ('catboost','linear','mlp')

def aggregate(cells):
    ids = sorted({c['dataset'] for c in cells})
    identities = [(c['dataset'],c['profile'],c['sample_seed'],c['size_multiplier']) for c in cells]
    require(len(ids)==100 and len(identities)==len(set(identities))==2400
            and set(identities)=={(d,p,s,z) for d in ids for p in PROFILES for s in SEEDS for z in SIZES},
            'complete all100 all-profile matrix required')
    groups = defaultdict(list)
    for c in cells:
        require(c['fit_seed']==11 and c['counts_as_dope_win'] is False
                and all(c[k] is None for k in ('mfs_v2','ptf_v1','release_safe_l3','paired_superiority')),
                'validation panel acquired a gated claim')
        groups[(c['dataset'],c['profile'],c['size_multiplier'])].append(c)
    summary = []
    for (d,p,z),values in sorted(groups.items()):
        values.sort(key=lambda r:r['sample_seed']);utility={}
        require(len({v['charged_artifact_bytes'] for v in values})==1,'profile byte charges disagree')
        for a in AUDITORS:
            scores=[]
            for v in values:
                u=v['utility'].get(a,{}) if v['utility'] else {}
                value=u.get('retention') if v['status']=='ok' and u.get('informative') is True else None
                require(value is None or (isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)),
                        'nonfinite common utility')
                scores.append(value)
            complete=all(v is not None for v in scores)
            utility[a]={'complete_informative_sample_group':complete,
                        'sample_retention_values':scores,'median_retention':median(scores) if complete else None}
        summary.append({'dataset':d,'profile':p,'size_multiplier':z,'fit_seed':11,'sample_cells':3,
                        'charged_artifact_bytes':values[0]['charged_artifact_bytes'],
                        'statuses':dict(Counter(v['status'] for v in values)),'utility':utility})
    panels=[]
    for p in PROFILES:
        for z in SIZES:
            selected=[g for g in summary if g['profile']==p and g['size_multiplier']==z]
            audits={}
            for a in AUDITORS:
                scores=[g['utility'][a]['median_retention'] for g in selected if g['utility'][a]['complete_informative_sample_group']]
                audits[a]={'complete_informative_lineages':len(scores),
                           'median_of_lineage_sample_medians':median(scores) if scores else None}
            panels.append({'profile':p,'size_multiplier':z,'lineages':100,
                           'lineage_status_counts':dict(Counter('ok' if g['statuses']=={'ok':3} else 'incomplete_or_unavailable' for g in selected)),
                           'utility':audits,'comparison_scope':'descriptive validation only; no paired superiority or fit uncertainty'})
    return summary,panels

def anchored(receipt_sha,report_sha):
    require(all(isinstance(v,str) and len(v)==64 and set(v)<=set('0123456789abcdef') for v in (receipt_sha,report_sha)),
            'publication requires external frozen receipt and accounting anchors')
    refs={}
    check(ROOT/'round.lock.json',ROUND_SHA,refs)
    check(ROOT/'receipt-lock-v1.json',receipt_sha,refs)
    anchor=read(ROOT/'receipt-lock-v1.json')
    require(anchor['round_sha256']==ROUND_SHA and anchor['fit_receipt_lock_sha256']==FIT_RECEIPT_SHA
            and anchor['official_tests_opened'] is False and anchor['mfs_v2'] is None and anchor['ptf_v1'] is None,
            'validation receipt lock identity differs')
    for name,h in anchor['refs'].items():check(name,h,refs)
    require(anchor['reconciliation_sha256']==report_sha,'reconciliation anchor differs')
    check(ROOT/'reconciliation-v1.json',report_sha,refs)
    return read(ROOT/'reconciliation-v1.json'),read(ROOT/'round.lock.json'),refs

def build(receipt_sha,report_sha):
    report,lock,refs=anchored(receipt_sha,report_sha)
    require(report['complete_matrix'] is True and report['logical_sample_cells']==2400
            and report['physical_batches_closed']==report['physical_batches_planned']==232
            and report['round_sha256']==ROUND_SHA,'validation matrix did not close')
    require(read(ROOT/'coordinator-exit.json')['exit_code']==0
            and read(ROOT/'completion.json')['counts']==report['physical_status_counts'],
            'validation coordinator did not close cleanly')
    require(lock['source_files']=={str(p):refs[str(p)] for p in (ROOT/'source').rglob('*') if p.is_file()}
            and not any(p.is_symlink() for p in (ROOT/'source').rglob('*')),'source inventory differs')
    require(lock['gpu_operations_enabled'] is False and lock['full_campaign_admitted'] is False
            and lock['official_tests_opened'] is False and lock['global_family_selected'] is False
            and all(lock[k] is None for k in ('mfs_v2','ptf_v1','release_safe','superiority')),
            'validation scope changed')
    runtime=shared_inventory(lock,refs)
    fits=fit_build();by_fit={c['fit_job_sha256']:c for c in fits['cells']}
    require(fits['receipt_lock_sha256']==FIT_RECEIPT_SHA,'fit matrix anchor differs')
    jobs={digest(j):j for j in lock['jobs']}
    require(len(jobs)==232,'physical matrix duplicated')
    frozen_logical={(c['dataset'],c['profile'],c['sample_seed'],c['size_multiplier']):c for c in lock['logical_cells']}
    require(len(frozen_logical)==2400,'logical freeze duplicated')
    cells=[];physical={}
    for b in report['batches']:
        key=b['physical_job_sha256'];job=jobs[key];out=ROOT/'attempts'/key/'attempt-0001'
        require(key not in physical and b['receipt_sha256']==refs[str(out/'receipt.json')],
                'physical receipt binding differs')
        receipt=read(out/'receipt.json')
        require(receipt['job']==job and receipt['round_sha256']==ROUND_SHA and receipt['status']==b['status']
                and receipt['official_tests_opened'] is False and receipt['native_selection_changed'] is False
                and receipt['new_generator_fits_started']==0 and receipt['global_family_selected'] is False
                and all(receipt[k] is None for k in ('mfs_v2','ptf_v1','release_safe','superiority')),
                'validation receipt identity or claims differ')
        require(not any(p.is_symlink() for p in out.rglob('*'))
                and {p.name for p in out.iterdir() if p.is_file() and p.name!='receipt.json'}==set(receipt['evidence_files']),
                'physical evidence gained an alias or file')
        for name,h in receipt['evidence_files'].items():
            require(Path(name).name==name and refs[str(out/name)]==h,'physical evidence digest differs')
        worker=safe(job['worker']['path']);model=safe(job['model_path'])
        require(job['final'] is False and job['fit_seed']==11 and job['track']=='common-numeric'
                and job['sample_seeds']==list(SEEDS) and job['size_multipliers']==list(SIZES),
                'physical job schedule or scope differs')
        require(not (worker/'test.csv').exists() and {p.name for p in worker.iterdir()}==set(job['worker']['files'])
                and not any(p.is_symlink() for p in worker.rglob('*')),'worker inventory changed')
        for name,h in job['worker']['files'].items():require(refs[str(worker/name)]==h,'worker lineage differs')
        require(refs[str(model)]==job['model_sha256'] and job['artifact_bytes']==model.stat().st_size+(worker/'projection.json').stat().st_size<=10240,
                'generator or complete byte charge differs')
        physical[key]=b
    require(set(physical)==set(jobs),'physical receipt matrix incomplete')
    require(math.isclose(sum(b['operation_seconds'] for b in physical.values()),report['new_operation_seconds'],abs_tol=1e-9),
            'physical cost reconciliation differs')
    for c in report['cells']:
        expected=frozen_logical[(c['dataset'],c['profile'],c['sample_seed'],c['size_multiplier'])]
        require(all(c[k]==v for k,v in expected.items()),'logical identity differs from preregistered matrix')
        fit=by_fit[c['fit_job_sha256']];key=c['physical_job_sha256'];measured=None;receipt_ref=None
        require((fit['dataset'],fit['profile'],fit['fit_seed'])==(c['dataset'],c['profile'],c['fit_seed'])
                and fit['receipt']['sha256']==c['fit_receipt_sha256'] and fit['model_sha256']==c['model_sha256']
                and fit['charged_artifact_bytes']==c['charged_artifact_bytes'], 'logical fit identity differs')
        if key is None:
            require(c['status']=='fit_unavailable' and fit['status']==c['unavailable_reason']=='charged_artifact_cap'
                    and c['sample_evidence'] is None,'failed fit acquired metrics')
        else:
            batch=physical[key];out=ROOT/'attempts'/key/'attempt-0001'
            require(c['status']==batch['status'] and c['validation_receipt_sha256']==batch['receipt_sha256']
                    and jobs[key]['model_sha256']==fit['model_sha256'] and jobs[key]['dataset']==fit['dataset']
                    and read(fit['receipt']['path'])['job']['worker']==jobs[key]['worker'],
                    'logical alias, worker or status differs')
            receipt_ref={'path':str(out/'receipt.json'),'sha256':batch['receipt_sha256']}
            if c['status']=='ok':
                sample=c['sample_evidence'];name=f"n{c['size_multiplier']}-seed{c['sample_seed']}"
                require(sample['metric_path']==str(out/(name+'.metric.json')) and sample in batch['samples']
                        and refs[sample['metric_path']]==sample['metric_sha256'], 'metric cell digest differs')
                measured=read(sample['metric_path'])
                require(measured['dependencies']==runtime['expected_versions']
                        and measured['implementation_sha256']==lock['metric_source_sha256']
                        and measured['gate_profile_complete'] is False and measured['mfs_v2'] is None,
                        'shared evaluator identity differs')
        cells.append({'dataset':c['dataset'],'method':'DOPE','profile':c['profile'],'fit_seed':11,
                      'sample_seed':c['sample_seed'],'size_multiplier':c['size_multiplier'],'status':c['status'],
                      'charged_artifact_bytes':fit['charged_artifact_bytes'],'artifact_within_l3_cap':fit['status']=='ok',
                      'fit_receipt':fit['receipt'],'validation_receipt':receipt_ref,'physical_job_sha256':key,
                      'metric_receipt':c['sample_evidence'],'utility':measured['utility'] if measured else None,
                      'copy_counts':measured['copy_counts'] if measured else None,
                      'real_vs_real_control_counts':measured['real_vs_real_control_counts'] if measured else None,
                      'null_loss':measured['null_loss'] if measured else None,
                      'marginal_ks_mean':measured['marginal_ks_mean'] if measured else None,
                      'pair_correlation_fidelity':measured['pair_correlation_fidelity'] if measured else None,
                      'c2st_auc':measured['c2st_auc'] if measured else None,
                      'mfs_v2':None,'ptf_v1':None,'release_safe_l3':None,'paired_superiority':None,'counts_as_dope_win':False})
    summary,panels=aggregate(cells)
    require(dict(Counter(c['status'] for c in cells))==report['logical_status_counts'],'logical counts differ')
    return {'format':'dope-s3-population-all-profile-validation-panel','version':1,
            'scope':'All100 training-derived S3 validation lineages, four unchanged GPU research profiles, one fit seed; bounded training cohorts, no test certification',
            'source_sha256':sha256(Path(__file__)),'source_locks':{'round':ROUND_SHA,'validation_receipts':receipt_sha,
             'validation_reconciliation':report_sha,'fit_receipts':FIT_RECEIPT_SHA,'metric':lock['metric_source_sha256']},
            'dataset_ids':fits['dataset_ids'],'datasets':100,'fit_seeds':[11],'sample_seeds':list(SEEDS),'sizes':list(SIZES),
            'logical_sample_cells':2400,'logical_status_counts':report['logical_status_counts'],
            'physical_sample_batches':232,'physical_status_counts':report['physical_status_counts'],
            'all_frozen_cells_accounted':True,'identical_artifact_worker_aliases_charge_once':True,
            'cells':cells,'summary':summary,'profile_panels':panels,
            'cost':{'shared_sampler_and_evaluator_seconds':report['new_operation_seconds'],
                    'coordinator_wall_seconds':report['coordinator_wall_seconds'],'host':'xbabe2',
                    'new_generator_fits':0,'new_tuning_trials':0,'new_gpu_fit_operation_seconds':fits['cost']['new_fit_operation_seconds'],
                    'gpu_fit_cost_accounting':fits['cost'],
                    'architecture_research_spend_separate_from_final_cell_parity':True,'job_attributed_energy_joules':None},
            'replay_policy':lock['replay_policy'],'metric_replay_dataset_ids':lock['metric_replay_dataset_ids'],
            'runtime':{'shared_versions':runtime['expected_versions'],'declared_import_inventory_verified':True,
                       'system_dynamic_library_closure_certified':False,'historical_full_runtime_closure_upgraded':False},
            'immutable_references':[{'path':str(ROOT/n),'sha256':refs[str(ROOT/n)]} for n in ('round.lock.json','receipt-lock-v1.json','reconciliation-v1.json','completion.json','coordinator-exit.json')],
            'missing_evidence':['final five fit seeds','n/2n/4n/8n final schedules','complete privacy attacks',
                                'projection-only utility cost','public-core paired analysis','five-lock full campaign admission'],
            'native_selection_changed':False,'global_family_selected':False,'full_campaign_admitted':False,
            'official_tests_opened':False,'production_certified':False,'mfs_v2':None,'ptf_v1':None,
            'release_safe_l3':None,'paired_superiority':None,'counts_as_dope_win':False}

def tables(report):
    summary,panels=aggregate(report['cells'])
    require(summary==report['summary'] and panels==report['profile_panels'],'public summary differs')
    out=io.StringIO();writer=csv.writer(out,lineterminator='\n')
    writer.writerow(['dataset','profile','fit_seed','size_multiplier','charged_bytes','statuses',*[a+'_median_retention' for a in AUDITORS]])
    for g in summary:
        writer.writerow([g['dataset'],g['profile'],11,g['size_multiplier'],g['charged_artifact_bytes'],json.dumps(g['statuses'],sort_keys=True),
                         *[g['utility'][a]['median_retention'] for a in AUDITORS]])
    lines=['# DOPE all100 GPU research validation','',report['scope'],'',
           'Single fit seed; three sample seeds. All four profiles remain visible. Retention is null-normalized TSTR/TRTR on training-derived validation. No fit uncertainty or paired superiority claim. Official tests sealed; MFS-v2/PTF-v1/release-safe scores null.','',
           '| Profile | Size | CatBoost informative lineages /100 | CatBoost median | Linear informative /100 | Linear median | MLP informative /100 | MLP median |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p in panels:
        values=[]
        for a in AUDITORS:
            v=p['utility'][a];s=v['median_of_lineage_sample_medians']
            values.extend([str(v['complete_informative_lineages']),'null' if s is None else f'{s:.6g}'])
        lines.append('| '+' | '.join([p['profile'],str(p['size_multiplier']),*values])+' |')
    lines+=['','Medians cover only complete informative three-sample groups; denominators and every failure remain explicit. They are descriptive aggregates, with no DOPE win or production family selection. Model plus projection bytes are charged. Bulk samples and runtime detail stay on scratch.','']
    return out.getvalue(),'\n'.join(lines)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--reconciliation-sha256',required=True)
    args=parser.parse_args()
    report=build(args.receipt_sha256,args.reconciliation_sha256)
    path=RESULTS/(NAME+'.json')
    path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(report),sort_keys=True,indent=2)+'\n')
    table,markdown=tables(report)
    path.with_suffix('.csv').write_text(table);path.with_suffix('.md').write_text(markdown)
    print(path)


if __name__=='__main__':main()
