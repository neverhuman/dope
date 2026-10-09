#!/usr/bin/env python3
"""Check and render the pinned additional-four-seed public metadata snapshot."""
import argparse,collections,csv,hashlib,io,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parent
FILES={'result.json','schema.json','README.md','seed-coverage.csv','operation-cost.csv','table.md','source-proof.schema.json'}
PROOF_PIN=(3042, 'a058b2589933594507048abc3bbfe3bc8c1e71e1a2b3f7c462d2adf817f9e1ea')

DIGEST_ENCODING = {'digest_algorithm': 'sha256', 'canonical_JSON': 'UTF-8 json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True); no trailing newline', 'fit_record_fields': ['dataset_lineage_id', 'fit_seed', 'receipt_bytes', 'receipt_sha256'], 'fit_sort_fields': ['dataset_lineage_id', 'fit_seed'], 'metric_record_fields': ['dataset_lineage_id', 'fit_seed', 'sample_seed', 'size_multiplier', 'receipt_bytes', 'receipt_sha256', 'synthetic_sha256_declared'], 'metric_sort_fields': ['dataset_lineage_id', 'fit_seed', 'sample_seed', 'size_multiplier'], 'per_fit_metric_subset': 'Filter complete metric records by dataset_lineage_id and fit_seed; retain every metric record field; use the same metric sort order.'}

def canonical_digest(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",",":"),allow_nan=False,ensure_ascii=True).encode("utf-8")).hexdigest()

def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate JSON key')
        out[k]=v
    return out

def decode(raw):
    def invalid(_):raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def verify_proof_raw(raw):
    assert (len(raw),hashlib.sha256(raw).hexdigest())==PROOF_PIN
    return decode(raw)

def verify_public_bytes(ref,raw):
    assert set(ref)=={'file','bytes','sha256'}and ref['file']in FILES
    assert type(ref['bytes'])is int and ref['bytes']>0
    assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
    return raw

def load():
    p=ROOT/'source-proof.json';assert not p.is_symlink();proof=verify_proof_raw(p.read_bytes())
    refs=proof['public_files'];assert len(refs)==len(FILES)and {x['file']for x in refs}==FILES
    for ref in refs:
        p=ROOT/ref['file'];assert not p.is_symlink();verify_public_bytes(ref,p.read_bytes())
    r=decode((ROOT/'result.json').read_bytes());validate(r)
    assert proof['receipt_counts']=={'fits':400,'sample_metrics':2400}and proof['receipt_stats_stable_before_and_after']is True
    assert proof['authenticated_runner_source']=={k:r['source_identity_scope'][k]for k in('runner_file','runner_bytes','runner_sha256')}
    assert proof["receipt_digest_encoding"]==DIGEST_ENCODING
    fit_records=sorted([{"dataset_lineage_id":x["dataset_lineage_id"],"fit_seed":x["fit_seed"],"receipt_bytes":x["fit_receipt_bytes"],"receipt_sha256":x["fit_receipt_sha256"]}for x in r["measured_fit_receipt_metadata"]],key=lambda x:(x["dataset_lineage_id"],x["fit_seed"]))
    assert canonical_digest(fit_records)==proof["receipt_set_hashes"]["fits"]
    return r

def validate(r):
    assert r['format']=='work-order-b-dope-additional-four-seed-dispositions-v1'
    history=r['historical_attempt_scope'];assert history['scope']=='current_closed_identity_snapshot_not_immutable_attempt_history'and history['all_historical_attempts_accounted']is None and history['retries_or_overwrites_reconciled']is None and history['failure_free_campaign_claim']is False and history['two_manager_intervals_are_full_attempt_ledger']is False
    s,c,h,b,src=r['snapshot'],r['coverage'],r['hardware_and_admission_scope'],r['cost_and_storage_scope'],r['source_identity_scope']
    assert s['fixed_snapshot']is True and s['method']=='DOPE'and s['configuration']=='features12_steps2048'and s['candidate']=='compact_neural_residual_symbolic'
    assert s['tier_requested']=='l3'and s['neural_target_weight']==2.0 and s['neural_structural_penalty']==0.0 and type(s['fit_deadline_seconds_requested'])is int and s['fit_deadline_seconds_requested']==600
    assert s['dataset_lineages']==100 and s['planned_additional_fit_slots']==s['accepted_fit_receipts']==400 and s['sample_metric_receipts_accounted']==2400
    assert s['additional_fit_seeds']==[23,37,53,71]and s['sample_seeds']==[101,211,307]and s['sample_sizes']==[1,4]
    assert s['official_tests_opened']is False and s['frozen_paper_cut_replaced']is False and s['native195_of500_count_increased_by_this_leaf']is False
    assert c['historical_seed11_joined']is False and c['completed_five_seed_lineages']is None and c['five_seed_stability']is None and c['quality_values_reduced']is False
    expected={str(z):{'accepted_fit_receipts':100,'sample_metric_receipts_accounted':600}for z in(23,37,53,71)}
    assert c['per_fit_seed']==expected and c['lineages_with_all_four_additional_seed_receipts']==100
    assert h['requested_device']=='cuda'and h['receipt_device_field_basis']=='producer_constant_not_measured_backend'and h['current_host_capacity_used_as_training_proof']is False
    assert all(h[k]is None for k in('actual_training_backend','GPU_training_verified','historical_per_fit_GPU_observation','co_tenant_or_exclusive_GPU_verified','historical_native_admission_verified'))
    assert src['runner_file']=='research/benchmark/review_fixes/fit_seeds.py'and src['runner_bytes']==22009 and src['runner_sha256']=='01f445f232b6570ecf481beaf00a3f710df8615b5610ea73a83932dfe2127884'
    assert src['binary_and_full_compilation_source_inventory_freshly_verified']is False
    rows=r['measured_fit_receipt_metadata'];assert type(rows)is list and len(rows)==400
    counts=collections.Counter();seen=set();lineages={};timers=[];declared=[];metric_total=0
    for row in rows:
        ds,seed=row['dataset_lineage_id'],row['fit_seed']
        assert type(ds)is str and re.fullmatch('[0-9a-f]{16}',ds)and type(seed)is int and seed in(23,37,53,71)
        assert (ds,seed)not in seen;seen.add((ds,seed));lineages.setdefault(ds,set()).add(seed);counts[seed]+=1
        assert row['status']=='ok'and type(row['fit_exit_code'])is int and row['fit_exit_code']==0
        assert type(row['fit_receipt_bytes'])is int and row['fit_receipt_bytes']>0
        for k in('fit_receipt_sha256','model_sha256_declared','train_sha256','validation_sha256','projection_sha256','binary_sha256_declared','runner_sha256','sample_metric_identity_and_hash_set_sha256'):
            assert type(row[k])is str and re.fullmatch('[0-9a-f]{64}',row[k])
        assert row['runner_sha256']==src['runner_sha256']and row['binary_sha256_declared']==src['binary_sha256_declared']and row['source_git_sha_declared']==src['source_git_sha_declared']
        t,z=row['fit_parent_wall_seconds'],row['receipt_declared_compile_artifact_bytes']
        assert type(t)in(int,float)and math.isfinite(t)and 0<=t<=660 and type(z)is int and z>=0
        timers.append(t);declared.append(z)
        assert type(row['sample_metric_receipts_accounted'])is int and row['sample_metric_receipts_accounted']==6 and type(row['sample_metric_status_ok'])is int and row['sample_metric_status_ok']==6
        metric_total+=row['sample_metric_receipts_accounted']
    assert len(lineages)==100 and all(x=={23,37,53,71}for x in lineages.values())and dict(counts)=={s:100 for s in(23,37,53,71)}and metric_total==2400
    assert math.fsum(timers)==b['successful_fit_parent_wall_seconds_sum']==1560.362 and b['successful_fit_timer_count']==400
    assert sum(declared)==b['receipt_declared_compile_artifact_bytes_sum']==462189
    assert b['manager_reported_CPU_seconds_sum']==35351.298 and b['manager_accounting_intervals']==2 and b['CPU_accounting_is_unit_report_not_fit_wall']is True and b['fit_wall_and_manager_CPU_added']is False
    assert b['artifact_bytes_basis']=='last_compile_log_JSON_artifact_bytes_value_copied_by_producer'and b['model_sample_binary_bytes_freshly_rehashed']is False
    assert all(b[k]is None for k in('sample_and_auditor_wall_seconds','whole_coordinator_parent_cost_seconds','actual_coordinator_parent_wait_exit_code','production_L3_projection_runtime_charged_inventory_bytes','retained_storage_bytes','actual_peak_RSS_bytes','actual_peak_GPU_resident_mib'))
    assert all(x is None for x in r['gated_claims'].values())
    def safe(v):
        if isinstance(v,str):assert not v.startswith('/')and '/home/'not in v and '/mnt/'not in v
        elif isinstance(v,dict):
            for k,z in v.items():assert k not in('argv','raw_cmdline','source_base64','source_values','headers','rows','weights','raw_errors');safe(z)
        elif isinstance(v,list):
            for z in v:safe(z)
        elif type(v)is float:assert math.isfinite(v)
    safe(r)

def csv_bytes(fields,rows):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()

def project(r):
    b=r['cost_and_storage_scope'];counts=r['coverage']['per_fit_seed']
    seeds=[{'fit_seed':s,'planned_additional_fits':100,'accepted_fit_receipts':counts[str(s)]['accepted_fit_receipts'],'sample_metric_receipts_accounted':counts[str(s)]['sample_metric_receipts_accounted']}for s in(23,37,53,71)]
    costs=[{'quantity':'successful_fit_parent_wall_seconds','value':b['successful_fit_parent_wall_seconds_sum'],'unit':'seconds','basis':'400 successful per-fit parent-wall receipt timers','available':'true'},{'quantity':'manager_reported_CPU_seconds','value':b['manager_reported_CPU_seconds_sum'],'unit':'seconds','basis':'two unit-manager accounting intervals; not additive with fit wall','available':'true'},{'quantity':'sample_and_auditor_wall_seconds','value':'','unit':'seconds','basis':'not recorded in successful sample metric receipts','available':'false'},{'quantity':'whole_coordinator_parent_cost_seconds','value':'','unit':'seconds','basis':'no genuine whole-coordinator parent-wait cost receipt','available':'false'},{'quantity':'receipt_declared_compile_artifact_bytes','value':b['receipt_declared_compile_artifact_bytes_sum'],'unit':'bytes','basis':'compile-log JSON declaration; not charged projection/runtime inventory','available':'true'}]
    lines=[('Accepted additional-seed fit receipts','400 / 400'),('Lineages with all four additional seeds','100 / 100'),('Accounted sample metric receipts','2400 / 2400'),('Successful fit parent-wall seconds',b['successful_fit_parent_wall_seconds_sum']),('Manager-reported CPU seconds, separate basis',b['manager_reported_CPU_seconds_sum']),('Receipt-declared compile artifact bytes',b['receipt_declared_compile_artifact_bytes_sum']),('Sample/auditor and whole-coordinator wall cost','Unavailable'),('Five-seed stability and measured GPU backend','Unavailable')]
    table='# DOPE additional four-seed receipt dispositions\n\n| Quantity | Value |\n|---|---:|\n'+''.join('| %s | %s |\n'%z for z in lines)+'\n| Fit seed | Accepted fit receipts | Accounted metric receipts |\n|---|---:|---:|\n'+''.join('| %s | %s | %s |\n'%(z['fit_seed'],z['accepted_fit_receipts'],z['sample_metric_receipts_accounted'])for z in seeds)
    table+='\nFit parent-wall and unit-manager CPU are different bases and must not be added. Artifact bytes are producer declarations, not fresh storage verification or production L3 charged inventory. Seed 11 is unjoined. No metric value or five-seed stability was reduced, no measured GPU backend is established, and this leaf does not increase the separate native 195/500 count or replace the frozen paper cut.\n'
    table+='Historical attempts and unreceipted retries are not fully accounted; this is not a failure-free campaign.\n'
    return {'seed-coverage.csv':csv_bytes(list(seeds[0]),seeds),'operation-cost.csv':csv_bytes(list(costs[0]),costs),'table.md':table.encode()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--output-dir',type=pathlib.Path);a=p.parse_args();r=load();out=project(r)
    if a.check:
        for n,b in out.items():assert (ROOT/n).read_bytes()==b
    else:
        assert a.output_dir is not None and not a.output_dir.exists();a.output_dir.mkdir(parents=True)
        for n,b in out.items():(a.output_dir/n).write_bytes(b)
    print(json.dumps({'status':'public_metadata_projection_verified','outputs':sorted(out),'private_inputs_reopened':False,'new_fits_or_measurements':0}))

if __name__=='__main__':
    try:main()
    except Exception:raise SystemExit('public metadata projection refused')
