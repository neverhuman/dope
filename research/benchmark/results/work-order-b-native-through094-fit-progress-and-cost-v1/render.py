#!/usr/bin/env python3
"""Reproduce hash-pinned public receipt metadata only; no private inputs or science."""
import argparse,csv,hashlib,io,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parent
FILES={'result.json', 'seed-coverage.csv', 'operation-cost.csv', 'schema.json', 'table.md', 'source-proof.schema.json', 'README.md'}
PROOF_PIN=(10560, 'e646d0207f789de6d663dfc375a9c49586de198598290245a31685425988d54b')

def pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result

def decode(raw):
    def invalid(_):raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def load():
    p=ROOT/'source-proof.json';assert not p.is_symlink()
    raw=p.read_bytes();assert (len(raw),hashlib.sha256(raw).hexdigest())==PROOF_PIN
    proof=decode(raw);files=proof['public_files']
    assert len(files)==len(FILES) and {x['file']for x in files}==FILES
    for x in files:
        assert set(x)=={'file','bytes','sha256'} and type(x['bytes'])is int and x['bytes']>0
        p=ROOT/x['file'];assert not p.is_symlink();raw=p.read_bytes()
        assert len(raw)==x['bytes']and hashlib.sha256(raw).hexdigest()==x['sha256']
    result=decode((ROOT/'result.json').read_bytes());validate(result);return result

def validate(r):
    assert r['format']=='work-order-b-native-through094-fit-progress-and-cost-v1'
    assert r['snapshot']['through_paid_queue_index']==94 and r['snapshot']['fixed_snapshot']is True
    c=r['logical500'];s=r['seed_coverage'];b=r['cost_basis'];ops=r['measured_operations'];m=r['excluded_metadata_waits']
    assert [c[k]for k in ['accepted','historical_artifact_caps','unresolved_infrastructure_slots','snapshot_unstarted','distinct_accounted_slots','physical_paid_trials']]==[195,2,7,296,204,205]
    assert c['scheduled_slots']==500 and c['dataset_lineages']==100 and c['accepted']+c['historical_artifact_caps']+c['unresolved_infrastructure_slots']+c['snapshot_unstarted']==500
    assert c['distinct_accounted_slots']==c['accepted']+c['historical_artifact_caps']+c['unresolved_infrastructure_slots']
    assert c['physical_paid_trials']==c['distinct_accounted_slots']+c['resolved_old_infrastructure_extra_trial']
    assert c['accepted_historical_seed11_reuse']==98 and c['accepted_new_population_fits']==97 and 98+97==c['accepted'] and c['infrastructure_is_method_failure']is False
    assert s['scheduled_fit_seeds']==[11,23,37,53,71] and s['accepted_by_seed']=={'11':98,'23':24,'37':26,'53':24,'71':23}
    assert s['accepted_seed_count_histogram']=={'0':2,'1':72,'2':0,'3':3,'4':1,'5':22}
    assert sum(s['accepted_by_seed'].values())==195 and sum(s['accepted_seed_count_histogram'].values())==100
    assert sum(int(k)*v for k,v in s['accepted_seed_count_histogram'].items())==195
    assert s['lineages_with_all_five_accepted_fit_identities']==22 and len(s['all_five_lineage_ids'])==len(set(s['all_five_lineage_ids']))==22
    assert all(re.fullmatch('[0-9a-f]{16}',x)for x in s['all_five_lineage_ids'])
    assert s['historical_reuse_runtime_upgraded']is False and s['mixed_runtime_equivalence_certified']is False and s['sample_completeness']is None and s['fivefit_stability']is None
    assert len(ops)==9 and [x['operation_index']for x in ops]==list(range(86,95)) and len({x['job_sha256']for x in ops})==9 and len({(x['dataset_lineage_id'],x['fit_seed'])for x in ops})==9
    for o in ops:
        assert o['status']=='ok' and o['attempt']==1 and o['parent_exit_code']==0 and o['host']=='xbabe2'
        assert 0<o['nested_operation_seconds_NONADDITIVE']<=o['parent_wait_seconds']<=600
        assert o['GPU_observation_count']>0 and o['gpu_target_operator_verified']is True and o['observed_providers_verified']is True
        assert 0<o['model_bytes']<=o['charged_model_plus_projection_bytes']<=10240 and o['charged_model_plus_projection_bytes']==o['model_bytes']+o['projection_bytes']
        assert o['co_tenant_observed']is False and o['shared_GPU_cost_observed']is False
        assert o['GPU_grant_resident_ceiling_mib']==11324
        assert o['true_peak_RAM_bytes']is None and o['true_peak_VRAM_mib']is None and o['full_system_runtime_closure_certified']is False and o['zero_impact_claim']is False
    assert sum(x['charged_model_plus_projection_bytes']for x in ops)==23161
    assert math.isclose(math.fsum(x['parent_wait_seconds']for x in ops),b['increment_parent_seconds086_to094'],rel_tol=0,abs_tol=1e-9)
    assert b['previous_through085_parent_seconds']==8284.137812728994 and b['new_operation_parent_seconds_through094']==9102.965393944178
    assert math.isclose(b['previous_through085_parent_seconds']+b['increment_parent_seconds086_to094'],b['new_operation_parent_seconds_through094'],rel_tol=0,abs_tol=1e-9)
    assert math.isclose(b['new_operation_parent_seconds_through094']+b['selected_historical_seed11_cell_seconds'],b['selected_historical_inclusive_cell_seconds_ALTERNATE'],rel_tol=0,abs_tol=1e-9)
    assert b['historical_alternate_added_to_primary']is False and b['nested_operator_driver_metadata_time_added']is False and b['all_campaign_R_and_D_or_metadata_seconds']is None
    assert [m['indices086_to094']['closed_unpaid_attempts'],m['index095']['closed_unpaid_attempts']]==[11,20]
    assert m['indices086_to094']['driver_wait_seconds_NONADDITIVE']==79.21575790969655 and m['index095']['driver_wait_seconds_NONADDITIVE']==150.7615494756028
    assert m['indices086_to094']['new_paid_trials']==m['index095']['new_paid_trials']==0 and m['indices086_to094']['method_failure_conclusion']is None and m['index095']['method_failure_conclusion']is None
    assert m['operator_wall_overlaps_native_and_metadata']is True and m['entire_service_wall_seconds']is None and m['actor_retirement_or_new_admission_claim']is False
    a=r['observations_scope086_to094'];assert a['accepted_operations']==a['nonempty_GPU_clock_operations']==a['co_tenant_observed_false_operations']==a['shared_GPU_cost_observed_false_operations']==9
    assert a['max_of_observed_own_GPU_resident_mib']==430 and a['max_of_observed_own_RSS_bytes']==918138880 and a['true_peak_RAM_bytes']is None and a['true_peak_VRAM_mib']is None
    assert a['whole_system_exclusivity_certified']is False and a['causal_zero_impact_identified']is False and a['future_QoS_guaranteed']is False and a['full_system_runtime_closure_certified']is False
    q=r['current_resource_and_runtime_scope'];assert q['GPU_grant_ceiling_upper_bound_mib']==11324 and q['RAM_grant_bytes']==17179869184 and q['reserved_CPU_slot']==list(range(80,96)) and q['fit_timeout_seconds']==600 and q['model_plus_projection_byte_cap']==10240 and q['official_tests_opened']is False
    assert all(v is None for v in r['gated_claims'].values())
    def safe(v):
        if isinstance(v,str):assert not v.startswith('/')and '/home/'not in v and '/mnt/'not in v
        elif isinstance(v,dict):
            for k,x in v.items():assert k not in ['argv','raw_cmdline','source_base64','logs','source_values','headers','rows','weights'];safe(x)
        elif isinstance(v,list):
            for x in v:safe(x)
        elif type(v)is float:assert math.isfinite(v)
    safe(r)

def csv_bytes(fields,rows):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()

def project(r):
    c=r['logical500'];b=r['cost_basis'];s=r['seed_coverage'];m=r['excluded_metadata_waits'];ops=r['measured_operations']
    fields=['operation_index','job_sha256','dataset_lineage_id','fit_seed','attempt','status','parent_exit_code','whole_native_parent_seconds','charged_model_plus_projection_bytes','GPU_observation_count','co_tenant_observed','shared_GPU_cost_observed','maximum_observed_own_GPU_resident_mib','maximum_observed_own_RSS_bytes'];rows=[]
    for o in ops:
        z={k:o[k]for k in fields if k!='whole_native_parent_seconds'};z['whole_native_parent_seconds']=format(o['parent_wait_seconds'],'.9f');rows.append(z)
    seedrows=[{'fit_seed':x,'accepted_fit_identities':s['accepted_by_seed'][str(x)],'historical_seed11_reuse':98 if x==11 else 0,'accepted_new_population_fits':0 if x==11 else s['accepted_by_seed'][str(x)]}for x in s['scheduled_fit_seeds']]
    pairs=[('Accepted fit identities',c['accepted']),('Historical seed11 reuses',c['accepted_historical_seed11_reuse']),('Accepted new population fits',c['accepted_new_population_fits']),('Historical artifact caps',c['historical_artifact_caps']),('Unresolved infrastructure slots',c['unresolved_infrastructure_slots']),('Snapshot unstarted slots',c['snapshot_unstarted']),('Distinct accounted logical slots',c['distinct_accounted_slots']),('Paid physical trial units',c['physical_paid_trials']),('Lineages with all five accepted fit identities',s['lineages_with_all_five_accepted_fit_identities']),('Primary native parent seconds',format(b['new_operation_parent_seconds_through094'],'.9f')),('Increment086–094 parent seconds',format(b['increment_parent_seconds086_to094'],'.9f')),('Historical-inclusive alternate seconds',format(b['selected_historical_inclusive_cell_seconds_ALTERNATE'],'.9f')),('Unpaid086–094 driver wait seconds (excluded)',format(m['indices086_to094']['driver_wait_seconds_NONADDITIVE'],'.9f')),('Unpaid095 driver wait seconds (excluded)',format(m['index095']['driver_wait_seconds_NONADDITIVE'],'.9f'))]
    table='# Fixed through094 fit identities and measured cost\n\n| Quantity | Value |\n|---|---:|\n'+''.join('| %s | %s |\n'%x for x in pairs)+'\n| Fit seed | Accepted identities |\n|---|---:|\n'+''.join('| %s | %s |\n'%(x,s['accepted_by_seed'][str(x)])for x in s['scheduled_fit_seeds'])
    table+='\nNine086–094 fits:23161 charged model-plus-projection bytes;430MiB maximum observed own GPU residency and918138880 bytes maximum observed own RSS. Observations are not true peaks. Their nonempty-clock co-tenant/shared-cost flags are false; no exclusivity or causal impact claim. Twenty unpaid095 refusals add no paid trial. Five-seed identities do not establish evaluation completeness or stability; all gated scores remain unavailable.\n'
    return {'operation-cost.csv':csv_bytes(fields,rows),'seed-coverage.csv':csv_bytes(['fit_seed','accepted_fit_identities','historical_seed11_reuse','accepted_new_population_fits'],seedrows),'table.md':table.encode()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--output-dir',type=pathlib.Path);a=p.parse_args();r=load();out=project(r)
    if a.check:
        for name,b in out.items():assert (ROOT/name).read_bytes()==b
    else:
        assert a.output_dir is not None and not a.output_dir.exists();a.output_dir.mkdir(parents=True)
        for name,b in out.items():(a.output_dir/name).write_bytes(b)
    print(json.dumps({'status':'public_metadata_projection_verified','private_inputs_reopened':False,'outputs':sorted(out)}))

if __name__=='__main__':main()
