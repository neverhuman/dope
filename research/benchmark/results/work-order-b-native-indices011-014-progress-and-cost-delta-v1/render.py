#!/usr/bin/env python3
"""Reproduce pinned public receipt metadata only; no private inputs or science."""
import argparse,csv,hashlib,io,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parent
FILES={'result.json','schema.json','README.md','operation-cost.csv','dataset-seed-coverage.csv','table.md','source-proof.schema.json'}
PROOF_PIN=(4156, '8ac9e7c761398b6a46b404a9c13b10e90908855df86e0fe971c8a5ba75b2aabb')

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
    assert r['format']=='work-order-b-native-indices011-014-progress-and-cost-delta-v1'
    c=r['logical500'];seeds=c['scheduled_fit_seeds'];rows=r['dataset_seed_coverage']
    assert c['accepted']==115 and c['accounted_slots']==124 and c['physical_paid_trials']==125 and c['historical_artifact_caps']==2 and c['unresolved_infrastructure_slots']==7 and c['snapshot_unstarted']==376
    assert c['scheduled_slots']==500 and c['dataset_lineages']==100 and seeds==[11,23,37,53,71]
    assert c['accepted']+c['historical_artifact_caps']+c['unresolved_infrastructure_slots']+c['snapshot_unstarted']==500
    assert c['accounted_slots']==c['accepted']+c['historical_artifact_caps']+c['unresolved_infrastructure_slots']
    assert c['physical_paid_trials']==c['accounted_slots']+c['resolved_old_infrastructure_extra_trial']
    assert c['accepted_historical_seed11_reuse']+c['accepted_new_fits']==c['accepted']
    assert len(rows)==100 and len({x['dataset']for x in rows})==100
    for x in rows:
        assert re.fullmatch('[0-9a-f]{16}',x['dataset'])
        assert x['accepted_seed_count']==sum(x['seed'+str(s)].startswith('ok')for s in seeds)
        assert x['all_five_accepted']==(x['accepted_seed_count']==5)
    assert sum(x['accepted_seed_count']for x in rows)==c['accepted']
    assert sum(x['seed'+str(s)]=='unstarted_at_paid_budget_snapshot'for x in rows for s in seeds)==c['snapshot_unstarted']
    assert sum(x['seed'+str(s)]=='infrastructure_unresolved'for x in rows for s in seeds)==c['unresolved_infrastructure_slots']
    for s in seeds:assert r['seed_coverage']['accepted_by_seed'][str(s)]==sum(x['seed'+str(s)].startswith('ok')for x in rows)
    assert [x['dataset']for x in rows if x['all_five_accepted']]==r['seed_coverage']['lineages_with_all_five_accepted_seeds']
    ops=r['measured_operations'];assert len(ops)==4 and [x['operation_index']for x in ops]==[11,12,13,14] and [x['fit_seed']for x in ops]==[23,37,53,71]
    assert all(x['dataset']=='06b0312a585b436e' and x['host']=='xbabe2'and x['attempt']==1 and x['paid_trial_increment']==1 for x in ops)
    for o in ops:
        assert o['parent_exit_code']==0 and 0<o['nested_operation_seconds_nonadditive']<=o['parent_wait_seconds']<=600
        assert o['official_tests_opened']is False and o['infrastructure_is_method_failure']is False
        assert o['underlying_failure_attribution']is None and o['zero_impact_claim']is False and o['causal_impact_identified']is False and o['full_system_runtime_closure_certified']is False
        for k in ['true_peak_RAM_bytes','true_peak_VRAM_mib','energy_joules','hardware_model','kernel_only_fit_seconds']:assert o[k]is None
    assert [o['status']for o in ops]==['infrastructure_failure','ok','ok','ok']
    assert len({o['job_sha256']for o in ops})==4
    assert [o['cell_inclusive_trial_count_after']for o in ops]==[2,3,4,5]
    assert [o['paid_trial_units_after']for o in ops]==[122,123,124,125]
    for o in ops:
        if o['status']=='ok':
            assert o['GPU_observation_count']>0 and o['gpu_target_operator_verified']is True and o['observed_providers_verified']is True
            assert o['charged_model_plus_projection_bytes']==o['model_bytes']+o['projection_bytes'] and o['model_bytes']>0 and o['projection_bytes']==1635 and o['failure_reason_code']is None
            assert o['co_tenant_observed']is False and o['shared_GPU_cost_observed']is False
        else:
            assert o['GPU_observation_count']==0 and o['gpu_target_operator_verified']is False and o['observed_providers_verified']is False and o['failure_reason_code']=='qos_measurement_unknown'
            for k in ['co_tenant_observed','shared_GPU_cost_observed','maximum_observed_own_GPU_resident_mib','maximum_observed_own_RSS_bytes','initial_observed_device_free_mib','minimum_observed_device_free_mib','maximum_observed_sampling_gap_seconds','model_bytes','model_sha256','projection_bytes','charged_model_plus_projection_bytes']:assert o[k]is None
    b=r['cost_basis'];assert math.isclose(sum(x['parent_wait_seconds']for x in ops),b['increment_parent_seconds'],rel_tol=0,abs_tol=1e-9)
    assert math.isclose(b['previous_new_operation_parent_seconds']+b['increment_parent_seconds'],b['cumulative_new_operation_parent_seconds'],rel_tol=0,abs_tol=1e-9)
    assert math.isclose(b['cumulative_new_operation_parent_seconds']+b['selected_historical_seed11_cell_seconds'],b['selected_historical_inclusive_cell_seconds'],rel_tol=0,abs_tol=1e-9)
    assert b['nested_operator_driver_or_metadata_time_added']is False and b['v13_operator_time_added_to_native_parent_cost']is False
    assert b['stale_counter_authoritative']is False and b['held_service_runtime_seconds']is None and b['held_service_runtime_receipt_bound']is False
    assert all(v is None for v in r['gated_claims'].values())
    assert r['resource_profile']['native_scientific_grant_RAM_bytes']==17179869184 and r['resource_profile']['reserved_CPU_slot']==list(range(80,96))and r['resource_profile']['fit_timeout_seconds']==600 and r['resource_profile']['new_resident4_pilot_applied']is False
    def safe(v):
        if isinstance(v,str):assert not v.startswith('/')and '/home/'not in v and '/mnt/'not in v
        elif isinstance(v,dict):
            for x in v.values():safe(x)
        elif isinstance(v,list):
            for x in v:safe(x)
        elif type(v)is float:assert math.isfinite(v)
    safe(r)

def csv_bytes(fields,rows):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()

def project(r):
    c=r['logical500'];b=r['cost_basis'];ops=r['measured_operations']
    fields=['operation_index','job_sha256','dataset','fit_seed','attempt','status','parent_exit_code','whole_native_parent_seconds','charged_model_plus_projection_bytes','GPU_observation_count','co_tenant_observed','shared_GPU_cost_observed','maximum_observed_own_GPU_resident_mib','maximum_observed_own_RSS_bytes'];rows=[]
    for o in ops:
        row={k:o[k]for k in fields if k!='whole_native_parent_seconds'};row['whole_native_parent_seconds']=format(o['parent_wait_seconds'],'.9f');rows.append(row)
    sf=['dataset','seed11','seed23','seed37','seed53','seed71','accepted_seed_count','all_five_accepted']
    pairs=[('Accepted logical fits',c['accepted']),('Historical artifact caps',c['historical_artifact_caps']),('Unresolved infrastructure slots',c['unresolved_infrastructure_slots']),('Snapshot unstarted slots',c['snapshot_unstarted']),('Distinct accounted slots',c['accounted_slots']),('Paid physical trial units',c['physical_paid_trials'])]+[('Index%03d whole native parent seconds'%o['operation_index'],format(o['parent_wait_seconds'],'.9f'))for o in ops]+[('Increment over frozen121 snapshot seconds',format(b['increment_parent_seconds'],'.9f')),('Cumulative new-operation parent seconds',format(b['cumulative_new_operation_parent_seconds'],'.9f')),('Selected historical-inclusive cell seconds',format(b['selected_historical_inclusive_cell_seconds'],'.9f')),('Lineages with all five accepted seeds',r['seed_coverage']['lineages_with_at_least_five_accepted_seeds'])]
    table='# Frozen indices011 through014 progress and measured cost\n\n| Quantity | Value |\n|---|---:|\n'+''.join('| %s | %s |\n'%x for x in pairs)+'\n| Operation | Status | GPU observations | Observed own GPU residency (MiB) | Observed own RSS (bytes) |\n|---|---|---:|---:|---:|\n'
    for o in ops:table+='| %03d | %s | %s | %s | %s |\n'%(o['operation_index'],o['status'],o['GPU_observation_count'],o['maximum_observed_own_GPU_resident_mib']if o['GPU_observation_count']else 'unavailable',o['maximum_observed_own_RSS_bytes']if o['GPU_observation_count']else 'unavailable')
    table+='\nResidency and RSS are retained observations, not true peaks. Index011 has no retained GPU observations; its sharing status and peaks are unavailable. Parent exit0 does not imply a successful fit. Alternate historical-inclusive totals and overlapping operator timers must not be added to the new-operation parent total. Quality, stability, privacy, and release claims remain unavailable.\n'
    return {'operation-cost.csv':csv_bytes(fields,rows),'dataset-seed-coverage.csv':csv_bytes(sf,r['dataset_seed_coverage']),'table.md':table.encode()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--output-dir',type=pathlib.Path);a=p.parse_args();r=load();out=project(r)
    if a.check:
        for name,b in out.items():assert (ROOT/name).read_bytes()==b
    else:
        assert a.output_dir is not None and not a.output_dir.exists();a.output_dir.mkdir(parents=True)
        for name,b in out.items():(a.output_dir/name).write_bytes(b)
    print(json.dumps({'status':'public_metadata_projection_verified','private_inputs_reopened':False,'outputs':sorted(out)}))
if __name__=='__main__':main()
