#!/usr/bin/env python3
"""Project committed rights-safe metadata only; no private receipt reads."""
import argparse,csv,hashlib,io,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parent
FILES={'result.json','schema.json','README.md','operation-cost.csv','dataset-seed-coverage.csv','table.md','source-proof.schema.json'}
PROOF_PIN=(4274, '97e332c16456de01739cb17f9ce4a2ec0dd3f467a4eb9d08db8f42501f64c0ee')
def load():
    proof_path=ROOT/'source-proof.json'; assert not proof_path.is_symlink()
    proof_bytes=proof_path.read_bytes()
    assert (len(proof_bytes),hashlib.sha256(proof_bytes).hexdigest())==PROOF_PIN
    proof=json.loads(proof_bytes)
    assert len(proof['public_files'])==len(FILES)
    assert set(x['file'] for x in proof['public_files'])==FILES
    for x in proof['public_files']:
        assert set(x)=={'file','bytes','sha256'} and x['file'] in FILES
        p=ROOT/x['file']; assert not p.is_symlink()
        b=p.read_bytes(); assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
    r=json.loads((ROOT/'result.json').read_bytes());validate(r);return r

def validate(r):
    assert r['format']=='work-order-b-native-index008-real-GPU-progress-and-cost-v1'
    c=r['logical500'];assert c['scheduled_slots']==500 and c['dataset_lineages']==100
    assert c['accepted']+c['historical_artifact_caps']+c['unresolved_infrastructure_slots']+c['snapshot_unstarted']==500
    assert c['accounted_slots']==c['accepted']+c['historical_artifact_caps']+c['unresolved_infrastructure_slots']
    assert c['accepted_historical_seed11_reuse']+c['accepted_new_fits']==c['accepted']
    assert c['physical_paid_trials']==c['accounted_slots']+c['resolved_old_infrastructure_extra_trial']
    rows=r['dataset_seed_coverage'];seeds=c['scheduled_fit_seeds'];assert len(rows)==100 and len({x['dataset'] for x in rows})==100 and seeds==[11,23,37,53,71]
    for x in rows:
        assert re.fullmatch('[0-9a-f]{16}',x['dataset'])
        assert x['accepted_seed_count']==sum(x['seed'+str(s)].startswith('ok') for s in seeds)
        assert x['all_five_accepted']==(x['accepted_seed_count']==5)
    assert sum(x['accepted_seed_count'] for x in rows)==c['accepted']
    assert sum(x['seed'+str(s)]=='unstarted_at_paid_budget_snapshot' for x in rows for s in seeds)==c['snapshot_unstarted']
    assert sum(x['seed'+str(s)]=='infrastructure_unresolved' for x in rows for s in seeds)==c['unresolved_infrastructure_slots']
    for s in seeds:assert r['seed_coverage']['accepted_by_seed'][str(s)]==sum(x['seed'+str(s)].startswith('ok') for x in rows)
    assert [x['dataset'] for x in rows if x['all_five_accepted']]==r['seed_coverage']['lineages_with_all_five_accepted_seeds']
    o=r['measured_positive_operation'];assert o['status']=='ok' and o['parent_exit_code']==0 and 0<o['nested_operation_seconds_nonadditive']<=o['parent_wait_seconds']<=600
    assert o['paid_trial_increment']==1 and o['charged_model_plus_projection_bytes']==o['model_bytes']+o['projection_bytes']
    assert o['gpu_target_operator_verified'] and o['observed_providers_verified'] and o['GPU_observation_count']>0
    assert o['co_tenant_observed'] is False and o['shared_GPU_cost_observed'] is False
    assert not o['zero_impact_claim'] and not o['causal_impact_identified'] and not o['full_system_runtime_closure_certified']
    for k in ['true_peak_RAM_bytes','true_peak_VRAM_mib','energy_joules','hardware_model','kernel_only_fit_seconds']:assert o[k] is None
    assert all(v is None for v in r['gated_claims'].values())
    b=r['cost_basis'];assert math.isclose(b['previous_new_operation_parent_seconds']+b['increment_parent_seconds'],b['cumulative_new_operation_parent_seconds'],rel_tol=0,abs_tol=1e-9)
    assert b['increment_parent_seconds']==o['parent_wait_seconds'] and b['nested_operator_driver_or_metadata_time_added'] is False and b['stale_counter_authoritative'] is False
    def safe(v):
        if isinstance(v,str):assert not v.startswith('/') and '/home/' not in v and '/mnt/' not in v
        elif isinstance(v,dict):
            for x in v.values():safe(x)
        elif isinstance(v,list):
            for x in v:safe(x)
        elif type(v) is float:assert math.isfinite(v)
    safe(r)

def csv_bytes(fields,rows):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()

def project(r):
    s=r['snapshot'];o=r['measured_positive_operation'];c=r['logical500'];b=r['cost_basis']
    fields=['job_sha256','dataset','fit_seed','attempt','status','parent_exit_code','whole_native_parent_seconds','charged_model_plus_projection_bytes','co_tenant_observed','shared_GPU_cost_observed','maximum_observed_own_GPU_resident_mib','maximum_observed_own_RSS_bytes']
    row={k:s[k] for k in ['job_sha256','dataset','fit_seed','attempt']};row.update({k:o[k] for k in fields[4:] if k!='whole_native_parent_seconds'});row['whole_native_parent_seconds']=format(o['parent_wait_seconds'],'.9f')
    seed_fields=['dataset','seed11','seed23','seed37','seed53','seed71','accepted_seed_count','all_five_accepted']
    pairs=[('Accepted logical fits',c['accepted']),('Historical artifact caps',c['historical_artifact_caps']),('Unresolved infrastructure slots',c['unresolved_infrastructure_slots']),('Snapshot unstarted slots',c['snapshot_unstarted']),('Paid physical trials',c['physical_paid_trials']),('Index008 whole native parent seconds',format(o['parent_wait_seconds'],'.9f')),('Cumulative new-operation parent seconds',format(b['cumulative_new_operation_parent_seconds'],'.9f')),('Selected historical-inclusive cell seconds',format(b['selected_historical_inclusive_cell_seconds'],'.9f')),('Observed own GPU residency MiB',o['maximum_observed_own_GPU_resident_mib']),('Observed own RSS bytes',o['maximum_observed_own_RSS_bytes']),('Lineages with all five accepted seeds',r['seed_coverage']['lineages_with_at_least_five_accepted_seeds'])]
    table='# Frozen index008 progress and measured cost\n\n| Quantity | Value |\n|---|---:|\n'+''.join(f'| {k} | {v} |\n' for k,v in pairs)+'\nResidency and RSS are observations, not true peaks. Parent time includes the whole operation and is not kernel-only fit time. Quality, stability, privacy, and release claims remain unavailable.\n'
    return {'operation-cost.csv':csv_bytes(fields,[row]),'dataset-seed-coverage.csv':csv_bytes(seed_fields,r['dataset_seed_coverage']),'table.md':table.encode()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--output-dir',type=pathlib.Path);a=p.parse_args();r=load();out=project(r)
    if a.check:
        for name,b in out.items():assert (ROOT/name).read_bytes()==b
    else:
        assert a.output_dir is not None and not a.output_dir.exists();a.output_dir.mkdir(parents=True)
        for name,b in out.items():(a.output_dir/name).write_bytes(b)
    print(json.dumps({'status':'public_metadata_projection_verified','private_inputs_reopened':False,'outputs':sorted(out)}))
if __name__=='__main__':main()
