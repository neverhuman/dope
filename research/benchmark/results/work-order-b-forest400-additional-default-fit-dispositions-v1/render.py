#!/usr/bin/env python3
"""Reproduce pinned Forest400 public receipt metadata; no private inputs or science."""
import argparse,collections,csv,hashlib,io,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parent
FILES=set(['result.json', 'schema.json', 'README.md', 'seed-coverage.csv', 'table.md', 'source-proof.schema.json'])
PROOF_PIN=(201585, '36735a8f612624ca68c4b957cfd59fdc7d2ae630dec29be8a05ee2ed3fce82c3')

def pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result

def decode(raw):
    def invalid(_):raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def csv_bytes(fields,rows):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()

def verify_proof_raw(raw):
    assert (len(raw),hashlib.sha256(raw).hexdigest())==PROOF_PIN
    return decode(raw)

def load():
    p=ROOT/'source-proof.json';assert not p.is_symlink()
    proof=verify_proof_raw(p.read_bytes());files=proof['public_files']
    assert len(files)==len(FILES) and {x['file']for x in files}==FILES
    for x in files:
        assert set(x)=={'file','bytes','sha256'} and type(x['bytes'])is int and x['bytes']>0
        p=ROOT/x['file'];assert not p.is_symlink();raw=p.read_bytes()
        assert len(raw)==x['bytes']and hashlib.sha256(raw).hexdigest()==x['sha256']
    r=decode((ROOT/'result.json').read_bytes());validate(r)
    assert {x['job_sha256']for x in proof['receipt_refs']}=={x['job_sha256']for x in r['measured_fit_metadata']}
    return r

def validate(r):
    assert r['format']=='work-order-b-forest400-additional-default-fit-dispositions-v1'
    s,c,b,q=r['snapshot'],r['coverage'],r['cost_and_storage_scope'],r['receipt_scope']
    assert s['fixed_snapshot']is True and s['method']=='Forest-Flow'and s['configuration']=='author_default'
    assert s['additional_fit_seeds']==[23,37,53,71]and s['dataset_lineages']==100 and s['planned_additional_fit_slots']==s['physical_closed']==400
    assert s['accepted_additional_default_fits']==291 and s['timeout_missing_artifacts']==109 and s['other_dispositions']==0 and s['new_tuning_trials']==0 and s['native_selection_complete']is None and s['official_tests_opened']is False
    assert c['completed_five_seed_lineages']is None and c['historical_fit11_joined']is False and c['sample_evaluation_completeness']is None and c['fivefit_stability']is None
    expected={'23':{'ok':73,'timeout':27},'37':{'ok':73,'timeout':27},'53':{'ok':73,'timeout':27},'71':{'ok':72,'timeout':28}}
    assert c['per_seed_status_counts']==expected and c['complete_four_additional_seed_lineages']==71
    rows=r['measured_fit_metadata'];assert len(rows)==400
    seen=set();lineages={};counts={str(seed):{'ok':0,'timeout':0}for seed in(23,37,53,71)};fit=[];elapsed=[];declared=[]
    for row in rows:
        jid,ds,seed,status=row['job_sha256'],row['dataset_lineage_id'],row['fit_seed'],row['status']
        assert isinstance(jid,str)and re.fullmatch('[0-9a-f]{64}',jid)and jid not in seen;seen.add(jid)
        assert isinstance(ds,str)and re.fullmatch('[0-9a-f]{16}',ds)and type(seed)is int and seed in(23,37,53,71)and status in('ok','timeout')
        assert seed not in lineages.setdefault(ds,{});lineages[ds][seed]=status;counts[str(seed)][status]+=1
        assert row['artifact_inventory_verified']is False and row['actual_peak_RSS_bytes']is None and row['actual_peak_GPU_resident_mib']is None
        if status=='timeout':assert all(row[k]is None for k in('successful_fit_seconds','successful_elapsed_seconds','receipt_declared_artifact_bytes'))
        else:
            f,e,z=row['successful_fit_seconds'],row['successful_elapsed_seconds'],row['receipt_declared_artifact_bytes']
            assert type(f)in(int,float)and type(e)in(int,float)and math.isfinite(f)and math.isfinite(e)and 0<=f<=e and type(z)is int and z>=0
            fit.append(f);elapsed.append(e);declared.append(z)
    assert len(lineages)==100 and all(set(x)=={23,37,53,71}for x in lineages.values())and counts==expected
    assert sum(all(v=='ok'for v in x.values())for x in lineages.values())==c['complete_four_additional_seed_lineages']
    assert math.fsum(fit)==b['successful_fit_seconds_sum']==49475.683213091455
    assert math.fsum(elapsed)==b['successful_elapsed_seconds_sum_ALTERNATE']==54299.77247472573
    assert sum(declared)==b['receipt_declared_cumulative_artifact_bytes']==72869335252
    assert b['successful_timers_are_alternate_nonadditive']is True and b['fit_and_elapsed_timers_added']is False and b['artifact_inventory_verified']is False
    assert b['artifact_bytes_basis']=='historical_successful_receipt_declarations_not_current_retained_disk'
    assert all(b[k]is None for k in('retained_disk_bytes','timeout_actual_elapsed_seconds_sum','timeout_artifact_bytes_sum','whole_round_parent_cost_seconds','actual_coordinator_wait_exit_code','actual_peak_RSS_bytes','actual_peak_GPU_resident_mib','co_tenant_observed','shared_GPU_cost_observed'))
    assert q['fit_timeout_seconds']==600 and q['CPU_affinity_declared']==[64,65,66,67]and q['RAM_cap_declared_bytes']==8589934592 and q['new_fits_or_evaluations_by_publication']==0 and q['runtime_or_artifact_reopened_by_publication']is False
    assert all(v is None for v in r['gated_claims'].values())
    h=r['historical_checkpoint'];assert h['physical_closed']==343 and h['additional_default_ok']==268 and h['additional_default_timeouts']==75 and h['kept_immutable']is True and h['earlier_13_published_cohorts_counted_in_this_leaf']is False
    def safe(v):
        if isinstance(v,str):assert not v.startswith('/')and '/home/'not in v and '/mnt/'not in v
        elif isinstance(v,dict):
            for k,x in v.items():assert k not in('argv','raw_cmdline','source_base64','logs','source_values','headers','rows','weights');safe(x)
        elif isinstance(v,list):
            for x in v:safe(x)
        elif type(v)is float:assert math.isfinite(v)
    safe(r)

def project(r):
    c=r['coverage'];b=r['cost_and_storage_scope'];s=r['snapshot'];counts=c['per_seed_status_counts'];seeds=s['additional_fit_seeds']
    rows=[dict(fit_seed=seed,planned_additional_fits=100,accepted_fit_receipts=counts[str(seed)]['ok'],timeout_missing_artifacts=counts[str(seed)]['timeout'])for seed in seeds]
    quantities=[('Closed additional-default slots','400 / 400'),('Accepted fit receipts',s['accepted_additional_default_fits']),('Timeout missing artifacts',s['timeout_missing_artifacts']),('Lineages with all four additional seeds accepted',c['complete_four_additional_seed_lineages']),('Complete five-seed lineages','Unavailable: fit seed 11 not joined'),('Successful fit seconds',str(b['successful_fit_seconds_sum'])),('Successful elapsed seconds (alternate)',str(b['successful_elapsed_seconds_sum_ALTERNATE'])),('Historical receipt-declared cumulative artifact bytes',b['receipt_declared_cumulative_artifact_bytes']),('Timeout elapsed / whole-round parent cost / true peaks','Unavailable')]
    table='# Forest400 additional-default fit dispositions\n\n| Quantity | Value |\n|---|---:|\n'+''.join('| %s | %s |\n'%x for x in quantities)+'\n| Fit seed | Accepted | Timeout missing artifacts |\n|---|---:|---:|\n'+''.join('| %s | %s | %s |\n'%(seed,counts[str(seed)]['ok'],counts[str(seed)]['timeout'])for seed in seeds)
    table+='\nThe two successful timer sums overlap and are alternate bases; do not add them. Artifact bytes are cumulative historical receipt declarations, not verified current retained disk. Timeouts are missing artifacts and contribute no comparative win. No common validation, stability across five fits, MFS, PTF, privacy, release or superiority score is measured by this publication.\n'
    return {'seed-coverage.csv':csv_bytes(['fit_seed','planned_additional_fits','accepted_fit_receipts','timeout_missing_artifacts'],rows),'table.md':table.encode()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--output-dir',type=pathlib.Path);a=p.parse_args();r=load();out=project(r)
    if a.check:
        for name,b in out.items():assert (ROOT/name).read_bytes()==b
    else:
        assert a.output_dir is not None and not a.output_dir.exists();a.output_dir.mkdir(parents=True)
        for name,b in out.items():(a.output_dir/name).write_bytes(b)
    print(json.dumps({'status':'public_metadata_projection_verified','private_inputs_reopened':False,'outputs':sorted(out)}))

if __name__=='__main__':main()
