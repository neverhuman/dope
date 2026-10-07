"""Reproduce closed evaluation-cost CSV/table/README using included pinned metadata only."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

PINS = {
 'delta.json': (7057,'fe98e7d0622099bd207d1c84ef82b1596daabc79372811ebc318d5da6cc42ff2'),
 'delta.schema.json': (10078,'88e65a5d37e7843593b61ba7b4b6694fa561705b84a0c3701d623900573a2bfc'),
 'source-proof.json': (5762,'279d05355ee014899ea6eb650f22278927261e52b7c34ee123f1e29486a23cd7'),
 'source-proof.schema.json': (9737,'5d20b8861df446801f47e44c533e2b9edb2e0cf27b230d63b4474f429d70cd98'),
}
ROWS = {
 'tabsyn_retained48_evaluation': (48,48,0,'xbabe1',list(range(96,104))),
 'arf_retained1116_evaluation': (1116,1200,0,'xbabe2',list(range(104,112))),
 'copula_chow_new2017_evaluation': (2017,2400,5,'xbabe1',list(range(96,104))),
 'retained5_utility_only1729': (5,None,5,'xbabe2',list(range(104,112))),
}
NULL_ROW_FIELDS = ('CPU_model','GPU_model','GPU_wall_seconds','peak_GPU_used_mib','energy_joules',
                   'core_seconds','co_tenant','shared_gpu_cost','actual_parent_wait_exit','actual_kernel_exit',
                   'whole_operation_wall_seconds','method_completion_claim')

def require(ok,reason):
 if not ok:raise ValueError(reason)

def pairs(items):
 out={}
 for key,value in items:
  require(key not in out,'duplicate_json_key');out[key]=value
 return out

def checked(directory,name):
 path=directory/name
 require(not any(p.is_symlink()for p in(path,*path.parents)),'public_input_alias')
 raw=path.read_bytes();length,pin=PINS[name]
 require(len(raw)==length and hashlib.sha256(raw).hexdigest()==pin,'public_input_pin_changed')
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:require(False,'nonfinite_json'))

def validate(value,schema):
 kind='null'if value is None else'boolean'if type(value)is bool else'integer'if type(value)is int else'number'if type(value)is float else'string'if type(value)is str else'array'if type(value)is list else'object'if type(value)is dict else'unsupported'
 allowed=schema.get('type');allowed=[allowed]if type(allowed)is str else allowed
 require(allowed is None or kind in allowed or(kind=='integer'and'number'in allowed),'schema_type')
 if 'enum'in schema:require(any(type(value)is type(v)and value==v for v in schema['enum']),'schema_enum')
 if 'const'in schema:require(type(value)is type(schema['const'])and value==schema['const'],'schema_const')
 if kind in('integer','number'):
  require(math.isfinite(value),'schema_nonfinite')
  if 'minimum'in schema:require(value>=schema['minimum'],'schema_minimum')
 if kind=='object':
  require(set(schema.get('required',[]))<=set(value),'schema_required')
  properties=schema.get('properties',{})
  require(schema.get('additionalProperties',True)or set(value)<=set(properties),'schema_extra_property')
  for k,v in value.items():
   if k in properties:validate(v,properties[k])
 if kind=='array':
  require(len(value)>=schema.get('minItems',0)and len(value)<=schema.get('maxItems',len(value)),'schema_array_length')
  if schema.get('uniqueItems'):require(len({json.dumps(v,sort_keys=True,allow_nan=False)for v in value})==len(value),'schema_repeated_item')
  for item in value:validate(item,schema.get('items',{}))

def check_contract(value):
 require(value['format']=='work-order-b-closed-evaluation-cost-delta-v5'and type(value['version'])is int and value['version']==1,'ledger_identity')
 require([r['id']for r in value['rows']]==list(ROWS),'ledger_row_scope')
 require(value['new_metric_physical_jobs']==3181 and value['new_utility_only_jobs']==5
         and value['retained_diagnostic_jobs_not_recharged']==5 and value['full_retained_panel_physical_diagnostics']==3186
         and value['full_retained_panel_logical_aliases']==3648,'physical_logical_scope')
 for key in('utility_outer_clock_added_to_metric_timer_sum','old_A952_seconds_added','v4_sample_or_qualification_clocks_added','official_tests_opened'):
  require(value[key]is False,'overlap_or_TEST_claim')
 for key in('campaign_wall_seconds','generator_fit_seconds','generator_sample_seconds','mfs_v2','ptf_v1','release_safe_l3','superiority','native_winner'):
  require(value[key]is None,'unmeasured_or_gated_claim')
 require(value['first12_publication_status']=='separate_committed_slice_excluded_from_this_delta','first12_base_claim')
 for row in value['rows']:
  count,aliases,retained,host,cores=ROWS[row['id']]
  require(row['new_physical_jobs']==row['typed_closed_receipt_count']==count and row['full_panel_logical_aliases']==aliases
          and row['retained_diagnostics_excluded']==retained,'row_alias_or_physical_scope')
  require(row['cpu_host']==host and row['cpu_affinity']==cores,'CPU_scope')
  require(row['ram_limit_bytes']==4294967296 and row['swap_limit_bytes']==0,'resource_cap_changed')
  require(type(row['peak_rss_bytes'])is int and 0<=row['peak_rss_bytes']<=row['ram_limit_bytes'],'RSS_measurement')
  require(row['rss_basis']=='process_lifetime_RUSAGE_SELF_ru_maxrss_times_1024','RSS_basis')
  require(row['clock_basis']=='per_job_time.perf_counter_wall_sum','timer_basis')
  for key in NULL_ROW_FIELDS:require(row[key]is None,'unknown_exit_hardware_sharing_or_cost')
  require(type(row['metric_timer_wall_seconds'])in(int,float)and math.isfinite(row['metric_timer_wall_seconds'])and row['metric_timer_wall_seconds']>=0,'timer_invalid')
  if row['id']=='retained5_utility_only1729':
   require(row['phase']=='utility_only'and row['whole_runner_wall_seconds']>=row['metric_timer_wall_seconds'],'utility_nested_clock')
  else:require(row['phase']=='evaluation'and row['whole_runner_wall_seconds']is None,'unmeasured_outer_clock')
 require(value['new_metric_timer_wall_seconds_sum']==sum(r['metric_timer_wall_seconds']for r in value['rows']),'timer_sum_changed')

def render(value):
 check_contract(value)
 headings=['id','method','phase','new_physical_jobs','full_panel_logical_aliases','metric_timer_wall_seconds',
           'whole_runner_wall_seconds','cpu_host','cpu_affinity','peak_rss_bytes','ram_limit_bytes',
           'host_memory_total_bytes','CPU_model','GPU_model','GPU_wall_seconds','energy_joules','core_seconds',
           'co_tenant','shared_gpu_cost','actual_parent_wait_exit','actual_kernel_exit']
 output=io.StringIO(newline='');writer=csv.writer(output,lineterminator='\n');writer.writerow(headings)
 for row in value['rows']:
  writer.writerow(['null'if row[k]is None else','.join(map(str,row[k]))if k=='cpu_affinity'else row[k]for k in headings])
 table=['| Evaluation scope | New physical actions | Full-panel logical aliases | Metric timer wall seconds | Outer runner seconds | Process high-water RSS bytes | Host / CPU cores |',
        '|---|---:|---:|---:|---:|---:|---|']
 for row in value['rows']:
  table.append('| '+row['method']+' | '+str(row['new_physical_jobs'])+' | '+str(row['full_panel_logical_aliases']if row['full_panel_logical_aliases']is not None else'—')+' | '+format(row['metric_timer_wall_seconds'],'.6f')+' | '+('null'if row['whole_runner_wall_seconds']is None else format(row['whole_runner_wall_seconds'],'.6f'))+' | '+str(row['peak_rss_bytes'])+' | '+row['cpu_host']+' / '+str(row['cpu_affinity'][0])+'–'+str(row['cpu_affinity'][-1])+' |')
 table='\n'.join(table)+'\n'
 readme='''# Closed retained evaluation costs: v5

Four authenticated closed CPU evaluation scopes add no generator fit or sampling costs. TabSyn covers 48 retained evaluations across eight lineages; ARF covers 1,116 physical evaluations / 1,200 logical aliases; GaussianCopula and Chow-Liu cover 2,017 new physical evaluations plus five retained diagnostics / 2,400 logical aliases. Five utility-only actions use those retained diagnostic job identities with auditor seed 1729.

'''+table+'''
The new per-job timers sum to **'''+repr(value['new_metric_timer_wall_seconds_sum'])+''' seconds**. These are wall durations from `time.perf_counter`, not CPU time or elapsed campaign time. The diagnostic driver timers include input verification/decoding, A952 diagnostics, utility auditors and final verification/capacity checks; preliminary capacity checks, module setup and receipt writing are outside those timers. The utility-only timer covers input verification/decoding and utility; its separate 7.970948356203735-second outer runner includes setup and postchecks. That outer clock is not added to the timer sum.

Logical aliases never multiply physical costs. The five retained A952 diagnostic timers (31.6338860578835 seconds) are excluded entirely; three overlap ROOT's frozen first12 batch and all five overlap its completed16. ROOT's first12 publication was merged as [PR169](https://github.com/neverhuman/dope/pull/169) at `774c72f16ebae006ad627eddeacf2f5c727fa70b` after exact-head review of `c01048e58a698d9ff4eab1b47cec2d963eb75f0a`. It is a separate committed slice; its evaluator clocks remain excluded from this delta. The previous v4 sample/qualification/engineering clocks are excluded. No cross-ledger campaign total is claimed.

RSS is the process lifetime `RUSAGE_SELF.ru_maxrss` high-water mark in bytes, not summed family RSS, an allocation cap or an additive cost. Each execution records a 4 GiB RAM bound and zero swap; the utility-only bound is retained accepted-launch evidence, not a later live kernel measurement. Host total RAM comes from each frozen receipt lock. CPU/GPU models, GPU device use/time, energy, core-seconds, co-tenancy and shared-GPU cost are unknown and remain null. Parent/kernel exit codes remain unknown; collected-unit defaults and PID absence are not exit0 evidence.

Evaluation receipts are closed; this does not certify full method training or a native winner. Official TEST stays sealed. MFS-v2, PTF-v1, release-safe and superiority remain null. The immutable source and receipt locks remain in private custody. This public leaf contains aggregate costs and digests; no rows, models, samples, headers or source extrema.

## Reproduce

Run `python3 -I -S -B research/benchmark/results/work-order-b-cost-closure-delta-v5/render.py --check` after adoption. It authenticates the included JSON/proof/schemas before decoding and compares CSV, table and README in memory. It needs no private receipt payload, numerical dependency or host call.

- `delta.json` and `delta.schema.json`: measured cost rows and strict metadata shape.
- `costs.csv` / `costs.md`: exact CSV numbers / rounded readable table; CSV `null` means unmeasured.
- `source-proof.json` and its schema: input digests, scoped verification, exclusions and remaining gaps.
- `render.py`: committed-input regenerator.
'''
 return {'costs.csv':output.getvalue().encode(),'costs.md':table.encode(),'README.md':readme.encode()}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args()
 value=checked(args.directory,'delta.json');proof=checked(args.directory,'source-proof.json')
 validate(value,checked(args.directory,'delta.schema.json'));validate(proof,checked(args.directory,'source-proof.schema.json'))
 expected=render(value)
 for name,raw in expected.items():
  path=args.directory/name;require(not any(p.is_symlink()for p in(path,*path.parents)),'public_leaf_alias')
  if args.check:require(path.read_bytes()==raw,'public_leaf_changed:'+name)
  else:
   require(not path.exists(),'public_leaf_exists:'+name)
   with path.open('xb')as stream:stream.write(raw)
 print(json.dumps({'status':'PASS','metadata_only':True,'checked':args.check,'rows':4,'public_files':sorted(expected)}))

if __name__=='__main__':main()
