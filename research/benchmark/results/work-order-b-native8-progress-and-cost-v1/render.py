"""Render frozen native8 public metadata; never read private campaign inputs."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

PINS = {'delta.json': (12829, 'f9c04198e77fbefab8a3d41ee6441c0cddb306f32043bdfbfaf32f3b474239fc'), 'delta.schema.json': (12340, 'b818180261d53164ec3d810689ab10265bcc166e499ed5d394ee3c018d475689'), 'source-proof.json': (5617, '38c61384f90d1d9c3f760a7c2375ca1ffc14229ea3385454aacc44984170775d'), 'source-proof.schema.json': (3355, '693bc277d8bce03de857cef557438ea69f611417ce8425636de44dcf293da848')}

def require(condition,reason):
 if not condition:raise ValueError(reason)

def pairs(items):
 result={}
 for key,value in items:
  require(key not in result,'duplicate_json_key');result[key]=value
 return result

def bad_constant(value):
 raise ValueError('nonfinite_json_constant')

def checked(directory,name):
 path=directory/name
 require(not any(p.is_symlink() for p in (path,*path.parents)),'public_input_alias')
 raw=path.read_bytes();size,digest=PINS[name]
 require(len(raw)==size and hashlib.sha256(raw).hexdigest()==digest,'public_input_pin_changed')
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad_constant)

def validate(value,schema):
 if schema is False:raise ValueError('schema_false')
 kind=schema['type']
 types={'object':lambda:type(value)is dict,'array':lambda:type(value)is list,
        'string':lambda:type(value)is str,'boolean':lambda:type(value)is bool,
        'integer':lambda:type(value)is int,'number':lambda:type(value)in(int,float) and math.isfinite(value),
        'null':lambda:value is None}
 require(kind in types and types[kind](),'schema_type')
 if 'const'in schema:require(value==schema['const'],'schema_const')
 if kind=='object':
  require(set(schema['required'])<=set(value),'schema_required')
  require(schema['additionalProperties']is False and set(value)<=set(schema['properties']),'schema_extra_key')
  for key,item in value.items():validate(item,schema['properties'][key])
 elif kind=='array':
  require(schema['minItems']<=len(value)<=schema['maxItems'],'schema_array_length')
  for item in value:validate(item,schema['items'])
 elif kind=='string':
  if 'pattern'in schema:require(re.fullmatch(schema['pattern'],value)is not None,'schema_pattern')
  if 'minLength'in schema:require(len(value)>=schema['minLength'],'schema_string_length')
 elif kind in('integer','number'):
  require(value>=schema.get('minimum',0),'schema_minimum')

def contract(data,proof):
 require(data['format']=='work-order-b-native8-progress-cost-v1' and proof['format']=='work-order-b-native8-source-proof-v1','format')
 progress=data['progress'];require(progress=={'total_logical_slots':500,'closed_logical_slots':108,'ok':106,'historical_seed11_ok_reuse':98,'new_native_ok':8,'historical_artifact_caps':2,'unstarted_in_frozen_snapshot':392},'frozen_progress')
 rows=data['operations'];require(len(rows)==8 and [r['ordinal']for r in rows]==list(range(1,9)),'ordinal')
 require(len({(r['job_sha256'],r['attempt'])for r in rows})==8,'duplicate_job_attempt')
 require(len({r['terminal_sha256']for r in rows})==8,'terminal_alias')
 require(len({r['parent_receipt_sha256']for r in rows})==8,'parent_alias')
 require([r['seed']for r in rows]==[23,37,53,71,23,37,53,71],'seed_order')
 refs={r['source_id']:r for r in proof['inputs']};require(len(refs)==len(proof['inputs']),'duplicate_input_id')
 for row in rows:
  require(row['attempt']==1 and row['status']=='ok' and row['actual_parent_wait_exit_code']==0,'terminal_status')
  require(row['co_tenant']is False and row['shared_gpu_cost']is False,'sharing_fact')
  require(row['resident_measurement_kind']=='sum_verified_owned_pid_start_tick_resident_mib','resident_basis')
  require(row['maximum_observed_own_resident_mib']==430,'resident_observation')
  for prefix,suffix in [('native-operation-','terminal'),('native-operation-','parent')]:
   ref=refs[f"{prefix}{row['ordinal']:02d}-{suffix}"]
   byte_key='terminal_bytes'if suffix=='terminal'else'parent_receipt_bytes'
   sha_key='terminal_sha256'if suffix=='terminal'else'parent_receipt_sha256'
   require(ref['bytes']==row[byte_key] and ref['sha256']==row[sha_key],'public_receipt_join')
 cost=data['cost'];parent=sum(r['parent_elapsed_seconds']for r in rows)
 require(parent==cost['sum8_parent_seconds']==cost['charged_new_whole_actual_seconds'],'parent_charge_sum')
 require(sum(r['parent_elapsed_seconds']for r in rows[:-1])==cost['prior7_parent_seconds'] and rows[-1]['parent_elapsed_seconds']==cost['new1_parent_seconds'],'prefix_increment')
 require(sum(r['operation_elapsed_seconds']for r in rows)==cost['sum8_nested_operation_seconds_nonadditive'],'nested_sum')
 require(sum(r['charged_artifact_bytes_model_plus_projection']for r in rows)==cost['charged_artifact_bytes_model_plus_projection_sum']==23581,'artifact_charge_sum')
 budget=data['budget'];require(budget['maximum_inclusive_trials_per_cell']==8 and budget['maximum_inclusive_seconds_per_cell']==43200 and budget['maximum_new_attempt_actual_seconds']==600 and budget['new_operation_compute_ceiling_seconds']==240000,'budget_caps')
 require(budget['native_operations_started_stored_value']==0 and budget['stored_counter_is_authoritative_fit_count']is False,'stale_counter')
 require(budget['changed_cell_trial_count_before']==4 and budget['changed_cell_trial_count_after']==5 and budget['initial_trial_counter_sum']==100 and budget['current_trial_counter_sum']==108,'trial_charge')
 require(data['scope']=='cumulative_measured_native8_snapshot_not_additive_campaign_delta','cost_scope')
 require(proof['metadata_only']is True and proof['public_regenerator_needs_private_receipts']is False,'public_scope')
 require(proof['true_peak_or_kernel_time_inferred']is False and proof['historical_reuse_or_qualification_cost_recharged']is False,'excluded_inference')


CSV_FIELDS = (
 'ordinal','job_sha256','logical_identity_sha256','attempt','seed','status',
 'actual_parent_wait_exit_code','parent_elapsed_seconds','operation_elapsed_seconds',
 'charged_artifact_bytes_model_plus_projection','maximum_observed_own_resident_mib',
 'maximum_observed_sampling_gap_seconds','co_tenant','shared_gpu_cost',
)

def csv_text(data):
 out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=CSV_FIELDS,lineterminator='\n')
 writer.writeheader()
 for row in data['operations']:
  writer.writerow({key:('false' if row[key] is False else 'true' if row[key] is True else row[key]) for key in CSV_FIELDS})
 return out.getvalue()

def table_text(data):
 lines=['| Operation | Seed | Status | Parent elapsed (s) | Nested operation (s) | Charged model + projection (B) | Maximum observed owned GPU residency (MiB) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
 for row in data['operations']:
  lines.append(f"| {row['ordinal']} | {row['seed']} | {row['status']} | {row['parent_elapsed_seconds']:.6f} | {row['operation_elapsed_seconds']:.6f} | {row['charged_artifact_bytes_model_plus_projection']} | {row['maximum_observed_own_resident_mib']} |")
 lines.append(f"| **Eight-operation snapshot** | | | **{data['cost']['sum8_parent_seconds']:.6f}** | **{data['cost']['sum8_nested_operation_seconds_nonadditive']:.6f}** | **{data['cost']['charged_artifact_bytes_model_plus_projection_sum']}** | **Observation only** |")
 return '\n'.join(lines)+'\n'

def readme_text(data):
 return f'''# DOPE native8 progress and measured operation cost

This frozen snapshot has **108 closed logical slots of 500**: 106 `ok` (98 historical seed11 reuse and eight newly measured native fits) and two historical artifact caps. The other 392 slots are unstarted in the frozen continuation snapshot. These counts do not establish completion of the 500-slot campaign or stability across five fits. Metadata GPU refusals are not method failures or scientific fits.

The eight actual new fit operations use seeds 23, 37, 53 and 71 twice each. Their inclusive controller parent timers sum to **{data['cost']['sum8_parent_seconds']:.6f} seconds**. The prior seven contribute {data['cost']['prior7_parent_seconds']:.6f} seconds and the eighth adds {data['cost']['new1_parent_seconds']:.6f} seconds. This is a cumulative native8 snapshot; do not add it to another ledger containing any of the same receipt digests.

{table_text(data)}
The nested operation timers are shown separately and must not be added to parent time. Neither timer is GPU kernel time or a separate fit-only/sample-only clock. Historical reused costs, qualification/R&D clocks, metadata refusal clocks and evaluation ledgers are excluded. A campaign-wide elapsed total is unknown.

The 23,581 charged bytes are the receipts' model-plus-projection accounting; no model or projection body was opened. All eight receipts explicitly record `co_tenant=false` and `shared_gpu_cost=false`. These receipt-scoped observations do not certify future exclusivity or foreign mutex participation.

The 430 MiB values are **maximum observed owned GPU residency**, from the sum of verified owned PID/start-tick GPU-memory observations. They do not measure host RSS. Sampling gaps are recorded per operation in JSON/CSV. They are not true peaks or a certified VRAM cap. True peak RAM/VRAM, hardware models, CPU time, GPU kernel time and energy remain null.

Historical trials and the existing deadline are preserved. Per-cell caps remain eight trials and 43,200 seconds, each new operation retains its 600-second ceiling, and the new-operation budget retains 240,000 seconds. The eighth operation changes one cell's inclusive trial count from four to five. The charged ledger's stored `native_operations_started=0` was not updated by the controller and is preserved as a non-authoritative field; eight actual fits are established by their immutable terminal and parent receipts.

`source-proof.json` lists immutable SHA256/byte references without private paths or payloads. The renderer hashes all four included JSON/schema buffers before decoding, validates strict public schemas and receipt joins, and regenerates this README, `costs.md` and `costs.csv` using only included rights-safe JSON:

```sh
python3 -I -S -B render.py --check
python3 -I -S -B render.py --output-dir PATH
```

Official TEST remains sealed. MFS-v2, PTF-v1, release, superiority, full runtime certification, native winner and five-fit stability claims remain null. Later operations are outside this frozen snapshot.
'''


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--output-dir',type=Path)
 args=parser.parse_args();require(not(args.check and args.output_dir is not None),'mode_conflict')
 here=Path(__file__).resolve().parent
 data=checked(here,'delta.json');schema=checked(here,'delta.schema.json')
 proof=checked(here,'source-proof.json');proof_schema=checked(here,'source-proof.schema.json')
 validate(data,schema);validate(proof,proof_schema);contract(data,proof)
 generated={'costs.csv':csv_text(data),'costs.md':table_text(data),'README.md':readme_text(data)}
 if args.check:
  for name,text in generated.items():
   path=here/name;require(not path.is_symlink() and path.read_bytes()==text.encode(),'public_generated_file_changed')
 else:
  output=here if args.output_dir is None else args.output_dir;output.mkdir(parents=True,exist_ok=True)
  require(not output.is_symlink(),'output_alias')
  for name,text in generated.items():
   path=output/name;require(not path.is_symlink(),'output_file_alias');path.write_text(text)
 print(json.dumps({'operations':8,'closed_logical_slots':108,'parent_seconds':data['cost']['sum8_parent_seconds'],'public_inputs_only':True},sort_keys=True))

if __name__=='__main__':main()
