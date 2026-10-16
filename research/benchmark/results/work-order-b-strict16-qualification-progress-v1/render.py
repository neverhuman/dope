#!/usr/bin/env python3
"""Regenerate this fixed, public scalar snapshot using only pinned public JSON."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

PINS = {
    'delta.json': (7323, '0bc9236ef855a19b24a830c1832e65ca67ea1bf76b3a1899f92d6311a9adbbb4'),
    'delta.schema.json': (17096, '3a4b457f182f19e0ca6241b9db3201f72f548df44e132168501135c309f6c135'),
    'source-proof.json': (5803, '504c8441936b05012c40d86a799cf1eddbe709145a9e27244bce82bca8a9985a'),
    'source-proof.schema.json': (10510, 'c99cb1af2ef8bf0bdb03d6bcc395d95a121152a0e60ad936cc286d47f46dc1db'),
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate_json_key')
        result[key] = value
    return result


def bad_constant(value):
    raise ValueError('nonfinite_json_constant')


def checked(directory, name):
    path = directory / name
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'public_input_alias')
    raw = path.read_bytes()
    size, digest = PINS[name]
    require(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
            'public_input_pin_changed')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def validate(value, schema):
    if schema is False:
        raise ValueError('schema_false')
    kind = schema['type']
    types = {
        'object': lambda: type(value) is dict,
        'array': lambda: type(value) is list,
        'string': lambda: type(value) is str,
        'boolean': lambda: type(value) is bool,
        'integer': lambda: type(value) is int,
        'number': lambda: type(value) in (int, float) and math.isfinite(value),
        'null': lambda: value is None,
    }
    require(kind in types and types[kind](), 'schema_type')
    if 'const' in schema:
        require(value == schema['const'], 'schema_const')
    if 'enum' in schema:
        require(value in schema['enum'], 'schema_enum')
    if kind == 'object':
        require(set(schema['required']) <= set(value), 'schema_required')
        require(schema['additionalProperties'] is False and
                set(value) <= set(schema['properties']), 'schema_extra_key')
        for key, item in value.items():
            validate(item, schema['properties'][key])
    elif kind == 'array':
        require(schema['minItems'] <= len(value) <= schema['maxItems'], 'schema_array_length')
        for item in value:
            validate(item, schema['items'])
    elif kind == 'string':
        if 'pattern' in schema:
            require(re.fullmatch(schema['pattern'], value) is not None, 'schema_pattern')
        require(len(value) >= schema.get('minLength', 0), 'schema_string_length')
        if 'maxLength' in schema:
            require(len(value) <= schema['maxLength'], 'schema_string_length')
    elif kind in ('integer', 'number'):
        require(value >= schema.get('minimum', 0), 'schema_minimum')
        if 'maximum' in schema:
            require(value <= schema['maximum'], 'schema_maximum')


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)


def contract(data, proof):
    require(data['format'] == 'work-order-b-strict16-qualification-progress-v1' and
            proof['format'] == 'work-order-b-strict16-qualification-source-proof-v1', 'format')
    snapshot, progress, cost = data['snapshot'], data['progress'], data['cost']
    require(snapshot['anchor'] == 'first_strict16_native_terminal' and
            snapshot['live_later_operations_included'] is False, 'fixed_snapshot')
    expected_progress = {
        'total_logical_slots': 500, 'scientific_terminal_slots': 110, 'ok': 108,
        'retained_seed11_ok': 98, 'new_native_ok': 10, 'historical_artifact_caps': 2,
        'pending_logical_slots_in_snapshot': 390, 'first_strict16_native_ok': 1,
        'remaining_strict16_physical_jobs_in_snapshot': 390,
        'historical_paid_prefit_infrastructure_attempts': 1,
        'historical_infrastructure_is_same_logical_slot_as_first_strict16_success': True,
        'historical_failed_physical_attempt_preserved': True,
        'infra_attempt_adds_an_extra_logical_slot': False,
        'inclusive_paid_trial_count': 111, 'stored_native_operations_started': 0,
        'stored_native_operations_started_is_authoritative': False,
    }
    require(progress == expected_progress, 'progress_accounting')
    require(progress['ok'] + progress['historical_artifact_caps'] ==
            progress['scientific_terminal_slots'] and
            progress['scientific_terminal_slots'] + progress['pending_logical_slots_in_snapshot'] ==
            progress['total_logical_slots'], 'logical_partition')
    rows = data['operations']
    ids = ['qualification_1', 'qualification_2', 'first_strict16_native']
    require([r['id'] for r in rows] == ids and
            len({(r['job_sha256'], r['attempt']) for r in rows}) == 3, 'operation_dedup')
    require([r['phase'] for r in rows] ==
            ['generated_qualification', 'generated_qualification', 'native_profile_continuation']
            and [r['seed'] for r in rows] == [11, 11, 37], 'operation_scope')
    require([r['charged_model_plus_projection_bytes'] for r in rows] == [2323, 2323, 762],
            'charged_bytes')
    for row in rows:
        require(row['status'] == 'ok' and row['attempt'] == 1 and
                row['actual_parent_wait_exit'] == 0 and row['co_tenant'] is False and
                row['shared_GPU_cost'] is False and row['GPU_target_operator_verified'] is True
                and row['observed_provider_validation_verified'] is True, 'operation_status')
        require(0 < row['nested_operation_seconds_nonadditive'] <= row['parent_elapsed_seconds']
                <= 600, 'operation_clock')
        require(row['resident_measurement_kind'] ==
                'sum_verified_owned_pid_start_tick_resident_mib', 'observation_basis')
    inputs = proof['input_metadata']
    require(len(inputs) == 11 and len({r['role'] for r in inputs}) == 11, 'input_dedup')
    refs = {r['role']: {'bytes': r['bytes'], 'sha256': r['sha256']} for r in inputs}
    require(proof['operation_joins'] == [
        {k: row[k] for k in ('id', 'job_sha256', 'receipt_ref', 'parent_ref')} for row in rows
    ], 'receipt_join')
    for row, prefix in zip(rows, ('qualification_1', 'qualification_2', 'first_native')):
        require(row['receipt_ref'] == refs[prefix + '_receipt'] and
                row['parent_ref'] == refs[prefix + '_parent'], 'input_operation_join')
    q = data['qualification']
    require(q['generated_operations'] == q['ok'] == 2 and q['seed'] == 11 and
            close(q['generated_R_and_D_parent_seconds'], 337.20003135968) and
            close(sum(r['parent_elapsed_seconds'] for r in rows[:2]),
                  q['generated_R_and_D_parent_seconds']) and
            q['generated_total_ceiling_seconds'] == 1200 and
            q['charged_into_final_cell_budget'] is False and
            close(q['historical_qualification_seconds_separate'], 388.65127065824345),
            'qualification_accounting')
    require(q['parity'] == proof['qualification_parity'] and
            all(q['parity'][key] is True for key in
                ('model_bytes_equal', 'projection_bytes_equal', 'inspection_bytes_equal')),
            'qualification_parity')
    require(q['owned_controller_and_keeper_closed'] is True and q['actual_outer_wait_exit'] == 0
            and q['full_system_runtime_closure_certified'] is False and
            q['production_certified'] is False, 'qualification_lifecycle')
    alias = proof['alias_join']
    require(alias['successor_physical_job_sha256'] == rows[2]['job_sha256'] and
            alias['original_physical_job_sha256'] ==
            '0064b9d563f16e44475468d7855c8f0472fd5d6b480a35f89736a5d4af7d3d0b' and
            alias['seed'] == 37 and alias['original_status'] == 'charged_prefit_infrastructure_failure'
            and alias['paid_infrastructure_history_preserved'] is True and
            alias['release_or_scientific_equivalence_certified'] is False, 'alias_history')
    clocks = {
        'previous_native_and_prefit_paid_parent_seconds': 814.5702133062296,
        'new_first_strict16_native_parent_seconds': 109.85987342894077,
        'cumulative_new_native_and_prefit_paid_parent_seconds': 924.4300867351703,
        'historical_prefit_infrastructure_parent_seconds': 18.162802021950483,
        'new_native_ok_parent_seconds_cumulative': 906.2672847132198,
    }
    require(all(close(cost[k], v) for k, v in clocks.items()), 'native_cost')
    require(close(rows[2]['parent_elapsed_seconds'], cost['new_first_strict16_native_parent_seconds'])
            and close(cost['previous_native_and_prefit_paid_parent_seconds'] +
                      cost['new_first_strict16_native_parent_seconds'],
                      cost['cumulative_new_native_and_prefit_paid_parent_seconds'])
            and close(cost['new_native_ok_parent_seconds_cumulative'] +
                      cost['historical_prefit_infrastructure_parent_seconds'],
                      cost['cumulative_new_native_and_prefit_paid_parent_seconds']), 'parent_cost_join')
    require(all(cost[k] is False for k in
                ('nested_elapsed_added_to_parent', 'qualification_added_to_native_paid_total',
                 'retained_seed11_cost_recharged', 'native8_committed_slice_recharged')), 'no_doublecount')
    require(all(cost[k] is None for k in
                ('campaign_total_wall_seconds', 'GPU_kernel_seconds', 'CPU_core_seconds',
                 'energy_joules', 'true_peak_host_RAM_bytes', 'true_peak_VRAM_bytes', 'hardware_model')),
            'unmeasured_cost_null')
    scope = data['scope']
    require(scope['per_operation_co_tenant_and_shared_GPU_flags_are_measured'] is True and
            scope['prior9_native_aggregate_co_tenant'] is None and
            scope['foreign_DLTrain_campaign_mutex_participation'] is None and
            scope['fixed_campaign_serial_mutex_scope'] == 'our_campaign_jobs_only' and
            scope['true_peak_or_zero_impact_certification'] is False and
            scope['official_tests_opened'] is False, 'sharing_scope')
    require(all(scope[k] is None for k in
                ('sample_completeness', 'fivefit_stability', 'method_complete', 'MFS_v2', 'PTF_v1',
                 'release_safe', 'superiority', 'native_winner')), 'gated_null')
    require(proof['preserved_budget'] == {
        'inclusive_trials': 111, 'cumulative_paid_seconds': 924.4300867351703,
        'historical_prefit_infrastructure_seconds': 18.162802021950483,
    }, 'budget_join')
    require(snapshot['anchor_receipt_sha256'] == rows[2]['receipt_ref']['sha256'], 'snapshot_receipt')
    resource = data['resource_profile']
    require(resource['declared_RAM_envelope_bytes'] == 17179869184 and resource['host'] == 'xbabe2'
            and resource['GPU_device'] == 0 and resource['CPU_affinity'] == list(range(80, 96))
            and resource['cell_trial_cap'] == 8 and resource['cell_inclusive_seconds_cap'] == 43200
            and resource['new_operation_ceiling_seconds'] == 600 and
            resource['new_operation_budget_seconds'] == 240000, 'resource_profile')
    source = proof['source_scope']
    require(resource['runtime_binding_sha256'] == source['runtime_binding_sha256'] and
            resource['binary_sha256'] == source['binary_sha256'] and
            resource['custody_source_sha256'] == source['execution_source_sha256']['source/custody.py']
            and source['full_system_runtime_closure_certified'] is False and
            source['historical_runtime_closure_upgraded'] is False and
            source['production_certified'] is False, 'source_join')
    require(proof['custody'] == {
        'input_metadata_hashes_verified_before_decode': True,
        'models_projections_samples_rows_arrays_TEST_unopened': True,
        'private_receipt_paths_included': False, 'private_model_or_sample_payload_included': False,
        'public_regenerator_requires_private_inputs': False, 'host_calls_or_new_measurements': 0,
    }, 'public_custody')
    require(proof['overlap'] == {
        'snapshot_native8_included_in_native_paid_history': True, 'native8_cost_added_again': False,
        'closed_evaluation_ledgers_included': False, 'qualification_R_and_D_in_native_paid_history': False,
        'later_live_operations_included': False,
    }, 'overlap')


def csv_text(data):
    fields = ['id', 'phase', 'seed', 'attempt', 'status', 'parent_elapsed_seconds',
              'nested_operation_seconds_nonadditive', 'charged_model_plus_projection_bytes',
              'maximum_observed_own_GPU_resident_mib', 'maximum_observed_sampling_gap_seconds',
              'co_tenant', 'shared_GPU_cost', 'receipt_sha256', 'parent_sha256']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in data['operations']:
        out = {key: row[key] for key in fields if key not in ('receipt_sha256', 'parent_sha256')}
        out.update(receipt_sha256=row['receipt_ref']['sha256'], parent_sha256=row['parent_ref']['sha256'])
        for key in ('co_tenant', 'shared_GPU_cost'):
            out[key] = str(out[key]).lower()
        writer.writerow(out)
    return stream.getvalue()


def table_text(data):
    lines = ['| Operation | Seed | Parent seconds | Nested seconds (nonadditive) | Model + projection bytes | Maximum observed own GPU MiB | Co-tenant |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in data['operations']:
        lines.append(f"| {row['id']} | {row['seed']} | {row['parent_elapsed_seconds']:.6f} | "
                     f"{row['nested_operation_seconds_nonadditive']:.6f} | "
                     f"{row['charged_model_plus_projection_bytes']} | "
                     f"{row['maximum_observed_own_GPU_resident_mib']} | false |")
    return '\n'.join(lines) + '\n'


def readme_text(data):
    before = '''# Strict16 qualification and first native continuation

This fixed snapshot ends at the first successful native operation under the declared 16 GiB RAM profile. It contains two completed generated qualification operations and one native continuation operation. Later operations are excluded.

The 500 logical slots contain **110 scientific terminals: 108 OK and 2 historical artifact caps**, with **390 pending**. The OK total includes 98 retained seed11 successes and 10 new native successes. These counts do not establish fivefit stability, sample completeness, or method completion.

## Measured operations

'''
    after = '''
The two generated qualifications each charged 2,323 model plus projection bytes and produced matching model, projection, and inspection hashes. Their **337.20003135968 parent seconds** are separate R&D, outside the final cell budget. Historical qualification cost remains separate.

The first strict16 native operation charged 762 bytes and **109.85987342894077 parent seconds**. Native and prefit infrastructure paid history totals **924.4300867351703 seconds across 111 trials**. This includes the preserved **18.162802021950483 seconds** of historical prefit infrastructure. The successful successor resolves that same logical slot; the failed physical attempt remains in the history and does not add a 501st slot.

## Cost and observation scope

Each row uses its recorded parent controller elapsed time. Nested operation elapsed time is shown for reconciliation and is **nonadditive**. Neither clock is a GPU kernel fit timer. Retained seed11 costs, the earlier native8 slice, closed evaluation ledgers, and qualification clocks are not charged again into this native total. Campaign total wall time, CPU core seconds, energy, hardware model, and true RAM/VRAM peaks are null.

The three displayed receipts record `co_tenant=false` and `shared_GPU_cost=false`. Sharing for the prior nine native operations and foreign DLTrain participation in the campaign mutex remain null. The fixed serial mutex scopes our campaign jobs. No full runtime, production, or zero impact certification is claimed.

The 430 MiB values are maximum **observed** owned process GPU residency, using verified PID/start identities. Recorded maximum sampling gaps are retained in JSON and CSV. These values are not true peaks, complete host RSS measurements, or capacity qualification for another dataset.

The profile retains CPU80–95 on xbabe2, GPU0, a declared 16 GiB RAM envelope, 8 trials per cell, 43,200 inclusive seconds per cell, a 600 second operation ceiling, and a 240,000 second new operation budget. The stored `native_operations_started=0` counter is explicitly non-authoritative; receipt and charged trial joins determine progress.

Official TEST data remains sealed. MFS v2, PTF v1, release safety, superiority, native winner, sample completeness, fivefit stability, and method completion remain null.

## Reproduction

`source-proof.json` contains only immutable metadata digests, byte counts, source pins, and scalar joins. It includes no private receipt paths, model/sample payloads, or source values. The standard library renderer verifies the four committed JSON/schema buffers before decoding and never reopens private inputs.

```sh
python3 -I -S -B research/benchmark/results/work-order-b-strict16-qualification-progress-v1/render.py --check
python3 -I -S -B research/benchmark/results/work-order-b-strict16-qualification-progress-v1/test_render.py
```

Running the renderer without `--check` regenerates this README, `costs.md`, and `costs.csv` from the committed scalar ledger.
'''
    return before + table_text(data) + after


def outputs(data):
    return {'README.md': readme_text(data), 'costs.md': table_text(data), 'costs.csv': csv_text(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    data, schema = checked(directory, 'delta.json'), checked(directory, 'delta.schema.json')
    proof, proof_schema = checked(directory, 'source-proof.json'), checked(directory, 'source-proof.schema.json')
    validate(data, schema)
    validate(proof, proof_schema)
    contract(data, proof)
    for name, content in outputs(data).items():
        path = directory / name
        require(not path.is_symlink(), 'public_output_alias')
        if args.check:
            require(path.read_text(encoding='utf-8') == content, 'public_output_changed')
        else:
            path.write_text(content, encoding='utf-8')
    print('strict16 qualification snapshot: schema, joins, and public regeneration PASS')


if __name__ == '__main__':
    main()
