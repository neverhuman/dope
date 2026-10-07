"""Static metadata predicates derived from the frozen common A952 queue adapter.

The source adapter remains authenticated provenance and is never executed.
Callers supply an authenticated JSON reader that tracks every transitive receipt
and rehashes those references before publishing. No I/O or source loading here.
"""
import hashlib
import json
from typing import Callable

BASE_CONFIG = {'path': '/home/ubuntu/dope/.agent/worktrees/integration/target/tabsyn-common-evaluator-round-candidate-v3/ROOT.operation-template.DISABLED.v3.json', 'bytes': 9509, 'sha256': '35d2093808020d4421542355f0823a8b7b37125282dd750e3b25d15ff8e04130'}


TS_CONFIG = {'path': '/home/ubuntu/dope/.agent/worktrees/integration/target/tabsyn-common-evaluator-round-candidate-v3/TS143.operation-template.DISABLED.v3.json', 'bytes': 9551, 'sha256': '6deb28b186ce98a68625f55a55667ab6ffdd3173ab4a670d9f1b15eec9b1af69'}


LEGACY_BASE_CONFIG = {'bytes': 14650, 'path': '/home/ubuntu/dope/.agent/worktrees/integration/target/work-order-b-expanded-metric-round-v1/runtime-final-rehash-repair-v3/kind-one.runtime-final-v3.config.disabled.json', 'sha256': 'fcddb060479cd6697f50ef047b15e97f98384d63d8fe932062e7a123ac1f4d22'}


LEGACY_ROOT_ROUND = {'bytes': 3401474, 'path': '/home/ubuntu/dope/.agent/worktrees/integration/target/work-order-b-expanded-metric-round-v1/runtime-final-rehash-repair-v3/round-proposal-v5.runtime-final.json', 'sha256': '3f4781dea37756382f1a92f249060766236980e65bf17c258b88f833831a0fe9'}


KIND_PROVENANCE_BRIDGE = {'path': '/home/ubuntu/dope/.agent/worktrees/integration/target/tabsyn-common-evaluator-round-candidate-v3/kind-map-provenance.bridge.DISABLED.v3.json', 'bytes': 113539, 'sha256': '64dfa7448a87bbef14a205e270a6be56b9ff0e1c4eec7cf68b15f8dac266dd55'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def validate_jobs(lock):
    TS = lock.get('scope') == 'tabsyn_closed143_common_validation_v2'
    require((TS and lock['matrix'] is None and lock['kind_jobs'] == {} and len(lock['jobs']) == 143
             and lock['historical_parent_exit'] is None) or
            (not TS and len(lock['kind_jobs']) == 100 and len(lock['jobs']) == 5724),
            'frozen_job_counts_changed')
    for jobs in (lock['kind_jobs'], lock['jobs']):
        require(all(hashlib.sha256(canonical(job)).hexdigest() == key for key, job in jobs.items()),
                'canonical_job_changed')
    workers = [job['worker_key'] for job in lock['kind_jobs'].values()]
    require(len(set(workers)) == (0 if TS else 100), 'kind_worker_duplicate')
    require(lock['deadline_seconds_per_job'] == 600 and lock['ram_cap_bytes'] == 16 * 1024 ** 3
            and lock['max_cores'] == 16 and lock['gpu_allowed'] is False, 'frozen_caps_changed')


def positive(lock, record, checked: Callable[[dict[str, object]], dict[str, object]]):
    """Authenticate a completed one-action result; never turn a failure into reuse."""
    config = checked(record['config_ref'])
    request = checked(config['request_ref'])
    operation = checked(record['operation_ref'])
    stdout = checked(record['stdout_ref'])
    reused_kind = config['action'] == 'kind-one'
    baseline = checked(LEGACY_BASE_CONFIG if reused_kind else TS_CONFIG
                       if config['action'] == 'tabsyn-metric-one' else BASE_CONFIG)
    original = checked(LEGACY_ROOT_ROUND) if reused_kind else lock
    if reused_kind:
        bridge = checked(KIND_PROVENANCE_BRIDGE)
        require(lock['kind_jobs'] == original['kind_jobs'] and lock['kind_rule'] == original['kind_rule']
                and record in bridge['positive_kind_receipts']
                and bridge['historical_metric_results_reused_for_A952'] == 0, 'kind_reuse_provenance_changed')
    require(canonical(checked(config['original_round_ref'])) == canonical(original)
            and config['source_files'] == baseline['source_files']
            and config['source_root'] == baseline['source_root']
            and config['runtime_ref'] == baseline['runtime_ref']
            and config['preparation_root'] in (baseline['preparation_root'], '/home/ubuntu/dope-scratch-x2/work-order-b-expanded-metric-serial-v1'), 'positive_frozen_science_changed')
    action = config['action']
    require(action in ('kind-one', 'metric-one', 'tabsyn-metric-one'), 'unrecognized_positive_action')
    if action == 'tabsyn-metric-one':
        require(lock.get('scope') == 'tabsyn_closed143_common_validation_v2'
                and stdout.get('method') == 'TabSyn' and stdout.get('historical_parent_exit') is None
                and stdout.get('production_certified') is False, 'TS_actual_operation_scope_changed')
    jobs = lock['kind_jobs'] if action == 'kind-one' else lock['jobs']
    key = request['job_sha256']
    require(key in jobs and canonical(request['job']) == canonical(jobs[key]), 'positive_job_changed')
    require(operation['config_ref'] == record['config_ref'] and operation['action'] == action
            and operation['status'] == 'actual_exit_recorded' and operation['actual_Popen_exit'] == 0
            and operation['scope_result_eligible'] is True and operation['reason'] is None
            and operation['remaining_captured_identities'] == []
            and operation['unknown_operation_group_identities'] == [], 'positive_actual_custody_missing')
    require(stdout['status'] == 'ok' and stdout['job_sha256'] == key
            and stdout['official_tests_opened'] is False, 'positive_worker_result_missing')
    if action == 'kind-one':
        kind = stdout['kind_map']; job = jobs[key]
        require(hashlib.sha256(canonical(kind)).hexdigest() == stdout['map_sha256']
                and kind['worker_key'] == job['worker_key'] and kind['train_sha256'] == job['train_sha256']
                and kind['rule_sha256'] == job['kind_rule_sha256']
                and kind['producer_source_sha256'] == job['producer_source_sha256']
                and kind['rule'] == lock['kind_rule'] and kind['official_tests_opened'] is False
                and kind['validation_or_synthetic_used_for_types'] is False
                and type(kind['dimensions']) is int and kind['dimensions'] == len(kind['column_kinds'])
                and set(kind['column_kinds']) <= {'categorical', 'continuous'}, 'kind_map_changed')
    return action, key, stdout
