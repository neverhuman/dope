"""A source-bound per-operation grant, separate from immutable runtime custody.

The reviewed runner owns fresh capacity RPC, CUDA candidate closure, source and
worker custody, the whole operation watchdog, and the inherited GPU lease.
The public adapter retains its generated CPU contract when no grant is supplied.
"""
import fcntl
import hashlib
from importlib.machinery import SourceFileLoader
import math
import json
import os
from pathlib import Path
import stat
import time
import types

from research.benchmark.tabsyn_runtime_guard import bound, require, safe

AUTHOR = 'cb5ac0f74ec36ee88e7a974a393dfbef50d42da7'
LEASE = '/home/ubuntu/dope-scratch-x3/.gpu-fit.lock'
OWNER_ONE_POLICY_SHA256 = '3b5e46786727a6f09938ea4276c37070fae63824c7b1e6ff5b0ea8be88012f94'


class _VerifiedGuardLoader(SourceFileLoader):
    """Compile the captured, verified source buffer for module initialization."""

    def __init__(self, path, body):
        super().__init__('tabsyn_operation_guard', str(path))
        self.verified_body = body

    def get_code(self, fullname):
        require(fullname == self.name, 'TabSyn guard module identity differs')
        return compile(self.verified_body, self.path, 'exec',
                       dont_inherit=True, optimize=0)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def identity(pid):
    path = Path('/proc')/str(pid)
    tail = (path/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, uid=path.stat().st_uid, start_ticks=tail[19])


def held_lease(fd, path=LEASE):
    require(type(fd) is int and fd >= 3, 'TabSyn inherited lease descriptor differs')
    path = safe(path)
    expected, actual = path.stat(), os.fstat(fd)
    require(stat.S_ISREG(actual.st_mode) and actual.st_uid == os.getuid()
            and (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino),
            'TabSyn inherited lease identity differs')
    with path.open('a') as other:
        try:
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(other.fileno(), fcntl.LOCK_UN)
            require(False, 'TabSyn GPU lease was not held')
    # A foreign holder cannot make an unrelated open descriptor pass this check.
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        require(False, 'TabSyn inherited lease is not the holder')


def source_binding(code):
    selected={k:code[k] for k in ('runtime_sha256','source_commit','author_files')}
    selected['source_files']={k:v['sha256'] for k,v in code['inventory']['files'].items()}
    scope=code['numeric_population_scope']
    selected['epoch_policy']={k:scope[k] for k in ('epoch_budget','epoch_policy_sha256','diffusion_author_patience','config_sha256')}
    if 'operational_qualification_policy' in scope:
        selected['operational_qualification_policy_sha256'] = scope['operational_qualification_policy']['sha256']
    return hashlib.sha256(json.dumps(selected,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def qualification(scope, runtime_sha256, mode, code):
    count = 2
    qualification_format = 'dope-tabsyn-epoch-policy-generated-CUDA-two-replay-closure'
    policy_ref = scope.get('operational_qualification_policy')
    if policy_ref is not None:
        require(type(policy_ref) is dict and type(policy_ref.get('sha256')) is str
                and policy_ref['sha256'] == OWNER_ONE_POLICY_SHA256,
                'TabSyn owner qualification amendment differs')
        policy = bound(policy_ref['path'], OWNER_ONE_POLICY_SHA256)
        require(policy['format'] == 'dope-tabsyn-owner-operational-qualification-amendment'
                and policy['scope'] == 'work-order-B-tabsyn-scaled-100-lineage-research'
                and type(policy['required_actual_generated_CUDA_closures']) is int
                and policy['required_actual_generated_CUDA_closures'] == 1,
                'TabSyn owner qualification amendment differs')
        count = 1
        qualification_format = 'dope-tabsyn-owner-one-generated-CUDA-closure-v1'
    ref = scope['generated_cuda_qualification']
    q = bound(ref['path'], ref['sha256'])
    require(q['format'] == qualification_format
            and q['status'] == 'declared_epoch_fit_sample_replay_verified'
            and q['epoch_policy_sha256']==scope['epoch_policy_sha256']
            and q['epoch_budget']==scope['epoch_budget']
            and q['diffusion_author_patience']==scope['diffusion_author_patience']==500
            and type(q['diffusion_author_patience']) is int
            and q['config_sha256']==scope['config_sha256']
            and q['timeout_checkpoint_eligible'] is False
            and q['runtime_sha256'] == runtime_sha256
            and q['source_binding_sha256']==source_binding(code)
            and q['adapter_source_sha256']==code['inventory']['files']['research/benchmark/tabsyn_adapter.py']['sha256']
            and q['actual_outer_process_exit_verified'] is True
            and q['official_tests_opened'] is False and type(q['real_fits_started']) is int
            and q['real_fits_started'] == 0
            and q['scope'] == 'numeric_generated_CUDA_only',
            'TabSyn actual generated CUDA qualification missing')
    if policy_ref is not None:
        require(type(q.get('operational_qualification_policy_sha256')) is str
                and q['operational_qualification_policy_sha256'] == OWNER_ONE_POLICY_SHA256,
                'TabSyn generated owner qualification binding differs')
    # The reviewed guard revalidates the referenced receipts, their actual
    # exit proofs, source-bound qualifier, candidate bytes and mapped providers.
    require(type(q['input_receipt_refs']) is list and type(q['root_actual_exit_refs']) is list
            and len(q['input_receipt_refs']) == len(q['root_actual_exit_refs']) == count,
            'TabSyn generated replay closure differs')
    if mode == 'native':
        ref = scope['native_cuda_qualification']
        q = bound(ref['path'], ref['sha256'])
        require(q['format'] == 'dope-tabsyn-generated-native-CUDA-closure'
                and q['status'] == 'ok' and type(q['actual_exit_code']) is int
                and q['actual_exit_code'] == 0 and q['runtime_sha256'] == runtime_sha256
                and q['source_binding_sha256']==source_binding(code)
                and q['epoch_policy_sha256']==scope['epoch_policy_sha256']
                and q['native_adapter_sha256']==code['inventory']['files']['research/benchmark/tabsyn_adapter.py']['sha256']
                and q['author_mle_sha256'] == scope['author_mle_sha256']
                and q['TRAIN_target_helper_sha256'] == scope['TRAIN_target_helper_sha256']
                and type(q['auditor_configurations']) is int and q['auditor_configurations'] == 36
                and type(q['postselection_refits']) is int and q['postselection_refits'] == 4
                and q['actual_outer_process_exit_verified'] is True
                and q['official_tests_opened'] is False,
                'TabSyn actual native CUDA qualification missing')


def verify_operation(admission, mode, config_sha256, runtime_sha256, code_path, code_sha256,
                     phase='before_dependencies'):
    require(type(admission) is dict and set(admission) == {'operation_path', 'operation_sha256'}
            and mode in ('fit', 'sample', 'native')
            and phase in ('before_dependencies','final_integrity'), 'TabSyn operation grant required')
    code = bound(code_path, code_sha256)
    require(code['format']=='dope-tabsyn-adapter-controls' and type(code['version']) is int
            and code['version']==1 and code['runtime_sha256']==runtime_sha256
            and code['execution_admitted'] is False, 'TabSyn historical control custody differs')
    scope = code['numeric_population_scope']
    require(scope['source_commit'] == AUTHOR and scope['real_data_allowed'] is True
            and scope['partition'] == 'official_training_derived_validation'
            and scope['official_tests_opened'] is False
            and scope['gpu_lease_path'] == LEASE
            and scope['maximum_trials'] == 8 and type(scope['maximum_trials']) is int
            and scope['maximum_lineage_seconds'] == 43200
            and type(scope['maximum_lineage_seconds']) is int
            and scope['whole_operation_seconds'] == 600
            and type(scope['whole_operation_seconds']) is int
            and scope['gpu_vram_cap_bytes'] == 16*2**30
            and type(scope['gpu_vram_cap_bytes']) is int
            and type(scope['epoch_policy_sha256']) is str
            and len(scope['epoch_policy_sha256'])==64
            and all(c in '0123456789abcdef' for c in scope['epoch_policy_sha256'])
            and config_sha256 in scope['config_sha256'], 'TabSyn numeric population scope differs')
    qualification(scope, runtime_sha256, mode, code)
    root = safe(code['inventory']['root'])
    relative = 'source/tabsyn_operation_guard.py'
    path = safe(root/relative)
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == code['inventory']['files'][relative]['sha256'],
            'TabSyn operation guard source changed')
    # The loader consumes exactly the captured, hash-verified buffer. Its
    # get_code override excludes all source rereads and cached bytecode paths.
    guard = types.ModuleType('tabsyn_operation_guard'); guard.__file__ = str(path)
    _VerifiedGuardLoader(path, body).exec_module(guard)
    result = guard.admit_tabsyn_operation(admission['operation_path'], admission['operation_sha256'],
        mode, config_sha256, runtime_sha256, code_path, code_sha256, phase)
    require(result['phase']==phase and result['mode'] == mode and result['config_sha256'] == config_sha256
            and result['runtime_sha256'] == runtime_sha256 and result['code_sha256'] == code_sha256
            and result['operation_sha256'] == admission['operation_sha256']
            and result['actual_caller'] == identity(os.getpid())
            and result['controller'] == identity(os.getppid())
            and result['cpu_slot'] == list(range(80, 96))
            and sorted(os.sched_getaffinity(0)) == result['cpu_slot']
            and result['fresh_capacity_rpc_verified'] is True
            and result['current_candidate_bytes_and_providers_verified'] is True
            and result['worker_and_request_custody_verified'] is True
            and result['official_tests_opened'] is False, 'TabSyn actual operation identity differs')
    start, deadline = result['started_monotonic'], result['deadline_monotonic']
    require(finite(start) and finite(deadline) and 0 < deadline-start <= 600
            and start <= time.monotonic() < deadline, 'TabSyn whole operation deadline differs')
    count, charged = result['prior_fit_attempts'], result['charged_prior_lineage_seconds']
    require(type(count) is int and 0 <= count <= (7 if mode == 'fit' else 8)
            and finite(charged) and 0 <= charged < 43200
            and charged + deadline-start <= 43200
            and result['failed_fit_and_admission_wait_costs_included'] is True,
            'TabSyn carried lineage budget differs')
    epochs = result['epoch_budget']
    require(epochs == scope['epoch_budget'] and type(epochs) is dict
            and result['epoch_policy_sha256']==scope['epoch_policy_sha256']
            and set(epochs) == {'vae_epochs', 'diffusion_epochs'}
            and all(type(epochs[k]) is int and 0 < epochs[k] <= n for k, n in
                    [('vae_epochs', 4000), ('diffusion_epochs', 10001)]),
            'TabSyn frozen epoch budget differs')
    held_lease(result['inherited_gpu_lease_fd'])
    require(time.monotonic() < deadline, 'TabSyn admission exceeded operation deadline')
    return result
