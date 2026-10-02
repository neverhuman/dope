"""Use the unchanged author wrapper only after before-import runtime checks."""
from pathlib import Path
import json
import math
import os
import socket
import sys
import time
import hashlib
import importlib.util

common_path = Path(__file__).with_name('common.py')
assert not common_path.is_symlink() and not any(p.is_symlink() for p in common_path.parents)
assert hashlib.sha256(common_path.read_bytes()).hexdigest() == sys.argv[-1]
spec = importlib.util.spec_from_file_location('common', common_path)
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)

def run():
    mode, request_path, expected_round = sys.argv[1:-1]
    lock = common.check(expected_round)
    assert sys.flags.no_site and sys.flags.isolated and sys.flags.dont_write_bytecode
    common.verify_runtime(lock)
    request = common.read(common.safe(request_path)); job = request['job']
    assert job in lock['jobs'] and job['method'] == 'TabDDPM' and job['final'] is False
    common.check_worker(job)
    host = socket.gethostname().split('.')[0]
    assert host in common.HOSTS
    output = common.safe(request['receipt']); assert output.is_relative_to(common.ROOT)
    parent = output.parent; assert parent == common.ROOT/'attempts'/common.digest(job)/'attempt-0001'
    assert set(os.sched_getaffinity(0)) == set(lock['cpu_slots'][host])
    if mode in ('fit','sample'):
        assert host in ('xbabe1','xbabe3')
        snapshot = common.load_file('frozen_inventory',common.ROOT/'source/inventory_hosts.py').local()
        assert not snapshot['active_gpu_processes'] and snapshot['gpus'][0]['memory_free_mib'] >= 17*1024
    else: assert mode == 'native' and host == 'xbabe2'
    # The pinned interpreter's isolated startup cannot import site initializers.
    sys.path.insert(0,lock['environment_site'])
    entry = common.load_file('original_entry',lock['original_entry'])
    adapter = entry.check_all(); adapter.check()
    if mode == 'fit':
        assert request['artifact'] == str(parent/'artifact')
        import torch
        assert torch.cuda.is_available()
        torch.set_num_threads(16)
        torch.cuda.set_per_process_memory_fraction(lock['gpu_vram_cap_bytes']/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        result = adapter.fit(request)
        assert result['trained_gpu'] is True and result['peak_gpu_allocated_bytes'] <= lock['gpu_vram_cap_bytes']
    elif mode == 'sample':
        assert request['artifact'] == str(parent/'artifact') and request['seed'] in lock['sample_seeds']
        assert request['rows'] == job['worker']['cohort']['train_rows'] and request['device'] == 'cuda'
        assert common.safe(request['output']).parent == parent
        result = adapter.sample(request)
    else:
        original = common.load_file('original_native_coordinator',lock['original_native_coordinator'])
        prior = common.read(lock['original_round'])
        native_lock = prior | {'worker':job['worker'],'sample_seeds':lock['sample_seeds']}
        # Frozen native code uses its own five synthetic seeds and original
        # author CatBoost configuration. It has no shared-evaluator selection.
        start = time.monotonic(); original.native(parent,native_lock)
        result = common.read(parent/'native-kpi.json')
        assert math.isfinite(result['value']) and len(result['components']) == 5
        result |= {'implementation_sha256':lock['native_objective']['implementation_sha256'],
                   'validation_sha256':job['worker']['files']['validation.csv'],
                   'native_seconds':time.monotonic()-start}
    result |= {'round_sha256':expected_round,'job_sha256':common.digest(job),'host':host,
               'official_tests_opened':False,'mfs_v2':None,'ptf_v1':None}
    common.once(output,result)

if __name__ == '__main__': run()
