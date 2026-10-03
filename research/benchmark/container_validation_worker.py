"""CPU research worker for compressed members; requires a separately frozen round."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import types


def require(condition):
    if not condition:
        raise ValueError('compressed research worker integrity failed')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe(path, base):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)))
    require(path.resolve(strict=True).is_relative_to(base))
    require('evaluator' not in path.parts and path.name != 'test.csv')
    return path


def load(name, path):
    require(name not in sys.modules)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def prepare(root, expected, request):
    require(socket.gethostname().split('.')[0] in ('xbabe1', 'xbabe2', 'xbabe3'))
    require(sys.dont_write_bytecode is True)
    root = Path(root)
    base = root.parent
    safe(root, base)
    request = safe(request, base)
    require(request.name == 'request.json' and request.is_relative_to(root / 'attempts'))
    require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected) is not None)
    require(sha(safe(root / 'round.lock.json', base)) == expected)
    lock = read(root / 'round.lock.json')
    require(lock['execution_admitted'] is True and lock['execution_lock_frozen'] is True)
    require(lock['clean_sdv_predecessor_closed'] is True and lock['density_priority_satisfied'] is True)
    require(lock['official_tests_opened'] is False and lock['gpu_operations_enabled'] is False)
    require(lock['global_family_selected'] is False and lock['gate_profile_complete'] is False)
    require(all(lock[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    require(os.environ.get('LD_LIBRARY_PATH') == lock['native_library_directory'])
    source = safe(root / 'source', base)
    require(not any(p.is_symlink() for p in source.rglob('*')))
    require({str(p) for p in source.rglob('*') if p.is_file()} == set(lock['source_files']))
    required = {'entry.py', 'common.py', 'shared_runtime.py', 'native_runtime.py',
                'research_container.py', 'research_container_members.py'}
    require({p.name for p in source.iterdir()} == required)
    for path, expected_hash in lock['source_files'].items():
        require(sha(safe(path, base)) == expected_hash)
    require(sha(source / 'entry.py') == sha(__file__))
    request_blob = request.read_bytes()
    req = json.loads(request_blob)
    job = req['job']
    frozen_jobs = {digest(row) for row in lock['jobs']}
    require(len(frozen_jobs) == len(lock['jobs']))
    require(req['round_sha256'] == expected and digest(job) in frozen_jobs)
    require(re.fullmatch('attempt-[0-9]{4}', request.parent.name) is not None)
    require(request.parent.parent.name == digest(job))
    require(request.parent.parent.parent == root / 'attempts')
    require(request.parent.stat().st_uid == os.getuid() and request.parent.stat().st_mode & 0o777 == 0o700)
    require(job['final'] is False and job['track'] == 'common-numeric')
    require(job['split'] == 'official_training_derived_validation' and type(job['fit_seed']) is int and job['fit_seed'] == 11)
    require(job['sample_seeds'] == [101, 211, 307] and job['size_multipliers'] == [1, 4])
    require(all(type(v) is int for v in (*job['sample_seeds'], *job['size_multipliers'])))
    require(job['original_generator_binary_sha256'] == lock['gpu_binary_sha256'])
    for path, expected_hash in lock['frozen_references'].items():
        require(sha(safe(path, base)) == expected_hash)
    parent = safe(lock['fit_receipt_lock_path'], base)
    require(sha(parent) == lock['fit_receipt_lock_sha256'])
    for path, expected_hash in read(parent)['refs'].items():
        require(sha(safe(path, base)) == expected_hash)
    worker = safe(job['worker']['path'], base)
    require(set(job['worker']['files']) == {'train.csv', 'validation.csv', 'projection.json',
                                         'row-group-assignments.json', 'worker-manifest.json'})
    require({p.name for p in worker.iterdir()} == set(job['worker']['files']))
    for name, expected_hash in job['worker']['files'].items():
        require(sha(safe(worker / name, base)) == expected_hash)
    require(type(job['worker']['train_rows']) is int and job['worker']['train_rows'] > 0)
    require(job['expected_original_projection_sha256'] == job['worker']['files']['projection.json'])
    original = safe(job['original_model_path_for_sampling_replay'], base)
    require(sha(original) == job['expected_original_model_sha256'])
    artifact = safe(job['compressed_artifact_path'], base)
    with artifact.open('rb') as stream:
        blob = stream.read(10_241)
    # Sources and external parent/data identities precede loading even the codec.
    common = load('common', source / 'common.py')
    common.check(expected)
    package_name = '_dope_container_validation'
    require(package_name not in sys.modules)
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    package.research_container = load(package_name + '.research_container', source / 'research_container.py')
    members_module = load(package_name + '.research_container_members', source / 'research_container_members.py')
    members = members_module.decode_verified_members(blob,
        artifact_sha256=job['compressed_artifact_sha256'], artifact_bytes=job['compressed_artifact_bytes'],
        model_sha256=job['expected_original_model_sha256'],
        projection_sha256=job['expected_original_projection_sha256'])
    require(json.loads(members.projection)['task'] == 'regression')
    for path, expected_hash in lock['runtime_lock_files'].items():
        require(sha(safe(path, base)) == expected_hash)
    shared = load('shared_runtime', source / 'shared_runtime.py').verify()
    require(shared['metric_sha256'] == lock['metric_source_sha256'])
    native = load('native_runtime', source / 'native_runtime.py')
    require(native.verify_runtime(lock)['torch_library_directory'] == lock['native_library_directory'])
    return lock, job, worker, original, members, shared, native, hashlib.sha256(request_blob).hexdigest()


def once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def sample(lock, native, kernel, rows, seed, output, start):
    require(not output.exists() and not output.is_symlink())
    native.verify_runtime(lock)
    remaining = 600 - (time.monotonic() - start)
    require(remaining > 0)
    with output.with_suffix('.log').open('x') as log:
        subprocess.run([lock['gpu_binary'], 'sample', '--kernel', str(kernel), '--rows', str(rows),
                        '--seed', str(seed), '--out', str(output)], cwd=output.parent,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
    safe(output, output.parent)


def run(root, expected, request):
    start = time.monotonic()
    lock, job, worker, original, members, runtime, native, request_hash = prepare(root, expected, request)
    out = Path(request).parent
    owned = out / 'decoded-members'
    owned.mkdir(mode=0o700)
    kernel = owned / 'model.dpk'
    projection = owned / 'projection.json'
    for path, body in ((kernel, members.model), (projection, members.projection)):
        with path.open('xb') as stream:
            stream.write(body)
        path.chmod(0o400)
    metrics = load('population_shared_metric', Path(runtime['metric_source']))
    samples = []
    for size in job['size_multipliers']:
        for seed in job['sample_seeds']:
            require(sha(safe(kernel, out)) == job['expected_original_model_sha256'])
            require(sha(safe(projection, out)) == job['expected_original_projection_sha256'])
            rows = job['worker']['train_rows'] * size
            name = f'n{size}-seed{seed}'
            primary = out / (name + '.csv')
            sample(lock, native, kernel, rows, seed, primary, start)
            primary_hash = sha(primary)
            require(sha(safe(original, Path(root).parent)) == job['expected_original_model_sha256'])
            replay = out / (name + '.original.csv')
            sample(lock, native, original, rows, seed, replay, start)
            require(sha(replay) == primary_hash)
            if size == 1 and seed == 101:
                repeat = out / (name + '.repeat.csv')
                sample(lock, native, kernel, rows, seed, repeat, start)
                require(sha(repeat) == primary_hash)
            value = metrics.measure(worker / 'train.csv', worker / 'validation.csv', primary, 'regression', seed)
            require(value['dependencies'] == runtime['expected_versions'])
            require(value['rows']['synthetic'] == rows and value['mfs_v2'] is None)
            require(value['gate_profile_complete'] is False and set(value['utility']) == {'linear', 'catboost', 'mlp'})
            if size == 4 and seed == 101:
                replayed_metric = metrics.measure(worker / 'train.csv', worker / 'validation.csv', primary, 'regression', seed)
                require({k: v for k, v in value.items() if k != 'metric_seconds'} ==
                        {k: v for k, v in replayed_metric.items() if k != 'metric_seconds'})
                once(out / (name + '.metric-replay.json'), replayed_metric)
            require(sha(primary) == primary_hash and time.monotonic() - start < 600)
            once(out / (name + '.metric.json'), value)
            samples.append({'size_multiplier': size, 'sample_seed': seed, 'rows': rows,
                            'sample_file': primary.name, 'sample_sha256': primary_hash,
                            'metric_file': name + '.metric.json', 'metric_sha256': sha(out / (name + '.metric.json')),
                            'original_kernel_replay': 'exact',
                            'metric_replay': 'exact' if size == 4 and seed == 101 else 'not_repeated',
                            'sample_replay': 'exact' if size == 1 and seed == 101 else 'not_repeated'})
    for name, expected_hash in job['worker']['files'].items():
        require(sha(safe(worker / name, Path(root).parent)) == expected_hash)
    require(sha(safe(job['compressed_artifact_path'], Path(root).parent)) == members.artifact_sha256)
    require(sha(safe(kernel, out)) == job['expected_original_model_sha256'])
    require(sha(safe(projection, out)) == job['expected_original_projection_sha256'])
    require(sha(safe(request, Path(root).parent)) == request_hash)
    elapsed = time.monotonic() - start
    if elapsed >= 600:
        raise TimeoutError('compressed research batch deadline exhausted')
    once(out / 'batch.json', {'job_sha256': digest(job), 'round_sha256': expected,
        'request_sha256': request_hash, 'metric_source_sha256': lock['metric_source_sha256'],
        'samples': samples, 'artifact_bytes': members.artifact_bytes,
        'artifact_sha256': members.artifact_sha256, 'original_kernel_replays_exact': True,
        'new_generator_fits_started': 0, 'native_selection_changed': False, 'global_family_selected': False,
        'official_tests_opened': False, 'gate_profile_complete': False, 'mfs_v2': None, 'ptf_v1': None,
        'elapsed_seconds': elapsed})


if __name__ == '__main__':
    run(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
