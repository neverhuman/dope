"""Replay stored runtime file declarations; make no loaded-library claim."""
import hashlib
import os
from pathlib import Path
import stat

from research.benchmark import density_publication_inputs as inputs
from research.benchmark import density_metric_replay as metrics


def files_under(root, files):
    """Strict regular file inventory, including bytecode; no aliases or specials."""
    root = inputs.safe(root)
    selected = {Path(p): row for p, row in files.items() if Path(p).is_relative_to(root)}
    actual = set()
    for path in root.rglob('*'):
        mode = path.lstat().st_mode
        inputs.require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'runtime alias or special')
        if stat.S_ISREG(mode): actual.add(path)
    inputs.require(actual == set(selected), 'runtime regular file inventory changed')
    refs = {}
    for path, row in selected.items():
        inputs.safe(path)
        inputs.require(type(row['bytes']) is int and row['bytes'] >= 0
                       and path.stat().st_size == row['bytes'], 'runtime byte charge differs')
        inputs.digest_string(row['sha256'])
        refs[str(path)] = row['sha256']
    inputs.verify_refs(refs)
    return refs


def rehash_system_file(path, expected, size=None):
    """Check fixed manifest-owned system paths without invoking their code."""
    inputs.digest_string(expected)
    path = Path(path)
    resolved = path.resolve(strict=True)
    inputs.require(stat.S_ISREG(resolved.stat().st_mode), 'system file not regular')
    if size is not None:
        inputs.require(type(size) is int and size >= 0 and path.stat().st_size == size,
                       'system file byte count changed')
    h = hashlib.sha256()
    with resolved.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    inputs.require(h.hexdigest() == expected, 'system file digest changed')


def replay(lock):
    prep_path = Path(lock['density_runtime_preparation_path'])
    prep = inputs.bound_json(prep_path, lock['density_runtime_preparation_sha256'])
    inventory_path = prep_path.parent / 'runtime-inventory.lock.json'
    inv = inputs.bound_json(inventory_path, prep['runtime_inventory_sha256'])
    python = inputs.safe(prep['python'])
    prefix = python.parents[1]
    inputs.require(python.name == 'python3.12' and python.parent.name == 'bin'
        and prep['stdlib_root'] == str(prefix / 'lib/python3.12')
        and prep['stdlib_dynload_root'] == str(prefix / 'lib/python3.12/lib-dynload')
        and prep['environment_site'] == str(prefix / 'lib/python3.12/site-packages')
        and prep['package_root'] == str(prep_path.parent / 'package')
        and prep['absent_zip'] == str(prefix / 'lib/python312.zip')
        and inv['roots'] == [str(prefix), prep['package_root']]
        and inv['aliases'] == []
        and set(inv['inventory_modes']) == set(inv['roots'])
        and all(v == 'all' for v in inv['inventory_modes'].values()), 'sampler runtime declaration differs')
    inputs.require(str(python) in inv['files']
        and all(str(prefix / 'lib/python3.12' / n) in inv['files']
                for n in ('os.py', 'encodings/__init__.py'))
        and inv['files'][str(Path(prep['package_root']) / 'research/benchmark/adapters.py')]['sha256']
        == prep['source_adapter_sha256'] == lock['adapter_sha256']
        and prep['environment'] == {'copulas': '0.14.1', 'numpy': '1.26.4',
                                   'pandas': '2.1.4', 'scipy': '1.16.3'}, 'sampler source differs')
    refs = {str(inventory_path): prep['runtime_inventory_sha256']}
    for root in inv['roots']: refs.update(files_under(root, inv['files']))
    inputs.require(set(inv['files']) == {p for p in refs if p != str(inventory_path)},
                   'unassigned sampler inventory file')
    inputs.require(not os.path.lexists(prep['absent_zip']), 'sampler archive appeared')

    runtime = inputs.bound_json(metrics.RUNTIME_PATH, metrics.RUNTIME_SHA256)
    auxiliary_path = inputs.BASE / 'shared-validation-python312-auxiliary-v1/auxiliary.lock.json'
    auxiliary_sha = lock['runtime_lock_files'][str(auxiliary_path)]
    aux = inputs.bound_json(auxiliary_path, auxiliary_sha)
    inputs.require(aux['runtime_sha256'] == metrics.RUNTIME_SHA256
                   and runtime['metric_sha256'] == metrics.METRIC_SHA256
                   == lock['metric_source_sha256'], 'shared runtime declarations differ')
    for row in (runtime, aux):
        site = inputs.safe(row['site'])
        absolute = {str(site / name): item for name, item in row['files'].items()}
        refs.update(files_under(site, absolute))
    library = Path('/usr/lib/python3.12')
    actual_files, actual_aliases = set(), set()
    for path in library.rglob('*'):
        mode = path.lstat().st_mode
        inputs.require(stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode),
                       'system runtime special file')
        if stat.S_ISLNK(mode): actual_aliases.add(str(path))
        if path.is_file(): actual_files.add(str(path))
    inputs.require(actual_files == set(aux['stdlib_all_files'])
                   and actual_aliases == set(aux['stdlib_aliases']), 'system import inventory changed')
    for name, row in aux['stdlib_all_files'].items():
        rehash_system_file(name, row['sha256'], row['bytes'])
    for name, row in aux['stdlib_aliases'].items():
        path = Path(name)
        inputs.require(str(path.readlink()) == row['target']
                       and str(path.resolve(strict=True)) == row['resolved'], 'system alias changed')
        rehash_system_file(path, row['sha256'])
    inputs.require(runtime['python'] == '/usr/bin/python3.12'
                   and str(Path('/usr/bin/python3').readlink()) == runtime['python_symlink']
                   and runtime['absent_zip'] == '/usr/lib/python312.zip'
                   and not os.path.lexists(runtime['absent_zip']), 'system interpreter declaration differs')
    rehash_system_file(runtime['python'], runtime['python_sha256'])
    inputs.require(not any(inputs.safe(runtime['empty_bytecode_cache']).iterdir()),
                   'shared bytecode cache not empty')
    inputs.verify_refs(refs)
    return dict(scratch_runtime_refs=refs, scratch_files_hashed=len(refs),
                system_import_files_hashed=len(actual_files), system_aliases_checked=len(actual_aliases),
                stored_file_inventory_replayed=True, dependencies_initialized=False,
                sampler_or_learner_invoked=False, system_dynamic_library_closure_certified=False,
                historical_full_runtime_closure_upgraded=False, publication_admitted=False)
