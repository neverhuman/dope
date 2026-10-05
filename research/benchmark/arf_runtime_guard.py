"""Check all owned ARF import providers before loading third-party dependencies."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys


def require(ok, reason):
    if not ok: raise ValueError(reason)


def digest(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'frozen runtime digest required')


def safe(path, base):
    p = Path(path)
    require(p.is_absolute() and p == p.resolve(strict=True) and p.is_relative_to(base)
            and not any(q.is_symlink() for q in (p, *p.parents)), 'runtime path not owned')
    return p


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def bound(path, expected, base):
    digest(expected); p = safe(path, base)
    require(stat.S_ISREG(p.stat().st_mode), 'runtime manifest not regular')
    data = p.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'runtime manifest changed')
    return decode(data)


def pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, 'duplicate ARF JSON key')
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=pairs,
                      parse_float=finite_float,
                      parse_constant=lambda _: require(False, 'nonfinite ARF JSON'))


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), 'nonfinite ARF JSON')
    return result


def inventory(root, files, base, directories):
    root = safe(root, base); declared = {Path(p) for p in files}
    require(root.is_dir() and declared and all(p.is_relative_to(root) and p != root for p in declared),
            'runtime inventory invalid')
    expected_dirs = {Path(p) for p in directories}
    require(all(p != root and p.is_relative_to(root) for p in expected_dirs),
            'runtime directory inventory invalid')
    actual, actual_dirs = set(), set()
    for p in root.rglob('*'):
        mode = p.lstat().st_mode
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), 'runtime alias or special entry')
        if stat.S_ISREG(mode): actual.add(p)
        else: actual_dirs.add(p)
    require(actual == declared and actual_dirs == expected_dirs, 'runtime file inventory changed')
    for path, row in files.items():
        p = safe(path, base); digest(row['sha256'])
        require(type(row['bytes']) is int and row['bytes'] >= 0 and p.stat().st_size == row['bytes']
                and hash_file(p) == row['sha256'], 'runtime provider changed')


def verify(path, expected, base):
    """No third-party imports; return only a fully checked import path declaration."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'ARF CPU environment required')
    lock = bound(path, expected, base)
    require(lock['format'] == 'dope-arf-pandas-compatible-runtime-preparation'
            and type(lock['version']) is int and lock['version'] == 2
            and type(lock['new_generator_fits_started']) is int and lock['new_generator_fits_started'] == 0
            and lock['official_tests_opened'] is False and lock['execution_admitted'] is False
            and lock['author_commit'] == '8b63c1b3999981125b4af2828ff52cba8e29169d',
            'ARF runtime preparation declaration differs')
    inventories = []
    for ref in lock['inventory_locks']:
        inv = bound(ref['path'], ref['sha256'], base)
        inventories.append(inv)
        for root in inv['roots']:
            selected = {p: row for p, row in inv['files'].items() if Path(p).is_relative_to(root)}
            inventory(root, selected, base, lock['directory_inventories'][root])
        require(not inv.get('aliases'), 'ARF preparation aliases are not admitted')
        require(set(inv['files']) == {p for p in inv['files']
            if any(Path(p).is_relative_to(root) for root in inv['roots'])}, 'unassigned runtime file')
    source = safe(lock['source_root'], base)
    module = safe(__file__, base)
    require(module.is_relative_to(source)
            and lock['source_files'].get(str(module), {}).get('sha256') == hash_file(module),
            'executing ARF guard is not the frozen source')
    inventory(source, lock['source_files'], base, lock['directory_inventories'][str(source)])
    python = safe(lock['python'], base); digest(lock['python_sha256'])
    require(hash_file(python) == lock['python_sha256']
            and Path(sys.executable).resolve(strict=True) == python,
            'resolved ARF interpreter differs')
    prefix = python.parents[1]
    require(lock['stdlib_root'] == str(prefix / 'lib/python3.12')
            and lock['stdlib_dynload_root'] == str(prefix / 'lib/python3.12/lib-dynload')
            and lock['absent_zip'] == str(prefix / 'lib/python312.zip')
            and not os.path.lexists(lock['absent_zip']), 'interpreter import chain differs')
    cache = safe(lock['empty_cache'], base)
    require(not list(cache.rglob('*')) and sys.pycache_prefix == str(cache)
            and sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode,
            'isolated empty-cache interpreter required')
    paths = [str(source), *lock['package_sites'], lock['stdlib_root'], lock['stdlib_dynload_root']]
    declared = {Path(root) for inv in inventories for root in inv['roots']}
    require(all(safe(p, base).is_dir() and any(Path(p).is_relative_to(root) for root in declared)
                for p in paths[1:]), 'import provider not inventoried')
    sys.path[:] = paths
    return lock


def probe(path, expected, base):
    lock = verify(path, expected, base)
    import numpy, pandas, scipy, sklearn
    from arfpy.arf import arf
    versions = dict(numpy=numpy.__version__, pandas=pandas.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__)
    require(versions == lock['expected_versions'] and callable(arf), 'ARF dependency contract differs')
    print(json.dumps(dict(versions=versions, author_class_imported=True, generator_fits_started=0,
        official_tests_opened=False, historical_runtime_closure_upgraded=False,
        system_dynamic_library_closure_certified=False)), flush=True)
