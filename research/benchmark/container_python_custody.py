"""Check frozen Python import inputs without starting the candidate interpreter."""
import hashlib
import json
from pathlib import Path
import re

BASE = Path('/mnt/fast-scratch/dope-benchmark')
PYTHON = Path('/usr/bin/python3.12')
PYTHON_ALIAS = Path('/usr/bin/python3')
STDLIB = Path('/usr/lib/python3.12')
ABSENT_ZIP = Path('/usr/lib/python312.zip')


def require(condition):
    if not condition:
        raise ValueError('compressed Python custody rejected')


def frozen_digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None)
    return value


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def unaliased(path):
    require(path.is_absolute())
    require(not any(p.is_symlink() for p in (path, *path.parents)))
    require(path.resolve(strict=True) == path)
    return path


def scratch(path):
    require(type(path) is str)
    result = unaliased(Path(path))
    require(result.is_relative_to(BASE) and result != BASE)
    return result


def file_identity(path, row):
    require(type(row) is dict and type(row.get('bytes')) is int and row['bytes'] >= 0)
    expected = frozen_digest(row.get('sha256'))
    require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == expected)


def package_tree(root, files):
    require(root.is_dir() and type(files) is dict and bool(files))
    paths = list(root.rglob('*'))
    require(not any(path.is_symlink() for path in paths))
    require({path.relative_to(root).as_posix() for path in paths if path.is_file()} == set(files))
    for name, row in files.items():
        require(type(name) is str and Path(name).as_posix() == name)
        require(not Path(name).is_absolute() and '..' not in Path(name).parts)
        file_identity(unaliased(root / name), row)


def stdlib_tree(files, aliases):
    unaliased(STDLIB)
    require(STDLIB.is_dir() and type(files) is dict and bool(files) and type(aliases) is dict)
    paths = list(STDLIB.rglob('*'))
    require({str(p) for p in paths if p.is_file()} == set(files))
    require({str(p) for p in paths if p.is_symlink()} == set(aliases))
    for name, row in aliases.items():
        path = Path(name)
        require(path.is_relative_to(STDLIB) and path.is_symlink() and not path.is_dir())
        require(type(row) is dict and type(row.get('target')) is str and type(row.get('resolved')) is str)
        require(str(path.readlink()) == row['target'] and str(path.resolve(strict=True)) == row['resolved'])
        require(sha(path) == frozen_digest(row.get('sha256')))
    for name, row in files.items():
        path = Path(name)
        require(path.is_relative_to(STDLIB))
        for parent in path.parents:
            require(not parent.is_symlink())
        if not path.is_symlink():
            unaliased(path)
        file_identity(path, row)


def verify_import_inputs(runtime_path, runtime_sha256, auxiliary_path, auxiliary_sha256):
    """External digests select manifests; checking files grants no job admission.

    Call from an already trusted coordinator before starting the worker Python.
    This never imports an auditor, executes code, or verifies system ELF closure.
    """
    try:
        expected_runtime = frozen_digest(runtime_sha256)
        expected_auxiliary = frozen_digest(auxiliary_sha256)
        runtime_file, auxiliary_file = scratch(runtime_path), scratch(auxiliary_path)
        require(sha(runtime_file) == expected_runtime and sha(auxiliary_file) == expected_auxiliary)
        runtime = json.loads(runtime_file.read_text())
        auxiliary = json.loads(auxiliary_file.read_text())
        require(type(runtime) is dict and type(auxiliary) is dict)
        require(auxiliary['runtime_sha256'] == expected_runtime)
        require(auxiliary['runtime_path'] == str(runtime_file))
        require(runtime['python'] == str(PYTHON) and runtime['absent_zip'] == str(ABSENT_ZIP))
        unaliased(PYTHON)
        require(sha(PYTHON) == frozen_digest(runtime['python_sha256']))
        unaliased(PYTHON_ALIAS.parent)
        require(PYTHON_ALIAS.is_symlink() and type(runtime['python_symlink']) is str)
        require(str(PYTHON_ALIAS.readlink()) == runtime['python_symlink'])
        require(PYTHON_ALIAS.resolve(strict=True) == PYTHON)
        require(not ABSENT_ZIP.exists() and not ABSENT_ZIP.is_symlink())
        cache = scratch(runtime['empty_bytecode_cache'])
        require(cache.is_dir() and not list(cache.rglob('*')))
        for manifest in (runtime, auxiliary):
            package_tree(scratch(manifest['site']), manifest['files'])
        stdlib_tree(auxiliary['stdlib_all_files'], auxiliary['stdlib_aliases'])
        metric = scratch(runtime['metric_source'])
        require(sha(metric) == frozen_digest(runtime['metric_sha256']))
        require(sha(runtime_file) == expected_runtime and sha(auxiliary_file) == expected_auxiliary)
        return {'format': 'dope-compressed-python-import-input-custody', 'version': 1,
                'runtime_sha256': expected_runtime, 'auxiliary_sha256': expected_auxiliary,
                'package_files': len(runtime['files']) + len(auxiliary['files']),
                'stdlib_files': len(auxiliary['stdlib_all_files']),
                'python_import_inputs_verified': True, 'full_runtime_closure_certified': False,
                'execution_admitted': False, 'processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed Python custody rejected') from None
