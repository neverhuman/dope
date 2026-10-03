"""Check declared ELF inputs from a trusted coordinator; never load them."""
import hashlib
import json
from pathlib import Path

from . import container_elf_inputs as elf
from . import container_python_custody as python


def require(condition):
    if not condition:
        raise ValueError('compressed ELF file custody rejected')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def owned_manifest(path, expected):
    path = python.scratch(path)
    expected = python.frozen_digest(expected)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected)
    value = json.loads(data)
    require(type(value) is dict)
    return path, expected, value


def selected_inputs(runtime, auxiliary):
    inputs = [(python.PYTHON, runtime['python_sha256'], 'interpreter', None)]
    for manifest in (runtime, auxiliary):
        root = python.scratch(manifest['site'])
        for name, row in manifest['files'].items():
            if name.endswith('.so') or '.so.' in Path(name).name:
                inputs.append((root / name, row['sha256'], 'package-extension', None))
    for name, row in auxiliary['stdlib_all_files'].items():
        if name.endswith('.so') or '.so.' in Path(name).name:
            inputs.append((Path(name), row['sha256'], 'stdlib-extension',
                           auxiliary['stdlib_aliases'].get(name)))
    return inputs


def verify_declared_elf_inputs(declarations_path, declarations_sha256,
                              runtime_path, runtime_sha256, auxiliary_path, auxiliary_sha256):
    """Recheck frozen files and declarations, not actual loader resolution.

    The caller must already trust this coordinator and its complete source and
    interpreter closure. This does not start a candidate or admit a job.
    """
    try:
        manifest, runtime, auxiliary = [owned_manifest(path, expected) for path, expected in
            [(declarations_path, declarations_sha256), (runtime_path, runtime_sha256),
             (auxiliary_path, auxiliary_sha256)]]
        lock = manifest[2]
        require(lock['format'] == 'dope-shared-validation-system-input-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        require(lock['runtime_path'] == str(runtime[0]) and lock['runtime_sha256'] == runtime[1])
        require(lock['auxiliary_path'] == str(auxiliary[0]) and lock['auxiliary_sha256'] == auxiliary[1])
        for key in ('official_tests_opened', 'loader_resolution_verified',
                    'system_library_file_inventory_complete', 'full_runtime_closure_certified',
                    'cross_host_runtime_identity_verified', 'sdv_predecessor_closed',
                    'density_priority_satisfied', 'capacity_admitted', 'execution_admitted',
                    'selection_performed'):
            require(lock[key] is False)
        require(all(lock[key] is None for key in
                    ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        for key in ('candidate_libraries_loaded', 'generator_fits', 'auditor_fits', 'samples_generated', 'gaps'):
            require(type(lock[key]) is int and lock[key] == 0)
        helper = Path(elf.__file__)
        helper_sha = python.frozen_digest(lock['helper_sha256'])
        require(python.sha(python.unaliased(helper)) == helper_sha)
        python.verify_import_inputs(str(runtime[0]), runtime[1], str(auxiliary[0]), auxiliary[1])
        inputs = selected_inputs(runtime[2], auxiliary[2])
        require(type(lock['objects']) is list and len(lock['objects']) == len(inputs))
        require(len({str(path) for path, *_ in inputs}) == len(inputs))
        rows = {row['path']: row for row in lock['objects']}
        require(len(rows) == len(inputs) and set(rows) == {str(path) for path, *_ in inputs})
        for key in ('declaration_objects', 'declarations_recorded'):
            require(type(lock[key]) is int and lock[key] == len(inputs))
        for path, expected, group, alias in inputs:
            blob = path.read_bytes()
            expected = python.frozen_digest(expected)
            require(hashlib.sha256(blob).hexdigest() == expected)
            actual = {'path': str(path), 'group': group, 'expected_sha256': expected,
                      'declared_alias': alias, 'bytes': len(blob), 'file_hash_verified': True,
                      'status': 'declarations_recorded',
                      'declarations': elf.inspect_elf_inputs(blob, expected)}
            require(canonical(rows[str(path)]) == canonical(actual))
        # Recheck complete trees and aliases after all owned-byte parsing.
        python.verify_import_inputs(str(runtime[0]), runtime[1], str(auxiliary[0]), auxiliary[1])
        for path, expected, _ in (manifest, runtime, auxiliary):
            require(python.sha(python.unaliased(path)) == expected)
        require(python.sha(python.unaliased(helper)) == helper_sha)
        return {'format': 'dope-compressed-declared-elf-file-custody', 'version': 1,
                'declarations_sha256': manifest[1], 'runtime_sha256': runtime[1],
                'auxiliary_sha256': auxiliary[1], 'declared_elf_files_verified': len(inputs),
                'python_import_inputs_verified': True, 'loader_resolution_verified': False,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'candidate_processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed ELF file custody rejected') from None
