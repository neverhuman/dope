"""Inspect owned loader configuration declarations, without searching providers."""
import hashlib
import fnmatch
import os
from pathlib import Path
import re

from . import container_loader_custody as loader


def require(condition):
    if not condition:
        raise ValueError('compressed loader configuration rejected')


def inspect_configuration(blob, expected_sha256):
    """Accept the observed ASCII absolute-directory/include declaration profile."""
    require(type(blob) is bytes and type(expected_sha256) is str)
    require(re.fullmatch('[0-9a-f]{64}', expected_sha256) is not None)
    require(len(blob) <= 1048576 and hashlib.sha256(blob).hexdigest() == expected_sha256)
    try:
        text = blob.decode('ascii')
    except UnicodeError:
        raise ValueError('compressed loader configuration rejected') from None
    require('\x00' not in text)
    records = []
    for ordinal, line in enumerate(text.split('\n')):
        value = line.split('#', 1)[0].strip()
        if not value:
            continue
        tokens = value.split()
        if tokens[0] == 'include':
            require(len(tokens) == 2)
            kind, path = 'include', tokens[1]
            require(not any(c in os.path.dirname(path) for c in '*?[]'))
            require(re.fullmatch('[A-Za-z0-9._*-]+', os.path.basename(path)) is not None)
        else:
            require(len(tokens) == 1)
            kind, path = 'directory', tokens[0]
            require(not any(c in path for c in '*?[]'))
        require(path.startswith('/') and not path.startswith('//') and '$' not in path)
        require(os.path.normpath(path) == path and '\\' not in path)
        records.append({'line': ordinal + 1, 'kind': kind, 'value': path})
        require(len(records) <= 4096)
    return {'format': 'dope-compressed-loader-configuration-declarations', 'version': 1,
            'configuration_sha256': expected_sha256, 'records': records,
            'include_expansion_verified': False, 'actual_loader_selection_verified': False,
            'complete_provider_search_verified': False, 'full_runtime_closure_certified': False,
            'execution_admitted': False, 'candidate_processes_started': 0,
            'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
            'release_safe': None, 'superiority': None}


def verify_configuration_inputs(search_path, search_sha256):
    """Bind this declaration profile to selected directory/absence snapshots.

    Call from a trusted coordinator after Python/root ELF custody. This does not
    establish actual cache construction, loader search order or execution.
    """
    try:
        manifest = loader.system.owned_manifest(search_path, search_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-shared-validation-loader-search-input-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('actual_loader_selection_verified', 'complete_provider_search_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            require(lock[key] is False)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        require(type(lock['candidate_processes_started']) is int and lock['candidate_processes_started'] == 0)
        helper = loader.system.python.unaliased(Path(__file__))
        expected = loader.system.python.frozen_digest(lock['configuration_helper_sha256'])
        require(loader.system.python.sha(helper) == expected)
        parent = loader.system.owned_manifest(lock['loader_path'], lock['loader_sha256'])
        loader.verify_loader_inputs(str(parent[0]), parent[1])
        selected = parent[2]
        records, directories = [], []
        for path, row in selected['files'].items():
            if row['role'] == 'configuration':
                parsed = inspect_configuration(Path(row['resolved']).read_bytes(), row['sha256'])
                records.append({'path': path, 'sha256': row['sha256'], 'declarations': parsed})
                directories.extend(item['value'] for item in parsed['records'] if item['kind'] == 'directory')
        require(loader.system.canonical(records) == loader.system.canonical(lock['configuration_files']))
        require(loader.system.canonical(directories) == loader.system.canonical(lock['configuration_directory_declarations']))
        added, absent = lock['additional_directories'], lock['additional_absent']
        require(type(added) is dict and type(absent) is dict and not set(added) & set(absent))
        missing = set(directories) - set(selected['directories']) - set(selected['absent'])
        require(set(added) | set(absent) == missing)
        require(type(lock['missing_configuration_directory_records']) is int
                and lock['missing_configuration_directory_records'] == len(missing))
        for path, row in added.items():
            require(loader.system.canonical(loader.directory_identity(path)) == loader.system.canonical(row))
        for path, row in absent.items():
            actual = loader.path_identity(path)
            require(actual.get('absent') is True and loader.system.canonical(actual) == loader.system.canonical(row))
        snapshots = dict(selected['directories'], **added)
        absent_paths = set(selected['absent']) | set(absent)
        configs = {row['path']: row['declarations']['records'] for row in records}
        pending, seen = [str(loader.absolute(lock['configuration_root_path']))], set()
        while pending:
            path = pending.pop(0)
            if path in seen:
                continue
            require(path in configs)
            seen.add(path)
            for item in configs[path]:
                if item['kind'] != 'include':
                    continue
                directory, pattern = os.path.split(item['value'])
                require(directory in snapshots or directory in absent_paths)
                children = snapshots[directory]['children'] if directory in snapshots else []
                for child in children:
                    if child['name'].startswith('.') and not pattern.startswith('.'):
                        continue
                    if fnmatch.fnmatchcase(child['name'], pattern):
                        require(child['kind'] in ('file', 'symlink'))
                        included = str(Path(directory) / child['name'])
                        require(included in configs)
                        pending.append(included)
            require(len(pending) + len(seen) <= 4096)
        require(seen == set(configs))
        # Snapshot proof is point-in-time, not an execution lease.
        for path, row in added.items():
            require(loader.system.canonical(loader.directory_identity(path)) == loader.system.canonical(row))
        for path, row in absent.items():
            require(loader.system.canonical(loader.path_identity(path)) == loader.system.canonical(row))
        for path, expected_digest, _ in (manifest, parent):
            require(loader.system.python.sha(loader.system.python.unaliased(path)) == expected_digest)
        require(loader.system.python.sha(helper) == expected)
        loader.verify_loader_inputs(str(parent[0]), parent[1])
        return {'format': 'dope-compressed-loader-configuration-input-custody', 'version': 1,
                'search_sha256': manifest[1], 'configuration_files_verified': len(records),
                'configured_directory_declarations_verified': len(directories),
                'additional_directory_snapshots_verified': len(added),
                'additional_absence_snapshots_verified': len(absent),
                'include_file_membership_verified': True, 'actual_loader_selection_verified': False,
                'complete_provider_search_verified': False, 'full_runtime_closure_certified': False,
                'execution_admitted': False, 'candidate_processes_started': 0,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed loader configuration rejected') from None
