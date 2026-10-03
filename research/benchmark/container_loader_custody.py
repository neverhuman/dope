"""Recheck selected loader input snapshots; never resolve or load a library."""
import hashlib
import os
from pathlib import Path
import stat

from . import container_system_custody as system


def require(condition):
    if not condition:
        raise ValueError('compressed loader input custody rejected')


def absolute(value):
    require(type(value) is str and value.startswith('/') and not value.startswith('//'))
    require(os.path.normpath(value) == value)
    return Path(value)


def path_identity(value):
    """Walk aliases component by component, including symlink-target dot parts."""
    requested = absolute(value)
    parts, current, aliases = list(requested.parts[1:]), Path('/'), []
    while parts:
        part = parts.pop(0)
        if part in ('.', '..'):
            if part == '..':
                current = current.parent
            continue
        candidate = current / part
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            return {'path': value, 'missing_at': str(candidate), 'aliases': aliases, 'absent': True}
        if stat.S_ISLNK(info.st_mode):
            require(len(aliases) < 40)
            target = str(candidate.readlink())
            aliases.append({'path': str(candidate), 'target': target,
                            'mode': stat.S_IMODE(info.st_mode), 'uid': info.st_uid, 'gid': info.st_gid})
            target_path = Path(target)
            if target_path.is_absolute():
                current = Path('/')
                parts = list(target_path.parts[1:]) + parts
            else:
                parts = list(target_path.parts) + parts
        else:
            require(not parts or stat.S_ISDIR(info.st_mode))
            current = candidate
    info = current.lstat()
    kind = ('directory' if stat.S_ISDIR(info.st_mode) else
            'file' if stat.S_ISREG(info.st_mode) else 'special')
    return {'path': value, 'resolved': str(current), 'aliases': aliases,
            'mode': stat.S_IMODE(info.st_mode), 'uid': info.st_uid, 'gid': info.st_gid, 'kind': kind}


def directory_identity(value):
    result = path_identity(value)
    require(result.get('kind') == 'directory')
    children = []
    for child in Path(result['resolved']).iterdir():
        info = child.lstat()
        kind = ('symlink' if stat.S_ISLNK(info.st_mode) else
                'directory' if stat.S_ISDIR(info.st_mode) else
                'file' if stat.S_ISREG(info.st_mode) else 'special')
        row = {'name': child.name, 'kind': kind}
        if kind == 'symlink':
            row['target'] = str(child.readlink())
        children.append(row)
    result['children'] = sorted(children, key=lambda row: row['name'])
    return result


def verify_loader_inputs(loader_path, loader_sha256):
    """Externally pin a selected proposal, not a complete loader search policy.

    Call only from an already trusted coordinator after Python/ELF file custody
    checks. This starts no interpreter or library and grants no job admission.
    """
    try:
        manifest = system.owned_manifest(loader_path, loader_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-shared-validation-loader-input-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('official_tests_opened', 'actual_loader_selection_verified',
                    'complete_provider_search_verified', 'cache_default_preload_hwcaps_dlopen_complete',
                    'cross_host_verified', 'full_runtime_closure_certified', 'execution_admitted'):
            require(lock[key] is False)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        for key in ('candidate_libraries_loaded', 'generator_fits', 'auditor_fits', 'samples_generated'):
            require(type(lock[key]) is int and lock[key] == 0)
        require(type(lock['gaps']) is list and not lock['gaps'])
        parents = [system.owned_manifest(lock[key + '_path'], lock[key + '_sha256'])
                   for key in ('unresolved', 'declarations', 'cache_receipt')]
        unresolved, declarations, cache = [parent[2] for parent in parents]
        require(unresolved['declarations_path'] == str(parents[1][0])
                and unresolved['declarations_sha256'] == parents[1][1])
        helper = system.python.unaliased(Path(system.elf.__file__))
        helper_sha = system.python.frozen_digest(declarations['helper_sha256'])
        require(system.python.sha(helper) == helper_sha)
        files, directories, absent = [lock[key] for key in ('files', 'directories', 'absent')]
        require(all(type(group) is dict for group in (files, directories, absent)) and bool(files))
        require(len(set(files) | set(directories) | set(absent))
                == len(files) + len(directories) + len(absent))
        for path, row in files.items():
            actual = path_identity(path)
            require(actual.get('kind') == 'file' and type(row) is dict)
            blob = Path(actual['resolved']).read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            require(digest == system.python.frozen_digest(row['sha256']))
            role = row['role']
            require(role in ('metadata-tool', 'cache', 'configuration', 'preload', 'tunables',
                             'candidate-provider', 'interpreter'))
            actual.update(bytes=len(blob), sha256=digest, role=role)
            if role in ('candidate-provider', 'interpreter'):
                actual['declarations'] = system.elf.inspect_elf_inputs(blob, digest)
            require(system.canonical(actual) == system.canonical(row))
        argv = cache['metadata_tool_argv']
        require(type(argv) is list and len(argv) == 4 and argv[1:3] == ['-p', '-C'])
        require(files[argv[0]]['role'] == 'metadata-tool' and files[argv[3]]['role'] == 'cache')
        require(files[argv[0]]['sha256'] == system.python.frozen_digest(cache['tool_sha256'])
                and files[argv[3]]['sha256'] == system.python.frozen_digest(cache['cache_sha256']))
        transcripts = [(system.python.scratch(str(parents[2][0].parent / ('cache-print.' + name))),
                        system.python.frozen_digest(cache[name + '_sha256'])) for name in ('stdout', 'stderr')]
        for path, row in directories.items():
            require(system.canonical(directory_identity(path)) == system.canonical(row))
        for path, row in absent.items():
            actual = path_identity(path)
            require(actual.get('absent') is True and system.canonical(actual) == system.canonical(row))
        # Repeat snapshots and file hashes after parsing; success is not an atomic execution lease.
        for path, row in files.items():
            metadata = {key: value for key, value in row.items()
                        if key not in ('bytes', 'sha256', 'role', 'declarations')}
            require(system.canonical(path_identity(path)) == system.canonical(metadata))
            require(system.python.sha(Path(row['resolved'])) == row['sha256'])
        for path, row in directories.items():
            require(system.canonical(directory_identity(path)) == system.canonical(row))
        for path, row in absent.items():
            require(system.canonical(path_identity(path)) == system.canonical(row))
        for path, expected in [(path, expected) for path, expected, _ in [manifest, *parents]] + transcripts:
            require(system.python.sha(system.python.unaliased(path)) == expected)
        require(system.python.sha(helper) == helper_sha)
        return {'format': 'dope-compressed-selected-loader-input-custody', 'version': 1,
                'loader_sha256': manifest[1], 'selected_files_verified': len(files),
                'directory_snapshots_verified': len(directories), 'absence_records_verified': len(absent),
                'selected_loader_inputs_verified': True, 'actual_loader_selection_verified': False,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'candidate_processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed loader input custody rejected') from None
