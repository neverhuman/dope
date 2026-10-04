"""Own the fifteen startup guard sources without importing or dispatching them."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

BASE = Path('/mnt/fast-scratch/dope-benchmark')
MAX_BYTES = 16 * 1024 * 1024
SOURCE_NAMES = frozenset({
    'container_bootstrap_filesystem.py', 'container_bootstrap_invocation.py',
    'container_elf_inputs.py', 'container_loader_catalog.py',
    'container_loader_configuration.py', 'container_loader_custody.py',
    'container_loader_environment.py', 'container_native_child_environment.py',
    'container_native_library_custody.py', 'container_native_runtime_inventory.py',
    'container_native_sampler_custody.py', 'container_python_custody.py',
    'container_search_path_custody.py', 'container_startup.py',
    'container_system_custody.py',
})


def require(condition):
    if not condition:
        raise ValueError('compressed guard source custody rejected')


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None)
    return value


def unaliased(path):
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)))
    require(path.resolve(strict=True) == path)
    return path


def scratch(value):
    require(type(value) is str and Path(value).as_posix() == value)
    path = unaliased(Path(value))
    require(path.is_relative_to(BASE) and path != BASE)
    require('evaluator' not in path.parts and 'test.csv' not in path.parts)
    return path


def read_regular(path):
    unaliased(path)
    observed = path.lstat()
    require(stat.S_ISREG(observed.st_mode) and observed.st_uid == os.getuid())
    require(observed.st_size <= MAX_BYTES and stat.S_IMODE(observed.st_mode) & 0o022 == 0)
    with path.open('rb') as stream:
        body = stream.read(MAX_BYTES + 1)
    require(len(body) <= MAX_BYTES and len(body) == observed.st_size)
    return body, observed


class _DeadlineExhausted(Exception):
    pass


def remaining(start):
    require(type(start) in (int, float) and math.isfinite(start) and start >= 0)
    now = time.monotonic()
    require(type(now) in (int, float) and math.isfinite(now) and now >= start)
    seconds = 600 - (now - start)
    if seconds <= 0:
        raise _DeadlineExhausted()
    return seconds


@dataclass(frozen=True)
class StartupSources:
    """Owned immutable bytes; later pathname reads cannot replace these inputs."""
    sources: tuple
    manifest_sha256: str
    outer_seconds_remaining: float

    def receipt(self):
        return {'format': 'dope-compressed-startup-source-custody', 'version': 1,
                'manifest_sha256': self.manifest_sha256, 'source_files_verified': len(self.sources),
                'outer_seconds_remaining': self.outer_seconds_remaining,
                'sources_imported': False, 'candidate_processes_started': 0,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}


def verify_startup_sources(manifest_path, manifest_sha256, *, batch_started_at):
    """The caller already owns this checker and its interpreter/import closure.

    Check before importing staged guards. This owns application source bytes only;
    it verifies no interpreter/system closure, atomic lease, capacity or dispatch.
    """
    try:
        remaining(batch_started_at)
        expected = digest(manifest_sha256)
        manifest = scratch(manifest_path)
        manifest_blob, _ = read_regular(manifest)
        require(hashlib.sha256(manifest_blob).hexdigest() == expected)
        lock = json.loads(manifest_blob)
        require(type(lock) is dict and lock['format'] == 'dope-startup-source-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('execution_admitted', 'full_runtime_closure_certified', 'official_tests_opened'):
            require(lock[key] is False)
        require(type(lock['candidate_processes_started']) is int and lock['candidate_processes_started'] == 0)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        helper = unaliased(Path(__file__))
        helper_sha = digest(lock['guard_source_helper_sha256'])
        require(hashlib.sha256(read_regular(helper)[0]).hexdigest() == helper_sha)
        source = scratch(lock['source_directory'])
        require(type(lock['source_files']) is dict and set(lock['source_files']) == SOURCE_NAMES)

        def inspect_sources():
            unaliased(source)
            observed = source.lstat()
            require(stat.S_ISDIR(observed.st_mode) and observed.st_uid == os.getuid()
                    and stat.S_IMODE(observed.st_mode) == 0o700)
            require({p.name for p in source.iterdir()} == SOURCE_NAMES)
            owned = []
            for name in sorted(SOURCE_NAMES):
                remaining(batch_started_at)
                row = lock['source_files'][name]
                require(type(row) is dict and type(row['bytes']) is int and row['bytes'] >= 0)
                require(type(row['uid']) is int and type(row['mode']) is int)
                expected_source = digest(row['sha256'])
                body, info = read_regular(source / name)
                require(len(body) == row['bytes'] and info.st_uid == row['uid']
                        and stat.S_IMODE(info.st_mode) == row['mode'])
                require(hashlib.sha256(body).hexdigest() == expected_source)
                owned.append((name, body))
            return tuple(owned)

        owned = inspect_sources()
        require(inspect_sources() == owned)
        require(hashlib.sha256(read_regular(helper)[0]).hexdigest() == helper_sha)
        require(read_regular(manifest)[0] == manifest_blob)
        seconds = remaining(batch_started_at)
        return StartupSources(owned, expected, seconds)
    except _DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError('compressed guard source custody rejected') from None
