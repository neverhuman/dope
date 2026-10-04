"""Bind original native library-directory files; never resolve or load them."""
from pathlib import Path
import stat

from . import container_loader_custody as loader
from . import container_native_sampler_custody as sampler


def require(condition):
    if not condition:
        raise ValueError('compressed native library custody rejected')


def verify_native_library_directory(library_path, library_sha256, *, batch_started_at):
    """Trusted caller establishes startup/native guards on its original timer.

    Recheck one recorded native directory and its selected files, not a complete
    provider search, runtime closure or executable child environment.
    """
    system = sampler.system
    python = system.python
    try:
        manifest = sampler.owned_regular(library_path, library_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-native-library-directory-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('actual_loader_selection_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            require(lock[key] is False)
        require(type(lock['candidate_processes_started']) is int and lock['candidate_processes_started'] == 0)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        helpers = [(python.unaliased(Path(module.__file__)), python.frozen_digest(lock[key]))
                   for module, key in ((loader, 'loader_helper_sha256'), (sampler, 'sampler_helper_sha256'))]
        helpers.append((python.unaliased(Path(__file__)), python.frozen_digest(lock['library_helper_sha256'])))
        for path, expected in helpers:
            require(python.sha(path) == expected)
        native = sampler.owned_regular(lock['sampler_path'], lock['sampler_sha256'])
        proposal = sampler.owned_regular(native[2]['proposal_path'], native[2]['proposal_sha256'])
        proposal_blob = proposal[0].read_bytes()
        plan = sampler.clock_call(lambda: sampler.bootstrap.prepare_invocation(proposal_blob, proposal[1],
                                                                               batch_started_at=batch_started_at))
        validation = sampler.owned_regular(lock['validation_path'], lock['validation_sha256'])
        reference = validation[2]['host_runtimes'][plan.host]
        runtime = sampler.owned_regular(reference['path'], reference['sha256'])
        require(runtime[2]['host'] == plan.host)
        require(runtime[2]['binary'] == validation[2]['gpu_binary'] == native[2]['binary_path'])
        require(python.frozen_digest(runtime[2]['binary_sha256'])
                == python.frozen_digest(validation[2]['gpu_binary_sha256'])
                == python.frozen_digest(native[2]['binary_sha256']))
        directory = str(loader.absolute(lock['library_directory']))
        require(directory == validation[2]['native_library_directory'] == runtime[2]['torch_library_directory'])
        frozen_paths = runtime[2]['root_inventory_files'][directory]
        require(type(frozen_paths) is list and all(type(path) is str for path in frozen_paths))
        require(bool(frozen_paths) and len(set(frozen_paths)) == len(frozen_paths))
        files = lock['files']
        require(type(files) is dict and set(files) == set(frozen_paths))
        snapshot = lock['directory_snapshot']

        def inspect_directory():
            actual = loader.directory_identity(directory)
            require(system.canonical(actual) == system.canonical(snapshot))
            require(all(child['kind'] in ('file', 'symlink') for child in actual['children']))
            require({str(Path(directory) / child['name']) for child in actual['children']} == set(files))

        def inspect_files():
            for path, row in files.items():
                sampler.clock_call(plan.remaining_seconds)
                actual = loader.path_identity(path)
                require(actual.get('kind') == 'file' and type(row) is dict)
                require(system.canonical(actual) == system.canonical(row['identity']))
                original = runtime[2]['files'][path]
                require(type(row['bytes']) is int and row['bytes'] >= 0 and type(original['bytes']) is int)
                expected = python.frozen_digest(row['sha256'])
                require(original['bytes'] == row['bytes'] and original['resolved_path'] == actual['resolved']
                        and python.frozen_digest(original['sha256']) == expected)
                resolved = Path(actual['resolved'])
                require(stat.S_ISREG(resolved.stat().st_mode) and resolved.stat().st_size == row['bytes'])
                require(python.sha(resolved) == expected)

        inspect_directory()
        inspect_files()
        inspect_files()
        inspect_directory()
        for path, expected, _ in (manifest, native, proposal, validation, runtime):
            python.unaliased(path)
            require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size <= 16 * 1024 * 1024)
            require(python.sha(python.unaliased(path)) == expected)
        for path, expected in helpers:
            require(stat.S_ISREG(path.stat().st_mode))
            require(python.sha(python.unaliased(path)) == expected)
        remaining = sampler.clock_call(plan.remaining_seconds)
        return {'format': 'dope-compressed-native-library-directory-custody', 'version': 1,
                'library_sha256': manifest[1], 'native_sampler_sha256': native[1],
                'original_native_directory_verified': True, 'selected_directory_files_verified': len(files),
                'literal_alias_snapshots_verified': True, 'outer_seconds_remaining': remaining,
                'actual_loader_selection_verified': False, 'full_runtime_closure_certified': False,
                'execution_admitted': False, 'candidate_processes_started': 0,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}
    except sampler._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed native library custody rejected') from None
