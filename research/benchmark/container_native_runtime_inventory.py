"""Recheck every file in one original native inventory without loading it."""
from pathlib import Path
import stat

from . import container_native_child_environment as environment


def require(condition):
    if not condition:
        raise ValueError('compressed native runtime inventory rejected')


def verify_native_runtime_inventory(inventory_path, inventory_sha256, *, batch_started_at):
    """Already trusted caller separately establishes startup/native prerequisites.

    Recorded file coverage is not proof of all dynamic loads or system closure.
    """
    library = environment.library
    sampler = library.sampler
    python = sampler.system.python
    try:
        manifest = sampler.owned_regular(inventory_path, inventory_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-native-runtime-inventory-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('actual_linker_transcript_replayed', 'actual_loader_selection_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            require(lock[key] is False)
        require(type(lock['candidate_processes_started']) is int and lock['candidate_processes_started'] == 0)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        helpers = [(python.unaliased(Path(module.__file__)), python.frozen_digest(lock[key]))
                   for module, key in ((environment, 'environment_helper_sha256'),
                                       (library.loader, 'loader_helper_sha256'))]
        helpers.append((python.unaliased(Path(__file__)), python.frozen_digest(lock['inventory_helper_sha256'])))
        for path, expected in helpers:
            require(stat.S_ISREG(path.stat().st_mode) and python.sha(path) == expected)
        child = sampler.owned_regular(lock['environment_path'], lock['environment_sha256'])
        selected = sampler.owned_regular(child[2]['library_path'], child[2]['library_sha256'])
        native = sampler.owned_regular(selected[2]['sampler_path'], selected[2]['sampler_sha256'])
        proposal = sampler.owned_regular(child[2]['proposal_path'], child[2]['proposal_sha256'])
        proposal_blob = proposal[0].read_bytes()
        plan = sampler.clock_call(lambda: sampler.bootstrap.prepare_invocation(proposal_blob, proposal[1],
                                                                               batch_started_at=batch_started_at))
        validation = sampler.owned_regular(selected[2]['validation_path'], selected[2]['validation_sha256'])
        reference = validation[2]['host_runtimes'][plan.host]
        runtime = sampler.owned_regular(reference['path'], reference['sha256'])
        require(runtime[2]['host'] == plan.host and runtime[2]['binary'] == native[2]['binary_path'])
        require(python.frozen_digest(runtime[2]['binary_sha256']) == python.frozen_digest(native[2]['binary_sha256']))
        files = lock['files']
        require(type(files) is dict and bool(files) and set(files) == set(runtime[2]['files']))
        child_receipt = sampler.clock_call(lambda: environment.verify_native_child_environment(
            str(child[0]), child[1], batch_started_at=batch_started_at))

        def inspect_files():
            for name, row in files.items():
                sampler.clock_call(plan.remaining_seconds)
                actual = library.loader.path_identity(name)
                require(actual.get('kind') == 'file' and type(row) is dict)
                require(sampler.system.canonical(actual) == sampler.system.canonical(row['identity']))
                original = runtime[2]['files'][name]
                require(type(row['bytes']) is int and row['bytes'] >= 0 and type(original['bytes']) is int)
                expected = python.frozen_digest(row['sha256'])
                require(row['bytes'] == original['bytes'] and actual['resolved'] == original['resolved_path']
                        and expected == python.frozen_digest(original['sha256']))
                resolved = Path(actual['resolved'])
                require(stat.S_ISREG(resolved.stat().st_mode) and resolved.stat().st_size == row['bytes'])
                require(python.sha(resolved) == expected)

        inspect_files()
        inspect_files()
        for path, expected, _ in (manifest, child, selected, native, proposal, validation, runtime):
            python.unaliased(path)
            require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size <= 16 * 1024 * 1024)
            require(python.sha(path) == expected)
        for path, expected in helpers:
            python.unaliased(path)
            require(stat.S_ISREG(path.stat().st_mode) and python.sha(path) == expected)
        remaining = sampler.clock_call(plan.remaining_seconds)
        return {'format': 'dope-compressed-native-runtime-inventory-custody', 'version': 1,
                'inventory_sha256': manifest[1], 'environment_sha256': child[1],
                'original_runtime_sha256': runtime[1], 'recorded_native_files_verified': len(files),
                'selected_library_files_verified': child_receipt['selected_library_files_verified'],
                'literal_alias_snapshots_verified': True, 'outer_seconds_remaining': remaining,
                'actual_linker_transcript_replayed': False, 'actual_loader_selection_verified': False,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'candidate_processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except sampler._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed native runtime inventory rejected') from None
