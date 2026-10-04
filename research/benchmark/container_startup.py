"""Replay declared startup custody on one original timer; never dispatch."""
from pathlib import Path
import stat

from . import container_bootstrap_filesystem as filesystem
from . import container_native_runtime_inventory as inventory


def require(condition):
    if not condition:
        raise ValueError('compressed startup custody rejected')


def verify_proposed_startup(startup_path, startup_sha256, *, batch_started_at):
    """Trusted caller owns sources and complete interpreter closure before import.

    These recorded custody checks do not admit execution or certify system closure.
    """
    sampler = inventory.environment.library.sampler
    system = sampler.system
    python = system.python
    try:
        manifest = sampler.owned_regular(startup_path, startup_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-proposed-startup-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('actual_loader_selection_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            require(lock[key] is False)
        require(type(lock['candidate_processes_started']) is int and lock['candidate_processes_started'] == 0)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        helpers = [(python.unaliased(Path(module.__file__)), python.frozen_digest(lock[key]))
                   for module, key in ((filesystem, 'filesystem_helper_sha256'),
                                       (sampler, 'sampler_helper_sha256'),
                                       (inventory, 'inventory_helper_sha256'),
                                       (system, 'system_helper_sha256'))]
        helpers.append((python.unaliased(Path(__file__)), python.frozen_digest(lock['startup_helper_sha256'])))

        def inspect_helpers():
            for path, expected in helpers:
                python.unaliased(path)
                require(stat.S_ISREG(path.stat().st_mode) and python.sha(path) == expected)

        inspect_helpers()
        fs = sampler.owned_regular(lock['filesystem_path'], lock['filesystem_sha256'])
        native = sampler.owned_regular(lock['sampler_path'], lock['sampler_sha256'])
        recorded = sampler.owned_regular(lock['inventory_path'], lock['inventory_sha256'])
        child = sampler.owned_regular(recorded[2]['environment_path'], recorded[2]['environment_sha256'])
        selected = sampler.owned_regular(child[2]['library_path'], child[2]['library_sha256'])
        require(selected[2]['sampler_path'] == str(native[0]) and selected[2]['sampler_sha256'] == native[1])
        environment = sampler.owned_regular(fs[2]['environment_path'], fs[2]['environment_sha256'])
        require(environment[2]['proposal_path'] == native[2]['proposal_path']
                and environment[2]['proposal_sha256'] == native[2]['proposal_sha256'])
        projection = sampler.owned_regular(environment[2]['projection_path'], environment[2]['projection_sha256'])
        declarations = sampler.owned_regular(projection[2]['declarations_path'], projection[2]['declarations_sha256'])
        runtime = sampler.owned_regular(declarations[2]['runtime_path'], declarations[2]['runtime_sha256'])
        auxiliary = sampler.owned_regular(declarations[2]['auxiliary_path'], declarations[2]['auxiliary_sha256'])
        proposal = sampler.owned_regular(child[2]['proposal_path'], child[2]['proposal_sha256'])
        proposal_blob = proposal[0].read_bytes()
        plan = sampler.clock_call(lambda: sampler.bootstrap.prepare_invocation(
            proposal_blob, proposal[1], batch_started_at=batch_started_at))
        sampler.clock_call(plan.remaining_seconds)
        roots = system.verify_declared_elf_inputs(str(declarations[0]), declarations[1],
            str(runtime[0]), runtime[1], str(auxiliary[0]), auxiliary[1])
        sampler.clock_call(plan.remaining_seconds)
        staged = sampler.clock_call(lambda: filesystem.verify_proposed_filesystem(
            str(fs[0]), fs[1], batch_started_at=batch_started_at))
        executable = sampler.clock_call(lambda: sampler.verify_sampler_declarations(
            str(native[0]), native[1], batch_started_at=batch_started_at))
        files = sampler.clock_call(lambda: inventory.verify_native_runtime_inventory(
            str(recorded[0]), recorded[1], batch_started_at=batch_started_at))
        for path, expected, _ in (manifest, fs, native, recorded, child, selected, environment,
                                  projection, declarations, runtime, auxiliary, proposal):
            python.unaliased(path)
            require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size <= 16 * 1024 * 1024)
            require(python.sha(path) == expected)
        inspect_helpers()
        remaining = sampler.clock_call(plan.remaining_seconds)
        return {'format': 'dope-compressed-proposed-startup-custody', 'version': 1,
                'startup_sha256': manifest[1], 'filesystem_sha256': fs[1],
                'native_sampler_sha256': native[1], 'inventory_sha256': recorded[1],
                'declared_python_elf_files_verified': roots['declared_elf_files_verified'],
                'staged_source_files_verified': staged['source_files_verified'],
                'native_elf_roots_verified': executable['native_elf_roots_verified'],
                'recorded_native_files_verified': files['recorded_native_files_verified'],
                'outer_seconds_remaining': remaining, 'actual_loader_selection_verified': False,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'candidate_processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except sampler._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed startup custody rejected') from None
