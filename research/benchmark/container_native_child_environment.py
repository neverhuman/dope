"""Bind a proposed child environment to original selected native inputs."""
import hashlib
from pathlib import Path
import stat

from . import container_native_library_custody as library

PRELOAD = '/etc/ld.so.preload'


def require(condition):
    if not condition:
        raise ValueError('compressed native child environment rejected')


def verify_native_child_environment(environment_path, environment_sha256, *, batch_started_at):
    """Trusted startup/native prerequisites remain separate on the original timer.

    Check a pure declaration only. No child process, provider search or execution
    lease is established by these snapshots.
    """
    sampler = library.sampler
    python = sampler.system.python
    try:
        manifest = sampler.owned_regular(environment_path, environment_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-native-child-environment-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('actual_child_environment_verified', 'actual_loader_selection_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            require(lock[key] is False)
        require(type(lock['candidate_processes_started']) is int and lock['candidate_processes_started'] == 0)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        helpers = [(python.unaliased(Path(module.__file__)), python.frozen_digest(lock[key]))
                   for module, key in ((library, 'library_helper_sha256'),
                                       (sampler.bootstrap, 'bootstrap_helper_sha256'),
                                       (library.loader, 'loader_helper_sha256'))]
        helpers.append((python.unaliased(Path(__file__)), python.frozen_digest(lock['environment_helper_sha256'])))
        for path, expected in helpers:
            require(stat.S_ISREG(path.stat().st_mode) and python.sha(path) == expected)
        selected = sampler.owned_regular(lock['library_path'], lock['library_sha256'])
        native = sampler.owned_regular(selected[2]['sampler_path'], selected[2]['sampler_sha256'])
        original = sampler.owned_regular(native[2]['proposal_path'], native[2]['proposal_sha256'])
        proposal = sampler.owned_regular(lock['proposal_path'], lock['proposal_sha256'])
        require(sampler.system.canonical({k: v for k, v in proposal[2].items() if k != 'native_library_directory'})
                == sampler.system.canonical({k: v for k, v in original[2].items() if k != 'native_library_directory'}))
        proposal_blob = proposal[0].read_bytes()
        plan = sampler.clock_call(lambda: sampler.bootstrap.prepare_invocation(proposal_blob, proposal[1],
                                                                               batch_started_at=batch_started_at))
        require(proposal[2]['native_library_directory'] == selected[2]['library_directory'])
        require(sampler.system.canonical(list(plan.environment)) == sampler.system.canonical(lock['environment']))
        request = sampler.owned_regular(proposal[2]['request'], native[2]['request_sha256'])
        require(request[2]['round_sha256'] == proposal[2]['round_sha256'])
        job_sha = hashlib.sha256(sampler.system.canonical(request[2]['job']).encode()).hexdigest()
        require(job_sha == proposal[2]['job_sha256'] == python.frozen_digest(native[2]['job_sha256']))
        loader = sampler.owned_regular(lock['loader_path'], lock['loader_sha256'])
        require(type(loader[2]['absent']) is dict and PRELOAD in loader[2]['absent'])
        preload = loader[2]['absent'][PRELOAD]

        def check_preload():
            observed = library.loader.path_identity(PRELOAD)
            require(observed.get('absent') is True
                    and sampler.system.canonical(observed) == sampler.system.canonical(preload))

        check_preload()
        selected_receipt = sampler.clock_call(lambda: library.verify_native_library_directory(
            str(selected[0]), selected[1], batch_started_at=batch_started_at))
        check_preload()
        for path, expected, _ in (manifest, selected, native, original, proposal, request, loader):
            python.unaliased(path)
            require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size <= 16 * 1024 * 1024)
            require(python.sha(path) == expected)
        for path, expected in helpers:
            python.unaliased(path)
            require(stat.S_ISREG(path.stat().st_mode) and python.sha(path) == expected)
        check_preload()
        remaining = sampler.clock_call(plan.remaining_seconds)
        return {'format': 'dope-compressed-native-child-environment-custody', 'version': 1,
                'environment_sha256': manifest[1], 'proposal_sha256': proposal[1],
                'library_sha256': selected[1], 'job_sha256': job_sha,
                'proposed_environment_entries_verified': len(plan.environment),
                'declared_native_directory_verified': True, 'preload_absence_snapshot_verified': True,
                'selected_library_files_verified': selected_receipt['selected_directory_files_verified'],
                'outer_seconds_remaining': remaining, 'actual_child_environment_verified': False,
                'actual_loader_selection_verified': False, 'full_runtime_closure_certified': False,
                'execution_admitted': False, 'candidate_processes_started': 0,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}
    except sampler._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed native child environment rejected') from None
