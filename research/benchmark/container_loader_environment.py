"""Bind a restricted invocation to selected loader snapshots; never start it."""
from pathlib import Path

from . import container_bootstrap_invocation as bootstrap
from . import container_search_path_custody as search

PRELOAD = '/etc/ld.so.preload'


def require(condition):
    if not condition:
        raise ValueError('compressed loader environment rejected')


def verify_proposed_environment(environment_path, environment_sha256, *, batch_started_at):
    """Use the caller's original outer timer; snapshots grant no execution lease."""
    system = search.catalog.loader.system
    try:
        manifest = system.owned_manifest(environment_path, environment_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-proposed-loader-environment-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('actual_loader_selection_verified', 'complete_provider_search_verified',
                    'full_runtime_closure_certified', 'execution_admitted', 'official_tests_opened'):
            require(lock[key] is False)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        require(type(lock['candidate_processes_started']) is int and lock['candidate_processes_started'] == 0)
        helpers = [(system.python.unaliased(Path(module.__file__)), system.python.frozen_digest(lock[key]))
                   for module, key in [(bootstrap, 'bootstrap_helper_sha256'),
                                       (search, 'search_helper_sha256')]]
        helpers.append((system.python.unaliased(Path(__file__)),
                        system.python.frozen_digest(lock['environment_helper_sha256'])))
        for path, expected in helpers:
            require(system.python.sha(path) == expected)
        proposal = system.owned_manifest(lock['proposal_path'], lock['proposal_sha256'])
        # The restricted plan is derived before inspecting any loader files.
        blob = proposal[0].read_bytes()
        plan = bootstrap.prepare_invocation(blob, proposal[1], batch_started_at=batch_started_at)
        require(system.canonical(list(plan.environment)) == system.canonical(lock['environment']))
        projection = system.owned_manifest(lock['projection_path'], lock['projection_sha256'])
        search.verify_declared_search_paths(str(projection[0]), projection[1])
        config = system.owned_manifest(projection[2]['search_path'], projection[2]['search_sha256'])
        parent = system.owned_manifest(config[2]['loader_path'], config[2]['loader_sha256'])
        library = dict(plan.environment)['LD_LIBRARY_PATH']
        require(library in parent[2]['directories'] or library in config[2]['additional_directories'])
        require(PRELOAD in parent[2]['absent'] or PRELOAD in config[2]['additional_absent'])
        # No candidate, host, transport or resource admission follows from this plan.
        search.verify_declared_search_paths(str(projection[0]), projection[1])
        for path, expected, _ in (manifest, proposal, projection, config, parent):
            require(system.python.sha(system.python.unaliased(path)) == expected)
        for path, expected in helpers:
            require(system.python.sha(path) == expected)
        remaining = plan.remaining_seconds()
        return {'format': 'dope-compressed-proposed-loader-environment-custody', 'version': 1,
                'environment_sha256': manifest[1], 'proposal_sha256': proposal[1],
                'restricted_environment_entries_verified': len(plan.environment),
                'declared_library_directory_snapshot_verified': True,
                'preload_absence_snapshot_verified': True, 'outer_seconds_remaining': remaining,
                'actual_loader_selection_verified': False, 'complete_provider_search_verified': False,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'candidate_processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except TimeoutError:
        raise
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed loader environment rejected') from None
