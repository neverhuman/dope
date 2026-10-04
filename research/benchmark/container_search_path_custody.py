"""Bind declared pathname projections to snapshots; never resolve a loader."""
import os
from pathlib import Path
import re

from . import container_loader_catalog as catalog
from . import container_loader_configuration as configuration


def require(condition):
    if not condition:
        raise ValueError('compressed declared search path rejected')


def project_component(origin, value):
    """Project literal ORIGIN tokens using the declared pathname, not glibc origin."""
    require(type(origin) is str and type(value) is str)
    require(origin.startswith('/') and not origin.startswith('//') and '\x00' not in origin)
    require(os.path.normpath(origin) == origin)
    expanded = re.sub(r'\$(?:\{ORIGIN\}|ORIGIN(?=/|$))',
                      lambda match: str(Path(origin).parent), value)
    require(bool(expanded) and expanded.startswith('/') and not expanded.startswith('//'))
    require('$' not in expanded and '\x00' not in expanded)
    return os.path.normpath(expanded)


def verify_declared_search_paths(projection_path, projection_sha256):
    """Call after trusted Python/root ELF custody; this grants no execution lease."""
    system = catalog.loader.system
    try:
        manifest = system.owned_manifest(projection_path, projection_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-declared-search-path-projection-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('actual_loader_origin_verified', 'actual_loader_selection_verified',
                    'complete_provider_search_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            require(lock[key] is False)
        for key in ('candidate_processes_started', 'unsupported_components', 'unbound_snapshots'):
            require(type(lock[key]) is int and lock[key] == 0)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        helper = system.python.unaliased(Path(__file__))
        helper_sha = system.python.frozen_digest(lock['projection_helper_sha256'])
        require(system.python.sha(helper) == helper_sha)
        search = system.owned_manifest(lock['search_path'], lock['search_sha256'])
        configuration.verify_configuration_inputs(str(search[0]), search[1])
        parent = system.owned_manifest(search[2]['loader_path'], search[2]['loader_sha256'])
        require(lock['loader_path'] == str(parent[0]) and lock['loader_sha256'] == parent[1])
        catalog.verify_candidate_catalog(str(parent[0]), parent[1])
        declarations = system.owned_manifest(parent[2]['declarations_path'], parent[2]['declarations_sha256'])
        require(lock['declarations_path'] == str(declarations[0])
                and lock['declarations_sha256'] == declarations[1])
        sources = [(row['path'], row['expected_sha256'], 'root', row['declarations']['declarations'])
                   for row in declarations[2]['objects']]
        sources.extend((path, row['sha256'], 'selected-provider', row['declarations']['declarations'])
                       for path, row in parent[2]['files'].items()
                       if row['role'] in ('candidate-provider', 'interpreter'))
        directories = set(parent[2]['directories']) | set(search[2]['additional_directories'])
        absent = set(parent[2]['absent']) | set(search[2]['additional_absent'])
        require(not directories & absent)
        records = []
        for path, digest, group, values in sources:
            for kind in ('rpath', 'runpath'):
                for ordinal, entry in enumerate(values[kind]):
                    for index, value in enumerate(entry.split(':')):
                        projected = project_component(path, value)
                        require(projected in directories or projected in absent)
                        records.append({'origin': {'path': path, 'sha256': digest, 'group': group},
                            'kind': kind, 'ordinal': ordinal, 'component_ordinal': index,
                            'declared_value': value, 'supported_projection': True,
                            'declared_path_projection': projected,
                            'snapshot_kind': 'directory' if projected in directories else 'absent'})
                        require(len(records) <= catalog.MAX_EDGES)
        require(system.canonical(records) == system.canonical(lock['records']))
        # Recheck bodies and snapshots after projection; this is not an atomic lease.
        configuration.verify_configuration_inputs(str(search[0]), search[1])
        catalog.verify_candidate_catalog(str(parent[0]), parent[1])
        for path, expected, _ in (manifest, search, parent, declarations):
            require(system.python.sha(system.python.unaliased(path)) == expected)
        require(system.python.sha(helper) == helper_sha)
        return {'format': 'dope-compressed-declared-search-path-custody', 'version': 1,
                'projection_sha256': manifest[1], 'declared_components_verified': len(records),
                'root_components_verified': sum(row['origin']['group'] == 'root' for row in records),
                'selected_provider_components_verified': sum(row['origin']['group'] == 'selected-provider' for row in records),
                'declared_path_snapshots_verified': True, 'actual_loader_origin_verified': False,
                'actual_loader_selection_verified': False, 'complete_provider_search_verified': False,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'candidate_processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed declared search path rejected') from None
