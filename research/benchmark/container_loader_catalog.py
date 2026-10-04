"""Replay a frozen candidate enumeration policy; never select or load providers."""
from collections import defaultdict, deque
import hashlib
from pathlib import Path
import re

from . import container_loader_custody as loader

POLICY = ('package basenames and x86-64 cache entries plus direct absolute names; '
          'not actual loader selection or complete search')
MAX_EDGES = 100000


def require(condition):
    if not condition:
        raise ValueError('compressed loader candidate catalog rejected')


def declared_edges(objects):
    edges = []
    for row in objects:
        origin = {'path': row['path'], 'sha256': row['expected_sha256']}
        inspected = row['declarations']
        if inspected['interpreter'] is not None:
            edges.append(dict(origin=origin, kind='interpreter', ordinal=0,
                              declared_value=inspected['interpreter']))
        values = inspected['declarations']
        require(not any(values[key] for key in ('audit', 'depaudit', 'filter', 'auxiliary')))
        for kind in ('needed', 'rpath', 'runpath'):
            edges.extend(dict(origin=origin, kind=kind, ordinal=i, declared_value=value)
                         for i, value in enumerate(values[kind]))
        require(len(edges) <= MAX_EDGES)
    return edges


def replay_edges(objects, roots, files, transcript):
    cache, bundled = defaultdict(list), defaultdict(list)
    pattern = re.compile(r'^\s*(\S+)\s+\(([^)]+)\)\s+=>\s+(/\S+)\s*$')
    for line in transcript.decode('utf-8').splitlines():
        match = pattern.fullmatch(line)
        if match and 'x86-64' in match[2]:
            cache[match[1]].append(str(loader.absolute(match[3])))
    for row in objects:
        if row['group'] == 'package-extension':
            bundled[Path(row['path']).name].append(row['path'])
    seen = {row['path'] for row in objects}
    require(len(seen) == len(objects))
    queue, edges = deque(), []

    def enqueue(path):
        require(path in files and files[path]['role'] in ('candidate-provider', 'interpreter'))
        row = files[path]
        values = row['declarations']['declarations']
        require(not any(values[key] for key in ('audit', 'depaudit', 'filter', 'auxiliary')))
        queue.extend(dict(origin={'path': path, 'sha256': row['sha256']}, kind='needed',
                          ordinal=i, declared_value=name) for i, name in enumerate(values['needed']))
        require(len(edges) + len(queue) <= MAX_EDGES)

    for edge in roots:
        if edge['kind'] == 'needed':
            queue.append(edge)
        elif edge['kind'] == 'interpreter':
            path = str(loader.absolute(edge['declared_value']))
            enqueue(path)
            edges.append(dict(edge, candidate_paths=[path]))
    while queue:
        edge = queue.popleft()
        name = edge['declared_value']
        choices = list(dict.fromkeys(bundled.get(name, []) + cache.get(name, [])))
        if '/' in name:
            require('$' not in name)
            choices = [str(loader.absolute(name))]
        require(bool(choices))
        edges.append(dict(edge, candidate_paths=choices))
        for path in choices:
            require(path in files and files[path]['role'] in ('candidate-provider', 'interpreter'))
            if path not in seen:
                seen.add(path)
                enqueue(path)
        require(len(edges) + len(queue) <= MAX_EDGES)
    return edges


def verify_candidate_catalog(loader_path, loader_sha256):
    """Call after Python/root ELF custody from a trusted coordinator.

    Package-basename/cache order is a catalog policy, not glibc search order.
    Competing candidates remain unresolved and execution remains unadmitted.
    """
    try:
        loader.verify_loader_inputs(loader_path, loader_sha256)
        manifest = loader.system.owned_manifest(loader_path, loader_sha256)
        lock = manifest[2]
        require(lock['candidate_enumeration_policy'] == POLICY)
        parents = [loader.system.owned_manifest(lock[key + '_path'], lock[key + '_sha256'])
                   for key in ('declarations', 'unresolved', 'cache_receipt')]
        declarations, unresolved, cache = [item[2] for item in parents]
        objects = declarations['objects']
        require(type(objects) is list and 0 < len(objects) <= MAX_EDGES)
        for row in objects:
            path = loader.absolute(row['path'])
            expected = loader.system.python.frozen_digest(row['expected_sha256'])
            blob = path.read_bytes()
            require(hashlib.sha256(blob).hexdigest() == expected)
            require(loader.system.canonical(loader.system.elf.inspect_elf_inputs(blob, expected))
                    == loader.system.canonical(row['declarations']))
        roots = declared_edges(objects)
        require(loader.system.canonical(roots) == loader.system.canonical(unresolved['declared_edges']))
        transcript_path = loader.system.python.scratch(str(parents[2][0].parent / 'cache-print.stdout'))
        transcript = transcript_path.read_bytes()
        require(hashlib.sha256(transcript).hexdigest()
                == loader.system.python.frozen_digest(cache['stdout_sha256']))
        edges = replay_edges(objects, roots, lock['files'], transcript)
        require(loader.system.canonical(edges) == loader.system.canonical(lock['candidate_edges']))
        competing = sum(len({(lock['files'][path]['resolved'], lock['files'][path]['sha256'])
                             for path in edge['candidate_paths']}) > 1
                        for edge in edges)
        for row in objects:
            require(loader.system.python.sha(loader.absolute(row['path'])) == row['expected_sha256'])
        for path, expected, _ in [manifest, *parents]:
            require(loader.system.python.sha(loader.system.python.unaliased(path)) == expected)
        loader.verify_loader_inputs(loader_path, loader_sha256)
        return {'format': 'dope-compressed-loader-candidate-catalog-replay', 'version': 1,
                'loader_sha256': manifest[1], 'ordered_candidate_edges_verified': len(edges),
                'competing_provider_identity_edges': competing, 'ordered_candidate_catalog_verified': True,
                'actual_loader_selection_verified': False, 'complete_provider_search_verified': False,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'candidate_processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed loader candidate catalog rejected') from None
