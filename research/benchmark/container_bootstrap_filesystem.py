"""Inspect proposed startup files without executing or admitting the worker."""
import hashlib
import os
from pathlib import Path
import stat

from . import container_loader_environment as environment

SOURCE_NAMES = {'entry.py', 'common.py', 'shared_runtime.py', 'native_runtime.py',
                'research_container.py', 'research_container_members.py'}


def require(condition):
    if not condition:
        raise ValueError('compressed bootstrap filesystem rejected')


def verify_proposed_filesystem(filesystem_path, filesystem_sha256, *, batch_started_at):
    """The already trusted caller supplies the original whole-batch timer.

    This checks staging files for an unadmitted descriptor, not a final round,
    import closure, atomic execution lease or permission to dispatch a process.
    """
    system = environment.search.catalog.loader.system
    python = system.python
    try:
        manifest = system.owned_manifest(filesystem_path, filesystem_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-proposed-bootstrap-filesystem-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('final_campaign_or_execution_lock', 'execution_admitted',
                    'full_runtime_closure_certified', 'official_tests_opened'):
            require(lock[key] is False)
        require(type(lock['candidate_processes_started']) is int and lock['candidate_processes_started'] == 0)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        helper = python.unaliased(Path(__file__))
        helper_sha = python.frozen_digest(lock['filesystem_helper_sha256'])
        require(python.sha(helper) == helper_sha)
        parent = system.owned_manifest(lock['environment_path'], lock['environment_sha256'])
        environment.deadline_call(lambda: environment.verify_proposed_environment(
            str(parent[0]), parent[1], batch_started_at=batch_started_at))
        proposal = system.owned_manifest(parent[2]['proposal_path'], parent[2]['proposal_sha256'])
        proposal_blob = proposal[0].read_bytes()
        plan = environment.deadline_call(lambda: environment.bootstrap.prepare_invocation(
            proposal_blob, proposal[1], batch_started_at=batch_started_at))
        root = python.scratch(proposal[2]['root'])
        descriptor = system.owned_manifest(str(root / 'round-proposal.json'), proposal[2]['round_sha256'])
        require(descriptor[2]['format'] == 'dope-unadmitted-bootstrap-round-proposal')
        require(type(descriptor[2]['version']) is int and descriptor[2]['version'] == 1)
        require(descriptor[2]['final_campaign_or_execution_lock'] is False
                and descriptor[2]['execution_admitted'] is False and descriptor[2]['official_tests_opened'] is False)
        jobs = descriptor[2]['frozen_preview_job_digests']
        require(type(jobs) is list and bool(jobs))
        require(all(python.frozen_digest(value) == value for value in jobs))
        require(len(set(jobs)) == len(jobs) and proposal[2]['job_sha256'] in jobs)
        source = python.scratch(str(root / 'source'))
        require(type(lock['source_files']) is dict and set(lock['source_files']) == SOURCE_NAMES)
        request = python.scratch(proposal[2]['request'])
        cache = python.scratch(proposal[2]['empty_bytecode_cache'])

        def inspect_files():
            require(source.is_dir() and {p.name for p in source.iterdir()} == SOURCE_NAMES)
            for name, row in lock['source_files'].items():
                path = python.unaliased(source / name)
                require(stat.S_ISREG(path.stat().st_mode))
                python.file_identity(path, row)
            require(root / 'source/entry.py' == Path(plan.argv[6]))
            parent_stat = request.parent.stat()
            require(request.parent.is_dir() and parent_stat.st_uid == os.getuid()
                    and stat.S_IMODE(parent_stat.st_mode) == 0o700)
            python.unaliased(request)
            require(stat.S_ISREG(request.stat().st_mode))
            python.file_identity(request, lock['request_file'])
            python.unaliased(cache)
            require(cache.is_dir() and not list(cache.iterdir()))

        inspect_files()
        # Only externally owned regular request bytes reach JSON decoding.
        req = system.owned_manifest(str(request), lock['request_file']['sha256'])[2]
        require(req['round_sha256'] == descriptor[1])
        job_sha = hashlib.sha256(system.canonical(req['job']).encode()).hexdigest()
        require(job_sha == proposal[2]['job_sha256'] and job_sha in jobs)
        environment.deadline_call(lambda: environment.verify_proposed_environment(
            str(parent[0]), parent[1], batch_started_at=batch_started_at))
        inspect_files()
        for path, expected, _ in (manifest, parent, proposal, descriptor):
            require(python.sha(python.unaliased(path)) == expected)
        require(python.sha(helper) == helper_sha)
        remaining = environment.deadline_call(plan.remaining_seconds)
        return {'format': 'dope-compressed-proposed-bootstrap-filesystem-custody', 'version': 1,
                'filesystem_sha256': manifest[1], 'environment_sha256': parent[1],
                'source_files_verified': len(SOURCE_NAMES), 'request_job_sha256': job_sha,
                'empty_cache_verified': True, 'outer_seconds_remaining': remaining,
                'final_campaign_or_execution_lock': False, 'execution_admitted': False,
                'full_runtime_closure_certified': False, 'candidate_processes_started': 0,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}
    except environment._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed bootstrap filesystem rejected') from None
