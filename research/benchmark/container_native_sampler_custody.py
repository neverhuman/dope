"""Bind native sampler ELF declarations to owned fit/job metadata; never run it."""
import hashlib
from pathlib import Path
import stat

from . import container_bootstrap_invocation as bootstrap
from . import container_elf_inputs as elf
from . import container_system_custody as system


class _DeadlineExhausted(Exception):
    pass


def clock_call(action):
    try:
        return action()
    except TimeoutError:
        raise _DeadlineExhausted() from None


def require(condition):
    if not condition:
        raise ValueError('compressed native sampler custody rejected')


def owned_regular(path, expected):
    path = system.python.scratch(path)
    require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size <= 16 * 1024 * 1024)
    return system.owned_manifest(str(path), expected)


def verify_sampler_declarations(sampler_path, sampler_sha256, *, batch_started_at):
    """Already trusted caller supplies the original timer and separate guards.

    This adds a native executable root omitted from the Python ELF inventory.
    It does not resolve providers, load the executable or admit its execution.
    """
    python = system.python
    try:
        manifest = owned_regular(sampler_path, sampler_sha256)
        lock = manifest[2]
        require(lock['format'] == 'dope-native-sampler-elf-preparation')
        require(type(lock['version']) is int and lock['version'] == 1)
        for key in ('loader_resolution_verified', 'full_runtime_closure_certified',
                    'execution_admitted', 'official_tests_opened'):
            require(lock[key] is False)
        require(type(lock['candidate_processes_started']) is int and lock['candidate_processes_started'] == 0)
        require(all(lock[key] is None for key in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')))
        helpers = [(python.unaliased(Path(module.__file__)), python.frozen_digest(lock[key]))
                   for module, key in ((elf, 'elf_helper_sha256'), (bootstrap, 'bootstrap_helper_sha256'))]
        helpers.append((python.unaliased(Path(__file__)), python.frozen_digest(lock['sampler_helper_sha256'])))
        for path, expected in helpers:
            require(python.sha(path) == expected)
        proposal = owned_regular(lock['proposal_path'], lock['proposal_sha256'])
        proposal_blob = proposal[0].read_bytes()
        plan = clock_call(lambda: bootstrap.prepare_invocation(proposal_blob, proposal[1],
                                                               batch_started_at=batch_started_at))
        request = owned_regular(proposal[2]['request'], lock['request_sha256'])
        require(request[2]['round_sha256'] == proposal[2]['round_sha256'])
        job = request[2]['job']
        job_sha = hashlib.sha256(system.canonical(job).encode()).hexdigest()
        require(job_sha == proposal[2]['job_sha256'] == python.frozen_digest(lock['job_sha256']))
        fit = owned_regular(lock['fit_round_path'], lock['fit_round_sha256'])
        binary = python.scratch(lock['binary_path'])
        expected = python.frozen_digest(lock['binary_sha256'])
        require(str(binary) == fit[2]['gpu_binary'])
        require(expected == python.frozen_digest(fit[2]['gpu_binary_sha256'])
                == python.frozen_digest(job['original_generator_binary_sha256']))
        require(system.canonical(job['original_product_source_files_sha256'])
                == system.canonical(fit[2]['binary_product_source_sha256']))
        require(type(lock['binary_bytes']) is int and lock['binary_bytes'] >= 64)
        require(stat.S_ISREG(binary.stat().st_mode) and binary.stat().st_size == lock['binary_bytes'])
        blob = binary.read_bytes()
        require(len(blob) == lock['binary_bytes'] and hashlib.sha256(blob).hexdigest() == expected)
        declarations = elf.inspect_elf_inputs(blob, expected)
        require(system.canonical(declarations) == system.canonical(lock['declarations']))
        for path, frozen, _ in (manifest, proposal, request, fit):
            require(python.sha(python.unaliased(path)) == frozen)
        python.unaliased(binary)
        require(stat.S_ISREG(binary.stat().st_mode) and binary.stat().st_size == lock['binary_bytes'])
        require(python.sha(binary) == expected)
        for path, frozen in helpers:
            require(python.sha(path) == frozen)
        remaining = clock_call(plan.remaining_seconds)
        return {'format': 'dope-compressed-native-sampler-elf-custody', 'version': 1,
                'sampler_sha256': manifest[1], 'binary_sha256': expected, 'job_sha256': job_sha,
                'native_elf_roots_verified': 1, 'needed_declarations': len(declarations['declarations']['needed']),
                'outer_seconds_remaining': remaining, 'loader_resolution_verified': False,
                'full_runtime_closure_certified': False, 'execution_admitted': False,
                'candidate_processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except _DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        raise ValueError('compressed native sampler custody rejected') from None
