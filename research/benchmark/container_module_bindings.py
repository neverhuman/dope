"""Bind compiled startup guards to verified source paths without initializing them."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from . import container_owned_code as compiler

PACKAGE = '_dope_owned_startup_guards'


def require(condition):
    if not condition:
        raise ValueError('compressed guard module binding rejected')


@dataclass(frozen=True)
class ModuleBinding:
    name: str
    package: str
    file: str
    source_sha256: str
    code: object


@dataclass(frozen=True)
class StartupModuleBindings:
    modules: tuple
    manifest_sha256: str
    compiler_sha256: str
    binding_helper_sha256: str
    outer_seconds_remaining: float

    def receipt(self):
        return {'format': 'dope-compressed-startup-module-bindings', 'version': 1,
                'manifest_sha256': self.manifest_sha256,
                'compiler_sha256': self.compiler_sha256,
                'binding_helper_sha256': self.binding_helper_sha256,
                'source_files_bound': len(self.modules),
                'outer_seconds_remaining': self.outer_seconds_remaining,
                'source_initializers_run': False, 'modules_installed': False,
                'candidate_processes_started': 0, 'execution_admitted': False,
                'full_runtime_closure_certified': False, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}


def prepare_module_bindings(manifest_path, manifest_sha256, compiler_sha256,
                            binding_helper_sha256, *, batch_started_at):
    """Caller already owns all three seed helpers and interpreter/import closure.

    Return immutable descriptors, not imported modules. A future closed bootstrap
    must install these exact code objects, names and __file__ bindings; it must
    separately prevent ordinary import/cache fallback and namespace collisions.
    """
    sources = compiler.sources
    try:
        sources.remaining(batch_started_at)
        expected_helper = sources.digest(binding_helper_sha256)
        helper = sources.unaliased(Path(__file__))
        require(hashlib.sha256(sources.read_regular(helper)[0]).hexdigest() == expected_helper)
        try:
            compiled = compiler.prepare_startup_code(manifest_path, manifest_sha256,
                compiler_sha256, batch_started_at=batch_started_at)
            owned = sources.verify_startup_sources(manifest_path, manifest_sha256,
                batch_started_at=batch_started_at)
        except TimeoutError:
            # Both trusted helpers sanitize ordinary I/O timeouts into ValueError.
            raise sources._DeadlineExhausted() from None
        hashes = tuple((name, hashlib.sha256(body).hexdigest()) for name, body in owned.sources)
        require(hashes == compiled.source_sha256 and owned.manifest_sha256 == compiled.manifest_sha256)
        sources.remaining(batch_started_at)
        manifest = sources.scratch(manifest_path)
        manifest_blob = sources.read_regular(manifest)[0]
        require(hashlib.sha256(manifest_blob).hexdigest() == compiled.manifest_sha256)
        directory = sources.scratch(json.loads(manifest_blob)['source_directory'])
        require(tuple(name for name, _ in compiled.modules) == tuple(name for name, _ in hashes))
        modules = tuple(ModuleBinding(PACKAGE + '.' + name[:-3], PACKAGE,
            str(directory / name), sha, code)
            for (name, code), (_, sha) in zip(compiled.modules, hashes, strict=True))
        require(len(modules) == len(sources.SOURCE_NAMES))
        # Recheck all bindings after construction; point-in-time custody is not a lease.
        try:
            require(sources.verify_startup_sources(manifest_path, manifest_sha256,
                batch_started_at=batch_started_at).sources == owned.sources)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        sources.remaining(batch_started_at)
        compiler_path = sources.unaliased(Path(compiler.__file__))
        require(hashlib.sha256(sources.read_regular(compiler_path)[0]).hexdigest()
                == compiled.compiler_sha256)
        sources.remaining(batch_started_at)
        require(hashlib.sha256(sources.read_regular(helper)[0]).hexdigest() == expected_helper)
        sources.remaining(batch_started_at)
        require(sources.read_regular(manifest)[0] == manifest_blob)
        remaining = sources.remaining(batch_started_at)
        return StartupModuleBindings(modules, compiled.manifest_sha256,
            compiled.compiler_sha256, expected_helper, remaining)
    except sources._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError('compressed guard module binding rejected') from None
