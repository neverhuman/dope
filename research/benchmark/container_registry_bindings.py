"""Prepare private registry additions without installing or initializing guards."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import ModuleType

from . import container_namespace_bindings as namespaces


def require(condition):
    if not condition:
        raise ValueError('compressed guard registry preparation rejected')


def require_vacant(registry, package):
    require(type(registry) is dict)
    # Check exact key types before equality, hashing or prefix operations.
    keys = tuple(registry)
    require(all(type(key) is str for key in keys))
    require(all(key != package and not key.startswith(package + '.') for key in keys))


@dataclass(frozen=True)
class PreparedRegistryBindings:
    namespaces: object
    registry: object
    additions: tuple
    helper_sha256: str

    def receipt(self):
        hook = self.namespaces.builtin_bindings.import_hook
        return {'format': 'dope-compressed-startup-registry-preparation', 'version': 1,
                'manifest_sha256': hook.plan.manifest_sha256, 'helper_sha256': self.helper_sha256,
                'registry_additions_planned': len(self.additions),
                'registry_exclusion_snapshot_checked': True,
                'builtins_bound_in_private_namespaces': True,
                'registry_entries_installed': 0, 'source_initializers_run': False,
                'actual_import_resolution_verified': False, 'candidate_processes_started': 0,
                'execution_admitted': False, 'full_runtime_closure_certified': False,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}


def prepare_registry_bindings(manifest_path, manifest_sha256, compiler_sha256,
                              binding_helper_sha256, declaration_helper_sha256,
                              import_helper_sha256, builtin_helper_sha256,
                              namespace_helper_sha256, helper_sha256,
                              stdlib_modules, builtin_module, registry, *, batch_started_at):
    """Caller already owns eight seed helpers, interpreter closure and references.

    Check private-name exclusion before parent work and after final custody reads.
    Return additions without modifying the supplied registry. Mutable snapshots
    do not attest loaded code or provide atomic exclusion for a later installer.
    """
    selector = namespaces.selector
    imports = selector.imports
    binding = imports.declarations.binding
    compiler = binding.compiler
    sources = compiler.sources
    try:
        sources.remaining(batch_started_at)
        expected_helper = sources.digest(helper_sha256)
        helper = sources.unaliased(Path(__file__))
        require(hashlib.sha256(sources.read_regular(helper)[0]).hexdigest() == expected_helper)
        require_vacant(registry, binding.PACKAGE)
        source_helper_sha256 = hashlib.sha256(sources.read_regular(Path(sources.__file__))[0]).hexdigest()
        try:
            prepared = namespaces.prepare_guard_namespaces(manifest_path, manifest_sha256,
                compiler_sha256, binding_helper_sha256, declaration_helper_sha256,
                import_helper_sha256, builtin_helper_sha256, namespace_helper_sha256,
                stdlib_modules, builtin_module, batch_started_at=batch_started_at)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        require(type(prepared) is namespaces.PreparedGuardNamespaces)
        selected = prepared.builtin_bindings
        hook = selected.import_hook
        require(hook.plan.initialization_order == tuple(name for name, _ in selected.namespace_bindings))
        additions = ((binding.PACKAGE, hook.package),) + tuple(
            (name, hook.guard_modules[name]) for name in hook.plan.initialization_order)
        require(len(additions) == len(sources.SOURCE_NAMES) + 1)
        try:
            owned = sources.verify_startup_sources(manifest_path, manifest_sha256,
                batch_started_at=batch_started_at)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        require(owned.manifest_sha256 == hook.plan.manifest_sha256
                and {filename[:-3]: hashlib.sha256(body).hexdigest() for filename, body in owned.sources}
                == {module.name.rsplit('.', 1)[1]: module.source_sha256 for module in hook.plan.modules})
        for path, expected in ((Path(sources.__file__), source_helper_sha256),
                               (Path(compiler.__file__), sources.digest(compiler_sha256)),
                               (Path(binding.__file__), sources.digest(binding_helper_sha256)),
                               (Path(imports.declarations.__file__), sources.digest(declaration_helper_sha256)),
                               (Path(imports.__file__), sources.digest(import_helper_sha256)),
                               (Path(selector.__file__), sources.digest(builtin_helper_sha256)),
                               (Path(namespaces.__file__), sources.digest(namespace_helper_sha256)),
                               (helper, expected_helper)):
            sources.remaining(batch_started_at)
            require(hashlib.sha256(sources.read_regular(path)[0]).hexdigest() == expected)
        sources.remaining(batch_started_at)
        manifest = sources.scratch(manifest_path)
        require(hashlib.sha256(sources.read_regular(manifest)[0]).hexdigest() == hook.plan.manifest_sha256)
        require(type(hook.package) is ModuleType)
        package_values = hook.package.__dict__
        require(type(package_values.get('__name__')) is str and package_values['__name__'] == binding.PACKAGE
                and '__path__' not in package_values)
        for descriptor in hook.plan.modules:
            module = hook.guard_modules[descriptor.name]
            require(type(module) is ModuleType)
            values = module.__dict__
            require(set(values) == namespaces.EMPTY_KEYS | {'__builtins__'})
            for key, expected in (('__name__', descriptor.name), ('__package__', descriptor.package),
                                  ('__file__', descriptor.file)):
                require(type(values[key]) is str and values[key] == expected)
            require(all(values[key] is None for key in ('__doc__', '__loader__', '__spec__', '__cached__')))
            require(values['__builtins__'] is selected.builtin_references
                    and hook.package.__dict__.get(descriptor.name.rsplit('.', 1)[1]) is module)
        require_vacant(registry, binding.PACKAGE)
        sources.remaining(batch_started_at)
        return PreparedRegistryBindings(prepared, registry, additions, expected_helper)
    except sources._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError('compressed guard registry preparation rejected') from None
