"""Prepare an owned private registry overlay without activating or running it."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType, ModuleType

from . import container_registry_bindings as registry_plan


def require(condition):
    if not condition:
        raise ValueError('compressed private registry preparation rejected')


@dataclass(frozen=True)
class PreparedPrivateRegistry:
    bindings: object
    registry_references: object
    helper_sha256: str

    def receipt(self):
        parent = self.bindings.receipt()
        return parent | {'format': 'dope-compressed-startup-private-registry',
                         'helper_sha256': self.helper_sha256,
                         'private_registry_prepared': True,
                         'private_guard_references_bound': len(self.bindings.additions),
                         'private_registry_references': len(self.registry_references),
                         'live_registry_entries_installed': 0}


def prepare_private_registry(manifest_path, manifest_sha256, compiler_sha256,
                             binding_helper_sha256, declaration_helper_sha256,
                             import_helper_sha256, builtin_helper_sha256,
                             namespace_helper_sha256, registry_helper_sha256,
                             helper_sha256, stdlib_modules, builtin_module, registry, *, batch_started_at):
    """Caller owns nine seed helpers, interpreter closure and supplied references.

    Copy reference selections before parent work and bind additions in a new
    private dictionary. Never modify or activate the supplied registry. Mutable
    referenced objects and exclusion snapshots provide no atomic execution lease.
    """
    namespaces = registry_plan.namespaces
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
        registry_plan.require_vacant(registry, binding.PACKAGE)
        snapshot = dict(registry)
        source_helper_sha = hashlib.sha256(sources.read_regular(Path(sources.__file__))[0]).hexdigest()
        try:
            prepared = registry_plan.prepare_registry_bindings(manifest_path, manifest_sha256,
                compiler_sha256, binding_helper_sha256, declaration_helper_sha256,
                import_helper_sha256, builtin_helper_sha256, namespace_helper_sha256,
                registry_helper_sha256, stdlib_modules, builtin_module, registry,
                batch_started_at=batch_started_at)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        require(type(prepared) is registry_plan.PreparedRegistryBindings and prepared.registry is registry)
        hook = prepared.namespaces.builtin_bindings.import_hook
        additions = dict(prepared.additions)
        require(tuple(name for name, _ in prepared.additions)
                == (binding.PACKAGE, *hook.plan.initialization_order)
                and len(additions) == len(sources.SOURCE_NAMES) + 1)
        overlay = snapshot | additions
        try:
            owned = sources.verify_startup_sources(manifest_path, manifest_sha256,
                batch_started_at=batch_started_at)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        require(owned.manifest_sha256 == hook.plan.manifest_sha256
                and {name[:-3]: hashlib.sha256(body).hexdigest() for name, body in owned.sources}
                == {module.name.rsplit('.', 1)[1]: module.source_sha256 for module in hook.plan.modules})
        for path, expected in ((Path(sources.__file__), source_helper_sha),
                               (Path(compiler.__file__), sources.digest(compiler_sha256)),
                               (Path(binding.__file__), sources.digest(binding_helper_sha256)),
                               (Path(imports.declarations.__file__), sources.digest(declaration_helper_sha256)),
                               (Path(imports.__file__), sources.digest(import_helper_sha256)),
                               (Path(selector.__file__), sources.digest(builtin_helper_sha256)),
                               (Path(namespaces.__file__), sources.digest(namespace_helper_sha256)),
                               (Path(registry_plan.__file__), sources.digest(registry_helper_sha256)),
                               (helper, expected_helper)):
            sources.remaining(batch_started_at)
            require(hashlib.sha256(sources.read_regular(path)[0]).hexdigest() == expected)
        sources.remaining(batch_started_at)
        manifest = sources.scratch(manifest_path)
        require(hashlib.sha256(sources.read_regular(manifest)[0]).hexdigest() == hook.plan.manifest_sha256)
        require(type(hook.package) is ModuleType and overlay[binding.PACKAGE] is hook.package)
        package_values = hook.package.__dict__
        require(type(package_values.get('__name__')) is str and package_values['__name__'] == binding.PACKAGE
                and '__path__' not in package_values)
        for descriptor in hook.plan.modules:
            module = hook.guard_modules[descriptor.name]
            require(type(module) is ModuleType and overlay[descriptor.name] is module)
            values = module.__dict__
            require(set(values) == namespaces.EMPTY_KEYS | {'__builtins__'})
            for key, expected in (('__name__', descriptor.name), ('__package__', descriptor.package),
                                  ('__file__', descriptor.file)):
                require(type(values[key]) is str and values[key] == expected)
            require(all(values[key] is None for key in ('__doc__', '__loader__', '__spec__', '__cached__')))
            require(values['__builtins__'] is prepared.namespaces.builtin_bindings.builtin_references
                    and package_values.get(descriptor.name.rsplit('.', 1)[1]) is module)
        registry_plan.require_vacant(registry, binding.PACKAGE)
        require(set(registry) == set(snapshot) and all(registry[name] is value for name, value in snapshot.items()))
        require(set(overlay) == set(snapshot) | set(additions)
                and all(overlay[name] is value for name, value in snapshot.items()))
        sources.remaining(batch_started_at)
        return PreparedPrivateRegistry(prepared, MappingProxyType(overlay), expected_helper)
    except sources._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError('compressed private registry preparation rejected') from None
