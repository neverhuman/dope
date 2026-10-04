"""Prepare declared imports against preowned references without module initialization."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType, ModuleType

from . import container_import_declarations as declarations


def require(condition):
    if not condition:
        raise ValueError('compressed guard import hook rejected')


@dataclass(frozen=True)
class PreparedImportHook:
    plan: object
    stdlib_modules: object
    guard_modules: object
    package: object
    batch_started_at: float
    helper_sha256: str

    def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
        sources = declarations.binding.compiler.sources
        try:
            sources.remaining(self.batch_started_at)
            require(type(name) is str and type(level) is int and level in (0, 1))
            require(type(globals) is dict and (fromlist is None or type(fromlist) is tuple))
            members = () if fromlist is None else fromlist
            require(all(type(member) is str and member != '*' for member in members))
            caller = globals.get('__name__')
            require(type(caller) is str and caller in self.guard_modules)
            require(globals is self.guard_modules[caller].__dict__)
            descriptor = next(module for module in self.plan.modules if module.name == caller)
            require(type(globals.get('__package__')) is str and globals['__package__'] == descriptor.package)
            require(type(globals.get('__file__')) is str and globals['__file__'] == descriptor.file)
            rows = dict(self.plan.imports)[caller]
            if level == 0:
                require(name in self.stdlib_modules)
                if members:
                    require(all(any(row[:3] == ('absolute', name, member) for row in rows)
                                for member in members))
                    require(all(member in self.stdlib_modules[name].__dict__ for member in members))
                else:
                    require(any(row[:3] == ('absolute', name, None) for row in rows))
                result = self.stdlib_modules[name]
            else:
                require(name == '' and members)
                for member in members:
                    target = descriptor.package + '.' + member
                    require(any(row[:3] == ('relative', target, None) for row in rows))
                    require(target in self.guard_modules
                            and self.package.__dict__.get(member) is self.guard_modules[target])
                result = self.package
            sources.remaining(self.batch_started_at)
            return result
        except sources._DeadlineExhausted:
            raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
        except (KeyError, TypeError, ValueError, RuntimeError, StopIteration):
            raise ValueError('compressed guard import hook rejected') from None

    def receipt(self):
        return {'format': 'dope-compressed-startup-import-hook', 'version': 1,
                'manifest_sha256': self.plan.manifest_sha256, 'helper_sha256': self.helper_sha256,
                'guard_namespaces_prepared': len(self.guard_modules),
                'stdlib_references_supplied': len(self.stdlib_modules),
                'source_initializers_run': False, 'modules_installed': False,
                'actual_import_resolution_verified': False, 'candidate_processes_started': 0,
                'execution_admitted': False, 'full_runtime_closure_certified': False,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}


def prepare_import_hook(manifest_path, manifest_sha256, compiler_sha256, binding_helper_sha256,
                        declaration_helper_sha256, helper_sha256, stdlib_modules, *, batch_started_at):
    """Caller owns five seed helpers, interpreter closure and supplied module references.

    No standard-library identity proof follows from module names or self-file hashes.
    Empty namespaces and this hook prepare later initialization; no initializer runs.
    """
    binding = declarations.binding
    compiler = binding.compiler
    sources = compiler.sources
    try:
        sources.remaining(batch_started_at)
        expected_helper = sources.digest(helper_sha256)
        helper = sources.unaliased(Path(__file__))
        require(hashlib.sha256(sources.read_regular(helper)[0]).hexdigest() == expected_helper)
        require(type(stdlib_modules) is dict and all(type(name) is str for name in stdlib_modules))
        references = dict(stdlib_modules)
        require(set(references) == declarations.STDLIB_NAMES)
        for name, module in references.items():
            require(type(module) is ModuleType and type(module.__dict__.get('__name__')) is str
                    and module.__dict__['__name__'] == name)
        try:
            plan = declarations.prepare_import_declarations(manifest_path, manifest_sha256,
                compiler_sha256, binding_helper_sha256, declaration_helper_sha256,
                batch_started_at=batch_started_at)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        package = ModuleType(binding.PACKAGE)
        modules = {}
        for descriptor in plan.modules:
            sources.remaining(batch_started_at)
            module = ModuleType(descriptor.name)
            module.__dict__.update(__package__=descriptor.package, __file__=descriptor.file,
                                   __cached__=None, __loader__=None)
            modules[descriptor.name] = module
            package.__dict__[descriptor.name.rsplit('.', 1)[1]] = module
        for path, expected in ((Path(binding.__file__), sources.digest(binding_helper_sha256)),
                               (Path(compiler.__file__), sources.digest(compiler_sha256)),
                               (Path(declarations.__file__), sources.digest(declaration_helper_sha256)),
                               (helper, expected_helper)):
            sources.remaining(batch_started_at)
            require(hashlib.sha256(sources.read_regular(path)[0]).hexdigest() == expected)
        sources.remaining(batch_started_at)
        manifest = sources.scratch(manifest_path)
        require(hashlib.sha256(sources.read_regular(manifest)[0]).hexdigest() == plan.manifest_sha256)
        sources.remaining(batch_started_at)
        return PreparedImportHook(plan, MappingProxyType(references),
            MappingProxyType(modules), package, batch_started_at, expected_helper)
    except sources._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError('compressed guard import hook rejected') from None
