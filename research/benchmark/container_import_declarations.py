"""Plan declared startup guard imports without resolving or initializing modules."""
import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path

from . import container_module_bindings as binding

STDLIB_NAMES = frozenset({'collections', 'dataclasses', 'fnmatch', 'hashlib', 'json',
                        'math', 'os', 'pathlib', 're', 'stat', 'struct', 'time'})


def require(condition):
    if not condition:
        raise ValueError('compressed guard import declarations rejected')


@dataclass(frozen=True)
class StartupImportDeclarations:
    modules: tuple
    imports: tuple
    initialization_order: tuple
    manifest_sha256: str
    import_helper_sha256: str
    outer_seconds_remaining: float

    def receipt(self):
        return {'format': 'dope-compressed-startup-import-declarations', 'version': 1,
                'manifest_sha256': self.manifest_sha256,
                'import_helper_sha256': self.import_helper_sha256,
                'source_files_planned': len(self.modules),
                'declared_imports': sum(len(rows) for _, rows in self.imports),
                'outer_seconds_remaining': self.outer_seconds_remaining,
                'actual_import_resolution_verified': False, 'source_initializers_run': False,
                'modules_installed': False, 'candidate_processes_started': 0,
                'execution_admitted': False, 'full_runtime_closure_certified': False,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'release_safe': None, 'superiority': None}


def prepare_import_declarations(manifest_path, manifest_sha256, compiler_sha256,
                                binding_helper_sha256, import_helper_sha256, *, batch_started_at):
    """Caller already owns four seed helpers and complete interpreter/import closure.

    Inspect static declarations in owned bytes only. This does not intercept an
    import, resolve a standard-library member or admit a module initializer.
    """
    compiler = binding.compiler
    sources = compiler.sources
    try:
        sources.remaining(batch_started_at)
        expected_helper = sources.digest(import_helper_sha256)
        helper = sources.unaliased(Path(__file__))
        require(hashlib.sha256(sources.read_regular(helper)[0]).hexdigest() == expected_helper)
        try:
            plan = binding.prepare_module_bindings(manifest_path, manifest_sha256, compiler_sha256,
                binding_helper_sha256, batch_started_at=batch_started_at)
            owned = sources.verify_startup_sources(manifest_path, manifest_sha256,
                batch_started_at=batch_started_at)
        except TimeoutError:
            # Trusted guards reserve their public TimeoutError for the original deadline.
            raise sources._DeadlineExhausted() from None
        require(owned.manifest_sha256 == plan.manifest_sha256)
        imports, dependencies = [], {}
        for (filename, body), module in zip(owned.sources, plan.modules, strict=True):
            sources.remaining(batch_started_at)
            require(module.name == binding.PACKAGE + '.' + filename[:-3]
                    and hashlib.sha256(body).hexdigest() == module.source_sha256)
            rows, edges = [], set()
            for node in ast.walk(ast.parse(body, filename='<owned guard import declarations>')):
                if isinstance(node, ast.Import):
                    for item in node.names:
                        require(item.name in STDLIB_NAMES)
                        rows.append(('absolute', item.name, None, item.asname))
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 1:
                        require(node.module is None)
                        for item in node.names:
                            require(item.name + '.py' in sources.SOURCE_NAMES)
                            target = binding.PACKAGE + '.' + item.name
                            edges.add(target)
                            rows.append(('relative', target, None, item.asname))
                    else:
                        require(node.level == 0 and node.module in STDLIB_NAMES)
                        for item in node.names:
                            require(item.name != '*')
                            rows.append(('absolute', node.module, item.name, item.asname))
            imports.append((module.name, tuple(rows)))
            dependencies[module.name] = edges
            sources.remaining(batch_started_at)
        require(len(dependencies) == len(sources.SOURCE_NAMES))
        order, pending = [], dict(dependencies)
        while pending:
            sources.remaining(batch_started_at)
            ready = sorted(name for name, edges in pending.items() if edges <= set(order))
            require(ready)
            order.extend(ready)
            for name in ready: del pending[name]
        try:
            require(sources.verify_startup_sources(manifest_path, manifest_sha256,
                batch_started_at=batch_started_at).sources == owned.sources)
        except TimeoutError:
            raise sources._DeadlineExhausted() from None
        for path, expected in ((Path(binding.__file__), sources.digest(binding_helper_sha256)),
                               (Path(compiler.__file__), sources.digest(compiler_sha256)),
                               (helper, expected_helper)):
            sources.remaining(batch_started_at)
            require(hashlib.sha256(sources.read_regular(path)[0]).hexdigest() == expected)
        sources.remaining(batch_started_at)
        manifest = sources.scratch(manifest_path)
        require(hashlib.sha256(sources.read_regular(manifest)[0]).hexdigest() == plan.manifest_sha256)
        remaining = sources.remaining(batch_started_at)
        return StartupImportDeclarations(plan.modules, tuple(imports), tuple(order),
            plan.manifest_sha256, expected_helper, remaining)
    except sources._DeadlineExhausted:
        raise TimeoutError('compressed bootstrap batch deadline exhausted') from None
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, OverflowError, SyntaxError):
        raise ValueError('compressed guard import declarations rejected') from None
