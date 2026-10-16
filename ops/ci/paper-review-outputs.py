#!/usr/bin/env python3
"""Run the declared Lane B paper-reduction graph; never discover experiments."""
import argparse
from dataclasses import dataclass, field
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / 'research/benchmark/review_fixes/paper-build.json'
LOCK = REPO / 'ops/ci/paper-review-inputs.json'
LOCK_SHA256 = 'e17f3461f1d322abe036e9c7c80e5abf3b0534d91563ff4b6b30a7a914629b96'
NATIVE_BYTE_PIN = 'docs/whitepaper/receipts/native-byte-conformance-v1.json'
NATIVE_BYTE_CHECKER = 'docs/whitepaper/scripts/native_byte_conformance.py'
CONTROL_LEDGER = 'research/benchmark/results/review-fixes-controls-v1/scalar-cells.jsonl'
CONTROL_PANEL = 'research/benchmark/results/review-fixes-controls-v1/panel.json'
CONTROL_RENDERER = 'docs/whitepaper/scripts/review_controls.py'
PREFIX = 'research/benchmark/results/'
PRIVACY_LEDGER = PREFIX + 'review-fixes-privacy-v1/scalar-cells.jsonl'
PRIVACY_PANEL = PREFIX + 'review-fixes-privacy-v1/panel.json'
TABSYN_LEDGER = PREFIX + 'review-fixes-tabsyn-v1/scalar-cells.jsonl'
TABSYN_PANEL = PREFIX + 'review-fixes-tabsyn-v1/panel.json'
TABSYN_BINDING = PREFIX + 'review-fixes-tabsyn-v1/binding.json'
FIDELITY_LEDGER = PREFIX + 'review-fixes-tabsyn-fidelity-v1/scalar-cells.jsonl'
FIDELITY_PANEL = PREFIX + 'review-fixes-fidelity-v1/panel.json'
FIDELITY_RENDERER = 'docs/whitepaper/scripts/review_fidelity.py'
V2_LEDGER = PREFIX + 'review-fixes-v2/scalar-cells.jsonl'
V2_FITS = PREFIX + 'review-fixes-v2/fits.jsonl'
TABSYN_SEEDS_FITS = PREFIX + 'review-fixes-tabsyn-seeds-v2/fits.jsonl'
V2_PANEL = PREFIX + 'review-fixes-v2/panel.json'
V2_RECORD = PREFIX + 's3-lineage-record.json'
V2_PREDECLARE = 'research/benchmark/review_fixes/predeclare_v2.json'
V2_RENDERER = 'docs/whitepaper/scripts/review_v2_emit.py'
V2_FAMILY = PREFIX + 'review-fixes-v2/family.json'
V2_FAMILY_RENDERER = 'docs/whitepaper/scripts/review_v2_family.py'
V2_ALL_FIVE = PREFIX + 'review-fixes-v2/all-five.json'
V2_ALL_FIVE_RENDERER = 'docs/whitepaper/scripts/review_v2_all_five.py'
JSONL_INPUTS = (CONTROL_LEDGER, PRIVACY_LEDGER, TABSYN_LEDGER, FIDELITY_LEDGER, V2_LEDGER, V2_FITS,
                TABSYN_SEEDS_FITS)
OUTPUT_PREFIXES = ('review-', 'v2-')
_ADMISSION = object()
AUXILIARY_OUTPUTS = (
    'README.md', 'docs/readme/pareto.png', 'docs/readme/paired.png',
    'docs/whitepaper/figures/teaser.pdf', 'docs/whitepaper/figures/pareto.pdf',
    'docs/whitepaper/generated/teaser-figure-hashes.json',
    'docs/whitepaper/generated/pareto-figure-hashes.json',
    'docs/whitepaper/generated/readme-figure-hashes.json',
)


@dataclass(frozen=True)
class AuthenticatedPaperBuild:
    """Private, in-process admission; never a serialized authentication receipt.

    This binds trusted-driver control flow, not hostile Python callers.
    Bytes and tuples keep graph, code, inputs and expected renders immutable.
    """
    root: Path
    graph: bytes
    inputs: tuple
    code: tuple
    consumed: tuple
    expected: tuple
    _admission: object = field(repr=False)
    auxiliary_expected: tuple = ()
    source_identity: tuple = ()

    def revalidate(self):
        if self._admission is not _ADMISSION or self.root != REPO:
            raise ValueError('invalid paper build context')
        if source_identity(self.inputs + self.code) != self.source_identity:
            raise ValueError('stale paper build source identity')
        for name, raw in self.inputs + self.code + self.consumed:
            if safe_bytes(self.root, name) != raw:
                raise ValueError('stale paper build context')
        declared = validate_graph(json.loads(self.graph), dict(self.inputs))
        original = tuple((name, raw) for name, raw in self.consumed if name not in AUXILIARY_OUTPUTS)
        if code_snapshot() != self.code or consumed_snapshot(declared) != original:
            raise ValueError('stale paper build dependency inventory')


def safe_bytes(root, name):
    path = root / name
    if path.resolve() != root.resolve() / name or path.is_symlink():
        raise ValueError('redirected paper build dependency')
    return path.read_bytes()


def source_identity(files):
    identity = []
    for name, _ in files:
        stat = (REPO / name).stat(follow_symlinks=False)
        identity.append((name, stat.st_dev, stat.st_ino, stat.st_ctime_ns))
    return tuple(identity)


def validate_graph(graph, files):
    if (not isinstance(graph, dict) or graph.get('format') != 'dope-review-paper-build-v1'
            or not isinstance(graph.get('scripts'), list) or not graph['scripts']):
        raise ValueError('invalid review paper graph')
    declared, scripts = set(), set()
    for entry in graph['scripts']:
        if not isinstance(entry, dict) or set(entry) != {'path', 'args', 'check_args', 'outputs'}:
            raise ValueError('invalid review renderer declaration')
        script = Path(entry['path'])
        if (script.as_posix() != entry['path'] or script.parent.as_posix() != 'docs/whitepaper/scripts'
                or not script.name.startswith('review_') or script.suffix != '.py'
                or entry['path'] in scripts):
            raise ValueError('invalid review reduction script')
        scripts.add(entry['path'])
        args = entry['args']
        if (not isinstance(args, list) or len(args) != 2 or args[0] != '--from-panel'
                or args[1] not in files):
            raise ValueError('review renderer requires one independently pinned public panel')
        if entry['check_args'] not in (['--check'], ['--check-parser']):
            raise ValueError('invalid review renderer check arguments')
        if not isinstance(entry['outputs'], list) or not entry['outputs']:
            raise ValueError('incomplete review output declaration')
        for name in entry['outputs']:
            path = Path(name)
            if (path.as_posix() != name or path.parent.as_posix() not in
                    ('docs/whitepaper/generated', 'docs/whitepaper/figures')
                    or not path.name.startswith(OUTPUT_PREFIXES) or name in declared):
                raise ValueError('invalid review output path')
            declared.add(name)
    return declared


def authenticated(path, digest, size=None):
    if path.is_symlink():
        raise ValueError('symlinked review paper input')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest or (size is not None and len(raw) != size):
        raise ValueError('review paper input digest or length mismatch')
    return raw


def _load_authenticated_graph():
    if not GRAPH.is_file():
        raise ValueError('paper build requires a declared graph')
    lock_raw = authenticated(LOCK, LOCK_SHA256)
    lock = json.loads(lock_raw)
    if lock.get('format') != 'dope-paper-review-inputs-v1':
        raise ValueError('invalid review paper input lock')
    files = lock['files']
    graph_name = GRAPH.relative_to(REPO).as_posix()
    if graph_name not in files:
        raise ValueError('review graph has no input pin')
    inputs = {}
    for name, pin in files.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name
                or (relative.suffix != '.json' and name not in JSONL_INPUTS)
                or not (name == NATIVE_BYTE_PIN or name.startswith(
                    ('research/benchmark/results/', 'research/benchmark/review_fixes/')))):
            raise ValueError('invalid review paper input path')
        path = REPO / relative
        if path.resolve() != REPO.resolve() / relative:
            raise ValueError('redirected review paper input')
        if type(pin['bytes']) is not int or pin['bytes'] <= 0:
            raise ValueError('invalid review paper input length')
        inputs[name] = authenticated(path, pin['sha256'], pin['bytes'])
    # Every public dependency is authenticated before decoding the build graph.
    graph = json.loads(inputs[graph_name])
    validate_graph(graph, files)
    for entry in graph['scripts']:
        safe_bytes(REPO, entry['path'])
    return graph, inputs, lock_raw


def _authenticated_graph():
    graph, inputs, lock_raw = _load_authenticated_graph()
    replay_graph(graph, inputs)
    return graph, inputs, lock_raw


def replay_graph(graph, inputs):
    replayed = set()
    required_panels = {CONTROL_RENDERER: CONTROL_PANEL, FIDELITY_RENDERER: FIDELITY_PANEL,
                       V2_RENDERER: V2_PANEL, V2_FAMILY_RENDERER: V2_FAMILY,
                       V2_ALL_FIVE_RENDERER: V2_ALL_FIVE}
    for entry in graph['scripts']:
        panel = entry['args'][1]
        renderer = entry['path']
        if renderer in required_panels and panel != required_panels[renderer]:
            raise ValueError('review renderer requires its own independently pinned panel')
        if renderer == CONTROL_RENDERER:
            if CONTROL_LEDGER not in inputs:
                raise ValueError('control renderer requires the pinned scalar ledger and panel')
            kind = 'controls'
        elif renderer == FIDELITY_RENDERER:
            kind = 'fidelity'
        elif renderer == V2_RENDERER:
            kind = 'v2'
        elif renderer == V2_FAMILY_RENDERER:
            if V2_PANEL not in inputs:
                raise ValueError('v2 family renderer requires the pinned family file and v2 panel')
            kind = 'v2-family'
        elif renderer == V2_ALL_FIVE_RENDERER:
            if V2_PANEL not in inputs:
                raise ValueError('v2 all-five renderer requires the pinned all-five file and v2 panel')
            kind = 'v2-all-five'
        elif panel == PRIVACY_PANEL:
            kind = 'privacy'
        elif panel == TABSYN_PANEL:
            kind = 'tabsyn'
        else:
            continue
        # Only identical reducer contracts may share a replay. A generic or
        # differently bound renderer cannot mark another reducer as complete.
        key = (kind, panel)
        if key in replayed:
            continue
        if kind == 'controls':
            authenticated_controls(inputs)
        elif kind in ('privacy', 'tabsyn'):
            authenticated_bound(inputs, kind)
        elif kind == 'fidelity':
            authenticated_fidelity(inputs)
        elif kind == 'v2':
            authenticated_v2(inputs)
        elif kind == 'v2-all-five':
            authenticated_v2_reduction(inputs, 'v2_all_five', V2_ALL_FIVE)
        else:
            authenticated_v2_family(inputs)
        replayed.add(key)


def authenticated_graph():
    return _authenticated_graph()[0]


def review_inventory(root):
    return {p.relative_to(root).as_posix() for directory in ('generated', 'figures')
            for prefix in OUTPUT_PREFIXES
            for p in (root / 'docs/whitepaper' / directory).glob(prefix + '*') if p.is_file()}


def rendered_inventory(root):
    return {p.relative_to(root).as_posix() for directory in
            ('docs/whitepaper/generated', 'docs/whitepaper/figures', 'docs/readme')
            for p in (root / directory).rglob('*') if p.is_file()}


def code_snapshot():
    # Bind the transitive local Python implementation, including reducers and
    # renderer helpers. No numerical source, lock or result is modified.
    paths = set()
    for directory in ('research', 'docs/whitepaper/scripts'):
        paths.update(p for p in (REPO / directory).rglob('*.py')
                     if 'tests' not in p.parts and not p.name.startswith('test_'))
    paths.update((REPO / 'ops/ci').glob('paper-*.py'))
    # readme_benchmark.py reads the L3 byte cap from the embedded release contract.
    paths.update(REPO / name for name in ('docs/whitepaper/scripts/build_pdf.sh',
                                         'ops/ci/paper-architecture.tex',
                                         'production/kpi-contract.json'))
    mfs_panel = REPO / 'research/benchmark/results/mfs-v3-density-panel.json'
    if mfs_panel.is_file():
        paths.add(mfs_panel)
    return tuple((p.relative_to(REPO).as_posix(), safe_bytes(REPO, p.relative_to(REPO)))
                 for p in sorted(paths))


def consumed_snapshot(declared):
    paths = set()
    for directory in ('docs/whitepaper', 'docs/readme'):
        for p in (REPO / directory).rglob('*'):
            name = p.relative_to(REPO).as_posix()
            if (p.is_file() and 'scripts' not in p.parts and 'drafts' not in p.parts
                    and (p.parent.name in ('generated', 'figures', 'readme')
                         or p.suffix in ('.tex', '.bib', '.cls', '.sty', '.bst'))
                    and name not in declared and name not in AUXILIARY_OUTPUTS
                    and name not in ('docs/whitepaper/figures/architecture.pdf',
                                     'docs/whitepaper/generated/architecture-figure-hashes.json')):
                paths.add(p)
    return tuple((p.relative_to(REPO).as_posix(), safe_bytes(REPO, p.relative_to(REPO)))
                 for p in sorted(paths))


def authenticate_build(*, log_directory=None):
    code = code_snapshot()
    graph, inputs, lock_raw = _load_authenticated_graph()
    declared = validate_graph(graph, inputs)
    consumed = consumed_snapshot(declared)
    has_headline = 'docs/whitepaper/generated/v2-headline.json' in declared
    if has_headline:
        consumed += (('README.md', safe_bytes(REPO, 'README.md')),)
    captured_inputs = tuple(sorted(inputs.items())) + ((LOCK.relative_to(REPO).as_posix(), lock_raw),)
    verified = AuthenticatedPaperBuild(REPO, json.dumps(graph).encode(), captured_inputs,
                                       code, consumed, (), _ADMISSION,
                                       source_identity=source_identity(captured_inputs + code))
    verified.revalidate()
    # Render once from captured inputs/code in an isolated target directory.
    # This gives expected bytes, rather than trusting hashes of caller outputs.
    target = REPO / 'target/paper-render'
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as directory:
        root = Path(directory)
        for name, raw in verified.inputs + verified.code + tuple(
                (name, raw) for name, raw in verified.consumed if name == 'README.md'):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root / 'docs/whitepaper/generated').mkdir(parents=True, exist_ok=True)
        # Keep exact commands/exits and renderer failures after scratch cleanup.
        log_path = (target / (root.name + '.log') if log_directory is None
                    else log_directory / 'render.log')
        command_path = (target / (root.name + '.json') if log_directory is None
                        else log_directory / 'render-commands.json')
        commands = []

        def run_renderer(command, log, *, cwd=root):
            started = time.monotonic()
            result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            commands.append({'command': command, 'cwd': str(cwd), 'log': str(log_path),
                             'exit': result.returncode, 'elapsed_seconds': time.monotonic() - started})
            command_path.write_text(json.dumps(commands, indent=2) + '\n')
            result.check_returncode()

        with log_path.open('wb') as log:
            if NATIVE_BYTE_PIN in inputs:
                # Check only captured bytes in a separate scratch root; do not
                # seed the numerical render inventory with committed outputs.
                conformance_root = root / 'native-byte-conformance'
                captured = dict(verified.inputs + verified.code + verified.consumed)
                for name in (NATIVE_BYTE_PIN, NATIVE_BYTE_CHECKER, 'README.md',
                             'docs/whitepaper/supplement.tex',
                             'docs/whitepaper/generated/native-byte-conformance.tex'):
                    path = conformance_root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(captured[name])
                run_renderer([sys.executable, '-B', str(conformance_root / NATIVE_BYTE_CHECKER)],
                             log, cwd=conformance_root)
            # Numerical replay runs once from the same captured local code and
            # inputs as rendering. No live imported reducer, cached module or
            # caller-supplied authentication receipt is reused by the driver.
            loader = (
                "import importlib.util, pathlib, sys; "
                "root=pathlib.Path(sys.argv[1]); "
                "spec=importlib.util.spec_from_file_location('paper_admission',root/'ops/ci/paper-review-outputs.py'); "
                "gate=importlib.util.module_from_spec(spec); sys.modules[spec.name]=gate; "
                "spec.loader.exec_module(gate); "
                "gate.LOCK=root/sys.argv[2]; gate.LOCK_SHA256=sys.argv[3]; "
                "gate.authenticated_graph()"
            )
            run_renderer([sys.executable, '-B', '-c', loader, str(root),
                          LOCK.relative_to(REPO).as_posix(), LOCK_SHA256], log)
            for entry in graph['scripts']:
                prior = {name: safe_bytes(root, name) for name in review_inventory(root)}
                if entry['check_args'] == ['--check-parser']:
                    run_renderer([sys.executable, '-B', str(root / entry['path']),
                                  *entry['args'], *entry['check_args']], log)
                run_renderer([sys.executable, '-B', str(root / entry['path']), *entry['args']], log)
                actual = review_inventory(root)
                if (actual != set(prior) | set(entry['outputs'])
                        or any(safe_bytes(root, name) != raw for name, raw in prior.items())):
                    raise ValueError('review renderer differs from actual rendered inventory')
        if rendered_inventory(root) != declared:
            raise ValueError('review build graph differs from actual rendered inventory')
        expected = tuple((name, safe_bytes(root, name)) for name in sorted(declared))
        auxiliary = ()
        if has_headline:
            with log_path.open('ab') as log:
                for script, args in [('v2_figures.py', []),
                                     ('readme_benchmark.py', ['--write-readme'])]:
                    run_renderer([sys.executable, '-B', str(root / 'docs/whitepaper/scripts' / script),
                                  *args], log)
            auxiliary = tuple((name, safe_bytes(root, name)) for name in AUXILIARY_OUTPUTS)
            if dict(auxiliary)['README.md'] != dict(consumed)['README.md']:
                raise ValueError('README differs from expected rendering')
            if (rendered_inventory(root) != declared | (set(AUXILIARY_OUTPUTS) - {'README.md'})
                    or any(safe_bytes(root, name) != raw for name, raw in expected)):
                raise ValueError('auxiliary renderer changed expected review inventory')
    verified.revalidate()
    return AuthenticatedPaperBuild(verified.root, verified.graph, verified.inputs,
                                   verified.code, verified.consumed, expected, _ADMISSION, auxiliary,
                                   verified.source_identity)


def render_review_outputs(verified, *, check):
    if not isinstance(verified, AuthenticatedPaperBuild):
        raise ValueError('invalid paper build context')
    verified.revalidate()
    declared = validate_graph(json.loads(verified.graph), dict(verified.inputs))
    if {name for name, _ in verified.expected} != declared:
        raise ValueError('incomplete expected review inventory')
    if review_inventory(REPO) - declared:
        raise ValueError('review build graph does not cover every committed output')
    for name, _ in verified.expected:
        if (REPO / name).resolve() != REPO.resolve() / name:
            raise ValueError('redirected review output')
    for name, raw in verified.expected:
        if check:
            if safe_bytes(REPO, name) != raw:
                raise ValueError('generated review output differs from expected render')
        else:
            path = REPO / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    if review_inventory(REPO) != declared:
        raise ValueError('missing declared review output')
    verified.revalidate()


def authenticated_controls(inputs):
    # Decode only authenticated public bytes, and reject an open test flag before
    # importing the reducer. Never reopen a ledger after checking its digest.
    cells = [json.loads(line) for line in inputs[CONTROL_LEDGER].splitlines()]
    if not cells:
        raise ValueError('control ledger is empty')
    if any(not isinstance(cell, dict) or cell.get('official_tests_opened') is not False
           for cell in cells):
        raise ValueError('control ledger official test flag is not closed')
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from research.benchmark.review_fixes.bind_control_cells import (
        assert_public_scalar, canonical_panel, identity_of,
    )
    from docs.whitepaper.scripts.review_controls import payload_from_cells

    seen = set()
    for cell in cells:
        assert_public_scalar(cell)
        identity = identity_of(cell)
        if identity in seen:
            raise ValueError('control ledger contains a duplicate identity')
        seen.add(identity)
    recomputed = canonical_panel(payload_from_cells(cells)).encode()
    if recomputed != inputs[CONTROL_PANEL]:
        raise ValueError('control scalar ledger does not reproduce the pinned panel')


def closed_ledger(raw):
    cells = [json.loads(line) for line in raw.splitlines()]
    if not cells or any(not isinstance(c, dict) or c.get('official_tests_opened') is not False
                        or c.get('formal_dp') is not False for c in cells):
        raise ValueError('public scalar ledger official test or formal privacy flag refused')


def authenticated_bound(inputs, kind):
    ledger, panel = (PRIVACY_LEDGER, PRIVACY_PANEL) if kind == 'privacy' else (TABSYN_LEDGER, TABSYN_PANEL)
    if ledger not in inputs:
        raise ValueError('bound renderer requires its pinned scalar ledger')
    closed_ledger(inputs[ledger])
    if kind == 'tabsyn' and TABSYN_BINDING not in inputs:
        raise ValueError('TabSyn renderer requires its pinned binding')
    binding = json.loads(inputs[TABSYN_BINDING]) if kind == 'tabsyn' else None
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from research.benchmark.review_fixes.bind_review_cells import canonical, replay
    recomputed = canonical(replay(kind, inputs[ledger], inputs, binding)).encode()
    if recomputed != inputs[panel]:
        raise ValueError('public scalar ledger does not reproduce its pinned panel')


def authenticated_fidelity(inputs):
    if FIDELITY_LEDGER not in inputs:
        raise ValueError('fidelity renderer requires its pinned scalar ledger')
    closed_ledger(inputs[FIDELITY_LEDGER])
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from docs.whitepaper.scripts.review_fidelity import INPUTS, payload_from_inputs
    from research.benchmark.review_fixes.bind_review_cells import canonical
    if any(name not in inputs for name in INPUTS):
        raise ValueError('fidelity renderer requires every independently pinned source')
    if canonical(payload_from_inputs(inputs)).encode() != inputs[FIDELITY_PANEL]:
        raise ValueError('fidelity scalar sources do not reproduce the pinned panel')


def authenticated_v2(inputs):
    """Recompute the five-seed panel from its pinned scalar and fit ledgers before any v2 output is written."""
    names = (V2_LEDGER, V2_FITS, V2_RECORD, V2_PREDECLARE)
    if any(name not in inputs for name in names):
        raise ValueError('v2 renderer requires its pinned ledgers, lineage record and predeclaration')
    cells = [json.loads(line) for line in inputs[V2_LEDGER].splitlines()]
    fits = [json.loads(line) for line in inputs[V2_FITS].splitlines()]
    if not cells or any(not isinstance(item, dict) or item.get('official_tests_opened') is not False
                        for item in cells + fits):
        raise ValueError('v2 ledger official test flag is not closed')
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from research.benchmark.review_fixes import v2_panel
    v2_panel.check_fits(cells, fits)
    panel = v2_panel.reduce(cells, v2_panel.names_of(json.loads(inputs[V2_RECORD])),
                            predeclare=inputs[V2_PREDECLARE], fits=fits)
    if v2_panel.canonical(panel).encode() != inputs[V2_PANEL]:
        raise ValueError('v2 scalar ledger does not reproduce the pinned panel')


def authenticated_v2_family(inputs):
    """Recompute the source-family sensitivity from the pinned v2 scalar ledger before it is rendered."""
    names = (V2_LEDGER, V2_RECORD, V2_PREDECLARE)
    if any(name not in inputs for name in names):
        raise ValueError('v2 family renderer requires its pinned ledger, lineage record and predeclaration')
    cells = [json.loads(line) for line in inputs[V2_LEDGER].splitlines()]
    if not cells or any(not isinstance(item, dict) or item.get('official_tests_opened') is not False
                        for item in cells):
        raise ValueError('v2 ledger official test flag is not closed')
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from research.benchmark.review_fixes import v2_family, v2_panel
    family = v2_family.reduce(cells, v2_panel.names_of(json.loads(inputs[V2_RECORD])),
                              predeclare=inputs[V2_PREDECLARE])
    if v2_panel.canonical(family).encode() != inputs[V2_FAMILY]:
        raise ValueError('v2 scalar ledger does not reproduce the pinned family file')


def authenticated_v2_reduction(inputs, module, output):
    """Recompute one v2 sensitivity file from the pinned v2 scalar ledger before it is rendered."""
    names = (V2_LEDGER, V2_RECORD, V2_PREDECLARE)
    if any(name not in inputs for name in names):
        raise ValueError('v2 sensitivity renderer requires its pinned ledger, lineage record and predeclaration')
    cells = [json.loads(line) for line in inputs[V2_LEDGER].splitlines()]
    if not cells or any(not isinstance(item, dict) or item.get('official_tests_opened') is not False
                        for item in cells):
        raise ValueError('v2 ledger official test flag is not closed')
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import importlib
    from research.benchmark.review_fixes import v2_panel
    reducer = importlib.import_module(f'research.benchmark.review_fixes.{module}')
    result = reducer.reduce(cells, v2_panel.names_of(json.loads(inputs[V2_RECORD])), predeclare=inputs[V2_PREDECLARE])
    if v2_panel.canonical(result).encode() != inputs[output]:
        raise ValueError(f'v2 scalar ledger does not reproduce the pinned {module} output')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--check-inputs', action='store_true',
                        help='authenticate and replay inputs without writing paper outputs')
    args = parser.parse_args()
    if args.check_inputs:
        authenticated_graph()
        print('review paper inputs authenticated and declared scalar panels reproduced')
        return
    verified = authenticate_build()
    render_review_outputs(verified, check=args.check)


if __name__ == '__main__':
    main()
