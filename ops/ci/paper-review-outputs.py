#!/usr/bin/env python3
"""Run the declared Lane B paper-reduction graph; never discover experiments."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / 'research/benchmark/review_fixes/paper-build.json'
LOCK = REPO / 'ops/ci/paper-review-inputs.json'
LOCK_SHA256 = '49cf19ab42257408dbbeaeffcbaedabcebd2352df1f930ecb74b430e91c0120c'
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
V2_PANEL = PREFIX + 'review-fixes-v2/panel.json'
V2_RECORD = PREFIX + 's3-lineage-record.json'
V2_PREDECLARE = 'research/benchmark/review_fixes/predeclare_v2.json'
V2_RENDERER = 'docs/whitepaper/scripts/review_v2_emit.py'
JSONL_INPUTS = (CONTROL_LEDGER, PRIVACY_LEDGER, TABSYN_LEDGER, FIDELITY_LEDGER, V2_LEDGER, V2_FITS)
OUTPUT_PREFIXES = ('review-', 'v2-')


def authenticated(path, digest, size=None):
    if path.is_symlink():
        raise ValueError('symlinked review paper input')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest or (size is not None and len(raw) != size):
        raise ValueError('review paper input digest or length mismatch')
    return raw


def authenticated_graph():
    lock = json.loads(authenticated(LOCK, LOCK_SHA256))
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
                or not name.startswith(('research/benchmark/results/', 'research/benchmark/review_fixes/'))):
            raise ValueError('invalid review paper input path')
        path = REPO / relative
        if path.resolve() != REPO.resolve() / relative:
            raise ValueError('redirected review paper input')
        if type(pin['bytes']) is not int or pin['bytes'] <= 0:
            raise ValueError('invalid review paper input length')
        inputs[name] = authenticated(path, pin['sha256'], pin['bytes'])
    # Every public dependency is authenticated before decoding the build graph.
    graph = json.loads(inputs[graph_name])
    replayed = set()
    for entry in graph['scripts']:
        args = entry.get('args', [])
        if len(args) != 2 or args[0] != '--from-panel' or args[1] not in files:
            raise ValueError('review renderer requires one independently pinned public panel')
        if entry['path'] == CONTROL_RENDERER:
            if args[1] != CONTROL_PANEL or CONTROL_LEDGER not in inputs:
                raise ValueError('control renderer requires the pinned scalar ledger and panel')
            authenticated_controls(inputs)
        panel = args[1]
        if panel in replayed:
            continue
        if panel == PRIVACY_PANEL:
            authenticated_bound(inputs, 'privacy')
        elif panel == TABSYN_PANEL:
            authenticated_bound(inputs, 'tabsyn')
        elif entry['path'] == FIDELITY_RENDERER:
            if panel != FIDELITY_PANEL:
                raise ValueError('fidelity renderer requires the pinned fidelity panel')
            authenticated_fidelity(inputs)
        elif entry['path'] == V2_RENDERER:
            if panel != V2_PANEL:
                raise ValueError('v2 renderer requires the pinned v2 panel')
            authenticated_v2(inputs)
        replayed.add(panel)
    return graph


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--check-inputs', action='store_true',
                        help='authenticate and replay inputs without writing paper outputs')
    args = parser.parse_args()
    if not GRAPH.is_file():
        if list((REPO / 'docs/whitepaper/generated').glob('review-*')):
            raise ValueError('review outputs require a declared paper-build graph')
        print('review paper graph has no committed outputs yet')
        return
    graph = authenticated_graph()
    if graph.get('format') != 'dope-review-paper-build-v1':
        raise ValueError('invalid review paper graph')
    if args.check_inputs:
        print('review paper inputs authenticated; control scalar panel reproduced; declared wave2 and v2 scalar panels reproduced')
        return
    declared = set()
    for entry in graph['scripts']:
        script = Path(entry['path'])
        if (script.parent.as_posix() != 'docs/whitepaper/scripts'
                or not script.name.startswith('review_') or script.suffix != '.py'):
            raise ValueError('invalid review reduction script')
        command = [sys.executable, str(REPO / script), *entry.get('args', [])]
        if args.check:
            command.extend(entry['check_args'])
        subprocess.run(command, cwd=REPO, check=True)
        for relative in entry['outputs']:
            path = Path(relative)
            if (path.parent.as_posix() not in ('docs/whitepaper/generated', 'docs/whitepaper/figures')
                    or not path.name.startswith(OUTPUT_PREFIXES)):
                raise ValueError('invalid review output path')
            if not (REPO / path).is_file():
                raise ValueError('missing declared review output')
            declared.add(path.as_posix())
    actual = {p.relative_to(REPO).as_posix() for directory in ('generated', 'figures')
              for prefix in OUTPUT_PREFIXES
              for p in (REPO / 'docs/whitepaper' / directory).glob(prefix + '*') if p.is_file()}
    if actual != declared:
        raise ValueError('review build graph does not cover every committed output')


if __name__ == '__main__':
    main()
