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
LOCK_SHA256 = '36738416d2913c68c13a853d735011453e6740a40817032547ca702f9025528a'
CONTROL_LEDGER = 'research/benchmark/results/review-fixes-controls-v1/scalar-cells.jsonl'
CONTROL_PANEL = 'research/benchmark/results/review-fixes-controls-v1/panel.json'
CONTROL_RENDERER = 'docs/whitepaper/scripts/review_controls.py'


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
                or (relative.suffix != '.json' and name != CONTROL_LEDGER)
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
    for entry in graph['scripts']:
        args = entry.get('args', [])
        if len(args) != 2 or args[0] != '--from-panel' or args[1] not in files:
            raise ValueError('review renderer requires one independently pinned public panel')
        if entry['path'] == CONTROL_RENDERER:
            if args[1] != CONTROL_PANEL or CONTROL_LEDGER not in inputs:
                raise ValueError('control renderer requires the pinned scalar ledger and panel')
            authenticated_controls(inputs)
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
        print('review paper inputs authenticated; control scalar panel reproduced')
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
                    or not path.name.startswith('review-')):
                raise ValueError('invalid review output path')
            if not (REPO / path).is_file():
                raise ValueError('missing declared review output')
            declared.add(path.as_posix())
    actual = {p.relative_to(REPO).as_posix() for directory in ('generated', 'figures')
              for p in (REPO / 'docs/whitepaper' / directory).glob('review-*') if p.is_file()}
    if actual != declared:
        raise ValueError('review build graph does not cover every committed output')


if __name__ == '__main__':
    main()
