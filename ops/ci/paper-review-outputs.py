#!/usr/bin/env python3
"""Run the declared Lane B paper-reduction graph; never discover experiments."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / 'research/benchmark/review_fixes/paper-build.json'
LOCK = REPO / 'ops/ci/paper-review-inputs.json'
LOCK_SHA256 = '6b35b5124c1216e75c1ca200c44a909986b5a265551a041676063843167729b2'


class Replay(NamedTuple):
    """A renderer whose pinned panel must be recomputed from a pinned scalar ledger.

    reduce(cells, extras) receives the decoded ledger and the authenticated bytes
    of extras, in order, and returns the canonical panel bytes.
    """
    label: str
    panel: str
    ledger: str
    extras: tuple
    reduce: Callable


def unique_cells(cells, check, identity_of, label):
    seen = set()
    for cell in cells:
        check(cell)
        identity = identity_of(cell)
        if identity in seen:
            raise ValueError(f'{label} ledger contains a duplicate identity')
        seen.add(identity)


def replay_controls(cells, _extras):
    from research.benchmark.review_fixes.bind_control_cells import (
        assert_public_scalar, canonical_panel, identity_of,
    )
    from docs.whitepaper.scripts.review_controls import payload_from_cells

    unique_cells(cells, assert_public_scalar, identity_of, 'control')
    return canonical_panel(payload_from_cells(cells)).encode()


def replay_privacy(cells, _extras):
    from research.benchmark.review_fixes.bind_privacy_cells import replay

    return replay(cells).encode()


def replay_tabsyn(cells, extras):
    from research.benchmark.review_fixes import bind_tabsyn_cells as bind

    receipts, gate, eight_panel = extras
    return bind.replay(cells, bind.decode_jsonl(receipts, 'TabSyn evaluator receipts'),
                       json.loads(gate), json.loads(eight_panel)).encode()


TABSYN = 'research/benchmark/results/review-fixes-tabsyn-v1/'
REPLAYS = {
    'docs/whitepaper/scripts/review_controls.py': Replay(
        'control',
        'research/benchmark/results/review-fixes-controls-v1/panel.json',
        'research/benchmark/results/review-fixes-controls-v1/scalar-cells.jsonl',
        (), replay_controls),
    'docs/whitepaper/scripts/review_privacy.py': Replay(
        'privacy',
        'research/benchmark/results/review-fixes-privacy-v1/panel.json',
        'research/benchmark/results/review-fixes-privacy-v1/scalar-cells.jsonl',
        (), replay_privacy),
    'docs/whitepaper/scripts/review_tabsyn.py': Replay(
        'TabSyn',
        TABSYN + 'panel.json',
        TABSYN + 'scalar-cells.jsonl',
        (TABSYN + 'evaluator-replay.jsonl', TABSYN + 'parser-check.json',
         'research/benchmark/results/tabsyn-eight-lineage-validation/panel.json'),
        replay_tabsyn),
}


def registered_ledgers():
    """The only line-delimited inputs the lock may pin."""
    return {name for spec in REPLAYS.values() for name in (spec.ledger, *spec.extras)
            if name.endswith('.jsonl')}


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
    ledgers = registered_ledgers()
    inputs = {}
    for name, pin in files.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name
                or (relative.suffix != '.json' and name not in ledgers)
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
        spec = REPLAYS.get(entry['path'])
        if spec is not None:
            if args[1] != spec.panel or any(name not in inputs for name in (spec.ledger, *spec.extras)):
                raise ValueError(f'{spec.label} renderer requires the pinned scalar ledger and panel')
            authenticated_replay(spec, inputs)
    return graph


def closed_records(raw, label):
    records = [json.loads(line) for line in raw.splitlines()]
    if not records:
        raise ValueError(f'{label} ledger is empty')
    if any(not isinstance(record, dict) or record.get('official_tests_opened') is not False
           for record in records):
        raise ValueError(f'{label} ledger official test flag is not closed')
    return records


def authenticated_replay(spec, inputs):
    # Decode only authenticated public bytes, and reject an open test flag before
    # importing the reducer. Never reopen a ledger after checking its digest.
    cells = closed_records(inputs[spec.ledger], spec.label)
    for name in spec.extras:
        if name.endswith('.jsonl'):
            closed_records(inputs[name], f'{spec.label} {Path(name).stem}')
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    recomputed = spec.reduce(cells, tuple(inputs[name] for name in spec.extras))
    if recomputed != inputs[spec.panel]:
        raise ValueError(f'{spec.label} scalar ledger does not reproduce the pinned panel')


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
        labels = [REPLAYS[entry['path']].label for entry in graph['scripts'] if entry['path'] in REPLAYS]
        print('review paper inputs authenticated; scalar panels reproduced: ' + (', '.join(labels) or 'none'))
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
