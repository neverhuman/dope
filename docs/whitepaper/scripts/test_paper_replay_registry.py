"""Replay-registry admission for every registered renderer, on public toy fixtures."""

import builtins
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'paper_review_replay_registry', ROOT / 'ops/ci/paper-review-outputs.py')
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)
GRAPH_NAME = 'research/benchmark/review_fixes/paper-build.json'
REDUCER_PACKAGES = ('research.benchmark.review_fixes', 'docs.whitepaper.scripts')


def toy_cell(**overrides):
    cell = {'dataset': 'public-toy', 'official_tests_opened': False, 'value': 0.5}
    cell.update(overrides)
    return cell


class ReplayRegistryTests(unittest.TestCase):
    def setUp(self):
        target = ROOT / 'target/paper-replay-registry-tests'
        target.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reducers, replays, scripts = {}, {}, []
        for renderer, spec in GATE.REPLAYS.items():
            self.write(spec.panel, json.dumps({'renderer': renderer}, sort_keys=True) + '\n')
            self.write_cells(spec.ledger, [toy_cell()])
            for name in spec.extras:
                self.write(name, json.dumps(toy_cell()) + '\n' if name.endswith('.jsonl') else '{}\n')
            reduce = Mock(side_effect=lambda cells, extras, panel=spec.panel: (self.root / panel).read_bytes())
            self.reducers[renderer] = reduce
            replays[renderer] = spec._replace(reduce=reduce)
            scripts.append({'path': renderer, 'args': ['--from-panel', spec.panel],
                            'check_args': ['--check'], 'outputs': []})
        self.graph = {'format': 'dope-review-paper-build-v1', 'scripts': scripts}
        self.write(GRAPH_NAME, json.dumps(self.graph))
        self.lock = self.root / 'lock.json'
        self.registry = replays
        for name, value in [('REPO', self.root), ('GRAPH', self.root / GRAPH_NAME),
                            ('LOCK', self.lock), ('REPLAYS', replays)]:
            p = patch.object(GATE, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.relock()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_cells(self, name, cells):
        self.write(name, ''.join(json.dumps(cell, sort_keys=True) + '\n' for cell in cells))

    def pinned_names(self):
        names = {GRAPH_NAME}
        for spec in self.registry.values():
            names.update((spec.panel, spec.ledger, *spec.extras))
        return names

    def relock(self, names=None):
        files = {}
        for name in sorted(self.pinned_names() if names is None else names):
            raw = (self.root / name).read_bytes()
            files[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        raw = (json.dumps({'format': 'dope-paper-review-inputs-v1', 'files': files}) + '\n').encode()
        self.lock.write_bytes(raw)
        p = patch.object(GATE, 'LOCK_SHA256', hashlib.sha256(raw).hexdigest())
        p.start()
        self.addCleanup(p.stop)

    def refuse(self, reason):
        with patch.object(GATE.subprocess, 'run') as render:
            with patch('sys.argv', ['paper-review-outputs.py']):
                with self.assertRaisesRegex(ValueError, reason):
                    GATE.main()
            render.assert_not_called()

    def test_every_registered_panel_is_replayed_from_its_ledger(self):
        self.assertEqual(GATE.authenticated_graph(), self.graph)
        for renderer, reduce in self.reducers.items():
            spec = self.registry[renderer]
            extras = tuple((self.root / name).read_bytes() for name in spec.extras)
            reduce.assert_called_once_with([toy_cell()], extras)

    def test_registry_names_the_four_review_renderers(self):
        self.assertEqual({spec.label for spec in GATE.REPLAYS.values()}, {'control', 'privacy', 'TabSyn', 'v2'})
        self.assertIn('research/benchmark/results/review-fixes-tabsyn-v1/evaluator-replay.jsonl',
                      GATE.registered_ledgers())

    def test_unregistered_jsonl_is_refused_even_beside_a_registered_ledger(self):
        name = 'research/benchmark/results/review-fixes-tabsyn-v1/scalar-cells.v2.jsonl'
        self.write_cells(name, [toy_cell()])
        self.relock(self.pinned_names() | {name})
        self.refuse('invalid review paper input path')
        for reduce in self.reducers.values():
            reduce.assert_not_called()

    def test_ledger_hash_mismatch_is_refused_before_decode(self):
        for spec in self.registry.values():
            with self.subTest(ledger=spec.ledger):
                path = self.root / spec.ledger
                saved = path.read_bytes()
                path.write_bytes(saved + b' ')
                loads = json.loads
                with patch.object(GATE.json, 'loads', wraps=loads) as decode:
                    self.refuse('digest or length mismatch')
                    self.assertEqual(decode.call_count, 1)  # The authenticated lock only.
                path.write_bytes(saved)
        for reduce in self.reducers.values():
            reduce.assert_not_called()

    def test_open_official_flag_is_refused_before_any_reducer_import(self):
        real_import = builtins.__import__
        for renderer, spec in GATE.REPLAYS.items():
            for flag in (True, 1, 0, None, 'false', 'missing'):
                with self.subTest(renderer=renderer, flag=flag):
                    cell = toy_cell()
                    if flag == 'missing':
                        del cell['official_tests_opened']
                    else:
                        cell['official_tests_opened'] = flag
                    self.write_cells(spec.ledger, [cell])
                    self.relock()
                    requested = []

                    def spy(name, globals=None, locals=None, fromlist=(), level=0):
                        requested.append(name)
                        return real_import(name, globals, locals, fromlist, level)

                    # The real reducers import their binder and renderer lazily.
                    with patch.object(GATE, 'REPLAYS', {**self.registry, renderer: spec}):
                        with patch('builtins.__import__', side_effect=spy):
                            with self.assertRaisesRegex(ValueError, f'{spec.label} ledger official test flag'):
                                GATE.authenticated_graph()
                    self.assertFalse([name for name in requested if name.startswith(REDUCER_PACKAGES)])
                    self.write_cells(spec.ledger, [toy_cell()])
        self.relock()

    def test_open_flag_in_evaluator_receipts_is_refused_before_reduction(self):
        spec = GATE.REPLAYS['docs/whitepaper/scripts/review_tabsyn.py']
        receipts = next(name for name in spec.extras if name.endswith('.jsonl'))
        self.write_cells(receipts, [toy_cell(official_tests_opened=True)])
        self.relock()
        self.refuse('TabSyn evaluator-replay ledger official test flag')
        self.reducers['docs/whitepaper/scripts/review_tabsyn.py'].assert_not_called()

    def test_recomputed_panel_mismatch_is_refused_for_every_renderer(self):
        for renderer, reduce in self.reducers.items():
            with self.subTest(renderer=renderer):
                reduce.side_effect = None
                reduce.return_value = b'{"different": "public toy"}\n'
                self.refuse(f'{self.registry[renderer].label} scalar ledger does not reproduce')
                panel = self.registry[renderer].panel
                reduce.side_effect = lambda cells, extras, panel=panel: (self.root / panel).read_bytes()

    def test_registered_renderer_cannot_read_another_pinned_panel(self):
        for index, entry in enumerate(self.graph['scripts']):
            with self.subTest(renderer=entry['path']):
                other = self.graph['scripts'][index - 1]['args'][1]
                graph = json.loads(json.dumps(self.graph))
                graph['scripts'][index]['args'][1] = other
                self.write(GRAPH_NAME, json.dumps(graph))
                self.relock()
                label = self.registry[entry['path']].label
                self.refuse(f'{label} renderer requires the pinned scalar ledger and panel')
        self.write(GRAPH_NAME, json.dumps(self.graph))
        self.relock()

    def test_unpinned_ledger_or_extra_is_refused(self):
        for spec in self.registry.values():
            for name in (spec.ledger, *spec.extras):
                with self.subTest(name=name):
                    self.relock(self.pinned_names() - {name})
                    self.refuse(f'{spec.label} renderer requires the pinned scalar ledger and panel')

    def test_input_preflight_names_every_replayed_panel(self):
        with patch.object(GATE.subprocess, 'run') as render:
            with patch('sys.argv', ['paper-review-outputs.py', '--check-inputs']):
                with patch('builtins.print') as printed:
                    GATE.main()
            render.assert_not_called()
        self.assertIn('control, privacy, TabSyn', printed.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
