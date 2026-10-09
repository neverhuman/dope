"""Pre-render control admission with public scalar fixtures; no sealed inputs."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'paper_review_control_gate', ROOT / 'ops/ci/paper-review-outputs.py')
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def scalar_cell():
    return {
        'dataset': 'public-fixture', 'kind': 'predictor_only', 'fit_seed': 11,
        'split_seed': None, 'sample_seed': 101, 'size': 1, 'status': 'ok',
        'reason': None, 'official_tests_opened': False, 'synthetic_sha256': 'ab' * 32,
        'metrics': {
            'c2st_auc': 0.5, 'marginal_ks_mean': 0.1, 'null_loss': 0.2,
            'pair_correlation_fidelity': 0.3,
            'rows': {'synthetic': 4, 'train': 2, 'validation': 1},
            'utility': {
                name: {'informative': True, 'low_signal_noninferior': None,
                       'retention': 0.5, 'trtr_loss': 0.2, 'tstr_loss': 0.2}
                for name in ('catboost', 'linear', 'mlp')
            },
        },
    }


class ControlEmitTests(unittest.TestCase):
    def setUp(self):
        target = ROOT / 'target/paper-control-input-tests'
        target.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.graph_name = 'research/benchmark/review_fixes/paper-build.json'
        self.graph = self.root / self.graph_name
        self.ledger = self.root / GATE.CONTROL_LEDGER
        self.panel = self.root / GATE.CONTROL_PANEL
        self.lock = self.root / 'lock.json'
        self.output_name = 'docs/whitepaper/generated/review-controls.tex'
        self.output = self.root / self.output_name
        for path in (self.graph, self.ledger, self.panel, self.output):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.write_text(json.dumps({'format': 'dope-review-paper-build-v1', 'scripts': [{
            'path': GATE.CONTROL_RENDERER,
            'args': ['--from-panel', GATE.CONTROL_PANEL], 'check_args': ['--check'],
            'outputs': [self.output_name],
        }]}))
        self.panel_payload = {'paired': [], 'predictor_fit_seeds': [], 'split_seeds': []}
        self.panel.write_text(json.dumps(self.panel_payload, indent=2, sort_keys=True) + '\n')
        self.output.write_bytes(b'existing public table\n')
        self.write_cells([scalar_cell()])
        for name, value in [('REPO', self.root), ('GRAPH', self.graph), ('LOCK', self.lock)]:
            p = patch.object(GATE, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.relock()

    def write_cells(self, cells):
        self.ledger.write_text(''.join(json.dumps(cell, sort_keys=True) + '\n' for cell in cells))

    def relock(self, include_ledger=True, extra_name=None):
        names = [self.graph_name, GATE.CONTROL_PANEL]
        if include_ledger:
            names.append(GATE.CONTROL_LEDGER)
        if extra_name:
            names.append(extra_name)
        files = {}
        for name in names:
            raw = (self.root / name).read_bytes()
            files[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        raw = (json.dumps({'format': 'dope-paper-review-inputs-v1', 'files': files}) + '\n').encode()
        self.lock.write_bytes(raw)
        p = patch.object(GATE, 'LOCK_SHA256', hashlib.sha256(raw).hexdigest())
        p.start()
        self.addCleanup(p.stop)

    def refuse_emit(self, reason):
        before = self.output.read_bytes()
        with patch('docs.whitepaper.scripts.review_controls.payload_from_cells') as reduce:
            with patch.object(GATE.subprocess, 'run') as render:
                with patch('sys.argv', ['paper-review-outputs.py']):
                    with self.assertRaisesRegex(ValueError, reason):
                        GATE.main()
                render.assert_not_called()
            reduce.assert_not_called()
        self.assertEqual(self.output.read_bytes(), before)

    def test_ledger_hash_mismatch_refuses_emit_before_decode(self):
        self.ledger.write_bytes(self.ledger.read_bytes() + b' ')
        loads = json.loads
        with patch.object(GATE.json, 'loads', wraps=loads) as decode:
            self.refuse_emit('digest or length mismatch')
            self.assertEqual(decode.call_count, 1)  # The authenticated lock only.

    def test_official_test_flags_refuse_emit_even_with_matching_hashes(self):
        for value in (True, 1, 0, None, 'false'):
            with self.subTest(flag=value):
                cell = scalar_cell()
                cell['official_tests_opened'] = value
                self.write_cells([cell])
                self.relock()
                self.refuse_emit('official test flag')

    def test_missing_official_test_flag_refuses_emit(self):
        cell = scalar_cell()
        del cell['official_tests_opened']
        self.write_cells([cell])
        self.relock()
        self.refuse_emit('official test flag')

    def test_unpinned_ledger_refuses_emit(self):
        self.relock(include_ledger=False)
        self.refuse_emit('pinned scalar ledger')

    def test_redirected_ledger_refuses_emit(self):
        saved = self.ledger.with_suffix('.saved')
        self.ledger.rename(saved)
        self.ledger.symlink_to(saved)
        self.refuse_emit('redirected')

    def test_duplicate_identity_refuses_emit(self):
        self.write_cells([scalar_cell(), scalar_cell()])
        self.relock()
        self.refuse_emit('duplicate identity')

    def test_non_scalar_keys_refuse_emit(self):
        cell = scalar_cell()
        cell['train_rows'] = ['public toy value']
        self.write_cells([cell])
        self.relock()
        self.refuse_emit('public scalar set')

    def test_empty_ledger_refuses_emit(self):
        self.ledger.write_bytes(b'\n')
        self.relock()
        with self.assertRaises(json.JSONDecodeError):
            GATE.authenticated_graph()

    def test_unregistered_jsonl_path_refuses_emit(self):
        name = 'research/benchmark/results/unregistered.jsonl'
        (self.root / name).write_text('{}\n')
        self.relock(extra_name=name)
        self.refuse_emit('invalid review paper input path')

    def test_recomputed_panel_mismatch_refuses_emit(self):
        before = self.output.read_bytes()
        with patch('docs.whitepaper.scripts.review_controls.payload_from_cells',
                   return_value={'different': 'public fixture'}) as reduce:
            with patch.object(GATE.subprocess, 'run') as render:
                with patch('sys.argv', ['paper-review-outputs.py']):
                    with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                        GATE.main()
                render.assert_not_called()
            reduce.assert_called_once()
        self.assertEqual(self.output.read_bytes(), before)

    def test_matching_scalar_panel_admits_emit(self):
        with patch('docs.whitepaper.scripts.review_controls.payload_from_cells',
                   return_value=self.panel_payload) as reduce:
            with patch.object(GATE.subprocess, 'run') as render:
                with patch('sys.argv', ['paper-review-outputs.py']):
                    GATE.main()
                render.assert_called_once()
            reduce.assert_called_once_with([scalar_cell()])

    def test_input_only_preflight_replays_without_emitting(self):
        before = self.output.read_bytes()
        with patch('docs.whitepaper.scripts.review_controls.payload_from_cells',
                   return_value=self.panel_payload) as reduce:
            with patch.object(GATE.subprocess, 'run') as render:
                with patch('sys.argv', ['paper-review-outputs.py', '--check-inputs']):
                    GATE.main()
                render.assert_not_called()
            reduce.assert_called_once_with([scalar_cell()])
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
