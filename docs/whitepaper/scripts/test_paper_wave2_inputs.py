"""Wave2 pre-render refusal on independently pinned, scalar-only toy inputs."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('wave2_paper_gate', ROOT / 'ops/ci/paper-review-outputs.py')
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class Wave2EmitTests(unittest.TestCase):
    def fixture(self, kind):
        target = ROOT / 'target/paper-wave2-input-tests'; target.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=target); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        if kind == 'privacy':
            ledger, panel, script = GATE.PRIVACY_LEDGER, GATE.PRIVACY_PANEL, 'docs/whitepaper/scripts/review_privacy.py'
            deps = []
        elif kind == 'tabsyn':
            ledger, panel, script = GATE.TABSYN_LEDGER, GATE.TABSYN_PANEL, 'docs/whitepaper/scripts/review_tabsyn.py'
            deps = [GATE.TABSYN_BINDING]
        else:
            from docs.whitepaper.scripts.review_fidelity import INPUTS
            ledger, panel, script = GATE.FIDELITY_LEDGER, GATE.FIDELITY_PANEL, GATE.FIDELITY_RENDERER
            deps = list(INPUTS)
        self.ledger_name, self.panel_name = ledger, panel
        self.graph_name = 'research/benchmark/review_fixes/paper-build.json'
        self.names = list(dict.fromkeys([self.graph_name, ledger, panel, *deps]))
        for name in self.names:
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{}\n')
        self.graph = self.root / self.graph_name
        self.graph.write_text(json.dumps({'format': 'dope-review-paper-build-v1', 'scripts': [{
            'path': script, 'args': ['--from-panel', panel], 'check_args': ['--check'],
            'outputs': ['docs/whitepaper/generated/review-fixture.tex']}]}))
        self.ledger = self.root / ledger
        self.write_flags(False)
        (self.root / panel).write_text(json.dumps({'ok': True}, indent=2, sort_keys=True) + '\n')
        self.output = self.root / 'docs/whitepaper/generated/review-fixture.tex'
        self.output.parent.mkdir(parents=True); self.output.write_bytes(b'existing public output\n')
        self.lock = self.root / 'lock.json'
        for name, value in [('REPO', self.root), ('GRAPH', self.graph), ('LOCK', self.lock)]:
            p = patch.object(GATE, name, value); p.start(); self.addCleanup(p.stop)
        self.relock()

    def write_flags(self, value, field='official_tests_opened'):
        cell = {'official_tests_opened': False, 'formal_dp': False}; cell[field] = value
        self.ledger.write_text(json.dumps(cell) + '\n')

    def relock(self, drop=None):
        files = {}
        for name in self.names:
            if name == drop:
                continue
            raw = (self.root / name).read_bytes()
            files[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        raw = json.dumps({'format': 'dope-paper-review-inputs-v1', 'files': files}).encode()
        self.lock.write_bytes(raw)
        p = patch.object(GATE, 'LOCK_SHA256', hashlib.sha256(raw).hexdigest()); p.start(); self.addCleanup(p.stop)

    def refuse(self, reason):
        before = self.output.read_bytes()
        with patch('research.benchmark.review_fixes.bind_review_cells.replay') as replay, \
                patch('docs.whitepaper.scripts.review_fidelity.payload_from_inputs') as fidelity, \
                patch.object(GATE.subprocess, 'run') as emit, patch('sys.argv', ['paper-review-outputs.py']):
            with self.assertRaisesRegex(ValueError, reason):
                GATE.main()
            replay.assert_not_called(); fidelity.assert_not_called(); emit.assert_not_called()
        self.assertEqual(self.output.read_bytes(), before)

    def test_each_ledger_hash_mismatch_refuses_before_decode_or_emit(self):
        for kind in ('privacy', 'tabsyn', 'fidelity'):
            with self.subTest(kind=kind):
                self.fixture(kind); self.ledger.write_bytes(b'changed bytes')
                loads = json.loads
                with patch.object(GATE.json, 'loads', wraps=loads) as decode:
                    self.refuse('digest or length mismatch')
                    self.assertEqual(decode.call_count, 1)

    def test_each_open_nonliteral_or_missing_closed_flag_refuses_emit(self):
        for kind in ('privacy', 'tabsyn', 'fidelity'):
            self.fixture(kind)
            for field in ('official_tests_opened', 'formal_dp'):
                for value in (True, 1, 0, None, 'false'):
                    with self.subTest(kind=kind, field=field, value=value):
                        self.write_flags(value, field); self.relock(); self.refuse('flag refused')
                self.ledger.write_bytes(b'{}\n'); self.relock(); self.refuse('flag refused')

    def test_each_missing_ledger_pin_refuses_emit(self):
        for kind in ('privacy', 'tabsyn', 'fidelity'):
            with self.subTest(kind=kind):
                self.fixture(kind); self.relock(drop=self.ledger_name); self.refuse('pinned scalar ledger')

    def test_each_redirected_ledger_refuses_even_when_bytes_match(self):
        for kind in ('privacy', 'tabsyn', 'fidelity'):
            with self.subTest(kind=kind):
                self.fixture(kind); saved = self.ledger.with_suffix('.saved')
                self.ledger.rename(saved); self.ledger.symlink_to(saved); self.refuse('redirected')

    def test_bound_panel_mismatch_refuses_all_renderers(self):
        for kind in ('privacy', 'tabsyn'):
            with self.subTest(kind=kind):
                self.fixture(kind)
                with patch('research.benchmark.review_fixes.bind_review_cells.replay', return_value={'ok': False}), \
                        patch.object(GATE.subprocess, 'run') as emit, patch('sys.argv', ['paper-review-outputs.py']):
                    with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                        GATE.main()
                    emit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
