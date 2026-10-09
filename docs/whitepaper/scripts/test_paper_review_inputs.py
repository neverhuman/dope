"""Public paper input authentication, using fixtures under the lane target."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('paper_review_inputs', ROOT / 'ops/ci/paper-review-outputs.py')
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)
PREPARE_SPEC = importlib.util.spec_from_file_location('paper_prepare_inputs', ROOT / 'ops/ci/paper-prepare.py')
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(PREPARE)


class CommittedInputTests(unittest.TestCase):
    def test_committed_graph_and_all_public_inputs_match_the_lock(self):
        self.assertEqual(REVIEW.authenticated_graph()['format'], 'dope-review-paper-build-v1')

    def test_tost_constants_and_supplement_use_the_original_declaration(self):
        text = PREPARE.tost_protocol_tex()
        declaration = json.loads(PREPARE.authenticated(PREPARE.PREDECLARE, PREPARE.PREDECLARE_SHA256))
        self.assertIn(r'\newcommand{\TostEquivalenceMargin}{' + str(declaration['tost']['margin_retention']) + '}', text)
        self.assertIn(r'\newcommand{\TostAlpha}{' + str(declaration['tost']['alpha']) + '}', text)
        self.assertEqual((ROOT / 'docs/whitepaper/generated/tost-protocol.tex').read_text(), text)
        supplement = (ROOT / 'docs/whitepaper/supplement.tex').read_text()
        self.assertIn(r'\input{generated/tost-protocol.tex}', supplement)
        self.assertIn(r'\pm\TostEquivalenceMargin', supplement)
        self.assertIn(r'\alpha=\TostAlpha', supplement)

    def test_tost_declaration_drift_and_redirection_fail_before_decode(self):
        target = ROOT / 'target/paper-review-input-tests'
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            fixture = Path(directory) / 'predeclare.json'
            fixture.write_bytes(PREPARE.PREDECLARE.read_bytes() + b' ')
            with patch.object(PREPARE, 'PREDECLARE', fixture), patch.object(PREPARE.json, 'loads') as decode:
                with self.assertRaises(ValueError):
                    PREPARE.tost_protocol_tex()
                decode.assert_not_called()
            link = Path(directory) / 'redirect.json'
            link.symlink_to(PREPARE.PREDECLARE)
            with patch.object(PREPARE, 'PREDECLARE', link), patch.object(PREPARE.json, 'loads') as decode:
                with self.assertRaises(ValueError):
                    PREPARE.tost_protocol_tex()
                decode.assert_not_called()


class ReviewInputTests(unittest.TestCase):
    def setUp(self):
        target = ROOT / 'target/paper-review-input-tests'
        target.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.graph_name = 'research/benchmark/review_fixes/paper-build.json'
        self.panel_name = 'research/benchmark/results/review-fixes-fixture/panel.json'
        self.graph = {'format': 'dope-review-paper-build-v1', 'scripts': [{
            'path': 'docs/whitepaper/scripts/review_fixture.py',
            'args': ['--from-panel', self.panel_name], 'check_args': ['--check'], 'outputs': [],
        }]}
        self.panel = self.root / self.panel_name
        self.panel.parent.mkdir(parents=True)
        self.panel.write_bytes(b'{"median": 0.5}\n')
        graph = self.root / self.graph_name
        graph.parent.mkdir(parents=True)
        graph.write_text(json.dumps(self.graph))
        self.lock = self.root / 'lock.json'
        for name, value in [('REPO', self.root), ('GRAPH', graph), ('LOCK', self.lock)]:
            p = patch.object(REVIEW, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.relock([self.graph_name, self.panel_name])

    def relock(self, names):
        files = {}
        for name in names:
            raw = (self.root / name).read_bytes()
            files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        raw = json.dumps({'format': 'dope-paper-review-inputs-v1', 'files': files}).encode()
        self.lock.write_bytes(raw)
        p = patch.object(REVIEW, 'LOCK_SHA256', hashlib.sha256(raw).hexdigest())
        p.start()
        self.addCleanup(p.stop)

    def test_valid_public_graph(self):
        self.assertEqual(REVIEW.authenticated_graph(), self.graph)

    def test_changed_lock_fails_before_json_decoding(self):
        self.lock.write_bytes(b'{broken')
        with patch.object(REVIEW.json, 'loads') as decode:
            with self.assertRaisesRegex(ValueError, 'digest'):
                REVIEW.authenticated_graph()
            decode.assert_not_called()

    def test_changed_panel_fails_before_graph_decode_or_renderer(self):
        self.panel.write_bytes(b'{broken')
        loads = json.loads
        with patch.object(REVIEW.json, 'loads', wraps=loads) as decode:
            with patch.object(REVIEW.subprocess, 'run') as renderer:
                with patch('sys.argv', ['paper-review-outputs.py', '--check']):
                    with self.assertRaisesRegex(ValueError, 'digest'):
                        REVIEW.main()
                renderer.assert_not_called()
            self.assertEqual(decode.call_count, 1)  # Only the authenticated lock.

    def test_unpinned_graph_is_rejected(self):
        self.relock([self.panel_name])
        with self.assertRaisesRegex(ValueError, 'graph has no input pin'):
            REVIEW.authenticated_graph()

    def test_redirected_panel_is_rejected_even_when_bytes_match(self):
        saved = self.panel.with_suffix('.saved')
        self.panel.rename(saved)
        self.panel.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, 'redirected'):
            REVIEW.authenticated_graph()

    def test_renderer_cannot_use_an_unpinned_panel(self):
        self.graph['scripts'][0]['args'][1] = 'research/benchmark/results/unpinned.json'
        REVIEW.GRAPH.write_text(json.dumps(self.graph))
        self.relock([self.graph_name, self.panel_name])
        with self.assertRaisesRegex(ValueError, 'independently pinned'):
            REVIEW.authenticated_graph()


if __name__ == '__main__':
    unittest.main()
