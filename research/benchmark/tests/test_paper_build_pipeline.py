"""Bounded fixtures prove admission, expected bytes and PDF control flow."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'docs/whitepaper/scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('paper_build_pipeline', SCRIPTS / 'paper_build.py')
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)
REVIEW = BUILD.REVIEW


class PipelineTests(unittest.TestCase):
    def setUp(self):
        target = ROOT / 'target/paper-pipeline-tests'
        target.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.graph_name = 'research/benchmark/review_fixes/paper-build.json'
        self.panel_name = 'research/benchmark/results/review-fixes-fixture/panel.json'
        self.output = 'docs/whitepaper/generated/review-fixture.tex'
        self.renderer = 'docs/whitepaper/scripts/review_fixture.py'
        self.graph = {'format': 'dope-review-paper-build-v1', 'scripts': [{
            'path': self.renderer, 'args': ['--from-panel', self.panel_name],
            'check_args': ['--check'], 'outputs': [self.output]}]}
        self.write(self.panel_name, b'{"official_tests_opened": false, "median": 0.5}\n')
        self.write(self.renderer, b'''import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
panel=json.loads((root/sys.argv[2]).read_bytes())
assert panel['official_tests_opened'] is False
out=root/'docs/whitepaper/generated/review-fixture.tex'
out.write_text('median='+str(panel['median'])+'\\n')
''')
        self.write('docs/whitepaper/scripts/build_pdf.sh', b'# fixture wrapper\n')
        self.write('ops/ci/paper-architecture.tex', b'% fixture architecture\n')
        self.write('research/benchmark/review_fixes/reducer_fixture.py', b'# transitive reducer\n')
        self.write('docs/whitepaper/generated/numbers.tex', b'legacy=42\n')
        self.write('docs/whitepaper/figures/architecture.tex', b'% tracked figure\n')
        author = ('\\author{Jepson Taylor$^1$, Alton Alexander$^1$\\\\[0.15cm]\n'
                  '$^1$NEVERHUMAN Research}')
        for stem in ('dope-mfs', 'supplement'):
            text = author + '\n'
            if stem == 'dope-mfs':
                text += '% BEGIN JOURNAL AVAILABILITY\nhttps://github.com/neverhuman/dope\n% END JOURNAL AVAILABILITY\n'
            self.write(f'docs/whitepaper/{stem}.tex', text.encode())
        for obj, key, value in [(REVIEW, 'REPO', self.root), (REVIEW, 'GRAPH', self.root / self.graph_name),
                                (REVIEW, 'LOCK', self.root / 'ops/ci/paper-review-inputs.json'),
                                (BUILD, 'REPO', self.root), (BUILD, 'PAPER', self.root / 'docs/whitepaper')]:
            p = patch.object(obj, key, value)
            p.start(); self.addCleanup(p.stop)
        self.relock()

    def write(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def relock(self):
        self.write(self.graph_name, json.dumps(self.graph).encode())
        files = {}
        for name in (self.graph_name, self.panel_name):
            raw = (self.root / name).read_bytes()
            files[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        raw = json.dumps({'format': 'dope-paper-review-inputs-v1', 'files': files}).encode()
        REVIEW.LOCK.write_bytes(raw)
        p = patch.object(REVIEW, 'LOCK_SHA256', hashlib.sha256(raw).hexdigest())
        p.start(); self.addCleanup(p.stop)

    def admit(self):
        return REVIEW.authenticate_build()

    def add_auxiliary_renderers(self):
        headline = 'docs/whitepaper/generated/v2-headline.json'
        self.graph['scripts'][0]['outputs'].append(headline)
        path = self.root / self.renderer
        path.write_text(path.read_text() + f"(root / {headline!r}).write_text(json.dumps(panel))\n")
        self.write('README.md', b'fixture README\n')
        self.write('docs/whitepaper/scripts/v2_figures.py', f'''import json
from pathlib import Path
root = Path(__file__).resolve().parents[3]
panel = json.loads((root / {headline!r}).read_text())
assert panel['official_tests_opened'] is False
for name in {REVIEW.AUXILIARY_OUTPUTS!r}:
    if name != 'README.md':
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('median=' + str(panel['median']) + '\\n')
'''.encode())
        self.write('docs/whitepaper/scripts/readme_benchmark.py', b'# fixture README already matches\n')
        self.relock()

    def publish(self, verified):
        REVIEW.render_review_outputs(verified, check=False)

    def test_expected_bytes_and_immutable_context_reject_stale_input_and_code(self):
        verified = self.admit()
        self.assertEqual(dict(verified.expected), {self.output: b'median=0.5\n'})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            verified.graph = b'{}'
        self.publish(verified)
        REVIEW.render_review_outputs(verified, check=True)
        for name in (self.panel_name, self.renderer, 'docs/whitepaper/generated/numbers.tex',
                     'research/benchmark/review_fixes/reducer_fixture.py',
                     'docs/whitepaper/dope-mfs.tex', self.graph_name, 'ops/ci/paper-review-inputs.json'):
            path = self.root / name
            raw = path.read_bytes()
            path.write_bytes(raw + b' ')
            with self.assertRaisesRegex(ValueError, 'stale'):
                REVIEW.render_review_outputs(verified, check=True)
            path.write_bytes(raw)

    def test_added_dependencies_and_undeclared_outputs_fail_closed(self):
        verified = self.admit(); self.publish(verified)
        self.write('docs/whitepaper/scripts/new_helper.py', b'# new import\n')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            REVIEW.render_review_outputs(verified, check=True)
        (self.root / 'docs/whitepaper/scripts/new_helper.py').unlink()
        self.write('docs/whitepaper/generated/review-extra.tex', b'extra\n')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            REVIEW.render_review_outputs(verified, check=True)
        fresh = self.admit()
        with self.assertRaisesRegex(ValueError, 'cover every'):
            REVIEW.render_review_outputs(fresh, check=True)

    def test_generated_macro_tamper_rejected_instead_of_repaired_by_check(self):
        verified = self.admit(); self.publish(verified)
        self.write(self.output, b'median=9000\n')
        with self.assertRaisesRegex(ValueError, 'expected render'):
            REVIEW.render_review_outputs(verified, check=True)
        self.assertEqual((self.root / self.output).read_bytes(), b'median=9000\n')
        # A fresh context computes expected bytes from pinned input, never from
        # the tampered generated file's own fingerprint.
        fresh = self.admit()
        with self.assertRaisesRegex(ValueError, 'expected render'):
            REVIEW.render_review_outputs(fresh, check=True)

    def test_missing_graph_invalid_declarations_and_tamper_fail_before_publication(self):
        REVIEW.GRAPH.unlink()
        with self.assertRaisesRegex(ValueError, 'declared graph'):
            self.admit()
        self.relock()
        for field, value in [('path', '../escape.py'), ('args', ['--from-panel', '../escape.json']),
                             ('check_args', ['--write-panel']), ('outputs', []),
                             ('outputs', ['docs/whitepaper/generated/../../escape'])]:
            original = self.graph['scripts'][0][field]
            self.graph['scripts'][0][field] = value; self.relock()
            with self.assertRaises(ValueError):
                self.admit()
            self.assertFalse((self.root / self.output).exists())
            self.graph['scripts'][0][field] = original
        self.graph['format'] = 'wrong'; self.relock()
        with self.assertRaisesRegex(ValueError, 'invalid review paper graph'):
            self.admit()

    def test_actual_render_inventory_must_equal_declared_inventory(self):
        self.graph['scripts'][0]['outputs'].append('docs/whitepaper/generated/review-absent.tex')
        self.relock()
        with self.assertRaisesRegex(ValueError, 'actual rendered inventory'):
            self.admit()
        self.assertFalse((self.root / self.output).exists())

    def test_auxiliary_expected_bytes_share_the_same_immutable_context(self):
        self.add_auxiliary_renderers()
        verified = self.admit(); self.publish(verified)
        receipt = BUILD.BuildReceipt('auxiliary-fixture')
        self.assertIs(BUILD.auxiliary_outputs(verified, check=False, receipt=receipt), verified)
        before = {name: (self.root / name).stat().st_mtime_ns for name in REVIEW.AUXILIARY_OUTPUTS}
        self.assertIs(BUILD.auxiliary_outputs(verified, check=True, receipt=receipt), verified)
        self.assertEqual(before, {name: (self.root / name).stat().st_mtime_ns for name in before})
        self.write('docs/readme/pareto.png', b'tampered image')
        with self.assertRaisesRegex(ValueError, 'expected rendering'):
            BUILD.auxiliary_outputs(verified, check=True, receipt=receipt)
        with patch.object(BUILD, 'run', side_effect=AssertionError('engine must not start')):
            with self.assertRaisesRegex(ValueError, 'expected rendering'):
                BUILD.compile_pdfs(verified, receipt)

    def test_unexpected_nonreview_render_output_is_not_silently_published(self):
        script = self.root / self.renderer
        script.write_text(script.read_text() + "(root/'docs/whitepaper/generated/rogue.tex').write_text('extra')\n")
        with self.assertRaisesRegex(ValueError, 'actual rendered inventory'):
            self.admit()
        self.assertFalse((self.root / self.output).exists())

    def test_closed_flags_lock_panel_and_graph_are_authenticated(self):
        for name in ('ops/ci/paper-review-inputs.json', self.panel_name, self.graph_name):
            original = (self.root / name).read_bytes()
            self.write(name, original + b' ')
            with self.assertRaisesRegex(ValueError, 'digest'):
                self.admit()
            self.assertFalse((self.root / self.output).exists())
            self.write(name, original)
        # v2 admission refuses opened official tests before invoking a reducer.
        opened = json.dumps({'official_tests_opened': True}).encode()
        with self.assertRaisesRegex(ValueError, 'not closed'):
            REVIEW.authenticated_v2({name: opened for name in
                                     (REVIEW.V2_LEDGER, REVIEW.V2_FITS, REVIEW.V2_RECORD, REVIEW.V2_PREDECLARE)})

    def test_forged_receipt_and_skip_auth_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'invalid paper build context'):
            REVIEW.render_review_outputs({'authenticated': True}, check=False)
        with patch('sys.argv', ['paper_build.py', '--skip-auth']):
            with self.assertRaises(SystemExit) as error:
                BUILD.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / self.output).exists())

    def test_compiler_preserves_four_pdfs_flags_anonymity_and_revalidates(self):
        verified = self.admit(); self.publish(verified)
        receipt = BUILD.BuildReceipt('fixture')
        commands = []
        def compiler(command, receipt, **kwargs):
            commands.append(command)
            root = self.root / 'target/paper-build'
            if 'pdflatex' in command:
                self.write('target/paper-build/paper-architecture.pdf', b'architecture')
            elif 'latexmk' in command:
                stem = Path(command[-1]).stem
                self.write(f'target/paper-build/{stem}.pdf', stem.encode())
                self.write(f'target/paper-build/{stem}.log', b'clean log')
                self.assertEqual(kwargs['env']['SOURCE_DATE_EPOCH'], '1760000000')
                self.assertIn('-g', command)
                self.assertIn('-usepretex=\\pdftrailerid{}\\pdfsuppressptexinfo=-1', command)
                if stem.endswith('-anonymous'):
                    text = Path(command[-1]).read_text()
                    self.assertIn('Anonymous', text)
                    self.assertNotIn('Jepson Taylor', text)
            elif command[0] != 'pdfinfo':
                self.fail('unexpected compiler command')
        with patch.object(BUILD, 'run', side_effect=compiler), patch.dict('os.environ', {}, clear=True):
            BUILD.compile_pdfs(verified, receipt)
        self.assertEqual([Path(c[-1]).stem for c in commands if 'latexmk' in c], list(BUILD.PDF_STEMS))
        for stem in BUILD.PDF_STEMS:
            self.assertEqual((BUILD.PAPER / f'{stem}.pdf').read_bytes(), stem.encode())
        # Tamper between admission and compilation must precede any engine call.
        self.write(self.output, b'tampered')
        with patch.object(BUILD, 'run', side_effect=AssertionError('engine must not start')):
            with self.assertRaisesRegex(ValueError, 'expected render'):
                BUILD.compile_pdfs(verified, receipt)

    def test_zero_drift_accounts_for_compiler_outputs_only_after_compilation(self):
        # A synthetic repository isolates the real Git pathspec/drift contract.
        def git(*args):
            subprocess.run(['git', *args], cwd=self.root, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        git('init', '--quiet', '--initial-branch=fixture')
        for name in BUILD.COMPILER_OUTPUTS:
            self.write(name, b'committed compiler output')
        git('add', '-f', '.')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
        receipt = BUILD.BuildReceipt('drift-fixture')
        architecture = self.root / 'docs/whitepaper/figures/architecture.pdf'
        architecture.unlink()
        BUILD.output_drift(receipt, before_compile=True)
        with self.assertRaises(subprocess.CalledProcessError):
            BUILD.output_drift(receipt)
        architecture.write_bytes(b'committed compiler output')
        BUILD.output_drift(receipt)
        self.write('docs/whitepaper/generated/numbers.tex', b'tampered legacy macro')
        with self.assertRaises(subprocess.CalledProcessError):
            BUILD.output_drift(receipt, before_compile=True)

    def test_top_level_modes_share_real_admission_and_check_never_writes_outputs(self):
        self.add_auxiliary_renderers()
        for check, pdf_only in ((False, False), (True, False), (False, True)):
            if check or pdf_only:
                self.publish(self.admit())
            output = self.root / self.output
            initial_stat = output.stat().st_mtime_ns if output.exists() else None
            admitted = []
            authenticate = REVIEW.authenticate_build
            def admit(**kwargs):
                verified = authenticate(**kwargs); admitted.append(verified)
                return verified
            def compile(verified, receipt):
                self.assertIs(verified, admitted[0]); verified.revalidate()
                REVIEW.render_review_outputs(verified, check=True)
                for stem in BUILD.PDF_STEMS:
                    self.write(f'docs/whitepaper/{stem}.pdf', b'fixture pdf')
            # Only PDF engines and the outer Git gate are substituted;
            # authentication, renderers and byte comparisons use real files.
            with patch.object(REVIEW, 'authenticate_build', side_effect=admit), \
                    patch.object(BUILD, 'compile_pdfs', side_effect=compile), \
                    patch.object(BUILD, 'output_drift'), patch.object(BUILD, 'run'):
                verified = BUILD.build_paper(check=check, pdf_only=pdf_only)
            self.assertEqual(len(admitted), 1)
            self.assertEqual((self.root / self.output).read_bytes(), b'median=0.5\n')
            if check or pdf_only:
                self.assertEqual(output.stat().st_mtime_ns, initial_stat)
            self.assertIs(verified, admitted[0])


if __name__ == '__main__':
    unittest.main()
