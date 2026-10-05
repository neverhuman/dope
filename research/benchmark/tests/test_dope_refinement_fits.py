"""Reject incomplete training, rewritten custody and misleading failure labels."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research.benchmark import publish_dope_refinement_fits as pub


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix():
    return [dict(dataset=f'{d:016x}', method='DOPE', profile=p, fit_seed=11,
        status='ok', outcome_class='ok', kind='new_gpu_refinement_fit', immutable_prior_fit_reused=False,
        charged_artifact_bytes=100, model_bytes=90, projection_bytes=10,
        artifact_gate_failed=False, counts_as_method_failure=False, new_operation_seconds=1.0,
        **pub.GATES) for d in range(100) for p in pub.PROFILES]


class RefinementFits(unittest.TestCase):
    def test_complete_profiles_not_success_only_subset(self):
        cells = matrix(); self.assertEqual([r['closed'] for r in pub.summarize(cells)], [100, 100])
        for bad in (cells[:-1], cells[:-1] + [cells[0]]):
            with self.assertRaises(ValueError): pub.summarize(bad)

    def test_gates_and_integer_seed_are_strict(self):
        for key, value in [('fit_seed', 11.0), ('mfs_v2', .99), ('ptf_v1', .99),
                           ('production_certified', True), ('global_family_selected', True),
                           ('counts_as_dope_win', True), ('official_tests_opened', True)]:
            cells = matrix(); cells[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): pub.summarize(cells)

    def test_infrastructure_is_visible_without_method_failure(self):
        cells = matrix()
        for i, status in enumerate(('transport_or_prelaunch_failure', 'foreign_gpu_owner_appeared')):
            cells[i].update(status=status, outcome_class='infrastructure', charged_artifact_bytes=None,
                            model_bytes=None, projection_bytes=None)
        self.assertEqual(sum(r['status_counts'].get('foreign_gpu_owner_appeared', 0) for r in pub.summarize(cells)), 1)
        for key, value in [('outcome_class', 'method_failure'), ('counts_as_method_failure', True),
                           ('artifact_gate_failed', True)]:
            bad = copy.deepcopy(cells); bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): pub.summarize(bad)

    def test_projection_charge_cannot_turn_overrun_into_success(self):
        cells = matrix(); cells[0].update(charged_artifact_bytes=10241, model_bytes=10239, projection_bytes=2,
            status='charged_artifact_cap', outcome_class='artifact_gate_failure', artifact_gate_failed=True)
        self.assertEqual(pub.summarize(cells)[0]['status_counts']['charged_artifact_cap'], 1)
        for changes in ({'status': 'ok'}, {'charged_artifact_bytes': 10239}, {'artifact_gate_failed': False}):
            bad = copy.deepcopy(cells); bad[0].update(changes)
            with self.assertRaises(ValueError): pub.summarize(bad)

    def test_prior_reuse_is_not_a_new_fit_cost(self):
        cells = matrix(); cells[0].update(immutable_prior_fit_reused=True, kind='immutable_prior_reuse', new_operation_seconds=0)
        pub.summarize(cells)
        for changes in ({'new_operation_seconds': 1}, {'immutable_prior_fit_reused': 1}, {'kind': 'new_gpu_refinement_fit'}):
            bad = copy.deepcopy(cells); bad[0].update(changes)
            with self.assertRaises(ValueError): pub.summarize(bad)

    def test_rewritten_receipt_rejected_before_outcome_decode(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'round'; root.mkdir()
            receipt = root / 'receipt.json'; receipt.write_text('{}')
            report = root / 'reconciliation-v1.json'; report.write_text('not JSON')
            round_path = root / 'round.lock.json'; round_path.write_text('{}')
            rpin, pin = sha(round_path), sha(report)
            anchor = root / 'receipt-lock-v1.json'
            anchor.write_text(json.dumps(dict(complete_fit_matrix=True, round_sha256=rpin,
                reconciliation_sha256=pin, refs={str(receipt): sha(receipt), str(round_path): rpin})))
            apin = sha(anchor); receipt.write_text('{"status":"ok"}')
            with patch.object(pub, 'BASE', base), patch.object(pub, 'ROOT', root), \
                 patch.object(pub, 'ROUND', rpin), patch.object(pub, 'REPORT', pin), patch.object(pub, 'RECEIPTS', apin):
                with self.assertRaisesRegex(ValueError, 'immutable evidence changed'): pub.anchored()
                value = json.loads(anchor.read_bytes()); value['refs'][str(receipt)] = sha(receipt)
                anchor.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'immutable evidence changed'): pub.anchored()

    def test_flat_inventory_rejects_added_directory_and_symlink(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); code = root / 'entry.py'; code.write_text('pass\n'); files = {code.name: sha(code)}
            with patch.object(pub, 'BASE', root):
                pub.flat(root, files)
                alias = root / 'alias'; alias.symlink_to(code)
                with self.assertRaises(ValueError): pub.flat(root, files)
                alias.unlink(); (root / 'extra').mkdir()
                with self.assertRaises(ValueError): pub.flat(root, files)

    def test_digest_subclass_and_nonfinite_json_are_rejected(self):
        class Digest(str): pass
        with TemporaryDirectory() as tmp:
            root = Path(tmp); file = root / 'json'; file.write_text('{}')
            with patch.object(pub, 'BASE', root):
                with self.assertRaises(ValueError): pub.bound(file, Digest(sha(file)))
                file.write_text('{"cost":NaN}')
                with self.assertRaises(ValueError): pub.bound(file, sha(file))

    def fixture_cell(self, base):
        worker = base / 'worker'; worker.mkdir()
        for name in ('train.csv', 'validation.csv', 'projection.json', 'worker-manifest.json', 'row-group-assignments.json'):
            (worker / name).write_bytes(b'{}')
        job = dict(dataset='fixture', method='DOPE', seed=11, final=False, research_profile=pub.PROFILES[0],
            worker=dict(path=str(worker), files={p.name: sha(p) for p in worker.iterdir()}))
        key = pub.digest(job); out = base / 'attempts' / key / 'attempt-0001'; out.mkdir(parents=True)
        model = out / 'model.dpk'; model.write_bytes(b'x' * 90)
        fit = dict(job=job, job_sha256=key, status='ok', exit_code=0, gpu_target_operator_verified=True,
            gpu_process_observed=True, foreign_processes_signaled=False, official_tests_opened=False)
        fp = out / 'fit.json'; fp.write_text(json.dumps(fit))
        evidence = {p.name: sha(p) for p in out.iterdir()}
        receipt = dict(job=job, round_sha256=pub.ROUND, status='ok', kind='new_gpu_refinement_fit', attempt=1,
            official_tests_opened=False, counts_as_dope_win=False, mfs_v2=None, ptf_v1=None, evidence_files=evidence)
        rp = out / 'receipt.json'; rp.write_text(json.dumps(receipt))
        row = dict(original_fit_job=job, job_sha256=key, dataset='fixture', profile=pub.PROFILES[0], fit_seed=11,
            receipt_path=str(rp), receipt_sha256=sha(rp), status='ok', immutable_prior_fit_reused=False,
            model_path=str(model), model_sha256=sha(model), charged_artifact_bytes=92, projection_bytes_included=True,
            gpu_target_operator_verified=True, new_operation_seconds=1.0, host_operation_seconds={'xbabe2': 1.0})
        # fit.json's identity is transitively anchored by receipt.evidence_files.
        refs = {str(rp): sha(rp), str(model): sha(model), **{str(p): sha(p) for p in worker.iterdir()}}
        return row, refs, fp

    def test_transitive_fit_hash_precedes_fit_decoding(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp); row, refs, fp = self.fixture_cell(base)
            with patch.object(pub, 'BASE', base):
                self.assertTrue(pub.cell(row, dict(refs))['fit_receipt_written'])
                fp.write_text('{"gpu_target_operator_verified":true}')
                with self.assertRaises(ValueError): pub.cell(row, dict(refs))

    def test_canonical_final_and_seed_cannot_use_python_numeric_equality(self):
        for field, value in [('final', 0), ('final', 'false'), ('seed', 11.0)]:
            with TemporaryDirectory() as tmp:
                base = Path(tmp); row, refs, _ = self.fixture_cell(base)
                row['original_fit_job'][field] = value
                row['job_sha256'] = pub.digest(row['original_fit_job'])
                with patch.object(pub, 'BASE', base), self.assertRaises(ValueError): pub.cell(row, refs)

    def test_cell_projection_identity_and_source_header_never_leak(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp); row, refs, _ = self.fixture_cell(base); row['projection_bytes_included'] = False
            with patch.object(pub, 'BASE', base), self.assertRaisesRegex(ValueError, '^projection charge omitted$'):
                pub.cell(row, refs)

    def test_table_roundtrip_keeps_failures_and_null_scope(self):
        rows = matrix(); report = dict(cells=rows, summary=pub.summarize(rows), new_ok_fits=176, **pub.GATES,
            cost=dict(new_operation_seconds=200, prior_whole_discovery_confirmation_operation_seconds=10,
                cumulative_bounded_refinement_operation_seconds=210, scheduler_wall_seconds=300))
        csv_text, md = pub.tables(report)
        self.assertEqual(len(csv_text.splitlines()), 201); self.assertNotIn('\r', csv_text)
        self.assertIn('MFS-v2/PTF-v1/release/superiority null', md)
        self.assertIn('whole prior discovery/confirmation', md)


if __name__ == '__main__': unittest.main()
