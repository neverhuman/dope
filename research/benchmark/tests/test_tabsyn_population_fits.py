"""Opaque receipt fixtures and deterministic fit-ledger publication checks."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jsonschema

from research.benchmark import publish_tabsyn_population_fits as p


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def cell_fixture():
    identity = {'dataset': '0' * 16, 'fit_seed': 11, 'source_commit': p.AUTHOR,
                'worker_sha256': 'c' * 64, 'joint_train_sha256': 'd' * 64,
                'joint_validation_sha256': 'e' * 64, 'projection_sha256': 'f' * 64}
    config = {'config_sha256': p.CONFIG_SHA, 'vae_epochs': 200, 'diffusion_epochs': 1000}
    sources = {f'/opaque/source/{i}.py': f'{i}' * 64 for i in range(6)}
    model = {'complete': True, 'seed': 11, 'source_commit': p.AUTHOR, 'config': config,
             'declared_epoch_budget': {'vae_epochs': 200, 'diffusion_epochs': 1000},
             'author_epoch_baseline': {'vae_epochs': 4000, 'diffusion_epochs': 10001},
             'generated_fixture': False, 'official_tests_opened': False,
             **{k: None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority')}}
    names = ('decoder-final.safetensors', 'diffusion-epoch-0.safetensors', 'diffusion.safetensors',
             'encoder-final.safetensors', 'model.json', 'numeric-state.npz', 'projection.json',
             'vae-selected.safetensors')
    files = {n: {'bytes': 1, 'sha256': 'a' * 64} for n in names}
    files['projection.json']['sha256'] = identity['projection_sha256']
    files['model.json']['sha256'] = hashlib.sha256(encoded(model)).hexdigest()
    inventory = {'artifact_bytes': 8, 'projection_bytes': 1, 'files': files}
    request = {'learning_identity': identity, 'lineage_sha256': 'b' * 64,
               'model_source_commit': p.AUTHOR, 'generated_fixture': False,
               'official_tests_opened': False, 'hardware_qualification': None,
               'production_certified': False, 'source_files': sources, 'config': config,
               'author_default_4000_10001_track_preserved': True, 'author_patience_500_preserved': True}
    complete = {'status': 'ok', 'lineage_sha256': 'b' * 64, 'config_sha256': p.CONFIG_SHA,
                'hardware_qualification': None, 'production_certified': False,
                'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
                'operation_seconds': 1., 'result': {'inventory': inventory, 'result': model}}
    model['code_sha256'] = model['operation_admission_sha256'] = hashlib.sha256(encoded(request)).hexdigest()
    files['model.json']['sha256'] = hashlib.sha256(encoded(model)).hexdigest()
    refs = {n: {'path': '/opaque/' + n, 'sha256': hashlib.sha256(encoded(v)).hexdigest()}
            for n, v in [('request.json', request), ('complete.json', complete), ('model.json', model)]}
    exit_row = {'actual_worker_exit': 0, 'foreign_signals': 0, 'scientific_worker_pid': 7,
                'success_receipt_ref': refs['complete.json'], 'whole_operation_seconds': 2.}
    cuda = {'request_sha256': refs['request.json']['sha256'], 'scientific_body_sha256': p.ADAPTER_SHA,
            'pid': 7, 'hardware_qualification': None, 'real_TRAIN': True}
    refs.update({n: {'path': '/opaque/' + n, 'sha256': hashlib.sha256(encoded(v)).hexdigest()}
                 for n, v in [('exit.json', exit_row), ('worker.CUDA-initialized.receipt.json', cuda)]})
    receipts = {refs[n]['path']: v for n, v in [('request.json', request), ('complete.json', complete),
                ('exit.json', exit_row), ('model.json', model), ('worker.CUDA-initialized.receipt.json', cuda)]}
    row = {'dataset': identity['dataset'], 'lineage_sha256': 'b' * 64, 'status': 'fit_ok',
           'scientific_receipt_status': 'ok', 'actual_worker_exit': 0, 'fit_seed': 11,
           'config_sha256': p.CONFIG_SHA, 'artifact_file_sizes_rechecked': True, 'retained_first': False,
           'fit_request_ref': refs['request.json'], 'fit_complete_ref': refs['complete.json'],
           'fit_exit_ref': refs['exit.json'], 'artifact_inventory_receipt_ref': refs['complete.json'],
           'artifact_inventory': inventory, 'artifact_bytes': 8,
           'measured_whole_operation_seconds': 2., 'measured_scientific_seconds': 1.,
           'actual_metadata_refs': list(refs.values()), 'prior_history': {'allocated_operations': []},
           'validation_provenance': {'partition': 'official_training_derived_validation',
               'official_tests_opened': False, 'worker_sha256': identity['worker_sha256'],
               'projection_sha256': identity['projection_sha256']}}
    return row, {'learning_identity': identity, 'lineage_sha256': 'b' * 64}, receipts, sources


class TabSynPopulationFits(unittest.TestCase):
    def fixture_cell(self, fixture):
        row, plan, receipts, sources = fixture
        return p.fit_cell(row, plan, lambda ref: receipts[ref['path']], sources)

    def test_complete_actual_exit_and_request_required(self):
        self.assertEqual(self.fixture_cell(cell_fixture())['charged_artifact_bytes'], 8)
        for change in ('exit_nonzero', 'exit_bool', 'model_incomplete', 'request_seed', 'cuda_binding'):
            f = cell_fixture(); row, _, receipts, _ = f
            if change == 'exit_nonzero': receipts['/opaque/exit.json']['actual_worker_exit'] = 2
            if change == 'exit_bool': receipts['/opaque/exit.json']['actual_worker_exit'] = False
            if change == 'model_incomplete': receipts['/opaque/model.json']['complete'] = False
            if change == 'request_seed': receipts['/opaque/request.json']['learning_identity']['fit_seed'] = True
            if change == 'cuda_binding': receipts['/opaque/worker.CUDA-initialized.receipt.json']['request_sha256'] = '0' * 64
            with self.subTest(change=change), self.assertRaises(ValueError): self.fixture_cell(f)

    def test_complete_projection_and_model_bytes_cannot_be_laundered(self):
        for change in ('projection', 'total', 'missing_file'):
            f = cell_fixture(); row = f[0]
            if change == 'projection': row['artifact_inventory']['projection_bytes'] = 0
            if change == 'total': row['artifact_bytes'] = 7
            if change == 'missing_file': del row['artifact_inventory']['files']['encoder-final.safetensors']
            with self.subTest(change=change), self.assertRaises(ValueError): self.fixture_cell(f)

    def test_population_requires_exactly_100_distinct_lineages(self):
        cell = self.fixture_cell(cell_fixture())
        cells = [dict(cell, dataset_id=f'{i:016x}', lineage_sha256=f'{i:064x}', retained_first_fit=i == 0) for i in range(100)]
        self.assertEqual(p.summarize(cells)['charged_model_and_projection_bytes'], 800)
        for rows in (cells[:-1], cells[:-1] + [cells[0]]):
            with self.assertRaises(ValueError): p.summarize(rows)

    def test_verified_receipt_and_source_bytes_precede_decode(self):
        base = p.HERE.parents[1] / 'target/tmp'; base.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=base) as directory:
            root = Path(directory)
            source = b'opaque frozen source\n'; receipt = b'{"opaque":true}'
            files = {'receipt.json': receipt, 'source.py': source}
            lock = {'model_bulk_copied': False, 'source_bytes_executed': False,
                    'files': {n: {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)} for n, b in files.items()},
                    'metadata': {'/opaque/receipt': {'file': 'receipt.json', 'sha256': hashlib.sha256(receipt).hexdigest()}},
                    'source_files': {'/opaque/source': {'file': 'source.py', 'sha256': hashlib.sha256(source).hexdigest()}}}
            for name, data in files.items(): (root / name).write_bytes(data)
            anchor = root / 'input-custody.lock.json'; anchor.write_bytes(encoded(lock)); expected = hashlib.sha256(anchor.read_bytes()).hexdigest()
            self.assertEqual(p.anchored(root, expected)[1]['source.py'], source)
            for name in files:
                (root / name).write_bytes(b'invalid rewritten bytes')
                with self.subTest(name=name), self.assertRaises(ValueError): p.anchored(root, expected)
                (root / name).write_bytes(files[name])
            lock['files']['receipt.json']['sha256'] = '0' * 64; anchor.write_bytes(encoded(lock))
            with self.assertRaises(ValueError): p.anchored(root, expected)

    def report(self):
        return json.loads((p.RESULTS / (p.NAME + '.json')).read_bytes())

    def test_committed_totals_and_unmeasured_cost_separation(self):
        report = self.report(); s = p.summarize(report['cells'])
        self.assertEqual(s, report['summary']); self.assertEqual(s['charged_model_and_projection_bytes'], 8486920261)
        self.assertAlmostEqual(s['measured_whole_seconds'], 5226.959524283186)
        self.assertEqual(s['prior_unmeasured_allocations'], 35)
        self.assertEqual(s['prior_unmeasured_book_charge_seconds'], 21000)
        self.assertFalse(report['book_charges_are_measured_method_cost'])
        self.assertEqual(report['publisher_sha256'], hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest())

    def test_committed_schema_tables_regenerate_and_rights_safe(self):
        report = self.report(); path = p.RESULTS / (p.NAME + '.json')
        table, markdown = p.tables(report)
        self.assertEqual(table, path.with_suffix('.csv').read_text())
        self.assertEqual(markdown, path.with_suffix('.md').read_text())
        self.assertEqual(p.schema(report), json.loads(path.with_suffix('.schema.json').read_bytes()))
        jsonschema.validate(report, p.schema(report))
        self.assertEqual(len(table.splitlines()), 101)
        self.assertNotIn('native_metadata', path.read_text()); self.assertNotIn('columns', path.read_text())

    def test_no_quality_or_certification_claim_from_fit_success(self):
        for key in (*p.NULL_CLAIMS, *p.INCOMPLETE):
            report = self.report(); report[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError): p.tables(report)
            with self.subTest(schema=key), self.assertRaises(jsonschema.ValidationError): jsonschema.validate(report, p.schema(self.report()))

    def test_schema_requires_opaque_ids_scaled_epochs_and_exact_source_pins(self):
        original = self.report(); schema = p.schema(original)
        for change in ('dataset', 'epochs', 'source'):
            report = self.report()
            if change == 'dataset': report['cells'][0]['dataset_id'] = 'source-name'
            if change == 'epochs': report['scaled_epochs']['vae'] = 4000
            if change == 'source': report['source_files'][next(iter(report['source_files']))] = '0' * 64
            with self.subTest(change=change), self.assertRaises(jsonschema.ValidationError): jsonschema.validate(report, schema)


if __name__ == '__main__': unittest.main()
