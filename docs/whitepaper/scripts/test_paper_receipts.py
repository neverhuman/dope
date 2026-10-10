"""Original metadata custody and paper reduction regressions; no scratch or rows."""
import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonschema
import paper_receipts as receipts


class PaperReceipts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extracts, cls.preparation = receipts.reduce_originals()

    def test_original_matrix_and_schema(self):
        index = receipts.load_index()
        schema = json.loads((receipts.REPO / 'research/benchmark/results/paper-original-metadata-v1/index.schema.json').read_text())
        jsonschema.validate(index, schema)
        self.assertEqual(len(index['records']), 210)
        self.assertFalse(index['original_historical_frozen_hashes_reconstructed'])
        self.assertFalse(index['replay_replaces_scored_artifact'])

    def test_every_original_and_projection_json_validates(self):
        root=receipts.REPO/'research/benchmark/results/paper-original-metadata-v1'
        validators={kind:jsonschema.Draft202012Validator(json.loads((root/(kind+'.schema.json')).read_text()))
                    for kind in ('receipt','report','filename-inventory','beyond-prepare','field-map')}
        index=receipts.load_index()
        for record in index['records']:
            for key,kind in [('receipt_ref','receipt'),('report_ref','report')]:
                validators[kind].validate(json.loads(receipts.checked_bytes(record[key])))
        validators['filename-inventory'].validate(json.loads(receipts.checked_bytes(index['filename_inventory_ref'])))
        validators['beyond-prepare'].validate(self.preparation)
        validators['field-map'].validate(self.extracts['original-field-map.json'])

    def test_whole_document_selectors_resolve_to_real_roots(self):
        mapping=self.extracts['original-field-map.json']
        for key in ('filename_counts','prepare'):
            ref=mapping[key]
            self.assertEqual(ref['pointer'],'')
            document=json.loads(receipts.checked_bytes(ref))
            self.assertIsInstance(document,dict)
        inventory=json.loads(receipts.checked_bytes(mapping['filename_counts']))
        self.assertEqual(sum(r['train_csv'] for r in inventory['s3_worker']),
                         self.extracts['provenance-counts.json']['s3_train_csv'])
        preparation=json.loads(receipts.checked_bytes(mapping['prepare']))
        self.assertEqual(preparation,self.preparation)

    def test_repair_receipt_sources_unchanged_and_gates_null(self):
        root=receipts.REPO/'research/benchmark/results/paper-original-metadata-v1'
        document=json.loads((root/'repair.json').read_bytes())
        schema=json.loads((root/'repair.schema.json').read_bytes())
        jsonschema.Draft202012Validator(schema).validate(document)
        refs=[document[k] for k in ('historical_audit','original_index','original_field_map','historical_paper_snapshot')]
        refs+=document['unchanged_measured_outputs']
        for ref in refs:
            path=Path(ref['path']);self.assertFalse(path.is_absolute());self.assertNotIn('..',path.parts)
            raw=(receipts.REPO/path).read_bytes()
            self.assertEqual((len(raw),hashlib.sha256(raw).hexdigest()),(ref['bytes'],ref['sha256']))
        self.assertEqual({r['finding'] for r in document['repairs']},{'TRACE-01','SCOPE-01'})
        for key in ('mfs_v2','ptf_v1','release_safe_l3','superiority'):
            self.assertIsNone(document[key])
        paper=(receipts.REPO/'docs/whitepaper/dope-mfs.tex').read_text()
        self.assertNotIn('The fits use one fit seed, 11',paper)
        self.assertNotIn('Fit seed $11$ is the population actually measured',paper)
        # SCOPE-01: fit seed 11 is the historical population; the five-seed refit is the headline.
        self.assertIn('historical fit-seed-$11$ population',paper)
        self.assertIn('The five-seed median in the abstract is the estimate this paper stands on.',paper)
        macros=(receipts.REPO/'docs/whitepaper/generated/numbers.tex').read_text()
        self.assertIn('not measured for the displayed DOPE generator',macros)

    def test_originals_reproduce_all_existing_extracts(self):
        for name, expected in self.extracts.items():
            with self.subTest(extract=name):
                actual = json.loads((receipts.REPO / 'docs/whitepaper/generated' / name).read_text())
                self.assertEqual(expected, actual)
        self.assertEqual((self.preparation['families'], self.preparation['prepared'], self.preparation['failed']), (12,5,7))
        self.assertEqual(sum(r.get('reason_category') == 'projected_feature_range' for r in self.preparation['results']), 4)
        self.assertEqual(sum(r.get('reason_category') == 'cross_partition_overlap' for r in self.preparation['results']), 3)
        forbidden = {'target','detail','trace','rows','source_values','extrema'}
        self.assertTrue(all(not forbidden.intersection(r) for r in self.preparation['results']))

    def test_index_drift_rejected_before_json_decode(self):
        with patch.object(receipts, 'INDEX_SHA256', '0'*64), patch.object(receipts.json, 'loads') as decode:
            with self.assertRaisesRegex(ValueError, 'drift'):
                receipts.load_index()
            decode.assert_not_called()

    def test_receipt_report_and_log_drift_rejected_before_decode(self):
        record = receipts.load_index()['records'][0]
        for key in ('receipt_ref','report_ref','loss_trace_ref'):
            changed = copy.deepcopy(record); changed[key]['sha256'] = '0'*64
            with self.subTest(key=key):
                if key == 'loss_trace_ref':
                    with patch.object(receipts.gzip, 'GzipFile') as decompress:
                        with self.assertRaisesRegex(ValueError,'drift'): receipts.load_loss(changed)
                        decompress.assert_not_called()
                else:
                    with patch.object(receipts.json, 'loads', wraps=json.loads) as decode:
                        with self.assertRaisesRegex(ValueError,'drift'): receipts.load_fit(changed)
                        self.assertEqual(decode.call_count, 0 if key == 'receipt_ref' else 1)

    def test_invalid_length_rejected_before_file_read(self):
        ref=receipts.load_index()['records'][0]['receipt_ref']
        for length in (0, False, 1_000_000):
            with self.subTest(length=length), patch.object(Path,'open') as opening:
                with self.assertRaisesRegex(ValueError,'length'):
                    receipts.checked_bytes(dict(ref,bytes=length))
                opening.assert_not_called()

    def test_path_escape_and_directory_symlink_rejected(self):
        ref = receipts.load_index()['records'][0]['receipt_ref']
        for path in ('/etc/passwd', '../receipt.json', 'research/benchmark/results/paper-original-metadata-v1/../receipt.json'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError,'path'):
                receipts.checked_bytes(dict(ref,path=path))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'research').symlink_to(receipts.REPO/'research', target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink'):
                receipts.checked_bytes(ref,root)

    def test_missing_or_mixed_profile_rejected(self):
        index=receipts.load_index()
        for mutation in ('missing','duplicate','profile'):
            changed=copy.deepcopy(index)
            if mutation=='missing':changed['records'].pop()
            elif mutation=='duplicate':changed['records'][-1]=changed['records'][0]
            else:changed['records'][0]['profile']='features12_steps8192'
            with self.subTest(mutation=mutation), patch.object(receipts,'checked_bytes',return_value=json.dumps(changed).encode()):
                with self.assertRaisesRegex(ValueError,'matrix|profiles'):
                    receipts.load_index()

    def test_official_test_scope_cannot_be_promoted(self):
        index=receipts.load_index(); index['official_tests_opened']=True
        with patch.object(receipts,'checked_bytes',return_value=json.dumps(index).encode()):
            with self.assertRaisesRegex(ValueError,'scope'):receipts.load_index()
        record=receipts.load_index()['records'][0]
        receipt,report=receipts.load_fit(record);receipt['official_tests_opened']=1
        with patch.object(receipts,'checked_bytes',side_effect=[json.dumps(receipt).encode(),json.dumps(report).encode()]):
            with self.assertRaisesRegex(ValueError,'receipt'):receipts.load_fit(record)

    def test_loss_shape_sequence_finiteness_and_raw_digest(self):
        record=receipts.load_index()['records'][0]
        original=copy.deepcopy(record)
        original['loss_trace_ref']['original_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'log drift'):receipts.load_loss(original)
        for raw in (b'0\t1\t2\n', b'1\t1\t2\n', b'0\tnan\t2\n', b'0\t1\t2\t3\n'):
            ref=copy.deepcopy(record);ref['loss_trace_ref'].update(original_bytes=len(raw),original_sha256=hashlib.sha256(raw).hexdigest())
            with self.subTest(raw=raw), patch.object(receipts,'checked_bytes',return_value=gzip.compress(raw,mtime=0)):
                with self.assertRaisesRegex(ValueError,'log'):receipts.load_loss(ref)

    def test_filename_inventory_never_opens_a_csv(self):
        original=receipts.checked_bytes; paths=[]
        def capture(ref,repo=receipts.REPO):
            paths.append(ref['path']);return original(ref,repo)
        with patch.object(receipts,'checked_bytes',side_effect=capture):receipts.reduce_originals()
        self.assertTrue(paths)
        self.assertTrue(all(not p.endswith('.csv') for p in paths))
        self.assertEqual(self.extracts['provenance-counts.json']['s3_worker_test_csv'],0)
        self.assertEqual(self.extracts['provenance-counts.json']['beyond_evaluator_test_csv'],5)

    def test_field_map_pointers_resolve_to_original_receipts(self):
        index=receipts.load_index();records={(r['kind'],r['profile'],r['lineage_or_family']):r for r in index['records']}
        for row in self.extracts['original-field-map.json']['fits']:
            record=records[row['kind'],row['profile'],row['identity']]
            for field,key in [('elapsed_seconds','receipt_ref'),('artifact_bytes','receipt_ref'),('fit_rows','report_ref'),('loss','loss_trace_ref')]:
                self.assertEqual(row[field]['sha256'],record[key]['sha256'])
                self.assertEqual(row[field]['path'],record[key]['path'])
                if field!='loss':
                    document=json.loads(receipts.checked_bytes(record[key]));self.assertIn(row[field]['pointer'][1:],document)

    def test_changed_extract_cannot_enter_a_panel(self):
        real_read = Path.read_text
        def changed(path, *args, **kwargs):
            if path.name == 'replay-cost.json':return '{}'
            return real_read(path, *args, **kwargs)
        with patch.object(Path,'read_text',changed):
            with self.assertRaisesRegex(ValueError,'extract drift'):receipts.checked_extracts()

    def test_changed_loss_summary_rejected_before_decode(self):
        with patch.object(receipts,'LOSS_SUMMARY_SHA256','0'*64), patch.object(receipts.json,'loads') as decode:
            with self.assertRaisesRegex(ValueError,'summary drift'):receipts.checked_loss_summary()
            decode.assert_not_called()


if __name__=='__main__':unittest.main()
