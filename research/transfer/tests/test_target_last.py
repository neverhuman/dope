"""Target-last is an intake/consumer invariant, not a positional guess."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from research.transfer.target_last import (
    TargetLastError, normalize_csv, normalize_files, require_target_last, schema_columns,
)
from research.transfer.pilot_worker_inputs import _layout

ROOT=Path(__file__).resolve().parents[3]
def schema(order):
    return json.dumps({"format":"dope-table-schema","version":1,"columns":[
        {"id":{"a":"c0001","b":"c0002","y":"c0003"}[n],"source_name":n,
         "role":"target" if n=="y" else "feature","dtype":"continuous"} for n in order]}).encode()

class TargetLastTests(unittest.TestCase):
    def test_first_middle_and_last_all_produce_same_declared_feature_target_values(self):
        for order in [('y','a','b'),('a','y','b'),('a','b','y')]:
            raw=(','.join(order)+'\n'+','.join({'a':'1','b':'2','y':'9'}[n] for n in order)+'\n').encode()
            result=normalize_csv(raw,schema(order),target='y')
            self.assertEqual(result.csv,b'a,b,y\n1,2,9\n')
            self.assertEqual(require_target_last(result.csv,result.schema,expected_target='y'),(['a','b','y'],'y'))
            self.assertEqual(result.receipt['source_csv_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(result.receipt['source_column_indices_in_output_order'],[order.index(n) for n in ['a','b','y']])
            self.assertTrue(result.receipt['row_order_unchanged'])
            self.assertFalse(result.receipt['numeric_values_or_query_labels_parsed'])

    def test_canonical_input_is_byte_preserving_and_idempotent(self):
        raw=b'a,b,y\r\n"1",2,"9"\r\n';original=schema(('a','b','y'))
        first=normalize_csv(raw,original,target='y');again=normalize_csv(first.csv,first.schema,target='y')
        self.assertEqual((first.csv,first.schema),(raw,original))
        self.assertEqual((again.csv,again.schema),(raw,original))
        self.assertEqual(first.receipt['source_column_indices_in_output_order'],[0,1,2])

    def test_feature_order_comes_from_declaration_not_incoming_csv_position(self):
        result=normalize_csv(b'b,y,a\n2,9,1\n',schema(('y','a','b')),target='y')
        self.assertEqual(result.csv,b'a,b,y\n1,2,9\n')
        self.assertEqual(result.receipt['source_column_indices_in_output_order'],[2,0,1])

    def test_query_target_text_is_opaque_and_never_added_to_features(self):
        for value in ('PRIVATE_SENTINEL','NaN','1e999','not-a-number'):
            raw=('y,a,b\n'+value+',1,2\n').encode()
            result=normalize_csv(raw,schema(('y','a','b')),target='y')
            rows=list(csv.reader(io.StringIO(result.csv.decode())))
            self.assertEqual(rows[1][:-1],['1','2']);self.assertEqual(rows[1][-1],value)

    def test_declared_target_mismatch_is_refused_even_when_wrong_schema_target_is_last(self):
        value=json.loads(schema(('y','a','b')))
        value['columns'][0]['role']='feature';value['columns'][-1]['role']='target'
        with self.assertRaisesRegex(TargetLastError,'target_declaration_mismatch'):
            normalize_csv(b'y,a,b\n9,1,2\n',json.dumps(value).encode(),target='y')

    def test_consumers_reject_unconverted_schema_or_header(self):
        with self.assertRaisesRegex(TargetLastError,'target_must_be_last'):
            require_target_last(b'y,a,b\n9,1,2\n',schema(('y','a','b')))
        with self.assertRaisesRegex(TargetLastError,'target_last_header_mismatch'):
            require_target_last(b'y,a,b\n9,1,2\n',schema(('a','b','y')))
        with self.assertRaisesRegex(TargetLastError,'target_must_be_last'):
            _layout(schema(('y','a','b')))

    def test_duplicate_or_missing_target_and_bad_rows_fail_with_static_errors(self):
        values=[];base=json.loads(schema(('a','b','y')))
        for roles in (['feature','feature','feature'],['target','feature','target']):
            v=deepcopy(base)
            for c,r in zip(v['columns'],roles):c['role']=r
            values.append(json.dumps(v).encode())
        values.append(b'{"format":"dope-table-schema","format":"bad","version":1,"columns":[]}')
        for value in values:
            with self.assertRaises(TargetLastError) as caught:normalize_csv(b'a,b,y\n1,2,9\n',value,target='y')
            self.assertNotIn('PRIVATE_SENTINEL',str(caught.exception))
        for raw in (b'a,a,y\n1,2,9\n',b'a,b,y\n1,2\n',b'a,b,y\n1,2,9,4\n',b'PRIVATE_SENTINEL'):
            with self.assertRaises(TargetLastError) as caught:normalize_csv(raw,schema(('a','b','y')),target='y')
            self.assertNotIn('PRIVATE_SENTINEL',str(caught.exception))

    def test_control_fit_rejects_noncanonical_input_before_any_fitting(self):
        from research.transfer.tdl.generators import GaussianCopulaGenerator
        directory=ROOT/'target/target-last-tests';directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as td:
            p=Path(td);csv_path=p/'source.csv';schema_path=p/'schema.json'
            csv_path.write_bytes(b'y,a,b\n9,1,2\n8,3,4\n7,5,6\n');schema_path.write_bytes(schema(('y','a','b')))
            with mock.patch('research.transfer.tdl.generators._copula_state') as fit:
                with self.assertRaises(ValueError):GaussianCopulaGenerator().fit(csv_path,schema_path,11)
            fit.assert_not_called()

    def test_identifier_and_group_roles_cannot_become_python_model_features(self):
        for role in ('entity_key', 'group_split_key', 'timestamp', 'excluded_sensitive_identifier', 'unknown'):
            value=json.loads(schema(('a','b','y')))
            value['columns'][0]['role']=role
            with self.assertRaisesRegex(TargetLastError,"target_last_schema_invalid"):
                normalize_csv(b'a,b,y\n1,2,9\n',json.dumps(value).encode(),target='y')

    def test_serialization_receipt_distinguishes_cell_text_from_csv_bytes(self):
        raw=b'y,a,b\r\n"9","1",2\r\n'
        first=normalize_csv(raw,schema(('y','a','b')),target='y')
        self.assertEqual(first.receipt['csv_serialization'],'utf8_lf_minimal_csv_quoting')
        self.assertEqual(first.csv,b'a,b,y\n1,2,9\n')
        again=normalize_csv(first.csv,first.schema,target='y')
        self.assertEqual(again.receipt['csv_serialization'],'source_bytes_unchanged')

    def test_file_intake_preserves_originals_and_refuses_output_overwrite(self):
        directory=ROOT/'target/target-last-tests';directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as td:
            p=Path(td).resolve();raw=b'y,a,b\n9,1,2\n';source=p/'source.csv';source.write_bytes(raw)
            decl=p/'source.schema.json';decl.write_bytes(schema(('y','a','b')))
            result=normalize_files(source,decl,p/'canonical',target='y')
            self.assertEqual(source.read_bytes(),raw)
            self.assertEqual((p/'canonical/table.csv').read_bytes(),result.csv)
            self.assertTrue((p/'canonical/normalization.json').is_file())
            with self.assertRaises(FileExistsError):normalize_files(source,decl,p/'canonical',target='y')

if __name__=='__main__':unittest.main()
