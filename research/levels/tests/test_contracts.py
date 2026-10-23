import copy
import unittest
from research.levels.contracts import ModelConfig, ResearchError, levels, validate_corpus


class ContractTests(unittest.TestCase):
    def test_all_excludes_ultra_and_aliases_are_unambiguous(self):
        self.assertEqual(levels('all'),['low','medium','high'])
        self.assertEqual(levels('large,med,small'),['low','medium','high'])
        with self.assertRaises(ResearchError):levels('low,small')

    def test_unknown_graph_and_unbounded_shape_refuse(self):
        for config in [ModelConfig(width=100000),ModelConfig(graph='arbitrary-code'),ModelConfig(heads=7)]:
            with self.assertRaises(ResearchError):config.validate()

    def test_family_roles_duplicates_and_rights_refuse(self):
        items=[dict(family_id=f'family:{i}',meta_role=role,train_sha256=str(i)*64,
                    source_role='official_train',rights=dict(admitted=True,license_id='MIT'),
                    target_index=2,split_policy='iid') for i,role in [(1,'train'),(2,'development')]]
        manifest=dict(format='dope-meta-corpus-v1',datasets=items)
        validate_corpus(manifest)
        for field,value in [('family_id','family:1'),('train_sha256','1'*64),
                            ('family_id','unresolved:source'),('source_role','official_test')]:
            invalid=copy.deepcopy(manifest);invalid['datasets'][1][field]=value
            with self.assertRaises(ResearchError):validate_corpus(invalid)
        invalid=copy.deepcopy(manifest);invalid['datasets'][0]['rights']['admitted']=False
        with self.assertRaises(ResearchError):validate_corpus(invalid)
