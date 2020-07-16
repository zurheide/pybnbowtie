"""
Run this test with unittest:

    $ python -m unittest -v mapping_bowtie_test
"""


import unittest

# import xml.etree.ElementTree as ET
from treelib import Tree
# from bowtie.io.import_opsa import OPSA_Importer
from bowtie.mapping.mapping_bowtie import MappingBowTie


class Test_Import(unittest.TestCase):

    def test_check_path(self):
        tree = Tree()
        mapper = MappingBowTie(tree)

        res = mapper.check_path(['S1'], ['S1'])
        self.assertTrue(res)

    def test_check_path_2(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        try:
            res = mapper.check_path(['S1'], ['S2'])
        except RuntimeError:
            self.assertTrue(True)

    def test_check_path_3(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        try:
            res = mapper.check_path(['S1', 'S3', 'S5'],
                                    ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
        except RuntimeError:
            self.assertTrue(True)

    def test_check_path_4(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        try:
            res = mapper.check_path(['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
                                    ['S1', 'S6', 'S4', 'S5'])
        except RuntimeError:
            self.assertTrue(True)

    def test_check_path_5(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        res = mapper.check_path(['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
                                ['S1', 'S3', 'S4', 'S6'])
        self.assertTrue(res)

    def test_split_probability_table(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        p = [(('Alarm', 'true'), ('Sprinkler', 'true'), '0.2250'),
             (('Alarm', 'true'), ('Ignition', 'false'), '0.2250'),
             (('Alarm', 'true'), ('Ignition', 'false'), '0.225'),
             (('Alarm', 'false'), ('Sprinkler', 'true'), '0.9987'),
             (('Alarm', 'false'), ('Sprinkler', 'false'), '0.9987'),
             (('Alarm', 'true'), ('Ignition', 'true'), '0.0013'),
             (('Alarm', 'true'), ('Sprinkler', 'true'), '0.0013'),
             (('Alarm', 'false'), ('Sprinkler', 'true'), '0.775'),
             (('Alarm', 'false'), ('Ignition', 'true'), '0.9987'),
             (('Alarm', 'false'), ('Sprinkler', 'false'), '0.775'),
             (('Alarm', 'false'), ('Ignition', 'false'), '0.775'),
             (('Alarm', 'true'), ('Sprinkler', 'false'), '0.0013'),
             (('Alarm', 'true'), ('Sprinkler', 'false'), '0.225')]

        name_list, sans_name_list = mapper.split_probability_table('Alarm', p)

        self.assertEqual(len(name_list), 13)
        self.assertEqual(len(name_list[0]), 1)
        self.assertEqual(len(sans_name_list), 13)
        self.assertEqual(len(sans_name_list[0]), 1)

    def test_split_probability_table_2(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        p = [(('Alarm', 'true'), ('Sprinkler', 'true'), '0.2250'),
             (('Alarm', 'true'), ('Ignition', 'false'), '0.2250'),
             (('Alarm', 'true'), ('Ignition', 'false'), '0.225'),
             (('Alarm', 'false'), ('Sprinkler', 'true'), '0.9987'),
             (('Alarm', 'false'), ('Sprinkler', 'false'), '0.9987'),
             (('Alarm', 'true'), ('Ignition', 'true'), '0.0013'),
             (('Alarm', 'true'), ('Sprinkler', 'true'), '0.0013'),
             (('Alarm', 'false'), ('Sprinkler', 'true'), '0.775'),
             (('Alarm', 'false'), ('Ignition', 'true'), '0.9987'),
             (('Alarm', 'false'), ('Sprinkler', 'false'), '0.775'),
             (('Alarm', 'false'), ('Ignition', 'false'), '0.775'),
             (('Alarm', 'true'), ('Sprinkler', 'false'), '0.0013'),
             (('Alarm', 'true'), ('Sprinkler', 'false'), '0.225')]

        name_list, sans_name_list = mapper.split_probability_table('Lala', p)

        self.assertEqual(len(name_list), 13)
        self.assertEqual(len(name_list[0]), 0)
        self.assertEqual(len(sans_name_list), 13)
        self.assertEqual(len(sans_name_list[0]), 2)

    def test_split_probability_table_3(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        p = [(('e2', 'o22'), ('e1', 'o12'), '0.9'),
             (('e2', 'o21'), ('e1', 'o12'), '0.1'),
             (('e2', 'o22'), ('e1', 'o11'), '0.99'),
             (('e2', 'o21'), ('e1', 'o11'), '0.01')]

        name_list, sans_name_list = mapper.split_probability_table('e2', p)

        self.assertEqual(len(name_list), 4)
        self.assertEqual(len(name_list[0]), 1)
        self.assertEqual(len(sans_name_list), 4)
        self.assertEqual(len(sans_name_list[0]), 1)

    def test_states_in_probability(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        p = [(('e2', 'o22'), ('e1', 'o12'), '0.9'),
             (('e2', 'o21'), ('e1', 'o12'), '0.1'),
             (('e2', 'o22'), ('e1', 'o11'), '0.99'),
             (('e2', 'o21'), ('e1', 'o11'), '0.01')]

        res = mapper.states_in_probability(p)

        # print(res)
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res['e1']), 2)
        self.assertEqual(len(res['e2']), 2)
        # check if o11 and o12 are in e1 array
        self.assertIn('o11', res['e1'])
        self.assertIn('o12', res['e1'])
        # check if o11 and o12 are in e1 array
        self.assertIn('o21', res['e2'])
        self.assertIn('o22', res['e2'])

    def test_find_state_for_probabilities(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        p = [[('e1', 'o11'), ('e2', 'o21'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'o11'), ('e2', 'o21'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o11'), ('e2', 'o22'),
              ('failure leakage', 'fails'), 'c2'],
             [('e1', 'o11'), ('e2', 'o22'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o12'), ('e2', 'o21'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'o12'), ('e2', 'o21'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o12'), ('e2', 'o22'),
              ('failure leakage', 'fails'), 'c3'],
             [('e1', 'o12'), ('e2', 'o22'),
              ('failure leakage', 'works'), 'Safe']]

        v = [('e1', 'o12'), ('e2', 'o22'), ('failure leakage', 'works')]
        res = mapper.find_state_for_probabilities(p, v)

        self.assertEqual(res, 'Safe')

    def test_find_state_for_probabilities_2(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        p = [[('e1', 'o11'), ('e2', 'o21'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'o11'), ('e2', 'o21'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o11'), ('e2', 'o22'),
              ('failure leakage', 'fails'), 'c2'],
             [('e1', 'o11'), ('e2', 'o22'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o12'), ('e2', 'o21'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'o12'), ('e2', 'o21'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o12'), ('e2', 'o22'),
              ('failure leakage', 'fails'), 'c3'],
             [('e1', 'o12'), ('e2', 'o22'),
              ('failure leakage', 'works'), 'Safe']]

        v = [('e1', 'o12'), ('e2', 'o22'), ('failure leakage', 'fails')]
        res = mapper.find_state_for_probabilities(p, v)

        self.assertEqual(res, 'c3')

    def test_find_state_for_probabilities_3(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        p = [[('e1', 'o11'), ('e2', 'o21'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'o11'), ('e2', 'o21'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o11'), ('e2', 'o22'),
              ('failure leakage', 'fails'), 'c2'],
             [('e1', 'o11'), ('e2', 'o22'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o12'), ('e2', 'o21'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'o12'), ('e2', 'o21'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o12'), ('e2', 'o22'),
              ('failure leakage', 'fails'), 'c3'],
             [('e1', 'o12'), ('e2', 'o22'),
              ('failure leakage', 'works'), 'Safe']]

        v = [('x1', 'x12'), ('x2', 'ox2'), ('nada', 'rien')]
        res = mapper.find_state_for_probabilities(p, v)

        self.assertEqual(res, None)

    def test_derived_values_from_probabilities(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        p = [[('e1', 'o11'), ('e2', 'o21'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'o11'), ('e2', 'o21'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o11'), ('e2', 'o22'),
              ('failure leakage', 'fails'), 'c2'],
             [('e1', 'o11'), ('e2', 'o22'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o12'), ('e2', 'o21'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'o12'), ('e2', 'o21'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'o12'), ('e2', 'o22'),
              ('failure leakage', 'fails'), 'c3'],
             [('e1', 'o12'), ('e2', 'o22'),
              ('failure leakage', 'works'), 'Safe']]

        s = {'c3', 'c2', 'c1', 'Safe'}
        consequence = "Consequenzzzze"
        value_list, evidence, evidence_card, state_names = mapper.derived_values_from_probabilities(p, s, consequence)

        self.assertEqual(len(value_list), 4)
        self.assertEqual(len(evidence), 3)
        self.assertIn("failure leakage", evidence)
        self.assertIn("e1", evidence)
        self.assertIn("e2", evidence)

        self.assertEqual(len(evidence_card), 3)
        self.assertEqual(evidence_card[0], 2)
        self.assertEqual(evidence_card[1], 2)
        self.assertEqual(evidence_card[2], 2)

        self.assertEqual(len(state_names), 4)
        self.assertEqual(len(state_names[consequence]), 4)
        self.assertIn("c1", state_names[consequence])
        self.assertIn("c2", state_names[consequence])
        self.assertIn("c3", state_names[consequence])
        self.assertIn("Safe", state_names[consequence])
        self.assertIn("works", state_names["failure leakage"])
        self.assertIn("fails", state_names["failure leakage"])
        self.assertIn(None, state_names["e1"])
        self.assertIn(None, state_names["e2"])

    def test_derived_values_from_probabilities_2(self):
        tree = Tree()
        mapper = MappingBowTie(tree)
        p = [[('e1', 'yes'), ('e2', 'true'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'yes'), ('e2', 'true'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'yes'), ('e2', 'false'),
              ('failure leakage', 'fails'), 'c2'],
             [('e1', 'yes'), ('e2', 'false'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'no'), ('e2', 'true'),
              ('failure leakage', 'fails'), 'c1'],
             [('e1', 'no'), ('e2', 'true'),
              ('failure leakage', 'works'), 'Safe'],
             [('e1', 'no'), ('e2', 'false'),
              ('failure leakage', 'fails'), 'c3'],
             [('e1', 'no'), ('e2', 'false'),
              ('failure leakage', 'works'), 'Safe']]

        s = {'c3', 'c2', 'c1', 'Safe'}
        consequence = "Consequenzzzze"
        value_list, evidence, evidence_card, state_names = mapper.derived_values_from_probabilities(p, s, consequence)

        self.assertEqual(len(value_list), 4)
        self.assertEqual(len(evidence), 3)
        self.assertIn("failure leakage", evidence)
        self.assertIn("e1", evidence)
        self.assertIn("e2", evidence)

        self.assertEqual(len(evidence_card), 3)
        self.assertEqual(evidence_card[0], 2)
        self.assertEqual(evidence_card[1], 2)
        self.assertEqual(evidence_card[2], 2)

        self.assertEqual(len(state_names), 4)
        self.assertEqual(len(state_names[consequence]), 4)
        self.assertIn("c1", state_names[consequence])
        self.assertIn("c2", state_names[consequence])
        self.assertIn("c3", state_names[consequence])
        self.assertIn("Safe", state_names[consequence])
        self.assertIn("works", state_names["failure leakage"])
        self.assertIn("fails", state_names["failure leakage"])
        self.assertIn("yes", state_names["e1"])
        self.assertIn("no", state_names["e1"])
        self.assertIn("true", state_names["e2"])
        self.assertIn("false", state_names["e2"])


if __name__ == '__main__':
    unittest.main()
