import unittest

import xml.etree.ElementTree as ET
from treelib import Tree
from bowtie.io.import_opsa import OPSA_Importer


class Test_Import(unittest.TestCase):

    def test_import(self):
        # filename = "../data/test_et_v03_fails.xml"
        filename = "data/test_et_v03_fails.xml"

        # read file
        xml_root = ET.parse(filename).getroot()
        tree = Tree()
        tree_root = tree.create_node('root')
        importer = OPSA_Importer()
        importer.parse(xml_root, tree, tree_root)

        self.assertTrue(isinstance(tree, Tree))


if __name__ == '__main__':
    unittest.main()
