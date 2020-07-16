import xml.etree.ElementTree as ET
from treelib import Tree

from pgmpy.inference import VariableElimination

from bowtie.io.import_opsa import OPSA_Importer
from bowtie.mapping.mapping_bowtie import MappingBowTie


# Import a bow tie

# define opsa file
filename = '../data/test_et_v03_works.xml'
# filename = '../data/test_et_v03_true.xml'
filename = '../data/khakzad_dynamic_v03.xml'
# filename = '../data/Zarei_regulator_system_v01.xml'

# read file
xml_root = ET.parse(filename).getroot()
tree = Tree()
tree_root = tree.create_node('root')
importer = OPSA_Importer()
importer.parse(xml_root, tree, tree_root)

# show imported data
tree.show()

# map data
mapper = MappingBowTie(tree)
model = mapper.map()

# show model nodes
print('nodes')
print(model.nodes())

print('check bayesian model = {}'.format(model.check_model()))

# print CPD tables of bayesian network
for cpd in model.cpds:
    print(cpd)

# Inference of bayesian network
node_te = mapper.get_top_event(tree)
print('top event = {}'.format(node_te.tag))

infer = VariableElimination(model)
te_dist = infer.query(variables=[node_te.tag])
print(te_dist)

# Consequences
consequence = mapper.consequence_name
# print(consequence)
c_dist = infer.query(variables=[consequence])
print(c_dist)
print(c_dist.values)
