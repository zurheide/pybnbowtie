"""
This module holds the ``OPSA_Importer`` class. It takes a XML
document and parses it to a treelib tree.

Example of usage::

    import xml.etree.ElementTree as ET
    from treelib import Tree
    from bowtie.io.import_opsa import OPSA_Importer

    # read XML file, here opsa_input.xml
    xml_root = ET.parse("opsa_input.xml").getroot()
    # create and prepare tree for results
    tree = Tree()
    tree_root = tree.create_node('root')
    # create importer and parse
    importer = OPSA_Importer()
    importer.parse(xml_root, tree, tree_root)
    # show results
    tree.show()

"""


from treelib import Tree

from bowtie.dataresources.type_definitions import GateType, EventType
from bowtie.dataresources.opsa import OPSA_Container, OPSA_Gate, \
    OPSA_Basic_Event, OPSA_Initiating_Event, OPSA_Functional_Event, \
    OPSA_Sequence, OPSA_Path
from .utilities import tree_copy


class OPSA_Importer:
    """
    Class that parses a XML in opsa definition and returns a tree
    with the data for further processing.

    Implemented gates: ``OR``, ``AND``.
    """

    def check_root(self, root):
        """
        Check that the XML is of opsa-mef format. Returns True or raises
        an Exception.
        """
        expected_tag = 'opsa-mef'
        if root.tag != expected_tag:
            raise Exception('root.tag unknown: <{}>. Should be <{}>'
                            .format(root.tag, expected_tag))
        return True

    def tree_find_node(self, tree, node_name, data_class):
        '''
        Returns the first node in the tree with node_name and of type
        data_class. Returns the node or None.
        '''
        # convert the tree to a list for easier iterating
        aList = tree.all_nodes()

        for el in aList:
            if el.tag == node_name:
                if isinstance(el.data, data_class):
                    return el

        return None

    def tree_find_nodes(self, tree, node_name, data_class):
        """
        Returns a list with all found nodes with name_name
        and of class data_class.
        """

        aList = tree.all_nodes()

        result = []
        for el in aList:
            if node_name is None or el.tag == node_name:
                if isinstance(el.data, data_class):
                    result.append(el)

        return result

    def bd_define_basic_event(self, xml_node, tree):
        """
        Parses basic event definition (``define-basic-event``) in
        the ``model-data`` OPSA data structure.
        Reads data (probability and label) and stores it in the previously
        defined nodes.
        """

        # search for existing event (should already be created before)
        existing_nodes = self.tree_find_nodes(tree,
                                              xml_node.get('name'),
                                              OPSA_Basic_Event)

        if len(existing_nodes) < 1:
            raise RuntimeError('Node does not exist for tag:',
                               xml_node.get('name'))

        for a_node in existing_nodes:
            d = a_node.data

            for e in xml_node:

                if e.tag == 'float':
                    # while reading value convert it diretly to a float
                    d.set_probability(float(e.get('value')))
                elif e.tag == 'label':
                    d.set_label(e.text.strip())
                elif e.tag == 'attributes':
                    for attrib in e:
                        d.add_attribute(attrib.get('name'),
                                        attrib.get('value'))
                elif e.tag == 'parameter':
                    d.set_parameter(e.get('name'))
                else:
                    raise NotImplementedError('bd_define_basic_event not' +
                                              'implemented property:', e.tag)

    def define_parameter(self, xml_node, tree):
        """
        Parses parameters in the ``model-data`` OPSA data structure.
        Reads the parameter and stores it in the nodes that have this
        parameter defined.
        """

        name = xml_node.get('name')

        # read parameter values
        for e in xml_node:
            if e.tag == 'float':
                probability = float(e.get('value'))
            # elif e.tag == 'parameter':
            #    parameter = e.get('name')
            # elif e.tag == 'sub':
            #    pass
            # elif e.tag == 'exponential':
            #    print('*************************************')
            # elif e.tag == 'lognormal-deviate':
            #    print('*************************************')
            else:
                raise NotImplementedError('define_parameter not' +
                                          'implemented property:', e.tag)

        existing_paths = self.tree_find_nodes(tree, None, OPSA_Path)
        existing_basic_events = self.tree_find_nodes(tree, None,
                                                     OPSA_Basic_Event)

        found_parameters = False

        if existing_paths:
            for node in existing_paths:
                path = node.data
                if name == path.parameter:
                    if probability:
                        path.set_probability(probability)
                        found_parameters = True
        else:
            raise RuntimeError(' no objects with parameters found, perhaps' +
                               'other classes will fit')

        if existing_basic_events:
            for node in existing_basic_events:
                basic_event = node.data
                if name == basic_event.parameter:
                    if probability:
                        basic_event.set_probability(probability)
                        found_parameters = True

        if found_parameters is False:
            print('WARNING: parameter not used. Parameter name = <{}>'
                  .format(xml_node.get('name')))

    def parse_model_data_node(self, xml_node, tree):
        """
        Parses a single entry of the ``model-data`` OPSA data structure.
        Data can be of type ``define-basic-event`` or ``define-parameter``.
        """

        if xml_node.tag == 'define-basic-event':
            self.bd_define_basic_event(xml_node, tree)
        elif xml_node.tag == 'define-parameter':
            self.define_parameter(xml_node, tree)
        else:
            raise NotImplementedError('parse_model_data_node unhandled {}'
                                      .format(xml_node.tag))

    def parse_et_collect_formula(self, xml_node, tree, tree_parent):
        """
        Parses in the event tree (et) the collect-formula.
        """

        parent = tree_parent

        xml = xml_node
        if xml.tag == 'not':
            opsa_gate = OPSA_Gate(xml_node.tag, GateType.NOT)
            gate = tree.create_node(xml_node.tag, parent=parent,
                                    data=opsa_gate)
            parent = gate
            xml = xml[0]

        # now parse all data in the formula
        for elem in xml.iter():
            tag = elem.tag
            if tag == 'gate':
                # create a link
                opsa_gate = OPSA_Gate(elem.get('name'), GateType.LINK)
                gate = tree.create_node(elem.get('name'), parent=parent,
                                        data=opsa_gate)

    def parse_model_data(self, xml_node, tree):
        """
        Parses the ``model-data`` in the OPSA data structure.
        Calls for each entry the function ``parse_model_data_node``.
        """

        # traverse all nodes of the fault tree
        for subnode in xml_node:
            self.parse_model_data_node(subnode, tree)

    def et_define_functional_event(self, xml_node, tree, parent, a_dict):
        """
        Parses in the event tree (et) the functional event.
        The element contains mainly a description in the  ``label``.
        """

        # sanity check if functional events are previously stored in the tree
        existing_nodes = self.tree_find_nodes(tree, xml_node.get('name'),
                                              OPSA_Functional_Event)
        if len(existing_nodes) > 0:
            print('aaaargh, should not be, first we define the'
                  + 'functional-event, then we store them in the tree')
            raise RuntimeError('existing functional-events should not happen'
                               + 'at this stage of computation')

        fe = OPSA_Functional_Event(xml_node.get('name'))

        for e in xml_node:
            if e.tag == 'label':
                fe.set_label(e.text.strip())
            else:
                raise NotImplementedError('ft_define_basic_event not'
                                          + 'implemented property:', e.tag)

        a_dict[xml_node.get('name')] = fe

    def parse_et_collect_expression(self, xml_node):
        """
        Parses in the event tree (et) the collect-expression.
        The probability of the path is stored in the ``float`` element.
        """

        parameter = None
        probability = None

        for elem in xml_node.iter():
            tag = elem.tag

            if tag == 'parameter':
                parameter = elem.get('name')
            elif tag == 'sub':
                # just continue reading, not handled
                pass
            elif tag == 'float':
                probability = float(elem.get('value'))
            else:
                raise NotImplementedError('this tag is not jet handled', tag)

        return parameter, probability

    def parse_et_path(self, xml_node, tree, tree_parent, a_dict):
        """
        Parses in the event tree (et) a path.

        This function is recursive. If the path contains another fork it
        calls the caller function ``parse_et_fork``.
        """

        parent = tree_parent
        state = xml_node.get('state')

        # create artificial name for path
        path_name = 'Path' + '.' + tree_parent.data.name + '.' + state
        path = OPSA_Path(name=path_name, state=state)

        # now add the path object to the tree, we will change it later
        n = tree.create_node(path_name, parent=parent, data=path)
        parent = n    # now the path is the new parent

        for elem in xml_node:
            if elem.tag == 'collect-expression':
                param, prob = self.parse_et_collect_expression(elem[0])
                if param:
                    n.data.set_parameter(param)
                if prob:
                    n.data.set_probability(prob)
            elif elem.tag == 'collect-formula':
                self.parse_et_collect_formula(elem[0], tree, parent)
            elif elem.tag == 'sequence':
                seq = a_dict[elem.get('name')]

                if seq is None:
                    raise RuntimeError('Sequence <{}> is not defined'
                                       .format(elem.get('name')))
                tree.create_node(elem.get('name'), parent=parent, data=seq)
            elif elem.tag == 'fork':
                self.parse_et_fork(elem, tree, parent, a_dict)
            elif elem.tag == 'branch':
                # find branch in a_dict
                subtree = a_dict[elem.get('name')]
                if subtree is None:
                    raise RuntimeError('branch {} was not defined'
                                       .format(elem.get('name')))

                t = tree_copy(subtree)

                tree.paste(parent.identifier, t)
            else:
                raise NotImplementedError('parse_et_path:', elem.tag)

    def parse_et_fork(self, xml_node, tree, tree_parent, a_dict):
        """
        Parses in the event tree (et) the fork.
        The fork for the functional event holds a path. This path is then
        parsed by calling ``parse_et_path``.
        """

        parent = tree_parent
        func_elem_name = xml_node.get('functional-event')
        func_elem = a_dict.get(func_elem_name)

        if func_elem is None:
            raise RuntimeError('No functional-event to fork found with name'
                               + ' <{}>'.format(func_elem_name))

        # add functional-event (triggered from fork) to tree
        n = tree.create_node(func_elem.name, parent=parent, data=func_elem)
        if n:
            parent = n

        for elem in xml_node:
            if elem.tag == 'path':
                self.parse_et_path(elem, tree, parent, a_dict)
            else:
                raise RuntimeError('fork should only contain path elements'
                                   + 'and not <{}>:'.format(elem.tag))

    def parse_et_initial_state(self, xml_node, tree, tree_parent, a_dict):
        """
        Parses in the event tree (et) the initial state. Each path is
        parsed by calling ``parse_et_fork``.
        """

        for sub in xml_node:
            if sub.tag == 'fork':
                self.parse_et_fork(sub, tree, tree_parent, a_dict)
            else:
                raise NotImplementedError('unknown tag:', xml_node.tag)

    def et_define_sequence(self, xml_node, tree, tree_parent, a_dict):
        """
        Parses the event tree (et) sequence. Result of parsing is stored
        in ``a_dict`` for later usage.
        """

        # sanity check if functional events are previously stored in the tree
        existing_nodes = self.tree_find_nodes(tree, xml_node.get('name'),
                                              OPSA_Sequence)
        if len(existing_nodes) > 0:
            print('aaaargh, should not be, first we define the sequence, then'
                  + 'we store them in the tree')
            raise RuntimeError('existing functional-events should not happen'
                               + 'at this stage of computation')

        seq = OPSA_Sequence(xml_node.get('name'))

        for e in xml_node:
            if e.tag == 'label':
                seq.set_label(e.get('label'))
            elif e.tag == 'event-tree':
                seq.set_event_tree(e.get('name'))
            else:
                raise NotImplementedError('et_define_sequence not implemented'
                                          + 'property:', e.tag)

        a_dict[xml_node.get('name')] = seq

    def et_define_branch(self, xml_node, tree, tree_parent, a_dict):
        """
        Parses a branching data structure by calling ``parse_et_fork``.
        """

        subtree = Tree()
        parent = subtree.root

        for e in xml_node:
            self.parse_et_fork(e, subtree, parent, a_dict)

        if subtree.size() == 0:
            raise RuntimeError('Event tree branch contains no data')

        a_dict[xml_node.get('name')] = subtree

    def parse_event_tree_node(self, xml_node, tree, tree_parent, a_dict):
        """
        Parses the event tree by calling functions to parse the data for
        functional event, sequence, initial state and branch.

        Communication and preprocessed data is stored and shared in the
        ``a_dict`` dictionary.
        """

        if xml_node.tag == 'define-functional-event':
            self.et_define_functional_event(xml_node, tree, tree_parent,
                                            a_dict)
        elif xml_node.tag == 'define-sequence':
            self.et_define_sequence(xml_node, tree, tree_parent, a_dict)
        elif xml_node.tag == 'initial-state':
            self.parse_et_initial_state(xml_node, tree, tree_parent, a_dict)
        elif xml_node.tag == 'define-branch':
            self.et_define_branch(xml_node, tree, tree_parent, a_dict)
        else:
            raise NotImplementedError('parse_event_tree_node(...) does not'
                                      + 'handle {}'.format(xml_node.tag))

    def parse_event_tree(self, xml_node, tree, tree_parent):
        """
        Parses the event tree. Before parsing the initiating event is created
        (if it does not exist). Parses all subtree nodes of the event tree.
        """

        parent = tree_parent

        # check if there is an initiating-event that points to the
        # current event-tree
        existing_nodes = self.tree_find_nodes(tree, None,
                                              OPSA_Initiating_Event)
        if len(existing_nodes) > 0:
            # check if 'Initiating_Event.event-tree == xml_node.name'
            for node in existing_nodes:
                if node.data.event_tree == xml_node.get('name'):
                    parent = node

        # create new event-tree node in tree
        n = tree.create_node(xml_node.get('name'), parent=parent,
                             data=OPSA_Container(xml_node.get('name'),
                                                 EventType.EVENT_TREE))
        # traverse all nodes of the fault tree
        a_dict = {}
        for subnode in xml_node:
            self.parse_event_tree_node(subnode, tree, n, a_dict)

    def parse_gate_type(self, node):
        """
        Check if gate type is or, not, at_least.
        **Only add new gate types if they are properly handled.**
        Returns class ``GateType``.
        """
        tag = node.tag
        if tag == 'or':
            return GateType.OR
        if tag == 'and':
            return GateType.AND
        if tag == 'atleast':
            return GateType.ATLEAST

        raise NotImplementedError('gate type unknown {}'.format(node.tag))

    def ft_define_gate(self, xml_gate, tree, parent):
        """
        Parses in a fault tree (ft) a gate. Handles *AND*, *OR* and *AT_LEAST*
        (**Note:** *AT_LEAST* is not jet converted to a bayesian network).

        For large OPSA definitions reoccuring parts are left out by just
        stopping with the gate name that is defined somewhere else in the
        data structure. Therefore this function has to take into account if
        the gate does alredy exist.
        """

        gate_label = None
        for subnode in xml_gate:
            if (subnode.tag == 'and'
                    or subnode.tag == 'or'
                    or subnode.tag == 'atleast'):
                # get gate type and then create connected events, gates, ...
                # to this gate
                gate_type = self.parse_gate_type(subnode)
                existing_node = self.tree_find_node(tree, xml_gate.get('name'),
                                                    OPSA_Gate)
                if existing_node is None:
                    opsa_gate = OPSA_Gate(xml_gate.get('name'), gate_type,
                                          gate_label)
                    if gate_type == GateType.ATLEAST:
                        # refine the opsa_gate with atleast value
                        opsa_gate.set_atleast_min(subnode.get('min'))

                    gate = tree.create_node(xml_gate.get('name'),
                                            parent=parent,
                                            data=opsa_gate)
                else:
                    # existing_node
                    gate = existing_node
                    gate.data.gate_type = gate_type

                    if gate_type == GateType.ATLEAST:
                        gate.data.set_atleast_min(subnode.get('min'))

                if len(subnode) <= 0:
                    raise RuntimeError("No connections to gate: {}"
                                       .format(xml_gate.get('name')))

                for e in subnode:
                    if e.tag == 'gate':
                        opsa_gate = OPSA_Gate(e.get('name'), None)
                        tree.create_node(e.get('name'), parent=gate,
                                         data=opsa_gate)
                    elif e.tag == 'basic-event' or e.tag == 'event':
                        opsa_basic_event = OPSA_Basic_Event(e.get('name'),
                                                            probability=None)
                        tree.create_node(e.get('name'), parent=gate,
                                         data=opsa_basic_event)
                    else:
                        raise NotImplementedError('ft_define_gate does not'
                                                  + 'handle {}'.format(e.tag))

            elif subnode.tag == 'label':
                gate_label = subnode.text.strip()
                continue

            else:
                raise NotImplementedError(' Unknown tag <{}>'
                                          .format(subnode.tag))

    def ft_define_basic_event(self, xml_node, tree, parent):
        """
        Parses in a fault tree (ft) the basic event. Stores, if available,
        the probability and the label (description) of the basic event.
        Exponential probability is not jet implemented.
        """

        # search for existing event (should already be created before)
        existing_node = self.tree_find_node(tree, xml_node.get('name'),
                                            OPSA_Basic_Event)
        d = existing_node.data

        for e in xml_node:
            if e.tag == 'float':
                d.set_probability(float(e.get('value')))
            elif e.tag == 'label':
                d.set_label(e.text.strip())
            elif e.tag == 'exponential':
                print('*************************************')
                raise NotImplementedError('Probability <{}> unknown'
                                          .format(e.tag))
            else:
                raise NotImplementedError('ft_define_basic_event not'
                                          'implemented property: <{}> in'
                                          '<{}>'
                                          .format(e.tag, xml_node.get('name')))

    def parse_fault_tree_node(self, xml_node, tree, tree_parent):
        """
        Parses the fault tree by calling functions to parse the data for
        gate, basic_event and label of the fault tree.
        """
        if xml_node.tag == 'define-gate':
            self.ft_define_gate(xml_node, tree, tree_parent)
        elif xml_node.tag == 'define-basic-event':
            self.ft_define_basic_event(xml_node, tree, tree_parent)
        elif xml_node.tag == 'label':
            tree_parent.data.set_label(xml_node.text.strip())
        elif xml_node.tag == 'define-parameter':
            raise NotImplementedError('parse_fault_tree_node(...) does not'
                                      'implement <define-parameter>')
        else:
            raise NotImplementedError('parse_fault_tree_node(...) does not'
                                      'handle {}'.format(xml_node.tag))

    def parse_fault_tree(self, xml_node, tree, tree_parent):
        """
        Creates a new node in the tree for the fault tree. Then
        it parses the fault tree by calling ``parse_fault_tree_node``.
        """

        # create new fault-tree node in tree
        n = tree.create_node(xml_node.get('name'), parent=tree_parent,
                             data=OPSA_Container(xml_node.get('name'),
                                                 EventType.FAULT_TREE))

        # traverse all nodes of the fault tree
        for subnode in xml_node:
            self.parse_fault_tree_node(subnode, tree, n)

    def parse_initiating_event(self, xml_node, tree, tree_parent):
        """
        Creates a new node in the tree with an initiating event
        ``OPSA_Initiating_Event``.
        """

        # create the initiating event
        # create new fault-tree node in tree
        tree.create_node(xml_node.get('name'), parent=tree_parent,
                         data=OPSA_Initiating_Event(xml_node.get('name'),
                                                    EventType.INITIATING_EVENT,
                                                    xml_node.get('event-tree')
                                                    )
                         )

    def parse_containers(self, xml_node, tree, tree_parent):
        """
        Parses the XML and switches dependend on the container type
        to the appropriate parsing function.
        """
        if xml_node.tag == 'define-event-tree':
            self.parse_event_tree(xml_node, tree, tree_parent)
        elif xml_node.tag == 'define-fault-tree':
            self.parse_fault_tree(xml_node, tree, tree_parent)
        elif xml_node.tag == 'model-data':
            self.parse_model_data(xml_node, tree)
        elif xml_node.tag == 'define-initiating-event':
            self.parse_initiating_event(xml_node, tree, tree_parent)
        else:
            raise NotImplementedError('Container type unknown: {}'
                                      .format(xml_node.tag))

    def parse(self, xml_root, tree, tree_root):
        """
        Parse the xml_root and stores the data in tree.
        It iterates all the containers in the XML data.
        """

        self.check_root(xml_root)

        for container in xml_root:
            self.parse_containers(container, tree, tree_root)
        self.postprocess(tree)

    def postprocess(self, tree):
        """
        Correct flaws during loading. They are caused mainly from
        shortcuts in the XML data.

        If gate is not internal node, there must be somewhere a definition
        of that node. So find it and copy it to that gate.
        Continue until there are no more internal gates
        """

        max_iter = 1000
        i = 0
        found_leaf_gates = True

        while i < max_iter and found_leaf_gates:
            found_leaf_gates = self.find_leaf_gates(tree)
            i += 1

        if found_leaf_gates:
            raise RuntimeError('Still too many gates without children found,'
                               'perhaps increase max_iter = {}'
                               .format(max_iter))

    def find_leaf_gates(self, tree):
        """
        Find all the gates that are leafes. All gates should be internal,
        so it has to have basic events as leafes.
        """

        for node in tree.all_nodes_itr():
            if isinstance(node.data, OPSA_Gate):
                if node.is_leaf():
                    print('    Node is leaf {}\tid = {}'
                          .format(node.tag, node.identifier))
                    # find all all_nodes with same tag as node
                    found_nodes = self.tree_find_nodes(tree, node.tag,
                                                       OPSA_Gate)

                    if len(found_nodes) == 0:
                        raise RuntimeError('No internal gate found with {}'
                                           .format(node.tag))

                    for f in found_nodes:
                        if not f.is_leaf():
                            self.paste_gate(tree, node, f)
                            return True
        return False

    def paste_gate(self, tree, dest_node, source_node):
        """
        Paste the source_node data to the dest_node.
        """

        for child in tree.children(source_node.identifier):
            subtree = tree.subtree(child.identifier)
            t = tree_copy(subtree)
            tree.paste(dest_node.identifier, t)
