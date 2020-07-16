"""
The ``mapping_bowtie`` module holds the class ``MappingBowTie``. this
class maps the Bow-Tie that is stored in a tree structure to a
Bayesian network.

For the Bayesian network the pgmpy package is used.

Example::

    import xml.etree.ElementTree as ET
    from treelib import Tree
    from bowtie.io.import_opsa import OPSA_Importer
    from bowtie.mapping.mapping_bowtie import MappingBowTie

    # Note: importing is the same as the example in bowtie.io.import_opsa
    # read XML file, here opsa_input.xml
    xml_root = ET.parse("opsa_input.xml").getroot()
    # create and prepare tree for results
    tree = Tree()
    tree_root = tree.create_node('root')
    # create importer and parse
    importer = OPSA_Importer()
    importer.parse(xml_root, tree, tree_root)

    # now map the Bow-Tie to a Bayesian network
    mapper = MappingBowTie(tree)
    model = mapper.map()

    # show model nodes
    print('nodes')
    print(model.nodes())

    print('check bayesian model = {}'.format(model.check_model()))

"""

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from treelib import Tree

from bowtie.dataresources.type_definitions import GateType, EventType
from bowtie.dataresources.opsa import OPSA_Container, OPSA_Basic_Event, \
    OPSA_Gate, OPSA_Initiating_Event, OPSA_Functional_Event, OPSA_Path, \
    OPSA_Sequence
from bowtie.io.import_opsa import OPSA_Importer


class MappingBowTie:

    '''
    Convert a Bow-Tie into a Bayesian network.

    The conversion is based on the paper:

        Nima Khakzada, Faisal Khana, Paul Amyotte: *Dynamic safety analysis
        of process systems by mapping bow-tie into Bayesian network.*
        Process Safety and Environmental Protection, 2013.

    The function ``map()`` maps the Bow-Tie tree into a Bayesian network.
    The returned Bayesian network is of the package pgmpy.

    Implemented mapping of gates: ``OR``, ``AND``.
    '''

    def __init__(self, tree):
        self.tree = tree
        self.importer = OPSA_Importer()
        self.model = None
        # define some constants
        self.consequence_name = 'Consequence'
        self.safe_sequence = 'Safe_State'

    def find_event_tree(self, tree, top_event):
        """
        Find and return the event tree in the ``tree``.
        """

        # find the event tree
        events = self.importer.tree_find_nodes(tree, top_event.tag,
                                               OPSA_Initiating_Event)
        if len(events) != 1:
            raise RuntimeError('Initiating event for event tree not defined.'
                               + 'No connection from fault tree to event tree')
        event = events[0]

        event_trees = self.importer.tree_find_nodes(tree,
                                                    event.data.event_tree,
                                                    OPSA_Container)
        if len(event_trees) != 1:
            raise RuntimeError('More than one event tree for the Top Event')
        event_tree = event_trees[0]

        return event_tree

    def get_top_event(self, tree):
        """
        Return the top event of the fault tree in the ``tree``.
        """

        # look for first fault tree in the tree structure
        root_node = tree.get_node(tree.root)
        children = tree.children(root_node.identifier)
        for child in children:
            d = child.data
            if isinstance(d, OPSA_Container):
                if d.container_type == EventType.FAULT_TREE:
                    fault_tree = child
                    break
        if not fault_tree:
            raise RuntimeError('No fault tree given, so also no top event')

        # get children of fault tree. First element is the Top Event
        children = tree.children(fault_tree.identifier)
        if len(children) != 1:
            raise RuntimeError('Fault tree must have exactly ONE Top Event.'
                               + 'But has {}'.format(len(children)))

        return children[0]

    def probability_contains_values(self, prob_row, prob_values):
        """
        If all the elements of ``prob_values`` are in ``prob_row``
        return ``True`` otherwise ``False``.
        """

        if len(prob_row) - 1 < len(prob_values):
            list_a = prob_row[:-1]
            list_b = prob_values
        else:
            list_a = prob_values
            list_b = prob_row
        for element in list_a:
            inside = element in list_b
            if not inside:
                return False

        return True

    def find_state_for_probabilities(self, prob_list, prob_values):
        """
        Find the row in the ``prob_list`` that contains the value for
        ``prob_values``.
        """

        # have at least all the values of the prob_values in the prob_list
        for row in prob_list:
            if self.probability_contains_values(row, prob_values):
                return row[len(row) - 1]

        return None

    def find_tag_in_tree(self, tree, tag):
        """
        Find the nodes in the ``tree`` that have the tag ``tag``. Returns
        a list with the nodes.
        """
        res = []
        for node in tree.all_nodes_itr():
            if node.tag == tag:
                res.append(node)
        return res

    def states_in_probability(self, prob_list):
        """
        Return a dictionary with the states in the ``prob_list``.
        """
        tmp_dict = {}

        for row in prob_list:
            for element in row:
                if isinstance(element, tuple):
                    # is element[0] in res_dict, add element[1] in list
                    # not in res_dict, add list
                    if element[0] in tmp_dict:
                        tmp_dict[element[0]].append(element[1])
                    else:
                        tmp_dict[element[0]] = [element[1]]

        res = {}
        for k, v in tmp_dict.items():
            res[k] = list(set(v))

        return res

    def split_probability_table(self, name, prob_list):
        """
        Split the ``prob_list`` in two lists ``name_list`` and
        ``sans_name_list``. ``name_list``contains prob_list elements that
        have the name equal to ``name``, ``sans_name_list`` contains all
        the other elements. Returns ``name_list``, ``sans_name_list``.
        """

        name_list = []
        sans_name_list = []

        for row in prob_list:
            new_row = []
            new_row_name = []
            for element in row:
                if isinstance(element, tuple):
                    if element[0] == name:
                        new_row_name.append(element)
                    else:
                        new_row.append(element)
            sans_name_list.append(new_row)
            name_list.append(new_row_name)

        return name_list, sans_name_list

    def get_state_for_number(self, states, bit):
        """
        Return the state stored in ``states`` for the bit.
        Until now only 2 bits are allowed, i.e. two states.
        Values of ``states`` are compared if they belong to the given
        set of allowed states.

        Allowed states::

            first = ['works', 'true', 'yes']
            second = ['fails', 'false', 'no']

        """

        # some quick checks for the size of the bit
        if not (bit >= 0 or bit < 2):
            raise RuntimeError('number of states wrong {}'.format(bit))

        first = ['works', 'true', 'yes']
        second = ['fails', 'false', 'no']

        if bit == 0:
            for f in first:
                if f in states:
                    return f
        if bit == 1:
            for s in second:
                if s in states:
                    return s
        return None

    def derived_values_from_probabilities(self, prob_list, seq, variable_name):
        '''
        Returns the values, evidence, evidence_card.
        '''

        # get values for the states
        all_states = self.states_in_probability(prob_list)
        all_states_list = list(all_states)

        len_pos = (len(all_states))
        num_pos = 2**len_pos

        values = {}
        for s in seq:
            values[s] = [None] * num_pos

        for i in range(num_pos):
            bin_str = list(format(i, '08b'))
            bin_str.reverse()

            prob_values = []
            for n in range(len_pos):
                bit = int(bin_str[n])
                event = all_states_list[n]
                state = self.get_state_for_number(all_states[event], bit)

                # state = all_states[event][bit]
                prob_values.append((event, state))

            state = self.find_state_for_probabilities(prob_list, prob_values)

            for s in seq:
                if s == state:
                    x = 1.0
                    # x = 0.0
                else:
                    x = 0.0
                    # x = 1.0
                values[s][i] = x
        evidence = all_states_list.copy()
        evidence.reverse()

        # evidence card
        evidence_card = []
        for e in evidence:
            evidence_card.append(len(all_states[e]))

        state_names = {}
        state_names[variable_name] = list(seq)
        for ev in evidence:
            tmp = []
            for i in range(len(all_states[ev])):
                s = self.get_state_for_number(all_states[ev], i)
                tmp.append(s)

            # state_names[ev] = all_states[ev].copy()
            state_names[ev] = tmp

        value_list = []
        for s in seq:
            value_list.append(values[s])

        return value_list, evidence, evidence_card, state_names

    def consequence_probability_add_state(self, prob_list, event, state):
        """
        Add the state to the probability list ``prob_list``.
        """

        # insert the event with state
        prob_copy = []
        for row in prob_list:
            cons = row[len(row) - 1]
            if not isinstance(cons, str):
                raise RuntimeError("Last element of row ({}) should be"
                                   "sequence name type=str"
                                   .format(row))

            new_row = row.copy()
            t1 = (event, 'fails')
            new_row.insert(len(row) - 1, t1)
            prob_copy.append(new_row)

            new_row = row.copy()
            t2 = (event, 'works')
            new_row.insert(len(row) - 1, t2)
            new_row[len(new_row) - 1] = state
            prob_copy.append(new_row)

        return prob_copy

    def create_cdp_consequence(self, name, probs, top_event):
        '''
        Create a ``TabularCPD()`` for the consequences.
        '''

        new_probs = self.consequence_probability_add_state(probs,
                                                           top_event.tag,
                                                           'Safe')

        # get number of consequences
        s = set()
        for row in new_probs:
            seq_name = row[len(row) - 1]
            if not isinstance(seq_name, str):
                raise RuntimeError('Last element of row ({}) should be'
                                   'equence name'.format(row))
            s.add(seq_name)

        variable = name
        variable_card = len(s)
        values, evidence, evidence_card, state_names = \
            self.derived_values_from_probabilities(new_probs, s, name)

        cpd = self.fill_cpd_table(variable, variable_card, values,
                                  evidence, evidence_card, state_names)
        return cpd

    def sorted_probability_dict(self, prob_dict):
        """
        Return the ``values``, ``states`` from the prob_dict.
        """

        if len(prob_dict) != 2:
            raise RuntimeError('Only two states allowed, given: {}'
                               .format(list(prob_dict))
                               )

        first = ['works', 'true', 'yes']
        second = ['fails', 'false', 'no']

        values = []
        states = []
        prob_dict_list = list(prob_dict)

        found_first = False
        # find first value
        for f in first:
            if not found_first:
                if f in prob_dict_list:
                    values.append([prob_dict[f]])
                    states.append(f)

                    found_first = True

        found_second = False
        # find second value
        for f in second:
            if not found_second:
                if f in prob_dict_list:
                    values.append([prob_dict[f]])
                    states.append(f)

                    found_second = True

        if not (found_first and found_second):
            raise RuntimeError('Could not find first or second value')

        return values, states

    def create_cpd_functional_event_empty_probs(self, tree, name):
        """
        Return CPD table for node with name ``name``.
        The probability list is empty, because all the probabilities of
        node ``name`` have the same value.
        """

        # get the probabilities of the node "name"
        node_list = self.find_tag_in_tree(tree, name)
        # this probability list is empty, because all the probabilities of
        # node "name" have the same value. So we can just take the first one
        node = node_list[0]

        prob_dict = self.fe_prob_to_dict(tree, node)

        variable = name
        variable_card = len(prob_dict.keys())

        values, states = self.sorted_probability_dict(prob_dict)
        state_names = {name: states}

        # CPDs create
        cpd = TabularCPD(variable=variable,
                         variable_card=variable_card,
                         values=values,
                         state_names=state_names
                         )
        return cpd

    def create_cpd_functional_event_list(self, name, prob_list):
        """
        Returns CPD table for the prob_list.
        """

        # get values for the states
        all_states = self.states_in_probability(prob_list)
        all_states_list = list(all_states)

        if not (name in all_states_list):
            raise RuntimeError('Event ({}) cannot be found in the'
                               'probability list ({})'
                               .format(name, prob_list))

        all_states_wo_name = all_states.copy()
        all_states_wo_name.pop(name, None)

        all_states_list_wo_name = all_states_list.copy()
        all_states_list_wo_name.remove(name)

        _ = self.split_probability_table(name, prob_list)

        # here we have 2^len_pos possibilities (works/fails)
        len_pos = (len(all_states))
        num_pos = 2**len_pos

        if len_pos == len(prob_list):
            print(len_pos, len(prob_list))
            raise RuntimeError("Length missmatch of probability list and"
                               "computed possiblities")

        value_list = []
        # now cycle over all possibilities and numbers
        for i in range(num_pos):
            bin_str = list(format(i, '08b'))
            bin_str.reverse()

            prob_values = []
            for n in range(len_pos):
                bit = int(bin_str[n])
                if n == len_pos - 1:
                    event = name
                    state = all_states[event][bit]
                else:
                    event = all_states_list_wo_name[n - 1]
                    state = all_states[event][bit]

                prob_values.append((event, state))

            prob = self.find_state_for_probabilities(prob_list, prob_values)
            value_list.append(prob)

        len_name_states = len(all_states[name])
        num_events = int(num_pos / len_name_states)

        values = []
        n = 0
        for j in range(num_events):
            # fill for each event the list with posiblities
            tmp = [None] * len_name_states
            for i in range(len_name_states):
                tmp[i] = value_list[n]
                n += 1

            values.append(tmp)

        # cpd_s_sn = TabularCPD(variable='S', variable_card=2,
        #                      values=[[0.95, 0.2],
        #                              [0.05, 0.8]],
        #                      evidence=['I'],
        #                      evidence_card=[2],
        #                      state_names={'S': ['Bad', 'Good'],
        #                                   'I': ['Dumb', 'Intelligent']})
        #
        # +---------+---------+----------------+
        # | I       | I(Dumb) | I(Intelligent) |
        # +---------+---------+----------------+
        # | S(Bad)  | 0.95    | 0.2            |
        # +---------+---------+----------------+
        # | S(Good) | 0.05    | 0.8            |
        # +---------+---------+----------------+

        variable = name
        variable_card = len(all_states[name])

        evidence = all_states_list_wo_name
        # evidence card
        evidence_card = []
        for e in evidence:
            evidence_card.append(len(all_states[e]))
        state_names = all_states

        cpd = self.fill_cpd_table(variable, variable_card, values, evidence,
                                  evidence_card, state_names)

        return cpd

    def create_cdp_functional_event(self, tree, name, probs):
        """
        Create CPD for ``name`` with probability ``probs``.
        """

        cpd = None
        if len(probs) == 0:
            cpd = self.create_cpd_functional_event_empty_probs(tree, name)
        else:
            cpd = self.create_cpd_functional_event_list(name, probs)

        return cpd

    def get_functional_event_tuple(self, tree, path, sequence_name):
        """
        Create a functional event tuple for the nodes in the path that
        are of type OPSA_Functional_Event.
        """

        aList = []
        for ind, n in enumerate(path):
            node = tree.get_node(n)
            if isinstance(node.data, OPSA_Functional_Event):
                # get next element
                if ind + 1 > len(path):
                    for p in path:
                        print('node = {}'.format(tree.get_node(p).tag))
                    raise RuntimeError('Cannot access the next element of ({})'
                                       .format(node.tag))

                next_node = tree.get_node(path[ind + 1])

                if not isinstance(next_node.data, OPSA_Path):
                    raise RuntimeError('Functional event ({}) should be'
                                       'followed by a path'.format(node.tag))

                aList.append((node.tag, next_node.data.state))

        aList.append(sequence_name)
        res = ([x for x in aList])

        return res

    def check_causal_arc_elimination(self, elimination_dict, b_node):
        """
        Returns the value of the elimination_dict for b_node.
        """
        return elimination_dict[b_node]

    def get_next_element_of_path(self, tree, path, start_element,
                                 node_type=OPSA_Gate):
        """
        Returns the next element if the path beginning at start_element
        that is of type node_type.
        """

        start_index = self.get_list_index_of_element(path, start_element)

        if len(path) < start_index + 1:
            return None

        node = tree.get_node(path[start_index + 1])

        if isinstance(node.data, node_type):
            return node
        else:
            return None

    def fe_prob_to_dict(self, tree, node):
        """
        Returns probability of the node children.
        """
        children = tree.children(node.identifier)
        res = {}
        for child in children:
            res[child.data.state] = child.data.probability

        return res

    def perform_event_tree_checks(self, tree):
        """
        Perform some checks in the tree containing the event tree.
        Checks if consequence_name and safe_sequence are included.
        """

        fe = self.importer.tree_find_nodes(tree, self.consequence_name,
                                           OPSA_Functional_Event)
        if len(fe) > 0:
            raise RuntimeError('There has to be no functional event'
                               + 'named <{}>!'.format(self.consequence_name))

        seq = self.importer.tree_find_nodes(tree, self.safe_sequence,
                                            OPSA_Sequence)
        if len(seq) > 0:
            raise RuntimeError('There has to be no sequence named <{}>!'
                               .format(self.safe_sequence))

    def get_event_tree_subtree(self, tree, event_tree):
        """
        Returns a subtree and the root of that subtree that includes only
        the event_tree. The subtree is
        taken from the tree and starts at the event_tree.
        """

        subtree = tree.subtree(event_tree.identifier)
        subtree_root = subtree.get_node(subtree.root)

        return subtree, subtree_root

    def get_list_index_of_element(self, aList, element):
        """
        Checks if element is in aList. Returns None if not included or
        if included it returns the index of element in aList, so that
        element = aList[index].
        """

        cnt = aList.count(element)
        if cnt == 0:
            # no element in aList
            return None
        return aList.index(element)

    def check_path(self, complete_path, test_path):
        """
        Checks if all elements of ``test_path`` are stored in
        ``complete_path``. Returns True, otherwise raises an exception.
        """

        last_index = 0
        for element in test_path:
            index_complete = self.get_list_index_of_element(complete_path,
                                                            element)
            if index_complete is None:
                raise RuntimeError('Path incomplete: The element ({}) is'
                                   'not in the complete path. There should'
                                   'be a path from top event to consequence'
                                   'that contains all functional events.'
                                   'Please check this'
                                   .format(element))
            if last_index > index_complete:
                raise RuntimeError('Ordering wrong: The element ({})'
                                   'appears before ({}). Differs from other'
                                   'path definition.'
                                   .format(complete_path[last_index], element))
            last_index = index_complete

        return True

    def get_all_events_in_row(self, paths_list):
        """
        Make a guess and just take the path with the most elements for
        the complete path. Check if this is true (should be).

        Not sure if this works properly for all cases - perhaps users
        will abuse the event trees.
        """

        # start with the longest path
        complete_path = []
        for path in paths_list:
            if len(path) > len(complete_path):
                complete_path = path
        for path in paths_list:
            if not self.check_path(complete_path, path):
                raise RuntimeError('Path is not correct: {}'.format(path))

        return complete_path

    def get_functional_event_ordering(self, tree):
        """
        Returns all the functional events in all available paths.
        """

        # get all the paths for the functional_events
        all_paths = []
        for p in tree.paths_to_leaves():
            a_list = []
            for n in p:
                node = tree.get_node(n)
                if isinstance(node.data, OPSA_Functional_Event):
                    a_list.append(node.tag)
            all_paths.append(a_list)

        return self.get_all_events_in_row(all_paths)

    def get_probs_equal(self, tree, node_list):
        """
        Check if all the probabilities of the nodes in node_list are the same.
        If probabilities of nodes differ return value is False,
        otherwise True.
        """

        if len(node_list) == 0:
            raise RuntimeError("node_list is empty")

        first_prob = self.fe_prob_to_dict(tree, node_list[0])
        for node in node_list:
            sub_probs = self.fe_prob_to_dict(tree, node)
            if first_prob != sub_probs:
                return False

        return True

    def get_causal_arc_elemination(self, tree, fe_ordered):
        """
        Return dictionary with arc elimination.
        """

        # False: no arc elimination needed, probabilities are differeent
        # True: arc elimination needed (probabilities not different)
        result_dict = {}

        for element in fe_ordered:
            result_dict[element] = True
            events = self.importer.tree_find_nodes(tree, element,
                                                   OPSA_Functional_Event)
            prob_equal = self.get_probs_equal(tree, events)
            result_dict[element] = prob_equal

        return result_dict

    def create_cpd_tables_event_tree(self, model, subtree, result_dict,
                                     top_event):
        """
        Create and add CPD tables from result_dict.
        """

        for key, value in result_dict.items():
            if key == self.consequence_name:
                cpd = self.create_cdp_consequence(key, value,
                                                  top_event)
            else:
                cpd = self.create_cdp_functional_event(subtree, key, value)
            model.add_cpds(cpd)

    def map_path_previous(self, tree, elimination_dict):

        '''
        Before we can map the event tree to the dictionary,
        the number of states has to be checked, because otherwise
        downstream in the code we will have some problems with the CPD tables
        creation.
        So dummy-run the map_path without writing the data to the result_dict
        and do not store data if there are less then one state in map_path
        '''

        tmp_dict = {}
        for path in tree.paths_to_leaves():
            # collect the nodes of the path
            node_list = []
            for n in path:
                node = tree.get_node(n)
                node_list.append(node)

            for ind, node in enumerate(node_list):
                if not isinstance(node.data, OPSA_Functional_Event):
                    continue

                causal = self.check_causal_arc_elimination(elimination_dict,
                                                           node.tag)
                if causal:
                    continue

                path_element = self.get_next_element_of_path(tree, path,
                                                             node.identifier,
                                                             OPSA_Path)

                tuple_list = [(node.tag, path_element.data.state)]

                for prev_node_index in reversed(range(0, ind)):
                    prev_node = node_list[prev_node_index]

                    if not isinstance(prev_node.data, OPSA_Functional_Event):
                        continue
                    prev_path_element = \
                        self.get_next_element_of_path(tree,
                                                      path,
                                                      prev_node.identifier,
                                                      OPSA_Path)
                    tuple_list.append((prev_node.tag,
                                       prev_path_element.data.state))

                tuple_list.append((path_element.data.probability))
                t = tuple([(x) for x in tuple_list])

                # does key exist in tmp_dict? otherwise create a list
                if node.tag not in tmp_dict:
                    tmp_dict[node.tag] = []
                tmp_dict[node.tag].append(t)

        # get number of states and only use the ones with exactly two states
        possible_states = {}
        for key, value in tmp_dict.items():
            # get values for the states
            all_states = self.states_in_probability(value)
            all_states_list = list(all_states)

            possible_list = []

            for name in all_states_list:
                num_states = len(all_states[name])
                if num_states < 2:
                    continue
                if num_states > 2:
                    raise RuntimeError('No more than two (2) states allowed.'
                                       + 'Sorry. Functional Event ({})'
                                       .format(name))

                possible_list.append(name)

            if not (key in possible_states):
                possible_states[key] = []
            possible_states[key] = possible_list

        return possible_states

    def map_path(self, model, tree, result_dict, elimination_dict, top_event,
                 possible_states):
        '''
        - loop over all paths from top event to all sequences
            - create node_list with all the nodes of the path
            - loop over all elements of the path (node)
                - if the node is not a functional event, continue
                - loop over all previous elements of the path (prev_node)
                    - if prev_node is not a functional event, continue
                    - add edge prev_node -> node
                    - get next element in path and add to tupple_list
        - connect top event with consequences
        '''

        # loop over all functional elements in path
        for path in tree.paths_to_leaves():
            # collect the nodes of the path
            node_list = []
            for n in path:
                node = tree.get_node(n)
                node_list.append(node)

            for ind, node in enumerate(node_list):
                if not isinstance(node.data, OPSA_Functional_Event):
                    continue

                # connect the node with the consequence node
                model.add_edge(node.tag, self.consequence_name)

                # if causal elimination, we do not need a connection from any
                #           prev_node to actual node
                causal = self.check_causal_arc_elimination(elimination_dict,
                                                           node.tag)
                if causal:
                    continue

                path_element = self.get_next_element_of_path(tree, path,
                                                             node.identifier,
                                                             OPSA_Path)
                tuple_list = [(node.tag, path_element.data.state)]

                for prev_node_index in reversed(range(0, ind)):
                    prev_node = node_list[prev_node_index]

                    if not isinstance(prev_node.data, OPSA_Functional_Event):
                        continue

                    # is previous node a possibility? we checked this before!
                    if not (prev_node.tag in possible_states[node.tag]):
                        continue

                    model.add_edge(prev_node.tag, node.tag)

                    prev_path_element = \
                        self.get_next_element_of_path(tree, path,
                                                      prev_node.identifier,
                                                      OPSA_Path)
                    tuple_list.append((prev_node.tag,
                                       prev_path_element.data.state))

                tuple_list.append((path_element.data.probability))
                t = tuple([(x) for x in tuple_list])
                result_dict[node.tag].append(t)

            # finally, store the functional events and the states that lead
            #  to this sequence
            seq_node = tree.get_node(path[-1])
            t = self.get_functional_event_tuple(tree, path, seq_node.tag)
            result_dict[self.consequence_name].append(t)

        # connect top event with the consequences
        model.add_edge(top_event.tag, self.consequence_name)

    def map_functional_event_data(self, tree):
        """
        Returns a dictionary with the functional events and a dictionary
        with the causal arc elimination.
        """

        # result_dict stores the data for later processing to get the CPD/CPT
        result_dict = {}

        functional_events_ordered = self.get_functional_event_ordering(tree)
        elimination_dict = \
            self.get_causal_arc_elemination(tree, functional_events_ordered)

        for fe in functional_events_ordered:
            result_dict[fe] = []
        result_dict[self.consequence_name] = []
        return result_dict, elimination_dict

    def map_functional_events(self, model, tree, te, event_tree):
        """
        Maps the functional events of the event tree.
        """

        self.perform_event_tree_checks(tree)
        subtree, _ = self.get_event_tree_subtree(tree, event_tree)

        result_dict, elimination_dict = self.map_functional_event_data(subtree)
        possible_states = self.map_path_previous(subtree, elimination_dict)
        self.map_path(model, subtree, result_dict, elimination_dict, te,
                      possible_states)

        self.create_cpd_tables_event_tree(model, subtree, result_dict, te)

    def get_probability_values(self, gate_type, num):
        """
        Set the probability values for the gate types.

        Handled gate types are ``OR``and ``AND``.
        """

        if not (num >= 1 and num < 8):
            raise RuntimeError("number of functional events too high ({})"
                               .format(num))

        num_values = 2**num

        values_1 = []
        values_2 = []
        if gate_type == GateType.OR:

            for i in range(num_values):
                if i == 0:
                    values_1.append(1.0)
                    values_2.append(0.0)
                else:
                    values_1.append(0.0)
                    values_2.append(1.0)
            values = [values_1, values_2]
            return values
        elif gate_type == GateType.AND:
            for i in range(num_values):
                if i != num_values - 1:
                    values_1.append(1.0)
                    values_2.append(0.0)
                else:
                    values_1.append(0.0)
                    values_2.append(1.0)
            values = [values_1, values_2]
            return values
        else:
            raise NotImplementedError('gate_type not handled: {}'
                                      .format(gate_type))

        return None

    def get_evidence_card(self, model, children):
        """
        Returns the size of the variable_card. A little bit tricky.
        Explaination can be found in the source code.
        """

        evidence = []
        for child in children:
            cpd = model.get_cpds(child)
            if cpd is None:
                # OK, some explaination. While we fill the PGM we do not have
                # all the values already filled in. This means, that we do
                # not have the number of the variable_card. The algorithm
                # should be changed, so that we traverse from the leaf to
                # the root of the tree. Well should....
                # for now we just make an educated guess that
                # evidence_card == 2
                # TODO: fix this
                variable_card = 2
            else:
                variable_card = cpd.variable_card

            evidence.append(variable_card)
        return evidence

    def get_state_names(self, variable, children, states=['works', 'fails']):
        '''
        Create something like that::

          state_names={'AND': ['works', 'fails'],
                       'A': ['works', 'fails'],
                       'B': ['works', 'fails'],
                       'C': ['works', 'fails'],
                       'D': ['works', 'fails']})
        '''

        state_names = {}
        state_names[variable] = states
        for child in children:
            state_names[child] = states

        return state_names

    def fill_cpd_table(self, variable, variable_card, values, evidence,
                       evidence_card, state_names=None):
        """
        Function to shorten the source code (wrapper). Creates a
        ``TabularCPD()`` from the given values.
        """

        cpd = TabularCPD(variable=variable, variable_card=variable_card,
                         values=values,
                         evidence=evidence, evidence_card=evidence_card,
                         state_names=state_names)
        return cpd

    def create_cpd_gate(self, model, gate_node, child_nodes):
        """
        Create cpd table ``TabularCPD()`` for the gate.
        """

        num_children = len(child_nodes)
        children = [c.tag for c in child_nodes]
        variable = gate_node.tag
        # TODO: if the variables should ever be changed from 'fails' and
        #   'works' this has to be adapted here
        variable_card = 2

        values = self.get_probability_values(gate_node.data.gate_type,
                                             num_children)
        evidence = children
        evidence_card = self.get_evidence_card(model, children)
        state_names = self.get_state_names(variable, children)

        cpd = self.fill_cpd_table(variable, variable_card, values, evidence,
                                  evidence_card, state_names)
        model.add_cpds(cpd)

    def map_connections(self, model, tree):
        '''
        Maps the connections between the gates. Also creates
        the CPD tables for the connections
        '''
        nodes = self.importer.tree_find_nodes(tree, None, OPSA_Gate)

        for node in nodes:
            children = tree.children(node.identifier)
            for child in children:
                model.add_edge(child.tag, node.tag)
            # create CPD for all children
            self.create_cpd_gate(model, node, children)

    def map_gates(self, model, tree):
        '''
        Intermediate events (Gates) AND the Top Event are mapped
        from a tree structure to a Bayesian network (BN)
        '''
        nodes = self.importer.tree_find_nodes(tree, None, OPSA_Gate)
        for node in nodes:
            model.add_node(node.tag)

        self.map_connections(model, tree)

    def cpd_exists(self, model, name):
        """
        If a cpd with the variable name exist the function returns True
        otherwise False.
        """
        # find cpd for consequence
        cpds = model.cpds
        for cpd in cpds:
            if cpd.variable == name:
                return True
        return False

    def create_cpd_basic_event(self, model, node):
        """
        Creates the cpd ``TabularCPD()`` for the basic event.
        """
        # is there already a CPD for this nodes?
        if self.cpd_exists(model, node.tag):
            print('Warning: CPD already exists: {}'.format(node.tag))
            return

        # get probability
        probability = node.data.probability
        if probability is None:
            probability = 0.0
        else:
            probability = float(probability)

        if probability < 0.0 or probability > 1.0:
            print('WARNING: probability not set, so I chose 0.0')
            probability = 0.0

        # working: C = 0; failing: C = 1
        cpd = TabularCPD(variable=node.tag, variable_card=2,
                         values=[[1.0 - probability], [probability]],
                         state_names={node.tag: ['works', 'fails']}
                         )

        model.add_cpds(cpd)

    def map_basic_events(self, model, tree):
        """
        Maps the basic event to the Bayesian model. Create nodes and
        CPD tables.
        """
        # importer = OPSA_Importer()
        nodes = self.importer.tree_find_nodes(tree, None, OPSA_Basic_Event)

        for node in nodes:
            model.add_node(node.tag)
            self.create_cpd_basic_event(model, node)

    def map(self, tree=None):
        """
        Maps the Bow-Tie tree to a Bayesian network.
        If the tree is not given it must be set during class creation.

        The BayesianModel is returned.
        """

        print('mapping from Bow-Tie to Bayesian network')

        if tree:
            self.tree = tree

        model = BayesianModel()

        model = self.map_FT(self.tree, model)
        model = self.map_ET(self.tree, model)

        return model

    def map_FT(self, tree=None, model=None):
        """
        Maps only the *fault* tree to the Bayesian network.
        If parameter for ``tree`` is not given the class variable will be used.
        If parameter for model is not given, a new ``BayesianModel()``
        network will be created.
        Returns the model.
        """
        if tree is not None:
            self.tree = tree

        if not isinstance(self.tree, Tree):
            raise RuntimeError("tree is not a Tree: {}".format(tree))

        if model is None:
            model = BayesianModel()

        self.map_basic_events(model, self.tree)
        self.map_gates(model, self.tree)

        return model

    def map_ET(self, tree=None, model=None):
        """
        Maps only the *event* tree to the Bayesian network.
        If parameter for ``tree`` is not given the class variable will be used.
        If parameter for model is not given, a new ``BayesianModel()``
        network will be created.
        Returns the model.
        """
        if tree:
            self.tree = tree

        if not isinstance(self.tree, Tree):
            raise RuntimeError("tree is not a Tree: {}".format(tree))

        if model is None:
            model = BayesianModel()

        te = self.get_top_event(self.tree)

        event_tree = self.find_event_tree(self.tree, te)
        self.map_functional_events(model, self.tree, te, event_tree)

        return model
