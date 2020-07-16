'''
OPSA_Containers holds high level structures like event tree or faul tree

Definition of the Open-PSA data-Structures.
OPSA XML file will be imported and a tree structure with
this objects will be created.

Definition of OPSA structure can be found at the internet site of
https://open-psa.github.io

'''


class OPSA_Container():
    '''
    OPSA_Container holds high level structures like event tree or faul tree
    '''

    def __init__(self, name, container_type):
        """
        Set the name and the container_type.
        """
        self.name = name
        self.container_type = container_type
        self.label = None

    def set_label(self, label):
        """
        Sets a label (description) to the container.
        """
        self.label = label


class OPSA_Gate():
    """
    Definition of a gate, for example OR, AND, AT_LEAST, ....
    Available (but not necessary implemented) types of gates are defined
    in the class ``GateType``.
    """
    def __init__(self, name, gate_type, gate_label=None):
        self.name = name
        self.gate_type = gate_type
        self.gate_label = gate_label
        self.atleast_min = -1

    def set_atleast_min(self, minimun):
        """
        Set minimum number of active input for AT_LEAST gate.
        """
        self.atleast_min = minimun


class OPSA_Basic_Event():
    """
    Basic event in a fault tree. A basic event is connected to a gate.
    The basic event has to have a name. If no probability is given the
    creator must take care of the probability (e.g. 1.0).
    """
    def __init__(self, name, probability):
        self.name = name
        self.probability = probability
        self.label = None
        self.attribute = None
        self.parameter = None

    def set_label(self, label):
        """
        Sets a label (description) to the container.
        """
        self.label = label

    def set_probability(self, probability):
        """
        Set the probability of the ``OPSA_Basic_Event``. Probability is
        a float in the range of 0.0 to 1.0
        """
        self.probability = probability

    def add_attribute(self, name, value):
        """
        Add attribute to the ``OPSA_Basic_Event``. The storage is a ``set``
        so it is possible to overwrite existing values. This is not checked.
        """
        if self.attribute is None:
            self.attribute = {}
        self.attribute[name] = value

    def get_attribute(self):
        """
        Returns the ``set`` of attributes.
        """
        return self.attribute

    def set_parameter(self, parameter):
        """
        Sets parameter. Not implemented. Should be used for time dependend
        data.
        """
        self.parameter = parameter


class OPSA_Initiating_Event():
    """
    ``OPSA_Initiating_Event`` is a link from a top event to an
    event tree.
    """
    def __init__(self, name, container_type, event_tree):
        self.name = name
        self.container_type = container_type
        self.event_tree = event_tree


class OPSA_Functional_Event():
    """
    The functional event is a splitting event to two new different paths.
    This element forks the event tree and two ``OPSA_Path`` elements follow
    this functional events.

    Most often this is also called *safety gate* or *safety element*,
    for example a ignition prevention.
    """
    def __init__(self, name, attributes=None, label=None):
        self.name = name
        self.label = label
        self.attributes = attributes

    def set_label(self, label):
        """
        Sets a label (description) to the container.
        """
        self.label = label


class OPSA_Sequence():
    """
    The sequence is the end point of an event tree. In other places it is
    also called *consequence*.
    """
    def __init__(self, name, probability=None, label=None):
        self.name = name
        self.label = label
        self.probability = probability
        self.event_tree_name = None

    def set_label(self, label):
        """
        Sets a label (description) to the container.
        """
        self.label = label

    def set_event_tree(self, name):
        """
        Sets the name of the event tree to the sequence.
        """
        self.event_tree_name = name


class OPSA_Path():
    """
    A path follows a ``OPSA_Functional_Event``. There are always two possible
    paths after a functional event: YES/NO, true/false, works/fails.
    """
    def __init__(self, name, state, label=None, probability=None):
        self.name = name
        self.state = state
        self.label = label
        self.probability = probability
        self.parameter = None

    def set_expression(self, expression):
        """
        ZZZ TODO: Check this

        The expression defines the probability of this path. Both paths
        must give a probability of 1.0. This is not checked and must be
        done by the program logic.
        """
        self.expression = expression

    def set_parameter(self, parameter):
        self.parameter = parameter

    def set_probability(self, probability):
        """
        The expression defines the probability of this path. Both paths
        must give a probability of 1.0. This is not checked and must be
        done by the program logic.
        """
        self.probability = probability
