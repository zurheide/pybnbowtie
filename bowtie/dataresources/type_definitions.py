from enum import Enum, auto


class GateType(Enum):
    """
    Enumeration of available gate types.

    **Note:** not all of this types are already implemented.
    """
    OR = auto()
    AND = auto()
    ATLEAST = auto()
    NOT = auto()
    LINK = auto()
    UNKNOWN = auto()


class EventType(Enum):
    """
    Definition of OPSA data types.
    """
    FAULT_TREE = auto()
    EVENT_TREE = auto()
    GATE = auto()
    BASIC_EVENT = auto()
    INITIATING_EVENT = auto()
    FUNCTIONAL_EVENT = auto()
    SEQUENCE = auto()
    FORK = auto()
    PATH = auto()
    UNKNOWN = auto()
