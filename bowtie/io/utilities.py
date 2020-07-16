"""
Helper functions for functionality that is needed across the
code but finds no better place.

Includes mainly a function for a deep copy of a tree.
"""


from treelib import Tree


# Helper utilities

def root_traverse(tree, root, new_tree, new_root):
    """
    Traverse the tree starting at the root and create for each node
    a new node that links to the same data.

    Recursive function. Hopefully our trees are small.
    """
    # traverse children
    children = tree.children(root.identifier)
    for child in children:
        # copy node
        n = new_tree.create_node(tag=child.tag,
                                 parent=new_root,
                                 data=child.data)
        # traverse child node
        root_traverse(tree, child, new_tree, n)


def tree_copy(tree):
    """
    Function to copy a tree. Takes a tree from the ``treeLib`` and traverses
    it. For every existing node is a new node created. Only this ensures
    that the copy of the tree contains new nodes with new labels.

    Returns a tree with the same data but new nodes (with new labels).
    """

    root = tree.get_node(tree.root)

    # create new tree
    new_tree = Tree()
    # copy root node
    new_tree.create_node(tag=root.tag, data=root.data)
    new_root = new_tree.get_node(new_tree.root)

    # traverse the data, starting from the roots
    root_traverse(tree, root, new_tree, new_root)

    return new_tree
