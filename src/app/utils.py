from typing import Iterator, Callable

from anytree import Node, RenderTree


def _set_parent(node_dict: dict[str, Node],
                parent_name: str,
                child_name: str,
                keep_existing: bool = True) -> None:
    child = node_dict[child_name]
    parent = node_dict[parent_name]
    if (child.parent and child.parent is not parent):
        print('PARENTS DO NOT MATCH')
        print(f'The parent of {child_name} should be {parent_name}')
        if keep_existing:
            print(f"Keeping existing parent ({child.parent.name})")
            return
        print(f"Moving to new parent ({parent_name})")
    elif not child.parent:
        child.parent = parent


def build_nodes(data: dict[str, list[str]]) -> dict[str, Node]:
    """Build nodes and relationships from a raw dictionary.

    The dictionary is assumed to be in the following format:
     - its keys are the children
     - its values are the parents in chronological order
        (this means that the closest parent is at the end of the list)

    Args:
        data (dict[str, list[str]]): The input child-parents dict.

    Returns:
        dict[str, Node]: The built nodes with relationships.
    """
    # Create a dictionary to store nodes for quick reference
    node_dict: dict[str, Node] = {}

    # add all children
    for child in data:
        node_dict[child] = Node(child)

    # add all parents
    all_parents = (parent for parents in data.values() for parent in parents)
    for parent in all_parents:
        if parent in node_dict:
            continue
        node_dict[parent] = Node(parent)

    # add all direct relationships
    # to closest parent
    for child, parents in data.items():
        if parents:
            _set_parent(node_dict, parents[-1], child)

    # add relationships
    for child_name, parent_names in data.items():
        child = node_dict[child_name]
        for parent2, parent1 in zip(parent_names[:-1], parent_names[1:]):
            # parent2 is the parent of parent1
            _set_parent(node_dict, parent2, parent1)

    return node_dict


def find_roots(nodes: Iterator[Node]) -> list[Node]:
    """Find root nodes from the nodes.

    Args:
        nodes (Iterator[Node]): The nodes.

    Returns:
        list[Node]: The root nodes.
    """
    roots: set[Node] = set()

    for node in nodes:
        cur_node = node
        if not cur_node.parent:
            roots.add(cur_node)
    return list(roots)


def get_tree(root_node: Node) -> str:
    """Gets the string representation of the tree from the specified node.

    Args:
        root_node (Node): The root node.

    Returns:
        str: The string representation of the tree.
    """
    tree = ''
    for pre, _, node in RenderTree(root_node):
        tree += f"{pre}{node.name}\n"
    return tree


def get_all_trees(nodes: Iterator[Node],
                  model_link_func: Callable[[str], str]) -> list[str]:
    """Get all tree representations with links to corresponding models.

    Args:
        nodes (Iterator[Node]): All nodes.
        model_link_func (Callable[[str], str]):
            Function to generate the link from model version.

    Returns:
        list[str]: All trees with links to corresponding models.
    """
    trees = []
    for root in find_roots(nodes):
        tree_repr = ""
        for pre, _, node in RenderTree(root):
            # Get the link to the corresponding model
            model_link = model_link_func(node.name)
            full_link = f"<a href='{model_link}'>{model_link}</a>"
            tree_repr += f"{pre}{node.name} (Model link: {full_link})\n"
        trees.append(tree_repr)
    return trees
