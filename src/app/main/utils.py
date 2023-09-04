from typing import Iterator, Callable
import os
import sys
from functools import wraps
import time

from anytree import Node, RenderTree

import logging
from logging.handlers import TimedRotatingFileHandler


from .envs import LOG_PATH
from .envs import LOG_BACKUP_DAYS
from .envs import LOG_LEVEL

logger = logging.getLogger(__name__)


def _set_parent(node_dict: dict[str, Node],
                parent_name: str,
                child_name: str,
                keep_existing: bool = True) -> None:
    child = node_dict[child_name]
    parent = node_dict[parent_name]
    if (child.parent and child.parent is not parent):
        logger.info('PARENTS DO NOT MATCH')
        logger.info('The parent of %s should be %s', child_name, parent_name)
        if keep_existing:
            logger.info("Keeping existing parent (%s)", child.parent.name)
            return
        logger.info("Moving to new parent (%s)", parent_name)
    elif not child.parent:
        child.parent = parent


def build_nodes(data: dict[str, tuple[list[str], str]]) -> dict[str, Node]:
    """Build nodes and relationships from a raw dictionary.

    The dictionary is assumed to be in the following format:
     - its keys are the children
     - its values contains of:
        - List of parents in chronological order
           (this means that the closest parent is at the end of the list)
        - The category name

    Args:
        data (dict[str, tuple[list[str], str]]): The input.
            This maps a version to its parents and the category.

    Returns:
        dict[str, Node]: The built nodes with relationships.
    """
    # Create a dictionary to store nodes for quick reference
    node_dict: dict[str, Node] = {}

    # add all children
    for child in data:
        node_dict[child] = Node(child, category=data[child][1])

    # add all parents
    all_parents = ((parent, category) for (parents, category) in data.values()
                   for parent in parents)
    for (parent, category) in all_parents:
        if parent in node_dict:
            continue
        node_dict[parent] = Node(parent, category=category)

    # add all direct relationships
    # to closest parent
    for child, (parents, _) in data.items():
        if parents:
            _set_parent(node_dict, parents[-1], child)

    # add relationships
    for child_name, (parent_names, _) in data.items():
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
                  model_link_func: Callable[[str], str],
                  model_descr_func: Callable[[str], str],
                  ) -> list[tuple[list[tuple[str, str, str, str]], str]]:
    """Get all tree representations with links to corresponding models.

    Args:
        nodes (Iterator[Node]): All nodes.
        model_link_func (Callable[[str], str]):
            Function to generate the link from model version.
        model_descr_func (Callable[[str], str]):
            Function to get the model description from model version.

    Returns:
        list[tuple[list[tuple[str, str, str, str]], str]]: All trees with
            links to corresponding models alongside their respective
            categories.
    """
    # list of tree lines, and category
    # each line: prefix, name, link
    trees: list[tuple[list[tuple[str, str, str, str]], str]] = []
    for root in find_roots(nodes):
        tree_repr = []
        for pre, _, node in RenderTree(root):
            # Get the link to the corresponding model
            model_link = model_link_func(node.name)
            model_descr = model_descr_func(node.name)
            tree_repr.append((pre, model_descr, node.name, model_link))
        trees.append((tree_repr, root.category))
    return trees


def setup_logging(logger: logging.Logger) -> None:
    logger.setLevel(LOG_LEVEL)

    # Define the log file and log format
    log_file = os.path.join(os.path.dirname(__file__),
                            os.path.join(LOG_PATH, "medcatmlflow.log"))
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add a rotating file handler, which creates a new log file every day
    file_handler = TimedRotatingFileHandler(log_file, when="midnight",
                                            backupCount=LOG_BACKUP_DAYS)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Add a stream handler to log to stdout (Docker container's console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)


class ExpiringCache:

    def __init__(self, expiration_seconds):
        self.cache = {}
        self.expiration_seconds = expiration_seconds

    def get(self, key):
        value, timestamp = self.cache.get(key, (None, None))
        if timestamp is None:
            return None
        if time.time() - timestamp > self.expiration_seconds:
            return None
        return value

    def set(self, key, value):
        self.cache[key] = (value, time.time())

    def invalidate(self, key):
        if key in self.cache:
            del self.cache[key]


def expire_cache_after(seconds):
    def decorator(func):
        cache = ExpiringCache(seconds)

        @wraps(func)
        def wrapper(*args, **kwargs):
            cached_value = cache.get(args)
            if cached_value is not None:
                return cached_value
            result = func(*args, **kwargs)
            cache.set(args, result)
            return result

        return wrapper
    return decorator


class NoSuchModelExcepton(ValueError):

    def __init__(self, key: str, value: str) -> None:
        super().__init__(f"Could not find a model  where '{key}' = '{value}'")
        self.key = key
        self.value = value
