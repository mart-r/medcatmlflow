from src.app.main.utils import build_nodes, get_all_trees

from typing import Dict, Tuple, List

import unittest

EXAMPLE_DATA: Dict[str, Tuple[List[str], str]] = {
    # group T1
    "T1-v1": ([], "T1"),
    # missing T1-v2
    "T1-v3": (["T1-v1", "T1-v2"], "T1"),
    # group T2
    # missing first 3
    "T2-v4": (["T2-v1", "T2-v2", "T2-v3"], "T2"),
    "T2-v5": (["T2-v1", "T2-v2", "T2-v3", "T2-v4"], "T2"),
}

EXPECTED_TREE_KEYS = {
    'T1-v1',
    'T1-v2',
    'T1-v3',
    'T2-v1',
    'T2-v2',
    'T2-v3',
    'T2-v4',
    'T2-v5',
}

EXPECTED_TREE_ADDRESSES = {
    'T1-v1': '/T1-v1',
    'T1-v2': '/T1-v1/T1-v2',
    'T1-v3': '/T1-v1/T1-v2/T1-v3',
    'T2-v1': '/T2-v1',
    'T2-v2': '/T2-v1/T2-v2',
    'T2-v3': '/T2-v1/T2-v2/T2-v3',
    'T2-v4': '/T2-v1/T2-v2/T2-v3/T2-v4',
    'T2-v5': '/T2-v1/T2-v2/T2-v3/T2-v4/T2-v5',
}

EXPECTED_TREE_CATEGORIES = {
    'T1-v1': 'T1',
    'T1-v2': 'T1',
    'T1-v3': 'T1',
    'T2-v1': 'T2',
    'T2-v2': 'T2',
    'T2-v3': 'T2',
    'T2-v4': 'T2',
    'T2-v5': 'T2',
}

EXISTING_CATEGORIES = set([v[1] for v in EXAMPLE_DATA.values()])
DISCOVERED_NODES_PER_CATEGORY = dict([(cat, 
                                sum([cat in k for k in EXPECTED_TREE_ADDRESSES])
                                ) for cat in EXISTING_CATEGORIES])


class BuildNodesEmptyTests(unittest.TestCase):

    def test_empty_data_no_nodes(self):
        res = build_nodes({})
        self.assertFalse(res)


class BuildNodesSimpleTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.res = build_nodes(EXAMPLE_DATA)

    def test_has_correct_number_of_keys(self):
        self.assertEqual(len(self.res.keys()), len(EXPECTED_TREE_KEYS))

    def test_has_all_keys(self):
        for exp_key in EXPECTED_TREE_KEYS:
            with self.subTest(f"Key: {exp_key}"):
                self.assertIn(exp_key, self.res)

    def test_does_not_have_incorrect_keys(self):
        for got_key in self.res:
            with self.subTest(f"Key: {got_key}"):
                self.assertIn(got_key, EXPECTED_TREE_KEYS)

    def test_have_correct_categories(self):
        for key in self.res:
            with self.subTest(f"Key: {key}"):
                node = self.res[key]
                exp_cat = EXPECTED_TREE_CATEGORIES[key]
                self.assertEqual(node.category, exp_cat)

    def test_has_correct_paths(self):
        for key in self.res:
            with self.subTest(f"Key: {key}"):
                node = self.res[key]
                path_str = "/" + "/".join([n.name for n in node.path])
                exp_path = EXPECTED_TREE_ADDRESSES[key]
                self.assertEqual(path_str, exp_path)


class GetAllTreesTests(unittest.TestCase):

    @classmethod
    def get_model_link(cls, mn) -> str:
        return f"ML: {mn}"

    @classmethod
    def get_model_descr(cls, mn) -> str:
        return f"MD: {mn}"

    @classmethod
    def setUpClass(cls) -> None:
        nodes = build_nodes(EXAMPLE_DATA)
        cls.trees = get_all_trees(nodes.values(), cls.get_model_link, cls.get_model_descr)

    def test_nonempty(self):
        self.assertTrue(self.trees)

    def test_has_items_for_each_category(self):
        self.assertEqual(len(self.trees), len(EXISTING_CATEGORIES))

    def test_has_correct_categories(self):
        for _, cat in self.trees:
            with self.subTest(f"Category {cat}"):
                self.assertIn(cat, EXISTING_CATEGORIES)

    def test_has_correct_number_of_lines(self):
        total = 0
        for data, _ in self.trees:
            total += len(data)
        self.assertEqual(total, len(EXPECTED_TREE_ADDRESSES))

    def test_has_correct_nr_of_lines_per_category(self):
        for data, category in self.trees:
            expected = DISCOVERED_NODES_PER_CATEGORY[category]
            with self.subTest(f"Category: {category}"):
                self.assertEqual(len(data), expected)
