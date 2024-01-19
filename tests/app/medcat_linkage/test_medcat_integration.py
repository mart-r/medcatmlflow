from src.app.medcat_linkage.medcat_integration import (
    load_CAT, get_model_performance_with_dataset,
    get_cui_counts_for_model,
)

from medcat.cat import CAT

import os
from .. import TESTS_RESOURCES_PATH


from .helpers import TestCaseWithSpacyModel, TEST_MODEL_PACK_PATH


class LoadModelTests(TestCaseWithSpacyModel):

    def test_model_loads(self):
        cat = load_CAT(TEST_MODEL_PACK_PATH)
        self.assertIsInstance(cat, CAT)

    def test_second_call_returns_same(self):
        # i.e it remembers the instance
        cat1 = load_CAT(TEST_MODEL_PACK_PATH)
        cat2 = load_CAT(TEST_MODEL_PACK_PATH)
        self.assertIs(cat1, cat2)


class ModelPerformanceTests(TestCaseWithSpacyModel):
    dataset_path = os.path.join(TESTS_RESOURCES_PATH, "datasets",
                                "example_dataset.json")
    expected_performance = {
        'False positives': 0,
        'False negatives': 0,
        'True positives': 1,
        'Precision for each CUI':
            {
                'C0000239': 1.0
            },
        'Recall for each CUI':
            {
                'C0000239': 1.0
            },
        'F1 for each CUI':
            {
                'C0000239': 1.0
            },
        'Counts for each CUI':
            {
                'C0000239': 1
            },
        'Examples for each of the fp, fn, tp':
            {
                'fp': {},
                'fn': {},
                'tp':
                {
                    'C0000239':
                    [{
                        'text': 'Some virus attacked my second csv yesterday',
                        'cui': 'C0000239',
                        'start': 23,
                        'end': 33,
                        'source value': 'second csv',
                        'acc': 1.0,
                        'project name': 'Mock project #1',
                        'document name': 'Document # 1',
                        'project id': None,
                        'document id': None
                    }]
                }
                }
        }

    def test_preformance_as_expected(self):
        perf = get_model_performance_with_dataset(TEST_MODEL_PACK_PATH,
                                                  self.dataset_path)
        self.assertEqual(perf, self.expected_performance)


class GettingCUICountsForModelTests(TestCaseWithSpacyModel):
    cuis = ['C0000039', 'C0000139', 'C0000239']
    # no training
    expected_counts = {'C0000039': 0, 'C0000139': 0, 'C0000239': 0}

    def test_a(self):
        counts = get_cui_counts_for_model(TEST_MODEL_PACK_PATH, self.cuis)
        self.assertEqual(counts, self.expected_counts)
