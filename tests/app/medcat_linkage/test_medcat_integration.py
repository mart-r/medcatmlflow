from src.app.medcat_linkage.medcat_integration import load_CAT

from medcat.cat import CAT


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
