import unittest

from src.models.loader import ModelType, Loader


class TestModelType(unittest.TestCase):
    def check_with(self, name):
        with self.subTest(f"Testing name: {name}"):
            got = ModelType.get_type(name)
            self.assertIsInstance(got, ModelType)

    def test_static_getter_works_with_correct(self):
        for mt in ModelType:
            self.check_with(mt.name)
            self.check_with(mt.name.upper())
            self.check_with(mt.name.lower())

    def test_static_getter_fails_incorrect(self):
        for mt in ModelType:
            with self.assertRaises(ValueError):
                ModelType.get_type(mt.name + "+" + mt.name)


class TestLoader(unittest.TestCase):
    FAKE_MODEL_TYPE = "FAKE_MODEL_TYPE"
    FAKE_MODEL_PATH = "FAKE_MODEL_PATH"

    def setUp(self) -> None:
        self.loader = Loader(ModelType.MEDCAT)

    def test_load_model_fails_unknown_model_type(self):
        with self.assertRaises(ValueError):
            self.loader.model_type = self.FAKE_MODEL_TYPE
            self.loader.load_model(self.FAKE_MODEL_PATH)

    def test_load_regr_suite_fails_unknown_model_type(self):
        with self.assertRaises(ValueError):
            self.loader.model_type = self.FAKE_MODEL_TYPE
            self.loader.load_regr_suite([])
