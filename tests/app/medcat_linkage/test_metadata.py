from src.app.medcat_linkage.metadata import create_meta, ModelMetaData

import os
from collections import defaultdict

from .. import TESTS_RESOURCES_PATH
import unittest

import spacy
import shutil


TEST_MODEL_PACK_PATH = os.path.join(TESTS_RESOURCES_PATH, "model_pack")


class AllInDict(defaultdict):

    def __contains__(self, key):
        return True  # Always return True for any key

FAKE_HASH2MCT_DICT = AllInDict(lambda: -1)


class CreateMetaTests(unittest.TestCase):
    spacy_model_path = os.path.join(TEST_MODEL_PACK_PATH, "en_core_sci_md")

    @classmethod
    def setUpClass(cls) -> None:
        nlp = spacy.blank("en")
        nlp.meta["name"] = "core_sci_md"
        nlp.meta["version"] = "0.0.0"
        nlp.to_disk(cls.spacy_model_path)
    
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.spacy_model_path)

    # unittest
    def test_create_meta_returns_metadata(self):
        meta = create_meta(file_path=TEST_MODEL_PACK_PATH,
                           model_name='test model',
                           description='model describes stuff',
                           category='ontology#1',
                           run_id=-1,
                           hash2mct_id=FAKE_HASH2MCT_DICT)
        self.assertIsInstance(meta, ModelMetaData)

