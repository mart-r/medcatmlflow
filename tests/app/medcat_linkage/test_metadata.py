from src.app.medcat_linkage.metadata import create_meta, ModelMetaData


from .helpers import TestCaseWithSpacyModel, TEST_MODEL_PACK_PATH
from .helpers import FAKE_HASH2MCT_DICT


class CreateMetaTests(TestCaseWithSpacyModel):

    def test_create_meta_returns_metadata(self):
        meta = create_meta(file_path=TEST_MODEL_PACK_PATH,
                           model_name='test model',
                           description='model describes stuff',
                           category='ontology#1',
                           run_id=-1,
                           hash2mct_id=FAKE_HASH2MCT_DICT)
        self.assertIsInstance(meta, ModelMetaData)
