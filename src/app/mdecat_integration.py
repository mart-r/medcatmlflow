import copy

from .utils import ModelMetaData

from medcat.cat import CAT


def create_meta(file_path: str, model_name: str) -> ModelMetaData:
    # print('Loading CAT to get model info')
    cat = CAT.load_model_pack(file_path)
    version = cat.config.version.id
    version_history = ",".join(cat.config.version.history)
    # make sure it's a deep copy
    performance = copy.deepcopy(cat.config.version.performance)
    # print('Found the model info/data')
    return ModelMetaData(model_file_name=model_name, version=version,
                         version_history=version_history,
                         performance=performance)
