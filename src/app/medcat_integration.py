import copy

from typing import Optional

import logging

from .utils import ModelMetaData

from medcat.cat import CAT
from medcat.utils.versioning import ConfigUpgrader

from pydantic import ValidationError

import shutil

logger = logging.getLogger(__name__)


def _try_update(file_path: str, new_model: Optional[str] = None,
                overwrite: bool = True) -> CAT:
    if not new_model and overwrite:
        new_model = file_path
    if new_model.endswith('.zip'):
        new_model = file_path[:-4]
    if new_model == file_path or new_model + ".zip" == file_path:
        # can't overwrite open file
        logger.info("Direct overwrite is currently not supported")
        move_to_same = True
        new_model += "_copy"
        logger.info("So will change the temporary new model name to "
                    "%s and move back to %s later",
                    new_model, file_path)
    else:
        move_to_same = False
    logger.debug("Setting up upgrader for %s", file_path)
    upgrader = ConfigUpgrader(file_path)
    logger.debug("Starting the upgrade process")
    upgrader.upgrade(new_model, overwrite=overwrite)
    logger.debug("Loading from new file: %s", new_model)
    if move_to_same:
        logger.info("After upgrade, moving back to original path")
        if file_path.endswith(".zip"):
            file_path_w_zip = file_path
            file_path_no_zip = file_path[:-4]
        else:
            file_path_w_zip = file_path + ".zip"
            file_path_no_zip = file_path
        logger.info("Moving the directory %s -> %s", new_model,
                    file_path_no_zip)
        shutil.copytree(new_model, file_path_no_zip,
                        dirs_exist_ok=True)
        logger.info("Moving the zip %s -> %s", new_model + ".zip",
                    file_path_w_zip)
        shutil.copy(new_model + ".zip", file_path_w_zip)
    return CAT.load_model_pack(new_model)


def _load_CAT(file_path: str, new_model: Optional[str] = None,
              overwrite: bool = True) -> CAT:
    try:
        return CAT.load_model_pack(file_path)
    except ValidationError as e:
        logger.warning("Validation issue when loading CAT (%s). "
                       "Trying to load after fixing issue with "
                       "config.linking.filters.cuis",
                       file_path, exc_info=e)
    return _try_update(file_path, new_model, overwrite)


def create_meta(file_path: str, model_name: str) -> ModelMetaData:
    # print('Loading CAT to get model info')
    cat = _load_CAT(file_path)
    version = cat.config.version.id
    version_history = ",".join(cat.config.version.history)
    # make sure it's a deep copy
    performance = copy.deepcopy(cat.config.version.performance)
    # print('Found the model info/data')
    return ModelMetaData(model_file_name=model_name, version=version,
                         version_history=version_history,
                         performance=performance)
