from typing import Callable, Optional
import os

import logging

from ..main.envs import STORAGE_PATH
from ..main.models import db as flask_db, TestDataset

from ..medcat_linkage.medcat_integration import (
    get_model_performance_with_dataset as calc_performance,
    AllModelPerformanceResults
)
from ..medcat_linkage.metadata import ModelMetaData
from .cache import get_cached, add_to_cache as _add_to_cache

DATASET_PATH = os.path.join(STORAGE_PATH, "test_datasets")

logger = logging.getLogger(__name__)


def get_test_datasets() -> list[tuple[str, str, str, str]]:
    datasets: list[TestDataset] = TestDataset.query.all()
    return [(ds.category_name, ds.name, ds.description, ds.file_path)
            for ds in datasets]


def _get_ds_file(ds_name: str) -> str:
    return os.path.join(DATASET_PATH, ds_name)


def upload_test_dataset(
    file_saver: Callable[[str], None],
    category_name: str,
    ds_name: str,
    ds_description: str,
    overwrite: bool,
) -> Optional[str]:
    # Save the uploaded file to the desired location
    file_path = _get_ds_file(ds_name)

    if os.path.exists(file_path) and not overwrite:
        return f"Dataset file already exists: {ds_name}"

    # save on disk
    file_saver(file_path)

    # save info to databse
    descr = TestDataset(name=ds_name, category_name=category_name,
                        description=ds_description, file_path=file_path)

    flask_db.session.add(descr)
    flask_db.session.commit()


def delete_test_dataset(ds_name: str):
    found: Optional[TestDataset]
    found = TestDataset.query.filter_by(name=ds_name).first()
    if found:
        flask_db.session.delete(found)
        flask_db.session.commit()
        logger.info("Removing test dataset: %s", found)
    else:
        logger.warning("Unable to delete test dataset: '%s' - not found",
                       ds_name)
    file_path = _get_ds_file(ds_name)
    if os.path.exists(file_path):
        logger.info("Removing '%s' from '%s'", ds_name, file_path)
        os.remove(file_path)
    else:
        logger.warning("Unable to remove file '%s' - no such file", file_path)


def find_or_load_performance(
    models: list[ModelMetaData], datset_info: list[tuple[str, str]]
) -> AllModelPerformanceResults:
    all_results = {}
    for model in models:
        model_results = {}
        for dataset_name in datset_info:
            dataset_file_path = _get_ds_file(dataset_name)
            dataset_file_basename = os.path.basename(dataset_name)
            try:
                result = get_cached(model_id=model.id, ds_id=dataset_name)
            except ValueError:
                full_model_path = os.path.join(STORAGE_PATH,
                                               model.model_file_name)
                result = calc_performance(full_model_path, dataset_file_path)
                _add_to_cache(model.id, dataset_name, result)
            model_results[dataset_file_basename] = result
        all_results[model.name] = model_results
    return all_results
