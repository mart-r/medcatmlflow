from typing import Dict, TypedDict, Optional

import logging

from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.utils.versioning import ConfigUpgrader

from pydantic import ValidationError

import shutil
import os
import json

from ..main.utils import expire_cache_after


logger = logging.getLogger(__name__)


def _try_update_and_load(file_path: str, overwrite: bool = True) -> CAT:
    if file_path.endswith('.zip'):
        new_model = file_path[:-4] + '_cbdfix'
    else:
        new_model = file_path + '_cbdfix'
    logger.debug("Setting up upgrader for %s", file_path)
    upgrader = ConfigUpgrader(file_path)
    logger.debug("Starting the upgrade process")
    upgrader.upgrade(new_model, overwrite=overwrite)
    logger.debug("Loading from new file: %s", new_model)

    # remove original
    if file_path.endswith('.zip'):
        zip_path = file_path
        folder_path = file_path[:-4]
    else:
        zip_path = file_path + '.zip'
        folder_path = file_path
    logger.debug("Removing original: %s", file_path)
    # remove folder
    shutil.rmtree(folder_path)
    # remove zip
    os.remove(zip_path)

    logger.debug("Moving new: %s -> %s", new_model, folder_path)
    # move to original
    # move folder
    shutil.move(new_model, folder_path)
    # move zip
    shutil.move(new_model + ".zip", zip_path)
    return _load_CAT(zip_path)


@expire_cache_after(60)  # keep for 1 minute
def _load_CAT(file_path: str) -> CAT:
    return CAT.load_model_pack(file_path)


def load_CAT(file_path: str, overwrite: bool = True) -> CAT:
    """Load CAT or update and load CAT.

    If there's a ValidationError (common for older models)
    while loading the model this method will attempt to fix
    the underlying issue and load the subsequent model.

    The user can specify whether or not the fixed model
    overwrites the existing one.

    Args:
        file_path (str): The model ZIP to load.
        overwrite (bool, optional): Whether to overwrite old model when
            trying to update. Defaults to True.

    Returns:
        CAT: The loaded model.
    """
    try:
        return _load_CAT(file_path)
    except ValidationError as e:
        logger.warning("Validation issue when loading CAT (%s). "
                       "Trying to load after fixing issue with "
                       "config.linking.filters.cuis",
                       file_path, exc_info=e)
    return _try_update_and_load(file_path, overwrite)


def get_cdb_hash(cdb_file: str) -> str:
    """Get the hash of a CDB based on file.

    Args:
        cdb_file (str): The CDB file.

    Returns:
        str: The CDB hash.
    """
    cdb = CDB.load(cdb_file)
    cdb_hash = cdb.get_hash()
    del cdb
    return cdb_hash


def _load_data(dsf: str) -> dict:
    with open(dsf) as f:
        return json.load(f)


PerDatasetPerformanceResult = TypedDict(
    "PerformanceResult",
    {
        "False positives": int,
        "False negatives": int,
        "True positives": int,
        "Precision for each CUI": dict[str, float],
        "Recall for each CUI": dict[str, float],
        "F1 for each CUI": dict[str, float],
        "Counts for each CUI": dict[str, float],
        "Examples for each of the fp, fn, tp": dict[str, float],
    },
)
PerModelPerformanceResults = Dict[str, PerDatasetPerformanceResult]
AllModelPerformanceResults = Dict[str, PerModelPerformanceResults]


MODEL_2_PERF_MAP = {
    "fp": "False positives",
    "fn": "False negatives",
    "tp": "True positives",
    "prec": "Precision for each CUI",
    "recall": "Recall for each CUI",
    "f1": "F1 for each CUI",
    "counts": "Counts for each CUI",
    "examples": "Examples for each of the fp, fn, tp",
}

PERF_MAP_2_MODEL = {value: key for key, value in MODEL_2_PERF_MAP.items()}


def remap_to_perf_results(model_dict: dict) -> PerDatasetPerformanceResult:
    return {MODEL_2_PERF_MAP.get(key, key): value
            for key, value in model_dict.items()}


def remap_from_perf_results(res: PerDatasetPerformanceResult) -> dict:
    return {PERF_MAP_2_MODEL.get(key, key): value
            for key, value in res.items()}


def get_model_performance_with_dataset(model_file: str,
                                       dataset_file: str,
                                       cat: Optional[CAT] = None
                                       ) -> PerModelPerformanceResults:
    if cat is None:
        cat = _load_CAT(model_file)
    data = _load_data(dataset_file)
    (fps, fns, tps,
     cui_prec, cui_rec, cui_f1,
     cui_counts, examples) = cat._print_stats(data)
    return {
        "False positives": len(fps),
        "False negatives": len(fns),
        "True positives": len(tps),
        "Precision for each CUI": cui_prec,
        "Recall for each CUI": cui_rec,
        "F1 for each CUI": cui_f1,
        "Counts for each CUI": cui_counts,
        "Examples for each of the fp, fn, tp": examples
    }


def get_performance(models: list[tuple[str, str]],
                    dataset_files: list[str]) -> AllModelPerformanceResults:
    """Get the performance of models given the specified datasets.

    This method iterates over all models and all datasets.
    And it runs CAT._print_stats over each model-dataset pair.

    The end result is a dict in the following rough format:
    {
        "model_name":
        {
            "dataset name 1":
            {
                "False positives": int,
                "False negatives": int,
                "True positives": int,
                "Precision for each CUI": dict[str, float],
                "Recall for each CUI": dict[str, float],
                "F1 for each CUI": dict[str, float],
                "Counts for each CUI": dict[str, float],
                "Examples for each of the fp, fn, tp": dict[str, float]
            },
            ...
        },
        ...
    }

    Args:
        models (list[tuple[str, str]]):
            The model names and corresponding file names
        dataset_files (list[str]):
            The list of dataset files

    Returns:
        ModelPerformanceResults:
            The model's performance results
    """
    out = {}
    for model_name, model_file in models:
        cat = _load_CAT(model_file)
        per_model = {}
        for file_name in dataset_files:
            file_basename = os.path.basename(file_name)
            res = get_model_performance_with_dataset(file_name, file_name,
                                                     cat=cat)
            per_model[file_basename] = res
        out[model_name] = per_model
    return out
