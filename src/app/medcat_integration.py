import numbers

from typing import Any, Iterator

import logging

from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.utils.versioning import ConfigUpgrader

from pydantic import ValidationError

import shutil
import os
import json

logger = logging.getLogger(__name__)


def _try_update(file_path: str, overwrite: bool = True) -> CAT:
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
    shutil.move(new_model + '.zip', zip_path)
    return CAT.load_model_pack(zip_path)


def _load_CAT(file_path: str, overwrite: bool = True) -> CAT:
    try:
        return CAT.load_model_pack(file_path)
    except ValidationError as e:
        logger.warning("Validation issue when loading CAT (%s). "
                       "Trying to load after fixing issue with "
                       "config.linking.filters.cuis",
                       file_path, exc_info=e)
    return _try_update(file_path, overwrite)


def _remove_half(d: dict[str, Any], smallest: bool = True) -> None:
    # this expects all the values to be of the same
    # (or similar) type so they all can (or cannot)
    # be compared
    all_keys = list(d.keys())
    half_length = len(all_keys)//2
    key0 = all_keys[0]
    val0 = d[key0]
    is_numeric = isinstance(val0, numbers.Number)
    if is_numeric:
        # remove the smallest (or biggest) elements
        ordered_keys = sorted(all_keys, key=lambda k: d[k],
                              reverse=not smallest)
        for key in ordered_keys[:half_length]:
            del d[key]
    else:
        # remove random
        # i.e the first ones that come up
        for key in all_keys[:half_length]:
            del d[key]


def _attempt_fix_big(*dicts: list[dict[str, Any]],
                     limit: int = 5000) -> tuple[list[dict[str, Any]],
                                                 list[bool]]:
    dicts = list(dicts)
    changes = [False for _ in dicts]
    for i, d in enumerate(dicts):
        if not isinstance(d, dict):
            logger.info("Found a dict that needs unwrapped (%s)", d)
            # in case of Delegating dicts, for instance
            # simply unwrap
            d = dict(**d)
            dicts[i] = d
        s = str(d)
        while len(s) > limit:
            logger.info("Truncating dict from length %d to half size", len(s))
            _remove_half(d)
            s = str(d)
            changes[i] = True
    return tuple(dicts), changes


def get_cdb_hash(cdb_file: str) -> str:
    cdb = CDB.load(cdb_file)
    return cdb.get_hash()


def _iterate_datasets(dataset_files: list[str]) -> Iterator[tuple[str, dict]]:
    for dsf in dataset_files:
        with open(dsf) as f:
            data = json.load(f)
        yield dsf, data


def get_performance(models: list[tuple[str, str]],
                    dataset_files: list[str]) -> dict:
    out = {}
    for model_name, model_file in models:
        cat = CAT.load_model_pack(model_file)
        per_model = {}
        for file_name, data in _iterate_datasets(dataset_files):
            (fps, fns, tps,
             cui_prec, cui_rec, cui_f1,
             cui_counts, examples) = cat._print_stats(data)
            per_model[file_name] = {
                "False positives": fps,
                "False negatives": fns,
                "True postiives": tps,
                "Precision for each CUI": cui_prec,
                "Recall for each CUI": cui_rec,
                "F1 for each CUI": cui_f1,
                "Counts for each CUI": cui_counts,
                "Examples for each of the fp, fn, tp": examples
            }
        out[model_name] = per_model
    return out
