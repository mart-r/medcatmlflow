import copy
import logging
import os
from uuid import uuid4

from dataclasses import dataclass, field, fields
from typing import Optional, List

from mlflow.entities.model_registry import RegisteredModel

from .medcat_integration import load_CAT
from .mct_integration import get_mct_cdb_id

logger = logging.getLogger(__name__)


@dataclass
class ModelMetaData:
    # stuff to identify model and/or its metadata
    id: str
    name: str
    # stuff that describes a model
    description: str
    category: str
    version: str
    version_history: List[str]
    # tertiary descriptors
    cdb_hash: str
    stats: dict
    performance: dict
    changed_parts: List[str]
    # stuff that describes mlflow things
    model_file_name: str
    run_id: str
    mct_cdb_id: Optional[str] = field(default=None)

    def as_dict(self) -> dict:
        return dict((key, getattr(self, key)) for key in self.get_keys())

    @classmethod
    def get_keys(cls) -> List[str]:
        return [field.name for field in fields(cls)]

    @classmethod
    def from_mlflow_model(cls, model: RegisteredModel,
                          run_id: str) -> "ModelMetaData":
        kwargs = {}
        for key in cls.get_keys():
            kwargs[key] = model.tags[key]
        kwargs["run_id"] = run_id
        # fix all non-string values
        # TODO - do this better
        if ('performance' in kwargs
                and not isinstance(kwargs['performance'], dict)):
            # TODO - make the below better
            kwargs["performance"] = eval(kwargs["performance"])
        if ('stats' in kwargs
                and not isinstance(kwargs['stats'], dict)):
            # TODO - make the below better
            kwargs["stats"] = eval(kwargs["stats"])
        # str -> list
        kwargs["version_history"] = eval(kwargs["version_history"])
        return cls(**kwargs)


def _generate_new_model_id():
    return str(uuid4())


def create_meta(
    file_path: str,
    model_name: str,
    description: str,
    category: str,
    run_id: str,
    hash2mct_id: dict,
    existing_id: Optional[str] = None
) -> ModelMetaData:
    """Create model metadata.

    This will method load the model and read the data from the model
    and create a metadata object.

    The idea is that we then don't have to load the entire model
    every time we want to know something about it.

    Args:
        file_path (str): The path to the model .zip
        model_name (str): The (short) name of the model
        description (str): The model description
        category (str): The category of the model (e.g ontology)
        run_id (str): The internal run ID
        hash2mct_id (dict): The dictionary of CDB hashes mapped to MCT CDB ids
        existing_id (Optional[str], optional): The existing CDB id if knwon.
            Defaults to None.

    Returns:
        ModelMetaData: The resulting metadata.
    """
    model_file_name = os.path.basename(file_path)
    cat = load_CAT(file_path)
    version = cat.config.version.id
    version_history = cat.config.version.history.copy()
    # make sure it's a deep copy
    performance = copy.deepcopy(cat.config.version.performance)
    # in case something gets modified - nothing right now
    changed_parts: List[str] = []
    cdb_hash = cat.cdb.get_hash()
    if cdb_hash in hash2mct_id:
        mct_cdb_id = hash2mct_id[cdb_hash]
        logger.debug("Setting MCT CDB hash for '%s' to '%s' "
                     "based on existing models", cdb_hash, mct_cdb_id)
    else:
        mct_cdb_id = get_mct_cdb_id(cdb_hash)
        logger.debug("Setting MCT CDB hash for '%s' to '%s' "
                     "as read from the CDB", cdb_hash, mct_cdb_id)
    stats = cat.cdb.make_stats()
    if existing_id:
        model_id = existing_id
        logger.info("Using existing UUID of '%s' - "
                    "hopefully during recalculation of metadata", model_id)
    else:
        model_id = _generate_new_model_id()
    return ModelMetaData(
        id=model_id,
        name=model_name,
        description=description,
        category=category,
        version=version,
        version_history=version_history,
        cdb_hash=cdb_hash,
        stats=stats,
        performance=performance,
        changed_parts=changed_parts,
        model_file_name=model_file_name,
        run_id=run_id,
        mct_cdb_id=mct_cdb_id,
    )
