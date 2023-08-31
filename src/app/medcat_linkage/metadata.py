import copy
import logging

from dataclasses import dataclass, field, fields
from typing import Optional

from mlflow.entities.model_registry import RegisteredModel

from .medcat_integration import load_CAT
from .mct_integration import get_mct_cdb_id

logger = logging.getLogger(__name__)


@dataclass
class ModelMetaData:
    category: str
    run_id: str
    version: str
    description: str
    version_history: list[str]
    model_file_name: str
    performance: dict
    changed_parts: list[str]
    cdb_hash: str
    stats: dict
    mct_cdb_id: Optional[str] = field(default=None)

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "run_id": self.run_id,
            "version": self.version,
            "description": self.description,
            "version_history": self.version_history,
            "model_file_name": self.model_file_name,
            "performance": self.performance,
            "changed_parts": self.changed_parts,
            "cdb_hash": self.cdb_hash,
            "stats": self.stats,
            "mct_cdb_id": self.mct_cdb_id,
        }

    @classmethod
    def get_keys(cls) -> set[str]:
        return set(field.name for field in fields(cls))

    @classmethod
    def from_mlflow_model(cls, model: RegisteredModel,
                          run_id: str) -> "ModelMetaData":
        kwargs = {}
        for key in cls.get_keys():
            kwargs[key] = model.tags[key]
        kwargs['description'] = model.description
        kwargs['run_id'] = run_id
        return cls(**kwargs)


def create_meta(file_path: str,
                model_name: str,
                description: str,
                category: str,
                run_id: str,
                hash2mct_id: dict) -> ModelMetaData:
    cat = load_CAT(file_path)
    version = cat.config.version.id
    version_history = ",".join(cat.config.version.history)
    # make sure it's a deep copy
    performance = copy.deepcopy(cat.config.version.performance)
    # in case something gets modified - nothing right now
    changed_parts = []
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
    return ModelMetaData(
        category=category,
        run_id=run_id,
        model_file_name=model_name,
        version=version,
        description=description,
        version_history=version_history,
        performance=performance,
        changed_parts=changed_parts,
        cdb_hash=cdb_hash,
        stats=stats,
        mct_cdb_id=mct_cdb_id,
    )
