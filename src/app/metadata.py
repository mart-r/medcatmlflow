import copy

from dataclasses import dataclass, field, fields
from typing import Optional

from .medcat_integration import _attempt_fix_big, _load_CAT


@dataclass
class ModelMetaData:
    category: str
    version: str
    version_history: list[str]
    model_file_name: str
    performance: dict
    cui2average_confidence: dict
    cui2count_train: dict
    changed_parts: list[str]
    cdb_hash: str
    mct_cdb_id: Optional[str] = field(default=None)

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "version": self.version,
            "version_history": self.version_history,
            "model_file_name": self.model_file_name,
            "performance": self.performance,
            "cui2average_confidence": self.cui2average_confidence,
            "cui2count_train": self.cui2count_train,
            "changed_parts": self.changed_parts,
            "cdb_hash": self.cdb_hash,
            "mct_cdb_id": self.mct_cdb_id,
        }

    @classmethod
    def get_keys(cls) -> set[str]:
        return set(field.name for field in fields(cls))

    @classmethod
    def from_mlflow_model(cls, model) -> "ModelMetaData":
        kwargs = {}
        for key in cls.get_keys():
            kwargs[key] = model.tags[key]
        return cls(**kwargs)


def create_meta(file_path: str,
                model_name: str,
                category: str) -> ModelMetaData:
    cat = _load_CAT(file_path)
    version = cat.config.version.id
    version_history = ",".join(cat.config.version.history)
    # make sure it's a deep copy
    performance = copy.deepcopy(cat.config.version.performance)
    cui2average_confidence = copy.deepcopy(cat.cdb.cui2average_confidence)
    cui2count_train = copy.deepcopy(cat.cdb.cui2count_train)
    (cui2average_confidence, cui2count_train), changes = _attempt_fix_big(
        cui2average_confidence, cui2count_train
    )
    part_names = ["cui2average_confidence", "cui2count_train"]
    changed_parts = [
        part_name for part_name, change in zip(part_names, changes) if change
    ]
    return ModelMetaData(
        category=category,
        model_file_name=model_name,
        version=version,
        version_history=version_history,
        performance=performance,
        cui2average_confidence=cui2average_confidence,
        cui2count_train=cui2count_train,
        changed_parts=changed_parts,
        cdb_hash=cat.cdb.get_hash(),
    )
