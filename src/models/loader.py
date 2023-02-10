from enum import Enum, auto
from models.model import Model
from tracking.performance import RegressionSuite
from integration.medcat import get_medcat_model, get_medcat_regression


class ModelType(Enum):
    MEDCAT = auto()

    @classmethod
    def get_type(cls, name: str) -> "ModelType":
        upper_name = name.upper()
        try:
            return ModelType[upper_name]
        except KeyError:
            raise ValueError(f"Unknown model type: {name}")


class Loader:
    def __init__(self, model_type: ModelType) -> None:
        self.model_type = model_type

    def load_model(self, path: str) -> Model:
        if self.model_type == ModelType.MEDCAT:
            return get_medcat_model(path)
        raise ValueError(f"Model type not upported: {type}")

    def load_regr_suite(self, paths: list[str]) -> list[RegressionSuite]:
        if self.model_type == ModelType.MEDCAT:
            return get_medcat_regression(paths)
        raise ValueError(f"Model type not upported: {type}")
