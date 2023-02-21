import os
import logging

from mlflow.pyfunc import PythonModel, PythonModelContext

from medcat.cat import CAT
from medcat.utils.regression.checking import RegressionChecker
from medcat.utils.regression.results import MultiDescriptor
from medcat.utils.regression.targeting import TranslationLayer

from models.model import Model
from models.annotations import Annotation, Concept, ConceptBank
from models.annotations import get_concept_bank
from tracking.performance import RegressionSuite, RegressionMetrics


CONCEPT_BANK: ConceptBank = get_concept_bank()


MAX_ONTOLOGY_LENGTH = 20
MAX_DECRIPTION_LENGTH = 20

logger = logging.getLogger(__name__)


def get_ontology_and_version(model_card: dict) -> tuple[str, str]:
    # This is more or less copied from MedCAT unreleased repo
    # TODO - use the new release along with the method from there
    ont_list = model_card["Source Ontology"]
    if isinstance(ont_list, list):
        ont1 = ont_list[0]
    elif isinstance(ont_list, str):
        ont1 = ont_list
    else:
        raise KeyError(f"Unknown source ontology: {ont_list}")
    # find ontology
    if "SNOMED" in ont1.upper():
        return "SNOMED-CT", ont1
    elif "UMLS" in ont1.upper():
        return "UMLS", ont1
    elif "ICD" in ont1.upper():
        return "ICD", ont1
    else:
        raise ValueError(f"Unknown ontology: {ont1}")


def get_name_from_modelcard(d: dict) -> str:
    """Get model name from model card.

    Currently, this combines:
        The medCAT version
        The model ID
        The ontology

    Args:
        d (dict): The model card in dict form.

    Returns:
        str: The resulting model name.
    """
    version = d["MedCAT Version"]
    model_id = d["Model ID"]
    ontology, _ = get_ontology_and_version(d)
    return f"MedCAT_v{version}_{ontology}_{model_id}"


def get_annotations_from_raw(d: dict) -> list[Annotation]:
    """Get annoations from raw entity dict.

    The results of CAT.get_entities is expected.
    Example entity:
    {'pretty_name': 'Random', 'cui': '255226008', 'type_ids': ['7882689'],
        'types': [''], 'source_value': 'RANDOM', 'detected_name': 'random',
        'acc': 1.0, 'context_similarity': 1.0, 'start': 5, 'end': 11,
        'icd10': [], 'ontologies': ['20220803_SNOMED_UK_CLINICAL_EXT'],
        'snomed': [], 'id': 0, 'meta_anns': {}
    }

    Args:
        d (dict): The input dict.

    Returns:
        list[Annotation]: The list of annotations.
    """
    output: list[Annotation] = []
    for ann in d["entities"].values():
        cui: str = ann["cui"]
        name: str = ann["detected_name"]
        concept: Concept = CONCEPT_BANK.get_or_add_concept(cui, name)
        raw: str = ann["source_value"]
        start: int = ann["start"]
        end: int = ann["end"]
        ann = Annotation(concept, raw, start, end)
        output.append(ann)
    return output


class MedCATMLFlowWrapper(PythonModel):
    def load_context(self, context: PythonModelContext) -> None:
        self.cat = CAT.load_model_pack(context.artifacts["medcat_modelpack"])

    def predict(self, context: PythonModelContext, model_input: str):
        return self.cat.get_entities(model_input)


class MedCATModel(Model):
    """Describes the MEDCAT model."""

    def __init__(
        self, model_path: str, cat: CAT, mlflow_model: MedCATMLFlowWrapper
    ) -> None:
        self.model_path = model_path
        self.cat = cat
        self.mlflow_model = mlflow_model
        self._name = get_name_from_modelcard(cat.get_model_card(as_dict=True))

    def get_mlflow_model(self) -> PythonModel:
        return self.mlflow_model

    def get_model_name(self) -> str:
        return self._name

    def get_model_path(self) -> str:
        return self.model_path

    def get_model_tag(self) -> str:
        return str(self.cat.config.version.id)

    def annotate(self, document: str) -> list[Annotation]:
        out = self.cat.get_entities(document)
        return get_annotations_from_raw(out)


def get_medcat_model(model_pack_path: str) -> Model:
    """Get the medcat model from the model path.

    Loads the CAT instance and wraps it in MedCATModel.

    Args:
        model_pack_path (str): The path to the zip file.

    Returns:
        Model: The resulting model.
    """
    cat = CAT.load_model_pack(model_pack_path)
    mlflow_model = MedCATMLFlowWrapper()
    return MedCATModel(model_pack_path, cat, mlflow_model)


class MedCATRegressionSuit(RegressionSuite):
    """The MedCAT regression suite wrapper."""

    def __init__(self, name: str, checker: RegressionChecker) -> None:
        self.name = name
        self.checker = checker

    def get_name(self) -> str:
        return self.name

    def get_result(self, model: Model) -> RegressionMetrics:
        if not isinstance(model, MedCATModel):
            raise ValueError("Need to specify a MedCAT model")
        tl = TranslationLayer.from_CDB(model.cat.cdb)
        descr: MultiDescriptor = self.checker.check_model(model.cat, tl)
        # _extra = []
        extra = {}
        for part in descr.parts:
            for fd in part.failures:
                if fd.reason not in extra:
                    extra[fd.reason] = 0
                extra[fd.reason] += 1
                # _extra.append(fd.reason)
        # str([part.failures for part in descr.parts])
        extra_ordered = dict(sorted(extra.items(), key=lambda item: -item[1]))
        metrics = RegressionMetrics(
            success=descr.success,
            fail=descr.fail,
            extra=[str({k.name: v for k, v in extra_ordered.items()})],
        )
        return metrics


def get_medcat_regression(paths: list[str]) -> list[RegressionSuite]:
    suites: list[RegressionSuite] = []
    for path in paths:
        try:
            rc = RegressionChecker.from_yaml(path)
        except Exception as e:  # TODO - use specific exception
            logger.warning(
                "Exception raised while generating checker from %s",
                path,
                exc_info=e,  # pass exception
            )
            continue
        name = os.path.basename(path)[:-4]  # remove .yml
        suites.append(MedCATRegressionSuit(name, rc))
    return suites
