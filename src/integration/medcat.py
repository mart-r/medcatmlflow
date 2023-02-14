import os

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


def get_name_from_modelcard(d: dict) -> str:
    """Get model name from model card.

    Currently, this combines the description with the ontology.

    Args:
        d (dict): The model card in dict form.

    Returns:
        str: The resulting model name.
    """
    # example model card
    # 'Description': self.config.version.description,
    # 'Source Ontology': self.config.version.ontology,
    description = d["Description"]
    ontology = str(d["Source Ontology"])
    if len(ontology) > MAX_ONTOLOGY_LENGTH:
        ontology = ontology[:MAX_ONTOLOGY_LENGTH] + " ..."
    if len(description) > MAX_DECRIPTION_LENGTH:
        description = description[:MAX_DECRIPTION_LENGTH] + " ..."
    return f"CAT with ontology '{ontology}' and description '{description}'"


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


class MedCATModel(Model):
    """Describes the MEDCAT model."""

    def __init__(self, cat: CAT) -> None:
        self.cat = cat
        self._name = get_name_from_modelcard(cat.get_model_card(as_dict=True))

    def get_model_name(self) -> str:
        return self._name

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
    return MedCATModel(cat)


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
            # TODO - logging
            print("ISSUE with generating checker from", path, e)
            continue
        name = os.path.basename(path)[:-4]  # remove .yml
        suites.append(MedCATRegressionSuit(name, rc))
    return suites
