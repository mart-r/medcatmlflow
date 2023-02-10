from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Concept:
    """Describes a unique concept with a name and an ID."""

    name: str
    id: str


@dataclass
class Annotation:
    """Describes a specific annotaiton."""

    concept: Concept
    raw: str
    start: int
    end: int


ConceptMap = dict[str, list[Concept]]


class ConceptBank:
    def __init__(self) -> None:
        self._concept_id_map: ConceptMap = {}
        self._concept_name_map: ConceptMap = {}

    def has_concept(self, id: str, name: str) -> bool:
        if id not in self._concept_id_map:
            return False
        if name not in self._concept_name_map:
            return False
        # check per ID only - if the id-name pair exists somewhere
        # it will be in both dicts
        for id_concept in self._concept_id_map[id]:
            if id_concept.name == name:
                return True
        return False

    def add_concept(self, id: str, name: str) -> Concept:
        if self.has_concept(id, name):
            raise ValueError(f"Concept already in bank: ({id}, {name})")
        concept = Concept(id=id, name=name)
        self._add_concept(concept)
        return concept

    def get_or_add_concept(self, id: str, name: str) -> Concept:
        if id not in self._concept_id_map:
            return self.add_concept(id, name)
        if name not in self._concept_name_map:
            return self.add_concept(id, name)
        # checking to make sure there's a concept with the id-name pair
        id_concepts = self._concept_id_map[id]
        for concept in id_concepts:
            if concept.name == name:
                return concept
        return self.add_concept(id, name)

    @staticmethod
    def _add_to(map: ConceptMap, target: str, concept: Concept) -> None:
        if target not in map:
            map[target] = []
        map[target].append(concept)

    def _add_concept(self, concept: Concept) -> None:
        self._add_to(self._concept_id_map, concept.id, concept)
        self._add_to(self._concept_name_map, concept.name, concept)


@lru_cache
def get_concept_bank():
    return ConceptBank()
