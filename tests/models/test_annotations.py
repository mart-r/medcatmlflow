import unittest

from src.models.annotations import Concept, ConceptBank, get_concept_bank


class Test_get_concept_bank_method(unittest.TestCase):
    def test_get_method(self):
        self.assertIsInstance(get_concept_bank(), ConceptBank)

    def test_get_method_same_instance(self):
        cb1 = get_concept_bank()
        cb2 = get_concept_bank()
        self.assertIs(cb1, cb2)


class TestConceptBank(unittest.TestCase):
    def setUp(self) -> None:
        self.cb = ConceptBank()  # new instance every time
        self.concept_id = "123"
        self.concept_name = "some-concept"

    def test_concept_bank_does_not_have_new_concept(self):
        cid, cname = self.concept_id, self.concept_name
        self.assertFalse(self.cb.has_concept(cid, cname))

    def test_concept_bank_records_concept(self):
        cid, cname = self.concept_id, self.concept_name
        self.cb.add_concept(cid, cname)
        self.assertTrue(self.cb.has_concept(cid, cname))

    def test_concept_bank_does_not_have_concept_with_same_id_wrong_name(self):
        cid, cname = self.concept_id, self.concept_name
        self.cb.add_concept(cid, cname)
        new_name = cname + cname
        self.assertFalse(self.cb.has_concept(cid, new_name))

    def test_concept_bank_does_not_have_concept_with_same_name_wrong_id(self):
        cid, cname = self.concept_id, self.concept_name
        self.cb.add_concept(cid, cname)
        new_cid = cid + cid
        self.assertFalse(self.cb.has_concept(new_cid, cname))

    def test_concept_bank_get_or_add_adds(self):
        cid, cname = self.concept_id, self.concept_name
        concept = self.cb.get_or_add_concept(cid, cname)
        self.assertIsInstance(concept, Concept)

    def test_concept_bank_get_or_add_gets_same_2nd_time(self):
        cid, cname = self.concept_id, self.concept_name
        concept1 = self.cb.get_or_add_concept(cid, cname)
        concept2 = self.cb.get_or_add_concept(cid, cname)
        self.assertIs(concept1, concept2)

    def test_concept_bank_get_or_add_adds_with_correct_name_id(self):
        cid, cname = self.concept_id, self.concept_name
        concept = self.cb.get_or_add_concept(cid, cname)
        self.assertEqual(cid, concept.id)
        self.assertEqual(cname, concept.name)
