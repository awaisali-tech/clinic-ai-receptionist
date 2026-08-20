import unittest

from conversation.entity_resolver import ContextResolver, EntityCatalog
from conversation.intent_classifier import InformationNeed, IntentClassifier
from ingestion.document_builder import build_documents
from ingestion.loader import load_clinic_data
from ingestion.normalizer import normalize_clinic_data


class IntentClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = normalize_clinic_data(load_clinic_data("data/clinic_data.json"))
        cls.catalog = EntityCatalog.from_clinic_data(data)
        cls.classifier = IntentClassifier()

    def classify(self, query):
        resolution = ContextResolver(self.catalog).resolve(query)
        return self.classifier.classify(query, resolution)

    def test_representative_information_needs(self):
        cases = {
            "WHERE is Wellness Eye Clinic?!": InformationNeed.CLINIC_LOCATION,
            "What is Sunrise Medical Center's PHONE number?": (
                InformationNeed.CLINIC_CONTACT
            ),
            "What are the clinic's opening hours?": (
                InformationNeed.CLINIC_TIMINGS
            ),
            "When is Dr. Ayesha Khan available?": (
                InformationNeed.DOCTOR_AVAILABILITY
            ),
            "Tell me about Dr. Fahad Iqbal.": (
                InformationNeed.DOCTOR_INFORMATION
            ),
            "What services does Wellness Eye Clinic offer?": (
                InformationNeed.SERVICES
            ),
            "Do you accept walk-in patients?": InformationNeed.FAQ,
            "Tell me about Wellness Eye Clinic": (
                InformationNeed.CLINIC_INFORMATION
            ),
            "Parking information for Wellness Eye Clinic": (
                InformationNeed.GENERAL
            ),
            "Please help with an administrative matter": InformationNeed.GENERAL,
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(self.classify(query), expected)

    def test_follow_up_day_uses_resolved_doctor_context(self):
        resolver = ContextResolver(self.catalog)
        first = resolver.resolve("Tell me about Dr. Ayesha Khan")
        resolver.commit(first)
        follow_up = resolver.resolve("What about Saturday?")

        self.assertEqual(
            self.classifier.classify("What about Saturday?", follow_up),
            InformationNeed.DOCTOR_AVAILABILITY,
        )


class DocumentMetadataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = normalize_clinic_data(load_clinic_data("data/clinic_data.json"))
        cls.documents = build_documents(data)

    def test_doctor_metadata_contains_availability_and_associations(self):
        doctor = next(
            document
            for document in self.documents
            if document.metadata.get("doctor_name") == "Dr. Ayesha Khan"
        )

        self.assertEqual(doctor.metadata["document_type"], "doctor")
        self.assertEqual(doctor.metadata["clinic_name"], "Sunrise Medical Center")
        self.assertEqual(doctor.metadata["specialization"], "Pediatrics")
        self.assertEqual(doctor.metadata["availability"], "Mon-Fri 9:00am-2:00pm")
        self.assertIn("doctor_availability", doctor.metadata["information_types"])
        self.assertTrue(doctor.metadata["document_id"])

    def test_field_documents_expose_canonical_answerability_metadata(self):
        clinic = next(
            document
            for document in self.documents
            if document.metadata["document_id"] == "clinic_003:clinic"
        )
        timings = next(
            document
            for document in self.documents
            if document.metadata["document_id"] == "clinic_003:timings"
        )
        service = next(
            document
            for document in self.documents
            if document.metadata.get("service_name") == "Eye Exams"
        )

        self.assertEqual(
            clinic.metadata["address"],
            "78 Vision Road, Model Town, Lahore, Pakistan",
        )
        self.assertEqual(clinic.metadata["phone"], "+92-42-11223344")
        self.assertIn("clinic_contact", clinic.metadata["information_types"])
        self.assertEqual(timings.metadata["timings"]["Sun"], "Closed")
        self.assertEqual(service.metadata["information_types"], ("services",))


if __name__ == "__main__":
    unittest.main()
