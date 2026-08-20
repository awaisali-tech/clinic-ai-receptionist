import unittest

import numpy as np

from conversation.intent_classifier import InformationNeed
from ingestion.document_builder import RAGDocument
from retrieval.grounding_validator import GroundingValidator
from retrieval.hybrid_retriever import HybridRetriever, RetrievalCandidate
from retrieval.structured_retriever import StructuredRetriever
from retrieval.vector_retriever import RetrievalResult, VectorRetriever


def make_document(
    document_id,
    document_type,
    *,
    clinic="Clinic Alpha",
    doctor=None,
    specialization=None,
    information_types=(),
    text=None,
    **fields,
):
    metadata = {
        "document_id": document_id,
        "document_type": document_type,
        "clinic_name": clinic,
        "information_types": tuple(information_types),
        **fields,
    }
    if doctor:
        metadata["doctor_name"] = doctor
    if specialization:
        metadata["specialization"] = specialization
    return RAGDocument(text=text or f"Evidence for {document_id}", metadata=metadata)


def make_candidate(
    document,
    *,
    semantic_score=0.8,
    structured=True,
    entity_match=True,
    type_match=True,
):
    sources = []
    if structured:
        sources.append("structured")
    if semantic_score is not None:
        sources.append("vector")
    return RetrievalCandidate(
        document=document,
        semantic_score=semantic_score,
        structured_match=structured,
        entity_match=entity_match,
        document_type_match=type_match,
        sources=tuple(sources),
        final_score=0.8,
    )


class FakeVectorRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k=5, **kwargs):
        self.calls.append((query, top_k))
        return self.results[:top_k]


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.doctor = make_document(
            "doctor-a",
            "doctor",
            doctor="Dr. Alice Adams",
            specialization="Pediatrics",
            information_types=("doctor_information", "doctor_availability"),
            availability="Mon-Fri",
        )
        self.location = make_document(
            "clinic-a",
            "clinic",
            information_types=(
                "clinic_information",
                "clinic_location",
                "clinic_contact",
            ),
            address="1 Main Street",
            phone="123",
            email="alpha@example.test",
            about="Clinic Alpha",
        )

    def test_fuses_both_sources_deduplicates_and_ranks_answerable_type(self):
        vector = FakeVectorRetriever(
            [
                RetrievalResult(self.location, 0.82),
                RetrievalResult(self.doctor, 0.91),
            ]
        )
        retriever = HybridRetriever(
            StructuredRetriever([self.doctor, self.location]),
            vector,
        )

        results = retriever.retrieve(
            "Where is Clinic Alpha?",
            clinic_name="Clinic Alpha",
            clinic_source="explicit",
            information_need=InformationNeed.CLINIC_LOCATION,
            top_k=5,
        )

        self.assertEqual(len(results), 2)
        self.assertIs(results[0].document, self.location)
        self.assertEqual(results[0].sources, ("structured", "vector"))
        self.assertTrue(results[0].document_type_match)
        self.assertFalse(results[1].document_type_match)
        self.assertEqual(
            len({result.document.metadata["document_id"] for result in results}),
            2,
        )

    def test_explicit_entity_match_scores_above_inherited_match(self):
        vector = FakeVectorRetriever([RetrievalResult(self.location, 0.8)])
        retriever = HybridRetriever(
            StructuredRetriever([self.location]),
            vector,
        )

        explicit = retriever.retrieve(
            "Where is Clinic Alpha?",
            clinic_name="Clinic Alpha",
            clinic_source="explicit",
            information_need=InformationNeed.CLINIC_LOCATION,
            top_k=1,
        )[0]
        inherited = retriever.retrieve(
            "Where is it?",
            clinic_name="Clinic Alpha",
            clinic_source="inherited",
            information_need=InformationNeed.CLINIC_LOCATION,
            top_k=1,
        )[0]

        self.assertGreater(explicit.final_score, inherited.final_score)
        self.assertTrue(explicit.explicit_entity_match)
        self.assertTrue(inherited.inherited_entity_match)

    def test_inherited_doctor_does_not_overpower_location_need(self):
        vector = FakeVectorRetriever(
            [
                RetrievalResult(self.doctor, 0.95),
                RetrievalResult(self.location, 0.75),
            ]
        )
        retriever = HybridRetriever(
            StructuredRetriever([self.doctor, self.location]),
            vector,
        )

        results = retriever.retrieve(
            "Where is it?",
            doctor_name="Dr. Alice Adams",
            clinic_name="Clinic Alpha",
            specialization="Pediatrics",
            doctor_source="inherited",
            clinic_source="inherited",
            specialization_source="inherited",
            information_need=InformationNeed.CLINIC_LOCATION,
            top_k=2,
        )

        self.assertIs(results[0].document, self.location)
        self.assertTrue(results[0].document_type_match)


class GroundingAnswerabilityTests(unittest.TestCase):
    def setUp(self):
        self.validator = GroundingValidator()

    def validate(self, query, need, *candidates, doctor=None, clinic="Clinic Alpha"):
        return self.validator.validate(
            query=query,
            results=list(candidates),
            doctor_name=doctor,
            clinic_name=clinic,
            information_need=need,
        )

    def test_location_requires_location_document_and_address(self):
        doctor = make_document(
            "doctor",
            "doctor",
            information_types=("doctor_information",),
            doctor="Dr. Alice Adams",
        )
        location = make_document(
            "location",
            "clinic",
            information_types=("clinic_location",),
            address="1 Main Street",
        )

        wrong = self.validate(
            "Where is Clinic Alpha?",
            InformationNeed.CLINIC_LOCATION,
            make_candidate(doctor, type_match=False),
        )
        correct = self.validate(
            "Where is Clinic Alpha?",
            InformationNeed.CLINIC_LOCATION,
            make_candidate(location),
        )
        self.assertFalse(wrong.is_grounded)
        self.assertTrue(correct.is_grounded)

    def test_contact_requires_contact_field(self):
        empty_contact = make_document(
            "empty-contact",
            "clinic",
            information_types=("clinic_contact",),
        )
        contact = make_document(
            "contact",
            "clinic",
            information_types=("clinic_contact",),
            phone="123",
        )
        self.assertFalse(
            self.validate(
                "What is the phone number?",
                InformationNeed.CLINIC_CONTACT,
                make_candidate(empty_contact),
            ).is_grounded
        )
        self.assertTrue(
            self.validate(
                "What is the phone number?",
                InformationNeed.CLINIC_CONTACT,
                make_candidate(contact),
            ).is_grounded
        )

    def test_timings_reject_doctor_and_accept_timings_metadata(self):
        doctor = make_document(
            "doctor",
            "doctor",
            information_types=("doctor_information",),
            doctor="Dr. Alice Adams",
        )
        timings = make_document(
            "timings",
            "timings",
            information_types=("clinic_timings",),
            timings={"Sat": "10-2"},
        )
        self.assertFalse(
            self.validate(
                "When does Clinic Alpha open?",
                InformationNeed.CLINIC_TIMINGS,
                make_candidate(doctor, type_match=False),
            ).is_grounded
        )
        self.assertTrue(
            self.validate(
                "When does Clinic Alpha open?",
                InformationNeed.CLINIC_TIMINGS,
                make_candidate(timings),
            ).is_grounded
        )

    def test_doctor_availability_requires_availability_field(self):
        faq = make_document(
            "faq",
            "faq",
            doctor="Dr. Alice Adams",
            information_types=("faq",),
            faq_answer="Dr. Alice is popular.",
        )
        doctor = make_document(
            "doctor",
            "doctor",
            doctor="Dr. Alice Adams",
            information_types=("doctor_information", "doctor_availability"),
            availability="Mon-Fri",
        )
        self.assertFalse(
            self.validate(
                "When is Dr. Alice Adams available?",
                InformationNeed.DOCTOR_AVAILABILITY,
                make_candidate(faq, type_match=False),
                doctor="Dr. Alice Adams",
            ).is_grounded
        )
        self.assertTrue(
            self.validate(
                "When is Dr. Alice Adams available?",
                InformationNeed.DOCTOR_AVAILABILITY,
                make_candidate(doctor),
                doctor="Dr. Alice Adams",
            ).is_grounded
        )

    def test_services_reject_doctor_only_evidence(self):
        doctor = make_document(
            "doctor",
            "doctor",
            information_types=("doctor_information",),
        )
        service = make_document(
            "service",
            "service",
            information_types=("services",),
            service_name="Pediatrics",
        )
        self.assertFalse(
            self.validate(
                "What services are offered?",
                InformationNeed.SERVICES,
                make_candidate(doctor, type_match=False),
            ).is_grounded
        )
        self.assertTrue(
            self.validate(
                "What services are offered?",
                InformationNeed.SERVICES,
                make_candidate(service),
            ).is_grounded
        )

    def test_unknown_fact_requires_non_entity_lexical_support(self):
        clinic = make_document(
            "clinic",
            "clinic",
            information_types=("clinic_information",),
            about="Clinic Alpha provides general care.",
            text="Clinic Alpha provides general care.",
        )
        result = self.validate(
            "Does Clinic Alpha have parking?",
            InformationNeed.GENERAL,
            make_candidate(clinic, semantic_score=0.9),
        )
        self.assertFalse(result.is_grounded)


class VectorRetrieverTests(unittest.TestCase):
    def test_larger_candidate_pool_backfills_qualifying_results(self):
        documents = [
            make_document(f"doc-{index}", "faq", clinic=None)
            for index in range(4)
        ]

        class FakeStore:
            def __init__(self):
                self.requested_top_k = None

            def search(self, query_embedding, top_k):
                self.requested_top_k = top_k
                return list(
                    zip(documents, (0.10, 0.20, 0.80, 0.70))
                )

        class FakeEmbedder:
            def embed_query(self, query):
                return np.array([1.0], dtype=np.float32)

        store = FakeStore()
        retriever = VectorRetriever(store=store, embedder=FakeEmbedder())
        results = retriever.retrieve("query", top_k=2, min_score=0.35)

        self.assertGreater(store.requested_top_k, 2)
        self.assertEqual([result.score for result in results], [0.80, 0.70])

    def test_rejects_non_positive_top_k(self):
        class Unused:
            pass

        retriever = VectorRetriever(store=Unused(), embedder=Unused())
        with self.assertRaises(ValueError):
            retriever.retrieve("query", top_k=0)

    def test_non_finite_similarity_is_ignored(self):
        valid = make_document("valid", "faq", clinic=None)
        invalid = make_document("invalid", "faq", clinic=None)

        class FakeStore:
            def search(self, query_embedding, top_k):
                return [(invalid, float("nan")), (valid, 0.75)]

        class FakeEmbedder:
            def embed_query(self, query):
                return np.array([1.0], dtype=np.float32)

        retriever = VectorRetriever(FakeStore(), FakeEmbedder())
        results = retriever.retrieve("query", top_k=1)

        self.assertEqual([result.document for result in results], [valid])


if __name__ == "__main__":
    unittest.main()
