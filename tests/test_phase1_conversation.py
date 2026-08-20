import unittest
from unittest.mock import patch

from conversation.entity_resolver import ContextResolver, EntityCatalog
from conversation.intent_classifier import IntentClassifier
from ingestion.document_builder import RAGDocument
from orchestration.factory import SharedRAGResources, build_session_pipeline
from orchestration.pipeline import RAGPipeline
from retrieval.grounding_validator import GroundingResult, GroundingValidator
from retrieval.hybrid_retriever import RetrievalCandidate


CLINIC_DATA = {
    "clinics": [
        {
            "name": "Clinic Alpha",
            "doctors": [
                {
                    "name": "Dr. Alice Adams",
                    "specialization": "Pediatrics",
                }
            ],
        },
        {
            "name": "Clinic Beta",
            "doctors": [
                {
                    "name": "Dr. Bob Brown",
                    "specialization": "Dermatology",
                }
            ],
        },
    ]
}


def document(
    *,
    clinic: str | None = None,
    doctor: str | None = None,
    specialization: str | None = None,
    document_type: str = "doctor",
) -> RAGDocument:
    metadata = {"document_type": document_type}
    if clinic:
        metadata["clinic_name"] = clinic
    if doctor:
        metadata["doctor_name"] = doctor
    if specialization:
        metadata["specialization"] = specialization
    if document_type == "doctor":
        metadata["information_types"] = (
            "doctor_information",
            "doctor_availability",
        )
        metadata["availability"] = "Mon-Fri"
    elif document_type == "clinic":
        metadata["information_types"] = (
            "clinic_information",
            "clinic_location",
            "clinic_contact",
        )
        metadata.update(
            address="Test address",
            phone="123",
            email="clinic@example.test",
            about="Clinic information",
        )
    elif document_type == "faq":
        metadata["information_types"] = ("faq",)
        metadata["faq_answer"] = "Walk-in patients are accepted."
    return RAGDocument(
        text=f"Evidence: {metadata} Walk-in patients are accepted.",
        metadata=metadata,
    )


class CatalogRetriever:
    def __init__(self):
        self.calls: list[dict] = []
        self.override_document = None

    def retrieve(
        self,
        query,
        doctor_name=None,
        clinic_name=None,
        specialization=None,
        document_type=None,
        top_k=5,
        information_need=None,
        doctor_source=None,
        clinic_source=None,
        specialization_source=None,
    ):
        self.calls.append(
            {
                "query": query,
                "doctor_name": doctor_name,
                "clinic_name": clinic_name,
                "specialization": specialization,
                "document_type": document_type,
                "top_k": top_k,
                "information_need": information_need,
            }
        )

        if self.override_document is not None:
            result_document = self.override_document
        elif doctor_name == "Dr. Alice Adams":
            result_document = document(
                clinic="Clinic Alpha",
                doctor="Dr. Alice Adams",
                specialization="Pediatrics",
            )
        elif doctor_name == "Dr. Bob Brown":
            result_document = document(
                clinic="Clinic Beta",
                doctor="Dr. Bob Brown",
                specialization="Dermatology",
            )
        elif clinic_name:
            result_document = document(
                clinic=clinic_name,
                document_type="clinic",
            )
        elif specialization:
            result_document = document(
                clinic="Clinic Alpha",
                specialization=specialization,
            )
        else:
            result_document = document(
                clinic="Clinic Beta",
                document_type="faq",
            )

        return [
            RetrievalCandidate(
                document=result_document,
                semantic_score=0.9,
                structured_match=True,
                entity_match=True,
                document_type_match=True,
                sources=("fake",),
                final_score=0.9,
            )
        ]


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, query, evidence, conversation_context=None):
        self.calls.append((query, evidence, conversation_context))
        return "Deterministic answer"


class AllowGuard:
    def check(self, value):
        return True, value


class RejectNextGrounding:
    def __init__(self):
        self.reject = False
        self.validator = GroundingValidator()

    def validate(
        self,
        query,
        results,
        doctor_name=None,
        clinic_name=None,
        specialization=None,
        information_need=None,
    ):
        if self.reject:
            return GroundingResult(False, [], "Forced grounding failure.")
        return self.validator.validate(
            query=query,
            results=results,
            doctor_name=doctor_name,
            clinic_name=clinic_name,
            specialization=specialization,
            information_need=information_need,
        )


def make_resources(retriever=None):
    return SharedRAGResources(
        retriever=retriever or CatalogRetriever(),
        entity_catalog=EntityCatalog.from_clinic_data(CLINIC_DATA),
    )


def make_pipeline(resources, grounding_validator=None):
    return build_session_pipeline(
        resources,
        generator=FakeGenerator(),
        grounding_validator=grounding_validator or GroundingValidator(),
        input_guard=AllowGuard(),
        medical_guard=AllowGuard(),
        output_guard=AllowGuard(),
    )


class Phase1ConversationTests(unittest.TestCase):
    def test_session_state_is_isolated_while_resources_are_shared(self):
        resources = make_resources()
        session_a = make_pipeline(resources)
        session_b = make_pipeline(resources)

        self.assertIs(session_a.retriever, session_b.retriever)
        self.assertIsNot(session_a.context_resolver, session_b.context_resolver)

        session_a.run("When is Dr. Alice Adams available?")
        context_a = session_a.context_resolver.get_context()
        context_b = session_b.context_resolver.get_context()
        self.assertEqual(context_a.active_doctor, "Dr. Alice Adams")
        self.assertIsNone(context_b.active_doctor)

        session_b.run("When is Dr. Bob Brown available?")
        session_a.reset_conversation()

        self.assertIsNone(session_a.context_resolver.get_context().active_doctor)
        self.assertEqual(
            session_b.context_resolver.get_context().active_doctor,
            "Dr. Bob Brown",
        )

    def test_explicit_new_clinic_replaces_conflicting_stale_context(self):
        resources = make_resources()
        pipeline = make_pipeline(resources)

        pipeline.run("When is Dr. Alice Adams available?")
        result = pipeline.run("Where is Clinic Beta?")
        retrieval_call = resources.retriever.calls[-1]

        self.assertEqual(retrieval_call["clinic_name"], "Clinic Beta")
        self.assertIsNone(retrieval_call["doctor_name"])
        self.assertIsNone(retrieval_call["specialization"])
        self.assertEqual(result.clinic, "Clinic Beta")
        self.assertIsNone(result.doctor)
        self.assertIsNone(result.specialization)

    def test_elliptical_follow_up_inherits_committed_entities(self):
        resources = make_resources()
        pipeline = make_pipeline(resources)

        pipeline.run("Tell me about Dr. Alice Adams")
        result = pipeline.run("What about Saturday?")
        retrieval_call = resources.retriever.calls[-1]

        self.assertTrue(result.is_follow_up)
        self.assertEqual(retrieval_call["doctor_name"], "Dr. Alice Adams")
        self.assertEqual(retrieval_call["clinic_name"], "Clinic Alpha")
        self.assertEqual(retrieval_call["specialization"], "Pediatrics")

    def test_independent_administrative_question_has_no_stale_filters(self):
        resources = make_resources()
        pipeline = make_pipeline(resources)

        pipeline.run("Tell me about Dr. Alice Adams")
        result = pipeline.run("Do you accept walk-in patients?")
        retrieval_call = resources.retriever.calls[-1]

        self.assertFalse(result.is_follow_up)
        self.assertIsNone(retrieval_call["doctor_name"])
        self.assertIsNone(retrieval_call["clinic_name"])
        self.assertIsNone(retrieval_call["specialization"])
        self.assertIsNone(pipeline.context_resolver.get_context().active_doctor)

    def test_failed_grounding_does_not_commit_candidate_context(self):
        resources = make_resources()
        grounding = RejectNextGrounding()
        pipeline = make_pipeline(resources, grounding)

        pipeline.run("Tell me about Dr. Alice Adams")
        before = pipeline.context_resolver.get_context()
        grounding.reject = True
        resources.retriever.override_document = document(
            clinic="Invalid Clinic",
            doctor="Dr. Invalid Result",
        )

        result = pipeline.run("Where is Clinic Beta?")
        after = pipeline.context_resolver.get_context()

        self.assertFalse(result.grounded)
        self.assertEqual(after, before)

    def test_first_turn_explicit_entity_is_resolved_before_retrieval(self):
        resources = make_resources()
        pipeline = make_pipeline(resources)

        pipeline.run("Is Alice Adams available today?")
        retrieval_call = resources.retriever.calls[-1]

        self.assertEqual(retrieval_call["doctor_name"], "Dr. Alice Adams")
        self.assertIsNone(retrieval_call["clinic_name"])

    def test_resolution_records_explicit_and_inherited_provenance(self):
        resolver = ContextResolver(EntityCatalog.from_clinic_data(CLINIC_DATA))
        explicit = resolver.resolve("Tell me about Dr. Alice Adams")
        resolver.commit(explicit)
        inherited = resolver.resolve("When is she available?")

        self.assertTrue(explicit.doctor.is_explicit)
        self.assertTrue(inherited.doctor.is_inherited)
        self.assertTrue(inherited.clinic.is_inherited)

    def test_pipeline_dependency_injection_does_not_construct_groq(self):
        resources = make_resources()
        resolver = ContextResolver(resources.entity_catalog)
        generator = FakeGenerator()
        guard = AllowGuard()
        with patch("config.settings.GROQ_API_KEY", None):
            pipeline = RAGPipeline(
                retriever=resources.retriever,
                context_resolver=resolver,
                intent_classifier=IntentClassifier(),
                grounding_validator=GroundingValidator(),
                generator=generator,
                input_guard=guard,
                medical_guard=guard,
                output_guard=guard,
            )

        result = pipeline.run("Where is Clinic Alpha?")

        self.assertTrue(result.grounded)
        self.assertEqual(result.answer, "Deterministic answer")
        self.assertEqual(len(generator.calls), 1)


if __name__ == "__main__":
    unittest.main()
