import unittest

from conversation.entity_resolver import ContextResolver, EntityCatalog
from conversation.intent_classifier import IntentClassifier
from ingestion.document_builder import RAGDocument
from generation.prompt_builder import PromptBuilder
from orchestration.pipeline import AnswerStatus, RAGPipeline
from retrieval.grounding_validator import GroundingValidator
from retrieval.hybrid_retriever import RetrievalCandidate


CATALOG = EntityCatalog.from_clinic_data(
    {
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
)


def doctor_candidate(name, clinic, specialization):
    document = RAGDocument(
        text=(
            f"Doctor: {name}\nClinic: {clinic}\n"
            f"Specialization: {specialization}\nAvailability: Mon-Fri"
        ),
        metadata={
            "document_id": f"doctor:{name}",
            "document_type": "doctor",
            "information_types": (
                "doctor_information",
                "doctor_availability",
            ),
            "doctor_name": name,
            "clinic_name": clinic,
            "specialization": specialization,
            "availability": "Mon-Fri",
        },
    )
    return RetrievalCandidate(
        document=document,
        semantic_score=0.9,
        structured_match=True,
        entity_match=True,
        document_type_match=True,
        sources=("structured", "vector"),
        final_score=0.95,
        explicit_entity_match=True,
    )


class ScenarioRetriever:
    def __init__(self):
        self.return_results = True
        self.calls = []

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
        self.calls.append((query, information_need))
        if not self.return_results:
            return []
        if doctor_name == "Dr. Bob Brown":
            return [
                doctor_candidate(
                    "Dr. Bob Brown",
                    "Clinic Beta",
                    "Dermatology",
                )
            ]
        return [
            doctor_candidate(
                "Dr. Alice Adams",
                "Clinic Alpha",
                "Pediatrics",
            )
        ]


class FakeGenerator:
    def __init__(self):
        self.raise_error = False
        self.calls = []

    def generate(self, query, evidence, conversation_context=None):
        self.calls.append((query, evidence, conversation_context))
        if self.raise_error:
            raise RuntimeError("provider unavailable")
        return "Accepted administrative answer."


class ToggleGuard:
    def __init__(self, allowed=True, response="accepted"):
        self.allowed = allowed
        self.response = response

    def check(self, value):
        return self.allowed, value if self.allowed else self.response


def make_pipeline():
    retriever = ScenarioRetriever()
    generator = FakeGenerator()
    output_guard = ToggleGuard()
    pipeline = RAGPipeline(
        retriever=retriever,
        context_resolver=ContextResolver(CATALOG),
        intent_classifier=IntentClassifier(),
        grounding_validator=GroundingValidator(),
        generator=generator,
        input_guard=ToggleGuard(),
        medical_guard=ToggleGuard(),
        output_guard=output_guard,
    )
    return pipeline, retriever, generator, output_guard


class PipelineStatusTests(unittest.TestCase):
    def test_answered_status_commits_and_preserves_structured_evidence(self):
        pipeline, _, generator, _ = make_pipeline()

        result = pipeline.run("When is Dr. Alice Adams available?")

        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertTrue(result.grounded)
        self.assertEqual(result.evidence[0].document.metadata["doctor_name"], "Dr. Alice Adams")
        self.assertIs(generator.calls[0][1][0], result.evidence[0])
        self.assertEqual(
            pipeline.context_resolver.get_context().active_doctor,
            "Dr. Alice Adams",
        )

    def test_no_evidence_does_not_commit_new_candidate_context(self):
        pipeline, retriever, _, _ = make_pipeline()
        pipeline.run("When is Dr. Alice Adams available?")
        before = pipeline.context_resolver.get_context()
        retriever.return_results = False

        result = pipeline.run("When is Dr. Bob Brown available?")

        self.assertEqual(result.status, AnswerStatus.NO_EVIDENCE)
        self.assertEqual(pipeline.context_resolver.get_context(), before)

    def test_generation_failure_does_not_commit(self):
        pipeline, _, generator, _ = make_pipeline()
        pipeline.run("When is Dr. Alice Adams available?")
        before = pipeline.context_resolver.get_context()
        generator.raise_error = True

        result = pipeline.run("When is Dr. Bob Brown available?")

        self.assertEqual(result.status, AnswerStatus.GENERATION_FAILED)
        self.assertEqual(pipeline.context_resolver.get_context(), before)

    def test_output_rejection_is_explicit_and_does_not_commit(self):
        pipeline, _, _, output_guard = make_pipeline()
        pipeline.run("When is Dr. Alice Adams available?")
        before = pipeline.context_resolver.get_context()
        output_guard.allowed = False
        output_guard.response = "Safety fallback."

        result = pipeline.run("When is Dr. Bob Brown available?")

        self.assertEqual(result.status, AnswerStatus.OUTPUT_REJECTED)
        self.assertEqual(result.answer, "Safety fallback.")
        self.assertEqual(pipeline.context_resolver.get_context(), before)

    def test_input_and_medical_refusals_have_distinct_statuses(self):
        pipeline, _, _, _ = make_pipeline()
        pipeline.input_guard = ToggleGuard(False, "Invalid input.")
        input_result = pipeline.run("")

        pipeline.input_guard = ToggleGuard()
        pipeline.medical_guard = ToggleGuard(False, "Medical safety response.")
        medical_result = pipeline.run("Please diagnose me")

        self.assertEqual(input_result.status, AnswerStatus.INPUT_REJECTED)
        self.assertEqual(medical_result.status, AnswerStatus.SAFETY_REFUSAL)
        self.assertIsNone(pipeline.context_resolver.get_context().active_doctor)


class EvidencePromptTests(unittest.TestCase):
    def test_prompt_preserves_evidence_metadata_and_provenance(self):
        evidence = doctor_candidate(
            "Dr. Alice Adams",
            "Clinic Alpha",
            "Pediatrics",
        )

        messages = PromptBuilder().build(
            query="When is Dr. Alice Adams available?",
            evidence=[evidence],
        )
        prompt = messages[1]["content"]

        self.assertIn("Type: doctor", prompt)
        self.assertIn("Clinic: Clinic Alpha", prompt)
        self.assertIn("Doctor: Dr. Alice Adams", prompt)
        self.assertIn("Retrieved via: structured, vector", prompt)
        self.assertNotIn("0.95", prompt)


if __name__ == "__main__":
    unittest.main()
