from dataclasses import dataclass

from retrieval.hybrid_retriever import HybridRetriever, HybridResult
from retrieval.grounding_validator import GroundingValidator

from conversation.entity_resolver import ContextResolver

from generation.generator import Generator

from safety.input_guard import InputGuard
from safety.medical_guard import MedicalGuard
from safety.output_guard import OutputGuard


@dataclass
class PipelineResult:
    """
    Final result returned by the RAG pipeline.
    """

    query: str
    results: list[HybridResult]

    doctor: str | None = None
    clinic: str | None = None
    specialization: str | None = None

    is_follow_up: bool = False

    grounded: bool = False
    grounding_reason: str = ""

    answer: str = ""


class RAGPipeline:
    """
    Main orchestration layer for the clinic AI receptionist.

    Components are injected into the pipeline so each component
    can be tested independently.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        context_resolver: ContextResolver,
    ):
        self.retriever = retriever
        self.context_resolver = context_resolver

        self.grounding_validator = GroundingValidator()

        self.generator = Generator()

        self.input_guard = InputGuard()
        self.medical_guard = MedicalGuard()
        self.output_guard = OutputGuard()

    def run(
        self,
        query: str,
        top_k: int = 5,
    ) -> PipelineResult:
        """
        Process one user query.
        """

        # ==================================================
        # 1. INPUT SAFETY
        # ==================================================

        allowed, reason = self.input_guard.check(query)

        if not allowed:
            return PipelineResult(
                query=query,
                results=[],
                grounded=False,
                grounding_reason=reason,
                answer=reason,
            )

        # ==================================================
        # 2. MEDICAL SAFETY
        # ==================================================

        allowed, medical_response = (
            self.medical_guard.check(query)
        )

        if not allowed:
            return PipelineResult(
                query=query,
                results=[],
                grounded=False,
                grounding_reason="Medical advice request.",
                answer=medical_response,
            )

        # ==================================================
        # 3. RESOLVE CONVERSATION CONTEXT
        # ==================================================

        context = self.context_resolver.resolve(query)

        doctor = context.get("doctor")
        clinic = context.get("clinic")
        specialization = context.get("specialization")

        is_follow_up = context.get(
            "is_follow_up",
            False,
        )

        # ==================================================
        # 4. RETRIEVAL
        # ==================================================

        results = self.retriever.retrieve(
            query=query,
            doctor_name=doctor,
            clinic_name=clinic,
            specialization=specialization,
            top_k=top_k,
        )

        # ==================================================
        # 5. GROUNDING VALIDATION
        # ==================================================

        grounding = self.grounding_validator.validate(
            query=query,
            results=results,
            doctor_name=doctor,
            clinic_name=clinic,
        )

        # ==================================================
        # 6. UPDATE CONTEXT FROM RETRIEVAL
        # ==================================================

        if results:

            best_document = results[0].document

            retrieved_doctor = (
                best_document.metadata.get(
                    "doctor_name"
                )
            )

            retrieved_clinic = (
                best_document.metadata.get(
                    "clinic_name"
                )
            )

            retrieved_specialization = (
                best_document.metadata.get(
                    "specialization"
                )
            )

            doctor = doctor or retrieved_doctor

            clinic = clinic or retrieved_clinic

            specialization = (
                specialization
                or retrieved_specialization
            )

            self.context_resolver.update(
                doctor=doctor,
                clinic=clinic,
                specialization=specialization,
            )

        # ==================================================
        # 7. NO GROUNDING
        # ==================================================

        if not grounding.is_grounded:

            answer = (
                "I'm sorry, but that information is "
                "not available in the clinic records."
            )

            return PipelineResult(
                query=query,
                results=results,
                doctor=doctor,
                clinic=clinic,
                specialization=specialization,
                is_follow_up=is_follow_up,
                grounded=False,
                grounding_reason=grounding.reason,
                answer=answer,
            )

        # ==================================================
        # 8. PREPARE EVIDENCE
        # ==================================================

        evidence = [
            document.text
            for document in grounding.evidence
        ]

        # ==================================================
        # 9. GENERATE ANSWER
        # ==================================================

        answer = self.generator.generate(
            query=query,
            evidence=evidence,
            conversation_context={
                "doctor": doctor,
                "clinic": clinic,
                "specialization": specialization,
            },
        )

        # ==================================================
        # 10. OUTPUT SAFETY
        # ==================================================

        output_allowed, final_answer = (
            self.output_guard.check(answer)
        )

        answer = final_answer

        # ==================================================
        # 11. RETURN RESULT
        # ==================================================

        return PipelineResult(
            query=query,
            results=results,
            doctor=doctor,
            clinic=clinic,
            specialization=specialization,
            is_follow_up=is_follow_up,
            grounded=True,
            grounding_reason=grounding.reason,
            answer=answer,
        )

    def reset_conversation(self):
        """
        Clear the active conversation context.
        """

        self.context_resolver.context.reset()