from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from conversation.entity_resolver import ContextResolver, EntitySource, QueryResolution
from conversation.intent_classifier import InformationNeed
from generation.provider_errors import (
    GenerationFailure,
    ProviderError,
    ProviderFailureKind,
)
from privacy.models import PrivacyProcessingError, PrivacyResult
from privacy.redactor import PrivacyProcessor
from retrieval.grounding_validator import GroundingResult
from retrieval.hybrid_retriever import RetrievalCandidate
from safety.models import SafetyCategory, SafetyDecision


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    NO_EVIDENCE = "no_evidence"
    INPUT_REJECTED = "input_rejected"
    SAFETY_REFUSAL = "safety_refusal"
    GENERATION_FAILED = "generation_failed"
    OUTPUT_REJECTED = "output_rejected"


class RetrieverDependency(Protocol):
    def retrieve(
        self,
        query: str,
        doctor_name: str | None = None,
        clinic_name: str | None = None,
        specialization: str | None = None,
        document_type: str | None = None,
        top_k: int = 5,
        information_need: InformationNeed = InformationNeed.GENERAL,
        doctor_source: EntitySource | None = None,
        clinic_source: EntitySource | None = None,
        specialization_source: EntitySource | None = None,
    ) -> list[RetrievalCandidate]: ...


class GroundingDependency(Protocol):
    def validate(
        self,
        query: str,
        results: list[RetrievalCandidate],
        doctor_name: str | None = None,
        clinic_name: str | None = None,
        specialization: str | None = None,
        information_need: InformationNeed = InformationNeed.GENERAL,
    ) -> GroundingResult: ...


class GeneratorDependency(Protocol):
    def generate(
        self,
        query: str,
        evidence: list[Any],
        conversation_context: dict[str, str | None] | None = None,
    ) -> str: ...


class GuardDependency(Protocol):
    def check(self, value: str) -> tuple[bool, str]: ...


class IntentDependency(Protocol):
    def classify(
        self,
        query: str,
        resolution: QueryResolution | None = None,
    ) -> InformationNeed: ...


@dataclass
class PipelineResult:
    """Final answer outcome plus ranked and accepted retrieval evidence."""

    query: str
    results: list[RetrievalCandidate]
    status: AnswerStatus = AnswerStatus.NO_EVIDENCE
    evidence: list[RetrievalCandidate] = field(default_factory=list)
    information_need: InformationNeed = InformationNeed.GENERAL
    doctor: str | None = None
    clinic: str | None = None
    specialization: str | None = None
    is_follow_up: bool = False
    grounded: bool = False
    grounding_reason: str = ""
    answer: str = ""
    safety_category: SafetyCategory = SafetyCategory.SAFE_ADMINISTRATIVE
    privacy: PrivacyResult | None = None
    generation_failure: GenerationFailure | None = None

    @property
    def redaction_applied(self) -> bool:
        return bool(self.privacy and self.privacy.redaction_applied)


class RAGPipeline:
    """Provider-independent orchestration for one conversation session."""

    def __init__(
        self,
        retriever: RetrieverDependency,
        context_resolver: ContextResolver,
        intent_classifier: IntentDependency,
        grounding_validator: GroundingDependency,
        generator: GeneratorDependency,
        input_guard: GuardDependency,
        medical_guard: GuardDependency,
        output_guard: GuardDependency,
        privacy_processor: PrivacyProcessor | None = None,
    ):
        self.retriever = retriever
        self.context_resolver = context_resolver
        self.intent_classifier = intent_classifier
        self.grounding_validator = grounding_validator
        self.generator = generator
        self.input_guard = input_guard
        self.medical_guard = medical_guard
        self.output_guard = output_guard
        self.privacy_processor = privacy_processor or PrivacyProcessor()

    def run(self, query: str, top_k: int = 5) -> PipelineResult:
        """Process one user query and commit context only for accepted answers."""

        input_decision = self._guard_decision(
            self.input_guard,
            query,
            blocked_category=SafetyCategory.INVALID_INPUT,
        )
        if not input_decision.allowed:
            privacy = self._diagnostic_privacy(query)
            return PipelineResult(
                query=privacy.provider_query,
                results=[],
                status=AnswerStatus.INPUT_REJECTED,
                grounding_reason=input_decision.reason,
                answer=input_decision.user_response or input_decision.reason,
                safety_category=input_decision.category,
                privacy=privacy,
            )

        medical_decision = self._guard_decision(
            self.medical_guard,
            query,
            blocked_category=SafetyCategory.MEDICAL_ADVICE,
        )
        if not medical_decision.allowed:
            privacy = self._diagnostic_privacy(query)
            return PipelineResult(
                query=privacy.provider_query,
                results=[],
                status=AnswerStatus.SAFETY_REFUSAL,
                grounding_reason=medical_decision.reason,
                answer=(
                    medical_decision.user_response
                    or "This request cannot be handled by the clinic assistant."
                ),
                safety_category=medical_decision.category,
                privacy=privacy,
            )

        resolution = self.context_resolver.resolve(query)
        if resolution.reset_requested:
            self.context_resolver.reset()

        information_need = self.intent_classifier.classify(query, resolution)
        candidate_context = self.context_resolver.candidate_context(resolution)
        doctor = resolution.doctor.value
        clinic = resolution.clinic.value
        specialization = resolution.specialization.value

        protected_entities = tuple(
            value
            for value in (doctor, clinic, specialization)
            if value is not None
        )
        try:
            privacy = self.privacy_processor.process(
                query,
                protected_entities=protected_entities,
            )
        except PrivacyProcessingError:
            return PipelineResult(
                query="[REDACTED]",
                results=[],
                status=AnswerStatus.INPUT_REJECTED,
                grounding_reason="Privacy processing could not safely minimize the request.",
                answer=(
                    "I couldn't safely process personal information in that request. "
                    "Please remove it and ask only for the clinic information you need."
                ),
                safety_category=SafetyCategory.SENSITIVE_DATA,
                privacy=PrivacyResult(True, True, "[REDACTED]"),
            )

        retrieval_query = privacy.retrieval_query
        provider_query = privacy.provider_query
        request_category = (
            SafetyCategory.SENSITIVE_DATA
            if privacy.sensitive_data_detected
            else SafetyCategory.SAFE_ADMINISTRATIVE
        )

        results = self.retriever.retrieve(
            query=retrieval_query,
            doctor_name=doctor,
            clinic_name=clinic,
            specialization=specialization,
            top_k=top_k,
            information_need=information_need,
            doctor_source=resolution.doctor.source,
            clinic_source=resolution.clinic.source,
            specialization_source=resolution.specialization.source,
        )

        grounding = self.grounding_validator.validate(
            query=retrieval_query,
            results=results,
            doctor_name=doctor,
            clinic_name=clinic,
            specialization=specialization,
            information_need=information_need,
        )

        result_fields = {
            "query": provider_query,
            "results": results,
            "evidence": grounding.evidence,
            "information_need": information_need,
            "doctor": candidate_context.active_doctor,
            "clinic": candidate_context.active_clinic,
            "specialization": candidate_context.active_specialization,
            "is_follow_up": resolution.is_follow_up,
            "grounding_reason": grounding.reason,
            "safety_category": request_category,
            "privacy": privacy,
        }

        if not grounding.is_grounded:
            return PipelineResult(
                **result_fields,
                status=AnswerStatus.NO_EVIDENCE,
                grounded=False,
                answer=(
                    "I'm sorry, but that information is "
                    "not available in the clinic records."
                ),
            )

        try:
            answer = self.generator.generate(
                query=provider_query,
                evidence=grounding.evidence,
                conversation_context={
                    "doctor": candidate_context.active_doctor,
                    "clinic": candidate_context.active_clinic,
                    "specialization": (
                        candidate_context.active_specialization
                    ),
                },
            )
        except ProviderError as error:
            return PipelineResult(
                **result_fields,
                status=AnswerStatus.GENERATION_FAILED,
                grounded=True,
                answer=self._generation_failure_message(error.kind),
                generation_failure=error.failure,
            )
        except Exception:
            failure = GenerationFailure(
                kind=ProviderFailureKind.UNKNOWN,
                retryable=False,
                attempts=0,
            )
            return PipelineResult(
                **result_fields,
                status=AnswerStatus.GENERATION_FAILED,
                grounded=True,
                answer=self._generation_failure_message(failure.kind),
                generation_failure=failure,
            )

        output_decision = self._guard_decision(
            self.output_guard,
            answer,
            blocked_category=SafetyCategory.MEDICAL_ADVICE,
            allowed_response=answer,
        )
        final_answer = output_decision.user_response or answer.strip()
        if not output_decision.allowed:
            return PipelineResult(
                **{
                    **result_fields,
                    "safety_category": output_decision.category,
                },
                status=AnswerStatus.OUTPUT_REJECTED,
                grounded=True,
                answer=final_answer,
            )

        committed = self.context_resolver.commit(resolution)
        answered_fields = {
            **result_fields,
            "doctor": committed.active_doctor,
            "clinic": committed.active_clinic,
            "specialization": committed.active_specialization,
        }
        return PipelineResult(
            **answered_fields,
            status=AnswerStatus.ANSWERED,
            grounded=True,
            answer=final_answer,
        )

    def reset_conversation(self) -> None:
        self.context_resolver.reset()

    def _diagnostic_privacy(self, query: object) -> PrivacyResult:
        """Return a safe query representation for blocked-result diagnostics."""

        if not isinstance(query, str):
            return PrivacyResult(False, False, "")
        try:
            return self.privacy_processor.process(query)
        except PrivacyProcessingError:
            return PrivacyResult(True, True, "[REDACTED]")

    @staticmethod
    def _generation_failure_message(kind: ProviderFailureKind) -> str:
        if kind in {
            ProviderFailureKind.TIMEOUT,
            ProviderFailureKind.CONNECTION,
            ProviderFailureKind.RATE_LIMIT,
            ProviderFailureKind.SERVICE_UNAVAILABLE,
        }:
            return (
                "I'm unable to complete that request right now because the "
                "response service is temporarily unavailable. Please try again."
            )
        if kind in {
            ProviderFailureKind.CONFIGURATION,
            ProviderFailureKind.AUTHENTICATION,
            ProviderFailureKind.PERMISSION,
        }:
            return (
                "I'm unable to complete that request because the response "
                "service is not available. Please try again later."
            )
        return (
            "I couldn't generate an answer right now. "
            "Please try again shortly."
        )

    @staticmethod
    def _guard_decision(
        guard: GuardDependency,
        value: str,
        *,
        blocked_category: SafetyCategory,
        allowed_response: str | None = None,
    ) -> SafetyDecision:
        """Use typed guards while retaining injected tuple-guard compatibility."""

        evaluate = getattr(guard, "evaluate", None)
        if callable(evaluate):
            decision = evaluate(value)
            if isinstance(decision, SafetyDecision):
                return decision

        allowed, response = guard.check(value)
        if allowed:
            return SafetyDecision(
                True,
                SafetyCategory.SAFE_ADMINISTRATIVE,
                "Guard accepted the value.",
                allowed_response,
            )
        return SafetyDecision(
            False,
            blocked_category,
            response,
            response,
        )
