from __future__ import annotations

import math
from dataclasses import dataclass

from conversation.entity_resolver import normalize_entity_text
from conversation.intent_classifier import InformationNeed, information_need_spec
from ingestion.document_builder import RAGDocument
from retrieval.hybrid_retriever import RetrievalCandidate


@dataclass(frozen=True)
class GroundingConfig:
    minimum_results: int = 1
    minimum_semantic_score: float = 0.35


@dataclass(frozen=True)
class GroundingResult:
    """Answerability decision and the ranked candidates accepted as evidence."""

    is_grounded: bool
    evidence: list[RetrievalCandidate]
    reason: str

    @property
    def documents(self) -> list[RAGDocument]:
        return [candidate.document for candidate in self.evidence]


class GroundingValidator:
    """Validate entity-correct, field-aware evidence for an information need."""

    STOP_WORDS = frozenset(
        {
            "a",
            "about",
            "an",
            "and",
            "are",
            "at",
            "can",
            "could",
            "do",
            "does",
            "for",
            "have",
            "how",
            "i",
            "in",
            "is",
            "it",
            "me",
            "of",
            "on",
            "please",
            "tell",
            "the",
            "they",
            "to",
            "what",
            "when",
            "where",
            "which",
            "with",
            "you",
            "your",
        }
    )

    def __init__(
        self,
        minimum_results: int = 1,
        config: GroundingConfig | None = None,
    ):
        self.config = config or GroundingConfig(
            minimum_results=minimum_results
        )

    def validate(
        self,
        query: str,
        results: list[RetrievalCandidate],
        doctor_name: str | None = None,
        clinic_name: str | None = None,
        specialization: str | None = None,
        information_need: InformationNeed = InformationNeed.GENERAL,
    ) -> GroundingResult:
        if not results:
            return GroundingResult(
                is_grounded=False,
                evidence=[],
                reason="No relevant information was retrieved.",
            )

        entity_terms = self._entity_terms(
            doctor_name,
            clinic_name,
            specialization,
        )
        answerable = [
            candidate
            for candidate in results
            if self._is_answerable(
                candidate,
                query=query,
                information_need=information_need,
                entity_terms=entity_terms,
            )
        ]

        if len(answerable) < self.config.minimum_results:
            return GroundingResult(
                is_grounded=False,
                evidence=[],
                reason=(
                    "Retrieved candidates do not contain answerable "
                    f"{information_need.value} evidence."
                ),
            )

        return GroundingResult(
            is_grounded=True,
            evidence=answerable,
            reason=(
                "Answerable evidence was retrieved for "
                f"{information_need.value}."
            ),
        )

    def _is_answerable(
        self,
        candidate: RetrievalCandidate,
        *,
        query: str,
        information_need: InformationNeed,
        entity_terms: set[str],
    ) -> bool:
        spec = information_need_spec(information_need)
        metadata = candidate.document.metadata

        if not candidate.document_type_match or not candidate.entity_match:
            return False

        information_types = set(metadata.get("information_types", ()))
        if (
            spec.information_type is not None
            and spec.information_type not in information_types
        ):
            return False

        if spec.answer_fields and not any(
            metadata.get(field)
            for field in spec.answer_fields
        ):
            return False

        if candidate.semantic_score is not None and (
            not math.isfinite(candidate.semantic_score)
            or candidate.semantic_score < self.config.minimum_semantic_score
        ):
            return False

        if not candidate.structured_match and candidate.semantic_score is None:
            return False

        if spec.requires_semantic_support:
            if (
                candidate.semantic_score is None
                or candidate.semantic_score < self.config.minimum_semantic_score
            ):
                return False
            if not self._has_lexical_support(
                query,
                candidate.document,
                entity_terms,
            ):
                return False

        return True

    def _has_lexical_support(
        self,
        query: str,
        document: RAGDocument,
        entity_terms: set[str],
    ) -> bool:
        query_terms = set(normalize_entity_text(query).split())
        query_terms -= self.STOP_WORDS
        query_terms -= entity_terms
        if not query_terms:
            return False

        document_terms = set(normalize_entity_text(document.text).split())
        overlap = query_terms & document_terms
        required_overlap = min(2, len(query_terms))
        return len(overlap) >= required_overlap

    @staticmethod
    def _entity_terms(*values: str | None) -> set[str]:
        terms: set[str] = set()
        for value in values:
            if value:
                terms.update(normalize_entity_text(value).split())
        return terms
