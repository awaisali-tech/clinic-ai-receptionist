from __future__ import annotations

from dataclasses import dataclass

from conversation.entity_resolver import EntitySource
from conversation.intent_classifier import InformationNeed, information_need_spec
from ingestion.document_builder import RAGDocument
from retrieval.structured_retriever import StructuredRetriever
from retrieval.vector_retriever import VectorRetriever


@dataclass(frozen=True)
class RankingConfig:
    explicit_entity_weight: float = 0.30
    inherited_entity_weight: float = 0.18
    document_type_weight: float = 0.35
    semantic_weight: float = 0.30
    structured_source_weight: float = 0.05
    document_type_mismatch_penalty: float = 0.25
    explicit_entity_mismatch_penalty: float = 0.30
    inherited_entity_mismatch_penalty: float = 0.10
    vector_pool_multiplier: int = 4


@dataclass(frozen=True)
class RetrievalCandidate:
    document: RAGDocument
    semantic_score: float | None
    structured_match: bool
    entity_match: bool
    document_type_match: bool
    sources: tuple[str, ...]
    final_score: float
    explicit_entity_match: bool = False
    inherited_entity_match: bool = False

    @property
    def score(self) -> float:
        """Backward-compatible access to the deterministic final score."""

        return self.final_score

    @property
    def source(self) -> str:
        """Backward-compatible compact provenance label."""

        return "+".join(self.sources)


HybridResult = RetrievalCandidate


@dataclass
class _CandidateAccumulator:
    document: RAGDocument
    semantic_score: float | None = None
    structured_match: bool = False


class HybridRetriever:
    """Fuse structured and semantic candidates and rank them transparently."""

    def __init__(
        self,
        structured_retriever: StructuredRetriever,
        vector_retriever: VectorRetriever,
        ranking: RankingConfig | None = None,
    ):
        self.structured_retriever = structured_retriever
        self.vector_retriever = vector_retriever
        self.ranking = ranking or RankingConfig()

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
    ) -> list[RetrievalCandidate]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        spec = information_need_spec(information_need)
        compatible_types = (
            frozenset({document_type})
            if document_type
            else spec.document_types
        )
        entity_values = {
            "doctor": doctor_name,
            "clinic": clinic_name,
            "specialization": specialization,
        }
        entity_sources = {
            "doctor": doctor_source,
            "clinic": clinic_source,
            "specialization": specialization_source,
        }
        relevant_filters = {
            field: entity_values[field]
            for field in spec.entity_fields
            if entity_values[field]
        }

        accumulators: dict[str, _CandidateAccumulator] = {}

        if relevant_filters:
            entity_documents = self.structured_retriever.retrieve(
                doctor_name=relevant_filters.get("doctor"),
                clinic_name=relevant_filters.get("clinic"),
                specialization=relevant_filters.get("specialization"),
            )
            self._add_structured(accumulators, entity_documents)

        typed_documents = self.structured_retriever.retrieve(
            doctor_name=relevant_filters.get("doctor"),
            clinic_name=relevant_filters.get("clinic"),
            specialization=relevant_filters.get("specialization"),
            document_types=compatible_types,
        )
        self._add_structured(accumulators, typed_documents)

        vector_candidate_count = max(
            top_k * self.ranking.vector_pool_multiplier,
            top_k,
        )
        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=vector_candidate_count,
            candidate_pool_size=vector_candidate_count,
        )
        for result in vector_results:
            key = self._document_key(result.document)
            accumulator = accumulators.setdefault(
                key,
                _CandidateAccumulator(document=result.document),
            )
            if (
                accumulator.semantic_score is None
                or result.score > accumulator.semantic_score
            ):
                accumulator.semantic_score = result.score

        candidates = [
            self._to_candidate(
                accumulator,
                compatible_types=compatible_types,
                entity_values=entity_values,
                entity_sources=entity_sources,
                relevant_fields=spec.entity_fields,
            )
            for accumulator in accumulators.values()
        ]
        candidates.sort(
            key=lambda candidate: (
                -candidate.final_score,
                -(candidate.semantic_score or -1.0),
                self._document_key(candidate.document),
            )
        )
        return candidates[:top_k]

    def _to_candidate(
        self,
        accumulator: _CandidateAccumulator,
        *,
        compatible_types: frozenset[str],
        entity_values: dict[str, str | None],
        entity_sources: dict[str, EntitySource | None],
        relevant_fields: tuple[str, ...],
    ) -> RetrievalCandidate:
        metadata = accumulator.document.metadata
        document_type_match = metadata.get("document_type") in compatible_types

        requested_fields = [
            field
            for field in relevant_fields
            if entity_values.get(field)
        ]
        field_matches = {
            field: self._entity_matches(
                metadata,
                field,
                entity_values[field],
            )
            for field in requested_fields
        }
        entity_match = all(field_matches.values())
        explicit_requested = any(
            entity_sources.get(field) == "explicit"
            for field in requested_fields
        )
        inherited_requested = any(
            entity_sources.get(field) == "inherited"
            for field in requested_fields
        )
        explicit_entity_match = entity_match and explicit_requested
        inherited_entity_match = entity_match and inherited_requested

        score = 0.0
        if document_type_match:
            score += self.ranking.document_type_weight
        else:
            score -= self.ranking.document_type_mismatch_penalty

        if explicit_entity_match:
            score += self.ranking.explicit_entity_weight
        elif explicit_requested and not entity_match:
            score -= self.ranking.explicit_entity_mismatch_penalty

        if inherited_entity_match:
            score += self.ranking.inherited_entity_weight
        elif inherited_requested and not entity_match:
            score -= self.ranking.inherited_entity_mismatch_penalty

        if accumulator.semantic_score is not None:
            bounded_semantic_score = min(
                max(accumulator.semantic_score, 0.0),
                1.0,
            )
            score += self.ranking.semantic_weight * bounded_semantic_score

        if accumulator.structured_match:
            score += self.ranking.structured_source_weight

        sources = []
        if accumulator.structured_match:
            sources.append("structured")
        if accumulator.semantic_score is not None:
            sources.append("vector")

        return RetrievalCandidate(
            document=accumulator.document,
            semantic_score=accumulator.semantic_score,
            structured_match=accumulator.structured_match,
            entity_match=entity_match,
            document_type_match=document_type_match,
            sources=tuple(sources),
            final_score=round(score, 4),
            explicit_entity_match=explicit_entity_match,
            inherited_entity_match=inherited_entity_match,
        )

    @classmethod
    def _add_structured(
        cls,
        accumulators: dict[str, _CandidateAccumulator],
        documents: list[RAGDocument],
    ) -> None:
        for document in documents:
            key = cls._document_key(document)
            accumulator = accumulators.setdefault(
                key,
                _CandidateAccumulator(document=document),
            )
            accumulator.structured_match = True

    @staticmethod
    def _entity_matches(
        metadata: dict,
        field: str,
        expected: str | None,
    ) -> bool:
        if expected is None:
            return True

        metadata_fields = {
            "doctor": ("doctor_name",),
            "clinic": ("clinic_name",),
            "specialization": ("specialization", "service_name"),
        }[field]
        expected_value = expected.strip().casefold()
        return any(
            str(metadata.get(metadata_field, "")).strip().casefold()
            == expected_value
            for metadata_field in metadata_fields
        )

    @staticmethod
    def _document_key(document: RAGDocument) -> str:
        return str(
            document.metadata.get("document_id")
            or f"{document.metadata.get('document_type')}:{document.text}"
        )
