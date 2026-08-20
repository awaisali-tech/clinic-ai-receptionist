from __future__ import annotations

import math
from dataclasses import dataclass

from embeddings.embedder import Embedder
from ingestion.document_builder import RAGDocument
from retrieval.faiss_store import FAISSStore


@dataclass(frozen=True)
class VectorSearchConfig:
    minimum_score: float = 0.35
    candidate_pool_multiplier: int = 4


@dataclass(frozen=True)
class RetrievalResult:
    document: RAGDocument
    score: float

    @property
    def semantic_score(self) -> float:
        return self.score


class VectorRetriever:
    """Semantic candidate retrieval using normalized embeddings and FAISS."""

    def __init__(
        self,
        store: FAISSStore,
        embedder: Embedder,
        config: VectorSearchConfig | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or VectorSearchConfig()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        candidate_pool_size: int | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        threshold = (
            self.config.minimum_score
            if min_score is None
            else min_score
        )
        if not math.isfinite(threshold):
            raise ValueError("min_score must be finite.")

        pool_size = candidate_pool_size or max(
            top_k,
            top_k * self.config.candidate_pool_multiplier,
        )
        if pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive.")

        query_embedding = self.embedder.embed_query(query)
        results = self.store.search(query_embedding, top_k=pool_size)

        accepted: list[RetrievalResult] = []
        for document, score in results:
            if not math.isfinite(score) or score < threshold:
                continue
            accepted.append(RetrievalResult(document=document, score=score))
            if len(accepted) == top_k:
                break

        return accepted
