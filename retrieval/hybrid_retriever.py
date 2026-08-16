from dataclasses import dataclass

from ingestion.document_builder import RAGDocument
from retrieval.vector_retriever import VectorRetriever
from retrieval.structured_retriever import StructuredRetriever


@dataclass
class HybridResult:
    document: RAGDocument
    score: float
    source: str


class HybridRetriever:
    """
    Combines structured and semantic retrieval.

    Structured retrieval has priority when an exact
    entity is available. Vector retrieval provides
    semantic fallback.
    """

    def __init__(
        self,
        structured_retriever: StructuredRetriever,
        vector_retriever: VectorRetriever,
    ):
        self.structured_retriever = structured_retriever
        self.vector_retriever = vector_retriever

    def retrieve(
        self,
        query: str,
        doctor_name: str | None = None,
        clinic_name: str | None = None,
        specialization: str | None = None,
        document_type: str | None = None,
        top_k: int = 5,
    ) -> list[HybridResult]:

        # --------------------------------------------------
        # 1. Try structured retrieval first
        # --------------------------------------------------

        has_structured_filter = any(
            [
                doctor_name,
                clinic_name,
                specialization,
                document_type,
            ]
        )

        if has_structured_filter:

            structured_results = (
                self.structured_retriever.retrieve(
                    doctor_name=doctor_name,
                    clinic_name=clinic_name,
                    specialization=specialization,
                    document_type=document_type,
                )
            )

            if structured_results:

                return [
                    HybridResult(
                        document=document,
                        score=1.0,
                        source="structured",
                    )
                    for document in structured_results[:top_k]
                ]

        # --------------------------------------------------
        # 2. Fall back to semantic retrieval
        # --------------------------------------------------

        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        return [
            HybridResult(
                document=result.document,
                score=result.score,
                source="vector",
            )
            for result in vector_results
        ]