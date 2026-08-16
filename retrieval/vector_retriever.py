from dataclasses import dataclass

from embeddings.embedder import Embedder
from ingestion.document_builder import RAGDocument
from retrieval.faiss_store import FAISSStore


@dataclass
class RetrievalResult:
    document: RAGDocument
    score: float


class VectorRetriever:
    """
    Semantic retrieval using Sentence Transformers + FAISS.
    """

    def __init__(
        self,
        store: FAISSStore,
        embedder: Embedder,
    ):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> list[RetrievalResult]:

        query_embedding = self.embedder.embed_query(query)

        results = self.store.search(
            query_embedding,
            top_k=top_k,
        )

        filtered_results = []

        for document, score in results:

            if score >= min_score:
                filtered_results.append(
                    RetrievalResult(
                        document=document,
                        score=score,
                    )
                )

        return filtered_results