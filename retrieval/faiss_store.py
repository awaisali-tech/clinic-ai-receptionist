from pathlib import Path
import pickle

import faiss
import numpy as np

from ingestion.document_builder import RAGDocument


class FAISSStore:
    """
    FAISS-based vector store for clinic RAG documents.

    FAISS stores the vectors while the document list stores
    the corresponding text and metadata.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

        # Inner product works as cosine similarity because
        # our embeddings are normalized.
        self.index = faiss.IndexFlatIP(dimension)

        self.documents: list[RAGDocument] = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[RAGDocument],
    ) -> None:
        """
        Add embeddings and their corresponding documents.
        """

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match "
                "number of documents."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D NumPy array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        self.index.add(
            embeddings.astype(np.float32)
        )

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[RAGDocument, float]]:
        """
        Search for the most similar documents.

        Returns:
            List of (document, similarity_score).
        """

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            document = self.documents[index]

            results.append(
                (document, float(score))
            )

        return results

    def save(self, directory: str | Path) -> None:
        """
        Persist FAISS index and document metadata.
        """

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        with open(
            directory / "documents.pkl",
            "wb",
        ) as file:

            pickle.dump(
                self.documents,
                file,
            )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FAISSStore":
        """
        Load a previously saved FAISS store.
        """

        directory = Path(directory)

        index = faiss.read_index(
            str(directory / "index.faiss")
        )

        with open(
            directory / "documents.pkl",
            "rb",
        ) as file:

            documents = pickle.load(file)

        store = cls(
            dimension=index.d
        )

        store.index = index
        store.documents = documents

        return store

    @property
    def size(self) -> int:
        """Return number of indexed documents."""

        return self.index.ntotal