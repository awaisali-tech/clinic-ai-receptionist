from sentence_transformers import SentenceTransformer

from ingestion.document_builder import RAGDocument


MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """
    Converts RAG documents into dense vector embeddings.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        documents: list[RAGDocument],
    ):
        """
        Convert document text into embeddings.
        """

        texts = [document.text for document in documents]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings

    def embed_query(self, query: str):
        """
        Convert a user query into a single embedding.
        """

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding