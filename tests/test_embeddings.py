from ingestion.loader import load_clinic_data
from ingestion.validator import validate_clinic_data
from ingestion.normalizer import normalize_clinic_data
from ingestion.document_builder import build_documents
from embeddings.embedder import Embedder


DATA_PATH = "data/clinic_data.json"


def main():

    data = load_clinic_data(DATA_PATH)

    validate_clinic_data(data)

    normalized_data = normalize_clinic_data(data)

    documents = build_documents(normalized_data)

    print(f"Documents: {len(documents)}")

    embedder = Embedder()

    embeddings = embedder.embed_documents(documents)

    print(f"Embedding shape: {embeddings.shape}")

    query = "When is Dr. Ayesha Khan available?"

    query_embedding = embedder.embed_query(query)

    print(f"Query embedding shape: {query_embedding.shape}")

    print("\nEmbedding test passed!")


if __name__ == "__main__":
    main()