from ingestion.loader import load_clinic_data
from ingestion.validator import validate_clinic_data
from ingestion.normalizer import normalize_clinic_data
from ingestion.document_builder import build_documents
from embeddings.embedder import Embedder
from retrieval.faiss_store import FAISSStore
from retrieval.vector_retriever import VectorRetriever
from retrieval.structured_retriever import StructuredRetriever


DATA_PATH = "data/clinic_data.json"


def main():
    # 1. Load data
    data = load_clinic_data(DATA_PATH)
    print("✓ Data loaded")

    # 2. Validate data
    validate_clinic_data(data)
    print("✓ Data validated")

    # 3. Normalize data
    normalized_data = normalize_clinic_data(data)
    print("✓ Data normalized")

    # 4. Build RAG documents
    documents = build_documents(normalized_data)
    print(f"✓ Documents created: {len(documents)}")

    # 5. Create embeddings
    embedder = Embedder()

    embeddings = embedder.embed_documents(documents)
    print(f"✓ Embeddings shape: {embeddings.shape}")

    # 6. Create FAISS store
    dimension = embeddings.shape[1]

    store = FAISSStore(
        dimension=dimension
    )

    store.add(
        embeddings,
        documents,
    )

    print(f"✓ FAISS documents: {store.size}")

    # 7. Test direct FAISS search
    query = "When is Dr. Ayesha Khan available?"

    print(f"\nQuery: {query}\n")

    query_embedding = embedder.embed_query(query)

    results = store.search(
        query_embedding,
        top_k=3,
    )

    print("Direct FAISS results:")
    print("=" * 60)

    for rank, (document, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult #{rank}")
        print(f"Score: {score:.4f}")

        print(
            f"Type: "
            f"{document.metadata.get('document_type')}"
        )

        print(
            f"Clinic: "
            f"{document.metadata.get('clinic_name')}"
        )

        print(f"Text:\n{document.text}")

        print("-" * 60)

    # 8. Test VectorRetriever
    retriever = VectorRetriever(
        store=store,
        embedder=embedder,
    )

    vector_results = retriever.retrieve(
        query="When is Dr. Ayesha Khan available?",
        top_k=3,
    )

    print("\nVectorRetriever results:")
    print("=" * 60)

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):
        print(f"\nResult #{rank}")
        print(f"Score: {result.score:.4f}")

        print(
            f"Type: "
            f"{result.document.metadata.get('document_type')}"
        )

        print(
            f"Doctor: "
            f"{result.document.metadata.get('doctor_name')}"
        )

        print(
            f"Clinic: "
            f"{result.document.metadata.get('clinic_name')}"
        )

        print(
            f"Text:\n{result.document.text}"
        )

        print("-" * 60)

    # 10. Test StructuredRetriever

    structured_retriever = StructuredRetriever(
        documents
    )

    structured_results = structured_retriever.retrieve(
        doctor_name="Dr. Ayesha Khan",
    )

    print("\nStructuredRetriever results:")
    print("=" * 60)

    for rank, document in enumerate(
        structured_results,
        start=1,
    ):
        print(f"\nResult #{rank}")

        print(
            f"Doctor: "
            f"{document.metadata.get('doctor_name')}"
        )

        print(
            f"Clinic: "
            f"{document.metadata.get('clinic_name')}"
        )

        print(
            f"Type: "
            f"{document.metadata.get('document_type')}"
        )

        print(
            f"Text:\n{document.text}"
        )

        print("-" * 60)

    # 9. Save FAISS store
    store.save("vectorstore/faiss")

    print("\n✓ FAISS store saved successfully!")




if __name__ == "__main__":
    main()