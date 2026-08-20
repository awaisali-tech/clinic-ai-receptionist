from ingestion.loader import load_clinic_data
from ingestion.validator import validate_clinic_data
from ingestion.normalizer import normalize_clinic_data
from ingestion.document_builder import build_documents

from embeddings.embedder import Embedder

from retrieval.faiss_store import FAISSStore
from retrieval.vector_retriever import VectorRetriever
from retrieval.structured_retriever import StructuredRetriever
from retrieval.hybrid_retriever import HybridRetriever

from conversation.entity_resolver import EntityCatalog

from orchestration.factory import SharedRAGResources, build_session_pipeline


DATA_PATH = "data/clinic_data.json"


def build_pipeline():
    """
    Build the complete RAG pipeline.
    """

    # ==================================================
    # 1. LOAD AND PREPARE DATA
    # ==================================================

    data = load_clinic_data(DATA_PATH)

    validate_clinic_data(data)

    normalized_data = normalize_clinic_data(data)

    documents = build_documents(normalized_data)

    # ==================================================
    # 2. CREATE EMBEDDINGS
    # ==================================================

    embedder = Embedder()

    embeddings = embedder.embed_documents(documents)

    # ==================================================
    # 3. BUILD FAISS STORE
    # ==================================================

    dimension = embeddings.shape[1]

    store = FAISSStore(
        dimension=dimension
    )

    store.add(
        embeddings,
        documents,
    )

    # ==================================================
    # 4. CREATE RETRIEVERS
    # ==================================================

    vector_retriever = VectorRetriever(
        store=store,
        embedder=embedder,
    )

    structured_retriever = StructuredRetriever(
        documents=documents,
    )

    hybrid_retriever = HybridRetriever(
        structured_retriever=structured_retriever,
        vector_retriever=vector_retriever,
    )

    # ==================================================
    # 5. SESSION PIPELINE
    # ==================================================

    pipeline = build_session_pipeline(
        SharedRAGResources(
            retriever=hybrid_retriever,
            entity_catalog=EntityCatalog.from_clinic_data(
                normalized_data
            ),
        )
    )

    return pipeline


def print_result(result):
    """
    Print a PipelineResult in a readable format.
    """

    print("\nPipeline result:")
    print(f"Doctor: {result.doctor}")
    print(f"Clinic: {result.clinic}")
    print(f"Specialization: {result.specialization}")
    print(f"Follow-up: {result.is_follow_up}")
    print(f"Grounded: {result.grounded}")
    print(f"Grounding reason: {result.grounding_reason}")

    print("\nRetrieved documents:")

    if not result.results:
        print("No documents retrieved.")

    for rank, item in enumerate(
        result.results,
        start=1,
    ):
        print(
            f"\n#{rank} "
            f"[{item.source}] "
            f"score={item.score:.4f}"
        )

        print(item.document.text)

    print("\nAnswer:")
    print(result.answer)


def main():

    # ==================================================
    # BUILD PIPELINE
    # ==================================================

    print("Building RAG pipeline...")
    print("=" * 60)

    pipeline = build_pipeline()

    print("✓ Pipeline initialized")

    # ==================================================
    # TURN 1
    # ==================================================

    query = "When is Dr. Ayesha Khan available?"

    print(f"\nUser: {query}")

    result = pipeline.run(
        query=query,
        top_k=3,
    )

    print_result(result)

    # ==================================================
    # TURN 2 - FOLLOW-UP
    # ==================================================

    query = "What about Saturday?"

    print(f"\n{'=' * 60}")
    print(f"User: {query}")

    result = pipeline.run(
        query=query,
        top_k=3,
    )

    print_result(result)

    # ==================================================
    # TURN 3 - FOLLOW-UP
    # ==================================================

    query = "What time?"

    print(f"\n{'=' * 60}")
    print(f"User: {query}")

    result = pipeline.run(
        query=query,
        top_k=3,
    )

    print_result(result)

    # ==================================================
    # RESET CONVERSATION
    # ==================================================

    print(f"\n{'=' * 60}")
    print("Resetting conversation...")

    pipeline.reset_conversation()

    query = "I have a different question"

    print(f"\nUser: {query}")

    result = pipeline.run(
        query=query,
        top_k=3,
    )

    print_result(result)


if __name__ == "__main__":
    main()
