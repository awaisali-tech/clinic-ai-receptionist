from dataclasses import dataclass

from ingestion.loader import load_clinic_data
from ingestion.validator import validate_clinic_data
from ingestion.normalizer import normalize_clinic_data
from ingestion.document_builder import build_documents

from embeddings.embedder import Embedder

from retrieval.faiss_store import FAISSStore
from retrieval.vector_retriever import VectorRetriever
from retrieval.structured_retriever import StructuredRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.grounding_validator import GroundingValidator
from conversation.intent_classifier import InformationNeed


DATA_PATH = "data/clinic_data.json"


@dataclass
class TestResult:
    document: object
    score: float
    source: str


def build_retriever():

    data = load_clinic_data(DATA_PATH)

    validate_clinic_data(data)

    normalized_data = normalize_clinic_data(data)

    documents = build_documents(normalized_data)

    embedder = Embedder()

    embeddings = embedder.embed_documents(documents)

    store = FAISSStore(
        dimension=embeddings.shape[1]
    )

    store.add(
        embeddings,
        documents,
    )

    vector_retriever = VectorRetriever(
        store=store,
        embedder=embedder,
    )

    structured_retriever = StructuredRetriever(
        documents=documents,
    )

    return HybridRetriever(
        structured_retriever=structured_retriever,
        vector_retriever=vector_retriever,
    )


def main():

    print("Building retriever...")
    print("=" * 60)

    retriever = build_retriever()

    validator = GroundingValidator()

    # -----------------------------------------
    # Test 1: Known doctor
    # -----------------------------------------

    query = "When is Dr. Ayesha Khan available?"

    results = retriever.retrieve(
        query=query,
        doctor_name="Dr. Ayesha Khan",
        information_need=InformationNeed.DOCTOR_AVAILABILITY,
        top_k=3,
    )

    grounding = validator.validate(
        query=query,
        results=results,
        doctor_name="Dr. Ayesha Khan",
        information_need=InformationNeed.DOCTOR_AVAILABILITY,
    )

    print("\nTest 1: Known doctor")
    print(f"Grounded: {grounding.is_grounded}")
    print(f"Reason: {grounding.reason}")

    assert grounding.is_grounded

    # -----------------------------------------
    # Test 2: Unknown doctor
    # -----------------------------------------

    query = "When is Dr. Unknown available?"

    results = retriever.retrieve(
        query=query,
        doctor_name="Dr. Unknown",
        information_need=InformationNeed.DOCTOR_AVAILABILITY,
        top_k=3,
    )

    grounding = validator.validate(
        query=query,
        results=results,
        doctor_name="Dr. Unknown",
        information_need=InformationNeed.DOCTOR_AVAILABILITY,
    )

    print("\nTest 2: Unknown doctor")
    print(f"Grounded: {grounding.is_grounded}")
    print(f"Reason: {grounding.reason}")

    assert not grounding.is_grounded

    # -----------------------------------------
    # Test 3: No results
    # -----------------------------------------

    grounding = validator.validate(
        query="Some completely unknown question",
        results=[],
    )

    print("\nTest 3: No results")
    print(f"Grounded: {grounding.is_grounded}")
    print(f"Reason: {grounding.reason}")

    assert not grounding.is_grounded

    print("\n✓ All grounding tests passed!")


if __name__ == "__main__":
    main()
