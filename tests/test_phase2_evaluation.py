import unittest

from conversation.entity_resolver import EntityCatalog, normalize_entity_text
from evaluation.retrieval_evaluation import (
    RETRIEVAL_EVALUATION_CASES,
    run_retrieval_evaluation,
)
from ingestion.document_builder import build_documents
from ingestion.loader import load_clinic_data
from ingestion.normalizer import normalize_clinic_data
from orchestration.factory import SharedRAGResources, build_session_pipeline
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.structured_retriever import StructuredRetriever
from retrieval.vector_retriever import RetrievalResult


class LexicalVectorRetriever:
    """Offline semantic stand-in for deterministic evaluation tests."""

    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=5, **kwargs):
        query_terms = set(normalize_entity_text(query).split())
        scored = []
        for document in self.documents:
            document_terms = set(normalize_entity_text(document.text).split())
            overlap = len(query_terms & document_terms)
            if overlap:
                scored.append(
                    RetrievalResult(
                        document=document,
                        score=min(0.40 + overlap * 0.08, 0.95),
                    )
                )
        scored.sort(key=lambda result: -result.score)
        return scored[:top_k]


class FakeGenerator:
    def generate(self, query, evidence, conversation_context=None):
        return "Deterministic evaluation answer."


class AllowGuard:
    def check(self, value):
        return True, value


class RetrievalEvaluationTests(unittest.TestCase):
    def test_curated_evaluation_fixture(self):
        data = normalize_clinic_data(load_clinic_data("data/clinic_data.json"))
        documents = build_documents(data)
        retriever = HybridRetriever(
            StructuredRetriever(documents),
            LexicalVectorRetriever(documents),
        )
        pipeline = build_session_pipeline(
            SharedRAGResources(
                retriever=retriever,
                entity_catalog=EntityCatalog.from_clinic_data(data),
            ),
            generator=FakeGenerator(),
            input_guard=AllowGuard(),
            medical_guard=AllowGuard(),
            output_guard=AllowGuard(),
        )

        report = run_retrieval_evaluation(pipeline)

        self.assertEqual(
            report.passed,
            len(RETRIEVAL_EVALUATION_CASES),
            msg=str(report.failures),
        )


if __name__ == "__main__":
    unittest.main()
