from __future__ import annotations

from dataclasses import dataclass

from conversation.entity_resolver import ContextResolver, EntityCatalog
from conversation.intent_classifier import IntentClassifier
from generation.generator import Generator
from privacy.redactor import PrivacyProcessor
from orchestration.pipeline import (
    GeneratorDependency,
    GroundingDependency,
    GuardDependency,
    IntentDependency,
    RAGPipeline,
    RetrieverDependency,
)
from retrieval.grounding_validator import GroundingValidator
from safety.input_guard import InputGuard
from safety.medical_guard import MedicalGuard
from safety.output_guard import OutputGuard


@dataclass(frozen=True)
class SharedRAGResources:
    """Expensive, stateless resources safe to reuse across sessions."""

    retriever: RetrieverDependency
    entity_catalog: EntityCatalog


def build_session_pipeline(
    resources: SharedRAGResources,
    *,
    generator: GeneratorDependency | None = None,
    grounding_validator: GroundingDependency | None = None,
    intent_classifier: IntentDependency | None = None,
    input_guard: GuardDependency | None = None,
    medical_guard: GuardDependency | None = None,
    output_guard: GuardDependency | None = None,
    privacy_processor: PrivacyProcessor | None = None,
) -> RAGPipeline:
    """Create a pipeline with fresh mutable state for one browser session."""

    return RAGPipeline(
        retriever=resources.retriever,
        context_resolver=ContextResolver(resources.entity_catalog),
        intent_classifier=(
            intent_classifier
            if intent_classifier is not None
            else IntentClassifier()
        ),
        grounding_validator=(
            grounding_validator
            if grounding_validator is not None
            else GroundingValidator()
        ),
        generator=generator if generator is not None else Generator(),
        input_guard=input_guard if input_guard is not None else InputGuard(),
        medical_guard=(
            medical_guard if medical_guard is not None else MedicalGuard()
        ),
        output_guard=output_guard if output_guard is not None else OutputGuard(),
        privacy_processor=(
            privacy_processor
            if privacy_processor is not None
            else PrivacyProcessor()
        ),
    )
