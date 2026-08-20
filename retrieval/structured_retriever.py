from __future__ import annotations

from collections.abc import Collection

from ingestion.document_builder import RAGDocument


class StructuredRetriever:
    """Deterministic metadata candidate retrieval over the small corpus."""

    def __init__(self, documents: list[RAGDocument]):
        self.documents = documents

    def retrieve(
        self,
        doctor_name: str | None = None,
        clinic_name: str | None = None,
        specialization: str | None = None,
        document_type: str | None = None,
        document_types: Collection[str] | None = None,
    ) -> list[RAGDocument]:
        accepted_types = {
            value.casefold()
            for value in (document_types or ())
        }
        if document_type:
            accepted_types.add(document_type.casefold())

        results: list[RAGDocument] = []
        for document in self.documents:
            metadata = document.metadata

            if doctor_name and not self._equals(
                metadata.get("doctor_name"),
                doctor_name,
            ):
                continue

            if clinic_name and not self._equals(
                metadata.get("clinic_name"),
                clinic_name,
            ):
                continue

            if specialization and not any(
                self._equals(metadata.get(field), specialization)
                for field in ("specialization", "service_name")
            ):
                continue

            if accepted_types and (
                str(metadata.get("document_type", "")).casefold()
                not in accepted_types
            ):
                continue

            results.append(document)

        return results

    @staticmethod
    def _equals(stored: object, requested: str) -> bool:
        return str(stored or "").strip().casefold() == requested.strip().casefold()
