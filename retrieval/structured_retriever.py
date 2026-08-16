from ingestion.document_builder import RAGDocument


class StructuredRetriever:
    """
    Exact/metadata-based retrieval.

    Used when the query contains known entities such as:
    - doctor name
    - clinic name
    - specialization
    - document type
    """

    def __init__(self, documents: list[RAGDocument]):
        self.documents = documents

    def retrieve(
        self,
        doctor_name: str | None = None,
        clinic_name: str | None = None,
        specialization: str | None = None,
        document_type: str | None = None,
    ) -> list[RAGDocument]:

        results = []

        for document in self.documents:

            metadata = document.metadata

            # Doctor filter
            if doctor_name:
                stored_doctor = metadata.get(
                    "doctor_name",
                    "",
                ).lower()

                if doctor_name.lower() not in stored_doctor:
                    continue

            # Clinic filter
            if clinic_name:
                stored_clinic = metadata.get(
                    "clinic_name",
                    "",
                ).lower()

                if clinic_name.lower() not in stored_clinic:
                    continue

            # Specialization filter
            if specialization:
                stored_specialization = metadata.get(
                    "specialization",
                    "",
                ).lower()

                if (
                    specialization.lower()
                    not in stored_specialization
                ):
                    continue

            # Document type filter
            if document_type:
                stored_type = metadata.get(
                    "document_type",
                    "",
                ).lower()

                if stored_type != document_type.lower():
                    continue

            results.append(document)

        return results