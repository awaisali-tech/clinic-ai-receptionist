from dataclasses import dataclass

from ingestion.document_builder import RAGDocument


@dataclass
class GroundingResult:
    """
    Result of checking whether retrieved documents
    contain enough evidence to answer a query.
    """

    is_grounded: bool
    evidence: list[RAGDocument]
    reason: str


class GroundingValidator:
    """
    Validates whether retrieval produced usable evidence.

    Important:
    Retrieval success does NOT automatically mean
    that the answer is grounded.

    The validator checks:
        1. Whether evidence exists.
        2. Whether the evidence contains relevant text.
        3. Whether the requested entity is present.
    """

    def __init__(
        self,
        minimum_results: int = 1,
    ):
        self.minimum_results = minimum_results

    def validate(
        self,
        query: str,
        results: list,
        doctor_name: str | None = None,
        clinic_name: str | None = None,
    ) -> GroundingResult:

        # -----------------------------------------
        # 1. No retrieval results
        # -----------------------------------------

        if not results:

            return GroundingResult(
                is_grounded=False,
                evidence=[],
                reason="No relevant information was retrieved.",
            )

        # -----------------------------------------
        # 2. Extract documents
        # -----------------------------------------

        documents = [
            result.document
            for result in results
            if hasattr(result, "document")
        ]

        if len(documents) < self.minimum_results:

            return GroundingResult(
                is_grounded=False,
                evidence=documents,
                reason="Insufficient retrieval evidence.",
            )

        # -----------------------------------------
        # 3. Verify requested doctor
        # -----------------------------------------

        if doctor_name:

            doctor_found = any(
                document.metadata.get("doctor_name")
                == doctor_name
                for document in documents
            )

            if not doctor_found:

                return GroundingResult(
                    is_grounded=False,
                    evidence=documents,
                    reason=(
                        f"No retrieved evidence was found for "
                        f"{doctor_name}."
                    ),
                )

        # -----------------------------------------
        # 4. Verify requested clinic
        # -----------------------------------------

        if clinic_name:

            clinic_found = any(
                document.metadata.get("clinic_name")
                == clinic_name
                for document in documents
            )

            if not clinic_found:

                return GroundingResult(
                    is_grounded=False,
                    evidence=documents,
                    reason=(
                        f"No retrieved evidence was found for "
                        f"{clinic_name}."
                    ),
                )

        # -----------------------------------------
        # 5. Evidence exists
        # -----------------------------------------

        return GroundingResult(
            is_grounded=True,
            evidence=documents,
            reason="Relevant grounded evidence was retrieved.",
        )