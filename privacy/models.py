from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PrivacyCategory(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    PERSONAL_NAME = "personal_name"
    PATIENT_IDENTIFIER = "patient_identifier"
    HEALTH_DETAILS = "health_details"


@dataclass(frozen=True)
class PrivacyResult:
    """Safe privacy diagnostics plus the minimized provider representation."""

    sensitive_data_detected: bool
    redaction_applied: bool
    provider_query: str
    categories: tuple[PrivacyCategory, ...] = ()

    @property
    def retrieval_query(self) -> str:
        """The minimized representation is also safe for local retrieval."""

        return self.provider_query


class PrivacyProcessingError(RuntimeError):
    """A privacy failure whose message never contains the source content."""
