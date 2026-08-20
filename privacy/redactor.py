from __future__ import annotations

import re

from privacy.detector import (
    EMAIL_PATTERN,
    HEALTH_DETAIL_PATTERN,
    HEALTH_TERMS,
    NAME_DECLARATION_PATTERN,
    PATIENT_ID_PATTERN,
    PHONE_PATTERN,
    PrivacyDetector,
)
from privacy.models import PrivacyCategory, PrivacyProcessingError, PrivacyResult


_PLACEHOLDERS = {
    PrivacyCategory.EMAIL: "[REDACTED_EMAIL]",
    PrivacyCategory.PHONE: "[REDACTED_PHONE]",
    PrivacyCategory.PERSONAL_NAME: "[REDACTED_NAME]",
    PrivacyCategory.PATIENT_IDENTIFIER: "[REDACTED_PATIENT_ID]",
    PrivacyCategory.HEALTH_DETAILS: "[REDACTED_HEALTH_DETAILS]",
}


class PrivacyProcessor:
    """Minimize direct identifiers and unrelated health narratives."""

    def __init__(self, detector: PrivacyDetector | None = None):
        self.detector = detector or PrivacyDetector()

    def process(
        self,
        query: str,
        *,
        protected_entities: tuple[str, ...] = (),
    ) -> PrivacyResult:
        try:
            if not isinstance(query, str):
                raise TypeError("query must be text")

            categories = self.detector.detect(
                query,
                protected_entities=protected_entities,
            )
            minimized = query

            if PrivacyCategory.HEALTH_DETAILS in categories:
                minimized = HEALTH_DETAIL_PATTERN.sub(
                    self._redact_health_match,
                    minimized,
                )
            if PrivacyCategory.EMAIL in categories:
                minimized = EMAIL_PATTERN.sub(
                    _PLACEHOLDERS[PrivacyCategory.EMAIL],
                    minimized,
                )
            if PrivacyCategory.PHONE in categories:
                minimized = PHONE_PATTERN.sub(
                    self._redact_phone_match,
                    minimized,
                )
            if PrivacyCategory.PATIENT_IDENTIFIER in categories:
                minimized = PATIENT_ID_PATTERN.sub(
                    _PLACEHOLDERS[PrivacyCategory.PATIENT_IDENTIFIER],
                    minimized,
                )
            if PrivacyCategory.PERSONAL_NAME in categories:
                minimized = NAME_DECLARATION_PATTERN.sub(
                    lambda match: self._redact_name_match(
                        match,
                        protected_entities,
                    ),
                    minimized,
                )

            minimized = self._normalize_spacing(minimized)
            if categories and self.detector.detect(
                minimized,
                protected_entities=protected_entities,
            ):
                raise PrivacyProcessingError(
                    "Privacy processing did not produce a safe provider query."
                )

            return PrivacyResult(
                sensitive_data_detected=bool(categories),
                redaction_applied=(
                    bool(categories) and minimized != query.strip()
                ),
                provider_query=minimized,
                categories=categories,
            )
        except PrivacyProcessingError:
            raise
        except Exception as error:
            raise PrivacyProcessingError(
                "Privacy processing failed; no provider query was created."
            ) from error

    @staticmethod
    def _redact_health_match(match: re.Match[str]) -> str:
        if HEALTH_TERMS.search(match.group(0)):
            return _PLACEHOLDERS[PrivacyCategory.HEALTH_DETAILS]
        return match.group(0)

    @staticmethod
    def _redact_phone_match(match: re.Match[str]) -> str:
        value = match.group(0)
        if PrivacyDetector._is_phone(value):
            return _PLACEHOLDERS[PrivacyCategory.PHONE]
        return value

    @staticmethod
    def _redact_name_match(
        match: re.Match[str],
        protected_entities: tuple[str, ...],
    ) -> str:
        candidate = match.group(1).strip()
        protected = {value.casefold() for value in protected_entities if value}
        non_names = {"looking", "asking", "trying", "interested", "wondering"}
        if candidate.casefold() in protected or candidate.casefold().split()[0] in non_names:
            return match.group(0)
        return _PLACEHOLDERS[PrivacyCategory.PERSONAL_NAME]

    @staticmethod
    def _normalize_spacing(value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(r"\s+([,.!?;])", r"\1", value)
        return value
