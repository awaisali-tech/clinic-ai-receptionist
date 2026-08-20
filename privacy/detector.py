from __future__ import annotations

import re

from privacy.models import PrivacyCategory


EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w-])",
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")

PATIENT_ID_PATTERN = re.compile(
    r"\b(?:patient\s*(?:id|number)|medical\s*record\s*(?:id|number)|"
    r"mrn|cnic)\s*(?:is|:|#)?\s*[A-Z0-9-]{4,}\b",
    re.IGNORECASE,
)

NAME_DECLARATION_PATTERN = re.compile(
    r"\b(?:my\s+name\s+is|i\s+am|i['’]?m)\s+"
    r"([A-Z][A-Za-z'’-]*(?:\s+[A-Z][A-Za-z'’-]*){0,2})"
    r"(?=\s+(?:and|my|when|where|what|which|does|is|can)\b|[,.!?;]|$)",
    re.IGNORECASE,
)

HEALTH_DETAIL_PATTERN = re.compile(
    r"\b(?:i\s+have|i['’]?ve\s+had|i\s+am\s+experiencing|"
    r"my\s+symptoms?\s+(?:are|include)|my\s+medical\s+history\s+(?:is|includes)|"
    r"i\s+was\s+diagnosed\s+with)\s+"
    r"([^.!?;]{1,180}?)"
    r"(?=\s+(?:what|when|where|which|does|is|are|can)\b|[.!?;]|$)",
    re.IGNORECASE,
)

HEALTH_TERMS = re.compile(
    r"\b(?:pain|ache|headache|migraine|rash|fever|cough|bleeding|"
    r"nausea|vomiting|dizzy|dizziness|blurred vision|symptoms?|"
    r"diabetes|cancer|infection|allergy|allergic|asthma|blood pressure|"
    r"pregnant|pregnancy|medication|prescription|diagnosis|condition)\b",
    re.IGNORECASE,
)


class PrivacyDetector:
    """Detect common structured identifiers and explicit health disclosures."""

    def detect(
        self,
        text: str,
        *,
        protected_entities: tuple[str, ...] = (),
    ) -> tuple[PrivacyCategory, ...]:
        categories: list[PrivacyCategory] = []

        if EMAIL_PATTERN.search(text):
            categories.append(PrivacyCategory.EMAIL)
        if any(self._is_phone(match.group(0)) for match in PHONE_PATTERN.finditer(text)):
            categories.append(PrivacyCategory.PHONE)
        if PATIENT_ID_PATTERN.search(text):
            categories.append(PrivacyCategory.PATIENT_IDENTIFIER)
        if self._has_unprotected_declared_name(text, protected_entities):
            categories.append(PrivacyCategory.PERSONAL_NAME)
        if any(
            HEALTH_TERMS.search(match.group(0))
            for match in HEALTH_DETAIL_PATTERN.finditer(text)
        ):
            categories.append(PrivacyCategory.HEALTH_DETAILS)

        return tuple(categories)

    @staticmethod
    def _is_phone(value: str) -> bool:
        return len(re.sub(r"\D", "", value)) >= 10

    @staticmethod
    def _has_unprotected_declared_name(
        text: str,
        protected_entities: tuple[str, ...],
    ) -> bool:
        protected = {value.casefold() for value in protected_entities if value}
        non_names = {"looking", "asking", "trying", "interested", "wondering"}
        for match in NAME_DECLARATION_PATTERN.finditer(text):
            candidate = match.group(1).strip().casefold()
            if candidate.split()[0] in non_names:
                continue
            if candidate not in protected:
                return True
        return False
