from __future__ import annotations

import re

from safety.models import SafetyCategory, SafetyDecision
from safety.policy import normalize_policy_text


class OutputGuard:
    """Reject patient-specific medical guidance and secret disclosure."""

    SAFE_FALLBACK = (
        "I can help with clinic information, doctors, services, availability, "
        "and contact details. For medical diagnosis or treatment advice, "
        "please consult a qualified healthcare professional."
    )

    _MEDICAL_OUTPUT_PATTERNS = (
        re.compile(
            r"\byou\s+have\s+(?:diabetes|cancer|an infection|a disease|"
            r"a medical condition|pneumonia|appendicitis|a stroke)\b"
        ),
        re.compile(
            r"\b(?:you|the patient)\s+(?:likely|probably|definitely)\s+have\b"
        ),
        re.compile(
            r"\b(?:your|the patient s)\s+diagnosis\s+(?:is|appears to be)\b|"
            r"\bi diagnose\b"
        ),
        re.compile(
            r"\b(?:your symptoms|these symptoms)\b.{0,25}"
            r"\b(?:indicate|confirm|mean|show)\b"
        ),
        re.compile(
            r"\byou should\s+(?:take|use|stop|start|increase|decrease)\b"
            r".{0,50}\b(?:medicine|medication|prescription|tablet|capsule|"
            r"insulin|antibiotic|aspirin|ibuprofen|paracetamol|acetaminophen|"
            r"dose|mg)\b"
        ),
        re.compile(
            r"\b(?:take|use)\s+\d+(?:\.\d+)?\s*"
            r"(?:mg|ml|mcg|units?|tablets?|capsules?)\b"
        ),
        re.compile(
            r"\b(?:your|recommended)\s+dos(?:e|age)\s+"
            r"(?:is|should be)\b"
        ),
        re.compile(
            r"\b(?:to treat your|for your)\b.{0,30}"
            r"\b(?:take|use|apply)\b"
        ),
    )

    _SECRET_OUTPUT_PATTERNS = (
        re.compile(r"\b(?:groq api key|api key is|system prompt is)\b"),
        re.compile(r"\b(?:g?sk)[_ ][a-z0-9_]{12,}\b"),
    )

    def evaluate(self, response: object) -> SafetyDecision:
        if not isinstance(response, str) or not response.strip():
            return SafetyDecision(
                False,
                SafetyCategory.INVALID_INPUT,
                "Generated output was empty or malformed.",
                self.SAFE_FALLBACK,
            )

        normalized = normalize_policy_text(response)
        if any(
            pattern.search(normalized)
            for pattern in self._SECRET_OUTPUT_PATTERNS
        ):
            return SafetyDecision(
                False,
                SafetyCategory.PROMPT_INJECTION,
                "Generated output attempted to expose protected instructions or secrets.",
                self.SAFE_FALLBACK,
            )
        if any(
            pattern.search(normalized)
            for pattern in self._MEDICAL_OUTPUT_PATTERNS
        ):
            return SafetyDecision(
                False,
                SafetyCategory.MEDICAL_ADVICE,
                "Generated output contained personalized medical guidance.",
                self.SAFE_FALLBACK,
            )

        return SafetyDecision(
            True,
            SafetyCategory.SAFE_ADMINISTRATIVE,
            "Generated output remained within the administrative boundary.",
            response.strip(),
        )

    def check(self, response: str) -> tuple[bool, str]:
        """Compatibility adapter for the historical tuple contract."""

        decision = self.evaluate(response)
        return decision.allowed, decision.user_response or self.SAFE_FALLBACK
