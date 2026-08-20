from __future__ import annotations

import re
import unicodedata

from safety.models import SafetyCategory, SafetyDecision


ADMINISTRATIVE_RESPONSE = (
    "I can only help with clinic-related questions."
)

MEDICAL_ADVICE_RESPONSE = (
    "I can help with clinic information, doctors, services, and "
    "availability, but I cannot provide personalized medical diagnosis "
    "or treatment advice. Please consult a qualified healthcare professional."
)

URGENT_RESPONSE = (
    "This may require urgent medical attention. Please contact local "
    "emergency services or seek immediate in-person medical care."
)


def normalize_policy_text(value: str) -> str:
    """Normalize casing and punctuation while retaining word boundaries."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized)
    return " ".join(normalized.split())


class SafetyPolicy:
    """Central deterministic safety rules for an administrative assistant."""

    MAX_QUERY_LENGTH = 500

    _INJECTION_PATTERNS = (
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b.{0,45}"
            r"\b(?:previous|prior|system|developer|clinic|all)\b.{0,25}"
            r"\b(?:instruction|instructions|rule|rules|prompt)\b"
        ),
        re.compile(
            r"\b(?:reveal|show|print|display|repeat|expose)\b.{0,35}"
            r"\b(?:system|hidden|developer|internal)\b.{0,20}"
            r"\b(?:prompt|instruction|instructions|message|rules)\b"
        ),
        re.compile(
            r"\b(?:show|reveal|print|give|expose)\b.{0,30}"
            r"\b(?:api key|secret|secrets|environment variables?|credentials)\b"
        ),
        re.compile(
            r"\b(?:treat|reinterpret|use)\b.{0,35}\b(?:evidence|context)\b"
            r".{0,35}\b(?:system|instruction|instructions|commands)\b"
        ),
        re.compile(
            r"\b(?:forget|bypass|break|leave)\b.{0,30}"
            r"\b(?:clinic|scope|rules|restrictions)\b.{0,25}"
            r"\b(?:answer anything|unrestricted|anything)\b"
        ),
    )

    _URGENT_PATTERNS = (
        re.compile(
            r"\b(?:severe|extreme|sudden)\b.{0,25}"
            r"\b(?:difficulty breathing|trouble breathing|shortness of breath)\b"
        ),
        re.compile(r"\b(?:can not|cannot|cant|unable to)\s+breathe\b"),
        re.compile(r"\b(?:not breathing|unconscious|unresponsive)\b"),
        re.compile(r"\b(?:severe|crushing|sudden)\s+chest\s+pain\b"),
        re.compile(
            r"\b(?:uncontrolled|heavy|severe)\s+bleeding\b|"
            r"\bbleeding\b.{0,20}\b(?:will not|wont|cannot)\s+stop\b"
        ),
        re.compile(
            r"\b(?:face droop|face drooping|slurred speech|"
            r"sudden one sided weakness|sudden weakness on one side)\b"
        ),
        re.compile(
            r"\b(?:severe allergic reaction|anaphylaxis|"
            r"tongue swelling|throat swelling|throat is closing)\b"
        ),
        re.compile(r"\b(?:overdose|attempted suicide|suicidal|kill myself)\b"),
    )

    _MEDICAL_ADVICE_PATTERNS = (
        re.compile(
            r"\b(?:can you|could you|please)?\s*diagnos(?:e|is)\b"
            r".{0,40}\b(?:me|my|these symptoms|this)\b"
        ),
        re.compile(
            r"\b(?:what|which)\s+(?:disease|illness|condition)\b"
            r".{0,25}\b(?:do i have|is this|have i got)\b"
        ),
        re.compile(r"\bdo i have\s+(?:a\s+)?[a-z0-9 ]{2,40}\??$"),
        re.compile(
            r"\b(?:is|could|might)\s+(?:this|my)\b.{0,30}"
            r"\b(?:cancer|diabetes|infection|disease|condition)\b"
        ),
        re.compile(
            r"\b(?:what|which)\s+(?:medicine|medication|drug)\b"
            r".{0,40}\b(?:should|can|do)\s+i\s+(?:take|use)\b"
        ),
        re.compile(
            r"\b(?:what should i take|should i take|can i take)\b"
            r".{0,45}\b(?:pain|headache|rash|fever|symptom|medicine|medication)\b"
        ),
        re.compile(
            r"\b(?:should|can) i\s+(?:take|use)\b.{0,45}"
            r"\b(?:medicine|medication|insulin|antibiotic|cream|ointment|"
            r"aspirin|ibuprofen|paracetamol|acetaminophen)\b"
        ),
        re.compile(
            r"\bshould i\s+(?:stop|start|change|continue)\b.{0,45}"
            r"\b(?:medicine|medication|prescription|insulin|antibiotic|taking)\b"
        ),
        re.compile(
            r"\b(?:what|which|recommended|right|my)\s+dos(?:e|age)\b|"
            r"\bhow much\s+(?:insulin|medicine|medication)\b"
        ),
        re.compile(
            r"\b(?:how (?:do|can|should) i|tell me how to)\s+treat\b"
            r".{0,45}\b(?:my|this|pain|at home|symptoms?)\b"
        ),
        re.compile(
            r"\bwhat treatment\b.{0,35}\b(?:should i|get|is right for me|"
            r"for my)\b"
        ),
        re.compile(
            r"\bwhat should i do\b.{0,35}"
            r"\b(?:for|about)\b.{0,25}"
            r"\b(?:my|pain|rash|fever|symptoms?)\b"
        ),
    )

    def evaluate_input(self, query: object) -> SafetyDecision:
        if not isinstance(query, str):
            return SafetyDecision(
                False,
                SafetyCategory.INVALID_INPUT,
                "Input must be text.",
                "Invalid input.",
            )

        stripped = query.strip()
        if not stripped:
            return SafetyDecision(
                False,
                SafetyCategory.INVALID_INPUT,
                "Input was empty.",
                "Please enter a question.",
            )
        if len(stripped) > self.MAX_QUERY_LENGTH:
            return SafetyDecision(
                False,
                SafetyCategory.INVALID_INPUT,
                "Input exceeded the supported length.",
                "Your question is too long.",
            )

        normalized = normalize_policy_text(stripped)
        if any(pattern.search(normalized) for pattern in self._INJECTION_PATTERNS):
            return SafetyDecision(
                False,
                SafetyCategory.PROMPT_INJECTION,
                "The request attempted to manipulate assistant controls.",
                ADMINISTRATIVE_RESPONSE,
            )

        return self._safe()

    def evaluate_medical(self, query: object) -> SafetyDecision:
        if not isinstance(query, str):
            return SafetyDecision(
                False,
                SafetyCategory.INVALID_INPUT,
                "Input must be text.",
                "Invalid input.",
            )

        normalized = normalize_policy_text(query)
        if any(pattern.search(normalized) for pattern in self._URGENT_PATTERNS):
            return SafetyDecision(
                False,
                SafetyCategory.URGENT_MEDICAL,
                "The request contains a conservative urgent-risk signal.",
                URGENT_RESPONSE,
            )
        if any(
            pattern.search(normalized)
            for pattern in self._MEDICAL_ADVICE_PATTERNS
        ):
            return SafetyDecision(
                False,
                SafetyCategory.MEDICAL_ADVICE,
                "The request asks for personalized medical guidance.",
                MEDICAL_ADVICE_RESPONSE,
            )
        return self._safe()

    def evaluate(self, query: object) -> SafetyDecision:
        """Run the complete ordered input policy."""

        input_decision = self.evaluate_input(query)
        if not input_decision.allowed:
            return input_decision
        return self.evaluate_medical(query)

    @staticmethod
    def _safe() -> SafetyDecision:
        return SafetyDecision(
            True,
            SafetyCategory.SAFE_ADMINISTRATIVE,
            "Request is within the administrative safety boundary.",
        )
