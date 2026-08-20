from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SafetyCategory(str, Enum):
    """Deterministic request/output safety classifications."""

    SAFE_ADMINISTRATIVE = "safe_administrative"
    MEDICAL_ADVICE = "medical_advice"
    URGENT_MEDICAL = "urgent_medical"
    SENSITIVE_DATA = "sensitive_data"
    PROMPT_INJECTION = "prompt_injection"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class SafetyDecision:
    """A policy decision without copies of the evaluated sensitive text."""

    allowed: bool
    category: SafetyCategory
    reason: str
    user_response: str | None = None

    def legacy_tuple(self, accepted_message: str = "") -> tuple[bool, str]:
        """Return the historical ``(allowed, response)`` guard contract."""

        if self.allowed:
            return True, accepted_message
        return False, self.user_response or self.reason
