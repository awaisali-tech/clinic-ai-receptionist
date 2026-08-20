from safety.models import SafetyDecision
from safety.policy import MEDICAL_ADVICE_RESPONSE, SafetyPolicy


class MedicalGuard:
    """
    Detects medical-advice requests that the AI receptionist
    should not answer as a medical professional.
    """

    SAFE_RESPONSE = MEDICAL_ADVICE_RESPONSE

    def __init__(self, policy: SafetyPolicy | None = None):
        self.policy = policy or SafetyPolicy()

    def evaluate(self, query: object) -> SafetyDecision:
        return self.policy.evaluate_medical(query)

    def check(self, query: str) -> tuple[bool, str]:
        """
        Check whether a query is requesting medical advice.

        Returns:
            (allowed, response)
        """

        return self.evaluate(query).legacy_tuple()
