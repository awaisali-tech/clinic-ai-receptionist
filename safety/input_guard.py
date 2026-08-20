from safety.models import SafetyDecision
from safety.policy import SafetyPolicy


class InputGuard:
    """
    Validates and filters user input before it reaches
    the retrieval and generation pipeline.
    """

    MAX_QUERY_LENGTH = SafetyPolicy.MAX_QUERY_LENGTH

    def __init__(self, policy: SafetyPolicy | None = None):
        self.policy = policy or SafetyPolicy()

    def evaluate(self, query: object) -> SafetyDecision:
        return self.policy.evaluate_input(query)

    def check(self, query: str) -> tuple[bool, str]:
        """
        Validate user input.

        Returns:
            (allowed, reason)
        """

        return self.evaluate(query).legacy_tuple("Input accepted.")
