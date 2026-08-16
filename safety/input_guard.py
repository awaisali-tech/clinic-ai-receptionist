class InputGuard:
    """
    Validates and filters user input before it reaches
    the retrieval and generation pipeline.
    """

    MAX_QUERY_LENGTH = 500

    BLOCKED_PATTERNS = {
        "ignore previous instructions",
        "ignore all previous instructions",
        "system prompt",
        "reveal your instructions",
        "show me your prompt",
    }

    def check(self, query: str) -> tuple[bool, str]:
        """
        Validate user input.

        Returns:
            (allowed, reason)
        """

        if not isinstance(query, str):
            return False, "Invalid input."

        query = query.strip()

        if not query:
            return False, "Please enter a question."

        if len(query) > self.MAX_QUERY_LENGTH:
            return False, "Your question is too long."

        normalized_query = query.lower()

        for pattern in self.BLOCKED_PATTERNS:
            if pattern in normalized_query:
                return False, "I can only help with clinic-related questions."

        return True, "Input accepted."