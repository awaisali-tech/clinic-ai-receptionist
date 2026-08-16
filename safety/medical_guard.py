class MedicalGuard:
    """
    Detects medical-advice requests that the AI receptionist
    should not answer as a medical professional.
    """

    MEDICAL_PATTERNS = {
        "diagnose",
        "diagnosis",
        "what disease",
        "what illness",
        "what condition",
        "what medicine",
        "which medicine",
        "medication",
        "prescribe",
        "prescription",
        "treatment",
        "dosage",
        "dose",
        "should i take",
        "should i use",
        "symptoms",
        "what should i take",
    }

    SAFE_RESPONSE = (
        "I can help with clinic information, doctors, "
        "services, and availability, but I cannot provide "
        "medical diagnosis or treatment advice. "
        "Please consult a qualified healthcare professional."
    )

    def check(self, query: str) -> tuple[bool, str]:
        """
        Check whether a query is requesting medical advice.

        Returns:
            (allowed, response)
        """

        if not isinstance(query, str):
            return False, self.SAFE_RESPONSE

        normalized_query = query.lower().strip()

        for pattern in self.MEDICAL_PATTERNS:
            if pattern in normalized_query:
                return False, self.SAFE_RESPONSE

        return True, ""