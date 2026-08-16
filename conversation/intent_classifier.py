class IntentClassifier:
    """
    Classifies common receptionist queries.

    This classifier is intentionally deterministic.
    """

    INTENTS = {
        "doctor_availability",
        "clinic_hours",
        "clinic_location",
        "clinic_contact",
        "services",
        "faq",
        "medical_advice",
        "unknown",
    }

    def classify(self, query: str) -> str:

        text = query.lower().strip()

        # -----------------------------------------
        # Medical advice
        # -----------------------------------------

        medical_keywords = [
            "diagnose",
            "diagnosis",
            "treatment",
            "medicine",
            "medication",
            "symptoms",
            "what should i take",
            "what should i do",
        ]

        if any(
            keyword in text
            for keyword in medical_keywords
        ):
            return "medical_advice"

        # -----------------------------------------
        # Doctor availability
        # -----------------------------------------

        availability_keywords = [
            "available",
            "availability",
            "when is",
            "what time",
            "schedule",
            "appointment",
        ]

        if any(
            keyword in text
            for keyword in availability_keywords
        ):
            return "doctor_availability"

        # -----------------------------------------
        # Clinic hours
        # -----------------------------------------

        hours_keywords = [
            "opening hours",
            "opening time",
            "closing time",
            "open",
            "close",
            "hours",
            "timings",
        ]

        if any(
            keyword in text
            for keyword in hours_keywords
        ):
            return "clinic_hours"

        # -----------------------------------------
        # Location
        # -----------------------------------------

        location_keywords = [
            "where is",
            "where are",
            "location",
            "address",
            "located",
        ]

        if any(
            keyword in text
            for keyword in location_keywords
        ):
            return "clinic_location"

        # -----------------------------------------
        # Contact
        # -----------------------------------------

        contact_keywords = [
            "phone",
            "telephone",
            "contact",
            "email",
            "call",
        ]

        if any(
            keyword in text
            for keyword in contact_keywords
        ):
            return "clinic_contact"

        # -----------------------------------------
        # Services
        # -----------------------------------------

        service_keywords = [
            "services",
            "service",
            "what do you offer",
            "specialties",
            "speciality",
        ]

        if any(
            keyword in text
            for keyword in service_keywords
        ):
            return "services"

        return "unknown"