from typing import Any


def normalize_text(value: str) -> str:
    """Normalize a text value."""
    return " ".join(value.strip().split())


def normalize_clinic_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize clinic data without changing its meaning.

    Returns:
        A new normalized dataset.
    """

    normalized_data = {
        "clinics": []
    }

    for clinic in data["clinics"]:

        normalized_clinic = clinic.copy()

        # Basic clinic information
        normalized_clinic["id"] = normalize_text(
            str(clinic["id"])
        )

        normalized_clinic["name"] = normalize_text(
            clinic["name"]
        )

        normalized_clinic["about"] = normalize_text(
            clinic["about"]
        )

        # Location
        if isinstance(clinic["location"], dict):
            normalized_clinic["location"] = {
                key: normalize_text(str(value))
                for key, value in clinic["location"].items()
            }

        # Contact
        if isinstance(clinic["contact"], dict):
            normalized_clinic["contact"] = {
                key: normalize_text(str(value))
                for key, value in clinic["contact"].items()
            }

        # Services
        normalized_clinic["services"] = [
            normalize_text(service)
            if isinstance(service, str)
            else service
            for service in clinic["services"]
        ]

        # Doctors
        normalized_clinic["doctors"] = []

        for doctor in clinic["doctors"]:

            normalized_doctor = doctor.copy()

            for field in ["name", "specialization"]:
                if field in normalized_doctor:
                    normalized_doctor[field] = normalize_text(
                        normalized_doctor[field]
                    )

            normalized_clinic["doctors"].append(
                normalized_doctor
            )

        # FAQs
        normalized_clinic["faqs"] = []

        for faq in clinic["faqs"]:

            if isinstance(faq, dict):
                normalized_faq = faq.copy()

                for key, value in normalized_faq.items():
                    if isinstance(value, str):
                        normalized_faq[key] = normalize_text(value)

                normalized_clinic["faqs"].append(
                    normalized_faq
                )
            else:
                normalized_clinic["faqs"].append(faq)

        normalized_data["clinics"].append(
            normalized_clinic
        )

    return normalized_data