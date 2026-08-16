from typing import Any


REQUIRED_CLINIC_FIELDS = {
    "id",
    "name",
    "location",
    "contact",
    "services",
    "doctors",
    "timings",
    "faqs",
    "about",
}


def validate_clinic_data(data: dict[str, Any]) -> None:
    """
    Validate the basic structure of the clinic dataset.

    Raises:
        ValueError: If the dataset structure is invalid.
    """

    if not isinstance(data, dict):
        raise ValueError("Clinic data must be a dictionary.")

    if "clinics" not in data:
        raise ValueError("Missing required top-level field: 'clinics'.")

    clinics = data["clinics"]

    if not isinstance(clinics, list):
        raise ValueError("'clinics' must be a list.")

    if not clinics:
        raise ValueError("Clinic dataset cannot be empty.")

    for index, clinic in enumerate(clinics):

        if not isinstance(clinic, dict):
            raise ValueError(
                f"Clinic at index {index} must be an object."
            )

        missing_fields = REQUIRED_CLINIC_FIELDS - clinic.keys()

        if missing_fields:
            raise ValueError(
                f"Clinic at index {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if not isinstance(clinic["services"], list):
            raise ValueError(
                f"Clinic '{clinic['name']}' services must be a list."
            )

        if not isinstance(clinic["doctors"], list):
            raise ValueError(
                f"Clinic '{clinic['name']}' doctors must be a list."
            )

        if not isinstance(clinic["faqs"], list):
            raise ValueError(
                f"Clinic '{clinic['name']}' FAQs must be a list."
            )

    print(
        f"Validation successful: {len(clinics)} clinics found."
    )