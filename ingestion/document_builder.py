from dataclasses import dataclass
from typing import Any


@dataclass
class RAGDocument:
    """
    A searchable document used by the retrieval system.
    """

    text: str
    metadata: dict[str, Any]


def build_documents(data: dict[str, Any]) -> list[RAGDocument]:
    """
    Convert normalized clinic data into atomic RAG documents.

    Each document represents one specific piece of information:
    clinic, doctor, service, timing, or FAQ.
    """

    documents: list[RAGDocument] = []

    for clinic in data["clinics"]:

        clinic_id = clinic["id"]
        clinic_name = clinic["name"]

        # --------------------------------------------------
        # 1. Clinic information
        # --------------------------------------------------

        location = clinic["location"]
        contact = clinic["contact"]

        clinic_text = (
            f"Clinic: {clinic_name}\n"
            f"Location: {location}\n"
            f"Contact: {contact}\n"
            f"About: {clinic['about']}"
        )

        documents.append(
            RAGDocument(
                text=clinic_text,
                metadata={
                    "clinic_id": clinic_id,
                    "clinic_name": clinic_name,
                    "document_type": "clinic",
                },
            )
        )

        # --------------------------------------------------
        # 2. Doctors
        # --------------------------------------------------

        for doctor in clinic["doctors"]:

            doctor_name = doctor["name"]
            specialization = doctor["specialization"]

            doctor_text = (
                f"Doctor: {doctor_name}\n"
                f"Specialization: {specialization}\n"
                f"Clinic: {clinic_name}\n"
                f"Experience: "
                f"{doctor.get('experience_years', 'Not specified')} years\n"
                f"Availability: {doctor['availability']}"
            )

            documents.append(
                RAGDocument(
                    text=doctor_text,
                    metadata={
                        "clinic_id": clinic_id,
                        "clinic_name": clinic_name,
                        "document_type": "doctor",
                        "doctor_name": doctor_name,
                        "specialization": specialization,
                    },
                )
            )

        # --------------------------------------------------
        # 3. Services
        # --------------------------------------------------

        for service in clinic["services"]:

            service_text = (
                f"Service: {service}\n"
                f"Clinic: {clinic_name}"
            )

            documents.append(
                RAGDocument(
                    text=service_text,
                    metadata={
                        "clinic_id": clinic_id,
                        "clinic_name": clinic_name,
                        "document_type": "service",
                        "service_name": service,
                    },
                )
            )

        # --------------------------------------------------
        # 4. Clinic timings
        # --------------------------------------------------

        timings_text = (
            f"Clinic: {clinic_name}\n"
            f"Opening hours: {clinic['timings']}"
        )

        documents.append(
            RAGDocument(
                text=timings_text,
                metadata={
                    "clinic_id": clinic_id,
                    "clinic_name": clinic_name,
                    "document_type": "timings",
                },
            )
        )

        # --------------------------------------------------
        # 5. FAQs
        # --------------------------------------------------

        for faq in clinic["faqs"]:

            if isinstance(faq, dict):

                question = faq.get(
                    "question",
                    faq.get("q", "")
                )

                answer = faq.get(
                    "answer",
                    faq.get("a", "")
                )

                faq_text = (
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    f"Clinic: {clinic_name}"
                )

            else:
                faq_text = (
                    f"FAQ: {faq}\n"
                    f"Clinic: {clinic_name}"
                )

            documents.append(
                RAGDocument(
                    text=faq_text,
                    metadata={
                        "clinic_id": clinic_id,
                        "clinic_name": clinic_name,
                        "document_type": "faq",
                    },
                )
            )

    return documents