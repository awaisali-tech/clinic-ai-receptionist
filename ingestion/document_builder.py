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
                    "document_id": f"{clinic_id}:clinic",
                    "clinic_id": clinic_id,
                    "clinic_name": clinic_name,
                    "document_type": "clinic",
                    "information_types": (
                        "clinic_information",
                        "clinic_location",
                        "clinic_contact",
                    ),
                    "address": location.get("address"),
                    "phone": contact.get("phone"),
                    "email": contact.get("email"),
                    "about": clinic["about"],
                },
            )
        )

        # --------------------------------------------------
        # 2. Doctors
        # --------------------------------------------------

        for doctor_index, doctor in enumerate(clinic["doctors"]):

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
                        "document_id": (
                            f"{clinic_id}:doctor:{doctor_index}"
                        ),
                        "clinic_id": clinic_id,
                        "clinic_name": clinic_name,
                        "document_type": "doctor",
                        "information_types": (
                            "doctor_information",
                            "doctor_availability",
                        ),
                        "doctor_name": doctor_name,
                        "specialization": specialization,
                        "experience_years": doctor.get(
                            "experience_years"
                        ),
                        "availability": doctor["availability"],
                    },
                )
            )

        # --------------------------------------------------
        # 3. Services
        # --------------------------------------------------

        for service_index, service in enumerate(clinic["services"]):

            service_text = (
                f"Service: {service}\n"
                f"Clinic: {clinic_name}"
            )

            documents.append(
                RAGDocument(
                    text=service_text,
                    metadata={
                        "document_id": (
                            f"{clinic_id}:service:{service_index}"
                        ),
                        "clinic_id": clinic_id,
                        "clinic_name": clinic_name,
                        "document_type": "service",
                        "information_types": ("services",),
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
                    "document_id": f"{clinic_id}:timings",
                    "clinic_id": clinic_id,
                    "clinic_name": clinic_name,
                    "document_type": "timings",
                    "information_types": ("clinic_timings",),
                    "timings": dict(clinic["timings"]),
                },
            )
        )

        # --------------------------------------------------
        # 5. FAQs
        # --------------------------------------------------

        for faq_index, faq in enumerate(clinic["faqs"]):

            question = ""
            answer = ""

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
                        "document_id": f"{clinic_id}:faq:{faq_index}",
                        "clinic_id": clinic_id,
                        "clinic_name": clinic_name,
                        "document_type": "faq",
                        "information_types": ("faq",),
                        "faq_question": question,
                        "faq_answer": answer or str(faq),
                    },
                )
            )

    return documents
