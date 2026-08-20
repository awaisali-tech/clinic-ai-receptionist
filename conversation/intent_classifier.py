from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from conversation.entity_resolver import QueryResolution, normalize_entity_text


class InformationNeed(str, Enum):
    DOCTOR_INFORMATION = "doctor_information"
    DOCTOR_AVAILABILITY = "doctor_availability"
    CLINIC_LOCATION = "clinic_location"
    CLINIC_TIMINGS = "clinic_timings"
    CLINIC_CONTACT = "clinic_contact"
    SERVICES = "services"
    CLINIC_INFORMATION = "clinic_information"
    FAQ = "faq"
    GENERAL = "general"


@dataclass(frozen=True)
class InformationNeedSpec:
    document_types: frozenset[str]
    information_type: str | None
    entity_fields: tuple[str, ...]
    answer_fields: tuple[str, ...] = ()
    requires_semantic_support: bool = False


INFORMATION_NEED_SPECS: Mapping[InformationNeed, InformationNeedSpec] = (
    MappingProxyType(
        {
            InformationNeed.DOCTOR_INFORMATION: InformationNeedSpec(
                document_types=frozenset({"doctor"}),
                information_type="doctor_information",
                entity_fields=("doctor", "clinic", "specialization"),
                answer_fields=("doctor_name",),
            ),
            InformationNeed.DOCTOR_AVAILABILITY: InformationNeedSpec(
                document_types=frozenset({"doctor"}),
                information_type="doctor_availability",
                entity_fields=("doctor", "clinic", "specialization"),
                answer_fields=("availability",),
            ),
            InformationNeed.CLINIC_LOCATION: InformationNeedSpec(
                document_types=frozenset({"clinic"}),
                information_type="clinic_location",
                entity_fields=("clinic",),
                answer_fields=("address",),
            ),
            InformationNeed.CLINIC_TIMINGS: InformationNeedSpec(
                document_types=frozenset({"timings"}),
                information_type="clinic_timings",
                entity_fields=("clinic",),
                answer_fields=("timings",),
            ),
            InformationNeed.CLINIC_CONTACT: InformationNeedSpec(
                document_types=frozenset({"clinic"}),
                information_type="clinic_contact",
                entity_fields=("clinic",),
                answer_fields=("phone", "email"),
            ),
            InformationNeed.SERVICES: InformationNeedSpec(
                document_types=frozenset({"service"}),
                information_type="services",
                entity_fields=("clinic", "specialization"),
                answer_fields=("service_name",),
            ),
            InformationNeed.CLINIC_INFORMATION: InformationNeedSpec(
                document_types=frozenset({"clinic"}),
                information_type="clinic_information",
                entity_fields=("clinic",),
                answer_fields=("about",),
            ),
            InformationNeed.FAQ: InformationNeedSpec(
                document_types=frozenset({"faq"}),
                information_type="faq",
                entity_fields=("clinic",),
                answer_fields=("faq_answer",),
                requires_semantic_support=True,
            ),
            InformationNeed.GENERAL: InformationNeedSpec(
                document_types=frozenset(
                    {"clinic", "doctor", "service", "timings", "faq"}
                ),
                information_type=None,
                entity_fields=("doctor", "clinic", "specialization"),
                requires_semantic_support=True,
            ),
        }
    )
)


def information_need_spec(need: InformationNeed) -> InformationNeedSpec:
    return INFORMATION_NEED_SPECS[need]


class IntentClassifier:
    """Deterministically classify administrative information needs."""

    LOCATION_PATTERN = re.compile(
        r"\b(where (?:is|are)|address|location|located|directions?)\b"
    )
    CONTACT_PATTERN = re.compile(
        r"\b(phone(?: number)?|telephone|email|contact(?: details?| number)?|call)\b"
    )
    TIMINGS_PATTERN = re.compile(
        r"\b(opening hours?|closing (?:hours?|time)|clinic hours?|timings?|"
        r"when .* (?:open|close)|what time .* (?:open|close)|open today|"
        r"closed today)\b"
    )
    AVAILABILITY_PATTERN = re.compile(
        r"\b(available|availability|doctor schedule|consultation schedule)\b"
    )
    SERVICE_PATTERN = re.compile(
        r"\b(services?|specialties|specialities|specializations|specialisations)\b"
    )
    DOCTOR_PATTERN = re.compile(
        r"\b(doctors?|physicians?|experience|specialization|specialisation)\b"
    )
    DAY_PATTERN = re.compile(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b"
    )
    FAQ_PATTERN = re.compile(
        r"^(do|does|can|could|are|is|will|what|how)\b"
    )

    def classify(
        self,
        query: str,
        resolution: QueryResolution | None = None,
    ) -> InformationNeed:
        text = normalize_entity_text(query)
        doctor = resolution.doctor.value if resolution else None
        clinic = resolution.clinic.value if resolution else None

        if self.LOCATION_PATTERN.search(text):
            return InformationNeed.CLINIC_LOCATION

        if self.TIMINGS_PATTERN.search(text):
            return InformationNeed.CLINIC_TIMINGS

        if clinic and re.search(r"\b(open|close|closed)\b", text):
            return InformationNeed.CLINIC_TIMINGS

        if self.AVAILABILITY_PATTERN.search(text) or (
            doctor and "schedule" in text
        ):
            return InformationNeed.DOCTOR_AVAILABILITY

        if self.DAY_PATTERN.search(text) and resolution and resolution.is_follow_up:
            if doctor:
                return InformationNeed.DOCTOR_AVAILABILITY
            if clinic:
                return InformationNeed.CLINIC_TIMINGS

        if self.SERVICE_PATTERN.search(text) or re.search(r"\boffer\b", text):
            return InformationNeed.SERVICES

        if "contact lens" not in text and self.CONTACT_PATTERN.search(text):
            return InformationNeed.CLINIC_CONTACT

        if self.DOCTOR_PATTERN.search(text) or (
            doctor
            and any(
                phrase in text
                for phrase in ("about", "tell me", "who is", "experience")
            )
        ):
            return InformationNeed.DOCTOR_INFORMATION

        if (
            (
                "about" in text
                or "clinic information" in text
                or "general information" in text
            )
            and (clinic or "clinic" in text or "center" in text)
        ):
            return InformationNeed.CLINIC_INFORMATION

        if self.FAQ_PATTERN.search(text):
            return InformationNeed.FAQ

        if doctor:
            return InformationNeed.DOCTOR_INFORMATION

        return InformationNeed.GENERAL
