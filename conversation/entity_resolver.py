from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal


EntitySource = Literal["explicit", "inherited"]


def normalize_entity_text(value: str) -> str:
    """Normalize user and catalog text for deterministic phrase matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        character if character.isalnum() else " "
        for character in normalized
    )
    return " ".join(normalized.split())


@dataclass(frozen=True)
class DoctorEntity:
    name: str
    clinic: str
    specialization: str


@dataclass(frozen=True)
class EntityCatalog:
    """Canonical entity names and doctor relationships from clinic data."""

    clinics: tuple[str, ...] = ()
    doctors: tuple[DoctorEntity, ...] = ()
    specializations: tuple[str, ...] = ()

    @classmethod
    def from_clinic_data(cls, data: dict[str, Any]) -> "EntityCatalog":
        clinics: list[str] = []
        doctors: list[DoctorEntity] = []
        specializations: list[str] = []

        for clinic in data.get("clinics", []):
            clinic_name = clinic["name"]
            clinics.append(clinic_name)

            for doctor in clinic.get("doctors", []):
                specialization = doctor["specialization"]
                doctors.append(
                    DoctorEntity(
                        name=doctor["name"],
                        clinic=clinic_name,
                        specialization=specialization,
                    )
                )
                specializations.append(specialization)

        return cls(
            clinics=tuple(dict.fromkeys(clinics)),
            doctors=tuple(doctors),
            specializations=tuple(dict.fromkeys(specializations)),
        )

    def doctor(self, name: str | None) -> DoctorEntity | None:
        if name is None:
            return None

        normalized_name = normalize_entity_text(name)
        return next(
            (
                doctor
                for doctor in self.doctors
                if normalize_entity_text(doctor.name) == normalized_name
            ),
            None,
        )

    def match(self, query: str) -> tuple[str | None, str | None, str | None]:
        """Return unambiguous doctor, clinic, and specialization mentions."""

        normalized_query = normalize_entity_text(query)
        doctor = self._match_doctor(normalized_query)
        clinic = self._match_value(normalized_query, self.clinics)
        specialization = self._match_value(
            normalized_query,
            self.specializations,
        )
        return doctor, clinic, specialization

    def _match_doctor(self, normalized_query: str) -> str | None:
        matches: list[str] = []

        for doctor in self.doctors:
            aliases = {normalize_entity_text(doctor.name)}
            for name in tuple(aliases):
                if name.startswith("dr "):
                    aliases.add(name.removeprefix("dr "))
                if name.startswith("doctor "):
                    aliases.add(name.removeprefix("doctor "))

            if any(self._contains_phrase(normalized_query, alias) for alias in aliases):
                matches.append(doctor.name)

        unique_matches = tuple(dict.fromkeys(matches))
        return unique_matches[0] if len(unique_matches) == 1 else None

    @classmethod
    def _match_value(
        cls,
        normalized_query: str,
        values: tuple[str, ...],
    ) -> str | None:
        matches = tuple(
            value
            for value in values
            if cls._contains_phrase(
                normalized_query,
                normalize_entity_text(value),
            )
        )
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        if not phrase:
            return False
        return f" {phrase} " in f" {text} "


@dataclass
class ConversationContext:
    """The last successfully committed conversational entity context."""

    active_doctor: str | None = None
    active_clinic: str | None = None
    active_specialization: str | None = None

    def reset(self) -> None:
        self.active_doctor = None
        self.active_clinic = None
        self.active_specialization = None

    def copy(self) -> "ConversationContext":
        return ConversationContext(
            active_doctor=self.active_doctor,
            active_clinic=self.active_clinic,
            active_specialization=self.active_specialization,
        )


@dataclass(frozen=True)
class ResolvedEntity:
    """A resolved canonical entity together with its conversational origin."""

    value: str | None = None
    source: EntitySource | None = None

    @property
    def is_explicit(self) -> bool:
        return self.source == "explicit"

    @property
    def is_inherited(self) -> bool:
        return self.source == "inherited"


@dataclass(frozen=True)
class QueryResolution:
    """Typed, non-mutating resolution result for one user query."""

    query: str
    doctor: ResolvedEntity
    clinic: ResolvedEntity
    specialization: ResolvedEntity
    is_follow_up: bool = False
    reset_requested: bool = False

    @property
    def has_explicit_entities(self) -> bool:
        return any(
            entity.is_explicit
            for entity in (self.doctor, self.clinic, self.specialization)
        )


class ContextResolver:
    """Resolve explicit entities and safe conversational inheritance."""

    FOLLOW_UP_PHRASES = {
        "what about",
        "how about",
        "what time",
        "which day",
        "which days",
        "is she",
        "is he",
        "does she",
        "does he",
        "does it",
        "what about that",
    }

    RESET_WORDS = {
        "new question",
        "different question",
        "another question",
        "start over",
    }

    PRONOUN_PATTERN = re.compile(
        r"\b(she|he|her|him|they|them|it|that doctor|that clinic)\b"
    )
    DAY_PATTERN = re.compile(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b"
    )

    def __init__(self, catalog: EntityCatalog | None = None):
        self.catalog = catalog or EntityCatalog()
        self._context = ConversationContext()

    def get_context(self) -> ConversationContext:
        """Return a snapshot so callers cannot mutate stored state directly."""

        return self._context.copy()

    def reset(self) -> None:
        self._context.reset()

    def update(
        self,
        doctor: str | None = None,
        clinic: str | None = None,
        specialization: str | None = None,
    ) -> None:
        """Explicitly seed context; retained for callers outside the pipeline."""

        if doctor:
            self._context.active_doctor = doctor
        if clinic:
            self._context.active_clinic = clinic
        if specialization:
            self._context.active_specialization = specialization

    def resolve(self, query: str) -> QueryResolution:
        """Resolve a query without changing the committed conversation state."""

        normalized_query = normalize_entity_text(query)
        reset_requested = any(
            self._contains_phrase(normalized_query, normalize_entity_text(phrase))
            for phrase in self.RESET_WORDS
        )
        explicit_doctor, explicit_clinic, explicit_specialization = (
            self.catalog.match(query)
        )

        is_follow_up = (
            not reset_requested
            and self._has_context()
            and self._looks_like_follow_up(normalized_query)
        )

        doctor = self._resolved_entity(explicit_doctor)
        clinic = self._resolved_entity(explicit_clinic)
        specialization = self._resolved_entity(explicit_specialization)

        if is_follow_up:
            doctor = self._inherit_doctor(
                doctor,
                explicit_clinic,
                explicit_specialization,
            )
            clinic = self._inherit_clinic(
                clinic,
                explicit_doctor,
                explicit_specialization,
            )
            specialization = self._inherit_specialization(
                specialization,
                explicit_doctor,
                explicit_clinic,
            )

        return QueryResolution(
            query=query,
            doctor=doctor,
            clinic=clinic,
            specialization=specialization,
            is_follow_up=is_follow_up,
            reset_requested=reset_requested,
        )

    def candidate_context(self, resolution: QueryResolution) -> ConversationContext:
        """Build the context that may be committed after a successful turn."""

        doctor = resolution.doctor.value
        clinic = resolution.clinic.value
        specialization = resolution.specialization.value

        doctor_entity = self.catalog.doctor(doctor)
        if resolution.doctor.is_explicit and doctor_entity is not None:
            clinic = clinic or doctor_entity.clinic
            specialization = specialization or doctor_entity.specialization

        return ConversationContext(
            active_doctor=doctor,
            active_clinic=clinic,
            active_specialization=specialization,
        )

    def commit(self, resolution: QueryResolution) -> ConversationContext:
        """Atomically replace stored context after a successful pipeline turn."""

        candidate = self.candidate_context(resolution)
        self._context = candidate.copy()
        return candidate

    def _resolved_entity(self, explicit_value: str | None) -> ResolvedEntity:
        if explicit_value is None:
            return ResolvedEntity()
        return ResolvedEntity(value=explicit_value, source="explicit")

    def _inherit_doctor(
        self,
        resolved: ResolvedEntity,
        explicit_clinic: str | None,
        explicit_specialization: str | None,
    ) -> ResolvedEntity:
        if resolved.value is not None or self._context.active_doctor is None:
            return resolved

        doctor = self.catalog.doctor(self._context.active_doctor)
        if doctor is not None:
            if explicit_clinic and doctor.clinic != explicit_clinic:
                return resolved
            if explicit_specialization and doctor.specialization != explicit_specialization:
                return resolved

        return ResolvedEntity(self._context.active_doctor, "inherited")

    def _inherit_clinic(
        self,
        resolved: ResolvedEntity,
        explicit_doctor: str | None,
        explicit_specialization: str | None,
    ) -> ResolvedEntity:
        active_clinic = self._context.active_clinic
        if resolved.value is not None or active_clinic is None:
            return resolved

        explicit_doctor_entity = self.catalog.doctor(explicit_doctor)
        if explicit_doctor_entity and explicit_doctor_entity.clinic != active_clinic:
            return resolved
        if (
            explicit_specialization
            and self._context.active_specialization != explicit_specialization
        ):
            return resolved

        return ResolvedEntity(active_clinic, "inherited")

    def _inherit_specialization(
        self,
        resolved: ResolvedEntity,
        explicit_doctor: str | None,
        explicit_clinic: str | None,
    ) -> ResolvedEntity:
        active_specialization = self._context.active_specialization
        if resolved.value is not None or active_specialization is None:
            return resolved

        explicit_doctor_entity = self.catalog.doctor(explicit_doctor)
        if (
            explicit_doctor_entity
            and explicit_doctor_entity.specialization != active_specialization
        ):
            return resolved
        if explicit_clinic and self._context.active_clinic != explicit_clinic:
            return resolved

        return ResolvedEntity(active_specialization, "inherited")

    def _has_context(self) -> bool:
        return any(
            (
                self._context.active_doctor,
                self._context.active_clinic,
                self._context.active_specialization,
            )
        )

    def _looks_like_follow_up(self, normalized_query: str) -> bool:
        if any(
            self._contains_phrase(normalized_query, normalize_entity_text(phrase))
            for phrase in self.FOLLOW_UP_PHRASES
        ):
            return True
        if self.PRONOUN_PATTERN.search(normalized_query):
            return True
        return bool(
            self.DAY_PATTERN.search(normalized_query)
            and len(normalized_query.split()) <= 6
        )

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        return f" {phrase} " in f" {text} "
