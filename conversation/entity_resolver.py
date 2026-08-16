from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationContext:
    """
    Stores the active entities for the current conversation.
    """

    active_doctor: Optional[str] = None
    active_clinic: Optional[str] = None
    active_specialization: Optional[str] = None

    def reset(self):
        """Clear the current conversation context."""

        self.active_doctor = None
        self.active_clinic = None
        self.active_specialization = None


class ContextResolver:
    """
    Resolves references to previously mentioned entities.

    Example:

        User:
        "When is Dr. Ayesha Khan available?"

        Context:
        active_doctor = "Dr. Ayesha Khan"

        User:
        "What about Saturday?"

        Resolved query:
        "When is Dr. Ayesha Khan available on Saturday?"
    """

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

    def __init__(self):
        self.context = ConversationContext()

    def update(
        self,
        doctor: Optional[str] = None,
        clinic: Optional[str] = None,
        specialization: Optional[str] = None,
    ):
        """
        Update active entities.
        """

        if doctor:
            self.context.active_doctor = doctor

        if clinic:
            self.context.active_clinic = clinic

        if specialization:
            self.context.active_specialization = specialization

    def resolve(
        self,
        query: str,
    ) -> dict:
        """
        Resolve the current query using conversation context.
        """

        normalized_query = query.lower().strip()

        # -----------------------------------------
        # Explicit reset
        # -----------------------------------------

        if any(
            phrase in normalized_query
            for phrase in self.RESET_WORDS
        ):
            self.context.reset()

            return {
                "query": query,
                "doctor": None,
                "clinic": None,
                "specialization": None,
                "is_follow_up": False,
            }

        # -----------------------------------------
        # Detect whether this looks like a follow-up
        # -----------------------------------------

        is_follow_up = any(
    phrase in normalized_query
    for phrase in self.FOLLOW_UP_PHRASES
)

        # -----------------------------------------
        # If it is a follow-up, use active context
        # -----------------------------------------

        if is_follow_up:

            return {
                "query": query,
                "doctor": self.context.active_doctor,
                "clinic": self.context.active_clinic,
                "specialization": self.context.active_specialization,
                "is_follow_up": True,
            }

        # -----------------------------------------
        # Otherwise return current context only
        # when appropriate.
        # -----------------------------------------

        return {
            "query": query,
            "doctor": self.context.active_doctor,
            "clinic": self.context.active_clinic,
            "specialization": self.context.active_specialization,
            "is_follow_up": False,
        }