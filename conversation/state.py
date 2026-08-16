from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationState:
    """
    Represents the current state of one user conversation.
    """

    active_doctor: Optional[str] = None
    active_clinic: Optional[str] = None
    active_specialization: Optional[str] = None

    last_query: Optional[str] = None
    last_intent: Optional[str] = None

    turn_count: int = 0

    metadata: dict = field(default_factory=dict)

    def clear_entities(self):
        """Clear active clinic-related entities."""

        self.active_doctor = None
        self.active_clinic = None
        self.active_specialization = None

    def reset(self):
        """Completely reset conversation state."""

        self.clear_entities()

        self.last_query = None
        self.last_intent = None

        self.turn_count = 0

        self.metadata.clear()