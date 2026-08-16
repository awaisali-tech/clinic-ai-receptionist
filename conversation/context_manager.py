from conversation.state import ConversationState


class ContextManager:
    """
    Manages conversation state across multiple turns.
    """

    def __init__(self):
        self.state = ConversationState()

    def update(
        self,
        query: str,
        intent: str | None = None,
        doctor: str | None = None,
        clinic: str | None = None,
        specialization: str | None = None,
    ):
        """
        Update conversation state using information
        extracted from the current user query.
        """

        self.state.turn_count += 1
        self.state.last_query = query

        if intent:
            self.state.last_intent = intent

        if doctor:
            self.state.active_doctor = doctor

        if clinic:
            self.state.active_clinic = clinic

        if specialization:
            self.state.active_specialization = specialization

    def get_state(self) -> ConversationState:
        """Return current conversation state."""

        return self.state

    def reset(self):
        """Reset the complete conversation."""

        self.state.reset()

    def clear_entities(self):
        """Clear only active entities."""

        self.state.clear_entities()