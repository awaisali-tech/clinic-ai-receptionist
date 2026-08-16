class PromptBuilder:
    """
    Builds grounded prompts for the clinic AI receptionist.

    The model must answer using only the supplied
    clinic evidence.
    """

    SYSTEM_PROMPT = """
You are an AI receptionist for a medical clinic.

Your job is to help users with administrative and
clinic-information questions.

You may answer questions about:
- Clinics
- Doctors
- Specializations
- Availability
- Clinic services
- Contact information
- Locations
- General clinic information

IMPORTANT RULES:

1. Use ONLY the information provided in the retrieved
   clinic evidence.

2. NEVER invent or assume information.

3. If the retrieved evidence does not contain the
   requested information, say that the information is
   not available in the clinic records.

4. Do not provide medical diagnosis.

5. Do not prescribe medicines.

6. Do not recommend specific medical treatments.

7. If a user asks for medical advice, recommend that
   they consult a qualified healthcare professional.

8. Keep answers clear, concise, and professional.

9. When answering about doctor availability, use the
   exact schedule provided in the evidence.

10. If the user asks about a day that is not included
    in a doctor's availability, do not assume the doctor
    is available on that day.

You are a clinic receptionist, not a doctor.
"""

    def build(
        self,
        query: str,
        evidence: list[str],
        conversation_context: dict | None = None,
    ) -> list[dict[str, str]]:
        """
        Build messages suitable for the Groq chat API.
        """

        evidence_text = self._format_evidence(
            evidence
        )

        context_text = self._format_context(
            conversation_context
        )

        user_prompt = f"""
Clinic evidence:

{evidence_text}

Conversation context:

{context_text}

User question:

{query}

Answer the user's question using only the
clinic evidence above.
"""

        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ]

    @staticmethod
    def _format_evidence(
        evidence: list[str],
    ) -> str:

        if not evidence:
            return "No clinic evidence was retrieved."

        return "\n\n".join(
            f"[Evidence {index}]\n{text}"
            for index, text in enumerate(
                evidence,
                start=1,
            )
        )

    @staticmethod
    def _format_context(
        context: dict | None,
    ) -> str:

        if not context:
            return "No previous conversation context."

        parts = []

        doctor = context.get("doctor")
        clinic = context.get("clinic")
        specialization = context.get(
            "specialization"
        )

        if doctor:
            parts.append(f"Doctor: {doctor}")

        if clinic:
            parts.append(f"Clinic: {clinic}")

        if specialization:
            parts.append(
                f"Specialization: {specialization}"
            )

        if not parts:
            return "No active entities."

        return "\n".join(parts)