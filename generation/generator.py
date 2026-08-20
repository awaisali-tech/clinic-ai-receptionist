from collections.abc import Callable
from typing import Any

from generation.groq_client import GroqClient
from generation.prompt_builder import PromptBuilder


class Generator:
    """
    Generates the final clinic receptionist response.

    This class connects the PromptBuilder with the
    GroqClient.
    """

    def __init__(
        self,
        groq_client: GroqClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        client_factory: Callable[[], GroqClient] = GroqClient,
    ):
        self.groq_client = groq_client
        self._client_factory = client_factory
        self.prompt_builder = (
            prompt_builder or PromptBuilder()
        )

    def generate(
        self,
        query: str,
        evidence: list[Any],
        conversation_context: dict | None = None,
    ) -> str:
        """
        Generate an answer from a privacy-minimized query and grounded evidence.
        """

        messages = self.prompt_builder.build(
            query=query,
            evidence=evidence,
            conversation_context=conversation_context,
        )

        if self.groq_client is None:
            self.groq_client = self._client_factory()

        return self.groq_client.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
