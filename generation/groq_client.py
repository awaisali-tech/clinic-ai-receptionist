from groq import Groq

from config.settings import GROQ_API_KEY, GROQ_MODEL


class GroqClient:
    """
    Small wrapper around the Groq API.

    Keeps API-specific code separate from
    the rest of the application.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(
            api_key=self.api_key
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        """
        Send messages to Groq and return the generated text.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()