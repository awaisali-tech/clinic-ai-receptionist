from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx
from groq import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    Groq,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from config.settings import GROQ_API_KEY, GROQ_MODEL
from generation.provider_errors import ProviderError, ProviderFailureKind
from generation.reliability import ProviderReliabilityConfig


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
        reliability_config: ProviderReliabilityConfig | None = None,
        *,
        sdk_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        sdk_factory: Callable[..., Any] = Groq,
    ):
        self.reliability_config = (
            reliability_config or ProviderReliabilityConfig.from_environment()
        )
        self.api_key = api_key if api_key is not None else GROQ_API_KEY
        self.model = model if model is not None else GROQ_MODEL
        self._sleep = sleep

        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ProviderError(
                ProviderFailureKind.CONFIGURATION,
                retryable=False,
                attempts=0,
            ) from None
        if not isinstance(self.model, str) or not self.model.strip():
            raise ProviderError(
                ProviderFailureKind.CONFIGURATION,
                retryable=False,
                attempts=0,
            ) from None

        self._timeout = httpx.Timeout(
            self.reliability_config.timeout_seconds,
            connect=self.reliability_config.connect_timeout_seconds,
        )
        if sdk_client is not None:
            self.client = sdk_client
            return

        try:
            self.client = sdk_factory(
                api_key=self.api_key,
                timeout=self._timeout,
                max_retries=0,
            )
        except Exception:
            raise ProviderError(
                ProviderFailureKind.CONFIGURATION,
                retryable=False,
                attempts=0,
            ) from None

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        """
        Send messages to Groq and return the generated text.
        """

        for attempt in range(1, self.reliability_config.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self._timeout,
                )
            except Exception as error:
                provider_error = self._translate_exception(error, attempt)
                if (
                    provider_error.retryable
                    and attempt < self.reliability_config.max_attempts
                ):
                    self._sleep(
                        self.reliability_config.backoff_seconds(
                            attempt,
                            self._retry_after_seconds(error),
                        )
                    )
                    continue
                raise provider_error from None

            return self._parse_response(response, attempt)

        raise ProviderError(
            ProviderFailureKind.UNKNOWN,
            retryable=False,
            attempts=self.reliability_config.max_attempts,
        ) from None

    @staticmethod
    def _parse_response(response: Any, attempt: int) -> str:
        try:
            choices = getattr(response, "choices", None)
            if not isinstance(choices, (list, tuple)) or not choices:
                raise TypeError
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise TypeError
            return content.strip()
        except Exception:
            raise ProviderError(
                ProviderFailureKind.MALFORMED_RESPONSE,
                retryable=False,
                attempts=attempt,
            ) from None

    @staticmethod
    def _translate_exception(error: Exception, attempt: int) -> ProviderError:
        if isinstance(error, ProviderError):
            return error.with_attempts(attempt)
        if isinstance(error, (APITimeoutError, httpx.TimeoutException)):
            return ProviderError(
                ProviderFailureKind.TIMEOUT,
                retryable=True,
                attempts=attempt,
            )
        if isinstance(error, (APIConnectionError, httpx.TransportError)):
            return ProviderError(
                ProviderFailureKind.CONNECTION,
                retryable=True,
                attempts=attempt,
            )
        if isinstance(error, RateLimitError):
            return ProviderError(
                ProviderFailureKind.RATE_LIMIT,
                retryable=True,
                attempts=attempt,
            )
        if isinstance(error, AuthenticationError):
            return ProviderError(
                ProviderFailureKind.AUTHENTICATION,
                retryable=False,
                attempts=attempt,
            )
        if isinstance(error, PermissionDeniedError):
            return ProviderError(
                ProviderFailureKind.PERMISSION,
                retryable=False,
                attempts=attempt,
            )
        if isinstance(
            error,
            (BadRequestError, UnprocessableEntityError, NotFoundError),
        ):
            return ProviderError(
                ProviderFailureKind.INVALID_REQUEST,
                retryable=False,
                attempts=attempt,
            )
        if isinstance(error, (InternalServerError, ConflictError)):
            return ProviderError(
                ProviderFailureKind.SERVICE_UNAVAILABLE,
                retryable=True,
                attempts=attempt,
            )
        if isinstance(error, APIResponseValidationError):
            return ProviderError(
                ProviderFailureKind.MALFORMED_RESPONSE,
                retryable=False,
                attempts=attempt,
            )
        if isinstance(error, APIStatusError):
            return GroqClient._translate_status(error, attempt)
        if isinstance(error, APIError):
            return ProviderError(
                ProviderFailureKind.UNKNOWN,
                retryable=False,
                attempts=attempt,
            )
        return ProviderError(
            ProviderFailureKind.UNKNOWN,
            retryable=False,
            attempts=attempt,
        )

    @staticmethod
    def _translate_status(error: APIStatusError, attempt: int) -> ProviderError:
        status = getattr(error, "status_code", None)
        if status == 401:
            kind, retryable = ProviderFailureKind.AUTHENTICATION, False
        elif status == 403:
            kind, retryable = ProviderFailureKind.PERMISSION, False
        elif status == 408:
            kind, retryable = ProviderFailureKind.TIMEOUT, True
        elif status == 429:
            kind, retryable = ProviderFailureKind.RATE_LIMIT, True
        elif status == 409 or (isinstance(status, int) and status >= 500):
            kind, retryable = ProviderFailureKind.SERVICE_UNAVAILABLE, True
        elif status in {400, 404, 422}:
            kind, retryable = ProviderFailureKind.INVALID_REQUEST, False
        else:
            kind, retryable = ProviderFailureKind.UNKNOWN, False
        return ProviderError(kind, retryable=retryable, attempts=attempt)

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if not isinstance(headers, Mapping):
            return None

        milliseconds = headers.get("retry-after-ms")
        seconds = headers.get("retry-after")
        try:
            if milliseconds is not None:
                return float(milliseconds) / 1000.0
            if seconds is not None:
                return float(seconds)
        except (TypeError, ValueError):
            return None
        return None
