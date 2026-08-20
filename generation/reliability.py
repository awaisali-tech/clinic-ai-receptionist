from __future__ import annotations

import math
import os
from dataclasses import dataclass

from generation.provider_errors import ProviderError, ProviderFailureKind


@dataclass(frozen=True)
class ProviderReliabilityConfig:
    """Narrow, immutable reliability settings for synchronous generation."""

    timeout_seconds: float = 20.0
    connect_timeout_seconds: float = 5.0
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        numeric_values = (
            self.timeout_seconds,
            self.connect_timeout_seconds,
            self.initial_backoff_seconds,
            self.max_backoff_seconds,
        )
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in numeric_values
        ):
            self._invalid()
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or not 1 <= self.max_attempts <= 5
        ):
            self._invalid()
        if self.connect_timeout_seconds > self.timeout_seconds:
            self._invalid()
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            self._invalid()

    @classmethod
    def from_environment(cls) -> "ProviderReliabilityConfig":
        return cls(
            timeout_seconds=cls._float_env(
                "GROQ_TIMEOUT_SECONDS",
                cls.timeout_seconds,
            ),
            connect_timeout_seconds=cls._float_env(
                "GROQ_CONNECT_TIMEOUT_SECONDS",
                cls.connect_timeout_seconds,
            ),
            max_attempts=cls._int_env(
                "GROQ_MAX_ATTEMPTS",
                cls.max_attempts,
            ),
            initial_backoff_seconds=cls._float_env(
                "GROQ_INITIAL_BACKOFF_SECONDS",
                cls.initial_backoff_seconds,
            ),
            max_backoff_seconds=cls._float_env(
                "GROQ_MAX_BACKOFF_SECONDS",
                cls.max_backoff_seconds,
            ),
        )

    def backoff_seconds(
        self,
        failed_attempt: int,
        retry_after_seconds: float | None = None,
    ) -> float:
        if (
            retry_after_seconds is not None
            and math.isfinite(retry_after_seconds)
            and retry_after_seconds > 0
        ):
            return min(retry_after_seconds, self.max_backoff_seconds)

        exponential = self.initial_backoff_seconds * (2 ** (failed_attempt - 1))
        return min(exponential, self.max_backoff_seconds)

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            ProviderReliabilityConfig._invalid()

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            ProviderReliabilityConfig._invalid()

    @staticmethod
    def _invalid() -> None:
        raise ProviderError(
            ProviderFailureKind.CONFIGURATION,
            retryable=False,
            attempts=0,
        ) from None
