from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProviderFailureKind(str, Enum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_REQUEST = "invalid_request"
    MALFORMED_RESPONSE = "malformed_response"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GenerationFailure:
    """Provider-neutral diagnostics safe to retain on a pipeline result."""

    kind: ProviderFailureKind
    retryable: bool
    attempts: int


class ProviderError(RuntimeError):
    """A provider-neutral exception that never stores raw provider details."""

    def __init__(
        self,
        kind: ProviderFailureKind,
        *,
        retryable: bool,
        attempts: int,
    ):
        self.failure = GenerationFailure(
            kind=kind,
            retryable=retryable,
            attempts=attempts,
        )
        super().__init__(f"Provider request failed: {kind.value}.")

    @property
    def kind(self) -> ProviderFailureKind:
        return self.failure.kind

    @property
    def retryable(self) -> bool:
        return self.failure.retryable

    @property
    def attempts(self) -> int:
        return self.failure.attempts

    def with_attempts(self, attempts: int) -> "ProviderError":
        return ProviderError(
            self.kind,
            retryable=self.retryable,
            attempts=attempts,
        )
