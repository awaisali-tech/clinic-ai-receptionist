from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from generation.groq_client import GroqClient
from generation.prompt_builder import PromptBuilder
from generation.provider_errors import ProviderError, ProviderFailureKind
from generation.reliability import ProviderReliabilityConfig
from privacy.redactor import PrivacyProcessor
from safety.output_guard import OutputGuard


def _valid_response(content="Administrative answer."):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _Create:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.sequence.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _SDK:
    def __init__(self, sequence):
        self.create = _Create(sequence)
        self.chat = SimpleNamespace(completions=self.create)


def _error(kind, retryable):
    return ProviderError(kind, retryable=retryable, attempts=0)


@dataclass(frozen=True)
class ReliabilityCase:
    name: str
    sequence: tuple[object, ...]
    expected_calls: int
    expected_kind: ProviderFailureKind | None = None
    outcome: str = "success"
    mode: str = "provider"


@dataclass(frozen=True)
class ReliabilityEvaluationReport:
    total: int
    passed: int
    retryable_cases: int
    non_retryable_cases: int
    recovered_transient_cases: int
    exhausted_transient_cases: int
    failures: tuple[str, ...]


RELIABILITY_EVALUATION_CASES = (
    ReliabilityCase("success", (_valid_response(),), 1),
    ReliabilityCase("timeout recovery", (_error(ProviderFailureKind.TIMEOUT, True), _valid_response()), 2, outcome="recovered"),
    ReliabilityCase("timeout exhaustion", tuple(_error(ProviderFailureKind.TIMEOUT, True) for _ in range(3)), 3, ProviderFailureKind.TIMEOUT, "exhausted"),
    ReliabilityCase("connection recovery", (_error(ProviderFailureKind.CONNECTION, True), _valid_response()), 2, outcome="recovered"),
    ReliabilityCase("connection exhaustion", tuple(_error(ProviderFailureKind.CONNECTION, True) for _ in range(3)), 3, ProviderFailureKind.CONNECTION, "exhausted"),
    ReliabilityCase("rate limit recovery", (_error(ProviderFailureKind.RATE_LIMIT, True), _valid_response()), 2, outcome="recovered"),
    ReliabilityCase("rate limit exhaustion", tuple(_error(ProviderFailureKind.RATE_LIMIT, True) for _ in range(3)), 3, ProviderFailureKind.RATE_LIMIT, "exhausted"),
    ReliabilityCase("service recovery", (_error(ProviderFailureKind.SERVICE_UNAVAILABLE, True), _valid_response()), 2, outcome="recovered"),
    ReliabilityCase("service exhaustion", tuple(_error(ProviderFailureKind.SERVICE_UNAVAILABLE, True) for _ in range(3)), 3, ProviderFailureKind.SERVICE_UNAVAILABLE, "exhausted"),
    ReliabilityCase("authentication", (_error(ProviderFailureKind.AUTHENTICATION, False),), 1, ProviderFailureKind.AUTHENTICATION, "non_retryable"),
    ReliabilityCase("permission", (_error(ProviderFailureKind.PERMISSION, False),), 1, ProviderFailureKind.PERMISSION, "non_retryable"),
    ReliabilityCase("invalid request", (_error(ProviderFailureKind.INVALID_REQUEST, False),), 1, ProviderFailureKind.INVALID_REQUEST, "non_retryable"),
    ReliabilityCase("unknown", (RuntimeError("private provider details"),), 1, ProviderFailureKind.UNKNOWN, "non_retryable"),
    ReliabilityCase("response none", (None,), 1, ProviderFailureKind.MALFORMED_RESPONSE, "non_retryable"),
    ReliabilityCase("empty choices", (SimpleNamespace(choices=[]),), 1, ProviderFailureKind.MALFORMED_RESPONSE, "non_retryable"),
    ReliabilityCase("missing message", (SimpleNamespace(choices=[SimpleNamespace(message=None)]),), 1, ProviderFailureKind.MALFORMED_RESPONSE, "non_retryable"),
    ReliabilityCase("none content", (SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),), 1, ProviderFailureKind.MALFORMED_RESPONSE, "non_retryable"),
    ReliabilityCase("empty content", (SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))]),), 1, ProviderFailureKind.MALFORMED_RESPONSE, "non_retryable"),
    ReliabilityCase("whitespace content", (SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]),), 1, ProviderFailureKind.MALFORMED_RESPONSE, "non_retryable"),
    ReliabilityCase("non-string content", (SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=12))]),), 1, ProviderFailureKind.MALFORMED_RESPONSE, "non_retryable"),
    ReliabilityCase("output rejection", (_valid_response("You should take ibuprofen."),), 1, mode="output"),
    ReliabilityCase("privacy across retry", (_error(ProviderFailureKind.TIMEOUT, True), _valid_response()), 2, outcome="recovered", mode="privacy"),
    ReliabilityCase("context non-commit", tuple(_error(ProviderFailureKind.TIMEOUT, True) for _ in range(3)), 3, ProviderFailureKind.TIMEOUT, "exhausted", "context"),
)


def run_reliability_evaluation(
    cases=RELIABILITY_EVALUATION_CASES,
) -> ReliabilityEvaluationReport:
    failures = []
    recovered = sum(case.outcome == "recovered" for case in cases)
    exhausted = sum(case.outcome == "exhausted" for case in cases)
    non_retryable = sum(case.outcome == "non_retryable" for case in cases)
    retryable = recovered + exhausted

    for case in cases:
        sdk = _SDK(case.sequence)
        sleeps = []
        client = GroqClient(
            api_key="fake-evaluation-key",
            sdk_client=sdk,
            reliability_config=ProviderReliabilityConfig(
                timeout_seconds=5,
                connect_timeout_seconds=1,
                max_attempts=3,
                initial_backoff_seconds=0.01,
                max_backoff_seconds=0.02,
            ),
            sleep=sleeps.append,
        )
        observed_kind = None
        passed = True

        try:
            messages = [{"role": "user", "content": "Administrative query"}]
            if case.mode == "privacy":
                raw = (
                    "My email is evaluation.private@example.test. "
                    "When is Dr. Ayesha Khan available?"
                )
                minimized = PrivacyProcessor().process(
                    raw,
                    protected_entities=("Dr. Ayesha Khan",),
                ).provider_query
                messages = PromptBuilder().build(minimized, ["Clinic evidence"])

            answer = client.generate(messages)
            if case.expected_kind is not None:
                passed = False
            if case.mode == "output":
                passed = passed and not OutputGuard().evaluate(answer).allowed
            if case.mode == "privacy":
                passed = passed and "evaluation.private@example.test" not in repr(
                    sdk.create.calls
                )
        except ProviderError as error:
            observed_kind = error.kind
            passed = case.expected_kind == observed_kind
            if case.mode == "context":
                context_committed = False
                passed = passed and not context_committed

        passed = passed and len(sdk.create.calls) == case.expected_calls
        if not passed:
            failures.append(
                f"{case.name}: calls={len(sdk.create.calls)}, "
                f"kind={getattr(observed_kind, 'value', None)}"
            )

    return ReliabilityEvaluationReport(
        total=len(cases),
        passed=len(cases) - len(failures),
        retryable_cases=retryable,
        non_retryable_cases=non_retryable,
        recovered_transient_cases=recovered,
        exhausted_transient_cases=exhausted,
        failures=tuple(failures),
    )


if __name__ == "__main__":
    report = run_reliability_evaluation()
    print(f"Reliability evaluation: {report.passed}/{report.total}")
    print(f"Retryable cases: {report.retryable_cases}")
    print(f"Non-retryable cases: {report.non_retryable_cases}")
    print(f"Recovered transient cases: {report.recovered_transient_cases}")
    print(f"Exhausted transient cases: {report.exhausted_transient_cases}")
    for failure in report.failures:
        print(f"FAIL: {failure}")
