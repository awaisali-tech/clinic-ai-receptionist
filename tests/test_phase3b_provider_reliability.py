from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from groq import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from conversation.entity_resolver import ContextResolver, EntityCatalog
from conversation.intent_classifier import IntentClassifier
from generation.generator import Generator
from generation.groq_client import GroqClient
from generation.provider_errors import ProviderError, ProviderFailureKind
from generation.reliability import ProviderReliabilityConfig
from ingestion.document_builder import RAGDocument
from orchestration.pipeline import AnswerStatus, RAGPipeline
from retrieval.grounding_validator import GroundingResult
from retrieval.hybrid_retriever import RetrievalCandidate
from safety.input_guard import InputGuard
from safety.medical_guard import MedicalGuard
from safety.output_guard import OutputGuard


def valid_response(content: str = "Administrative answer."):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


class SequencedCreate:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.sequence.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeSDKClient:
    def __init__(self, sequence):
        self.create = SequencedCreate(sequence)
        self.chat = SimpleNamespace(completions=self.create)


def reliability_config(max_attempts: int = 3):
    return ProviderReliabilityConfig(
        timeout_seconds=10.0,
        connect_timeout_seconds=2.0,
        max_attempts=max_attempts,
        initial_backoff_seconds=0.1,
        max_backoff_seconds=0.4,
    )


def provider_client(sequence, *, max_attempts=3):
    sdk = FakeSDKClient(sequence)
    sleeps = []
    client = GroqClient(
        api_key="fake-key-for-tests",
        sdk_client=sdk,
        reliability_config=reliability_config(max_attempts),
        sleep=sleeps.append,
    )
    return client, sdk, sleeps


def request():
    return httpx.Request("POST", "https://provider.invalid/chat")


def status_exception(exception_type, status, message="safe test failure", headers=None):
    response = httpx.Response(
        status,
        request=request(),
        headers=headers or {},
    )
    if exception_type is APIResponseValidationError:
        return exception_type(response=response, body=None, message=message)
    return exception_type(message, response=response, body=None)


class ProviderConfigurationTests(unittest.TestCase):
    def test_sdk_retries_are_disabled_and_timeout_is_explicit(self):
        captured = {}
        sdk = FakeSDKClient([valid_response()])

        def factory(**kwargs):
            captured.update(kwargs)
            return sdk

        client = GroqClient(
            api_key="fake-key-for-tests",
            reliability_config=reliability_config(),
            sdk_factory=factory,
            sleep=lambda _: None,
        )
        client.generate([])

        self.assertEqual(captured["max_retries"], 0)
        self.assertIsInstance(captured["timeout"], httpx.Timeout)
        self.assertIsInstance(sdk.create.calls[0]["timeout"], httpx.Timeout)

    def test_missing_key_is_typed_configuration_failure(self):
        with self.assertRaises(ProviderError) as caught:
            GroqClient(api_key="", sdk_client=FakeSDKClient([]))

        self.assertEqual(caught.exception.kind, ProviderFailureKind.CONFIGURATION)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(caught.exception.attempts, 0)

    def test_invalid_reliability_configuration_is_typed(self):
        with self.assertRaises(ProviderError) as caught:
            ProviderReliabilityConfig(max_attempts=0)
        self.assertEqual(caught.exception.kind, ProviderFailureKind.CONFIGURATION)

        with patch.dict(os.environ, {"GROQ_MAX_ATTEMPTS": "not-a-number"}):
            with self.assertRaises(ProviderError):
                ProviderReliabilityConfig.from_environment()


class ProviderRetryTests(unittest.TestCase):
    def test_timeout_timeout_success_uses_three_attempts(self):
        client, sdk, sleeps = provider_client(
            [
                APITimeoutError(request()),
                APITimeoutError(request()),
                valid_response(" recovered "),
            ]
        )

        answer = client.generate([])

        self.assertEqual(answer, "recovered")
        self.assertEqual(len(sdk.create.calls), 3)
        self.assertEqual(sleeps, [0.1, 0.2])

    def test_timeout_exhaustion_is_bounded(self):
        client, sdk, sleeps = provider_client(
            [APITimeoutError(request()) for _ in range(3)]
        )

        with self.assertRaises(ProviderError) as caught:
            client.generate([])

        self.assertEqual(caught.exception.kind, ProviderFailureKind.TIMEOUT)
        self.assertEqual(caught.exception.attempts, 3)
        self.assertEqual(len(sdk.create.calls), 3)
        self.assertEqual(len(sleeps), 2)

    def test_connection_failure_then_success(self):
        client, sdk, sleeps = provider_client(
            [APIConnectionError(request=request()), valid_response()]
        )
        self.assertEqual(client.generate([]), "Administrative answer.")
        self.assertEqual(len(sdk.create.calls), 2)
        self.assertEqual(sleeps, [0.1])

    def test_transport_reset_and_read_timeout_are_transient(self):
        cases = (
            (
                httpx.ReadError("connection reset test", request=request()),
                ProviderFailureKind.CONNECTION,
            ),
            (
                httpx.ReadTimeout("read timeout test", request=request()),
                ProviderFailureKind.TIMEOUT,
            ),
        )
        for error, expected_kind in cases:
            with self.subTest(kind=expected_kind.value):
                client, sdk, _ = provider_client([error, valid_response()])
                self.assertEqual(client.generate([]), "Administrative answer.")
                self.assertEqual(len(sdk.create.calls), 2)

    def test_service_failure_then_success(self):
        error = status_exception(InternalServerError, 503)
        client, sdk, _ = provider_client([error, valid_response()])
        self.assertEqual(client.generate([]), "Administrative answer.")
        self.assertEqual(len(sdk.create.calls), 2)

    def test_rate_limit_recovery_and_bounded_retry_after(self):
        error = status_exception(
            RateLimitError,
            429,
            message="quota details that must not escape",
            headers={"retry-after": "30"},
        )
        client, sdk, sleeps = provider_client([error, valid_response()])

        self.assertEqual(client.generate([]), "Administrative answer.")
        self.assertEqual(len(sdk.create.calls), 2)
        self.assertEqual(sleeps, [0.4])

    def test_rate_limit_exhaustion(self):
        sequence = [status_exception(RateLimitError, 429) for _ in range(3)]
        client, sdk, _ = provider_client(sequence)

        with self.assertRaises(ProviderError) as caught:
            client.generate([])

        self.assertEqual(caught.exception.kind, ProviderFailureKind.RATE_LIMIT)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.attempts, 3)
        self.assertEqual(len(sdk.create.calls), 3)


class ProviderNonRetryTests(unittest.TestCase):
    def test_authentication_permission_and_invalid_request_are_not_retried(self):
        cases = (
            (
                status_exception(AuthenticationError, 401, "fake bearer secret"),
                ProviderFailureKind.AUTHENTICATION,
            ),
            (
                status_exception(PermissionDeniedError, 403, "account details"),
                ProviderFailureKind.PERMISSION,
            ),
            (
                status_exception(BadRequestError, 400, "request body details"),
                ProviderFailureKind.INVALID_REQUEST,
            ),
        )
        for error, kind in cases:
            with self.subTest(kind=kind.value):
                client, sdk, sleeps = provider_client([error])
                with self.assertRaises(ProviderError) as caught:
                    client.generate([])
                self.assertEqual(caught.exception.kind, kind)
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(caught.exception.attempts, 1)
                self.assertEqual(len(sdk.create.calls), 1)
                self.assertEqual(sleeps, [])
                self.assertNotIn("details", str(caught.exception))
                self.assertNotIn("secret", str(caught.exception))

    def test_unknown_exception_is_safe_and_not_retried(self):
        client, sdk, sleeps = provider_client(
            [RuntimeError("fake api key and private request body")]
        )
        with self.assertRaises(ProviderError) as caught:
            client.generate([])

        self.assertEqual(caught.exception.kind, ProviderFailureKind.UNKNOWN)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(len(sdk.create.calls), 1)
        self.assertEqual(sleeps, [])
        self.assertNotIn("private request", str(caught.exception))


class MalformedResponseTests(unittest.TestCase):
    def test_malformed_shapes_are_non_retryable(self):
        malformed = (
            None,
            SimpleNamespace(),
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(message=None)]),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=123))]
            ),
        )
        for response in malformed:
            with self.subTest(response=repr(response)):
                client, sdk, sleeps = provider_client([response])
                with self.assertRaises(ProviderError) as caught:
                    client.generate([])
                self.assertEqual(
                    caught.exception.kind,
                    ProviderFailureKind.MALFORMED_RESPONSE,
                )
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(caught.exception.attempts, 1)
                self.assertEqual(len(sdk.create.calls), 1)
                self.assertEqual(sleeps, [])

    def test_sdk_response_validation_error_maps_to_malformed(self):
        error = status_exception(APIResponseValidationError, 200)
        client, _, _ = provider_client([error])
        with self.assertRaises(ProviderError) as caught:
            client.generate([])
        self.assertEqual(
            caught.exception.kind,
            ProviderFailureKind.MALFORMED_RESPONSE,
        )


CATALOG = EntityCatalog.from_clinic_data(
    {
        "clinics": [
            {
                "name": "Sunrise Medical Center",
                "doctors": [
                    {
                        "name": "Dr. Ayesha Khan",
                        "specialization": "Pediatrics",
                    }
                ],
            }
        ]
    }
)


def candidate():
    document = RAGDocument(
        text=(
            "Doctor: Dr. Ayesha Khan\nClinic: Sunrise Medical Center\n"
            "Availability: Mon-Fri 9:00am-2:00pm"
        ),
        metadata={
            "document_id": "doctor:ayesha",
            "document_type": "doctor",
            "information_types": (
                "doctor_information",
                "doctor_availability",
            ),
            "doctor_name": "Dr. Ayesha Khan",
            "clinic_name": "Sunrise Medical Center",
            "specialization": "Pediatrics",
            "availability": "Mon-Fri 9:00am-2:00pm",
        },
    )
    return RetrievalCandidate(
        document=document,
        semantic_score=0.9,
        structured_match=True,
        entity_match=True,
        document_type_match=True,
        sources=("structured",),
        final_score=0.9,
        explicit_entity_match=True,
    )


class CountingRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [candidate()]


class AlwaysGrounded:
    def validate(self, query, results, **kwargs):
        return GroundingResult(True, list(results), "Grounded test evidence.")


class CountingContextResolver(ContextResolver):
    def __init__(self):
        super().__init__(CATALOG)
        self.commit_count = 0

    def commit(self, resolution):
        self.commit_count += 1
        return super().commit(resolution)


def make_pipeline(generator):
    retriever = CountingRetriever()
    resolver = CountingContextResolver()
    pipeline = RAGPipeline(
        retriever=retriever,
        context_resolver=resolver,
        intent_classifier=IntentClassifier(),
        grounding_validator=AlwaysGrounded(),
        generator=generator,
        input_guard=InputGuard(),
        medical_guard=MedicalGuard(),
        output_guard=OutputGuard(),
    )
    return pipeline, retriever, resolver


class PipelineReliabilityTests(unittest.TestCase):
    def test_single_successful_provider_call_runs_output_safety_and_commits_once(self):
        client, sdk, _ = provider_client([valid_response()])
        pipeline, retriever, resolver = make_pipeline(Generator(client))

        result = pipeline.run("When is Dr. Ayesha Khan available?")

        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertEqual(len(sdk.create.calls), 1)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(resolver.commit_count, 1)

    def test_recovered_timeout_commits_once(self):
        client, sdk, _ = provider_client(
            [APITimeoutError(request()), valid_response()]
        )
        pipeline, retriever, resolver = make_pipeline(Generator(client))

        result = pipeline.run("When is Dr. Ayesha Khan available?")

        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertEqual(len(sdk.create.calls), 2)
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(resolver.commit_count, 1)

    def test_every_provider_failure_is_generation_failed_without_commit(self):
        cases = tuple(ProviderFailureKind)
        for kind in cases:
            with self.subTest(kind=kind.value):
                error = ProviderError(
                    kind,
                    retryable=False,
                    attempts=1,
                )

                class FailingGenerator:
                    def generate(self, **kwargs):
                        raise error

                pipeline, _, resolver = make_pipeline(FailingGenerator())
                result = pipeline.run("When is Dr. Ayesha Khan available?")

                self.assertEqual(result.status, AnswerStatus.GENERATION_FAILED)
                self.assertNotEqual(result.status, AnswerStatus.NO_EVIDENCE)
                self.assertEqual(result.generation_failure.kind, kind)
                self.assertEqual(resolver.commit_count, 0)
                self.assertIsNone(resolver.get_context().active_doctor)
                self.assertNotIn("ProviderError", result.answer)

    def test_malformed_response_is_generation_failed_not_output_rejected(self):
        client, _, _ = provider_client([SimpleNamespace(choices=[])])
        pipeline, _, resolver = make_pipeline(Generator(client))
        result = pipeline.run("When is Dr. Ayesha Khan available?")

        self.assertEqual(result.status, AnswerStatus.GENERATION_FAILED)
        self.assertEqual(
            result.generation_failure.kind,
            ProviderFailureKind.MALFORMED_RESPONSE,
        )
        self.assertEqual(resolver.commit_count, 0)

    def test_timeout_exhaustion_exposes_only_safe_failure_diagnostics(self):
        client, sdk, _ = provider_client(
            [APITimeoutError(request()) for _ in range(3)]
        )
        pipeline, _, resolver = make_pipeline(Generator(client))
        result = pipeline.run("When is Dr. Ayesha Khan available?")

        self.assertEqual(result.status, AnswerStatus.GENERATION_FAILED)
        self.assertEqual(result.generation_failure.kind, ProviderFailureKind.TIMEOUT)
        self.assertEqual(result.generation_failure.attempts, 3)
        self.assertEqual(len(sdk.create.calls), 3)
        self.assertEqual(resolver.commit_count, 0)
        self.assertNotIn("APITimeoutError", repr(result))

    def test_lazy_client_configuration_failure_is_controlled(self):
        def failing_factory():
            raise ProviderError(
                ProviderFailureKind.CONFIGURATION,
                retryable=False,
                attempts=0,
            )

        pipeline, _, resolver = make_pipeline(
            Generator(client_factory=failing_factory)
        )
        result = pipeline.run("When is Dr. Ayesha Khan available?")

        self.assertEqual(result.status, AnswerStatus.GENERATION_FAILED)
        self.assertEqual(
            result.generation_failure.kind,
            ProviderFailureKind.CONFIGURATION,
        )
        self.assertEqual(resolver.commit_count, 0)
        self.assertNotIn("GROQ_API_KEY", result.answer)

    def test_output_rejection_remains_distinct(self):
        client, _, _ = provider_client(
            [valid_response("You should take ibuprofen for your headache.")]
        )
        pipeline, _, resolver = make_pipeline(Generator(client))
        result = pipeline.run("When is Dr. Ayesha Khan available?")

        self.assertEqual(result.status, AnswerStatus.OUTPUT_REJECTED)
        self.assertIsNone(result.generation_failure)
        self.assertEqual(resolver.commit_count, 0)

    def test_safety_blocked_query_never_enters_retry_subsystem(self):
        client, sdk, sleeps = provider_client([valid_response()])
        pipeline, retriever, resolver = make_pipeline(Generator(client))
        result = pipeline.run("I cannot breathe. What medicine should I take?")

        self.assertEqual(result.status, AnswerStatus.SAFETY_REFUSAL)
        self.assertEqual(sdk.create.calls, [])
        self.assertEqual(sleeps, [])
        self.assertEqual(retriever.calls, [])
        self.assertEqual(resolver.commit_count, 0)

    def test_all_retries_use_the_same_minimized_prompt(self):
        client, sdk, _ = provider_client(
            [APITimeoutError(request()), valid_response()]
        )
        pipeline, _, _ = make_pipeline(Generator(client))
        result = pipeline.run(
            "I've had a private headache narrative for two days. "
            "My phone is 0300-555-0144. When is Dr. Ayesha Khan available?"
        )

        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertEqual(len(sdk.create.calls), 2)
        first_messages = sdk.create.calls[0]["messages"]
        second_messages = sdk.create.calls[1]["messages"]
        self.assertEqual(first_messages, second_messages)
        captured = repr(sdk.create.calls)
        self.assertNotIn("0300-555-0144", captured)
        self.assertNotIn("private headache narrative", captured)
        self.assertNotIn("0300-555-0144", repr(result))


if __name__ == "__main__":
    unittest.main()
