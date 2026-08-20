from __future__ import annotations

from dataclasses import dataclass, field

from conversation.intent_classifier import InformationNeed
from orchestration.pipeline import AnswerStatus, RAGPipeline


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    name: str
    query: str
    expected_need: InformationNeed
    expected_status: AnswerStatus
    expected_document_types: frozenset[str] = frozenset()
    expected_clinic: str | None = None
    expected_doctor: str | None = None
    setup_queries: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalEvaluationFailure:
    case: str
    reasons: tuple[str, ...]


@dataclass
class RetrievalEvaluationReport:
    total: int
    passed: int = 0
    failures: list[RetrievalEvaluationFailure] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


RETRIEVAL_EVALUATION_CASES = (
    RetrievalEvaluationCase(
        "wellness location",
        "Where is Wellness Eye Clinic?",
        InformationNeed.CLINIC_LOCATION,
        AnswerStatus.ANSWERED,
        frozenset({"clinic"}),
        expected_clinic="Wellness Eye Clinic",
    ),
    RetrievalEvaluationCase(
        "dental location",
        "What is the address of Green Leaf Dental & Oral Care?",
        InformationNeed.CLINIC_LOCATION,
        AnswerStatus.ANSWERED,
        frozenset({"clinic"}),
        expected_clinic="Green Leaf Dental & Oral Care",
    ),
    RetrievalEvaluationCase(
        "sunrise phone",
        "What is the phone number for Sunrise Medical Center?",
        InformationNeed.CLINIC_CONTACT,
        AnswerStatus.ANSWERED,
        frozenset({"clinic"}),
        expected_clinic="Sunrise Medical Center",
    ),
    RetrievalEvaluationCase(
        "wellness email",
        "How can I email Wellness Eye Clinic?",
        InformationNeed.CLINIC_CONTACT,
        AnswerStatus.ANSWERED,
        frozenset({"clinic"}),
        expected_clinic="Wellness Eye Clinic",
    ),
    RetrievalEvaluationCase(
        "sunrise hours",
        "What are Sunrise Medical Center opening hours?",
        InformationNeed.CLINIC_TIMINGS,
        AnswerStatus.ANSWERED,
        frozenset({"timings"}),
        expected_clinic="Sunrise Medical Center",
    ),
    RetrievalEvaluationCase(
        "dental sunday hours",
        "Is Green Leaf Dental & Oral Care open on Sunday?",
        InformationNeed.CLINIC_TIMINGS,
        AnswerStatus.ANSWERED,
        frozenset({"timings"}),
        expected_clinic="Green Leaf Dental & Oral Care",
    ),
    RetrievalEvaluationCase(
        "ayesha availability",
        "When is Dr. Ayesha Khan available?",
        InformationNeed.DOCTOR_AVAILABILITY,
        AnswerStatus.ANSWERED,
        frozenset({"doctor"}),
        expected_doctor="Dr. Ayesha Khan",
    ),
    RetrievalEvaluationCase(
        "nadia availability",
        "What is Dr. Nadia Rehman's availability?",
        InformationNeed.DOCTOR_AVAILABILITY,
        AnswerStatus.ANSWERED,
        frozenset({"doctor"}),
        expected_doctor="Dr. Nadia Rehman",
    ),
    RetrievalEvaluationCase(
        "fahad information",
        "Tell me about Dr. Fahad Iqbal's experience.",
        InformationNeed.DOCTOR_INFORMATION,
        AnswerStatus.ANSWERED,
        frozenset({"doctor"}),
        expected_doctor="Dr. Fahad Iqbal",
    ),
    RetrievalEvaluationCase(
        "dental doctors",
        "Which doctors work at Green Leaf Dental & Oral Care?",
        InformationNeed.DOCTOR_INFORMATION,
        AnswerStatus.ANSWERED,
        frozenset({"doctor"}),
        expected_clinic="Green Leaf Dental & Oral Care",
    ),
    RetrievalEvaluationCase(
        "wellness services",
        "What services does Wellness Eye Clinic offer?",
        InformationNeed.SERVICES,
        AnswerStatus.ANSWERED,
        frozenset({"service"}),
        expected_clinic="Wellness Eye Clinic",
    ),
    RetrievalEvaluationCase(
        "sunrise services",
        "List the services at Sunrise Medical Center.",
        InformationNeed.SERVICES,
        AnswerStatus.ANSWERED,
        frozenset({"service"}),
        expected_clinic="Sunrise Medical Center",
    ),
    RetrievalEvaluationCase(
        "walk in faq",
        "Do Sunrise Medical Center accept walk-in patients?",
        InformationNeed.FAQ,
        AnswerStatus.ANSWERED,
        frozenset({"faq"}),
        expected_clinic="Sunrise Medical Center",
    ),
    RetrievalEvaluationCase(
        "payment faq",
        "What payment methods do Sunrise Medical Center accept?",
        InformationNeed.FAQ,
        AnswerStatus.ANSWERED,
        frozenset({"faq"}),
        expected_clinic="Sunrise Medical Center",
    ),
    RetrievalEvaluationCase(
        "insurance faq",
        "Are eye exams covered by insurance at Wellness Eye Clinic?",
        InformationNeed.FAQ,
        AnswerStatus.ANSWERED,
        frozenset({"faq"}),
        expected_clinic="Wellness Eye Clinic",
    ),
    RetrievalEvaluationCase(
        "online booking faq",
        "Can I book an appointment online at Green Leaf Dental & Oral Care?",
        InformationNeed.FAQ,
        AnswerStatus.ANSWERED,
        frozenset({"faq"}),
        expected_clinic="Green Leaf Dental & Oral Care",
    ),
    RetrievalEvaluationCase(
        "clinic information",
        "Tell me about Wellness Eye Clinic.",
        InformationNeed.CLINIC_INFORMATION,
        AnswerStatus.ANSWERED,
        frozenset({"clinic"}),
        expected_clinic="Wellness Eye Clinic",
    ),
    RetrievalEvaluationCase(
        "contact lens service",
        "Do you offer contact lens fitting at Wellness Eye Clinic?",
        InformationNeed.SERVICES,
        AnswerStatus.ANSWERED,
        frozenset({"service"}),
        expected_clinic="Wellness Eye Clinic",
    ),
    RetrievalEvaluationCase(
        "unknown parking",
        "Does Wellness Eye Clinic have parking?",
        InformationNeed.FAQ,
        AnswerStatus.NO_EVIDENCE,
    ),
    RetrievalEvaluationCase(
        "unknown cafeteria",
        "Is there a cafeteria at Sunrise Medical Center?",
        InformationNeed.FAQ,
        AnswerStatus.NO_EVIDENCE,
    ),
    RetrievalEvaluationCase(
        "doctor day follow up",
        "What about Saturday?",
        InformationNeed.DOCTOR_AVAILABILITY,
        AnswerStatus.ANSWERED,
        frozenset({"doctor"}),
        expected_doctor="Dr. Ayesha Khan",
        setup_queries=("Tell me about Dr. Ayesha Khan.",),
    ),
    RetrievalEvaluationCase(
        "doctor pronoun follow up",
        "When is she available?",
        InformationNeed.DOCTOR_AVAILABILITY,
        AnswerStatus.ANSWERED,
        frozenset({"doctor"}),
        expected_doctor="Dr. Nadia Rehman",
        setup_queries=("Tell me about Dr. Nadia Rehman.",),
    ),
    RetrievalEvaluationCase(
        "explicit topic switch",
        "Where is Wellness Eye Clinic?",
        InformationNeed.CLINIC_LOCATION,
        AnswerStatus.ANSWERED,
        frozenset({"clinic"}),
        expected_clinic="Wellness Eye Clinic",
        setup_queries=("Tell me about Dr. Ayesha Khan.",),
    ),
)


def run_retrieval_evaluation(
    pipeline: RAGPipeline,
    cases: tuple[RetrievalEvaluationCase, ...] = RETRIEVAL_EVALUATION_CASES,
    *,
    top_k: int = 5,
) -> RetrievalEvaluationReport:
    report = RetrievalEvaluationReport(total=len(cases))

    for case in cases:
        pipeline.reset_conversation()
        for setup_query in case.setup_queries:
            pipeline.run(setup_query, top_k=top_k)

        result = pipeline.run(case.query, top_k=top_k)
        reasons: list[str] = []
        if result.information_need != case.expected_need:
            reasons.append(
                f"intent={result.information_need.value}, "
                f"expected={case.expected_need.value}"
            )
        if result.status != case.expected_status:
            reasons.append(
                f"status={result.status.value}, "
                f"expected={case.expected_status.value}"
            )

        if case.expected_status == AnswerStatus.ANSWERED:
            if not result.evidence:
                reasons.append("no accepted evidence")
            else:
                first_metadata = result.evidence[0].document.metadata
                if (
                    case.expected_document_types
                    and first_metadata.get("document_type")
                    not in case.expected_document_types
                ):
                    reasons.append(
                        f"document_type={first_metadata.get('document_type')}"
                    )
                if (
                    case.expected_clinic
                    and first_metadata.get("clinic_name") != case.expected_clinic
                ):
                    reasons.append(
                        f"clinic={first_metadata.get('clinic_name')}"
                    )
                if (
                    case.expected_doctor
                    and first_metadata.get("doctor_name") != case.expected_doctor
                ):
                    reasons.append(
                        f"doctor={first_metadata.get('doctor_name')}"
                    )
        elif result.evidence:
            reasons.append("unexpected accepted evidence")

        if reasons:
            report.failures.append(
                RetrievalEvaluationFailure(case.name, tuple(reasons))
            )
        else:
            report.passed += 1

    return report
