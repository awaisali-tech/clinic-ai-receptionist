import unittest

from evaluation.safety_privacy_evaluation import (
    SAFETY_PRIVACY_EVALUATION_CASES,
    run_safety_privacy_evaluation,
)


class SafetyPrivacyEvaluationTests(unittest.TestCase):
    def test_curated_policy_suite(self):
        report = run_safety_privacy_evaluation()

        self.assertGreaterEqual(len(SAFETY_PRIVACY_EVALUATION_CASES), 30)
        self.assertLessEqual(len(SAFETY_PRIVACY_EVALUATION_CASES), 50)
        self.assertEqual(report.failures, ())
        self.assertEqual(report.passed, report.total)


if __name__ == "__main__":
    unittest.main()
