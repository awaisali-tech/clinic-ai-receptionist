import unittest

from evaluation.reliability_evaluation import (
    RELIABILITY_EVALUATION_CASES,
    run_reliability_evaluation,
)


class ReliabilityEvaluationTests(unittest.TestCase):
    def test_curated_reliability_scenarios(self):
        report = run_reliability_evaluation()

        self.assertGreaterEqual(len(RELIABILITY_EVALUATION_CASES), 15)
        self.assertLessEqual(len(RELIABILITY_EVALUATION_CASES), 25)
        self.assertEqual(report.failures, ())
        self.assertEqual(report.passed, report.total)


if __name__ == "__main__":
    unittest.main()
