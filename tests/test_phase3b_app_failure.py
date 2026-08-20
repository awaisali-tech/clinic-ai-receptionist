from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from generation.provider_errors import ProviderError, ProviderFailureKind


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class FailingFactory:
    def __call__(self):
        raise ProviderError(
            ProviderFailureKind.CONFIGURATION,
            retryable=False,
            attempts=0,
        )


class HealthyFactory:
    def __call__(self):
        return HealthyPipeline()


class ExplodingPipeline:
    def run(self, query, top_k=5):
        raise RuntimeError("traceback sentinel and internal request details")

    def reset_conversation(self):
        pass


class HealthyPipeline:
    def run(self, query, top_k=5):
        return SimpleNamespace(
            answer="Recovered administrative answer.",
            evidence=[],
        )

    def reset_conversation(self):
        pass


def messages(app_test):
    return app_test.session_state.filtered_state["messages"]


class StreamlitFailureIsolationTests(unittest.TestCase):
    def test_pipeline_construction_failure_keeps_history_and_can_recover(self):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        app.session_state["_pipeline_factory"] = FailingFactory()

        app.chat_input[0].set_value("Where is Sunrise Medical Center?").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(messages(app)), 2)
        self.assertEqual(messages(app)[0]["role"], "user")
        self.assertEqual(messages(app)[1]["role"], "assistant")
        self.assertNotIn("CONFIGURATION", messages(app)[1]["content"])
        self.assertNotIn("ProviderError", messages(app)[1]["content"])
        self.assertIsNone(app.session_state.filtered_state["pipeline"])

        app.session_state["_pipeline_factory"] = HealthyFactory()
        app.chat_input[0].set_value("When does it open?").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(messages(app)), 4)
        self.assertEqual(
            messages(app)[-1]["content"],
            "Recovered administrative answer.",
        )

    def test_runtime_failure_has_safe_assistant_turn_and_next_turn_works(self):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        app.session_state["pipeline"] = ExplodingPipeline()

        app.chat_input[0].set_value("Where is Wellness Eye Clinic?").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(messages(app)), 2)
        self.assertEqual(messages(app)[-1]["role"], "assistant")
        self.assertNotIn("traceback sentinel", messages(app)[-1]["content"])
        self.assertNotIn("internal request", messages(app)[-1]["content"])
        self.assertIsNone(messages(app)[-1]["doctor_data"])
        self.assertIsInstance(
            app.session_state.filtered_state["pipeline"],
            ExplodingPipeline,
        )

        app.session_state["pipeline"] = HealthyPipeline()
        app.chat_input[0].set_value("What are the clinic hours?").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(messages(app)), 4)
        self.assertEqual(
            messages(app)[-1]["content"],
            "Recovered administrative answer.",
        )


if __name__ == "__main__":
    unittest.main()
