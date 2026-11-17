import sys
import types
import unittest
from unittest.mock import patch

provider_stub = types.ModuleType("services.llm_provider")
provider_stub.generate_response = lambda *args, **kwargs: "The user reported a concern."
sys.modules.setdefault("services.llm_provider", provider_stub)

from services.ai.episode_summarizer import EpisodeSummarizer


class EpisodeSummarizerTests(unittest.TestCase):
    def setUp(self):
        self.summarizer = EpisodeSummarizer()

    def test_only_user_messages_are_given_to_the_summarizer(self):
        conversation = [
            {"role": "user", "content": "I keep delaying my assignments."},
            {"role": "assistant", "content": "Try Pomodoro."},
            {"role": "user", "content": "I feel guilty afterward."},
        ]
        source = self.summarizer._user_statements(conversation)
        self.assertIn("delaying my assignments", source)
        self.assertIn("feel guilty", source)
        self.assertNotIn("Try Pomodoro", source)

    def test_rejects_transcript_shaped_output(self):
        self.assertFalse(self.summarizer._is_synthesized_summary(
            "User: I delayed my work.\nAssistant: Try planning."
        ))

    def test_rejects_overlong_output(self):
        self.assertFalse(self.summarizer._is_synthesized_summary("word " * 100))

    @patch("services.ai.episode_summarizer.generate_response")
    def test_retries_when_model_returns_transcript(self, llm):
        llm.side_effect = [
            "User: I delayed work.",
            "The user repeatedly delayed starting academic tasks and felt guilty afterward.",
        ]
        result = self.summarizer.summarize([
            {"role": "user", "content": "I delayed work and felt guilty."}
        ])
        self.assertIn("delayed starting", result)
        self.assertEqual(llm.call_count, 2)

    @patch("services.ai.episode_summarizer.generate_response")
    def test_does_not_return_transcript_after_failed_retry(self, llm):
        llm.return_value = "User: I delayed work."
        with self.assertRaises(ValueError):
            self.summarizer.summarize([
                {"role": "user", "content": "I delayed work."}
            ])
        self.assertEqual(llm.call_count, 2)


if __name__ == "__main__":
    unittest.main()
