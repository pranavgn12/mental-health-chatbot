import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

# Keep this unit test independent of a configured LLM API key.
provider_stub = types.ModuleType("services.llm_provider")
provider_stub.generate_response = lambda *args, **kwargs: "{}"
sys.modules.setdefault("services.llm_provider", provider_stub)

# These tests exercise validation only; no ORM session is created.
models_stub = types.ModuleType("models")
class _ColumnStub:
    def __eq__(self, other):
        return ("eq", other)
    def desc(self):
        return self
models_stub.EpisodeMood = type("EpisodeMood", (), {
    "episode_id": _ColumnStub(),
    "message_id": _ColumnStub(),
    "created_at": _ColumnStub(),
    "id": _ColumnStub(),
})
for _model_name in ("Episode", "EpisodePattern", "EpisodeReflection"):
    setattr(models_stub, _model_name, type(_model_name, (), {}))
sys.modules.setdefault("models", models_stub)

from services.ai.turn_analyzer import TurnAnalyzer
from services.ai.mood_detector import MoodDetector


class TurnAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = TurnAnalyzer()
        self.active_episode = SimpleNamespace(
            episode_id="EP1", title="Exam Anxiety", status="ACTIVE", summary="Exam-related fear."
        )

    def test_combined_payload_parses(self):
        parsed = self.analyzer._parse_response(
            '{"episode":{"action":"CONTINUE"},'
            '"mood":{"label":"sad","confidence":0.9,"evidence":"I feel low"},'
            '"reflection":"The user noticed preparation helps."}'
        )
        self.assertEqual(parsed["episode"]["action"], "CONTINUE")
        self.assertEqual(parsed["mood"]["label"], "sad")
        self.assertIn("preparation", parsed["reflection"])

    def test_mood_validation_keeps_confidence_threshold(self):
        accepted = self.analyzer._validate_mood({
            "label": "fear", "confidence": 0.80, "evidence": "I am scared."
        })
        rejected = self.analyzer._validate_mood({
            "label": "fear", "confidence": 0.70, "evidence": "I am scared."
        })
        self.assertEqual(accepted["label"], "fear")
        self.assertIsNone(rejected["label"])

    def test_active_episode_cannot_be_created_again(self):
        decision = self.analyzer._validate_episode(
            {"action": "CREATE", "title": "New topic"}, self.active_episode
        )
        self.assertEqual(decision["action"], "CONTINUE")

    def test_no_active_episode_cannot_be_closed(self):
        decision = self.analyzer._validate_episode(
            {"action": "CLOSE"}, None
        )
        self.assertEqual(decision["action"], "NONE")

    def test_reflection_is_suppressed_for_non_episode_turn(self):
        self.assertIsNone(
            self.analyzer._validate_reflection("Some insight", "NONE")
        )

    def test_empty_reflection_becomes_none(self):
        self.assertIsNone(
            self.analyzer._validate_reflection("  ", "CONTINUE")
        )

    def test_full_turn_analysis_uses_one_llm_request(self):
        response = (
            '{"episode":{"action":"NONE","title":null,"reason":"casual","confidence":0.9},'
            '"mood":{"label":null,"confidence":0.0,"evidence":""},'
            '"reflection":null}'
        )
        with __import__("unittest").mock.patch(
            "services.ai.turn_analyzer.generate_response", return_value=response
        ) as llm:
            self.analyzer.analyze(None, "Hello", conversation=[])
        llm.assert_called_once()


class FakeQuery:
    def __init__(self, first_result=None, all_result=None):
        self.first_result = first_result
        self.all_result = all_result or []

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def first(self):
        return self.first_result

    def all(self):
        return self.all_result


class MoodPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.detector = MoodDetector()
        self.episode = SimpleNamespace(episode_id="EP1")

    def test_missing_message_id_does_not_query_or_save(self):
        db = Mock()
        saved = self.detector.save_decision(
            db, self.episode,
            {"label": "fear", "confidence": 0.95, "evidence": "I feel scared."},
            message_id=None,
        )
        self.assertFalse(saved)
        db.query.assert_not_called()

    def test_retry_for_same_message_is_idempotent(self):
        db = Mock()
        db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(id=5)
        saved = self.detector.save_decision(
            db, self.episode,
            {"label": "fear", "confidence": 0.95, "evidence": "I feel scared."},
            message_id=5,
        )
        self.assertFalse(saved)
        db.add.assert_not_called()

    def test_confidence_below_threshold_is_not_saved(self):
        db = Mock()
        db.query.return_value.filter.return_value.first.return_value = None
        saved = self.detector.save_decision(
            db, self.episode,
            {"label": "fear", "confidence": 0.70, "evidence": "I feel scared."},
            message_id=6,
        )
        self.assertFalse(saved)
        # Only the idempotency query is made; no row is added.
        db.add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
