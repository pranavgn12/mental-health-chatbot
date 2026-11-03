import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

# The backend's database module creates a MySQL engine at import time. These
# unit tests only need the SQLAlchemy Base to import the models safely.
try:
    import database  # noqa: F401
except Exception:
    from sqlalchemy.orm import declarative_base
    database_stub = types.ModuleType("database")
    database_stub.Base = declarative_base()
    sys.modules["database"] = database_stub

provider_stub = types.ModuleType("services.llm_provider")
provider_stub.generate_response = lambda *args, **kwargs: '{"action":"CONTINUE","title":null,"reason":"test","confidence":0.9}'
sys.modules.setdefault("services.llm_provider", provider_stub)

# The dump redacts part of models.py, so use a minimal ORM model stub here.
class _ColumnStub:
    def __eq__(self, other):
        return ("eq", other)
    def asc(self):
        return self

models_stub = types.ModuleType("models")
models_stub.EpisodeMood = type("EpisodeMood", (), {
    "episode_id": _ColumnStub(), "message_id": _ColumnStub(),
    "created_at": _ColumnStub(), "id": _ColumnStub(),
})
sys.modules.setdefault("models", models_stub)

from services.ai.episode_detector import EpisodeDetector
from services.ai.mood_detector import MoodDetector


class FakeQuery:
    def __init__(self, first_result=None, all_result=None):
        self.first_result = first_result
        self.all_result = all_result or []

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return self.first_result

    def all(self):
        return self.all_result


class FakeDB:
    def __init__(self, queries):
        self.queries = iter(queries)

    def query(self, *args, **kwargs):
        return next(self.queries)

    def rollback(self):
        pass


class EpisodeDetectionTests(unittest.TestCase):
    def setUp(self):
        self.detector = EpisodeDetector()
        self.episode = SimpleNamespace(
            episode_id="EP_TEST123", title="Exam Anxiety", status="ACTIVE",
            summary=None,
        )

    def test_no_longer_feared_of_exams_closes_immediately(self):
        result = self.detector.detect(
            self.episode, "I am no longer feared of exams."
        )
        self.assertEqual(result["action"], "CLOSE")
        self.assertEqual(result["detection_method"], "explicit_resolution_rule")

    def test_not_afraid_anymore_closes(self):
        result = self.detector.detect(
            self.episode, "I don't feel afraid of exams anymore."
        )
        self.assertEqual(result["action"], "CLOSE")

    def test_dont_fear_exams_anymore_closes(self):
        result = self.detector.detect(
            self.episode, "I don't fear exams anymore."
        )
        self.assertEqual(result["action"], "CLOSE")

    @patch("services.ai.episode_detector.generate_response")
    def test_mild_improvement_does_not_force_close(self, llm):
        llm.return_value = json.dumps({
            "action": "CONTINUE", "title": None,
            "reason": "Still affected", "confidence": 0.9,
        })
        result = self.detector.detect(
            self.episode, "I feel a little better about exams today."
        )
        self.assertEqual(result["action"], "CONTINUE")
        llm.assert_called_once()

    @patch("services.ai.episode_detector.generate_response")
    def test_uncertain_resolution_does_not_force_close(self, llm):
        llm.return_value = json.dumps({
            "action": "CONTINUE", "title": None,
            "reason": "User is unsure", "confidence": 0.9,
        })
        result = self.detector.detect(
            self.episode, "I'm not sure if I'm no longer afraid of exams."
        )
        self.assertEqual(result["action"], "CONTINUE")

    def test_resolution_of_unrelated_named_topic_does_not_close(self):
        result = self.detector.detect(
            self.episode, "I am no longer afraid of dogs."
        )
        # Explicit named topic does not match the active exam episode; the
        # LLM fallback is stubbed to CONTINUE.
        self.assertEqual(result["action"], "CONTINUE")


class MoodTransitionTests(unittest.TestCase):
    def setUp(self):
        self.detector = MoodDetector()

    def test_first_mood_is_recorded(self):
        self.assertTrue(self.detector._is_transition("fear", []))

    def test_unchanged_mood_is_not_a_new_transition(self):
        previous = [SimpleNamespace(mood="fear")]
        self.assertFalse(self.detector._is_transition("fear", previous))

    def test_changed_mood_is_a_new_transition(self):
        previous = [SimpleNamespace(mood="fear")]
        self.assertTrue(self.detector._is_transition("happy", previous))

    def test_missing_message_id_skips_persistence(self):
        with patch("services.ai.mood_detector.generate_response") as llm:
            self.detector.update(
                db=Mock(), episode=SimpleNamespace(episode_id="EP1"),
                conversation=[], latest_message="I feel scared", reply="I understand",
                message_id=None,
            )
        llm.assert_not_called()

    def test_retry_for_same_message_is_idempotent(self):
        db = FakeDB([FakeQuery(first_result=SimpleNamespace(id=7))])
        with patch("services.ai.mood_detector.generate_response") as llm:
            self.detector.update(
                db=db, episode=SimpleNamespace(episode_id="EP1"),
                conversation=[], latest_message="I feel scared", reply="I understand",
                message_id=7,
            )
        llm.assert_not_called()

    def test_same_mood_does_not_create_another_row(self):
        db = FakeDB([
            FakeQuery(first_result=None),
            FakeQuery(all_result=[SimpleNamespace(mood="fear", created_at="2026-01-01")]),
        ])
        response = json.dumps({
            "mood": "fear", "confidence": 0.95,
            "evidence": "User says they feel scared.",
        })
        with patch("services.ai.mood_detector.generate_response", return_value=response):
            with patch.object(self.detector, "save") as save:
                self.detector.update(
                    db=db, episode=SimpleNamespace(episode_id="EP1", title="Exam Anxiety", status="ACTIVE", summary=None),
                    conversation=[], latest_message="I still feel scared", reply="I understand",
                    message_id=8,
                )
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
