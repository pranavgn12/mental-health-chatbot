import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Unit tests use stubs so no Groq key or MySQL instance is required.
try:
    import database  # noqa: F401
except Exception:
    from sqlalchemy.orm import declarative_base
    database_stub = types.ModuleType("database")
    database_stub.Base = declarative_base()
    sys.modules["database"] = database_stub

provider_stub = types.ModuleType("services.llm_provider")
provider_stub.generate_response = lambda *args, **kwargs: json.dumps({
    "action": "CONTINUE", "title": None, "reason": "test", "confidence": 0.9
})
sys.modules.setdefault("services.llm_provider", provider_stub)

class _ColumnStub:
    def __eq__(self, other):
        return ("eq", other)
    def desc(self):
        return self
    def asc(self):
        return self

models_stub = types.ModuleType("models")
class _EpisodeMood:
    episode_id = _ColumnStub()
    message_id = _ColumnStub()
    created_at = _ColumnStub()
    id = _ColumnStub()

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
models_stub.EpisodeMood = _EpisodeMood
for _model_name in ("Episode", "EpisodePattern", "EpisodeReflection", "LongTermMemory"):
    setattr(models_stub, _model_name, type(_model_name, (), {}))
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
    def limit(self, *args, **kwargs):
        return self
    def first(self):
        return self.first_result
    def all(self):
        return self.all_result


class FakeDB:
    def __init__(self, queries):
        self.queries = iter(queries)
        self.added = []
    def query(self, *args, **kwargs):
        return next(self.queries)
    def add(self, row):
        self.added.append(row)
    def commit(self):
        pass
    def refresh(self, row):
        pass
    def rollback(self):
        pass


class EpisodeDetectionTests(unittest.TestCase):
    def setUp(self):
        self.detector = EpisodeDetector()
        self.episode = SimpleNamespace(
            episode_id="EP_TEST123", title="Exam Anxiety", status="ACTIVE", summary=None,
        )

    def test_explicit_resolution_closes_immediately(self):
        result = self.detector.detect(self.episode, "I don't fear exams anymore.")
        self.assertEqual(result["action"], "CLOSE")
        self.assertEqual(result["detection_method"], "explicit_resolution_rule")

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

    @patch("services.ai.episode_detector.generate_response")
    def test_unrelated_resolution_does_not_close_active_episode(self, llm):
        llm.return_value = json.dumps({
            "action": "CONTINUE", "title": None,
            "reason": "Different topic", "confidence": 0.9,
        })
        result = self.detector.detect(
            self.episode, "I am no longer afraid of dogs."
        )
        self.assertEqual(result["action"], "CONTINUE")


class MoodPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.detector = MoodDetector()
        self.episode = SimpleNamespace(episode_id="EP1")
        self.decision = {
            "label": "fear", "confidence": 0.95, "evidence": "I feel scared."
        }

    def test_missing_message_id_skips_database(self):
        db = Mock()
        result = self.detector.save_decision(db, self.episode, self.decision, None)
        self.assertFalse(result)
        db.query.assert_not_called()

    def test_duplicate_message_id_is_idempotent(self):
        db = FakeDB([FakeQuery(first_result=SimpleNamespace(id=7))])
        result = self.detector.save_decision(db, self.episode, self.decision, 7)
        self.assertFalse(result)
        self.assertEqual(db.added, [])

    def test_low_confidence_is_rejected(self):
        db = FakeDB([FakeQuery(first_result=None)])
        decision = {**self.decision, "confidence": 0.70}
        result = self.detector.save_decision(db, self.episode, decision, 8)
        self.assertFalse(result)
        self.assertEqual(db.added, [])

    def test_repeated_consecutive_mood_is_not_saved(self):
        db = FakeDB([
            FakeQuery(first_result=None),
            FakeQuery(first_result=SimpleNamespace(mood="fear")),
        ])
        result = self.detector.save_decision(db, self.episode, self.decision, 9)
        self.assertFalse(result)
        self.assertEqual(db.added, [])

    def test_new_mood_transition_is_saved_without_llm(self):
        db = FakeDB([
            FakeQuery(first_result=None),
            FakeQuery(first_result=SimpleNamespace(mood="sad")),
        ])
        result = self.detector.save_decision(db, self.episode, self.decision, 10)
        self.assertTrue(result)
        self.assertEqual(len(db.added), 1)
        self.assertEqual(db.added[0].mood, "fear")


if __name__ == "__main__":
    unittest.main()
