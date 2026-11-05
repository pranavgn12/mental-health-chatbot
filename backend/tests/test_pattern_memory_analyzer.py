"""Regression tests for independent pattern and long-term-memory outcomes.

Run from backend/: python -m unittest tests.test_pattern_memory_analyzer -v
"""
import sys
import types
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Lightweight imports keep these tests independent of a real database/LLM.
provider_stub = types.ModuleType("services.llm_provider")
provider_stub.generate_response = lambda *args, **kwargs: "{}"
sys.modules.setdefault("services.llm_provider", provider_stub)

models_stub = types.ModuleType("models")
for model_name in ("Episode", "EpisodeMood", "EpisodePattern", "EpisodeReflection"):
    setattr(models_stub, model_name, type(model_name, (), {}))

class _ModelField:
    def __eq__(self, value):
        return ("user_id", value)

class _LongTermMemoryModel:
    user_id = _ModelField()

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

models_stub.LongTermMemory = _LongTermMemoryModel
sys.modules.setdefault("models", models_stub)

pattern_stub = types.ModuleType("services.ai.pattern_detector")
pattern_stub._build_episode_source = lambda *args, **kwargs: ""
pattern_stub._normalise_grounded_evidence = lambda evidence, allowed, sources: [
    item for item in (evidence or [])
    if isinstance(item, dict)
    and item.get("episode_id") in allowed
    and item.get("detail") in sources.get(item.get("episode_id"), set())
]
sys.modules.setdefault("services.ai.pattern_detector", pattern_stub)

profile_stub = types.ModuleType("services.ai.profile")
profile_stub.Profile = type("Profile", (), {"generate": lambda *args, **kwargs: []})
sys.modules.setdefault("services.ai.profile", profile_stub)

from services.ai.pattern_memory_analyzer import PatternMemoryAnalyzer
from services.ai.long_term_memory import LongTermMemory


class PatternMemoryAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = PatternMemoryAnalyzer()
        self.episode = SimpleNamespace(episode_id="EP_CURRENT", user_id=7)
        self.context = {
            "episode_sources": "Episode ID: EP_CURRENT\\nSummary: current",
            "existing_patterns_text": "No existing patterns.",
            "existing": [],
            "current_episode_id": "EP_CURRENT",
            "allowed_episode_ids": {"EP_OLD", "EP_CURRENT"},
            "source_lines_by_episode": {
                "EP_OLD": {"Summary: old evidence"},
                "EP_CURRENT": {"Summary: current evidence"},
            },
        }
        self.analyzer._build_pattern_context = Mock(return_value=self.context)
        self.analyzer._validate_patterns = Mock(return_value=[])
        self.analyzer._save_validated_patterns = Mock()
        self.db = Mock()
        self.db.begin_nested.return_value = nullcontext()

    def run_analyzer(self, response, previous_memory=None):
        with patch(
            "services.ai.pattern_memory_analyzer.generate_response",
            return_value=response,
        ):
            return self.analyzer.analyze_and_save(
                db=self.db,
                user_id=7,
                episode=self.episode,
                previous_memory=previous_memory,
            )

    def test_empty_memory_is_valid_when_no_previous_memory_exists(self):
        result = self.run_analyzer('{"patterns": [], "memory": ""}')
        self.assertEqual(result, "")
        self.analyzer._validate_patterns.assert_called_once()

    def test_empty_memory_does_not_block_pattern_persistence(self):
        self.analyzer._validate_patterns.return_value = [{"candidate": "valid"}]
        result = self.run_analyzer('{"patterns": [{"candidate": "valid"}], "memory": ""}')
        self.assertEqual(result, "")
        self.analyzer._save_validated_patterns.assert_called_once()
        self.db.flush.assert_called_once()

    def test_empty_model_memory_preserves_existing_memory(self):
        previous = "- User is preparing for examinations."
        result = self.run_analyzer('{"patterns": [], "memory": ""}', previous)
        self.assertEqual(result, previous)

    def test_invalid_memory_shape_preserves_previous_but_does_not_block_patterns(self):
        previous = "- User is preparing for examinations."
        self.analyzer._validate_patterns.return_value = [{"candidate": "valid"}]
        result = self.run_analyzer(
            '{"patterns": [{"candidate": "valid"}], "memory": "This is not a bullet."}',
            previous,
        )
        self.assertEqual(result, previous)
        self.analyzer._save_validated_patterns.assert_called_once()

    def test_pattern_validation_failure_does_not_discard_valid_memory(self):
        self.analyzer._validate_patterns.side_effect = ValueError("bad evidence")
        result = self.run_analyzer(
            '{"patterns": [{"candidate": "invalid"}], "memory": "- User values close friendships."}'
        )
        self.assertEqual(result, "- User values close friendships.")
        self.analyzer._save_validated_patterns.assert_not_called()

    def test_missing_memory_field_preserves_previous_memory_and_processes_patterns(self):
        previous = "- User is preparing for examinations."
        result = self.run_analyzer('{"patterns": []}', previous)
        self.assertEqual(result, previous)

    def test_malformed_json_still_raises_as_whole_response_is_unusable(self):
        with self.assertRaises(ValueError):
            self.run_analyzer("not json")

    def test_long_term_memory_commits_patterns_even_when_memory_is_empty(self):
        service = LongTermMemory()
        service.pattern_memory_analyzer.analyze_and_save = Mock(return_value="")
        service.profile_generator.generate = Mock()
        self.db.query.return_value.filter.return_value.with_for_update.return_value.one_or_none.return_value = None

        result = service.update(self.db, 7, self.episode)

        self.assertIsNone(result)
        self.db.commit.assert_called_once()
        self.db.add.assert_not_called()
        service.profile_generator.generate.assert_not_called()

    def test_long_term_memory_preserves_existing_value_when_analyzer_returns_empty(self):
        service = LongTermMemory()
        service.pattern_memory_analyzer.analyze_and_save = Mock(return_value="")
        service.profile_generator.generate = Mock()
        existing = SimpleNamespace(
            memory="- User values close friendships.",
            profile_themes=[{"theme": "relationships"}],
        )
        self.db.query.return_value.filter.return_value.with_for_update.return_value.one_or_none.return_value = existing

        result = service.update(self.db, 7, self.episode)

        self.assertEqual(result, existing.memory)
        self.assertEqual(existing.memory, "- User values close friendships.")
        self.assertEqual(existing.profile_themes, [{"theme": "relationships"}])
        self.db.commit.assert_called_once()
        service.profile_generator.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
