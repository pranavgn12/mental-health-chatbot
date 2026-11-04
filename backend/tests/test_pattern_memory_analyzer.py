import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

provider_stub = types.ModuleType("services.llm_provider")
provider_stub.generate_response = lambda *args, **kwargs: "{}"
sys.modules.setdefault("services.llm_provider", provider_stub)

models_stub = types.ModuleType("models")
for model_name in ("Episode", "EpisodeMood", "EpisodePattern", "EpisodeReflection"):
    setattr(models_stub, model_name, type(model_name, (), {}))
sys.modules.setdefault("models", models_stub)

from services.ai.pattern_memory_analyzer import PatternMemoryAnalyzer
from services.ai.pattern_detector import _normalise_grounded_evidence


class PatternMemoryAnalyzerTests(unittest.TestCase):
    def test_combined_response_requires_memory_and_pattern_array(self):
        valid = PatternMemoryAnalyzer._parse_response(
            '{"patterns":[],"memory":"Updated memory."}'
        )
        invalid = PatternMemoryAnalyzer._parse_response(
            '{"patterns":[],"memory":null}'
        )
        self.assertEqual(valid["memory"], "Updated memory.")
        self.assertIsNone(invalid.get("memory"))

    def test_combined_pattern_and_memory_stage_uses_one_llm_request(self):
        analyzer = PatternMemoryAnalyzer()
        analyzer._build_pattern_context = Mock(return_value={
            "episode_sources": "Episode ID: EP1\nSummary: test",
            "existing_patterns_text": "No existing patterns.",
            "existing": [],
            "allowed_episode_ids": {"EP1"},
            "source_lines_by_episode": {"EP1": {"Summary: test"}},
        })
        analyzer._save_patterns = Mock()
        response = '{"patterns": [], "memory": "Updated memory."}'
        with patch("services.ai.pattern_memory_analyzer.generate_response", return_value=response) as llm:
            result = analyzer.analyze_and_save(
                db=Mock(), user_id=1, episode=SimpleNamespace(summary="Episode summary"),
                previous_memory="Old memory",
            )
        self.assertEqual(result, "Updated memory.")
        llm.assert_called_once()
        analyzer._save_patterns.assert_called_once()

    def test_accepts_only_exact_evidence_from_known_episode(self):
        sources = {
            "EP1": {"Summary: User feels anxious before exams"},
            "EP2": {"Reflection: Preparation helps the user feel calmer"},
        }
        accepted = _normalise_grounded_evidence(
            [
                {"episode_id": "EP1", "detail": "Summary: User feels anxious before exams"},
                {"episode_id": "EP2", "detail": "Reflection: Preparation helps the user feel calmer"},
                {"episode_id": "EP_X", "detail": "invented"},
            ],
            {"EP1", "EP2"},
            sources,
        )
        self.assertEqual(len(accepted), 2)


if __name__ == "__main__":
    unittest.main()
