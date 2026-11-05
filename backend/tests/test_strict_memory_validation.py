"""Focused tests for strict pattern candidate validation.

Run from backend/: python -m unittest tests.test_strict_memory_validation -v
The tests use lightweight stubs and do not call the LLM or database.
"""
import sys
import types
import unittest
from types import SimpleNamespace

# Stub provider/models before importing the service in isolated unit tests.
provider = types.ModuleType("services.llm_provider")
provider.generate_response = lambda *a, **k: "{}"
sys.modules.setdefault("services.llm_provider", provider)

models = types.ModuleType("models")
for name in ("Episode", "EpisodeMood", "EpisodePattern", "EpisodeReflection"):
    setattr(models, name, type(name, (), {}))
sys.modules.setdefault("models", models)

# Stub source helpers to keep these tests independent from database models.
detector = types.ModuleType("services.ai.pattern_detector")
detector._build_episode_source = lambda *a: ""
detector._normalise_grounded_evidence = lambda evidence, allowed, sources: [
    e for e in (evidence or [])
    if isinstance(e, dict)
    and e.get("episode_id") in allowed
    and e.get("detail") in sources.get(e.get("episode_id"), set())
]
sys.modules.setdefault("services.ai.pattern_detector", detector)

from services.ai.pattern_memory_analyzer import PatternMemoryAnalyzer


class StrictValidationTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = PatternMemoryAnalyzer()
        self.context = {
            "current_episode_id": "E3",
            "allowed_episode_ids": {"E1", "E2", "E3"},
            "source_lines_by_episode": {
                "E1": {"Summary: felt anxious before exam"},
                "E2": {"Summary: felt anxious before presentation"},
                "E3": {"Summary: felt nervous today", "Reflection: things are improving"},
            },
            "existing": [],
        }
        self.episode = SimpleNamespace(episode_id="E3", user_id="U1")

    def candidate(self, **overrides):
        item = {
            "category": "mood",
            "description": "Experiences anxiety before evaluative situations",
            "confidence": 0.8,
            "evidence": [
                {"episode_id": "E1", "detail": "Summary: felt anxious before exam"},
                {"episode_id": "E2", "detail": "Summary: felt anxious before presentation"},
            ],
            "status": "ACTIVE",
            "resolution_evidence": None,
        }
        item.update(overrides)
        return item

    def test_accepts_two_distinct_episodes(self):
        result = self.analyzer._validate_patterns([self.candidate()], self.context, self.episode)
        self.assertEqual(len(result), 1)

    def test_rejects_two_lines_from_same_episode(self):
        item = self.candidate(evidence=[
            {"episode_id": "E1", "detail": "Summary: felt anxious before exam"},
            {"episode_id": "E1", "detail": "Summary: felt anxious before exam"},
        ])
        with self.assertRaises(ValueError):
            self.analyzer._validate_patterns([item], self.context, self.episode)

    def test_rejects_fabricated_evidence(self):
        item = self.candidate(evidence=[
            {"episode_id": "E1", "detail": "Summary: anxiety caused by childhood trauma"},
            {"episode_id": "E2", "detail": "Summary: felt anxious before presentation"},
        ])
        with self.assertRaises(ValueError):
            self.analyzer._validate_patterns([item], self.context, self.episode)

    def test_rejects_string_confidence(self):
        with self.assertRaises(ValueError):
            self.analyzer._validate_patterns([self.candidate(confidence="0.9")], self.context, self.episode)

    def test_rejects_unjustified_resolution(self):
        item = self.candidate(status="RESOLVED", resolution_evidence="Summary: felt nervous today")
        with self.assertRaises(ValueError):
            self.analyzer._validate_patterns([item], self.context, self.episode)

    def test_accepts_resolution_only_with_exact_current_evidence_and_existing_pattern(self):
        existing = SimpleNamespace(description="Experiences anxiety before evaluative situations")
        self.context["existing"] = [existing]
        item = self.candidate(status="RESOLVED", resolution_evidence="Reflection: things are improving")
        result = self.analyzer._validate_patterns([item], self.context, self.episode)
        self.assertEqual(result[0]["status"], "RESOLVED")

    def test_rejects_memory_without_bullets(self):
        with self.assertRaises(ValueError):
            self.analyzer._validate_memory_shape("The user likes music.")

    def test_rejects_more_than_eight_memory_bullets(self):
        with self.assertRaises(ValueError):
            self.analyzer._validate_memory_shape("\n".join(f"- Fact {i}" for i in range(9)))


if __name__ == "__main__":
    unittest.main()
