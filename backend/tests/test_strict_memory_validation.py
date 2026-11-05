"""Strict summary-evidence and memory-shape validation tests.

Run from backend/: python -m unittest tests.test_strict_memory_validation -v
"""
import sys
import types
import unittest
from types import SimpleNamespace

provider = types.ModuleType("services.llm_provider")
provider.generate_response = lambda *a, **k: '{"patterns": []}'
sys.modules.setdefault("services.llm_provider", provider)

models = types.ModuleType("models")
for name in (
    "Episode", "EpisodePattern", "LongTermMemory", "EpisodeMood", "EpisodeReflection"
):
    setattr(models, name, type(name, (), {}))
sys.modules.setdefault("models", models)

profile = types.ModuleType("services.ai.profile")
profile.Profile = type("Profile", (), {"generate": lambda *a, **k: []})
sys.modules.setdefault("services.ai.profile", profile)

from services.ai.pattern_memory_analyzer import PatternMemoryAnalyzer
from services.ai.long_term_memory import LongTermMemory


class StrictValidationTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = PatternMemoryAnalyzer()
        self.context = {
            "current_episode_id": "E3",
            "summary_by_episode": {
                "E1": "The user felt anxious before an exam.",
                "E2": "The user felt anxious before a presentation.",
                "E3": "The user reports managing evaluative stress better now.",
            },
            "episode_dates": {},
            "existing": [],
            "existing_by_id": {},
        }
        self.episode = SimpleNamespace(episode_id="E3", user_id="U1")

    def candidate(self, **overrides):
        item = {
            "pattern_id": None,
            "category": "mood",
            "description": "Experiences anxiety before evaluative situations",
            "confidence": 0.8,
            "evidence": [
                {"episode_id": "E1", "quote": "felt anxious before an exam"},
                {"episode_id": "E2", "quote": "felt anxious before a presentation"},
            ],
            "status": "ACTIVE",
            "resolution_evidence": None,
        }
        item.update(overrides)
        return item

    def test_requires_two_distinct_episode_summaries(self):
        candidate = self.candidate(evidence=[
            {"episode_id": "E1", "quote": "felt anxious before an exam"},
            {"episode_id": "E1", "quote": "felt anxious before an exam"},
        ])
        with self.assertRaisesRegex(ValueError, "two distinct"):
            self.analyzer._validate_pattern_candidate(candidate, self.context, self.episode)

    def test_rejects_fabricated_or_paraphrased_quote(self):
        candidate = self.candidate(evidence=[
            {"episode_id": "E1", "quote": "anxiety was caused by childhood trauma"},
            {"episode_id": "E2", "quote": "felt anxious before a presentation"},
        ])
        with self.assertRaisesRegex(ValueError, "two distinct"):
            self.analyzer._validate_pattern_candidate(candidate, self.context, self.episode)

    def test_rejects_unknown_episode_id(self):
        candidate = self.candidate(evidence=[
            {"episode_id": "E_UNKNOWN", "quote": "felt anxious before an exam"},
            {"episode_id": "E2", "quote": "felt anxious before a presentation"},
        ])
        with self.assertRaisesRegex(ValueError, "two distinct"):
            self.analyzer._validate_pattern_candidate(candidate, self.context, self.episode)

    def test_rejects_string_confidence(self):
        candidate = self.candidate(confidence="0.9")
        with self.assertRaisesRegex(ValueError, "JSON number"):
            self.analyzer._validate_pattern_candidate(candidate, self.context, self.episode)

    def test_new_pattern_must_be_active(self):
        candidate = self.candidate(status="RESOLVED")
        candidate["resolution_evidence"] = {
            "episode_id": "E3",
            "quote": "managing evaluative stress better now",
        }
        with self.assertRaisesRegex(ValueError, "new pattern must start as ACTIVE"):
            self.analyzer._validate_pattern_candidate(candidate, self.context, self.episode)

    def test_resolution_requires_existing_pattern_and_current_summary_excerpt(self):
        existing = SimpleNamespace(
            pattern_id="PAT_1", category="mood", description="Evaluative anxiety",
            confidence=0.8, status="ACTIVE",
        )
        self.context["existing"] = [existing]
        self.context["existing_by_id"] = {"PAT_1": existing}
        candidate = self.candidate(
            pattern_id="PAT_1",
            status="RESOLVED",
            resolution_evidence={
                "episode_id": "E3",
                "quote": "The user says all stress is permanently gone",
            },
        )
        with self.assertRaisesRegex(ValueError, "Resolution evidence"):
            self.analyzer._validate_pattern_candidate(candidate, self.context, self.episode)

    def test_accepts_resolution_with_current_summary_excerpt(self):
        existing = SimpleNamespace(
            pattern_id="PAT_1", category="mood", description="Evaluative anxiety",
            confidence=0.8, status="ACTIVE",
        )
        self.context["existing"] = [existing]
        self.context["existing_by_id"] = {"PAT_1": existing}
        candidate = self.candidate(
            pattern_id="PAT_1",
            status="RESOLVED",
            resolution_evidence={
                "episode_id": "E3",
                "quote": "managing evaluative stress better now",
            },
        )
        result = self.analyzer._validate_pattern_candidate(candidate, self.context, self.episode)
        self.assertEqual(result["status"], "RESOLVED")

    def test_rejects_memory_without_bullets(self):
        with self.assertRaises(ValueError):
            LongTermMemory._validate_memory_shape("The user likes music.")

    def test_rejects_more_than_eight_memory_bullets(self):
        with self.assertRaises(ValueError):
            LongTermMemory._validate_memory_shape("\n".join(f"- Fact {i}" for i in range(9)))


if __name__ == "__main__":
    unittest.main()
