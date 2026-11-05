"""Unit tests for summary-only pattern analysis and the 10-pattern cap.

Run from backend/: python -m unittest tests.test_pattern_memory_analyzer -v
"""
import sys
import types
import unittest
from datetime import datetime, timedelta
from contextlib import nullcontext
from unittest.mock import patch
from types import SimpleNamespace
from unittest.mock import Mock
from string import Template

provider_stub = types.ModuleType("services.llm_provider")
provider_stub.generate_response = lambda *args, **kwargs: '{"patterns": []}'
sys.modules.setdefault("services.llm_provider", provider_stub)

class _Column:
    def __eq__(self, other):
        return self

    def __ne__(self, other):
        return self

    def desc(self):
        return self

    def asc(self):
        return self


class _Episode:
    user_id = _Column()
    status = _Column()
    episode_id = _Column()
    closed_at = _Column()


class _EpisodePattern:
    user_id = _Column()

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _EpisodeMood:
    episode_id = _Column()
    created_at = _Column()


class _EpisodeReflection:
    episode_id = _Column()
    created_at = _Column()


models_stub = types.ModuleType("models")
models_stub.Episode = _Episode
models_stub.EpisodePattern = _EpisodePattern
models_stub.EpisodeMood = _EpisodeMood
models_stub.EpisodeReflection = _EpisodeReflection
models_stub.LongTermMemory = type("LongTermMemory", (), {})
sys.modules.setdefault("models", models_stub)

profile_stub = types.ModuleType("services.ai.profile")
profile_stub.Profile = type("Profile", (), {"generate": lambda *a, **k: []})
sys.modules.setdefault("services.ai.profile", profile_stub)

from services.ai.pattern_memory_analyzer import (
    MAX_STORED_PATTERNS,
    PATTERN_PROMPT,
    PatternMemoryAnalyzer,
)
from services.ai.long_term_memory import MEMORY_PROMPT, LongTermMemory


class PatternMemoryRedesignTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = PatternMemoryAnalyzer()
        self.current_id = "EP_CURRENT"
        self.context = {
            "current_episode_id": self.current_id,
            "summary_by_episode": {
                self.current_id: "The user says they are managing exam pressure better now.",
                "EP_OLD_1": "The user felt anxious because examinations were approaching.",
                "EP_OLD_2": "The user described feeling overwhelmed by exam preparation.",
            },
            "episode_dates": {},
            "existing": [],
            "existing_by_id": {},
        }
        self.episode = SimpleNamespace(episode_id=self.current_id, user_id=7)

    def _candidate(self, **overrides):
        candidate = {
            "pattern_id": None,
            "category": "trigger",
            "description": "Academic pressure is associated with anxiety.",
            "confidence": 0.82,
            "evidence": [
                {
                    "episode_id": "EP_OLD_1",
                    "quote": "felt anxious because examinations were approaching",
                    "reason": "Exam-related anxiety",
                },
                {
                    "episode_id": "EP_OLD_2",
                    "quote": "feeling overwhelmed by exam preparation",
                    "reason": "Exam preparation stress",
                },
            ],
            "status": "ACTIVE",
            "resolution_evidence": None,
        }
        candidate.update(overrides)
        return candidate

    def test_evidence_accepts_exact_summary_excerpt_case_and_whitespace_variants(self):
        result = self.analyzer._validate_summary_quote(
            "EP_OLD_1",
            "FELT ANXIOUS   because examinations were approaching",
            self.context,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["episode_id"], "EP_OLD_1")

    def test_evidence_rejects_paraphrase_that_is_not_a_summary_excerpt(self):
        result = self.analyzer._validate_summary_quote(
            "EP_OLD_1", "experienced a lot of stress about upcoming exams", self.context
        )
        self.assertIsNone(result)

    def test_new_pattern_requires_current_episode_and_another_episode(self):
        candidate = self._candidate()
        candidate["evidence"][0]["episode_id"] = self.current_id
        candidate["evidence"][0]["quote"] = "managing exam pressure better now"
        result = self.analyzer._validate_pattern_candidate(
            candidate, self.context, self.episode
        )
        self.assertEqual(result["status"], "ACTIVE")
        self.assertIsNone(result["match"])

    def test_active_pattern_update_must_include_current_episode(self):
        candidate = self._candidate()
        with self.assertRaisesRegex(ValueError, "current episode"):
            self.analyzer._validate_pattern_candidate(candidate, self.context, self.episode)

    def test_existing_pattern_can_be_reactivated_with_current_evidence(self):
        existing = SimpleNamespace(
            pattern_id="PAT_1", category="trigger", description="Academic stress",
            confidence=0.7, status="RESOLVED",
        )
        self.context["existing"] = [existing]
        self.context["existing_by_id"] = {"PAT_1": existing}
        candidate = self._candidate(
            pattern_id="PAT_1",
            description="A different description must not overwrite the stored identity.",
            evidence=[
                {"episode_id": self.current_id, "quote": "managing exam pressure better now"},
                {"episode_id": "EP_OLD_1", "quote": "felt anxious because examinations were approaching"},
            ],
        )
        result = self.analyzer._validate_pattern_candidate(
            candidate, self.context, self.episode
        )
        self.assertEqual(result["description"], "Academic stress")
        self.assertEqual(result["status"], "ACTIVE")

    def test_resolution_requires_exact_current_summary_evidence(self):
        existing = SimpleNamespace(
            pattern_id="PAT_1", category="trigger", description="Academic stress",
            confidence=0.7, status="ACTIVE",
        )
        self.context["existing"] = [existing]
        self.context["existing_by_id"] = {"PAT_1": existing}
        candidate = self._candidate(
            pattern_id="PAT_1",
            status="RESOLVED",
            resolution_evidence={
                "episode_id": self.current_id,
                "quote": "The academic stress is completely gone forever.",
            },
        )
        with self.assertRaisesRegex(ValueError, "Resolution evidence"):
            self.analyzer._validate_pattern_candidate(candidate, self.context, self.episode)

    def test_resolution_with_supported_current_excerpt_is_accepted(self):
        existing = SimpleNamespace(
            pattern_id="PAT_1", category="trigger", description="Academic stress",
            confidence=0.7, status="ACTIVE",
        )
        self.context["existing"] = [existing]
        self.context["existing_by_id"] = {"PAT_1": existing}
        candidate = self._candidate(
            pattern_id="PAT_1",
            status="RESOLVED",
            resolution_evidence={
                "episode_id": self.current_id,
                "quote": "managing exam pressure better now",
            },
        )
        result = self.analyzer._validate_pattern_candidate(
            candidate, self.context, self.episode
        )
        self.assertEqual(result["status"], "RESOLVED")

    def test_invalid_candidate_does_not_discard_valid_candidate(self):
        valid = self._candidate()
        valid["evidence"][0]["episode_id"] = self.current_id
        valid["evidence"][0]["quote"] = "managing exam pressure better now"
        invalid = self._candidate(description="Unsupported pattern")
        results = self.analyzer._validate_patterns(
            [invalid, valid], self.context, self.episode
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["description"], valid["description"])

    def test_ranker_returns_at_most_ten_and_prioritizes_recent_observation(self):
        now = datetime.now().astimezone().replace(tzinfo=None)
        rows = [
            SimpleNamespace(
                pattern_id=f"PAT_{i}",
                last_seen_at=now - timedelta(days=i),
                first_seen_at=None,
                updated_at=None,
                created_at=None,
                status="ACTIVE" if i % 2 else "RESOLVED",
                confidence=0.8,
            )
            for i in range(15)
        ]
        ranked = self.analyzer._rank_patterns(rows)[:MAX_STORED_PATTERNS]
        self.assertEqual(len(ranked), 10)
        self.assertEqual(ranked[0].pattern_id, "PAT_0")
        self.assertEqual(ranked[-1].pattern_id, "PAT_9")

    def test_database_is_trimmed_to_ten_and_new_pattern_can_replace_oldest(self):
        now = datetime.now().astimezone().replace(tzinfo=None)
        existing = [
            _EpisodePattern(
                pattern_id=f"OLD_{i}", user_id=7, category="general",
                description=f"Old pattern {i}", confidence=0.7, status="ACTIVE",
                evidence=[], last_seen_at=now - timedelta(days=20 - i),
                first_seen_at=now - timedelta(days=30 - i), created_at=now - timedelta(days=30 - i),
                updated_at=now - timedelta(days=20 - i),
            )
            for i in range(10)
        ]
        previous_episode = SimpleNamespace(
            episode_id="EP_OLD", user_id=7, status="CLOSED",
            summary="The user felt anxious about the upcoming exam.",
            closed_at=now - timedelta(days=1), created_at=now - timedelta(days=1),
        )
        current_episode = SimpleNamespace(
            episode_id=self.current_id, user_id=7, summary=(
                "The user described recurring anxiety due to academic examinations."
            ), closed_at=now, created_at=now,
        )

        class FakeQuery:
            def __init__(self, db, model):
                self.db, self.model, self.limit_value = db, model, None

            def filter(self, *args):
                return self

            def order_by(self, *args):
                return self

            def limit(self, value):
                self.limit_value = value
                return self

            def all(self):
                if self.model is _EpisodePattern:
                    return list(self.db.patterns)
                rows = list(self.db.prior_episodes)
                return rows[:self.limit_value] if self.limit_value else rows

        class FakeDB:
            def __init__(self):
                self.patterns = list(existing)
                self.prior_episodes = [previous_episode]

            def query(self, model):
                return FakeQuery(self, model)

            def delete(self, row):
                self.patterns.remove(row)

            def add(self, row):
                self.patterns.append(row)

            def flush(self):
                pass

            def begin_nested(self):
                return nullcontext()

        response = {
            "patterns": [{
                "pattern_id": None,
                "category": "trigger",
                "description": "Exam-related anxiety recurs across episodes.",
                "confidence": 0.9,
                "evidence": [
                    {"episode_id": self.current_id,
                     "quote": "recurring anxiety due to academic examinations"},
                    {"episode_id": "EP_OLD",
                     "quote": "felt anxious about the upcoming exam"},
                ],
                "status": "ACTIVE",
                "resolution_evidence": None,
            }]
        }
        db = FakeDB()
        with patch(
            "services.ai.pattern_memory_analyzer.generate_response",
            return_value=__import__("json").dumps(response),
        ) as llm:
            final_patterns = self.analyzer.analyze_and_save(
                db=db, user_id=7, episode=current_episode
            )

        self.assertEqual(len(db.patterns), 10)
        self.assertEqual(len(final_patterns), 10)
        self.assertIn("Exam-related anxiety recurs across episodes.",
                      [row["description"] for row in final_patterns])
        self.assertNotIn("OLD_0", [row.pattern_id for row in db.patterns])
        llm.assert_called_once()
        prompt = llm.call_args.args[0]
        self.assertIn("CURRENT EPISODE", prompt)
        self.assertIn("PREVIOUS CLOSED EPISODE", prompt)
        self.assertNotIn("Mood observation:", prompt)
        self.assertNotIn("Reflection:", prompt)

    def test_pattern_prompt_uses_episode_summaries_only(self):
        prompt = Template(PATTERN_PROMPT).substitute(
            episode_summaries="CURRENT EPISODE\\nEpisode ID: EP1\\nSummary: test",
            existing_patterns="[]",
        )
        self.assertIn("ONLY evidence source", prompt)
        self.assertIn("Do not use mood records, reflections", prompt)
        self.assertNotIn("Mood observation:", prompt)
        self.assertNotIn("Reflection:", prompt)

    def test_memory_generation_call_receives_moods_reflections_and_final_patterns(self):
        mood = SimpleNamespace(
            mood="anxious", confidence=0.91, evidence="Concern about examinations",
            created_at=datetime(2026, 11, 5, 10, 0, 0),
        )
        reflection = SimpleNamespace(
            reflection="Preparation helped the user feel more in control.",
            created_at=datetime(2026, 11, 5, 10, 5, 0),
        )

        class FakeQuery:
            def __init__(self, model):
                self.model = model

            def filter(self, *args):
                return self

            def order_by(self, *args):
                return self

            def all(self):
                if self.model is _EpisodeMood:
                    return [mood]
                if self.model is _EpisodeReflection:
                    return [reflection]
                return []

        class FakeDB:
            def query(self, model):
                return FakeQuery(model)

        episode = SimpleNamespace(
            episode_id="EP_CURRENT",
            summary="The user experienced exam pressure but prepared for it.",
        )
        patterns = [{
            "pattern_id": "PAT_1",
            "description": "Academic pressure is associated with anxiety.",
            "status": "ACTIVE",
            "confidence": 0.85,
            "category": "trigger",
        }]
        memory_service = LongTermMemory()
        response = '{"memory":"- Academic pressure has been a recurring source of anxiety."}'
        with patch("services.ai.long_term_memory.generate_response", return_value=response) as llm:
            result = memory_service._generate_memory(
                FakeDB(), episode, "- Previous durable fact", patterns
            )
        self.assertEqual(
            result,
            "- Academic pressure has been a recurring source of anxiety.",
        )
        prompt = llm.call_args.args[0]
        self.assertIn('"mood": "anxious"', prompt)
        self.assertIn("Preparation helped the user feel more in control.", prompt)
        self.assertIn("Academic pressure is associated with anxiety.", prompt)
        self.assertIn("The user experienced exam pressure", prompt)

    def test_memory_prompt_has_patterns_moods_and_reflections_inputs(self):
        prompt = Template(MEMORY_PROMPT).substitute(
            episode_summary="summary",
            mood_observations='[{"mood":"anxious"}]',
            reflections='[{"reflection":"preparedness helps"}]',
            validated_patterns='[{"description":"academic stress","status":"ACTIVE"}]',
            previous_memory="old memory",
        )
        self.assertIn('"mood":"anxious"', prompt)
        self.assertIn('"reflection":"preparedness helps"', prompt)
        self.assertIn('"status":"ACTIVE"', prompt)
        self.assertIn("CURRENT EPISODE SUMMARY", prompt)


if __name__ == "__main__":
    unittest.main()
