import unittest

from services.ai.pattern_detector import _normalise_grounded_evidence


class PatternEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.allowed = {"EP_CURRENT", "EP_PRIOR"}
        self.sources = {
            "EP_CURRENT": {"Mood observation: 2026-11-04T10:00:00 | mood=sad | confidence=0.9"},
            "EP_PRIOR": {"Summary: Felt anxious before the exam"},
        }

    def test_accepts_exact_evidence_for_allowed_episode(self):
        result = _normalise_grounded_evidence(
            [
                {"episode_id": "EP_CURRENT", "detail": next(iter(self.sources["EP_CURRENT"]))},
                {"episode_id": "EP_PRIOR", "detail": next(iter(self.sources["EP_PRIOR"]))},
            ],
            self.allowed,
            self.sources,
        )
        self.assertEqual(len(result), 2)

    def test_rejects_string_evidence_from_old_prompt_shape(self):
        result = _normalise_grounded_evidence(
            ["User reported stress before exams"],
            self.allowed,
            self.sources,
        )
        self.assertEqual(result, [])

    def test_rejects_unknown_episode_and_fabricated_detail(self):
        result = _normalise_grounded_evidence(
            [
                {"episode_id": "EP_MADE_UP", "detail": "Summary: Felt anxious before the exam"},
                {"episode_id": "EP_PRIOR", "detail": "A fabricated event"},
            ],
            self.allowed,
            self.sources,
        )
        self.assertEqual(result, [])

    def test_deduplicates_evidence(self):
        detail = next(iter(self.sources["EP_PRIOR"]))
        result = _normalise_grounded_evidence(
            [
                {"episode_id": "EP_PRIOR", "detail": detail},
                {"episode_id": "EP_PRIOR", "detail": detail},
            ],
            self.allowed,
            self.sources,
        )
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
