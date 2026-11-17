"""Tests for environment-backed feature settings."""

import os
import unittest
from unittest.mock import patch

from services.settings import reload_settings


class SettingsTests(unittest.TestCase):
    def tearDown(self):
        reload_settings()

    def test_defaults_keep_optional_modules_disabled(self):
        names = (
            "ENABLE_KNOWLEDGE_RAG", "ENABLE_EPISODIC_MEMORY",
            "ENABLE_LONG_TERM_MEMORY", "ENABLE_MEMORY_RAG",
            "ENABLE_MEMORY_CONSOLIDATION", "ENABLE_PATTERN_DETECTION",
            "ENABLE_REFLECTION", "ENABLE_MOOD_DETECTION", "BENCHMARK_MODE",
        )
        with patch.dict(os.environ, {}, clear=True):
            for name in names:
                os.environ.pop(name, None)
            settings = reload_settings()

        self.assertTrue(settings.enable_short_term_memory)
        self.assertFalse(settings.enable_knowledge_rag)
        self.assertFalse(settings.enable_episodic_memory)
        self.assertFalse(settings.enable_long_term_memory)
        self.assertFalse(settings.enable_memory_rag)
        self.assertFalse(settings.enable_memory_consolidation)
        self.assertFalse(settings.enable_pattern_detection)
        self.assertFalse(settings.enable_reflection)
        self.assertFalse(settings.enable_mood_detection)
        self.assertFalse(settings.benchmark_mode)

    def test_flags_parse_true_and_false(self):
        with patch.dict(os.environ, {
            "ENABLE_KNOWLEDGE_RAG": "yes",
            "ENABLE_EPISODIC_MEMORY": "1",
            "ENABLE_LONG_TERM_MEMORY": "off",
            "BENCHMARK_MODE": "true",
        }, clear=False):
            settings = reload_settings()

        self.assertTrue(settings.enable_knowledge_rag)
        self.assertTrue(settings.enable_episodic_memory)
        self.assertFalse(settings.enable_long_term_memory)
        self.assertTrue(settings.benchmark_mode)
        self.assertTrue(settings.enable_short_term_memory)

    def test_short_term_memory_cannot_be_disabled(self):
        with patch.dict(os.environ, {"ENABLE_SHORT_TERM_MEMORY": "false"}, clear=False):
            settings = reload_settings()
        self.assertTrue(settings.enable_short_term_memory)


if __name__ == "__main__":
    unittest.main()
