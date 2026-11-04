"""One-pass analysis of episode lifecycle, expressed mood, and reflection."""

import json
import logging
import re

from services.llm_provider import generate_response
from services.ai.episode_detector import EpisodeDetector
from services.ai.mood_detector import MOOD_LABELS, MIN_CONFIDENCE
from services.ai.prompts.turn_analyzer_prompt import PROMPT

logger = logging.getLogger(__name__)

MAX_CONTEXT_MESSAGES = 8
MAX_TIMELINE_ENTRIES = 5
MAX_REFLECTION_ENTRIES = 5


class TurnAnalyzer:
    """Produce episode, mood, and optional reflection decisions in one call."""

    def __init__(self, episode_detector=None):
        self.episode_detector = episode_detector or EpisodeDetector()

    def analyze(
        self,
        active_episode,
        latest_message,
        conversation=None,
        previous_moods=None,
        previous_reflections=None,
    ):
        prompt = self.build_prompt(
            active_episode=active_episode,
            latest_message=latest_message,
            conversation=conversation,
            previous_moods=previous_moods,
            previous_reflections=previous_reflections,
        )

        try:
            # One request returns all three structured decisions.
            payload = self._parse_response(generate_response(prompt))
        except Exception:
            # Analysis should not prevent the response from being generated.
            logger.exception("Turn analysis failed; using safe defaults.")
            payload = {}

        episode_decision = self._validate_episode(
            payload.get("episode") if isinstance(payload, dict) else None,
            active_episode,
        )
        mood_decision = self._validate_mood(
            payload.get("mood") if isinstance(payload, dict) else None,
        )
        reflection = self._validate_reflection(
            payload.get("reflection") if isinstance(payload, dict) else None,
            episode_decision["action"],
        )

        # Preserve the pre-existing deterministic explicit-resolution safeguard.
        if active_episode is not None and self.episode_detector.is_explicit_resolution(
            active_episode, latest_message
        ):
            episode_decision = {
                "action": "CLOSE",
                "title": None,
                "reason": (
                    "The user explicitly stated that the active issue no longer affects them."
                ),
                "confidence": 0.96,
                "detection_method": "explicit_resolution_rule",
            }

        return {
            "episode": episode_decision,
            "mood": mood_decision,
            "reflection": reflection,
        }

    def build_prompt(
        self,
        active_episode,
        latest_message,
        conversation=None,
        previous_moods=None,
        previous_reflections=None,
    ):
        if active_episode is None:
            episode_text = "No active episode."
        else:
            episode_text = (
                f"Episode ID: {active_episode.episode_id}\n"
                f"Title: {active_episode.title or 'Untitled'}\n"
                f"Status: {active_episode.status}\n"
                f"Summary: {active_episode.summary or 'Not yet summarized'}"
            )

        messages = list(conversation or [])
        # The latest user message is supplied separately; remove only a final
        # duplicate so an earlier identical message is not accidentally lost.
        if messages and isinstance(messages[-1], dict):
            last = messages[-1]
            if (
                str(last.get("role", "")).strip().lower() == "user"
                and str(last.get("content", "")).strip()
                == (latest_message or "").strip()
            ):
                messages = messages[:-1]

        context_lines = []
        for item in messages[-MAX_CONTEXT_MESSAGES:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = item.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                context_lines.append(f"{role.capitalize()}: {content.strip()[:1000]}")
        recent_context = "\n".join(context_lines) or "No prior conversation available."

        mood_lines = []
        for item in (previous_moods or [])[-MAX_TIMELINE_ENTRIES:]:
            mood = getattr(item, "mood", None)
            created_at = getattr(item, "created_at", None)
            if mood:
                mood_lines.append(f"{created_at}: {mood}" if created_at else str(mood))
        mood_timeline = "\n".join(mood_lines) or "No moods recorded for this episode."

        reflection_lines = []
        for item in (previous_reflections or [])[-MAX_REFLECTION_ENTRIES:]:
            reflection = getattr(item, "reflection", None)
            created_at = getattr(item, "created_at", None)
            if reflection and reflection.strip():
                prefix = f"{created_at}: " if created_at else ""
                reflection_lines.append(prefix + reflection.strip())
        reflection_history = "\n".join(reflection_lines) or "No previous reflections."

        return PROMPT.format(
            episode=episode_text,
            recent_context=recent_context,
            latest_message=latest_message or "",
            mood_timeline=mood_timeline,
            reflection_history=reflection_history,
            allowed_moods=", ".join(sorted(MOOD_LABELS)),
        )

    @staticmethod
    def _parse_response(response):
        """Parse one JSON object, tolerating a surrounding Markdown fence."""
        if not isinstance(response, str) or not response.strip():
            return {}
        content = response.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content)
        start = content.find("{")
        if start < 0:
            return {}
        try:
            value, _ = json.JSONDecoder().raw_decode(content[start:])
        except (json.JSONDecodeError, TypeError):
            logger.warning("TurnAnalyzer returned malformed JSON.")
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _normalise_confidence(value):
        if isinstance(value, bool):
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        return confidence if 0.0 <= confidence <= 1.0 else None

    def _validate_episode(self, decision, active_episode):
        if not isinstance(decision, dict):
            decision = {}
        action = str(decision.get("action", "")).strip().upper()
        if active_episode is not None:
            if action not in {"CONTINUE", "CLOSE"}:
                action = "CONTINUE"
        elif action not in {"CREATE", "NONE"}:
            action = "NONE"

        title = None
        if action == "CREATE":
            candidate = decision.get("title")
            title = (
                candidate.strip()[:255]
                if isinstance(candidate, str) and candidate.strip()
                else "Personal Concern"
            )
        result = {
            "action": action,
            "title": title,
            "reason": str(decision.get("reason", ""))[:400],
        }
        confidence = self._normalise_confidence(decision.get("confidence"))
        if confidence is not None:
            result["confidence"] = confidence
        return result

    def _validate_mood(self, decision):
        if not isinstance(decision, dict):
            return {"label": None, "confidence": None, "evidence": None}
        raw_label = decision.get("label")
        label = raw_label.strip().lower() if isinstance(raw_label, str) else None
        if label in {"", "none", "null", "unknown", "unclear"}:
            label = None
        if label not in MOOD_LABELS:
            return {"label": None, "confidence": None, "evidence": None}
        confidence = self._normalise_confidence(decision.get("confidence"))
        evidence = decision.get("evidence")
        if (
            confidence is None
            or confidence < MIN_CONFIDENCE
            or not isinstance(evidence, str)
            or not evidence.strip()
        ):
            return {"label": None, "confidence": None, "evidence": None}
        return {
            "label": label,
            "confidence": confidence,
            "evidence": evidence.strip()[:500],
        }

    @staticmethod
    def _validate_reflection(value, episode_action):
        # Do not create reflections for casual turns that do not belong to an episode.
        if episode_action == "NONE" or not isinstance(value, str):
            return None
        reflection = " ".join(value.split()).strip()
        if not reflection:
            return None
        return reflection[:2000]
