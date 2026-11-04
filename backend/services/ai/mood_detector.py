"""Mood timeline persistence; classification is performed by TurnAnalyzer."""

import logging

from models import EpisodeMood

logger = logging.getLogger(__name__)

MOOD_LABELS = {
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
    "neutral",
}
MIN_CONFIDENCE = 0.75
MAX_TIMELINE_ENTRIES = 5


class MoodDetector:
    """Load mood context and persist validated TurnAnalyzer decisions only."""

    def get_recent_timeline(self, db, episode):
        if episode is None:
            return []
        rows = (
            db.query(EpisodeMood)
            .filter(EpisodeMood.episode_id == episode.episode_id)
            .order_by(EpisodeMood.created_at.desc(), EpisodeMood.id.desc())
            .limit(MAX_TIMELINE_ENTRIES)
            .all()
        )
        return list(reversed(rows))

    def save_decision(self, db, episode, decision, message_id=None):
        if episode is None or message_id is None:
            return False

        # A retry for the same user message must never create another row.
        if (
            db.query(EpisodeMood)
            .filter(EpisodeMood.message_id == message_id)
            .first()
            is not None
        ):
            return False

        decision = decision if isinstance(decision, dict) else {}
        mood = self._normalise_mood(decision.get("label"))
        confidence = self._normalise_confidence(decision.get("confidence"))
        evidence = decision.get("evidence")
        if (
            mood is None
            or confidence is None
            or confidence < MIN_CONFIDENCE
            or not isinstance(evidence, str)
            or not evidence.strip()
        ):
            return False

        # Store transitions rather than repeating the same mood on every turn.
        latest = (
            db.query(EpisodeMood)
            .filter(EpisodeMood.episode_id == episode.episode_id)
            .order_by(EpisodeMood.created_at.desc(), EpisodeMood.id.desc())
            .first()
        )
        if latest is not None and (latest.mood or "").strip().lower() == mood:
            return False

        row = EpisodeMood(
            episode_id=episode.episode_id,
            mood=mood,
            message_id=message_id,
            confidence=confidence,
            evidence=evidence.strip()[:500],
        )
        try:
            db.add(row)
            db.commit()
            db.refresh(row)
            logger.debug(
                "Saved mood transition: episode=%s mood=%s message_id=%s",
                episode.episode_id,
                mood,
                message_id,
            )
            return True
        except Exception:
            db.rollback()
            logger.exception("Failed to save mood for episode %s", episode.episode_id)
            raise

    @staticmethod
    def _normalise_mood(value):
        if not isinstance(value, str):
            return None
        mood = value.strip().lower()
        if mood in {"", "none", "null", "unknown", "unclear"}:
            return None
        return mood if mood in MOOD_LABELS else None

    @staticmethod
    def _normalise_confidence(value):
        if isinstance(value, bool):
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        return confidence if 0.0 <= confidence <= 1.0 else None
