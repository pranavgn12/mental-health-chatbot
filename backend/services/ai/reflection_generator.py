"""Persistence helpers for reflections produced by TurnAnalyzer.

Reflection generation itself now happens inside TurnAnalyzer. This module does
not make an LLM request; it validates, deduplicates, and persists that result.
"""

import logging

from models import EpisodeReflection

logger = logging.getLogger(__name__)

MAX_PREVIOUS_REFLECTIONS = 5


class ReflectionGenerator:
    """Load reflection context and persist a precomputed reflection."""

    def get_recent_reflections(self, db, episode):
        if episode is None:
            return []
        rows = (
            db.query(EpisodeReflection)
            .filter(EpisodeReflection.episode_id == episode.episode_id)
            .order_by(EpisodeReflection.created_at.desc(), EpisodeReflection.id.desc())
            .limit(MAX_PREVIOUS_REFLECTIONS)
            .all()
        )
        return list(reversed(rows))

    def save_decision(self, db, episode, decision):
        """Save a meaningful reflection without calling the LLM again."""
        if episode is None or not isinstance(decision, str):
            return False

        reflection = " ".join(decision.split()).strip()
        if not reflection:
            return False

        latest = (
            db.query(EpisodeReflection)
            .filter(EpisodeReflection.episode_id == episode.episode_id)
            .order_by(EpisodeReflection.created_at.desc(), EpisodeReflection.id.desc())
            .first()
        )
        if latest is not None and latest.reflection:
            previous_normalized = " ".join(latest.reflection.split()).casefold()
            if previous_normalized == reflection.casefold():
                logger.debug("Skipping duplicate reflection for episode %s", episode.episode_id)
                return False

        row = EpisodeReflection(
            episode_id=episode.episode_id,
            reflection=reflection[:2000],
        )
        try:
            db.add(row)
            # This counter is retained for schema compatibility; reset it now
            # that reflection generation is event-driven rather than delayed.
            if hasattr(episode, "messages_since_last_reflection"):
                episode.messages_since_last_reflection = 0
            db.commit()
            db.refresh(row)
            logger.info("Saved reflection %s for episode %s", row.id, episode.episode_id)
            return True
        except Exception:
            db.rollback()
            logger.exception("Failed to save reflection for episode %s", episode.episode_id)
            raise
