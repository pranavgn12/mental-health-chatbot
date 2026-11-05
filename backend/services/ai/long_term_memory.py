import json
import logging

from sqlalchemy.orm import Session
from models import LongTermMemory as LongTermMemoryModel
from services.ai.pattern_memory_analyzer import PatternMemoryAnalyzer
from services.ai.profile import Profile

logger = logging.getLogger(__name__)


class LongTermMemory:
    def __init__(self):
        self.pattern_memory_analyzer = PatternMemoryAnalyzer()
        self.profile_generator = Profile()

    def update(self, db: Session, user_id, episode):
        """Best-effort consolidation; never propagate optional AI failures."""
        if episode is None or episode.user_id != user_id:
            logger.warning("Skipping memory update: missing/mismatched episode.")
            return None

        previous_memory = None
        try:
            record = (
                db.query(LongTermMemoryModel)
                .filter(LongTermMemoryModel.user_id == user_id)
                .with_for_update()
                .one_or_none()
            )
            previous_memory = record.memory if record and record.memory else None
            previous_themes = (
                record.profile_themes
                if record and isinstance(record.profile_themes, list) else []
            )

            # The analyzer stages independently validated pattern rows and
            # returns either validated memory, the prior memory, or an empty
            # string when the user has no durable memory yet.
            generated_memory = self.pattern_memory_analyzer.analyze_and_save(
                db=db, user_id=user_id, episode=episode,
                previous_memory=previous_memory,
            )
            updated_memory = (
                generated_memory.strip()
                if isinstance(generated_memory, str)
                else (previous_memory or "").strip()
            )
            prior_text = (previous_memory or "").strip()
            memory_changed = bool(updated_memory and updated_memory != prior_text)

            updated_themes = previous_themes
            if memory_changed:
                try:
                    generated = self.profile_generator.generate(
                        memory=updated_memory, previous_themes=previous_themes,
                    )
                    if not isinstance(generated, list):
                        raise ValueError("Profile generator did not return a list.")
                    valid = [item for item in generated if isinstance(item, dict)]
                    if valid:
                        json.dumps(valid)
                        updated_themes = valid
                except Exception:
                    logger.exception("Profile generation failed; preserving previous themes.")

                if record is None:
                    record = LongTermMemoryModel(
                        user_id=user_id, memory=updated_memory,
                        profile_themes=updated_themes,
                    )
                    db.add(record)
                else:
                    record.memory = updated_memory
                    record.profile_themes = updated_themes

            # Commit even when memory is empty/unchanged: valid pattern changes
            # may have been staged by the analyzer and must not be discarded.
            db.commit()
            if record is not None:
                db.refresh(record)
                return record.memory
            return previous_memory

        except Exception:
            db.rollback()
            logger.exception(
                "Long-term memory update failed for user=%s episode=%s; preserving prior memory.",
                user_id, getattr(episode, "episode_id", None),
            )
            return previous_memory

    def retrieve(self, db: Session, user_id):
        record = (
            db.query(LongTermMemoryModel)
            .filter(LongTermMemoryModel.user_id == user_id)
            .one_or_none()
        )
        return record.memory if record else None

    def retrieve_profile(self, db: Session, user_id):
        record = (
            db.query(LongTermMemoryModel)
            .filter(LongTermMemoryModel.user_id == user_id)
            .one_or_none()
        )
        return record.profile_themes or [] if record else []
