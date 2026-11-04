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
        """Update patterns and long-term memory with one LLM request.

        Profile theme generation remains a separate, optional request after the
        combined analysis. Pattern rows and the memory row are committed together.
        """
        if episode is None:
            raise ValueError("Episode does not exist.")
        if episode.user_id != user_id:
            raise ValueError("Episode does not belong to this user.")
        if not episode.summary or not episode.summary.strip():
            raise ValueError("Episode summary is empty.")

        try:
            logger.info(
                "Updating patterns and long-term memory for user=%s episode=%s",
                user_id,
                episode.episode_id,
            )
            record = (
                db.query(LongTermMemoryModel)
                .filter(LongTermMemoryModel.user_id == user_id)
                .with_for_update()
                .one_or_none()
            )
            previous_memory = record.memory if record and record.memory else None
            previous_themes = (
                record.profile_themes
                if record and isinstance(record.profile_themes, list)
                else []
            )

            # ONE LLM call returns both updated pattern candidates and memory.
            # Pattern evidence is then grounded/validated in Python.
            updated_memory = self.pattern_memory_analyzer.analyze_and_save(
                db=db,
                user_id=user_id,
                episode=episode,
                previous_memory=previous_memory,
            )
            if not updated_memory or not updated_memory.strip():
                raise ValueError("Combined pattern/memory analysis returned empty memory.")
            updated_memory = updated_memory.strip()

            # Profile generation intentionally remains separate. Failure here
            # must not block saving the updated memory or patterns.
            updated_themes = previous_themes
            try:
                generated_themes = self.profile_generator.generate(
                    memory=updated_memory,
                    previous_themes=previous_themes,
                )
                if not isinstance(generated_themes, list):
                    raise ValueError("Profile generator did not return a list.")
                valid_themes = [item for item in generated_themes if isinstance(item, dict)]
                if generated_themes and not valid_themes:
                    raise ValueError("Profile generator returned no valid theme objects.")
                if valid_themes:
                    json.dumps(valid_themes)  # Ensure JSON-column serializability.
                    updated_themes = valid_themes
                elif previous_themes:
                    logger.warning("No valid profile themes returned; preserving existing themes.")
            except Exception:
                logger.exception(
                    "Profile generation failed for user %s; preserving previous themes.",
                    user_id,
                )
                updated_themes = previous_themes

            if record is None:
                record = LongTermMemoryModel(
                    user_id=user_id,
                    memory=updated_memory,
                    profile_themes=updated_themes,
                )
                db.add(record)
            else:
                record.memory = updated_memory
                record.profile_themes = updated_themes

            # Commits the pattern changes staged by PatternMemoryAnalyzer too.
            db.commit()
            db.refresh(record)
            logger.info(
                "Memory update complete for user=%s; profile_themes=%d",
                user_id,
                len(record.profile_themes or []),
            )
            return record.memory

        except Exception:
            db.rollback()
            logger.exception(
                "Long-term memory update failed for user %s, episode %s",
                user_id,
                getattr(episode, "episode_id", None),
            )
            raise

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
