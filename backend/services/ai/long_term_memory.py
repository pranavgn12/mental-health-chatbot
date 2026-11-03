import json
import logging

from sqlalchemy.orm import Session

from models import (
    EpisodeReflection,
    EpisodePattern,
    LongTermMemory as LongTermMemoryModel
)

from services.ai.memory_consolidator import MemoryConsolidator
from services.ai.profile import Profile


logger = logging.getLogger(__name__)


class LongTermMemory:

    def __init__(self):
        self.memory_consolidator = MemoryConsolidator()
        self.profile_generator = Profile()

    # =====================================================
    # UPDATE LONG-TERM MEMORY AND PROFILE
    # =====================================================

    def update(
        self,
        db: Session,
        user_id,
        episode
    ):

        if episode is None:
            raise ValueError("Episode does not exist.")

        if episode.user_id != user_id:
            raise ValueError(
                "Episode does not belong to this user."
            )

        if not episode.summary or not episode.summary.strip():
            raise ValueError("Episode summary is empty.")

        try:

            print("\n========== LONG-TERM MEMORY UPDATE ==========")
            print("User ID:", user_id)
            print("Episode ID:", episode.episode_id)

            # -------------------------------------------------
            # 1. LOAD EXISTING MEMORY RECORD
            # -------------------------------------------------

            record = (
                db.query(LongTermMemoryModel)
                .filter(
                    LongTermMemoryModel.user_id == user_id
                )
                .with_for_update()
                .one_or_none()
            )

            previous_memory = (
                record.memory
                if record and record.memory
                else None
            )

            previous_themes = (
                record.profile_themes
                if record and isinstance(record.profile_themes, list)
                else []
            )

            print("\nPrevious Long-Term Memory:")
            print(previous_memory or "No previous memory.")

            print("\nPrevious Profile Themes:")
            print(previous_themes)

            # -------------------------------------------------
            # 2. LOAD EPISODE REFLECTIONS
            # -------------------------------------------------

            reflections = (
                db.query(EpisodeReflection)
                .filter(
                    EpisodeReflection.episode_id
                    == episode.episode_id
                )
                .order_by(
                    EpisodeReflection.created_at
                )
                .all()
            )

            print(
                "\nEpisode reflections:",
                len(reflections)
            )

            # -------------------------------------------------
            # 3. LOAD PERSISTENT PATTERNS
            # -------------------------------------------------

            patterns = (
                db.query(EpisodePattern)
                .filter(EpisodePattern.user_id == user_id,
                        EpisodePattern.status == "ACTIVE")
                .order_by(EpisodePattern.last_seen_at.desc())
                .all()
            )

            # -------------------------------------------------
            # 4. CONSOLIDATE LONG-TERM MEMORY
            # -------------------------------------------------

            updated_memory = self.memory_consolidator.consolidate(
                previous_memory=previous_memory,
                summary=episode.summary,
                reflections=reflections,
                patterns=patterns
            )

            if (
                not isinstance(updated_memory, str)
                or not updated_memory.strip()
            ):
                raise ValueError(
                    "Memory consolidation returned empty memory."
                )

            updated_memory = updated_memory.strip()

            print("\nUpdated Long-Term Memory:")
            print(updated_memory)

            # -------------------------------------------------
            # 5. GENERATE PROFILE THEMES
            # -------------------------------------------------

            # IMPORTANT:
            # Profile generation must never prevent memory saving.
            # If the LLM fails, retain the previous themes.

            updated_themes = previous_themes

            print("\nGenerating Profile Themes...")

            try:

                generated_themes = self.profile_generator.generate(
                    memory=updated_memory,
                    previous_themes=previous_themes
                )

                if not isinstance(generated_themes, list):
                    raise ValueError(
                        "Profile generator did not return a list."
                    )

                # Only accept dictionary-based themes.
                valid_themes = [
                    theme
                    for theme in generated_themes
                    if isinstance(theme, dict)
                ]

                if generated_themes and not valid_themes:
                    raise ValueError(
                        "Profile generator returned no valid theme objects."
                    )

                # An empty list is allowed when there are no meaningful
                # themes, but do not erase existing themes accidentally.
                if valid_themes:
                    # Verify that the result can be stored in a JSON column.
                    json.dumps(valid_themes)

                    updated_themes = valid_themes

                elif previous_themes:
                    logger.warning(
                        "Profile generator returned no themes. "
                        "Preserving previous themes for user %s.",
                        user_id
                    )

                print("\nProfile Themes:")
                print(updated_themes)

            except Exception:

                logger.exception(
                    "Profile generation failed for user %s. "
                    "Preserving previous themes.",
                    user_id
                )

                updated_themes = previous_themes

            # -------------------------------------------------
            # 6. SAVE MEMORY AND THEMES TOGETHER
            # -------------------------------------------------

            if record is None:

                record = LongTermMemoryModel(
                    user_id=user_id,
                    memory=updated_memory,
                    profile_themes=updated_themes
                )

                db.add(record)

            else:

                record.memory = updated_memory
                record.profile_themes = updated_themes

            # Commit consolidated memory even if Profile generation failed.
            db.commit()
            db.refresh(record)

            print("\n========== MEMORY UPDATE SUCCESS ==========")
            print("Memory saved:", bool(record.memory))
            print(
                "Profile themes saved:",
                len(record.profile_themes or [])
            )
            print("Updated at:", record.updated_at)
            print("===========================================\n")

            return record.memory

        except Exception:

            db.rollback()

            logger.exception(
                "Long-term memory update failed "
                "for user %s, episode %s",
                user_id,
                getattr(episode, "episode_id", None)
            )

            raise

    # =====================================================
    # RETRIEVE LONG-TERM MEMORY
    # =====================================================

    def retrieve(
        self,
        db: Session,
        user_id
    ):

        record = (
            db.query(LongTermMemoryModel)
            .filter(
                LongTermMemoryModel.user_id == user_id
            )
            .one_or_none()
        )

        if record is None:
            return None

        return record.memory

    # =====================================================
    # RETRIEVE PROFILE THEMES
    # =====================================================

    def retrieve_profile(
        self,
        db: Session,
        user_id
    ):

        record = (
            db.query(LongTermMemoryModel)
            .filter(
                LongTermMemoryModel.user_id == user_id
            )
            .one_or_none()
        )

        if record is None:
            return []

        return record.profile_themes or []




