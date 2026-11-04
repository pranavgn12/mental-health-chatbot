from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from models import (
    Episode,
    EpisodeMood,
    EpisodeReflection,
    LongTermMemory,
    EpisodePattern,
)
from auth_utils import get_current_user


router = APIRouter()


# =========================================================
# REQUEST MODELS
# =========================================================

class LongTermMemoryUpdate(BaseModel):
    memory: str


class PatternUpdateRequest(BaseModel):
    category: Literal["trigger", "coping", "mood", "behavior", "general"]
    description: str = Field(min_length=1, max_length=5000)
    status: Literal["ACTIVE", "RESOLVED"]


# =========================================================
# HELPERS
# =========================================================

def _date(value):
    """Serialize timestamps with an explicit UTC timezone."""
    if value is None:
        return None

    from datetime import timezone

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)

    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _user_id(user):
    """Extract the database user ID from the JWT."""
    try:
        return int(user["user_id"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Invalid user token"
        )


def _serialize_pattern(pattern):
    """Convert an EpisodePattern ORM record into an API response."""
    evidence = pattern.evidence or []
    if not isinstance(evidence, list):
        evidence = []

    return {
        "id": pattern.id,
        "pattern_id": pattern.pattern_id,
        "category": pattern.category,
        "description": pattern.description,
        "confidence": float(pattern.confidence or 0.0),
        "evidence": evidence,
        "status": pattern.status,
        "first_seen_at": _date(pattern.first_seen_at),
        "last_seen_at": _date(pattern.last_seen_at),
        "created_at": _date(pattern.created_at),
        "updated_at": _date(pattern.updated_at),
    }


# =========================================================
# PROFILE / LONG-TERM MEMORY
# =========================================================

@router.get("/profile")
def get_profile_memory(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Return the user's consolidated long-term memory and Profile themes."""
    user_id = _user_id(user)

    record = (
        db.query(LongTermMemory)
        .filter(LongTermMemory.user_id == user_id)
        .one_or_none()
    )

    # No long-term memory has been created yet.
    if record is None:
        return {
            "memory": "",
            "themes": [],
            "updated_at": None,
        }

    themes = record.profile_themes or []
    if not isinstance(themes, list):
        themes = []

    return {
        "memory": record.memory or "",
        "themes": themes,
        "updated_at": _date(record.updated_at),
    }


@router.put("/profile")
def update_profile_memory(
    payload: LongTermMemoryUpdate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Update consolidated memory while preserving existing Profile themes."""
    user_id = _user_id(user)

    record = (
        db.query(LongTermMemory)
        .filter(LongTermMemory.user_id == user_id)
        .one_or_none()
    )

    try:
        if record is None:
            record = LongTermMemory(
                user_id=user_id,
                memory=payload.memory,
                profile_themes=[],
            )
            db.add(record)
        else:
            record.memory = payload.memory

        db.commit()
        db.refresh(record)

        themes = record.profile_themes or []
        if not isinstance(themes, list):
            themes = []

        return {
            "memory": record.memory or "",
            "themes": themes,
            "updated_at": _date(record.updated_at),
        }

    except Exception:
        db.rollback()
        raise


# =========================================================
# EPISODES
# =========================================================

@router.get("/episodes")
def get_episodes(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = _user_id(user)

    episodes = (
        db.query(Episode)
        .filter(Episode.user_id == user_id)
        .order_by(Episode.updated_at.desc(), Episode.id.desc())
        .all()
    )

    return [
        {
            "id": episode.id,
            "episode_id": episode.episode_id,
            "title": episode.title,
            "summary": episode.summary,
            "status": episode.status,
            "created_at": _date(episode.created_at),
            "updated_at": _date(episode.updated_at),
            "closed_at": _date(episode.closed_at),
        }
        for episode in episodes
    ]


# =========================================================
# MOODS
# =========================================================

@router.get("/moods")
def get_moods(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = _user_id(user)

    rows = (
        db.query(EpisodeMood, Episode.title)
        .join(Episode, Episode.episode_id == EpisodeMood.episode_id)
        .filter(Episode.user_id == user_id)
        .order_by(EpisodeMood.created_at.desc(), EpisodeMood.id.desc())
        .all()
    )

    return [
        {
            "id": mood.id,
            "episode_id": mood.episode_id,
            "episode_title": episode_title,
            "mood": mood.mood,
            "created_at": _date(mood.created_at),
        }
        for mood, episode_title in rows
    ]


# =========================================================
# REFLECTIONS
# =========================================================

@router.get("/reflections")
def get_reflections(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = _user_id(user)

    rows = (
        db.query(EpisodeReflection, Episode.title)
        .join(Episode, Episode.episode_id == EpisodeReflection.episode_id)
        .filter(Episode.user_id == user_id)
        .order_by(
            EpisodeReflection.created_at.desc(),
            EpisodeReflection.id.desc(),
        )
        .all()
    )

    return [
        {
            "id": reflection.id,
            "episode_id": reflection.episode_id,
            "episode_title": episode_title,
            "reflection": reflection.reflection,
            "created_at": _date(reflection.created_at),
        }
        for reflection, episode_title in rows
    ]


# =========================================================
# PATTERNS
# =========================================================

@router.get("/patterns")
def get_patterns(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Return all detected patterns belonging to the signed-in user."""
    user_id = _user_id(user)

    patterns = (
        db.query(EpisodePattern)
        .filter(EpisodePattern.user_id == user_id)
        .order_by(
            EpisodePattern.last_seen_at.desc(),
            EpisodePattern.id.desc(),
        )
        .all()
    )

    return [_serialize_pattern(pattern) for pattern in patterns]


@router.put("/patterns/{pattern_id}")
def update_pattern(
    pattern_id: str,
    payload: PatternUpdateRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Edit a user's pattern without modifying its detector evidence."""
    user_id = _user_id(user)

    pattern = (
        db.query(EpisodePattern)
        .filter(
            EpisodePattern.pattern_id == pattern_id,
            EpisodePattern.user_id == user_id,
        )
        .one_or_none()
    )

    if pattern is None:
        raise HTTPException(
            status_code=404,
            detail="Pattern not found",
        )

    description = payload.description.strip()
    if not description:
        raise HTTPException(
            status_code=422,
            detail="Pattern description cannot be empty",
        )

    try:
        pattern.category = payload.category
        pattern.description = description
        pattern.status = payload.status

        db.commit()
        db.refresh(pattern)

        return _serialize_pattern(pattern)

    except Exception:
        db.rollback()
        raise


# =========================================================
# DELETE EPISODE
# =========================================================

@router.delete("/episodes/{episode_id}")
def delete_episode(
    episode_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = _user_id(user)

    episode = (
        db.query(Episode)
        .filter(
            Episode.episode_id == episode_id,
            Episode.user_id == user_id,
        )
        .first()
    )

    if not episode:
        raise HTTPException(
            status_code=404,
            detail="Episode not found",
        )

    # Delete dependent records explicitly.
    db.query(EpisodeMood).filter(
        EpisodeMood.episode_id == episode.episode_id
    ).delete(synchronize_session=False)

    db.query(EpisodeReflection).filter(
        EpisodeReflection.episode_id == episode.episode_id
    ).delete(synchronize_session=False)

    db.delete(episode)
    db.commit()

    return {
        "message": "Episode and its mood/reflection records deleted"
    }


# =========================================================
# DELETE MOOD
# =========================================================

@router.delete("/moods/{mood_id}")
def delete_mood(
    mood_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = _user_id(user)

    mood = (
        db.query(EpisodeMood)
        .join(Episode, Episode.episode_id == EpisodeMood.episode_id)
        .filter(
            EpisodeMood.id == mood_id,
            Episode.user_id == user_id,
        )
        .first()
    )

    if not mood:
        raise HTTPException(
            status_code=404,
            detail="Mood record not found",
        )

    db.delete(mood)
    db.commit()

    return {"message": "Mood record deleted"}


# =========================================================
# DELETE REFLECTION
# =========================================================

@router.delete("/reflections/{reflection_id}")
def delete_reflection(
    reflection_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    user_id = _user_id(user)

    reflection = (
        db.query(EpisodeReflection)
        .join(Episode, Episode.episode_id == EpisodeReflection.episode_id)
        .filter(
            EpisodeReflection.id == reflection_id,
            Episode.user_id == user_id,
        )
        .first()
    )

    if not reflection:
        raise HTTPException(
            status_code=404,
            detail="Reflection not found",
        )

    db.delete(reflection)
    db.commit()

    return {"message": "Reflection deleted"}

