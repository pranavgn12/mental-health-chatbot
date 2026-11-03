import logging
from uuid import uuid4
from datetime import datetime

from services.llm_provider import generate_response
from services.ai.json_parser import JSONParser
from models import EpisodePattern, EpisodeMood, EpisodeReflection, Episode

logger = logging.getLogger(__name__)

PATTERN_PROMPT = """You identify cautious, evidence-based recurring patterns in a mental-health support journal.
Do not diagnose or infer a disorder. A pattern must be supported by multiple observations or episodes.
Do not label a one-off event as recurring.

Evidence sources are grouped by episode ID. Use ONLY the episode IDs and source details shown below.
For every evidence item, copy one concise source detail EXACTLY as written. Do not paraphrase it.
A pattern should normally cite at least two distinct observations or two distinct episodes.
If evidence is insufficient, return no pattern.

Episode evidence sources:
{episode_sources}

Existing patterns:
{existing}

Return ONLY valid JSON in this exact shape:
{{
  "patterns": [
    {{
      "category": "trigger|coping|mood|behavior|general",
      "description": "concise recurring pattern",
      "confidence": 0.0,
      "evidence": [
        {{
          "episode_id": "an exact episode ID from the sources above",
          "detail": "an exact source detail copied from that episode"
        }}
      ],
      "status": "ACTIVE|RESOLVED"
    }}
  ]
}}

Rules:
- confidence must be a number from 0 to 1.
- evidence must be a non-empty array of objects with episode_id and detail.
- Every episode_id must be one of the IDs in the supplied evidence sources.
- Every detail must be copied exactly from a supplied source line for that episode.
- Do not invent dates, events, moods, quotes, or evidence.
- Return {{"patterns": []}} when evidence is insufficient.
"""


def _build_episode_source(episode, moods, reflections):
    """Build auditable source lines for one episode."""
    lines = [f"Episode ID: {episode.episode_id}"]
    summary = (episode.summary or "").strip()
    if summary:
        lines.append(f"Summary: {summary}")

    for mood in moods:
        timestamp = mood.created_at.isoformat() if mood.created_at else "unknown time"
        mood_line = (
            f"Mood observation: {timestamp} | mood={mood.mood} | "
            f"confidence={mood.confidence}"
        )
        lines.append(mood_line)
        if mood.evidence and mood.evidence.strip():
            lines.append(f"Mood evidence: {mood.evidence.strip()}")

    for reflection in reflections:
        if reflection.reflection and reflection.reflection.strip():
            lines.append(f"Reflection: {reflection.reflection.strip()}")

    return "\n".join(lines)


def _normalise_grounded_evidence(evidence, allowed_episode_ids, source_lines_by_episode):
    """Accept only evidence objects that quote a source line for an allowed episode."""
    if not isinstance(evidence, list):
        return []

    grounded = []
    seen = set()
    for entry in evidence:
        if not isinstance(entry, dict):
            continue
        episode_id = entry.get("episode_id")
        detail = entry.get("detail")
        if episode_id not in allowed_episode_ids:
            continue
        if not isinstance(detail, str) or not detail.strip():
            continue

        detail = detail.strip()
        source_lines = source_lines_by_episode.get(episode_id, set())
        if detail not in source_lines:
            continue

        key = (episode_id, detail.casefold())
        if key in seen:
            continue
        seen.add(key)
        grounded.append({"episode_id": episode_id, "detail": detail})

    return grounded


class PatternDetector:
    def __init__(self):
        self.parser = JSONParser()

    def detect_and_save(self, db, episode):
        current_moods = (
            db.query(EpisodeMood)
            .filter(EpisodeMood.episode_id == episode.episode_id)
            .order_by(EpisodeMood.created_at.asc())
            .all()
        )
        current_reflections = (
            db.query(EpisodeReflection)
            .filter(EpisodeReflection.episode_id == episode.episode_id)
            .order_by(EpisodeReflection.created_at.asc())
            .all()
        )
        existing = (
            db.query(EpisodePattern)
            .filter(EpisodePattern.user_id == episode.user_id)
            .all()
        )
        prior_episodes = (
            db.query(Episode)
            .filter(
                Episode.user_id == episode.user_id,
                Episode.status == "CLOSED",
                Episode.episode_id != episode.episode_id,
            )
            .order_by(Episode.closed_at.desc())
            .limit(8)
            .all()
        )

        episodes_for_analysis = [episode] + prior_episodes
        episode_sources = []
        source_lines_by_episode = {}

        for source_episode in episodes_for_analysis:
            if source_episode.episode_id == episode.episode_id:
                moods = current_moods
                reflections = current_reflections
            else:
                moods = (
                    db.query(EpisodeMood)
                    .filter(EpisodeMood.episode_id == source_episode.episode_id)
                    .order_by(EpisodeMood.created_at.asc())
                    .all()
                )
                reflections = (
                    db.query(EpisodeReflection)
                    .filter(EpisodeReflection.episode_id == source_episode.episode_id)
                    .order_by(EpisodeReflection.created_at.asc())
                    .all()
                )

            source_text = _build_episode_source(source_episode, moods, reflections)
            episode_sources.append(source_text)
            source_lines_by_episode[source_episode.episode_id] = {
                line for line in source_text.splitlines() if line.strip()
            }

        allowed_episode_ids = set(source_lines_by_episode)
        existing_text = "\n".join(
            f"{pattern.pattern_id}: {pattern.description} [{pattern.status}]"
            for pattern in existing
        ) or "None"

        prompt = PATTERN_PROMPT.format(
            episode_sources="\n\n".join(episode_sources),
            existing=existing_text,
        )
        raw = generate_response(prompt)
        parsed = self.parser.parse(raw) if raw else None

        if not isinstance(parsed, dict) or not isinstance(parsed.get("patterns"), list):
            logger.warning("Pattern detector returned invalid structure for %s", episode.episode_id)
            return []

        saved = []
        for item in parsed["patterns"]:
            if not isinstance(item, dict):
                continue

            description = item.get("description")
            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError):
                continue

            evidence = item.get("evidence")
            if (
                not isinstance(description, str)
                or not description.strip()
                or not 0.0 <= confidence <= 1.0
            ):
                continue

            grounded_evidence = _normalise_grounded_evidence(
                evidence, allowed_episode_ids, source_lines_by_episode
            )
            if len(grounded_evidence) < 2:
                # A single observation is not enough to establish a recurring pattern.
                continue

            category = str(item.get("category") or "general")[:80]
            status = (
                "RESOLVED"
                if str(item.get("status", "ACTIVE")).upper() == "RESOLVED"
                else "ACTIVE"
            )

            match = next(
                (
                    pattern for pattern in existing
                    if pattern.description.casefold() == description.strip().casefold()
                ),
                None,
            )
            if match:
                match.confidence = max(match.confidence or 0.0, confidence)
                match.last_seen_at = datetime.utcnow()
                match.status = status
                prior_evidence = list(match.evidence or [])
                known = {
                    (entry.get("episode_id"), str(entry.get("detail", "")).casefold())
                    for entry in prior_evidence if isinstance(entry, dict)
                }
                for entry in grounded_evidence:
                    key = (entry["episode_id"], entry["detail"].casefold())
                    if key not in known:
                        prior_evidence.append(entry)
                        known.add(key)
                match.evidence = prior_evidence
                saved.append(match)
            else:
                record = EpisodePattern(
                    pattern_id=f"PAT_{uuid4().hex[:10].upper()}",
                    user_id=episode.user_id,
                    category=category,
                    description=description.strip(),
                    confidence=confidence,
                    evidence=grounded_evidence,
                    status=status,
                )
                db.add(record)
                existing.append(record)
                saved.append(record)

        db.commit()
        return saved
