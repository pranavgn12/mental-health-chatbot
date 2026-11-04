"""Shared evidence-building and validation helpers for pattern analysis.

Pattern detection is no longer an independent LLM stage. The combined
PatternMemoryAnalyzer invokes the model once for patterns + long-term memory,
then uses the helpers in this module to validate evidence before persistence.
"""


def _build_episode_source(episode, moods, reflections):
    """Build auditable source lines for one episode."""
    lines = [f"Episode ID: {episode.episode_id}"]
    summary = (episode.summary or "").strip()
    if summary:
        lines.append(f"Summary: {summary}")

    for mood in moods:
        timestamp = mood.created_at.isoformat() if mood.created_at else "unknown time"
        lines.append(
            f"Mood observation: {timestamp} | mood={mood.mood} | confidence={mood.confidence}"
        )
        if mood.evidence and mood.evidence.strip():
            lines.append(f"Mood evidence: {mood.evidence.strip()}")

    for reflection in reflections:
        if reflection.reflection and reflection.reflection.strip():
            lines.append(f"Reflection: {reflection.reflection.strip()}")

    return "\n".join(lines)


def _normalise_grounded_evidence(evidence, allowed_episode_ids, source_lines_by_episode):
    """Accept only evidence objects copied exactly from allowed source lines."""
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
        if detail not in source_lines_by_episode.get(episode_id, set()):
            continue

        key = (episode_id, detail.casefold())
        if key not in seen:
            seen.add(key)
            grounded.append({"episode_id": episode_id, "detail": detail})

    return grounded
