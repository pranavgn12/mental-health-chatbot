"""One-call recurring-pattern detection and long-term-memory consolidation."""

import json
import logging
import re
from datetime import datetime
from uuid import uuid4

from services.llm_provider import generate_response
from services.ai.pattern_detector import (
    _build_episode_source,
    _normalise_grounded_evidence,
)
from models import Episode, EpisodeMood, EpisodePattern, EpisodeReflection

logger = logging.getLogger(__name__)

MAX_PRIOR_EPISODES = 8
MAX_OUTPUT_PATTERNS = 5

PROMPT = """
You perform two connected tasks for a mental-health support assistant:
(A) identify evidence-based recurring patterns, and
(B) update the user's long-term memory using those patterns.
Complete both tasks in this ONE response. Do not diagnose or infer a disorder.

EPISODE EVIDENCE (current episode first, then up to eight prior closed episodes)
-------------------------------------------------------------------------------
{episode_sources}

EXISTING PATTERNS
-----------------
{existing_patterns}

PREVIOUS LONG-TERM MEMORY
-------------------------
{previous_memory}

The current episode's summary and reflections appear in the first evidence block above.
Use those lines for memory consolidation; do not request or invent additional context.

TASK A — RECURRING PATTERNS
---------------------------
Identify recurring patterns supported by at least two distinct observations.
Use only the episode evidence supplied above. Do not treat a one-off event as
recurring. Return ONLY new patterns or existing patterns whose status or evidence
materially changed. Do not repeat unchanged existing patterns.
Return no more than five patterns in this response. Prefer the most important
and best-supported patterns. Update an existing pattern by returning its exact
existing description when it still represents the same pattern.
For each returned pattern, include exactly two distinct evidence items when
available, preferably from different episodes. Copy each complete source line
EXACTLY as written. Do not invent IDs, details, moods, dates, quotations, or
evidence. If there is insufficient evidence, return an empty patterns array.

Each pattern must contain:
- category: trigger, coping, mood, behavior, or general
- description: concise, cautious description
- confidence: number from 0 to 1
- evidence: array of objects with episode_id and exact detail
- status: ACTIVE or RESOLVED

TASK B — LONG-TERM MEMORY
-------------------------
Update the previous long-term memory using the current episode summary,
reflections, and the recurring patterns you identified in Task A.
- Merge useful new information with existing memory and remove duplicates.
- Preserve useful older information unless it is contradicted by newer evidence.
- Treat tentative patterns as hypotheses, not facts.
- Prefer recurring struggles, coping strategies that consistently help,
  long-term goals, important relationships, lasting preferences, and meaningful
  personal insights.
- Do not store temporary moods, one-time events, dates, small talk, or short-lived details.
- Keep the memory concise: at most 8 short bullet points, one sentence each.
- Preserve the most important durable information; omit low-value details.
- Return plain text for memory, not nested JSON or Markdown fences.

OUTPUT
------
Return ONLY one valid JSON object with exactly these top-level keys:
{{
  "patterns": [
    {{
      "category": "general",
      "description": "concise recurring pattern",
      "confidence": 0.0,
      "evidence": [
        {{"episode_id": "exact episode ID", "detail": "exact source line"}}
      ],
      "status": "ACTIVE"
    }}
  ],
  "memory": "Updated long-term memory text"
}}
"""


class PatternMemoryAnalyzer:
    """Detect patterns and consolidate long-term memory in one inference."""

    def analyze_and_save(self, db, user_id, episode, previous_memory):
        context = self._build_pattern_context(db, episode)

        prompt = PROMPT.format(
            episode_sources=context["episode_sources"],
            existing_patterns=context["existing_patterns_text"],
            previous_memory=(previous_memory or "No previous long-term memory."),
        )

        raw = generate_response(
            prompt,
            max_completion_tokens=8192,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        result = self._parse_response(raw)
        if not isinstance(result, dict):
            raise ValueError("Combined pattern/memory analyzer returned invalid JSON.")

        patterns = result.get("patterns")
        memory = result.get("memory")
        if not isinstance(patterns, list):
            raise ValueError("Combined analyzer did not return a patterns list.")
        if not isinstance(memory, str) or not memory.strip():
            raise ValueError("Combined analyzer returned empty long-term memory.")

        # Pattern records and memory are staged in the same DB transaction.
        # LongTermMemory commits them together after Profile generation.
        # Keep persistence bounded even if the model ignores the prompt limit.
        self._save_patterns(
            db=db,
            episode=episode,
            items=patterns[:MAX_OUTPUT_PATTERNS],
            context=context,
        )
        return memory.strip()

    def _build_pattern_context(self, db, episode):
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
            .limit(MAX_PRIOR_EPISODES)
            .all()
        )

        episode_sources = []
        source_lines_by_episode = {}
        for source_episode in [episode] + prior_episodes:
            if source_episode.episode_id == episode.episode_id:
                moods = current_moods
                episode_reflections = current_reflections
            else:
                moods = (
                    db.query(EpisodeMood)
                    .filter(EpisodeMood.episode_id == source_episode.episode_id)
                    .order_by(EpisodeMood.created_at.asc())
                    .all()
                )
                episode_reflections = (
                    db.query(EpisodeReflection)
                    .filter(EpisodeReflection.episode_id == source_episode.episode_id)
                    .order_by(EpisodeReflection.created_at.asc())
                    .all()
                )

            source = _build_episode_source(source_episode, moods, episode_reflections)
            episode_sources.append(source)
            source_lines_by_episode[source_episode.episode_id] = {
                line for line in source.splitlines() if line.strip()
            }

        existing_text = "\n".join(
            f"{row.pattern_id}: {row.description} [{row.status}]"
            for row in existing
        ) or "No existing patterns."

        return {
            "existing": existing,
            "allowed_episode_ids": set(source_lines_by_episode),
            "source_lines_by_episode": source_lines_by_episode,
            "episode_sources": "\n\n".join(episode_sources),
            "existing_patterns_text": existing_text,
        }

    def _save_patterns(self, db, episode, items, context):
        existing = context["existing"]
        allowed_episode_ids = context["allowed_episode_ids"]
        source_lines_by_episode = context["source_lines_by_episode"]

        for item in items:
            if not isinstance(item, dict):
                continue
            description = item.get("description")
            confidence_value = item.get("confidence")
            if isinstance(confidence_value, bool):
                continue
            try:
                confidence = float(confidence_value)
            except (TypeError, ValueError):
                continue
            if (
                not isinstance(description, str)
                or not description.strip()
                or not 0.0 <= confidence <= 1.0
            ):
                continue

            grounded_evidence = _normalise_grounded_evidence(
                item.get("evidence"),
                allowed_episode_ids,
                source_lines_by_episode,
            )
            if len(grounded_evidence) < 2:
                # Do not persist patterns without two grounded observations.
                continue
            grounded_evidence = grounded_evidence[:2]

            category = str(item.get("category") or "general")[:80]
            status = (
                "RESOLVED"
                if str(item.get("status", "ACTIVE")).upper() == "RESOLVED"
                else "ACTIVE"
            )
            normalized_description = description.strip()
            match = next(
                (
                    row for row in existing
                    if row.description.casefold() == normalized_description.casefold()
                ),
                None,
            )

            if match is not None:
                match.confidence = max(match.confidence or 0.0, confidence)
                match.last_seen_at = datetime.utcnow()
                match.status = status
                previous_evidence = list(match.evidence or [])
                known = {
                    (entry.get("episode_id"), str(entry.get("detail", "")).casefold())
                    for entry in previous_evidence
                    if isinstance(entry, dict)
                }
                for entry in grounded_evidence:
                    key = (entry["episode_id"], entry["detail"].casefold())
                    if key not in known:
                        previous_evidence.append(entry)
                        known.add(key)
                match.evidence = previous_evidence
            else:
                row = EpisodePattern(
                    pattern_id=f"PAT_{uuid4().hex[:10].upper()}",
                    user_id=episode.user_id,
                    category=category,
                    description=normalized_description,
                    confidence=confidence,
                    evidence=grounded_evidence,
                    status=status,
                )
                db.add(row)
                existing.append(row)

    @staticmethod
    def _parse_response(response):
        if not isinstance(response, str) or not response.strip():
            return None
        content = response.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content)
        start = content.find("{")
        if start < 0:
            return None
        try:
            result, _ = json.JSONDecoder().raw_decode(content[start:])
        except json.JSONDecodeError:
            logger.warning("Combined pattern/memory analyzer returned malformed JSON.")
            return None
        return result if isinstance(result, dict) else None
