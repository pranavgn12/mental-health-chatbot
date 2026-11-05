"""Best-effort pattern detection and independent long-term-memory generation."""
import json
import logging
import re
from datetime import datetime
from uuid import uuid4

from services.llm_provider import generate_response
from services.ai.pattern_detector import _build_episode_source, _normalise_grounded_evidence
from models import Episode, EpisodeMood, EpisodePattern, EpisodeReflection

logger = logging.getLogger(__name__)
MAX_PRIOR_EPISODES = 8
MAX_OUTPUT_PATTERNS = 5
CATEGORIES = {"trigger", "coping", "mood", "behavior", "general"}
STATUSES = {"ACTIVE", "RESOLVED"}

PATTERN_PROMPT = """
Identify cautious recurring patterns using only the evidence below.
A pattern requires at least two distinct exact observations. Never invent
episode IDs, dates, moods, quotations, or evidence. Return only changed/new
patterns. If evidence is insufficient, return {"patterns": []}.
category: trigger, coping, mood, behavior, or general.
confidence: JSON number 0..1 (e.g. 0.9, never "0. nine").
status: ACTIVE or RESOLVED.
Return valid JSON only, no Markdown fences, with exactly this structure:
{"patterns":[{"category":"general","description":"...","confidence":0.8,
"evidence":[{"episode_id":"exact ID","detail":"exact source line"},
{"episode_id":"exact ID","detail":"exact source line"}],"status":"ACTIVE"}]}

EPISODE EVIDENCE:
{episode_sources}

EXISTING PATTERNS:
{existing_patterns}
"""

MEMORY_PROMPT = """
Independently update durable long-term memory. Do NOT use detected patterns.
Use only the current episode and previous memory below. Preserve useful older
facts unless contradicted. Do not infer diagnoses, motives, traits, or recurring
patterns from one event. Prefer durable goals, relationships, preferences, and
coping strategies reported as helpful. At most 8 concise bullet points.
Return plain text only, not JSON or code fences.

CURRENT EPISODE
Title: {title}
Summary:
{summary}
Reflections:
{reflections}

PREVIOUS MEMORY:
{previous_memory}
"""


class PatternMemoryAnalyzer:
    def analyze_and_save(self, db, user_id, episode, previous_memory):
        context = self._build_pattern_context(db, episode)

        # Patterns are optional and isolated from memory generation.
        try:
            candidates = self._generate_patterns(context)
            if candidates:
                # Savepoint isolates pattern-write errors from the outer transaction.
                with db.begin_nested():
                    self._save_patterns(db, episode, candidates, context)
        except Exception:
            logger.exception(
                "Pattern analysis rejected/failed for user=%s episode=%s; ignoring patterns.",
                user_id, episode.episode_id,
            )

        # A separate LLM call means malformed pattern JSON cannot poison memory.
        try:
            memory = self._generate_memory(db, episode, previous_memory)
            if memory:
                return memory
        except Exception:
            logger.exception(
                "Memory generation failed for user=%s episode=%s; preserving previous memory.",
                user_id, episode.episode_id,
            )
        return previous_memory.strip() if isinstance(previous_memory, str) and previous_memory.strip() else None

    def _generate_patterns(self, context):
        raw = generate_response(
            PATTERN_PROMPT.format(
                episode_sources=context["episode_sources"],
                existing_patterns=context["existing_patterns_text"],
            ),
            max_completion_tokens=4096,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        result = self._parse_response(raw)
        if set(result) != {"patterns"} or not isinstance(result["patterns"], list):
            raise ValueError("Invalid pattern JSON schema.")

        valid = []
        for item in result["patterns"][:MAX_OUTPUT_PATTERNS]:
            candidate = self._validate_pattern(item, context)
            if candidate:
                valid.append(candidate)
            else:
                logger.warning("Discarding malformed or ungrounded pattern candidate.")
        return valid

    @staticmethod
    def _validate_pattern(item, context):
        if not isinstance(item, dict):
            return None
        category = str(item.get("category") or "general").strip().lower()
        status = str(item.get("status") or "ACTIVE").strip().upper()
        description = item.get("description")
        raw_confidence = item.get("confidence")
        if category not in CATEGORIES or status not in STATUSES:
            return None
        if not isinstance(description, str) or not description.strip():
            return None
        if isinstance(raw_confidence, bool):
            return None
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return None
        if not 0 <= confidence <= 1:
            return None

        grounded = _normalise_grounded_evidence(
            item.get("evidence"),
            context["allowed_episode_ids"],
            context["source_lines_by_episode"],
        )
        unique, seen = [], set()
        for entry in grounded:
            key = (entry.get("episode_id"), entry.get("detail"))
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        if len(unique) < 2:
            return None
        return {
            "category": category,
            "status": status,
            "description": description.strip(),
            "confidence": confidence,
            "evidence": unique[:2],
        }

    def _generate_memory(self, db, episode, previous_memory):
        reflections = (
            db.query(EpisodeReflection)
            .filter(EpisodeReflection.episode_id == episode.episode_id)
            .order_by(EpisodeReflection.created_at.asc())
            .all()
        )
        reflection_text = "\n".join(
            row.reflection.strip() for row in reflections
            if isinstance(row.reflection, str) and row.reflection.strip()
        ) or "No reflections recorded."
        summary = (episode.summary or "").strip()
        if not summary:
            raise ValueError("Episode summary is empty.")

        raw = generate_response(
            MEMORY_PROMPT.format(
                title=episode.title or "Untitled episode",
                summary=summary,
                reflections=reflection_text,
                previous_memory=previous_memory or "No previous memory.",
            ),
            max_completion_tokens=2048,
            reasoning_effort="low",
        )
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Memory generator returned empty content.")
        memory = re.sub(r"^\\s*```(?:text|markdown)?\\s*", "", raw.strip(), flags=re.I)
        memory = re.sub(r"\\s*```\\s*$", "", memory).strip()
        if not memory or memory.startswith("{") or len(memory) > 12000:
            raise ValueError("Memory generator returned malformed/oversized content.")
        return memory

    def _build_pattern_context(self, db, episode):
        current_moods = (
            db.query(EpisodeMood).filter(EpisodeMood.episode_id == episode.episode_id)
            .order_by(EpisodeMood.created_at.asc()).all()
        )
        current_reflections = (
            db.query(EpisodeReflection).filter(EpisodeReflection.episode_id == episode.episode_id)
            .order_by(EpisodeReflection.created_at.asc()).all()
        )
        existing = db.query(EpisodePattern).filter(EpisodePattern.user_id == episode.user_id).all()
        prior_episodes = (
            db.query(Episode)
            .filter(
                Episode.user_id == episode.user_id,
                Episode.status == "CLOSED",
                Episode.episode_id != episode.episode_id,
            )
            .order_by(Episode.closed_at.desc()).limit(MAX_PRIOR_EPISODES).all()
        )

        sources, lines_by_episode = [], {}
        for source_episode in [episode] + prior_episodes:
            if source_episode.episode_id == episode.episode_id:
                moods, reflections = current_moods, current_reflections
            else:
                moods = (
                    db.query(EpisodeMood).filter(EpisodeMood.episode_id == source_episode.episode_id)
                    .order_by(EpisodeMood.created_at.asc()).all()
                )
                reflections = (
                    db.query(EpisodeReflection).filter(EpisodeReflection.episode_id == source_episode.episode_id)
                    .order_by(EpisodeReflection.created_at.asc()).all()
                )
            source = _build_episode_source(source_episode, moods, reflections)
            sources.append(source)
            lines_by_episode[source_episode.episode_id] = {
                line for line in source.splitlines() if line.strip()
            }

        existing_text = "\n".join(
            f"{row.pattern_id}: {row.description} [{row.status}]" for row in existing
        ) or "No existing patterns."
        return {
            "existing": existing,
            "allowed_episode_ids": set(lines_by_episode),
            "source_lines_by_episode": lines_by_episode,
            "episode_sources": "\n\n".join(sources),
            "existing_patterns_text": existing_text,
        }

    def _save_patterns(self, db, episode, items, context):
        existing = context["existing"]
        for item in items:
            description = item["description"]
            match = next(
                (row for row in existing if row.description.casefold() == description.casefold()),
                None,
            )
            if match:
                match.confidence = max(match.confidence or 0.0, item["confidence"])
                match.last_seen_at = datetime.utcnow()
                match.status = item["status"]
                evidence = list(match.evidence or [])
                known = {
                    (entry.get("episode_id"), str(entry.get("detail", "")).casefold())
                    for entry in evidence if isinstance(entry, dict)
                }
                for entry in item["evidence"]:
                    key = (entry["episode_id"], entry["detail"].casefold())
                    if key not in known:
                        evidence.append(entry)
                        known.add(key)
                match.evidence = evidence
            else:
                row = EpisodePattern(
                    pattern_id=f"PAT_{uuid4().hex[:10].upper()}",
                    user_id=episode.user_id,
                    category=item["category"],
                    description=description,
                    confidence=item["confidence"],
                    evidence=item["evidence"],
                    status=item["status"],
                )
                db.add(row)
                existing.append(row)

    @staticmethod
    def _parse_response(response):
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Empty pattern response.")
        content = response.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\\s*", "", content, flags=re.I)
            content = re.sub(r"\\s*```$", "", content)
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed pattern JSON: %s", exc)
            raise ValueError("Malformed pattern JSON.") from exc
        if not isinstance(result, dict):
            raise ValueError("Pattern JSON root is not an object.")
        return result
