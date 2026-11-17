import logging

from services.llm_provider import generate_response
from services.ai.json_parser import JSONParser
from services.ai.prompts.episode_detector_prompt import PROMPT

logger = logging.getLogger(__name__)


class EpisodeDetector:
    """Use the LLM to interpret episode lifecycle from conversational context."""

    def __init__(self):
        self.json_parser = JSONParser()

    def detect(self, active_episode, latest_message, recent_messages=None):
        prompt = self.build_prompt(active_episode, latest_message, recent_messages)
        response = generate_response(prompt)
        decision = self.json_parser.parse(response)
        if not isinstance(decision, dict):
            raise ValueError("Episode detector returned a non-object response")

        action = str(decision.get("action", "")).strip().upper()
        allowed = {"CREATE", "CONTINUE", "CLOSE", "NONE"}
        if action not in allowed:
            logger.warning("Invalid episode action %r; applying safe default", action)
            action = "CONTINUE" if active_episode is not None else "NONE"

        # Keep only lifecycle invariants deterministic. The semantic decision
        # (including whether a concern is resolved) belongs to the LLM.
        if active_episode is not None and action in {"CREATE", "NONE"}:
            action = "CONTINUE"
        elif active_episode is None and action in {"CONTINUE", "CLOSE"}:
            action = "NONE"

        title = decision.get("title") if action == "CREATE" else None
        if action == "CREATE" and not isinstance(title, str):
            title = "Personal Concern"

        return {
            **decision,
            "action": action,
            "title": title,
            "reason": str(decision.get("reason", ""))[:400],
        }

    def build_prompt(self, active_episode, latest_message, recent_messages=None):
        if active_episode is None:
            episode = "None"
        else:
            episode = f"""
Episode ID: {active_episode.episode_id}
Title: {active_episode.title}
Status: {active_episode.status}
Summary: {active_episode.summary or 'Not yet summarized'}
"""

        context_lines = []
        for item in (recent_messages or [])[-8:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = item.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                if role == "user" and content.strip() == (latest_message or "").strip():
                    continue
                context_lines.append(f"{role.capitalize()}: {content.strip()[:1000]}")

        recent_context = "\n".join(context_lines) or "No recent context supplied."
        return PROMPT.format(
            episode=episode,
            message=latest_message or "",
            recent_context=recent_context,
        )
