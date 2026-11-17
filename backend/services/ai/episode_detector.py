import logging
import re

from services.llm_provider import generate_response
from services.ai.json_parser import JSONParser
from services.ai.prompts.episode_detector_prompt import PROMPT

logger = logging.getLogger(__name__)


# These are explicit first-person resolution statements, not generic positive
# sentiment. They let a user close an episode in one turn without depending
# entirely on the LLM interpreting a paraphrase correctly.
_RESOLUTION_PATTERNS = (
    re.compile(r"\bno longer\b.{0,90}\b(?:afraid|scared|fear\w*|anxious|worried|nervous|stressed|overwhelmed)\b", re.I),
    re.compile(r"\b(?:not|isn't|aren't|am not|don't|do not)\b.{0,55}\b(?:feel\s+)?(?:afraid|scared|fear\w*|anxious|worried|nervous|stressed)\b.{0,35}\b(?:anymore|now)\b", re.I),
    re.compile(r"\b(?:overcame|have overcome|overcome|conquered|got over|moved past)\b.{0,70}\b(?:fear|anxiety|worry|stress|problem|issue|difficulty)\b", re.I),
    re.compile(r"\b(?:this|that|it|the problem|the issue)\b.{0,35}\b(?:is|has been|feels)\b.{0,25}\b(?:resolved|over|sorted out|behind me)\b", re.I),
    re.compile(r"\b(?:doesn't|does not|no longer)\s+bother me\b", re.I),
    re.compile(r"\b(?:i am|i'm|i feel)\s+(?:now\s+)?(?:confident|comfortable|at ease)\b.{0,50}\b(?:about|with|regarding)\b", re.I),
    # Decisive first-person behaviour change: "I won't procrastinate anymore",
    # "I will not put things off again", "I've stopped avoiding people".
    re.compile(r"\b(?:i\s+)?(?:won't|will not|never)\s+(?:keep\s+)?[a-z][a-z' -]{1,65}\b(?:anymore|any more|again)\b", re.I),
    re.compile(r"\b(?:i have|i've|i am|i'm)\s+(?:finally\s+)?(?:stopped|quit|given up|overcome|overcame)\s+[a-z][a-z' -]{1,65}", re.I),
    re.compile(r"\b(?:i have|i've)\s+(?:fixed|changed|sorted out|dealt with)\s+(?:my|the|this|that)\s+[a-z][a-z' -]{1,55}", re.I),
)
_UNCERTAINTY_PREFIX = re.compile(
    r"\b(?:maybe|perhaps|not sure|don't know|do not know|wonder if|whether|might|may)\b",
    re.I,
)

_CLOSURE_REVIEW_CUES = re.compile(
    r"\b(?:resolved|resolution|fixed|sorted|stopped|quit|given up|won't|will not|never|overcame|overcome|got over|"
    r"moved on|moving on|past it|behind me|doesn't bother|does not bother|"
    r"no longer|anymore|not anxious|not worried|not afraid|confident|"
    r"comfortable|at ease|better|improved|at peace|accepting it|"
    r"not an issue|fine now|okay now|again|another thing|different topic|"
    r"different concern|new concern|unrelated)\b",
    re.I,
)


class EpisodeDetector:
    def __init__(self):
        self.json_parser = JSONParser()

    def detect(self, active_episode, latest_message, recent_messages=None):
        if active_episode is None:
            prompt = self.build_prompt(None, latest_message, recent_messages)
        else:
            # Explicit resolution language should not need a second confirmation.
            if self._explicit_resolution(active_episode, latest_message):
                return {
                    "action": "CLOSE",
                    "title": None,
                    "reason": "The user explicitly stated that the active issue no longer affects them.",
                    "confidence": 0.96,
                    "detection_method": "explicit_resolution_rule",
                }

            # Most messages during an active episode are routine updates. They
            # do not need another classification call: only messages containing
            # plausible resolution or topic-transition cues are reviewed by the LLM.
            if not _CLOSURE_REVIEW_CUES.search(latest_message or ""):
                return {
                    "action": "CONTINUE",
                    "title": None,
                    "reason": "No clear resolution cue; continue the active episode.",
                    "confidence": 0.90,
                    "detection_method": "deterministic_continue",
                }
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

        # The model must not create a second episode while one is active,
        # or close/create an episode when none exists.
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

    def is_explicit_resolution(self, active_episode, latest_message):
        """Public wrapper used by the combined TurnAnalyzer."""
        return self._explicit_resolution(active_episode, latest_message)

    def _explicit_resolution(self, active_episode, latest_message):
        message = " ".join((latest_message or "").strip().split())
        if not message:
            return False

        # Do not treat tentative/uncertain language as a firm resolution.
        if _UNCERTAINTY_PREFIX.search(message.lower()):
            return False

        if not any(pattern.search(message) for pattern in _RESOLUTION_PATTERNS):
            return False

        # Avoid closing an unrelated episode when the user explicitly names
        # a different topic (e.g. dogs vs. the active exam-anxiety episode).
        title = (getattr(active_episode, "title", "") or "").lower()
        summary = (getattr(active_episode, "summary", "") or "").lower()
        episode_text = f"{title} {summary}"
        message_lower = message.lower()

        # Compare meaningful words from the active episode with the message.
        # Prefix matching handles common forms such as procrastination/procrastinate
        # without maintaining a topic-specific list of every possible issue.
        stop_words = {
            "about", "after", "again", "also", "been", "being", "does", "feel",
            "from", "have", "into", "issue", "just", "more", "much", "over",
            "problem", "really", "that", "the", "their", "them", "then", "there",
            "this", "very", "with", "will", "would", "user", "because", "when",
            "while", "your", "they", "what", "some", "such", "still", "often",
        }
        episode_tokens = {
            token for token in re.findall(r"[a-z]{4,}", episode_text)
            if token not in stop_words
        }
        message_tokens = {
            token for token in re.findall(r"[a-z]{4,}", message_lower)
            if token not in stop_words
        }
        matched_topic = any(
            left == right or left.startswith(right[:4]) or right.startswith(left[:4])
            for left in episode_tokens
            for right in message_tokens
            if min(len(left), len(right)) >= 4
        )

        # Pronouns are acceptable only when the message explicitly points back
        # to the tracked concern; this supports concise replies such as "I won't
        # do that anymore" without treating unrelated commitments as closure.
        has_anaphora = bool(re.search(
            r"\b(?:it|this|that|the issue|the problem|the same thing)\b", message_lower
        ))
        return matched_topic or has_anaphora

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
                # Latest user message is shown separately below.
                if role == "user" and content.strip() == (latest_message or "").strip():
                    continue
                context_lines.append(f"{role.capitalize()}: {content.strip()[:1000]}")

        recent_context = "\n".join(context_lines) or "No recent context supplied."
        return PROMPT.format(
            episode=episode,
            message=latest_message or "",
            recent_context=recent_context,
        )


