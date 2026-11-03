import json
import logging

from services.llm_provider import generate_response
from .prompts.query_analysis import QUERY_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


_IMMEDIATE_DANGER_PHRASES = (
    "kill myself",
    "end my life",
    "suicidal right now",
    "about to hurt myself",
    "hurt myself right now",
    "self harm right now",
    "self-harm right now",
    "overdose",
    "in immediate danger",
)

_WEB_INTENT_PHRASES = (
    "search the internet",
    "search internet",
    "search online",
    "search the web",
    "search web",
    "web search",
    "internet search",
    "browse the web",
    "browse online",
    "look it up online",
    "look this up online",
    "find it online",
    "find online sources",
    "from the internet",
    "from internet",
    "use the internet",
    "use web search",
)

_CURRENT_INFO_PHRASES = (
    "latest",
    "current",
    "recent",
    "new research",
    "statistics",
    "2025",
    "2026",
    "this year",
    "this week",
    "this month",
    "today",
    "as of",
    "up-to-date",
    "up to date",
    "updated",
    "newest",
    "current guidelines",
)


def _parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object in model response")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Query analysis response must be a JSON object")
    return data


class QueryAnalyzer:
    """Decide whether local and/or web evidence is needed for the latest message."""

    @staticmethod
    def _is_immediate_danger(query):
        q = (query or "").casefold()
        return any(phrase in q for phrase in _IMMEDIATE_DANGER_PHRASES)

    @staticmethod
    def _requires_web(query):
        q = (query or "").casefold()
        explicit_request = any(phrase in q for phrase in _WEB_INTENT_PHRASES)
        current_information = any(phrase in q for phrase in _CURRENT_INFO_PHRASES)
        return explicit_request or current_information

    def analyze(self, user_query: str) -> dict:
        query = (user_query or "").strip()
        if not query:
            return self._no_rag("Empty query")

        immediate_danger = self._is_immediate_danger(query)
        try:
            raw = generate_response(
                [{"role": "user", "content": query}],
                QUERY_ANALYSIS_PROMPT,
                max_completion_tokens=256,
            )
            data = _parse_json(raw)

            needs_rag = data.get("needs_rag") is True
            use_local = data.get("use_local") is True and needs_rag
            use_web = data.get("use_web") is True and needs_rag
            reason = str(data.get("reason", ""))[:300]

            # Deterministic guard: an explicit request to search online, or a
            # clearly time-sensitive query, must not be silently routed away
            # from web retrieval. Preserve the prompt's immediate-danger rule.
            if self._requires_web(query) and not immediate_danger:
                needs_rag = True
                use_web = True
                reason = (reason + " Explicit/current web retrieval requested.").strip()[:400]

            if not (use_local or use_web):
                needs_rag = False

            return {
                "needs_rag": needs_rag,
                "use_local": use_local,
                "use_web": use_web,
                "reason": reason,
                "routing_method": "llm+web-intent-guard",
            }
        except Exception:
            logger.exception("RAG query analysis failed; using deterministic fallback.")
            return self._fallback(query)

    @staticmethod
    def _no_rag(reason):
        return {
            "needs_rag": False,
            "use_local": False,
            "use_web": False,
            "reason": reason,
            "routing_method": "fallback",
        }

    def _fallback(self, query):
        """Use explicit wording and temporal clues if the LLM router is unavailable."""
        q = (query or "").casefold()
        if self._is_immediate_danger(q):
            return self._no_rag("Immediate-danger query; defer to the safety protocol")

        explicit_knowledge = any(phrase in q for phrase in (
            "what is ",
            "explain ",
            "how do i ",
            "how can i ",
            "ways to ",
            "techniques for ",
            "symptoms of ",
            "difference between ",
        ))
        use_web = self._requires_web(q)
        if not explicit_knowledge and not use_web:
            return self._no_rag("Fallback did not identify a clear knowledge or web-search need")

        return {
            "needs_rag": True,
            "use_local": explicit_knowledge,
            "use_web": use_web,
            "reason": "Deterministic fallback after query-analysis failure",
            "routing_method": "fallback",
        }
