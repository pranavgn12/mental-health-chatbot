import json
import logging
import os

from services.llm_provider import generate_response
from .prompts.query_analysis import QUERY_ANALYSIS_PROMPT
from .identifier_utils import extract_identifiers

logger = logging.getLogger(__name__)

_IMMEDIATE_DANGER_PHRASES = (
    "kill myself", "end my life", "suicidal right now", "about to hurt myself",
    "hurt myself right now", "self harm right now", "self-harm right now",
    "overdose", "in immediate danger",
)
_WEB_INTENT_PHRASES = (
    "search the internet", "search internet", "search online", "search the web",
    "search web", "web search", "internet search", "browse the web",
    "browse online", "look it up online", "look this up online", "find it online",
    "find online sources", "from the internet", "from internet", "use the internet",
    "use web search",
)
_CURRENT_INFO_PHRASES = (
    "latest", "current", "recent", "new research", "statistics", "2025", "2026",
    "this year", "this week", "this month", "today", "as of", "up-to-date",
    "up to date", "updated", "newest", "current guidelines",
)
_INFORMATION_REQUEST_PHRASES = (
    "what ", "why ", "how ", "when ", "where ", "which ", "who ",
    "can you ", "could you ", "would you ", "should i ", "is it ",
    "are there ", "do i ", "does it ", "tell me ", "explain ",
    "help me understand", "give me ", "suggest ", "recommend ",
    "tips for ", "advice on ", "guidance on ", "techniques for ",
    "strategies for ", "ways to ", "symptoms of ", "difference between ",
    "research on ", "studies on ", "evidence about ", "compare ",
    "list ", "summarize ", "according to ", "based on ", "describe ",
    "outline ", "identify ", "what are ", "what were ", "which recommendations",
)


def _env_bool(name, default=False):
    value = os.getenv(name)
    return default if value is None else value.strip().casefold() in {"1", "true", "yes", "on"}


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
        raise ValueError("Query analysis response must be an object")
    return data


class QueryAnalyzer:
    """Route queries while respecting the RAG_ENABLE_WEB feature flag."""

    @staticmethod
    def _is_immediate_danger(query):
        q = (query or "").casefold()
        return any(phrase in q for phrase in _IMMEDIATE_DANGER_PHRASES)

    @staticmethod
    def _requires_web(query):
        q = (query or "").casefold()
        return (
            any(phrase in q for phrase in _WEB_INTENT_PHRASES)
            or any(phrase in q for phrase in _CURRENT_INFO_PHRASES)
        )

    @staticmethod
    def _web_enabled():
        # Web is OFF by default, making local-only evaluation safe by default.
        return _env_bool("RAG_ENABLE_WEB", False)

    def analyze(self, user_query: str) -> dict:
        query = (user_query or "").strip()
        if not query:
            return self._no_rag("Empty query")
        if self._is_immediate_danger(query):
            return self._no_rag("Immediate-danger wording detected; follow the separate safety response path")

        query_lower = query.casefold()
        information_seeking = (
            self._requires_web(query)
            or bool(extract_identifiers(query))
            or any(phrase in query_lower for phrase in _INFORMATION_REQUEST_PHRASES)
        )
        if not information_seeking:
            return self._no_rag("Conversational turn without an information-retrieval cue")

        web_enabled = self._web_enabled()

        # An explicit structured document reference is a direct local lookup.
        # This is generic (e.g. a section, table, standard or corpus-specific ID).
        if extract_identifiers(query) and not web_enabled:
            return {
                "needs_rag": True,
                "use_local": True,
                "use_web": False,
                "reason": "Explicit structured document reference; local knowledge retrieval required",
                "routing_method": "deterministic-document-reference",
            }

        try:
            raw = generate_response(
                [{"role": "user", "content": query}],
                QUERY_ANALYSIS_PROMPT,
                max_completion_tokens=256,
            )
            data = _parse_json(raw)
            needs_rag = data.get("needs_rag") is True
            use_local = data.get("use_local") is True and needs_rag
            use_web = data.get("use_web") is True and needs_rag and web_enabled
            reason = str(data.get("reason", ""))[:300]

            # Explicit/current web intent can only activate web if explicitly enabled.
            if web_enabled and self._requires_web(query):
                needs_rag = True
                use_web = True
                reason = (reason + " Web retrieval enabled for explicit/current request.").strip()[:400]
            if not (use_local or use_web):
                needs_rag = False
            return {
                "needs_rag": needs_rag,
                "use_local": use_local,
                "use_web": use_web,
                "reason": reason,
                "routing_method": "llm+web-feature-flag",
            }
        except Exception:
            logger.exception("RAG query analysis failed; using deterministic fallback.")
            return self._fallback(query, web_enabled=web_enabled)

    @staticmethod
    def _no_rag(reason):
        return {
            "needs_rag": False,
            "use_local": False,
            "use_web": False,
            "reason": reason,
            "routing_method": "fallback",
        }

    def _fallback(self, query, web_enabled=None):
        q = (query or "").casefold()
        if self._is_immediate_danger(q):
            return self._no_rag("Immediate-danger query; defer to safety protocol")
        if web_enabled is None:
            web_enabled = self._web_enabled()
        explicit_knowledge = any(phrase in q for phrase in (
            "what ", "why ", "how ", "explain ", "ways to ", "techniques for ",
            "symptoms of ", "difference between ", "research on ", "studies on ",
            "evidence about ", "scientific evidence", "papers on ", "recommendation",
            "guideline", "according to ", "from the document", "in the paper", "in the guideline",
        ))
        use_web = bool(web_enabled and self._requires_web(q))
        use_local = explicit_knowledge or (self._requires_web(q) and not web_enabled)
        if not use_local and not use_web:
            return self._no_rag("Fallback did not identify a clear knowledge retrieval need")
        return {
            "needs_rag": True,
            "use_local": bool(use_local),
            "use_web": bool(use_web),
            "reason": "Deterministic fallback after query-analysis failure",
            "routing_method": "fallback",
        }
