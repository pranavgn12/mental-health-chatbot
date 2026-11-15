import json
import logging
import os
import re

from services.llm_provider import generate_response
from .prompts.query_generation import QUERY_GENERATION_PROMPT
from .identifier_utils import extract_identifiers, format_identifier

logger = logging.getLogger(__name__)

class QueryGenerator:
    """Generate focused queries while preserving corpus-known document identifiers."""

    def __init__(self, max_queries=None):
        self.max_queries = max(2, int(max_queries or os.getenv("RAG_MAX_QUERIES", "8")))

    @staticmethod
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
            raise ValueError("No JSON object in query-generation response")
        data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise ValueError("Query-generation response must be an object")
        return data

    def generate(self, user_query: str, known_identifiers=None) -> list[str]:
        query = (user_query or "").strip()
        if not query:
            return []

        queries = [query]
        # Identifier expansion is corpus-aware when a catalogue is supplied by
        # LocalRetriever. This works for guideline IDs, section/table numbers,
        # standards, paper-specific labels, and other document reference styles.
        identifiers = extract_identifiers(query, known_identifiers=known_identifiers)

        if identifiers:
            for identifier in identifiers:
                readable_id = format_identifier(identifier)
                # Keep the user's requested attributes and topic in the query;
                # don't inject WHO-specific words or assumed table columns.
                queries.append(f'"{readable_id}" {query}')
                if len(queries) >= self.max_queries:
                    break
            return list(dict.fromkeys(queries))[:self.max_queries]

        try:
            raw = generate_response(
                [{"role": "user", "content": query}],
                QUERY_GENERATION_PROMPT,
                max_completion_tokens=192,
            )
            data = self._parse_json(raw)
            generated = data.get("queries", [])
            if isinstance(generated, list):
                for item in generated:
                    if isinstance(item, str) and item.strip():
                        candidate = item.strip()
                        if candidate.casefold() != query.casefold():
                            queries.append(candidate)
                            break
        except Exception:
            logger.exception("Query generation failed; using the original query only.")

        return list(dict.fromkeys(queries))[:self.max_queries]
