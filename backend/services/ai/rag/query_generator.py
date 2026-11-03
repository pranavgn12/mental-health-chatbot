import json
import logging

from services.llm_provider import generate_response
from .prompts.query_generation import QUERY_GENERATION_PROMPT

logger = logging.getLogger(__name__)


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
        raise ValueError("Query-generation response must be a JSON object")
    return data


class QueryGenerator:
    """Create a focused query pair without losing the user's original wording."""

    def generate(self, user_query: str) -> list[str]:
        query = (user_query or "").strip()
        if not query:
            return []

        secondary = None
        try:
            raw = generate_response(
                [{"role": "user", "content": query}],
                QUERY_GENERATION_PROMPT,
                max_completion_tokens=256,
            )
            data = _parse_json(raw)
            generated = data.get("queries", [])
            if isinstance(generated, list):
                for item in generated:
                    if not isinstance(item, str) or not item.strip():
                        continue
                    candidate = item.strip()
                    if candidate.casefold() != query.casefold():
                        secondary = candidate
                        break
            if secondary is None:
                logger.warning("Query generator returned no valid complementary query.")
        except Exception:
            logger.exception("Query generation failed; using deterministic fallback.")

        # Keep the exact user query first: this preserves dates, named entities,
        # explicit web-search intent, and other constraints that generation may omit.
        if not secondary:
            secondary = f"{query} official sources, recent evidence, and practical guidance"

        return [query, secondary]
