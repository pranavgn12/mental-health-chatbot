import json
import logging

from services.llm_provider import generate_response
from .prompts.relevance_evaluation import RELEVANCE_EVALUATION_PROMPT

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
        raise ValueError("Relevance response must be a JSON object")
    return data


class RelevanceEvaluator:
    """LLM relevance assessment with fallback only when evaluation output is unusable."""

    def evaluate(self, original_query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        compact = []
        for item in candidates[:20]:
            compact.append({
                "id": item.get("id"),
                "source_type": item.get("source_type"),
                "title": item.get("title"),
                "url": item.get("url"),
                "content": (item.get("content") or "")[:1400],
            })

        prompt = (
            f"Original user query:\n{original_query}\n\n"
            f"Retrieved candidates:\n{json.dumps(compact, ensure_ascii=False)}"
        )

        try:
            raw = generate_response(
                [{"role": "user", "content": prompt}],
                RELEVANCE_EVALUATION_PROMPT,
                max_completion_tokens=1024,
            )
            data = _parse_json(raw)
            rows = data.get("evaluations")
            if not isinstance(rows, list):
                raise ValueError("Missing evaluations list")

            by_id = {item.get("id"): item for item in candidates if item.get("id")}
            evaluated = []
            matched_rows = 0

            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate = by_id.get(row.get("id"))
                if not candidate:
                    continue
                matched_rows += 1
                try:
                    score = max(0.0, min(1.0, float(row.get("relevance", 0))))
                except (TypeError, ValueError):
                    score = 0.0

                if row.get("useful") is True and score >= 0.35:
                    evaluated.append({
                        **candidate,
                        "relevance": score,
                        "evaluation_reason": str(row.get("reason", ""))[:300],
                    })

            # If the evaluator returned no usable IDs, its output is malformed
            # for these candidates. Fall back rather than silently losing all evidence.
            if matched_rows == 0:
                logger.warning("RAG evaluator returned no matching candidate IDs; using fallback.")
                return self._fallback(candidates)

            # A valid evaluation that explicitly rejects all candidates is respected.
            return evaluated

        except Exception:
            logger.exception("RAG relevance evaluation failed; applying fallback ranking.")
            return self._fallback(candidates)

    @staticmethod
    def _fallback(candidates):
        """Keep retrieved candidates when the LLM evaluator is unavailable."""
        ranked = []
        for item in candidates:
            score = 0.50
            distance = item.get("distance")
            if distance is not None:
                try:
                    score = max(0.0, min(1.0, 1.0 - float(distance)))
                except (TypeError, ValueError):
                    pass

            search_score = item.get("search_score")
            if search_score is not None:
                try:
                    score = max(score, max(0.0, min(1.0, float(search_score))))
                except (TypeError, ValueError):
                    pass

            ranked.append({
                **item,
                "relevance": score,
                "evaluation_reason": "Fallback ranking; LLM evaluation unavailable",
            })
        return ranked
