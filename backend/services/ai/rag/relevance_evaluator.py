import json
import logging
import os
import re

from services.llm_provider import generate_response
from .prompts.relevance_evaluation import RELEVANCE_EVALUATION_PROMPT
from .identifier_utils import extract_identifiers

logger = logging.getLogger(__name__)
_USE_LLM_RERANK = os.getenv("RAG_LLM_RELEVANCE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _tokens(text):
    raw_tokens = [token.casefold() for token in re.findall(r"[A-Za-z]+\d+|[A-Za-z0-9]+", text or "")]
    canonical_ids = [
        identifier.casefold().replace("#", "")
        for identifier in extract_identifiers(text or "")
    ]
    return raw_tokens + canonical_ids


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
        raise ValueError("No JSON object in relevance response")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Relevance response must be an object")
    return data


class RelevanceEvaluator:
    """Deterministic evidence scoring with optional, non-authoritative LLM reranking."""

    def evaluate(self, original_query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        query_tokens = set(_tokens(original_query))
        requested_ids = {value.casefold() for value in extract_identifiers(original_query)}
        evaluated = []
        for candidate in candidates:
            content = candidate.get("content") or ""
            content_tokens = set(_tokens(content))
            overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            retrieval_score = float(candidate.get("retrieval_score") or 0.0)
            exact_ids = {value.casefold() for value in extract_identifiers(content)}
            id_coverage = (
                len(requested_ids & exact_ids) / len(requested_ids)
                if requested_ids else 0.0
            )
            exact_id_match = bool(requested_ids & exact_ids)
            has_evidence_fields = bool(re.search(
                r"strength of recommendation|certainty of evidence|recommendation strength|certainty",
                content,
                re.IGNORECASE,
            ))
            # Retrieval and lexical signals are deliberately combined; raw L2
            # distances are not treated as probabilities.
            score = 0.45 * retrieval_score + 0.30 * overlap + 0.20 * id_coverage
            if has_evidence_fields and any(term in original_query.casefold() for term in ("strength", "certainty", "evidence")):
                score += 0.05
            score = max(0.0, min(1.0, score))
            evaluated.append({
                **candidate,
                "identifier_coverage": id_coverage,
                "relevance": score,
                "deterministic_relevance": score,
                "evaluation_reason": "Hybrid retrieval, lexical overlap, exact-ID coverage and requested evidence-field matching.",
            })

        evaluated.sort(key=lambda item: item["relevance"], reverse=True)
        if not _USE_LLM_RERANK:
            return evaluated

        # LLM scores may adjust ordering slightly, but cannot discard an exact
        # recommendation-ID match or override the deterministic evidence score.
        compact = [{
            "id": item.get("id"),
            "title": item.get("title"),
            "content": (item.get("content") or "")[:3200],
        } for item in evaluated[:20]]
        prompt = (
            f"Original user query:\n{original_query}\n\n"
            f"Retrieved candidates:\n{json.dumps(compact, ensure_ascii=False)}"
        )
        try:
            raw = generate_response(
                [{"role": "user", "content": prompt}],
                RELEVANCE_EVALUATION_PROMPT,
                max_completion_tokens=1200,
            )
            rows = _parse_json(raw).get("evaluations")
            if not isinstance(rows, list):
                raise ValueError("Missing evaluations list")
            by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}
            for item in evaluated:
                row = by_id.get(item.get("id"))
                if not row:
                    continue
                try:
                    llm_score = max(0.0, min(1.0, float(row.get("relevance", 0.0))))
                except (TypeError, ValueError):
                    continue
                # 75% deterministic score, 25% model judgement.
                item["relevance"] = 0.75 * item["deterministic_relevance"] + 0.25 * llm_score
                item["evaluation_reason"] = str(row.get("reason", "LLM reranking plus deterministic score"))[:300]
            evaluated.sort(key=lambda item: item["relevance"], reverse=True)
        except Exception:
            logger.exception("LLM relevance reranking failed; retaining deterministic ranking.")
        return evaluated
