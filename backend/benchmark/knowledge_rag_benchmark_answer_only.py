#!/usr/bin/env python3
"""Knowledge RAG benchmark using the chatbot's production LocalRetriever.

The benchmark deliberately imports services.ai.rag.local_retriever.LocalRetriever
instead of implementing a second retriever. Ground-truth candidate discovery is
separate: evidence phrases are embedded against Chroma, source metadata matching
is whitespace/case normalized, and unmatched source labels fall back to an
unfiltered candidate search with an explicit warning.

Run from backend/:
    python benchmark/knowledge_rag_benchmark.py \
        --dataset benchmark/datasets/knowledge_rag.json --k 3 5 10

Run the offline tests:
    python benchmark/knowledge_rag_benchmark.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import math
import os
import re
import shutil
import sys
import textwrap
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_VERSION = "knowledge_rag_benchmark_v3"
CANDIDATE_DISCOVERY_VERSION = "source_resolved_evidence_pool_v2"
MAX_JUDGE_CHUNK_CHARS = 5000
MAX_ANSWER_CONTEXT_CHARS = 12000
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def find_backend_dir(explicit: str | None = None) -> Path:
    """Find the directory that contains services/ai/rag."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([SCRIPT_PATH.parent, *SCRIPT_PATH.parents])
    cwd = Path.cwd()
    candidates.extend([cwd, cwd / "backend", cwd.parent / "backend"])
    env_backend = os.getenv("MENTAL_HEALTH_BACKEND_DIR")
    if env_backend:
        candidates.insert(0, Path(env_backend).expanduser())
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if (resolved / "services" / "ai" / "rag" / "local_retriever.py").is_file():
            return resolved
    raise FileNotFoundError(
        "Could not locate backend/services/ai/rag/local_retriever.py. "
        "Place this script in backend/benchmark or pass --backend-dir."
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def resolve_path(raw: str | Path, base: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / candidate).resolve()


def normalize_source_label(value: Any) -> str:
    """Normalize source labels without changing meaningful filename tokens.

    The ingester in the supplied project archive stored some PDF basenames with
    embedded newlines. Collapsing Unicode whitespace makes those equal to the
    ordinary filename in the question dataset. We intentionally do not strip
    version suffixes or otherwise fuzzy-match different files.
    """
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).replace("\\", "/")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.rsplit("/", 1)[-1]
    return value.casefold()


def _extract_phrases(question: dict[str, Any]) -> list[dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    raw_many = question.get("evidence_phrases")
    if isinstance(raw_many, list):
        for index, item in enumerate(raw_many, start=1):
            if isinstance(item, str) and item.strip():
                phrases.append({"evidence_id": f"E{index:03d}", "phrase": item.strip()})
            elif isinstance(item, dict):
                phrase = item.get("evidence_phrase") or item.get("supporting_quote") or item.get("phrase")
                if isinstance(phrase, str) and phrase.strip():
                    phrases.append({
                        "evidence_id": str(item.get("evidence_id") or f"E{index:03d}"),
                        "phrase": phrase.strip(),
                        "source_document": item.get("source_document") or item.get("source"),
                    })
    if not phrases:
        raw = question.get("evidence_phrase")
        if isinstance(raw, str) and raw.strip():
            phrases.append({"evidence_id": "E001", "phrase": raw.strip()})
    if not phrases and isinstance(question.get("evidence"), list):
        for index, item in enumerate(question["evidence"], start=1):
            if not isinstance(item, dict):
                continue
            phrase = item.get("supporting_quote") or item.get("evidence_phrase") or item.get("phrase")
            if isinstance(phrase, str) and phrase.strip():
                phrases.append({
                    "evidence_id": str(item.get("evidence_id") or f"E{index:03d}"),
                    "phrase": phrase.strip(),
                    "source_document": item.get("source_document") or item.get("source"),
                })
    return phrases


def normalize_dataset(raw: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(raw, list):
        dataset_name, raw_questions, dataset_document = "knowledge_rag", raw, None
    elif isinstance(raw, dict):
        dataset_name = str(raw.get("dataset_name") or raw.get("dataset_id") or "knowledge_rag")
        raw_questions = raw.get("questions")
        dataset_document = raw.get("document")
        if isinstance(raw_questions, dict):
            # Also accept ground_truth.json/review_state-style question maps.
            mapped_questions = []
            for mapped_id, mapped_item in raw_questions.items():
                if isinstance(mapped_item, dict):
                    mapped_item = dict(mapped_item)
                    mapped_item.setdefault("question_id", str(mapped_id))
                mapped_questions.append(mapped_item)
            raw_questions = mapped_questions
        if raw_questions is None and (raw.get("question") or raw.get("q")):
            raw_questions = [raw]
    else:
        raise ValueError("Dataset JSON must be an object with 'questions' or a list of question objects.")

    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("No questions found. Expected a non-empty 'questions' array.")

    dataset_source = None
    if isinstance(dataset_document, str):
        dataset_source = dataset_document
    elif isinstance(dataset_document, dict):
        dataset_source = dataset_document.get("filename") or dataset_document.get("source")

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question entry {index} is not a JSON object.")
        q_text = item.get("question") or item.get("q")
        answer = item.get("reference_answer") or item.get("answer")
        phrases = _extract_phrases(item)
        if not isinstance(q_text, str) or not q_text.strip():
            raise ValueError(f"Question entry {index} has no question text.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Question entry {index} has no reference answer.")
        if not phrases:
            raise ValueError(f"Question entry {index} ({q_text[:60]!r}) has no evidence phrase.")
        qid = str(item.get("question_id") or item.get("id") or f"Q{index:03d}")
        if qid in seen_ids:
            raise ValueError(f"Duplicate question_id: {qid}")
        seen_ids.add(qid)
        source = item.get("source_document") or item.get("source") or item.get("document") or dataset_source
        for phrase in phrases:
            if not phrase.get("source_document"):
                phrase["source_document"] = source
        questions.append({
            "question_id": qid,
            "question": q_text.strip(),
            "reference_answer": answer.strip(),
            "source_document": source,
            "evidence_phrases": phrases,
        })
    return dataset_name, questions


def question_fingerprint(question: dict[str, Any]) -> str:
    """Stable fingerprint of every field that changes ground truth or answer labels."""
    canonical = {
        "question": str(question.get("question") or "").strip(),
        "reference_answer": str(question.get("reference_answer") or "").strip(),
        "source_document": normalize_source_label(question.get("source_document")),
        "evidence_phrases": [
            {
                "phrase": str(item.get("phrase") or "").strip(),
                "source_document": normalize_source_label(item.get("source_document")),
            }
            for item in question.get("evidence_phrases", [])
        ],
    }
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compact_text(value: Any, max_chars: int) -> str:
    """Bound LLM prompt size while preserving both the beginning and end of a chunk."""
    text_value = str(value or "")
    if max_chars <= 0 or len(text_value) <= max_chars:
        return text_value
    head = int(max_chars * 0.8)
    tail = max_chars - head
    return text_value[:head] + "\n\n[...content omitted to stay within benchmark prompt budget...]\n\n" + text_value[-tail:]


def print_side_by_side(left_title: str, left: str, right_title: str, right: str) -> None:
    width = shutil.get_terminal_size((120, 30)).columns
    col_width = max(30, min(74, (width - 7) // 2))
    left_lines = textwrap.wrap(left or "(not provided)", width=col_width) or [""]
    right_lines = textwrap.wrap(right or "(empty)", width=col_width) or [""]
    print("\n" + left_title[:col_width].ljust(col_width) + " | " + right_title[:col_width])
    print("-" * col_width + "-+-" + "-" * col_width)
    for i in range(max(len(left_lines), len(right_lines))):
        lhs = left_lines[i] if i < len(left_lines) else ""
        rhs = right_lines[i] if i < len(right_lines) else ""
        print(lhs.ljust(col_width) + " | " + rhs)


def build_source_catalog(collection: Any) -> dict[str, list[str]]:
    """Map normalized filenames to exact metadata values accepted by Chroma."""
    count = int(collection.count())
    if count <= 0:
        return {}
    data = collection.get(include=["metadatas"], limit=count)
    catalog: dict[str, set[str]] = defaultdict(set)
    for metadata in data.get("metadatas") or []:
        if not isinstance(metadata, dict):
            continue
        raw_source = metadata.get("source")
        if isinstance(raw_source, str) and raw_source.strip():
            catalog[normalize_source_label(raw_source)].add(raw_source)
    return {key: sorted(values) for key, values in catalog.items()}


def build_candidate_pool(
    collection: Any,
    question: dict[str, Any],
    top_n: int,
    source_catalog: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover evidence candidates, preferring exact normalized source matches.

    Chroma metadata filters still use the exact raw value stored during
    ingestion. A normalized lookup resolves newlines/case/spacing to those raw
    values. If a source label is not represented in the corpus (e.g. the old
    placeholder ``example_document.pdf``), an unfiltered query is run rather
    than silently returning zero candidates.
    """
    if int(collection.count()) <= 0:
        raise RuntimeError("The configured Chroma collection is empty.")
    source_catalog = source_catalog if source_catalog is not None else build_source_catalog(collection)
    best_by_id: dict[str, dict[str, Any]] = {}
    source_report: dict[str, Any] = {
        "expected_sources": [], "matched_sources": [], "used_global_fallback": False,
        "fallback_reasons": [],
    }
    queried: set[tuple[str, str]] = set()

    def query_phrase(phrase_record: dict[str, Any], raw_source: str | None) -> bool:
        """Run one evidence phrase query; return True when it yields content."""
        phrase = phrase_record["phrase"]
        dedupe_key = (phrase, raw_source or "<GLOBAL>")
        if dedupe_key in queried:
            return False
        queried.add(dedupe_key)
        args: dict[str, Any] = {
            "query_texts": [phrase],
            "n_results": min(max(1, int(top_n)), int(collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if raw_source is not None:
            args["where"] = {"source": raw_source}
        result = collection.query(**args)
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        found_content = False
        for i, chroma_id in enumerate(ids):
            content = docs[i] if i < len(docs) and docs[i] else ""
            if not content.strip():
                continue
            found_content = True
            metadata = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            distance = distances[i] if i < len(distances) and distances[i] is not None else None
            candidate = {
                "chroma_id": str(chroma_id),
                "content": str(content),
                "metadata": metadata,
                "distance": float(distance) if distance is not None else None,
                "matched_evidence_id": phrase_record["evidence_id"],
                "matched_evidence_phrase": phrase,
                "source_filter": raw_source,
            }
            previous = best_by_id.get(str(chroma_id))
            if previous is None or (
                distance is not None and (previous["distance"] is None or float(distance) < previous["distance"])
            ):
                best_by_id[str(chroma_id)] = candidate
        return found_content

    for phrase_record in question["evidence_phrases"]:
        source = phrase_record.get("source_document") or question.get("source_document")
        source_key = normalize_source_label(source)
        raw_matches = source_catalog.get(source_key, []) if source_key else []
        if source:
            source_report["expected_sources"].append(str(source))
        if raw_matches:
            source_report["matched_sources"].extend(
                raw for raw in raw_matches if raw not in source_report["matched_sources"]
            )
            found_in_source = False
            for raw_source in raw_matches:
                found_in_source = query_phrase(phrase_record, raw_source) or found_in_source
            # An existing filename with zero evidence hits is different from a
            # filename that does not exist. Still try the full collection so a
            # source/index mismatch cannot silently make the candidate pool empty.
            if not found_in_source:
                source_report["used_global_fallback"] = True
                source_report.setdefault("fallback_reasons", []).append(
                    f"No evidence candidates under expected source {source!r}; searched the full collection."
                )
                query_phrase(phrase_record, None)
        else:
            # No exact filename in corpus: do not kill the question. Search the
            # full local corpus and expose this mismatch in the output report.
            if source:
                source_report["used_global_fallback"] = True
                source_report.setdefault("fallback_reasons", []).append(
                    f"Expected source {source!r} is not present in indexed metadata; searched the full collection."
                )
            query_phrase(phrase_record, None)

    source_report["expected_sources"] = list(dict.fromkeys(source_report["expected_sources"]))
    source_report["matched_sources"] = list(dict.fromkeys(source_report["matched_sources"]))
    source_report["status"] = (
        "matched" if source_report["matched_sources"] and not source_report["used_global_fallback"]
        else "matched_source_global_fallback" if source_report["matched_sources"] and source_report["used_global_fallback"]
        else "unmatched_source_global_fallback" if source_report["used_global_fallback"]
        else "no_source_specified"
    )
    candidates = sorted(
        best_by_id.values(),
        key=lambda row: float("inf") if row["distance"] is None else row["distance"],
    )
    return candidates, source_report


def save_review_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "review_state.json", state)
    # Keep both judged decisions (including grade 0) and the positive subset.
    write_json(output_dir / "ground_truth.json", {
        "dataset_name": state["dataset_name"],
        "created_at_utc": state.get("created_at_utc"),
        "updated_at_utc": state["updated_at_utc"],
        "relevance_grade_definitions": {
            "0": "irrelevant or unsupported", "1": "related but insufficient",
            "2": "useful supporting evidence", "3": "directly supports the answer",
        },
        "binary_relevant_threshold": 2,
        "questions": list(state.get("annotations", {}).values()),
    })


class GroqJudge:
    """One Groq client/model for evidence judging, answer generation and judging."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Install groq and python-dotenv in the backend virtual environment.") from exc
        self.Groq = Groq
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            self.api_key = getpass.getpass("Groq API key: ").strip()
        if not self.api_key:
            raise RuntimeError("No Groq API key supplied.")
        self.client = self.Groq(api_key=self.api_key)
        self.calls = 0
        self.key_switches = 0
        self.skipped_requests = 0

    def validate(self) -> None:
        """Validate the key without spending a chat-completion token."""
        while True:
            try:
                self.client.models.list()
                return
            except Exception as exc:
                status = self._status(exc)
                if status in (401, 403, 429):
                    if self._switch_key_or_retry(f"API key validation failed: {type(exc).__name__}: {exc}", allow_retry=False):
                        continue
                raise

    @staticmethod
    def _status(exc: Exception) -> int | None:
        return getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)

    @staticmethod
    def _is_context_error(exc: Exception) -> bool:
        message = str(exc).casefold()
        return any(token in message for token in (
            "context_length_exceeded", "maximum context length", "too many tokens",
            "request too large", "token limit", "max context",
        ))

    def _switch_key_or_retry(self, reason: str, allow_retry: bool = True) -> bool:
        print(f"\nGroq request failed: {reason}")
        while True:
            choices = "[k] change key  [r] retry  [s] skip" if allow_retry else "[k] change key  [s] skip"
            choice = input(f"{choices}: ").strip().lower()
            if choice in {"k", "key"}:
                replacement = getpass.getpass("New Groq API key (hidden): ").strip()
                if not replacement:
                    print("No key entered.")
                    continue
                try:
                    client = self.Groq(api_key=replacement)
                    client.models.list()
                except Exception as exc:
                    print(f"New key validation failed: {type(exc).__name__}: {exc}")
                    continue
                self.api_key, self.client = replacement, client
                self.key_switches += 1
                print("New key accepted.")
                return True
            if allow_retry and choice in {"r", "retry"}:
                time.sleep(2)
                return True
            if choice in {"s", "skip"}:
                self.skipped_requests += 1
                return False
            print("Choose one of the displayed options.")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 900,
        json_mode: bool = False,
        label: str = "Groq request",
    ) -> str | None:
        request_format = {"type": "json_object"} if json_mode else None
        retry_count = 0
        max_tokens = min(max(32, int(max_tokens)), 1200)
        while True:
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                    "max_completion_tokens": max_tokens,
                    "reasoning_effort": "low",
                }
                if request_format:
                    kwargs["response_format"] = request_format
                response = self.client.chat.completions.create(**kwargs)
                self.calls += 1
                if not getattr(response, "choices", None):
                    raise RuntimeError("Groq returned no choices.")
                choice = response.choices[0]
                content = getattr(choice.message, "content", None) or ""
                if getattr(choice, "finish_reason", None) == "length":
                    # Retry at most once. Replaying a long prompt repeatedly can
                    # burn tokens, especially when the model's cap is the issue.
                    if retry_count == 0 and max_tokens < 1200:
                        max_tokens = min(max_tokens * 2, 1200)
                        retry_count += 1
                        print(f"{label}: output truncated; one retry with {max_tokens} completion tokens.")
                        continue
                    print(f"{label}: output remained truncated; request skipped to avoid repeated token use.")
                    self.skipped_requests += 1
                    return content.strip() or None
                return content.strip()
            except Exception as exc:
                status = self._status(exc)
                if self._is_context_error(exc) or status == 413:
                    # An API key switch cannot fix a prompt that exceeds the
                    # model context window. Do not repeat the same oversized call.
                    print(f"{label}: prompt exceeds model context; skipped without retry ({type(exc).__name__}).")
                    self.skipped_requests += 1
                    return None

                if status == 401:
                    if not self._switch_key_or_retry(f"{type(exc).__name__}: {exc}", allow_retry=False):
                        return None
                    retry_count = 0
                    continue

                if status == 429 or (status is not None and status >= 500):
                    if retry_count < 2:
                        delay = 3 if retry_count == 0 else 8
                        retry_count += 1
                        print(f"{label}: transient API error (HTTP {status}); one bounded backoff retry in {delay}s.")
                        time.sleep(delay)
                        continue
                    if not self._switch_key_or_retry(f"{type(exc).__name__}: {exc}", allow_retry=False):
                        return None
                    retry_count = 0
                    continue

                if retry_count < 1:
                    retry_count += 1
                    if not self._switch_key_or_retry(f"{type(exc).__name__}: {exc}"):
                        return None
                    continue
                if not self._switch_key_or_retry(f"{type(exc).__name__}: {exc}", allow_retry=False):
                    return None
                retry_count = 0


def parse_judge_json(raw: str) -> dict[str, Any]:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Judge did not return a JSON object: {cleaned[:250]}")
    payload = json.loads(cleaned[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Judge response JSON must be an object.")
    return payload


def llm_judge_chunk(judge: GroqJudge, question: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    metadata = candidate.get("metadata") or {}
    payload = {
        "question": question["question"],
        "reference_answer": question["reference_answer"],
        "evidence_phrases": [p["phrase"] for p in question["evidence_phrases"]],
        "expected_source": question.get("source_document"),
        "candidate_source": metadata.get("source"),
        "candidate_page_start": metadata.get("page_start"),
        "candidate_page_end": metadata.get("page_end"),
        "candidate_chunk": compact_text(candidate["content"], MAX_JUDGE_CHUNK_CHARS),
    }
    raw = judge.complete([
        {"role": "system", "content": (
            "You are a strict relevance annotator for a document retrieval benchmark. Judge only whether "
            "this chunk contains evidence that supports answering the question. Topic similarity alone is "
            "not enough. A chunk from another PDF may still support the answer, but mark source_match=false. "
            "Return JSON: {\"grade\":0|1|2|3,\"supports_answer\":boolean,\"source_match\":boolean,\"rationale\":\"brief\"}. "
            "3=direct/substantial evidence; 2=useful supporting evidence; 1=related but insufficient; "
            "0=irrelevant or unsupported. Do not use outside knowledge."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], max_tokens=400, json_mode=True, label="retrieved-chunk relevance judge")
    if raw is None:
        return None
    data = parse_judge_json(raw)
    grade = max(0, min(3, int(data.get("grade", 0))))
    return {
        "relevance_grade": grade,
        "supports_answer": bool(data.get("supports_answer", grade >= 2)),
        "source_match": bool(data.get("source_match", False)),
        "rationale": str(data.get("rationale", ""))[:500],
        "judge_model": judge.model,
        "judged_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def llm_judge_chunks_batch(
    judge: GroqJudge,
    question: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    """Judge several chunks in one completion to reduce request overhead.

    Each chunk is scored independently; the model is explicitly told not to
    compare candidates against one another. Missing IDs are left pending.
    """
    payload_candidates = []
    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        payload_candidates.append({
            "chroma_id": candidate["chroma_id"],
            "candidate_source": metadata.get("source"),
            "candidate_page_start": metadata.get("page_start"),
            "candidate_page_end": metadata.get("page_end"),
            "candidate_chunk": compact_text(candidate.get("content", ""), MAX_JUDGE_CHUNK_CHARS),
        })
    payload = {
        "question": question["question"],
        "reference_answer": question["reference_answer"],
        "evidence_phrases": [p["phrase"] for p in question["evidence_phrases"]],
        "expected_source": question.get("source_document"),
        "candidates": payload_candidates,
    }
    raw = judge.complete([
        {"role": "system", "content": (
            "You are a strict relevance annotator for a document retrieval benchmark. Evaluate EACH candidate "
            "independently against the question and reference answer. Do not compare candidates with each other, "
            "do not let one candidate influence another candidate's grade, and do not use outside knowledge. "
            "Topic similarity alone is not enough. A chunk from another PDF may support the answer, but mark "
            "source_match=false. Return one JSON object exactly shaped as "
            "{\"judgements\":[{\"chroma_id\":string,\"grade\":0|1|2|3,\"supports_answer\":boolean,\"source_match\":boolean,\"rationale\":string}]}. "
            "3=direct/substantial evidence; 2=useful supporting evidence; 1=related but insufficient; "
            "0=irrelevant or unsupported. Include every supplied chroma_id exactly once."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], max_tokens=min(1200, max(350, len(candidates) * 180)), json_mode=True, label="batched retrieved-chunk relevance judge")
    if raw is None:
        return None
    data = parse_judge_json(raw)
    rows = data.get("judgements")
    if not isinstance(rows, list):
        raise ValueError("Batched relevance response has no 'judgements' array.")
    output: dict[str, dict[str, Any]] = {}
    allowed_ids = {str(candidate["chroma_id"]) for candidate in candidates}
    for row in rows:
        if not isinstance(row, dict):
            continue
        chroma_id = str(row.get("chroma_id") or "")
        if chroma_id not in allowed_ids or chroma_id in output:
            continue
        grade = max(0, min(3, int(row.get("grade", 0))))
        output[chroma_id] = {
            "relevance_grade": grade,
            "supports_answer": bool(row.get("supports_answer", grade >= 2)),
            "source_match": bool(row.get("source_match", False)),
            "rationale": str(row.get("rationale", ""))[:300],
            "judge_model": judge.model,
            "judged_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    return output


def llm_judge_answer(
    judge: GroqJudge,
    question: dict[str, Any],
    generated: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    context = compact_text("\n\n".join(
        f"[Source: {chunk.get('source')}; pages: {chunk.get('page_start')}-{chunk.get('page_end')}]\n"
        f"{chunk.get('content', '')}"
        for chunk in retrieved_chunks[:5]
    ), MAX_ANSWER_CONTEXT_CHARS)
    payload = {
        "question": question["question"],
        "reference_answer": question["reference_answer"],
        "generated_answer": generated,
        "retrieved_context": context,
    }
    raw = judge.complete([
        {"role": "system", "content": (
            "You are a strict evaluator for a retrieval-augmented QA benchmark. Compare answer correctness, "
            "completeness and support in the supplied retrieved context. Accept valid paraphrases. Penalize "
            "unsupported claims, contradictions and missing key facts. Do not use outside knowledge. Return "
            "JSON: {\"grade\":0|1|2|3,\"label\":\"incorrect|partially_correct|mostly_correct|correct\", "
            "\"context_supported\":boolean,\"rationale\":\"brief\"}. 3=correct and adequately supported; "
            "2=mostly correct with minor omissions; 1=partially correct; 0=incorrect/contradicted/unsupported."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], max_tokens=450, json_mode=True, label="answer correctness judge")
    if raw is None:
        return None
    data = parse_judge_json(raw)
    grade = max(0, min(3, int(data.get("grade", 0))))
    default_label = {3: "correct", 2: "mostly_correct", 1: "partially_correct", 0: "incorrect"}[grade]
    return {
        "grade": grade,
        "label": str(data.get("label") or default_label),
        "context_supported": bool(data.get("context_supported", False)),
        "rationale": str(data.get("rationale", ""))[:500],
        "judge_model": judge.model,
        "judged_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def review_candidates(
    questions: list[dict[str, Any]],
    collection: Any,
    output_dir: Path,
    state: dict[str, Any],
    top_n: int,
    preview_chars: int,
    judge: GroqJudge,
    retry_failed: bool = False,
    judge_batch_size: int = 5,
) -> None:
    annotations = state.setdefault("annotations", {})
    source_catalog = None
    collection_count = int(collection.count())
    for q_index, question in enumerate(questions, start=1):
        qid = question["question_id"]
        annotation = annotations.setdefault(qid, {
            "question_id": qid,
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "source_document": question.get("source_document"),
            "evidence_phrases": [p["phrase"] for p in question["evidence_phrases"]],
            "relevant_chunks": {},
            "decisions": {},
        })
        current_fingerprint = question_fingerprint(question)
        # Completed candidate pools from this exact benchmark version/settings
        # are reused; this avoids re-embedding every evidence phrase on resume.
        if (
            annotation.get("question_fingerprint") == current_fingerprint
            and annotation.get("candidate_discovery_version") == CANDIDATE_DISCOVERY_VERSION
            and int(annotation.get("candidate_top_n", -1)) == int(top_n)
            and int(annotation.get("candidate_collection_count", -1)) == collection_count
            and bool(annotation.get("review_complete"))
        ):
            print(f"\nGROUND-TRUTH REVIEW {q_index}/{len(questions)} | {qid}: cached completed review reused")
            continue

        # Refresh dataset-controlled fields on resume, preserving valid grades.
        annotation.update({
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "source_document": question.get("source_document"),
            "evidence_phrases": [p["phrase"] for p in question["evidence_phrases"]],
            "question_fingerprint": current_fingerprint,
        })
        if source_catalog is None:
            source_catalog = build_source_catalog(collection)
        candidates, source_report = build_candidate_pool(collection, question, top_n, source_catalog)
        annotation["source_resolution"] = source_report
        annotation["candidate_count"] = len(candidates)
        annotation["candidate_ids"] = [candidate["chroma_id"] for candidate in candidates]
        annotation["candidate_discovery_version"] = CANDIDATE_DISCOVERY_VERSION
        annotation["candidate_top_n"] = int(top_n)
        annotation["candidate_collection_count"] = collection_count
        annotation["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
        print("\n" + "=" * 96)
        print(f"GROUND-TRUTH REVIEW {q_index}/{len(questions)} | {qid}")
        print(f"Question: {question['question']}")
        print(f"Expected source: {question.get('source_document') or '(not set)'}")
        print(f"Source resolution: {source_report['status']} | candidates: {len(candidates)}")
        if source_report["used_global_fallback"]:
            print("WARNING: source filename did not match indexed metadata; used global candidate search.")

        if not candidates:
            annotation["review_status"] = "no_candidates_found"
            annotation["review_complete"] = True
            save_review_state(output_dir, state)
            continue

        decisions = annotation.setdefault("decisions", {})
        relevant_chunks = annotation.setdefault("relevant_chunks", {})
        failed_candidates = annotation.setdefault("failed_candidates", {})
        skipped = 0
        pending_candidates = []
        for candidate in candidates:
            chunk_id = candidate["chroma_id"]
            if chunk_id in decisions:
                failed_candidates.pop(chunk_id, None)
                continue
            if chunk_id in failed_candidates and not retry_failed:
                skipped += 1
                print(f"Candidate {chunk_id} previously failed/skipped; not retrying without --retry-failed.")
                continue
            if retry_failed:
                failed_candidates.pop(chunk_id, None)
            pending_candidates.append(candidate)

        batch_size = max(1, int(judge_batch_size))
        total_batches = (len(pending_candidates) + batch_size - 1) // batch_size
        for batch_index in range(total_batches):
            batch = pending_candidates[batch_index * batch_size:(batch_index + 1) * batch_size]
            print(f"\nJudging candidate batch {batch_index + 1}/{total_batches} ({len(batch)} chunks)")
            for candidate_index, candidate in enumerate(batch, start=1):
                metadata = candidate.get("metadata") or {}
                preview = candidate["content"] if preview_chars <= 0 else candidate["content"][:preview_chars]
                print(f"  [{candidate_index}/{len(batch)}] {candidate['chroma_id']} | "
                      f"{metadata.get('source', '?')} | pages "
                      f"{metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}")
                print_side_by_side(
                    f"EVIDENCE PHRASE ({candidate['matched_evidence_id']})",
                    candidate["matched_evidence_phrase"],
                    "INDEXED CHUNK",
                    preview + ("\n...[preview truncated]" if len(preview) < len(candidate["content"]) else ""),
                )
            try:
                judged_batch = llm_judge_chunks_batch(judge, question, batch)
            except Exception as exc:
                judged_batch = None
                print(f"Batched relevance response could not be parsed: {type(exc).__name__}: {exc}")

            for candidate in batch:
                chunk_id = candidate["chroma_id"]
                judged = judged_batch.get(chunk_id) if judged_batch is not None else None
                if judged is None:
                    skipped += 1
                    failed_candidates[chunk_id] = {
                        "status": "skipped_or_failed",
                        "last_attempt_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    continue

                failed_candidates.pop(chunk_id, None)
                metadata = candidate.get("metadata") or {}
                grade = judged["relevance_grade"]
                decisions[chunk_id] = {
                    "accepted": grade >= 2,
                    "relevance_grade": grade,
                    "embedding_distance": candidate.get("distance"),
                    "matched_evidence_id": candidate["matched_evidence_id"],
                    "matched_evidence_phrase": candidate["matched_evidence_phrase"],
                    "source": metadata.get("source"),
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                    "heading_path": metadata.get("heading_path"),
                    "content": candidate["content"],
                    "llm_judgement": judged,
                }
                if grade > 0:
                    relevant_chunks[chunk_id] = {
                        "relevance": grade,
                        "source": metadata.get("source"),
                        "page_start": metadata.get("page_start"),
                        "page_end": metadata.get("page_end"),
                        "heading_path": metadata.get("heading_path"),
                        "evidence_phrase": candidate["matched_evidence_phrase"],
                        "judge_rationale": judged["rationale"],
                    }
                print(f"  {chunk_id}: relevance {grade}/3 | source match: {judged['source_match']} | {judged['rationale']}")
            # Persist once per batch rather than rewriting the full JSON state
            # after every single chunk judgement. This keeps checkpoint I/O bounded.
            save_review_state(output_dir, state)

        candidate_ids = [candidate["chroma_id"] for candidate in candidates]
        pending_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in decisions]
        pending = len(pending_ids)
        annotation["candidate_ids"] = candidate_ids
        annotation["reviewed_candidate_count"] = len(candidate_ids) - pending
        annotation["review_complete"] = pending == 0 and skipped == 0
        annotation["review_status"] = "completed" if annotation["review_complete"] else "partial_review_pending"
        annotation["review_pending_count"] = pending
        annotation["positive_ground_truth_count"] = sum(
            int(v.get("relevance", 0)) >= 2 for v in relevant_chunks.values()
        )
        save_review_state(output_dir, state)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(chunk_id in relevant for chunk_id in retrieved[:k]) / float(k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(chunk_id in relevant for chunk_id in retrieved[:k]) / float(len(relevant)) if relevant else math.nan


def reciprocal_rank(retrieved: list[str], relevant: set[str], max_k: int) -> float:
    for rank, chunk_id in enumerate(retrieved[:max_k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], grades: dict[str, int], k: int) -> float:
    def gain(grade: int) -> float:
        return (2.0 ** grade) - 1.0 if grade > 0 else 0.0
    dcg = sum(gain(grades.get(chunk_id, 0)) / math.log2(rank + 1)
              for rank, chunk_id in enumerate(retrieved[:k], start=1))
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else math.nan


def mean_defined(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
    return sum(usable) / len(usable) if usable else None


@lru_cache(maxsize=1)
def _load_sentence_model(model_name: str, device: str = "cpu"):
    """Load once per process; CPU is the safe default for this project laptop."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed in the backend environment.") from exc
    return SentenceTransformer(model_name, device=device)


def answer_embedding_similarity(
    reference_answer: str,
    generated_answer: str,
    model_name: str,
    device: str = "cpu",
) -> float:
    import numpy as np
    model = _load_sentence_model(model_name, device)
    vectors = model.encode(
        [reference_answer, generated_answer],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=2,
        show_progress_bar=False,
    )
    return float(np.dot(vectors[0], vectors[1]))


def generate_answer(
    judge: GroqJudge,
    question: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
) -> str | None:
    context = compact_text("\n\n".join(
        f"[Source: {chunk.get('source')}; pages: {chunk.get('page_start')}-{chunk.get('page_end')}]\n"
        f"{chunk.get('content', '')}"
        for chunk in retrieved_chunks[:5]
    ), MAX_ANSWER_CONTEXT_CHARS)
    prompt = (
        "Answer using only the retrieved context below. If the context is missing or insufficient, say so. "
        "Do not add unsupported facts. Keep the answer concise.\n\n"
        f"QUESTION:\n{question['question']}\n\nRETRIEVED CONTEXT:\n{context or '(No chunks retrieved.)'}\n\nANSWER:"
    )
    return judge.complete([
        {"role": "system", "content": "You are a factual QA component in a retrieval-augmented benchmark."},
        {"role": "user", "content": prompt},
    ], max_tokens=500, label="answer generation")


def review_generated_answers(
    questions: list[dict[str, Any]],
    retrieved_records: list[dict[str, Any]],
    state: dict[str, Any],
    output_dir: Path,
    model_name: str,
    judge: GroqJudge,
    embedding_device: str,
    retry_failed: bool = False,
) -> None:
    reviews = state.setdefault("answer_reviews", {})
    records_by_id = {str(item.get("question_id")): item for item in retrieved_records if item.get("question_id")}
    for index, question in enumerate(questions, start=1):
        qid = question["question_id"]
        record = records_by_id.get(qid)
        if record is None:
            print(f"\nANSWER BENCHMARK {index}/{len(questions)}: {qid} has no retrieval record; skipped.")
            continue
        existing = reviews.get(qid) or {}
        if not retry_failed and existing.get("generation_status") == "skipped_or_failed":
            record["answer_status"] = "generation_previously_skipped"
            continue
        if not retry_failed and existing.get("answer_judgement_status") == "skipped_or_failed":
            record["generated_answer"] = existing.get("generated_answer", "")
            record["answer_embedding_cosine_similarity"] = existing.get("embedding_cosine_similarity")
            record["answer_status"] = "answer_judgement_previously_skipped"
            continue
        if existing.get("llm_judgement"):
            record["generated_answer"] = existing.get("generated_answer", "")
            record["answer_embedding_cosine_similarity"] = existing.get("embedding_cosine_similarity")
            record["answer_llm_judgement"] = existing.get("llm_judgement")
            record["answer_status"] = "judged"
            continue

        print(f"\nANSWER GENERATION + JUDGING {index}/{len(questions)}: {qid}")
        generated = existing.get("generated_answer")
        similarity = existing.get("embedding_cosine_similarity")
        if not generated:
            generated = generate_answer(judge, question, record.get("retrieved_chunks", []))
            if generated is None:
                record["answer_status"] = "generation_skipped"
                reviews[qid] = {
                    "reference_answer": question["reference_answer"],
                    "generated_answer": None,
                    "generation_status": "skipped_or_failed",
                    "llm_judgement": None,
                }
                save_review_state(output_dir, state)
                continue

        # A prior run may have saved the answer but failed while loading the
        # embedding model. Retry the local cosine calculation without regenerating
        # the answer or spending another Groq completion.
        if similarity is None:
            try:
                similarity = answer_embedding_similarity(
                    question["reference_answer"], generated, model_name, embedding_device
                )
            except Exception as exc:
                print(f"Embedding similarity unavailable: {type(exc).__name__}: {exc}")
                similarity = None

        reviews[qid] = {
            "reference_answer": question["reference_answer"],
            "generated_answer": generated,
            "embedding_model": model_name,
            "embedding_device": embedding_device,
            "embedding_cosine_similarity": similarity,
            "llm_judgement": None,
        }
        save_review_state(output_dir, state)

        try:
            judgement = llm_judge_answer(judge, question, generated, record.get("retrieved_chunks", []))
        except Exception as exc:
            judgement = None
            print(f"Answer judge could not be parsed: {type(exc).__name__}: {exc}")
        if judgement is None:
            record["answer_status"] = "answer_judgement_skipped"
            reviews[qid]["answer_judgement_status"] = "skipped_or_failed"
            reviews[qid]["last_attempt_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_review_state(output_dir, state)
            continue

        record["generated_answer"] = generated
        record["answer_embedding_cosine_similarity"] = similarity
        record["answer_llm_judgement"] = judgement
        record["answer_status"] = "judged"
        print(f"\n{qid}: BGE cosine={similarity if similarity is not None else 'N/A'} | "
              f"Groq={judgement['grade']}/3 ({judgement['label']}) | context supported={judgement['context_supported']}")
        reviews[qid] = {
            "reference_answer": question["reference_answer"],
            "generated_answer": generated,
            "embedding_model": model_name,
            "embedding_device": embedding_device,
            "embedding_cosine_similarity": similarity,
            "llm_judgement": judgement,
            "answer_judgement_status": "completed",
            "judged_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_review_state(output_dir, state)



def write_answer_only_results(
    output_dir: Path,
    dataset_name: str,
    questions: list[dict[str, Any]],
    retrieval_records: list[dict[str, Any]],
    state: dict[str, Any],
    judge: GroqJudge,
    embedding_model: str,
    embedding_device: str,
) -> dict[str, Any]:
    """Write answer-only artifacts without recomputing retrieval metrics."""
    answer_reviews = state.get("answer_reviews") or {}
    judged = [item for item in answer_reviews.values() if item.get("llm_judgement")]
    grades = [int(item["llm_judgement"].get("grade", 0)) for item in judged]
    labels = [str(item["llm_judgement"].get("label", "unknown")) for item in judged]
    similarities = [float(item["embedding_cosine_similarity"]) for item in judged
                    if item.get("embedding_cosine_similarity") is not None]
    metrics = {
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_mode": "answer_only",
        "dataset_name": dataset_name,
        "question_count": len(questions),
        "judged_question_count": len(judged),
        "generation_completed_count": sum(bool(v.get("generated_answer")) for v in answer_reviews.values()),
        "generation_failed_count": sum(v.get("generation_status") == "skipped_or_failed" for v in answer_reviews.values()),
        "answer_judgement_failed_count": sum(v.get("answer_judgement_status") == "skipped_or_failed" for v in answer_reviews.values()),
        "judge_model": judge.model,
        "embedding_model": embedding_model,
        "embedding_device": embedding_device,
        "embedding_similarity_mean": mean_defined(similarities),
        "llm_judgement_counts": dict(Counter(labels)),
        "llm_exact_correct_rate": labels.count("correct") / len(judged) if judged else None,
        "llm_weighted_answer_score": mean_defined([grade / 3.0 for grade in grades]),
        "score_mapping": {"grade_3": 1.0, "grade_2": 0.6667, "grade_1": 0.3333, "grade_0": 0.0},
        "note": "Answer-only evaluation using the saved retrieval_results.json from the selected run; retrieval and relevance review were not rerun.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "groq_usage_this_answer_only_run": {
            "successful_completion_calls": judge.calls,
            "api_key_switches": judge.key_switches,
            "skipped_requests": judge.skipped_requests,
        },
    }
    rows = []
    for q in questions:
        review = answer_reviews.get(q["question_id"], {})
        judgement = review.get("llm_judgement") or {}
        rows.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "generated_answer": review.get("generated_answer", ""),
            "embedding_cosine_similarity": review.get("embedding_cosine_similarity"),
            "llm_grade": judgement.get("grade"),
            "llm_label": judgement.get("label", "not_judged"),
            "context_supported": judgement.get("context_supported"),
            "judge_rationale": judgement.get("rationale", ""),
            "normalized_score": int(judgement["grade"]) / 3.0 if judgement.get("grade") is not None else None,
        })
    write_json(output_dir / "answer_metrics.json", metrics)
    write_json(output_dir / "answer_review.json", {"answer_generation": metrics, "reviews": answer_reviews})
    write_json(output_dir / "answer_per_question.json", rows)
    write_csv(output_dir / "answer_per_question.csv", rows)
    # Keep the main metrics file in sync without replacing existing retrieval results.
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        combined = load_json(metrics_path)
        combined["answer_benchmark_enabled"] = True
        combined["answer_generation"] = metrics
        write_json(metrics_path, combined)
    state["answer_benchmark_enabled"] = True
    save_review_state(output_dir, state)
    return metrics


def retrieve_production_rankings(
    questions: list[dict[str, Any]],
    max_k: int,
    backend_dir: Path,
) -> list[dict[str, Any]]:
    """Call the exact production hybrid LocalRetriever used by the RAG package."""
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from services.ai.rag.local_retriever import LocalRetriever
    retriever = LocalRetriever()
    records: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        print(f"\nPRODUCTION RETRIEVAL {index}/{len(questions)} | {question['question_id']}")
        ranked = retriever.retrieve([question["question"]], top_k=max_k) or []
        chunks: list[dict[str, Any]] = []
        for rank, item in enumerate(ranked[:max_k], start=1):
            chunks.append({
                "rank": rank,
                "chroma_id": str(item.get("chroma_id") or ""),
                "retrieval_score": item.get("retrieval_score"),
                "bm25_score": item.get("bm25_score"),
                "distance": item.get("distance"),
                "retrieval_methods": item.get("retrieval_methods", []),
                "exact_id_match": item.get("exact_id_match"),
                "source": item.get("source"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "heading_path": item.get("heading_path"),
                "content_type": item.get("content_type"),
                "content": item.get("content"),
            })
        records.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "retrieved_count": len(chunks),
            "retrieved_chunks": chunks,
        })
        print(f"Retrieved {len(chunks)} chunks")
    return records


def evaluate_retrieval(
    questions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    state: dict[str, Any],
    ks: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return rows for every question and aggregate only fully supported positives."""
    per_question: list[dict[str, Any]] = []
    max_k = max(ks)
    records_by_id = {str(item.get("question_id")): item for item in records if item.get("question_id")}
    for question in questions:
        record = records_by_id.get(question["question_id"])
        record_missing = record is None
        if record is None:
            record = {"retrieved_chunks": []}
        annotation = (state.get("annotations") or {}).get(question["question_id"], {})
        decisions = annotation.get("decisions") or {}
        # Decisions contain grades 0-3; relevant_chunks contain positive grades
        # from older benchmark files. The decisions map is preferred when present.
        current_candidate_ids = annotation.get("candidate_ids") or []
        if decisions:
            # Only judge labels from this run's current candidate pool. Old
            # candidate IDs may remain in a resumed v1 state after discovery is
            # corrected; they must not inflate recall's ground-truth denominator.
            ids_to_use = current_candidate_ids if current_candidate_ids else list(decisions)
            grades = {
                str(cid): int(decisions[cid].get("relevance_grade", 0))
                for cid in ids_to_use if cid in decisions
            }
        else:
            grades = {
                str(cid): int(details.get("relevance", 0))
                for cid, details in (annotation.get("relevant_chunks") or {}).items()
            }
        relevant = {cid for cid, grade in grades.items() if grade >= 2}
        ranked_ids = [chunk.get("chroma_id", "") for chunk in record.get("retrieved_chunks", []) if chunk.get("chroma_id")]
        answer_review = (state.get("answer_reviews") or {}).get(question["question_id"], {})
        answer_judgement = answer_review.get("llm_judgement") or {}
        reviewed_count = len(decisions)
        candidate_ids = annotation.get("candidate_ids") or []
        expected_count = int(annotation.get("candidate_count") or len(candidate_ids) or reviewed_count)
        # Count only this question's current evidence candidates. A stale decision
        # from a previous candidate pool must not make an incomplete review look complete.
        current_candidates_reviewed = all(candidate_id in decisions for candidate_id in candidate_ids)
        review_complete = bool(annotation.get("review_complete", False)) and current_candidates_reviewed
        has_graded_relevance = any(grade >= 1 for grade in grades.values())
        if record_missing:
            status = "retrieval_record_missing"
        elif not review_complete:
            status = "incomplete_ground_truth_review"
        elif expected_count == 0:
            status = "no_candidates_found"
        elif relevant:
            status = "evaluated_with_positive_ground_truth"
        elif has_graded_relevance:
            status = "completed_weak_only_no_binary_relevant"
        else:
            status = "no_grade_1_or_higher_ground_truth"

        binary_evaluable = status == "evaluated_with_positive_ground_truth"
        graded_evaluable = review_complete and has_graded_relevance
        row: dict[str, Any] = {
            "question_id": question["question_id"],
            "question": question["question"],
            "evaluation_status": status,
            "retrieval_record_missing": record_missing,
            "ground_truth_decisions": reviewed_count,
            "ground_truth_candidate_count": expected_count,
            "ground_truth_review_complete": review_complete,
            "ground_truth_source_status": (annotation.get("source_resolution") or {}).get("status"),
            "ground_truth_source_fallback": (annotation.get("source_resolution") or {}).get("used_global_fallback"),
            "ground_truth_source_fallback_reasons": " | ".join((annotation.get("source_resolution") or {}).get("fallback_reasons", [])),
            "ground_truth_relevant_count": len(relevant),
            "ground_truth_weak_count": sum(grade == 1 for grade in grades.values()),
            "retrieved_count": len(ranked_ids),
            "first_relevant_rank": next((rank for rank, cid in enumerate(ranked_ids, start=1) if cid in relevant), None),
            "answer_embedding_similarity": answer_review.get("embedding_cosine_similarity"),
            "answer_llm_grade": answer_judgement.get("grade"),
            "answer_llm_label": answer_judgement.get("label"),
        }
        for k in ks:
            row[f"precision@{k}"] = precision_at_k(ranked_ids, relevant, k) if binary_evaluable else None
            row[f"recall@{k}"] = recall_at_k(ranked_ids, relevant, k) if binary_evaluable else None
            row[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, grades, k) if graded_evaluable else None
        row[f"mrr@{max_k}"] = reciprocal_rank(ranked_ids, relevant, max_k) if binary_evaluable else None
        per_question.append(row)

    evaluable = [r for r in per_question if r["evaluation_status"] == "evaluated_with_positive_ground_truth"]
    graded_evaluable = [r for r in per_question if r["evaluation_status"] in {
        "evaluated_with_positive_ground_truth", "completed_weak_only_no_binary_relevant"
    }]
    statuses = Counter(r["evaluation_status"] for r in per_question)
    macro: dict[str, float | None] = {}
    for k in ks:
        for metric in ("precision", "recall"):
            macro[f"{metric}@{k}"] = mean_defined([r[f"{metric}@{k}"] for r in evaluable])
        macro[f"ndcg@{k}"] = mean_defined([r[f"ndcg@{k}"] for r in graded_evaluable])
    macro[f"mrr@{max_k}"] = mean_defined([r[f"mrr@{max_k}"] for r in evaluable])
    summary = {
        "question_count": len(questions),
        "reported_question_count": len(per_question),
        "retrieval_evaluable_question_count": len(evaluable),
        "not_retrieval_evaluable_question_count": len(per_question) - len(evaluable),
        "retrieval_record_missing_count": sum(1 for row in per_question if row.get("retrieval_record_missing")),
        "graded_ndcg_evaluable_question_count": len(graded_evaluable),
        "evaluation_status_counts": dict(statuses),
        "macro_average_on_evaluable_questions": macro,
        "aggregation_note": (
            "Every dataset question appears in per_question_metrics. Precision, recall and MRR macro averages "
            "use questions with a completed review and at least one grade-2/3 chunk. Graded nDCG also includes "
            "completed questions with grade-1-only evidence. Questions with no candidates, only grade-0 labels, "
            "or incomplete judgments remain visible with an explicit status instead of being silently dropped."
        ),
    }
    return per_question, summary


def retrieval_cache_signature(questions: list[dict[str, Any]], max_k: int, backend_dir: Path, collection_count: int) -> str:
    retriever_path = backend_dir / "services" / "ai" / "rag" / "local_retriever.py"
    retriever_hash = hashlib.sha256(retriever_path.read_bytes()).hexdigest() if retriever_path.is_file() else "missing"
    db_path = Path(os.getenv("CHROMA_DB_DIR", str(backend_dir / "vector_db"))).expanduser()
    if not db_path.is_absolute():
        db_path = (backend_dir / db_path).resolve()
    db_stamp: list[tuple[str, int, int]] = []
    if db_path.is_dir():
        # Persisted Chroma files let the cache notice most re-indexes, including
        # replacements that leave the collection count unchanged.
        for file_path in db_path.rglob("*"):
            try:
                if file_path.is_file():
                    stat = file_path.stat()
                    db_stamp.append((str(file_path.relative_to(db_path)), stat.st_size, stat.st_mtime_ns))
            except OSError:
                continue
    payload = {
        "backend_dir": str(backend_dir.resolve()),
        "question_fingerprints": [question_fingerprint(q) for q in questions],
        "max_k": int(max_k),
        "collection_count": int(collection_count),
        "chroma_file_stamp": sorted(db_stamp),
        "retriever_sha256": retriever_hash,
        "embedding_model": os.getenv("KNOWLEDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "embedding_device": os.getenv("KNOWLEDGE_EMBEDDING_DEVICE", "cpu"),
        "collection": os.getenv("KNOWLEDGE_COLLECTION", "knowledge"),
        "chroma_db_dir": os.getenv("CHROMA_DB_DIR", ""),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_or_retrieve_rankings(
    questions: list[dict[str, Any]], max_k: int, backend_dir: Path, output_dir: Path, collection_count: int
) -> list[dict[str, Any]]:
    """Reuse rankings on resume so interruptions during answer judging do not rerun embeddings."""
    cache_path = output_dir / "retrieval_cache.json"
    signature = retrieval_cache_signature(questions, max_k, backend_dir, collection_count)
    if cache_path.is_file():
        try:
            cache = load_json(cache_path)
            records = cache.get("questions", [])
            expected_ids = [q["question_id"] for q in questions]
            cached_ids = [r.get("question_id") for r in records]
            if cache.get("signature") == signature and cached_ids == expected_ids:
                print(f"Reusing cached production retrieval rankings for {len(records)} questions.")
                return records
        except Exception as exc:
            print(f"Ignoring unreadable retrieval cache: {type(exc).__name__}: {exc}")
    records = retrieve_production_rankings(questions, max_k, backend_dir)
    write_json(cache_path, {"signature": signature, "questions": records})
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_results(
    output_dir: Path,
    dataset_name: str,
    questions: list[dict[str, Any]],
    retrieval_records: list[dict[str, Any]],
    per_question: list[dict[str, Any]],
    retrieval_summary: dict[str, Any],
    state: dict[str, Any],
    judge: GroqJudge,
    ks: list[int],
    embedding_model: str,
    embedding_device: str,
) -> None:
    answer_reviews = state.get("answer_reviews") or {}
    judged_reviews = [item for item in answer_reviews.values() if item.get("llm_judgement")]
    answer_grades = [int(item["llm_judgement"].get("grade", 0)) for item in judged_reviews]
    answer_labels = [str(item["llm_judgement"].get("label", "unknown")) for item in judged_reviews]
    answer_similarity = [float(item["embedding_cosine_similarity"]) for item in judged_reviews if item.get("embedding_cosine_similarity") is not None]
    answer_metrics = {
        "dataset_name": dataset_name,
        "question_count": len(questions),
        "judged_question_count": len(judged_reviews),
        "judge_model": judge.model,
        "embedding_model": embedding_model,
        "embedding_device": embedding_device,
        "embedding_similarity_mean": mean_defined(answer_similarity),
        "llm_judgement_counts": dict(Counter(answer_labels)),
        "llm_exact_correct_rate": answer_labels.count("correct") / len(judged_reviews) if judged_reviews else None,
        "llm_weighted_answer_score": mean_defined([grade / 3.0 for grade in answer_grades]),
        "score_mapping": {"grade_3": 1.0, "grade_2": 0.6667, "grade_1": 0.3333, "grade_0": 0.0},
        "note": "LLM-judged answer-generation results are separate from retrieval metrics; audit a sample manually.",
    }
    metrics = {
        "benchmark_version": BENCHMARK_VERSION,
        "answer_benchmark_enabled": bool(state.get("answer_benchmark_enabled", False)),
        "dataset_name": dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "retriever": "services.ai.rag.local_retriever.LocalRetriever",
        "retrieval_scope": "Production local hybrid retriever: Chroma vector search + BM25 + RRF, one original question per retrieval call, with production query ranking. No web or RAG context selector added.",
        "ground_truth": {
            "candidate_discovery": "Chroma vector query using evidence phrases; exact metadata source values are resolved through normalized source labels; global fallback used when the expected source is absent.",
            "judge_model": judge.model,
            "relevance_grades": {"0": "irrelevant", "1": "related but insufficient", "2": "useful supporting evidence", "3": "directly supports the answer"},
            "binary_relevant_threshold": 2,
        },
        "question_count": len(questions),
        "reported_question_count": len(per_question),
        "retrieval_evaluable_question_count": retrieval_summary["retrieval_evaluable_question_count"],
        "graded_ndcg_evaluable_question_count": retrieval_summary["graded_ndcg_evaluable_question_count"],
        "not_retrieval_evaluable_question_count": retrieval_summary["not_retrieval_evaluable_question_count"],
        "evaluation_status_counts": retrieval_summary["evaluation_status_counts"],
        "k_values": ks,
        "macro_average": retrieval_summary["macro_average_on_evaluable_questions"],
        "metric_definitions": {
            "precision_at_k": "Relevant retrieved chunks in top K divided by K.",
            "recall_at_k": "Relevant retrieved chunks in top K divided by all grade-2/3 judged relevant chunks in that question's evidence candidate pool.",
            "mrr": f"Reciprocal rank of first grade-2/3 chunk through rank {max(ks)}.",
            "ndcg": "Graded nDCG with gain 2^grade - 1; grades 1, 2 and 3 contribute.",
            "aggregation": retrieval_summary["aggregation_note"],
        },
        "embedding_model": os.getenv("KNOWLEDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "collection": os.getenv("KNOWLEDGE_COLLECTION", "knowledge"),
        "rrf_k": int(os.getenv("KNOWLEDGE_RRF_K", "60")),
        "groq_usage": {
            "model": judge.model,
            "successful_completion_calls": judge.calls,
            "api_key_switches": judge.key_switches,
            "skipped_requests": judge.skipped_requests,
            "api_key_values_saved": False,
        },
        "answer_generation": answer_metrics,
    }
    write_json(output_dir / "retrieval_results.json", {
        "retriever": metrics["retriever"],
        "questions": retrieval_records,
    })
    write_json(output_dir / "per_question_metrics.json", per_question)
    write_csv(output_dir / "per_question_metrics.csv", per_question)
    write_json(output_dir / "answer_metrics.json", answer_metrics)
    write_json(output_dir / "answer_review.json", {"answer_generation": answer_metrics, "reviews": answer_reviews})
    answer_rows = []
    for question in questions:
        review = answer_reviews.get(question["question_id"], {})
        judgement = review.get("llm_judgement") or {}
        answer_rows.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "generated_answer": review.get("generated_answer", ""),
            "embedding_cosine_similarity": review.get("embedding_cosine_similarity"),
            "llm_grade": judgement.get("grade"),
            "llm_label": judgement.get("label", "not_judged"),
            "context_supported": judgement.get("context_supported"),
            "judge_rationale": judgement.get("rationale", ""),
            "normalized_score": int(judgement["grade"]) / 3.0 if judgement.get("grade") is not None else None,
        })
    write_json(output_dir / "answer_per_question.json", answer_rows)
    write_csv(output_dir / "answer_per_question.csv", answer_rows)
    write_json(output_dir / "metrics.json", metrics)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the project's production Knowledge RAG retriever.")
    parser.add_argument("--dataset", help="JSON file containing question, reference_answer, evidence_phrase(s), source_document.")
    parser.add_argument("--backend-dir", help="Backend directory containing services/ai/rag.")
    parser.add_argument("--db-dir", help="Chroma directory; overrides CHROMA_DB_DIR.")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--top-n", type=int, default=15, help="Candidate chunks per evidence phrase during ground-truth review.")
    parser.add_argument("--judge-batch-size", type=int, default=5, help="Candidates evaluated per Groq relevance request (reduces request overhead).")
    parser.add_argument("--preview-chars", type=int, default=1800, help="Chunk preview length; 0 prints full chunks.")
    parser.add_argument("--groq-model", help="Groq model used for all LLM judging and answer generation.")
    parser.add_argument("--embedding-device", choices=["cpu", "cuda"], default="cpu", help="Answer similarity device. CPU is the recommended safe default.")
    parser.add_argument("--resume", help="Resume an existing run directory containing review_state.json.")
    parser.add_argument("--run-answer-benchmark", action="store_true", help="Opt in to answer generation + answer judging (up to two additional Groq calls per question).")
    parser.add_argument("--answer-only", action="store_true", help="Run only answer generation/judging from a prior --resume run; requires saved retrieval_results.json.")
    parser.add_argument("--skip-answer-benchmark", action="store_true", help="Deprecated compatibility flag; answer benchmarking is skipped by default.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previously skipped/failed LLM requests; this may spend additional tokens.")
    parser.add_argument("--self-test", action="store_true", help="Run offline benchmark unit tests; no Chroma/Groq required.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        import unittest
        from test_knowledge_rag_benchmark import BenchmarkTests
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BenchmarkTests))
        return 0 if result.wasSuccessful() else 1
    if not args.dataset:
        print("ERROR: --dataset is required unless --self-test is used.", file=sys.stderr)
        return 2
    if not args.k or any(k <= 0 for k in args.k) or args.top_n <= 0 or args.judge_batch_size <= 0:
        print("ERROR: K values and --top-n must be positive integers.", file=sys.stderr)
        return 2
    if args.answer_only and not args.resume:
        print("ERROR: --answer-only requires --resume pointing to an existing benchmark run.", file=sys.stderr)
        return 2
    ks = sorted(set(args.k))
    try:
        backend_dir = find_backend_dir(args.backend_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    dataset_path = resolve_path(args.dataset, backend_dir)
    if not dataset_path.is_file():
        print(f"ERROR: Dataset file not found: {dataset_path}", file=sys.stderr)
        return 2
    try:
        dataset_name, questions = normalize_dataset(load_json(dataset_path))
    except Exception as exc:
        print(f"ERROR: Invalid dataset: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.db_dir:
        os.environ["CHROMA_DB_DIR"] = str(resolve_path(args.db_dir, backend_dir))
    # The retriever reads the embedding model / collection settings at import.
    os.environ.setdefault("KNOWLEDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    os.environ.setdefault("KNOWLEDGE_EMBEDDING_DEVICE", "cpu")
    os.environ.setdefault("KNOWLEDGE_COLLECTION", "knowledge")

    if args.resume:
        output_dir = resolve_path(args.resume, backend_dir)
        state_path = output_dir / "review_state.json"
        if not state_path.is_file():
            print(f"ERROR: No review_state.json in resume directory: {output_dir}", file=sys.stderr)
            return 2
        state = load_json(state_path)
        expected_ids = [q["question_id"] for q in questions]
        if state.get("dataset_question_ids") != expected_ids:
            print("ERROR: Resume dataset IDs/order do not match the supplied dataset.", file=sys.stderr)
            return 2
        # Migrate previous run state in-place. Preserve valid judgments, but
        # rebuild candidate pools from the corrected discovery implementation.
        state["benchmark_mode"] = BENCHMARK_VERSION
        state.setdefault("annotations", {})
        state.setdefault("answer_reviews", {})
        # Never reuse relevance or answer labels when the corresponding question,
        # reference answer, source mapping or evidence phrase has changed.
        for question in questions:
            qid = question["question_id"]
            annotation = state["annotations"].get(qid)
            if not annotation:
                continue
            current_fp = question_fingerprint(question)
            old_fp = annotation.get("question_fingerprint")
            if not old_fp:
                legacy_question = {
                    "question": annotation.get("question", ""),
                    "reference_answer": annotation.get("reference_answer", ""),
                    "source_document": annotation.get("source_document"),
                    "evidence_phrases": [
                        {"phrase": phrase, "source_document": annotation.get("source_document")}
                        for phrase in annotation.get("evidence_phrases", [])
                    ],
                }
                old_fp = question_fingerprint(legacy_question)
            if old_fp != current_fp:
                print(f"Resume: {qid} changed; resetting stale ground-truth and answer judgments.")
                state["annotations"].pop(qid, None)
                state["answer_reviews"].pop(qid, None)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset_name).strip("_") or "knowledge_rag"
        output_dir = backend_dir / "benchmark" / "results" / f"{safe_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_question_ids": [q["question_id"] for q in questions],
            "benchmark_mode": BENCHMARK_VERSION,
            "annotations": {},
            "answer_reviews": {},
        }
        write_json(output_dir / "run_metadata.json", {
            "dataset_path": str(dataset_path),
            "question_count": len(questions),
            "k_values": ks,
            "candidate_top_n": args.top_n,
            "embedding_model": os.getenv("KNOWLEDGE_EMBEDDING_MODEL"),
            "collection": os.getenv("KNOWLEDGE_COLLECTION"),
            "created_at_utc": state["created_at_utc"],
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        })

    print(f"Backend: {backend_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Questions: {len(questions)} | K: {ks} | candidate top-N: {args.top_n} | judge batch size: {args.judge_batch_size}")
    max_candidate_labels = sum(args.top_n * len(q["evidence_phrases"]) for q in questions)
    max_candidate_calls = sum(
        (args.top_n * len(q["evidence_phrases"]) + args.judge_batch_size - 1) // args.judge_batch_size
        for q in questions
    )
    run_answer_benchmark = bool(args.run_answer_benchmark and not args.skip_answer_benchmark)
    answer_call_budget = 2 * len(questions) if run_answer_benchmark else 0
    print(f"Upper-bound LLM workload before resume-cache savings: up to {max_candidate_labels} chunk labels "
          f"in {max_candidate_calls} batched relevance calls; {answer_call_budget} answer calls.")
    if not run_answer_benchmark:
        print("Answer generation/judging is OFF by default; add --run-answer-benchmark to spend those extra tokens.")
    print("Retrieval under test: production LocalRetriever (Chroma vector + BM25 + RRF).")
    print("Ground-truth discovery uses normalized source names and falls back to global search on unmatched source labels.")

    collection = None
    collection_count = 0
    if not args.answer_only:
        try:
            from services.ai.rag.local_retriever import _get_collection, RRF_K
            collection = _get_collection()
            collection_count = int(collection.count())
            print(f"Chroma collection loaded: {getattr(collection, 'name', os.getenv('KNOWLEDGE_COLLECTION'))} ({collection_count} chunks)")
            if collection_count <= 0:
                raise RuntimeError("The collection is empty.")
        except Exception as exc:
            print(f"ERROR: Could not open the configured Chroma collection: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("Check CHROMA_DB_DIR, KNOWLEDGE_COLLECTION, and the backend virtual environment.", file=sys.stderr)
            return 2

    try:
        judge = GroqJudge(model=args.groq_model)
        print(f"Groq model: {judge.model} (same model for ground-truth judging, answer generation and answer judging)")
        judge.validate()
        print("Groq API key validation: successful (no chat-completion tokens used).")
    except Exception as exc:
        print(f"ERROR: Could not initialize Groq: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    run_metadata_path = output_dir / "run_metadata.json"
    metadata = load_json(run_metadata_path) if run_metadata_path.exists() else {}
    metadata.update({
        "benchmark_version": BENCHMARK_VERSION,
        "groq_model": judge.model,
        "ground_truth_judge": judge.model,
        "answer_generation_model": judge.model,
        "answer_judge": judge.model,
        "api_key_value_saved": False,
        "source_filename_matching": "Unicode/case/whitespace normalized; raw Chroma values used for where filters; global fallback when unmatched.",
    })
    write_json(run_metadata_path, metadata)

    if args.answer_only:
        saved_retrieval_path = output_dir / "retrieval_results.json"
        if not saved_retrieval_path.is_file():
            print(f"ERROR: Missing saved retrieval results: {saved_retrieval_path}", file=sys.stderr)
            return 2
        try:
            saved = load_json(saved_retrieval_path)
            retrieval_records = saved.get("questions", [])
            expected_ids = [q["question_id"] for q in questions]
            actual_ids = [r.get("question_id") for r in retrieval_records]
            if actual_ids != expected_ids:
                raise ValueError("Saved retrieval question IDs/order do not match the dataset.")
            print(f"ANSWER-ONLY MODE: loaded saved retrieval results for {len(retrieval_records)} questions.")
            review_generated_answers(
                questions, retrieval_records, state, output_dir,
                os.getenv("KNOWLEDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                judge, args.embedding_device, args.retry_failed,
            )
            answer_metrics = write_answer_only_results(
                output_dir, dataset_name, questions, retrieval_records, state, judge,
                os.getenv("KNOWLEDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL), args.embedding_device,
            )
        except KeyboardInterrupt:
            print("\nAnswer benchmark paused. Re-run the same --answer-only command with --retry-failed if needed.")
            save_review_state(output_dir, state)
            return 130
        except Exception as exc:
            print(f"ERROR: Answer-only benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            save_review_state(output_dir, state)
            return 1
        print("\n" + "=" * 72)
        print("KNOWLEDGE RAG ANSWER-ONLY BENCHMARK")
        print("=" * 72)
        print(f"Questions: {answer_metrics['question_count']}")
        print(f"Judged answers: {answer_metrics['judged_question_count']}")
        print(f"Mean BGE cosine similarity: {answer_metrics['embedding_similarity_mean']}")
        print(f"LLM exact-correct rate: {answer_metrics['llm_exact_correct_rate']}")
        print(f"LLM weighted answer score: {answer_metrics['llm_weighted_answer_score']}")
        print(f"Saved answer-only results: {output_dir / 'answer_metrics.json'}")
        print(f"Per-question answers: {output_dir / 'answer_per_question.csv'}")
        return 0

    try:
        review_candidates(questions, collection, output_dir, state, args.top_n, args.preview_chars, judge, args.retry_failed, args.judge_batch_size)
    except KeyboardInterrupt:
        print("\nBenchmark paused. Re-run with --resume to continue.")
        save_review_state(output_dir, state)
        return 130

    try:
        retrieval_records = load_or_retrieve_rankings(questions, max(ks), backend_dir, output_dir, collection_count)
        state["answer_benchmark_enabled"] = run_answer_benchmark
        if run_answer_benchmark:
            review_generated_answers(
                questions,
                retrieval_records,
                state,
                output_dir,
                os.getenv("KNOWLEDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                judge,
                args.embedding_device,
                args.retry_failed,
            )
        per_question, retrieval_summary = evaluate_retrieval(questions, retrieval_records, state, ks)
        write_results(
            output_dir,
            dataset_name,
            questions,
            retrieval_records,
            per_question,
            retrieval_summary,
            state,
            judge,
            ks,
            os.getenv("KNOWLEDGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            args.embedding_device,
        )
        save_review_state(output_dir, state)
    except KeyboardInterrupt:
        print("\nBenchmark paused during retrieval/answer evaluation. Re-run with --resume to continue.")
        save_review_state(output_dir, state)
        return 130
    except Exception as exc:
        print(f"ERROR: Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        save_review_state(output_dir, state)
        return 1

    print("\n" + "=" * 72)
    print("KNOWLEDGE RAG BENCHMARK")
    print("=" * 72)
    print(f"Dataset questions reported: {retrieval_summary['reported_question_count']}/{len(questions)}")
    print(f"Retrieval-evaluable questions: {retrieval_summary['retrieval_evaluable_question_count']}/{len(questions)}")
    print(f"Not evaluable: {retrieval_summary['not_retrieval_evaluable_question_count']}")
    print(f"Missing retrieval records: {retrieval_summary['retrieval_record_missing_count']}")
    print(f"Status counts: {retrieval_summary['evaluation_status_counts']}")
    for metric, value in retrieval_summary["macro_average_on_evaluable_questions"].items():
        print(f"{metric:16s}: {'N/A' if value is None else f'{value:.4f}'}")
    if run_answer_benchmark:
        answer_metrics = load_json(output_dir / "answer_metrics.json")
        print("\nSeparate answer-generation benchmark:")
        print(f"Mean answer embedding similarity: {answer_metrics['embedding_similarity_mean']}")
        print(f"LLM exact-correct rate: {answer_metrics['llm_exact_correct_rate']}")
        print(f"LLM weighted answer score: {answer_metrics['llm_weighted_answer_score']}")
    print(f"\nSaved results: {output_dir}")
    print("Every question is in per_question_metrics.csv/json; inspect evaluation_status for non-evaluable items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
