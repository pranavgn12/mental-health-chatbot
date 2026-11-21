#!/usr/bin/env python3
"""
Benchmark the production PatternMemoryAnalyzer against dataset.json.

Place this file at backend/benchmarks/benchmark_pattern_memory.py.
Place the dataset beside it at backend/benchmarks/pattern_benchmark.json.
Run from the backend directory:
  python benchmarks/benchmark_pattern_memory.py
  python benchmarks/benchmark_pattern_memory.py --api-keys key1,key2
  python benchmarks/benchmark_pattern_memory.py  # prompts for a replacement key if all keys fail

The analyzer and Groq judge share the same API-key pool and rotation state.

This runner:
- imports and invokes the real PatternMemoryAnalyzer (same prompt + validators);
- uses an isolated temporary SQLite DB, never the app's MySQL DB;
- loads BAAI/bge-small-en-v1.5 once and reuses that model for every case;
- uses CUDA automatically when available, with an explicit CPU fallback;
- combines cosine embedding similarity with BM25 lexical similarity;
- uses a Groq LLM judge for semantic-equivalence adjudication;
- reports detection, evidence, lifecycle, ID, false-positive and duplicate metrics.

Required packages are the backend's normal dependencies plus:
  pip install sentence-transformers groq python-dotenv

The dataset must follow the schema agreed for PatternMemoryAnalyzer:
dataset["cases"][i]["input"] and ["expected"]["patterns"].
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

# Load .env before setting the import-time placeholder below. Otherwise the
# placeholder would prevent python-dotenv from loading the real GROQ_API_KEY.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # The backend's llm_provider imports python-dotenv as a required dependency;
    # this guard keeps the failure message tied to that import if it is absent.
    pass

# ----------------------------- paths / imports -----------------------------

SCRIPT = Path(__file__).resolve()
BACKEND_DIR = SCRIPT.parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DB_HOST", "127.0.0.1")
os.environ.setdefault("DB_PORT", "3306")
os.environ.setdefault("DB_USER", "benchmark")
os.environ.setdefault("DB_PASSWORD", "benchmark")
os.environ.setdefault("DB_NAME", "benchmark")
# llm_provider constructs its Groq client at import time. The benchmark replaces
# that client with the rotating proxy after CLI arguments are parsed.
os.environ.setdefault("GROQ_API_KEY", "benchmark-placeholder-key")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base
from models import User, Episode, EpisodePattern
from services.ai.pattern_memory_analyzer import PatternMemoryAnalyzer
import services.llm_provider as llm_provider

try:
    from groq import Groq
except ImportError as exc:
    raise SystemExit("Missing groq package. Install it in the backend environment.") from exc

logger = logging.getLogger("pattern_benchmark")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOKEN_RE = re.compile(r"[a-z]+\d+[a-z0-9]*|[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)


# ------------------------------ key rotation -------------------------------

def split_keys(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


class RotatingGroq:
    """Rotate shared Groq keys on quota, auth, timeout, connection, or server failures.

    In an interactive terminal, ask for a replacement key after every configured
    key has failed, then retry the exact request that failed.
    """

    def __init__(self, keys: list[str], label: str, prompt_for_key_on_failure: bool = False):
        if not keys:
            raise ValueError(f"No API keys supplied for {label}.")
        self.keys = list(keys)
        self.label = label
        self.index = 0
        self.clients = [Groq(api_key=k, timeout=30.0, max_retries=0) for k in keys]
        self.prompt_for_key_on_failure = prompt_for_key_on_failure
        self.chat = _ChatProxy(self)

    def _ask_for_another_key(self) -> bool:
        if not self.prompt_for_key_on_failure or not sys.stdin.isatty():
            return False
        try:
            from getpass import getpass
            key = getpass(f"All {self.label} Groq keys failed. Enter another API key (blank to stop): ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        if not key:
            return False
        self.keys.append(key)
        self.clients.append(Groq(api_key=key, timeout=30.0, max_retries=0))
        self.index = len(self.clients) - 1
        logger.info("Added another %s API key; retrying with key %d.", self.label, self.index + 1)
        return True

    def _is_retryable(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        message = str(exc).casefold()
        error_type = type(exc).__name__.casefold()
        return (
            status in {401, 403, 408, 429, 500, 502, 503, 504}
            or any(term in error_type for term in ("timeout", "connection"))
            or any(
                term in message for term in (
                    "rate limit", "quota", "too many requests", "invalid api key",
                    "authentication", "timed out", "timeout", "connection error",
                    "connection reset", "server disconnected", "temporarily unavailable"
                )
            )
        )

    def call(self, fn, *args, **kwargs):
        tried = 0
        last_error = None
        while True:
            available = len(self.clients)
            while tried < available:
                client = self.clients[self.index]
                try:
                    result = fn(client, *args, **kwargs)
                    return result
                except Exception as exc:
                    last_error = exc
                    if not self._is_retryable(exc):
                        raise
                    tried += 1
                    if tried >= available:
                        break
                    old = self.index
                    self.index = (self.index + 1) % available
                    logger.warning(
                        "%s key %d failed (%s); switching to key %d",
                        self.label, old + 1, str(exc)[:180], self.index + 1
                    )
            if self._ask_for_another_key():
                tried = 0
                continue
            raise RuntimeError(
                f"All {len(self.clients)} {self.label} API keys failed. Last error: {last_error}"
            )


class _ChatProxy:
    def __init__(self, owner):
        self.owner = owner
        self.completions = _CompletionsProxy(owner)


class _CompletionsProxy:
    def __init__(self, owner):
        self.owner = owner

    def create(self, *args, **kwargs):
        return self.owner.call(
            lambda client, *a, **kw: client.chat.completions.create(*a, **kw),
            *args, **kwargs
        )


# ---------------------- same retriever-style similarity ---------------------

class BM25:
    """BM25 scorer using the same k1/b defaults as local_retriever._BM25Index."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = documents
        self.tokens = [TOKEN_RE.findall(x.casefold()) for x in documents]
        self.lengths = [len(x) for x in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.n = len(self.tokens)
        self.df = Counter()
        for row in self.tokens:
            self.df.update(set(row))
        self.k1, self.b = k1, b

    def score(self, query: str, index: int) -> float:
        qtokens = Counter(TOKEN_RE.findall(query.casefold()))
        if not qtokens or not self.n:
            return 0.0
        counts = Counter(self.tokens[index])
        dl = self.lengths[index]
        total = 0.0
        for token, qtf in qtokens.items():
            tf = counts.get(token, 0)
            df = self.df.get(token, 0)
            if not tf or not df:
                continue
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (
                1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0)
            )
            total += idf * ((tf * (self.k1 + 1.0)) / denom) * min(qtf, 2)
        return total


class SimilarityEngine:
    """One persistent BGE model; embeddings run on CUDA when available."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "auto"):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers and torch are required. Install them in the backend environment."
            ) from exc

        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("embedding device must be one of: auto, cuda, cpu")
        if device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested for embeddings, but torch.cuda.is_available() is False. "
                    "Use --embedding-device cpu or install a CUDA-enabled PyTorch build."
                )
            selected_device = "cuda"
        else:
            selected_device = "cpu"

        self.model_name = model_name
        self.device = selected_device
        logger.info("Loading embedding model %s on %s", model_name, selected_device)
        # This is the only model construction in the benchmark. The same object
        # remains resident on the selected device for all samples and all batches.
        self.model = SentenceTransformer(model_name, device=selected_device)
        if selected_device.startswith("cuda"):
            try:
                logger.info("Embedding GPU: %s", torch.cuda.get_device_name(0))
            except Exception:
                logger.info("Embedding GPU: CUDA device 0")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return [list(map(float, row)) for row in vectors]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if not na or not nb:
            return 0.0
        # Preserve the benchmark's original [0, 1] cosine scaling.
        return max(0.0, min(1.0, (dot / (na * nb) + 1.0) / 2.0))

    def pair_scores(self, predicted: list[str], expected: list[str]) -> list[list[dict]]:
        # Avoid model inference entirely when there are no possible pairs.
        if not predicted:
            return []
        if not expected:
            return [[] for _ in predicted]

        all_texts = predicted + expected
        vectors = self.embed(all_texts)
        pvecs = vectors[:len(predicted)]
        evecs = vectors[len(predicted):]
        bm25 = BM25(expected)
        matrix = []
        for i, p in enumerate(predicted):
            row = []
            lexical = [bm25.score(p, j) for j in range(len(expected))]
            max_lex = max(lexical, default=0.0)
            for j, _ in enumerate(expected):
                emb = self.cosine(pvecs[i], evecs[j])
                lex = lexical[j] / max_lex if max_lex > 0 else 0.0
                # Embeddings are primary; BM25 supplements with exact lexical overlap.
                hybrid = 0.75 * emb + 0.25 * lex
                row.append({
                    "embedding_cosine": round(emb, 4),
                    "bm25_normalized": round(lex, 4),
                    "hybrid_similarity": round(hybrid, 4)
                })
            matrix.append(row)
        return matrix


# ------------------------------- LLM judge ----------------------------------

JUDGE_SYSTEM = """You are a strict evaluator of recurring-pattern detection.
Compare predicted pattern descriptions with ground-truth descriptions.
Judge semantic equivalence, not wording similarity. A match must represent
the same underlying recurring tendency, trigger, coping strategy, or pattern.
Do not award a match merely because both mention the same broad topic.
Do not use evidence not supplied in the pair. Be conservative.
Return JSON only: {"judgments":[{"predicted_index":0,"expected_index":0,
"equivalent":true,"score":0.0,"reason":"brief explanation"}]}.
Score semantic equivalence from 0 to 1. A true match should normally score >=0.75.
"""

def judge_pairs(proxy: RotatingGroq, model: str, predicted: list[dict],
                expected: list[dict], pair_indices: list[tuple[int, int]] | None = None
                ) -> dict[tuple[int, int], dict]:
    pairs = []
    indices = pair_indices if pair_indices is not None else [
        (i, j) for i in range(len(predicted)) for j in range(len(expected))
    ]
    for i, j in indices:
        if i < 0 or i >= len(predicted) or j < 0 or j >= len(expected):
            continue
        p, e = predicted[i], expected[j]
        pairs.append({
            "predicted_index": i,
            "predicted": p.get("description", ""),
            "predicted_category": p.get("category"),
            "expected_index": j,
            "expected": e.get("description", ""),
            "expected_category": e.get("category")
        })
    if not pairs:
        return {}
    prompt = (
        "Evaluate every pair independently. Indices are zero-based. "
        "Return one judgment per supplied pair, with no omissions.\n\n"
        + json.dumps(pairs, ensure_ascii=False)
    )
    response = proxy.call(
        lambda c, **kw: c.chat.completions.create(**kw),
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_completion_tokens=4096,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    judgments = data.get("judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("LLM judge response must contain a judgments array.")

    result = {}
    for item in judgments:
        try:
            if not isinstance(item, dict) or not isinstance(item.get("equivalent"), bool):
                continue
            i = int(item["predicted_index"])
            j = int(item["expected_index"])
            score = float(item["score"])
            if not (0 <= i < len(predicted) and 0 <= j < len(expected)):
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                continue
            key = (i, j)
            result[key] = {
                "equivalent": item["equivalent"],
                "score": score,
                "reason": str(item.get("reason", ""))[:500]
            }
        except (KeyError, TypeError, ValueError):
            continue
    return result


# -------------------------- isolated database setup -------------------------

def create_isolated_session():
    temp_dir = tempfile.TemporaryDirectory(prefix="pattern_benchmark_")
    db_path = Path(temp_dir.name) / "benchmark.sqlite3"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        future=True
    )
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return temp_dir, engine, Session


def seed_case(db, case: dict, user_id: int = 1):
    """Translate benchmark JSON into the same ORM records queried by the analyzer."""
    user = User(id=user_id, username=f"benchmark_{user_id}", email=f"benchmark_{user_id}@local.test",
                password_hash="not-a-real-password")
    db.add(user)
    db.flush()

    inp = case["input"]
    now = datetime.utcnow()

    # Older episodes are ordered oldest -> newest in the dataset. Analyzer queries
    # closed_at DESC and takes the newest eight, matching production behavior.
    previous = inp.get("previous_episodes", [])
    for index, ep in enumerate(previous):
        closed_at = now - timedelta(days=(len(previous) - index + 1))
        db.add(Episode(
            episode_id=str(ep["episode_id"]),
            user_id=user_id,
            title=ep.get("title") or "Untitled episode",
            summary=ep.get("summary") or "",
            status="CLOSED",
            active_user_id=None,
            created_at=closed_at - timedelta(hours=1),
            closed_at=closed_at
        ))
    current = inp["current_episode"]
    current_episode = Episode(
        episode_id=str(current["episode_id"]),
        user_id=user_id,
        title=current.get("title") or "Current episode",
        summary=current.get("summary") or "",
        status="ACTIVE",
        active_user_id=user_id,
        created_at=now,
        closed_at=None
    )
    db.add(current_episode)
    db.flush()

    for index, pattern in enumerate(inp.get("existing_patterns", [])):
        seen = now - timedelta(days=30 - index)
        db.add(EpisodePattern(
            pattern_id=str(pattern["pattern_id"]),
            user_id=user_id,
            category=pattern["category"],
            description=pattern["description"],
            confidence=float(pattern.get("confidence", 0.5)),
            evidence=pattern.get("evidence", []),
            status=pattern.get("status", "ACTIVE"),
            first_seen_at=seen - timedelta(days=10),
            last_seen_at=seen,
            created_at=seen,
            updated_at=seen
        ))
    db.flush()
    return current_episode


class _AnalyzerFailureCapture(logging.Handler):
    """Detect failures swallowed and logged by PatternMemoryAnalyzer itself."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.failures: list[str] = []

    def emit(self, record):
        message = record.getMessage()
        if "Pattern analysis failed" in message:
            self.failures.append(message)


def run_analyzer_and_raise_on_logged_failure(analyzer, db, user_id, episode):
    analyzer_logger = logging.getLogger(PatternMemoryAnalyzer.__module__)
    capture = _AnalyzerFailureCapture()
    analyzer_logger.addHandler(capture)
    try:
        analyzer.analyze_and_save(db, user_id, episode)
    finally:
        analyzer_logger.removeHandler(capture)
    if capture.failures:
        raise RuntimeError("Analyzer swallowed an internal failure: " + " | ".join(capture.failures))


# ------------------------------ evaluation ---------------------------------

def evidence_metrics(predicted: list[dict], summaries: dict[str, str]) -> dict:
    total = valid = 0
    for p in predicted:
        for ev in p.get("evidence", []) or []:
            total += 1
            eid = str(ev.get("episode_id", ""))
            quote = str(ev.get("detail", ev.get("quote", "")))
            summary = summaries.get(eid, "")
            if quote and re.sub(r"\s+", " ", quote).casefold() in re.sub(
                r"\s+", " ", summary
            ).casefold():
                valid += 1
    return {
        "evidence_quote_validity": valid / total if total else 1.0,
        "evidence_quotes_total": total,
        "evidence_quotes_valid": valid
    }


def match_predictions(predicted: list[dict], expected: list[dict],
                      sim_matrix: list[list[dict]],
                      judge: dict[tuple[int, int], dict],
                      similarity_threshold: float,
                      judge_threshold: float) -> tuple[list[tuple[int, int]], list[dict]]:
    """One-to-one maximum-cardinality matching, with score-based tie-breaking."""
    details = []
    accepted_scores = {}
    for i, p in enumerate(predicted):
        for j, e in enumerate(expected):
            sim = sim_matrix[i][j] if i < len(sim_matrix) and j < len(sim_matrix[i]) else {}
            jv = judge.get((i, j), {})
            judge_score = float(jv.get("score", 0.0))
            equivalent = bool(jv.get("equivalent", False))
            hybrid = float(sim.get("hybrid_similarity", 0.0))
            combined = 0.65 * hybrid + 0.35 * judge_score
            accepted = equivalent and judge_score >= judge_threshold and hybrid >= similarity_threshold
            if accepted:
                accepted_scores[(i, j)] = combined
            details.append({
                "predicted_index": i,
                "expected_index": j,
                **sim,
                "judge_score": judge_score,
                "judge_equivalent": equivalent,
                "combined_score": round(combined, 4),
                "accepted_match": accepted,
                "judge_reason": jv.get("reason", "")
            })

    # Find the maximum-cardinality one-to-one matching, using combined score to
    # break ties. Greedy matching can lose a true positive when one prediction
    # has two acceptable candidates but another prediction has only one.
    @lru_cache(maxsize=None)
    def best_assignment(pred_index: int, used_expected_mask: int):
        if pred_index >= len(predicted):
            return (0, 0.0, ())

        best = best_assignment(pred_index + 1, used_expected_mask)
        for expected_index in range(len(expected)):
            if used_expected_mask & (1 << expected_index):
                continue
            edge_score = accepted_scores.get((pred_index, expected_index))
            if edge_score is None:
                continue
            count, total_score, pairs = best_assignment(
                pred_index + 1, used_expected_mask | (1 << expected_index)
            )
            candidate = (count + 1, total_score + edge_score,
                         ((pred_index, expected_index),) + pairs)
            if candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] > best[1]
            ):
                best = candidate
        return best

    matches = list(best_assignment(0, 0)[2])
    return matches, details


def lifecycle_expected_status(case: dict, expected_pattern: dict) -> str:
    return expected_pattern.get("status", "ACTIVE")


def evaluate_case(case: dict, predicted_records: list[dict], similarity: SimilarityEngine,
                  judge_proxy: RotatingGroq, judge_model: str,
                  sim_threshold: float, judge_threshold: float) -> dict:
    expected = case["expected"].get("patterns", [])
    inp = case["input"]
    summaries = {
        str(inp["current_episode"]["episode_id"]): inp["current_episode"].get("summary", "")
    }
    summaries.update({
        str(x["episode_id"]): x.get("summary", "")
        for x in inp.get("previous_episodes", [])
    })

    # Only new/changed patterns are returned by the production analyzer.
    predicted = predicted_records
    sim_matrix = similarity.pair_scores(
        [x.get("description", "") for x in predicted],
        [x.get("description", "") for x in expected]
    )
    judge = judge_pairs(judge_proxy, judge_model, predicted, expected)
    matches, pair_details = match_predictions(
        predicted, expected, sim_matrix, judge, sim_threshold, judge_threshold
    )

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(expected) - tp
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    evidence = evidence_metrics(predicted, summaries)
    lifecycle_total = lifecycle_correct = id_total = id_correct = 0
    for i, j in matches:
        p, e = predicted[i], expected[j]
        lifecycle_total += 1
        lifecycle_correct += int(p.get("status") == lifecycle_expected_status(case, e))
        if e.get("pattern_id") is not None:
            id_total += 1
            id_correct += int(str(p.get("pattern_id")) == str(e.get("pattern_id")))

    # Expected evidence coverage: expected evidence episode IDs represented by
    # at least one valid predicted evidence quote in the matching pattern.
    coverage_values = []
    for i, j in matches:
        exp_ids = {str(x["episode_id"]) for x in expected[j].get("evidence", [])}
        pred_ids = {
            str(x.get("episode_id")) for x in predicted[i].get("evidence", [])
            if str(x.get("episode_id")) in summaries
            and re.sub(r"\s+", " ", str(x.get("detail", x.get("quote", "")))).casefold()
                in re.sub(r"\s+", " ", summaries[str(x.get("episode_id"))]).casefold()
        }
        coverage_values.append(len(exp_ids & pred_ids) / len(exp_ids) if exp_ids else 1.0)

    # Duplicate predictions: descriptions judged equivalent to another prediction.
    duplicate_pairs = 0
    if len(predicted) > 1:
        unordered_pairs = [
            (i, j) for i in range(len(predicted)) for j in range(i + 1, len(predicted))
        ]
        dup_judgments = judge_pairs(
            judge_proxy, judge_model, predicted, predicted, pair_indices=unordered_pairs
        )
        for i in range(len(predicted)):
            for j in range(i + 1, len(predicted)):
                if dup_judgments.get((i, j), {}).get("equivalent", False):
                    duplicate_pairs += 1

    return {
        "case_id": case["case_id"],
        "scenario": case.get("evaluation", {}).get("scenario", "unknown"),
        "predicted_count": len(predicted),
        "expected_count": len(expected),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        **evidence,
        "evidence_episode_coverage": round(
            sum(coverage_values) / len(coverage_values), 4
        ) if coverage_values else (1.0 if not expected else 0.0),
        "lifecycle_accuracy": (
            round(lifecycle_correct / lifecycle_total, 4) if lifecycle_total else None
        ),
        "pattern_id_accuracy": (
            round(id_correct / id_total, 4) if id_total else None
        ),
        "duplicate_prediction_pairs": duplicate_pairs,
        "predictions": predicted,
        "expected": expected,
        "pairwise_similarity": pair_details
    }


# ---------------------------------- main ------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark PatternMemoryAnalyzer.")
    benchmark_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--dataset",
        default=str(benchmark_dir / "pattern_benchmark.json"),
        help="Path to validated dataset (default: pattern_benchmark.json beside this script)."
    )
    parser.add_argument("--api-keys", default=os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY"),
                        help="Comma-separated Groq keys for the analyzer (or GROQ_API_KEYS).")
    # The judge intentionally shares the analyzer's API-key pool.
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--judge-model", default=os.getenv("GROQ_JUDGE_MODEL") or
                        os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-device", choices=("auto", "cuda", "cpu"), default="auto",
        help="Embedding device (default: CUDA when available, otherwise CPU)."
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--judge-threshold", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=0, help="Run first N cases (0 = all).")
    parser.add_argument(
        "--output",
        default=str(benchmark_dir / "pattern_benchmark_results.json"),
        help="Results JSON path (default: save beside this script)."
    )
    parser.add_argument("--skip-judge", action="store_true",
                        help="Use embedding+BM25 matching only; no Groq judge calls.")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds between cases.")
    parser.add_argument(
        "--no-prompt-for-key-on-failure", action="store_true",
        help="Disable interactive API-key replacement prompts (automatic rotation still works)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Invalid dataset: expected a non-empty top-level 'cases' array.")
    if args.limit < 0:
        raise SystemExit("--limit must be zero (all cases) or a positive integer.")
    if not 0.0 <= args.similarity_threshold <= 1.0:
        raise SystemExit("--similarity-threshold must be between 0 and 1.")
    if not 0.0 <= args.judge_threshold <= 1.0:
        raise SystemExit("--judge-threshold must be between 0 and 1.")
    if args.pause < 0:
        raise SystemExit("--pause cannot be negative.")
    if args.limit:
        cases = cases[:args.limit]

    analyzer_keys = split_keys(args.api_keys)
    if not analyzer_keys:
        raise SystemExit("Provide --api-keys or GROQ_API_KEYS/GROQ_API_KEY.")

    # PatternMemoryAnalyzer imports generate_response directly, but that function
    # reads llm_provider.client at call time. Replace that shared client with a
    # rotating proxy so the real analyzer transparently retries on another key.
    analyzer_proxy = RotatingGroq(
        analyzer_keys, "benchmark", prompt_for_key_on_failure=(sys.stdin.isatty() and not args.no_prompt_for_key_on_failure)
    )
    llm_provider.client = analyzer_proxy
    llm_provider.MODEL = args.model

    # Share the exact same rotating key pool between analyzer and judge.
    # A key switch caused by either workload is therefore used by both.
    judge_proxy = analyzer_proxy if not args.skip_judge else None

    print(f"Loading embedding model: {args.embedding_model} (device={args.embedding_device})")
    similarity = SimilarityEngine(args.embedding_model, device=args.embedding_device)
    print(f"Embedding model loaded once on: {similarity.device}")

    temp_dir, engine, Session = create_isolated_session()
    analyzer = PatternMemoryAnalyzer()
    results = []
    start = time.time()

    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.get('case_id', 'UNKNOWN')} "
                  f"({case.get('evaluation', {}).get('scenario', 'unknown')})")
            # The schema has globally unique episode_id and pattern_id columns.
            # Reuse of synthetic IDs across cases would otherwise collide even
            # though each case uses a different user. Reset all tables per case.
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            db = Session()
            try:
                # Each case is isolated: a new user and fresh DB rows.
                user_id = index
                episode = seed_case(db, case, user_id=user_id)
                db.commit()

                # Snapshot the seeded patterns. analyze_and_save() returns the
                # retained store (including unchanged rows), so benchmark only
                # rows that were newly created or actually changed by this run.
                before_rows = (
                    db.query(EpisodePattern)
                    .filter(EpisodePattern.user_id == user_id)
                    .all()
                )
                before = {
                    row.pattern_id: {
                        "category": row.category,
                        "description": row.description,
                        "confidence": float(row.confidence or 0.0),
                        "status": row.status,
                        "evidence": row.evidence or [],
                        "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None
                    }
                    for row in before_rows
                }

                # Same production method, prompt, retrieval window, validators and writes.
                run_analyzer_and_raise_on_logged_failure(analyzer, db, user_id, episode)
                db.commit()

                # Reload persisted rows and isolate newly-created/updated patterns.
                stored = (
                    db.query(EpisodePattern)
                    .filter(EpisodePattern.user_id == user_id)
                    .all()
                )
                predicted_records = []
                for row in stored:
                    current = {
                        "category": row.category,
                        "description": row.description,
                        "confidence": float(row.confidence or 0.0),
                        "status": row.status,
                        "evidence": row.evidence or [],
                        "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None
                    }
                    old = before.get(row.pattern_id)
                    if old is not None and old == current:
                        continue
                    predicted_records.append({
                        "pattern_id": row.pattern_id,
                        "category": row.category,
                        "description": row.description,
                        "confidence": float(row.confidence or 0.0),
                        "status": row.status,
                        "evidence": row.evidence or []
                    })

                if args.skip_judge:
                    # Embeddings are computed exactly once inside this evaluator.
                    result = evaluate_without_judge(
                        case, predicted_records, similarity, args.similarity_threshold
                    )
                else:
                    result = evaluate_case(
                        case, predicted_records, similarity, judge_proxy,
                        args.judge_model, args.similarity_threshold, args.judge_threshold
                    )
                results.append(result)
                print(
                    f"  predicted={result['predicted_count']} expected={result['expected_count']} "
                    f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f}"
                )
            except Exception as exc:
                db.rollback()
                logger.exception("Case %s failed", case.get("case_id"))
                results.append({
                    "case_id": case.get("case_id"),
                    "scenario": case.get("evaluation", {}).get("scenario", "unknown"),
                    "error": str(exc),
                    "failed": True
                })
            finally:
                db.close()
            if args.pause and index < len(cases):
                time.sleep(args.pause)
    finally:
        engine.dispose()
        temp_dir.cleanup()

    successful = [x for x in results if not x.get("failed")]
    total_tp = sum(x.get("tp", 0) for x in successful)
    total_fp = sum(x.get("fp", 0) for x in successful)
    total_fn = sum(x.get("fn", 0) for x in successful)
    if not successful:
        # Do not emit apparently perfect metrics when every sample failed.
        precision = recall = f1 = 0.0
    else:
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else (1.0 if total_fn == 0 else 0.0)
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def mean_metric(key):
        vals = [x[key] for x in successful if isinstance(x.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    report = {
        "benchmark": "PatternMemoryAnalyzer",
        "dataset": str(dataset_path),
        "model": args.model,
        "judge_model": None if args.skip_judge else args.judge_model,
        "embedding_model": args.embedding_model,
        "embedding_device": similarity.device,
        "similarity_threshold": args.similarity_threshold,
        "judge_threshold": None if args.skip_judge else args.judge_threshold,
        "total_samples": len(cases),
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "runtime_seconds": round(time.time() - start, 2),
        "metrics": {
            "micro_precision": round(precision, 4),
            "micro_recall": round(recall, 4),
            "micro_f1": round(f1, 4),
            "macro_case_f1": round(mean_metric("f1") or 0.0, 4),
            "evidence_quote_validity": round(mean_metric("evidence_quote_validity") or 0.0, 4),
            "evidence_episode_coverage": round(mean_metric("evidence_episode_coverage") or 0.0, 4),
            "lifecycle_accuracy": mean_metric("lifecycle_accuracy"),
            "pattern_id_accuracy": mean_metric("pattern_id_accuracy"),
            "negative_case_false_positive_rate": (
                sum(1 for x in successful
                    if x.get("scenario") == "no_pattern" and x.get("predicted_count", 0) > 0)
                / max(1, sum(1 for x in successful if x.get("scenario") == "no_pattern"))
            ),
            "duplicate_prediction_pairs": (
                None if args.skip_judge else sum(
                    x.get("duplicate_prediction_pairs", 0) or 0 for x in successful
                )
            )
        },
        "by_scenario": {},
        "cases": results
    }
    for scenario in sorted({x.get("scenario", "unknown") for x in successful}):
        subset = [x for x in successful if x.get("scenario") == scenario]
        report["by_scenario"][scenario] = {
            "samples": len(subset),
            "mean_f1": round(sum(x.get("f1", 0.0) for x in subset) / len(subset), 4),
            "precision": round(
                sum(x.get("tp", 0) for x in subset) /
                max(1, sum(x.get("tp", 0) + x.get("fp", 0) for x in subset)), 4
            ),
            "recall": round(
                sum(x.get("tp", 0) for x in subset) /
                max(1, sum(x.get("tp", 0) + x.get("fn", 0) for x in subset)), 4
            )
        }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== PATTERN MEMORY BENCHMARK ==========")
    print(f"Samples: {len(cases)} | completed: {len(successful)} | failed: {len(results)-len(successful)}")
    print(f"Micro Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Evidence validity: {report['metrics']['evidence_quote_validity']:.4f}")
    print(f"Evidence episode coverage: {report['metrics']['evidence_episode_coverage']:.4f}")
    print(f"Lifecycle accuracy: {report['metrics']['lifecycle_accuracy']}")
    print(f"Pattern ID accuracy: {report['metrics']['pattern_id_accuracy']}")
    print(f"Negative-case false positive rate: {report['metrics']['negative_case_false_positive_rate']:.4f}")
    print(f"Results saved: {output_path}")


def evaluate_without_judge(case, predicted, similarity, threshold):
    expected = case["expected"].get("patterns", [])
    matrix = similarity.pair_scores(
        [x.get("description", "") for x in predicted],
        [x.get("description", "") for x in expected]
    )
    judge = {}
    for i, row in enumerate(matrix):
        for j, sim in enumerate(row):
            score = sim["hybrid_similarity"]
            judge[(i, j)] = {
                "equivalent": score >= threshold,
                "score": score,
                "reason": "Embedding/BM25-only mode"
            }
    matches, details = match_predictions(
        predicted, expected, matrix, judge, threshold, threshold
    )
    tp, fp, fn = len(matches), len(predicted)-len(matches), len(expected)-len(matches)
    p = tp/(tp+fp) if tp+fp else (1.0 if fn == 0 else 0.0)
    r = tp/(tp+fn) if tp+fn else 1.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    summaries = {str(case["input"]["current_episode"]["episode_id"]):
                 case["input"]["current_episode"].get("summary", "")}
    summaries.update({str(x["episode_id"]): x.get("summary", "")
                      for x in case["input"].get("previous_episodes", [])})
    ev = evidence_metrics(predicted, summaries)
    coverage_values = []
    for i, j in matches:
        expected_ids = {
            str(item["episode_id"]) for item in expected[j].get("evidence", [])
            if isinstance(item, dict) and item.get("episode_id") is not None
        }
        predicted_ids = {
            str(item.get("episode_id")) for item in predicted[i].get("evidence", [])
            if isinstance(item, dict)
            and str(item.get("episode_id")) in summaries
            and re.sub(r"\s+", " ", str(item.get("detail", item.get("quote", "")))).casefold()
                in re.sub(r"\s+", " ", summaries[str(item.get("episode_id"))]).casefold()
        }
        coverage_values.append(
            len(expected_ids & predicted_ids) / len(expected_ids) if expected_ids else 1.0
        )
    lifecycle = [
        int(predicted[i].get("status") == expected[j].get("status", "ACTIVE"))
        for i, j in matches
    ]
    id_checks = [
        int(str(predicted[i].get("pattern_id")) == str(expected[j].get("pattern_id")))
        for i, j in matches if expected[j].get("pattern_id") is not None
    ]
    return {
        "case_id": case["case_id"],
        "scenario": case.get("evaluation", {}).get("scenario", "unknown"),
        "predicted_count": len(predicted),
        "expected_count": len(expected),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        **ev,
        "evidence_episode_coverage": round(
            sum(coverage_values) / len(coverage_values), 4
        ) if coverage_values else (1.0 if not expected else 0.0),
        "lifecycle_accuracy": sum(lifecycle)/len(lifecycle) if lifecycle else None,
        "pattern_id_accuracy": sum(id_checks)/len(id_checks) if id_checks else None,
        "duplicate_prediction_pairs": None,
        "predictions": predicted,
        "expected": expected,
        "pairwise_similarity": details
    }


if __name__ == "__main__":
    main()
