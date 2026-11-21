#!/usr/bin/env python3
"""Benchmark the production TurnAnalyzer against human-annotated JSON samples.

Run from the backend directory:
    python benchmark/turn_analyzer_benchmark.py \
        --dataset benchmark/datasets/turn_analyzer_dataset.json

Resume an interrupted run without repeating successful samples:
    python benchmark/turn_analyzer_benchmark.py \
        --dataset benchmark/datasets/turn_analyzer_dataset.json --resume

The script calls the real services.ai.turn_analyzer.TurnAnalyzer and the real
services.llm_provider.generate_response function. It temporarily replaces the
provider's client when the user supplies a new Groq API key; it does not edit
production source files or write API keys to disk.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
BENCHMARK_VERSION = "turn_analyzer_benchmark_v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
ALLOWED_MOODS = {"happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"}
NULL_MOOD_TOKENS = {"", "none", "null", "unknown", "unclear"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_backend_dir(explicit: str | None = None) -> Path:
    """Locate the project backend that contains services/ai/turn_analyzer.py."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([SCRIPT_PATH.parent, *SCRIPT_PATH.parents])
    cwd = Path.cwd()
    candidates.extend([cwd, cwd / "backend", cwd.parent / "backend"])
    env_dir = os.getenv("MENTAL_HEALTH_BACKEND_DIR")
    if env_dir:
        candidates.insert(0, Path(env_dir).expanduser())

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if (resolved / "services" / "ai" / "turn_analyzer.py").is_file():
            return resolved
    raise FileNotFoundError(
        "Could not locate services/ai/turn_analyzer.py. Put this script in "
        "backend/benchmark/ or pass --backend-dir."
    )


def resolve_path(raw: str | Path, base: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (base / path).resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so an interrupted run leaves the last good report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temp_path.replace(path)


def canonical_fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_mood(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in NULL_MOOD_TOKENS:
        return None
    if normalized not in ALLOWED_MOODS:
        raise ValueError(f"Unsupported mood label {value!r}; allowed: {sorted(ALLOWED_MOODS)} or null.")
    return normalized


def normalize_reflection(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected_output.reflection must be a string or null.")
    normalized = " ".join(value.split()).strip()
    return normalized or None


def validate_dataset(raw: Any) -> list[dict[str, Any]]:
    """Validate the user's list-of-conversations JSON without rewriting it."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("Dataset root must be a non-empty JSON array of sample objects.")

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, sample in enumerate(raw, start=1):
        prefix = f"Sample #{index}"
        if not isinstance(sample, dict):
            raise ValueError(f"{prefix} must be a JSON object.")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{prefix} is missing a non-empty sample_id.")
        sample_id = sample_id.strip()
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)

        input_data = sample.get("input")
        expected = sample.get("expected_output")
        if not isinstance(input_data, dict):
            raise ValueError(f"{sample_id}: input must be an object.")
        if not isinstance(expected, dict):
            raise ValueError(f"{sample_id}: expected_output must be an object.")

        latest = input_data.get("latest_message")
        if not isinstance(latest, str) or not latest.strip():
            raise ValueError(f"{sample_id}: input.latest_message must be a non-empty string.")

        conversation = input_data.get("conversation", [])
        if not isinstance(conversation, list):
            raise ValueError(f"{sample_id}: input.conversation must be an array.")
        for message_index, message in enumerate(conversation, start=1):
            if not isinstance(message, dict):
                raise ValueError(f"{sample_id}: conversation item {message_index} must be an object.")
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"}:
                raise ValueError(f"{sample_id}: conversation item {message_index} role must be user or assistant.")
            if not isinstance(content, str):
                raise ValueError(f"{sample_id}: conversation item {message_index} content must be a string.")

        episode = input_data.get("active_episode")
        if episode is not None:
            if not isinstance(episode, dict):
                raise ValueError(f"{sample_id}: active_episode must be an object or null.")
            if not episode.get("episode_id"):
                raise ValueError(f"{sample_id}: active_episode.episode_id is required by TurnAnalyzer.")

        previous_moods = input_data.get("previous_moods", [])
        previous_reflections = input_data.get("previous_reflections", [])
        if not isinstance(previous_moods, list):
            raise ValueError(f"{sample_id}: previous_moods must be an array.")
        if not isinstance(previous_reflections, list):
            raise ValueError(f"{sample_id}: previous_reflections must be an array.")

        # Keep the dataset intact, but the runtime adapter below mirrors the
        # production orchestrator: without an active episode, no episode-scoped
        # mood/reflection history is available to TurnAnalyzer.

        for mood_index, item in enumerate(previous_moods, start=1):
            if not isinstance(item, dict) or not isinstance(item.get("mood"), str):
                raise ValueError(f"{sample_id}: previous_moods[{mood_index - 1}] needs a string 'mood'.")
        for reflection_index, item in enumerate(previous_reflections, start=1):
            if not isinstance(item, dict) or not isinstance(item.get("reflection"), str):
                raise ValueError(f"{sample_id}: previous_reflections[{reflection_index - 1}] needs a string 'reflection'.")

        expected_mood = expected.get("mood")
        if not isinstance(expected_mood, dict):
            raise ValueError(f"{sample_id}: expected_output.mood must be an object with a label.")
        normalized_expected_mood = normalize_mood(expected_mood.get("label"))
        expected_reflection = normalize_reflection(expected.get("reflection"))

        # Episode labels are optional for this benchmark. If supplied, they
        # are scored separately; reflection validation still follows the real
        # analyzer's dependency on its generated/validated episode action.
        expected_episode = expected.get("episode")
        if expected_episode is not None:
            if not isinstance(expected_episode, dict):
                raise ValueError(f"{sample_id}: expected_output.episode must be an object when provided.")
            action = str(expected_episode.get("action", "")).strip().upper()
            if action not in {"CREATE", "NONE", "CONTINUE", "CLOSE"}:
                raise ValueError(f"{sample_id}: expected_output.episode.action is invalid.")
            allowed_actions = {"CONTINUE", "CLOSE"} if episode is not None else {"CREATE", "NONE"}
            if action not in allowed_actions:
                raise ValueError(
                    f"{sample_id}: expected episode action {action} conflicts with active_episode "
                    f"({ 'active' if episode is not None else 'absent' }). Allowed: {sorted(allowed_actions)}."
                )
            if expected_reflection is not None and action == "NONE":
                raise ValueError(
                    f"{sample_id}: expected reflection is non-null but expected episode action is NONE; "
                    "the production TurnAnalyzer suppresses reflections for NONE actions."
                )

        normalized = dict(sample)
        normalized["sample_id"] = sample_id
        normalized["_expected_mood_label"] = normalized_expected_mood
        normalized["_expected_reflection_text"] = expected_reflection
        normalized["_sample_fingerprint"] = canonical_fingerprint(sample)
        validated.append(normalized)

    return validated


def make_namespace_records(records: list[dict[str, Any]], field: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(**record) for record in records]


def prepare_analyzer_inputs(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON dictionaries to the attribute-based records production uses."""
    input_data = sample["input"]
    active_episode_data = input_data.get("active_episode")
    active_episode = None
    if active_episode_data is not None:
        # The production ORM Episode exposes these four attributes to build_prompt.
        episode_values = {
            "episode_id": active_episode_data["episode_id"],
            "title": active_episode_data.get("title"),
            "status": active_episode_data.get("status", "ACTIVE"),
            "summary": active_episode_data.get("summary"),
        }
        active_episode = SimpleNamespace(**episode_values)

    # The production orchestrator retrieves mood/reflection history through
    # the active episode. If there is no active episode, pass empty histories
    # even if the source dataset contains them. The source JSON is not mutated.
    previous_moods = input_data.get("previous_moods", []) if active_episode is not None else []
    previous_reflections = input_data.get("previous_reflections", []) if active_episode is not None else []

    return {
        "active_episode": active_episode,
        "latest_message": input_data["latest_message"],
        "conversation": input_data.get("conversation", []),
        "previous_moods": make_namespace_records(previous_moods, "mood"),
        "previous_reflections": make_namespace_records(previous_reflections, "reflection"),
    }


def metric_prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if precision == 0 or recall == 0 else None)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_mood_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [item for item in samples if item.get("status") == "completed"]
    if not evaluable:
        return {"evaluated_samples": 0, "accuracy": None, "macro_precision": None, "macro_recall": None,
                "macro_f1": None, "per_class": {}, "confusion_matrix": {}}

    truth = [item["expected"]["mood"]["label"] for item in evaluable]
    predicted = [item["predicted"]["mood"].get("label") for item in evaluable]
    labels = sorted({"<null>" if value is None else value for value in truth + predicted})
    truth_norm = ["<null>" if value is None else value for value in truth]
    pred_norm = ["<null>" if value is None else value for value in predicted]

    matrix: dict[str, dict[str, int]] = {
        actual: {guess: 0 for guess in labels} for actual in labels
    }
    for actual, guess in zip(truth_norm, pred_norm):
        matrix[actual][guess] += 1

    per_class: dict[str, Any] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(matrix[label][guess] for guess in labels if guess != label)
        scores = metric_prf(tp, fp, fn)
        support = sum(matrix[label].values())
        per_class[label] = {"support": support, **scores}
        if scores["precision"] is not None:
            precisions.append(float(scores["precision"]))
        if scores["recall"] is not None:
            recalls.append(float(scores["recall"]))
        if scores["f1"] is not None:
            f1s.append(float(scores["f1"]))

    correct = sum(actual == guess for actual, guess in zip(truth_norm, pred_norm))
    return {
        "evaluated_samples": len(evaluable),
        "correct": correct,
        "incorrect": len(evaluable) - correct,
        "accuracy": correct / len(evaluable),
        "macro_precision": sum(precisions) / len(precisions) if precisions else None,
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "null_label_display": "<null>",
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def calculate_reflection_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [item for item in samples if item.get("status") == "completed"]
    tp = fp = fn = tn = 0
    for item in evaluable:
        expected = item["expected"].get("reflection") is not None
        predicted = item["predicted"].get("reflection") is not None
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif expected and not predicted:
            fn += 1
        else:
            tn += 1
    scores = metric_prf(tp, fp, fn)
    total = tp + fp + fn + tn
    return {
        "evaluated_samples": len(evaluable),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "accuracy": (tp + tn) / total if total else None,
        **scores,
        "quality_note": "Textual reflection quality is not scored automatically; inspect the expected/predicted reflection pairs in the sample results.",
    }


def calculate_episode_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [
        item for item in samples
        if item.get("status") == "completed" and item.get("expected", {}).get("episode") is not None
    ]
    if not evaluable:
        return {"evaluated_samples": 0, "accuracy": None, "note": "Add expected_output.episode.action to score episode decisions."}
    correct = 0
    for item in evaluable:
        expected_action = item["expected"]["episode"]["action"]
        predicted_action = item["predicted"].get("episode", {}).get("action")
        correct += expected_action == predicted_action
    return {"evaluated_samples": len(evaluable), "correct": correct, "accuracy": correct / len(evaluable)}


def calculate_category_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in samples:
        groups[str(item.get("category") or "uncategorized")].append(item)
    report: dict[str, Any] = {}
    for category, rows in sorted(groups.items()):
        completed = [row for row in rows if row.get("status") == "completed"]
        mood_accuracy = None
        if completed:
            mood_accuracy = sum(
                row["expected"]["mood"]["label"] == row["predicted"]["mood"].get("label")
                for row in completed
            ) / len(completed)
        reflection = calculate_reflection_metrics(rows)
        report[category] = {
            "samples": len(rows),
            "completed": len(completed),
            "mood_accuracy": mood_accuracy,
            "reflection_detection_f1": reflection["f1"],
        }
    return report


def build_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    completed = sum(item.get("status") == "completed" for item in samples)
    failed = sum(item.get("status") == "failed" for item in samples)
    skipped = sum(item.get("status") == "skipped" for item in samples)
    return {
        "total_samples": len(samples),
        "completed_samples": completed,
        "failed_samples": failed,
        "skipped_samples": skipped,
        "mood": calculate_mood_metrics(samples),
        "reflection_detection": calculate_reflection_metrics(samples),
        "episode": calculate_episode_metrics(samples),
        "by_category": calculate_category_metrics(samples),
    }


def is_json_object_response(response: Any) -> bool:
    """Check whether a returned completion contains the JSON object production expects."""
    if not isinstance(response, str) or not response.strip():
        return False
    content = response.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    start = content.find("{")
    if start < 0:
        return False
    try:
        value, _ = json.JSONDecoder().raw_decode(content[start:])
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(value, dict)


class GroqRequestController:
    """Use the production provider with interactive Groq key replacement on limits."""

    def __init__(self, provider_module: Any, groq_class: Any):
        self.provider = provider_module
        self.Groq = groq_class
        self.active_key = os.getenv("GROQ_API_KEY", "")
        self.model = provider_module.MODEL
        self.key_switches = 0
        self.request_attempts = 0
        self.successful_requests = 0
        self.skipped_requests = 0
        self.last_request: dict[str, Any] = {"status": "not_started"}
        self.last_raw_response: str | None = None

    @staticmethod
    def status_code(exc: Exception) -> int | None:
        return getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )

    def install_key(self, key: str) -> None:
        client = self.Groq(api_key=key)
        client.models.list()  # Validate without consuming a chat completion.
        self.provider.client = client
        self.active_key = key
        self.key_switches += 1
        print("Groq API key validated and activated. The key is held in memory only.")

    def validate_initial_key(self) -> None:
        while True:
            try:
                self.provider.client.models.list()
                print(f"Groq key validated. Model: {self.model}")
                return
            except Exception as exc:
                status = self.status_code(exc)
                print(f"Initial Groq key validation failed (HTTP {status or 'unknown'}: {type(exc).__name__}).")
                if not sys.stdin.isatty():
                    raise RuntimeError("Groq API key validation failed and no interactive terminal is available.") from exc
                choice = input("[k] enter another API key  [q] quit: ").strip().lower()
                if choice in {"q", "quit"}:
                    raise KeyboardInterrupt
                if choice in {"k", "key"}:
                    replacement = getpass.getpass("New Groq API key (hidden): ").strip()
                    if not replacement:
                        print("No API key entered.")
                        continue
                    try:
                        self.install_key(replacement)
                        return
                    except Exception as replacement_error:
                        print(f"Replacement key validation failed: {type(replacement_error).__name__}: {replacement_error}")

    def _prompt_after_limit(self, exc: Exception, status: int | None) -> str:
        if not sys.stdin.isatty():
            print("No interactive terminal: recording this request as skipped.")
            return "skip"

        print("\nGroq request limit/authentication error:")
        print(f"  HTTP status: {status or 'unknown'}")
        print(f"  Error type: {type(exc).__name__}")
        if status == 429:
            print("  A different Groq key may have separate rate limits or quota.")
            prompt = "[k] switch API key  [r] wait 30 seconds and retry  [s] skip sample  [q] quit: "
        elif status == 401:
            prompt = "[k] enter a valid API key  [s] skip sample  [q] quit: "
        else:
            prompt = "[k] switch API key  [r] retry  [s] skip sample  [q] quit: "

        while True:
            choice = input(prompt).strip().lower()
            if choice in {"k", "key"}:
                replacement = getpass.getpass("New Groq API key (hidden): ").strip()
                if not replacement:
                    print("No API key entered.")
                    continue
                try:
                    self.install_key(replacement)
                    return "retry"
                except Exception as key_error:
                    print(f"Replacement key validation failed: {type(key_error).__name__}: {key_error}")
                    continue
            if choice in {"r", "retry"} and status != 401:
                if status == 429:
                    print("Waiting 30 seconds before retrying the same key...")
                    time.sleep(30)
                else:
                    print("Retrying request...")
                    time.sleep(3)
                return "retry"
            if choice in {"s", "skip"}:
                return "skip"
            if choice in {"q", "quit"}:
                return "quit"
            print("Choose one of the displayed options.")

    def generate_response(self, *args: Any, **kwargs: Any) -> str:
        """Drop-in wrapper installed only in turn_analyzer's module namespace."""
        self.last_request = {"status": "running", "attempts": 0, "http_status": None, "error": None}
        transient_retries = 0
        while True:
            self.request_attempts += 1
            self.last_request["attempts"] += 1
            try:
                response = self.provider.generate_response(*args, **kwargs)
                self.successful_requests += 1
                self.last_raw_response = response if isinstance(response, str) else None
                self.last_request.update({"status": "ok", "http_status": None, "error": None})
                return response
            except Exception as exc:
                status = self.status_code(exc)
                self.last_request.update({
                    "http_status": status,
                    "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
                })

                if status in {401, 429}:
                    action = self._prompt_after_limit(exc, status)
                    if action == "retry":
                        continue
                    if action == "skip":
                        self.skipped_requests += 1
                        self.last_request["status"] = "skipped"
                        return ""
                    self.last_request["status"] = "interrupted"
                    raise KeyboardInterrupt

                if status is not None and status >= 500 and transient_retries < 2:
                    delay = 3 if transient_retries == 0 else 8
                    transient_retries += 1
                    print(f"Groq server error HTTP {status}; retrying in {delay}s ({transient_retries}/2).")
                    time.sleep(delay)
                    continue

                self.last_request["status"] = "failed"
                raise


class AnalyzerLogCapture(logging.Handler):
    """Detect parser/analyzer fallbacks that production intentionally swallows."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.last_failure: dict[str, str] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Turn analysis failed" in message:
            error_text = ""
            if record.exc_info:
                error_text = self.formatter.formatException(record.exc_info) if self.formatter else str(record.exc_info[1])
            self.last_failure = {
                "type": type(record.exc_info[1]).__name__ if record.exc_info else "AnalyzerError",
                "message": (error_text or message)[:1000],
            }
        elif "TurnAnalyzer returned malformed JSON" in message:
            self.last_failure = {
                "type": "MalformedJSON",
                "message": message,
            }


class ProviderUsageTracker:
    """Capture token usage from the production provider's existing usage hook."""

    def __init__(self, provider_module: Any):
        self.provider = provider_module
        self.original = provider_module._log_usage
        self.successful_usage_records = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cached_tokens = 0

    def install(self) -> None:
        original = self.original
        tracker = self

        def tracked(caller: str, response: Any, elapsed: float) -> None:
            usage = getattr(response, "usage", None)
            if usage is not None:
                tracker.successful_usage_records += 1
                tracker.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                tracker.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                tracker.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
                details = getattr(usage, "prompt_tokens_details", None)
                tracker.cached_tokens += int(getattr(details, "cached_tokens", 0) or 0)
            original(caller, response, elapsed)

        self.provider._log_usage = tracked

    def restore(self) -> None:
        self.provider._log_usage = self.original

    def as_dict(self) -> dict[str, int]:
        return {
            "successful_usage_records": self.successful_usage_records,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_tokens,
        }


def make_result_record(sample: dict[str, Any], prediction: dict[str, Any], request_state: dict[str, Any]) -> dict[str, Any]:
    expected_output = sample["expected_output"]
    expected_episode = expected_output.get("episode")
    expected_mood_label = sample["_expected_mood_label"]
    expected_reflection = sample["_expected_reflection_text"]

    predicted_mood = prediction.get("mood") if isinstance(prediction, dict) else {}
    if not isinstance(predicted_mood, dict):
        predicted_mood = {}
    predicted_mood_label = normalize_mood(predicted_mood.get("label"))

    predicted_reflection = normalize_reflection(prediction.get("reflection") if isinstance(prediction, dict) else None)
    predicted_episode = prediction.get("episode") if isinstance(prediction, dict) else {}
    if not isinstance(predicted_episode, dict):
        predicted_episode = {}

    request_status = request_state.get("status", "failed")
    result_status = "completed" if request_status == "ok" else request_status

    return {
        "sample_id": sample["sample_id"],
        "conversation_id": sample.get("conversation_id"),
        "category": sample.get("category", "uncategorized"),
        "sample_fingerprint": sample["_sample_fingerprint"],
        "status": result_status,
        "request": request_state,
        "expected": {
            "episode": expected_episode,
            "mood": {
                "label": expected_mood_label,
                "evidence": (expected_output.get("mood") or {}).get("evidence"),
            },
            "reflection": expected_reflection,
        },
        "predicted": {
            "episode": predicted_episode,
            "mood": {
                "label": predicted_mood_label,
                "confidence": predicted_mood.get("confidence"),
                "evidence": predicted_mood.get("evidence"),
            },
            "reflection": predicted_reflection,
        },
        "evaluation": {
            "mood_correct": predicted_mood_label == expected_mood_label,
            "reflection_detection_correct": (predicted_reflection is not None) == (expected_reflection is not None),
            "reflection_text_review_required": expected_reflection is not None and predicted_reflection is not None,
            "episode_correct": (
                predicted_episode.get("action") == expected_episode.get("action")
                if isinstance(expected_episode, dict) else None
            ),
            "reflection_quality_score": None,
        } if result_status == "completed" else None,
    }


def build_report(
    dataset_path: Path,
    output_path: Path,
    model: str,
    production_fingerprint: str,
    dataset_items: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    controller: GroqRequestController | None,
    usage: ProviderUsageTracker | None,
    started_at: str,
) -> dict[str, Any]:
    metrics = build_metrics(samples)
    return {
        "benchmark": BENCHMARK_VERSION,
        "model": model,
        "production_code_fingerprint": production_fingerprint,
        "dataset_file": dataset_path.name,
        "dataset_fingerprint": canonical_fingerprint([
            item.get("_sample_fingerprint") for item in dataset_items
        ]),
        "dataset_sample_count": len(dataset_items),
        "recorded_sample_count": len(samples),
        "remaining_sample_count": max(0, len(dataset_items) - len(samples)),
        "started_at_utc": started_at,
        "updated_at_utc": utc_now(),
        "output_file": output_path.name,
        "summary": metrics,
        "groq_usage": {
            "model": model,
            "api_key_switches": controller.key_switches if controller else 0,
            "request_attempts": controller.request_attempts if controller else 0,
            "successful_requests": controller.successful_requests if controller else 0,
            "skipped_requests": controller.skipped_requests if controller else 0,
            "api_key_values_saved": False,
            "token_usage": usage.as_dict() if usage else {},
        },
        "samples": samples,
    }


def production_code_fingerprint(backend_dir: Path) -> str:
    """Fingerprint the exact prompt, analyzer and Groq-provider code being tested."""
    relative_files = (
        Path("services/ai/turn_analyzer.py"),
        Path("services/ai/mood_detector.py"),
        Path("services/ai/prompts/turn_analyzer_prompt.py"),
        Path("services/llm_provider.py"),
    )
    digest = hashlib.sha256()
    for relative_path in relative_files:
        file_path = backend_dir / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Required production file is missing: {file_path}")
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def load_resume_records(
    output_path: Path, expected_model: str, expected_code_fingerprint: str
) -> dict[str, dict[str, Any]]:
    if not output_path.is_file():
        raise FileNotFoundError(f"Cannot resume because result file does not exist: {output_path}")
    previous = load_json(output_path)
    if not isinstance(previous, dict) or not isinstance(previous.get("samples"), list):
        raise ValueError(f"Cannot resume: {output_path} is not a TurnAnalyzer benchmark result file.")
    if previous.get("model") != expected_model:
        raise ValueError(
            f"Cannot resume with model {expected_model!r}; existing results use {previous.get('model')!r}. "
            "Choose a new --output path or use the original model."
        )
    if previous.get("production_code_fingerprint") != expected_code_fingerprint:
        raise ValueError(
            "Cannot resume because TurnAnalyzer, mood_detector, its prompt, or llm_provider.py changed since the previous run. "
            "Choose a new --output path to benchmark the changed code."
        )
    records: dict[str, dict[str, Any]] = {}
    for record in previous["samples"]:
        if isinstance(record, dict) and isinstance(record.get("sample_id"), str):
            records[record["sample_id"]] = record
    return records


def run_benchmark(args: argparse.Namespace) -> int:
    backend_dir = find_backend_dir(args.backend_dir)
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    # Load backend .env before importing llm_provider, because that module
    # creates its initial Groq client at import time.
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install python-dotenv in the backend environment.") from exc
    load_dotenv(backend_dir / ".env", override=False)

    dataset_path = resolve_path(args.dataset, backend_dir)
    output_path = resolve_path(args.output, backend_dir)
    raw_dataset = load_json(dataset_path)
    dataset = validate_dataset(raw_dataset)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer.")
        dataset = dataset[:args.limit]

    if args.validate_only:
        category_counts = Counter(str(item.get("category") or "uncategorized") for item in dataset)
        print(f"Dataset valid: {len(dataset)} samples.")
        print("Category counts:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        print("No Groq API request was made.")
        return 0

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        if not sys.stdin.isatty():
            raise RuntimeError("GROQ_API_KEY is not set and no interactive terminal is available.")
        api_key = getpass.getpass("Groq API key (hidden; not saved): ").strip()
    if not api_key:
        raise RuntimeError("No Groq API key provided.")
    os.environ["GROQ_API_KEY"] = api_key

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("Install the Groq SDK in the backend environment: pip install groq") from exc

    # Import the exact production provider and production TurnAnalyzer.
    import services.llm_provider as llm_provider
    import services.ai.turn_analyzer as turn_analyzer_module

    if args.model:
        llm_provider.MODEL = args.model.strip()
    model = llm_provider.MODEL or DEFAULT_MODEL
    production_fingerprint = production_code_fingerprint(backend_dir)
    llm_provider.client = Groq(api_key=api_key)

    controller = GroqRequestController(llm_provider, Groq)
    controller.model = model
    if not args.skip_key_validation:
        controller.validate_initial_key()

    # Swap only the symbol TurnAnalyzer uses for its provider function. The
    # controller delegates to the real production provider on each attempt.
    original_turn_generate = turn_analyzer_module.generate_response
    turn_analyzer_module.generate_response = controller.generate_response
    usage = ProviderUsageTracker(llm_provider)
    usage.install()

    analyzer = turn_analyzer_module.TurnAnalyzer()
    previous_records = load_resume_records(output_path, model, production_fingerprint) if args.resume else {}
    completed_records: dict[str, dict[str, Any]] = {}
    started_at = utc_now()
    analyzer_log_capture = AnalyzerLogCapture()
    analyzer_logger = logging.getLogger("services.ai.turn_analyzer")
    analyzer_logger.addHandler(analyzer_log_capture)

    try:
        for index, sample in enumerate(dataset, start=1):
            sample_id = sample["sample_id"]
            prior = previous_records.get(sample_id)
            if prior and prior.get("sample_fingerprint") == sample["_sample_fingerprint"]:
                if prior.get("status") == "completed":
                    completed_records[sample_id] = prior
                    print(f"[{index}/{len(dataset)}] {sample_id}: reused completed result (--resume).")
                    continue
                if prior.get("status") in {"failed", "skipped"} and not args.retry_failed:
                    completed_records[sample_id] = prior
                    print(f"[{index}/{len(dataset)}] {sample_id}: retained {prior.get('status')} result; use --retry-failed to retry.")
                    continue

            print(f"\n[{index}/{len(dataset)}] Running {sample_id} ({sample.get('category', 'uncategorized')})...")
            try:
                analyzer_inputs = prepare_analyzer_inputs(sample)
                controller.last_request = {"status": "running", "attempts": 0}
                controller.last_raw_response = None
                analyzer_log_capture.last_failure = None
                prediction = analyzer.analyze(**analyzer_inputs)
                request_state = dict(controller.last_request)
                if request_state.get("status") == "ok" and not is_json_object_response(controller.last_raw_response):
                    request_state["status"] = "failed"
                    request_state["error"] = {
                        "type": "InvalidJSONResponse",
                        "message": "Groq completion did not contain a parseable JSON object.",
                    }
                if analyzer_log_capture.last_failure is not None:
                    request_state["status"] = "failed"
                    request_state["error"] = analyzer_log_capture.last_failure
                if request_state.get("status") == "ok":
                    record = make_result_record(sample, prediction, request_state)
                    print(
                        f"  Mood: expected={record['expected']['mood']['label']!r}, "
                        f"predicted={record['predicted']['mood']['label']!r} | "
                        f"Reflection: expected={'yes' if record['expected']['reflection'] else 'no'}, "
                        f"predicted={'yes' if record['predicted']['reflection'] else 'no'}"
                    )
                elif request_state.get("status") == "skipped":
                    record = make_result_record(sample, prediction, request_state)
                    print("  Request skipped; sample excluded from metrics.")
                else:
                    record = make_result_record(sample, prediction, request_state)
                    record["status"] = "failed"
                    print(f"  Request failed ({request_state.get('error', {}).get('type', 'unknown')}); sample excluded from metrics.")
            except KeyboardInterrupt:
                print("\nBenchmark interrupted by user. Saving completed samples...")
                break
            except Exception as exc:
                request_state = dict(controller.last_request)
                request_state.setdefault("status", "failed")
                request_state["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:1000],
                }
                record = make_result_record(sample, {}, request_state)
                record["status"] = "failed"
                print(f"  Sample failed: {type(exc).__name__}: {exc}")

            completed_records[sample_id] = record
            ordered_records = [completed_records[s["sample_id"]] for s in dataset if s["sample_id"] in completed_records]
            report = build_report(dataset_path, output_path, model, production_fingerprint, dataset, ordered_records, controller, usage, started_at)
            write_json(output_path, report)
            print(f"  Saved progress: {output_path}")

    finally:
        analyzer_logger.removeHandler(analyzer_log_capture)
        turn_analyzer_module.generate_response = original_turn_generate
        usage.restore()

    ordered_records = [completed_records[s["sample_id"]] for s in dataset if s["sample_id"] in completed_records]
    report = build_report(dataset_path, output_path, model, production_fingerprint, dataset, ordered_records, controller, usage, started_at)
    write_json(output_path, report)

    print("\n========== TURN ANALYZER BENCHMARK SUMMARY ==========")
    summary = report["summary"]
    print(f"Samples: {summary['total_samples']} | completed: {summary['completed_samples']} | failed: {summary['failed_samples']} | skipped: {summary['skipped_samples']}")
    mood = summary["mood"]
    print(f"Mood accuracy: {mood['accuracy'] if mood['accuracy'] is not None else 'n/a'} | Macro F1: {mood['macro_f1'] if mood['macro_f1'] is not None else 'n/a'}")
    reflection = summary["reflection_detection"]
    print(f"Reflection detection: precision={reflection['precision']} recall={reflection['recall']} F1={reflection['f1']}")
    print(f"Groq requests: {report['groq_usage']['successful_requests']} successful, {report['groq_usage']['api_key_switches']} key switches")
    print(f"Token usage: {report['groq_usage']['token_usage']}")
    print(f"Results: {output_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the production TurnAnalyzer with a labelled conversation JSON array."
    )
    parser.add_argument("--dataset", required=True, help="JSON array of samples with input and expected_output.")
    parser.add_argument("--backend-dir", help="Backend directory containing services/ai/turn_analyzer.py.")
    parser.add_argument("--output", default="benchmark/results/turn_analyzer_results.json", help="Output JSON path (relative to backend unless absolute).")
    parser.add_argument("--model", help="Override GROQ_MODEL for this benchmark run; default comes from backend .env/provider.")
    parser.add_argument("--limit", type=int, help="Run only the first N dataset samples (useful for a small smoke test).")
    parser.add_argument("--resume", action="store_true", help="Reuse successful matching sample results from the existing output file.")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, rerun previously failed/skipped samples.")
    parser.add_argument("--skip-key-validation", action="store_true", help="Skip the initial models.list() key check; the first completion will validate the key.")
    parser.add_argument("--validate-only", action="store_true", help="Validate dataset schema and production-context consistency without calling Groq.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_benchmark(args)
    except KeyboardInterrupt:
        print("Benchmark stopped.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
