#!/usr/bin/env python3
"""Benchmark the production TurnAnalyzer episode-decision path.

Run from the backend directory:
    python benchmark/episode_detection_benchmark.py

Optional environment variables:
    GROQ_API_KEY       Primary Groq key
    GROQ_API_KEYS      Additional keys separated by commas, semicolons, or newlines
    GROQ_MODEL         Model override (defaults to the production provider default)

On a Groq rate-limit/quota response, the runner rotates to an unused configured
key. When no configured keys remain, it prompts for another key using hidden
terminal input. Keys are held in memory only and are never written to reports.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import time
import types
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = SCRIPT_DIR / "datasets" / "episode_detector_benchmark_50.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "episode_detection_benchmark_results.json"
ACTIONS = ("CREATE", "NONE", "CONTINUE", "CLOSE")


def load_environment() -> None:
    """Load the backend .env before importing the project's provider modules."""
    try:
        from dotenv import load_dotenv

        load_dotenv(BACKEND_ROOT / ".env", override=False)
    except ImportError:
        # The backend normally depends on python-dotenv. Environment variables
        # can still be supplied directly when it is not installed.
        pass


def parse_key_list() -> list[str]:
    values: list[str] = []
    primary = os.getenv("GROQ_API_KEY", "").strip()
    if primary:
        values.append(primary)

    extra = os.getenv("GROQ_API_KEYS", "")
    for item in re.split(r"[,;\n\r]+", extra):
        item = item.strip()
        if item:
            values.append(item)

    # Keep order while removing duplicates.
    return list(dict.fromkeys(values))


def acquire_initial_key(keys: list[str]) -> list[str]:
    if keys:
        return keys
    if not sys.stdin.isatty():
        raise RuntimeError(
            "No Groq API key found. Set GROQ_API_KEY or run interactively to enter one."
        )
    entered = getpass.getpass("Enter Groq API key (input hidden): ").strip()
    if not entered:
        raise RuntimeError("No API key supplied; benchmark was not started.")
    return [entered]


def is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").lower()
    message = str(exc).lower()
    if status == 429:
        return True
    if code in {"rate_limit_exceeded", "tokens_per_minute", "requests_per_minute"}:
        return True
    markers = (
        "rate limit",
        "rate_limit",
        "quota exceeded",
        "daily token limit",
        "tokens per minute",
        "requests per minute",
        "too many requests",
    )
    return any(marker in message for marker in markers)


def safe_error_text(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    parts = [type(exc).__name__]
    if status is not None:
        parts.append(f"HTTP {status}")
    if code:
        parts.append(f"code={code}")
    message = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[redacted]", str(exc))
    message = re.sub(r"gsk_[A-Za-z0-9_-]+", "[redacted-key]", message)
    if message:
        parts.append(message[:500])
    return " | ".join(parts)


class RotatingGroqGenerator:
    """Drop-in replacement for services.llm_provider.generate_response."""

    def __init__(self, api_keys: list[str], model: str):
        from groq import Groq

        self.Groq = Groq
        self.keys = list(api_keys)
        self.model = model
        self.current_index = 0
        self.exhausted: set[int] = set()
        self.clients: dict[int, Any] = {}
        self.last_raw_response: str | None = None
        self.last_error: str | None = None
        self.last_usage: dict[str, int] = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.rotation_count = 0

    def _client(self, index: int):
        if index not in self.clients:
            self.clients[index] = self.Groq(api_key=self.keys[index])
        return self.clients[index]

    def _next_unused_key(self) -> bool:
        for index in range(len(self.keys)):
            if index not in self.exhausted:
                self.current_index = index
                return True
        return False

    def _request_another_key(self) -> bool:
        if not sys.stdin.isatty():
            return False
        print("All configured Groq keys have hit a rate/quota limit.")
        entered = getpass.getpass("Paste another Groq API key (hidden; Enter to stop): ").strip()
        if not entered:
            return False
        if entered in self.keys:
            print("That key has already been used. Enter a different key.")
            return self._request_another_key()
        self.keys.append(entered)
        self.current_index = len(self.keys) - 1
        self.rotation_count += 1
        print(f"Added key {self.current_index + 1}; retrying the same sample.")
        return True

    @staticmethod
    def _build_messages(contents: Any, system_instruction: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        else:
            messages.extend(contents or [])
        return messages

    def __call__(
        self,
        contents,
        system_instruction=None,
        *,
        max_completion_tokens=1024,
        response_format=None,
        reasoning_effort="low",
    ) -> str:
        self.last_raw_response = None
        self.last_error = None
        self.last_usage = {}
        messages = self._build_messages(contents, system_instruction)

        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_completion_tokens": max_completion_tokens,
            "top_p": 1,
            "reasoning_effort": reasoning_effort,
        }
        if response_format is not None:
            request_params["response_format"] = response_format

        while True:
            if self.current_index in self.exhausted and not self._next_unused_key():
                if not self._request_another_key():
                    raise KeyboardInterrupt("No further API key supplied; stopping benchmark safely.")

            index = self.current_index
            started = time.perf_counter()
            try:
                response = self._client(index).chat.completions.create(**request_params)
                if not response.choices:
                    raise RuntimeError("Groq returned no completion choices")

                choice = response.choices[0]
                content = choice.message.content or ""
                if choice.finish_reason == "length":
                    raise RuntimeError(
                        f"Groq response was truncated at {max_completion_tokens} completion tokens"
                    )

                usage = getattr(response, "usage", None)
                if usage is not None:
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.last_usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": int(
                            getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
                            or (prompt_tokens + completion_tokens)
                        ),
                    }
                self.total_requests += 1
                self.last_raw_response = content
                self.last_usage["duration_seconds"] = round(time.perf_counter() - started, 3)
                return content

            except Exception as exc:
                if is_rate_limit_error(exc):
                    self.exhausted.add(index)
                    self.rotation_count += 1
                    print(
                        f"\n[Rate limit] Key {index + 1} reached its limit. "
                        "Switching key and retrying this sample..."
                    )
                    if self._next_unused_key():
                        continue
                    if self._request_another_key():
                        continue
                    raise KeyboardInterrupt(
                        "No replacement API key supplied after rate limit."
                    ) from None

                self.last_error = safe_error_text(exc)
                raise


def parse_raw_json(response: str | None) -> dict[str, Any]:
    if not isinstance(response, str) or not response.strip():
        return {}
    content = response.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    start = content.find("{")
    if start < 0:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(content[start:])
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def make_episode_object(active: dict[str, Any] | None):
    if active is None:
        return None
    # The benchmark dataset deliberately stores only episode_id and title.
    # These defaults mirror the database invariant for an active episode.
    return SimpleNamespace(
        episode_id=active.get("episode_id", "BENCHMARK_ACTIVE_EPISODE"),
        title=active.get("title") or "Untitled",
        status=active.get("status") or "ACTIVE",
        summary=active.get("summary"),
    )


def make_history_objects(items: Any, field: str) -> list[SimpleNamespace]:
    result = []
    for item in (items or [])[-5:]:
        if isinstance(item, str):
            result.append(SimpleNamespace(**{field: item, "created_at": None}))
        elif isinstance(item, dict):
            result.append(
                SimpleNamespace(
                    **{
                        field: item.get(field),
                        "created_at": item.get("created_at"),
                    }
                )
            )
    return result


def apply_orchestrator_action(action: str, active_episode) -> tuple[str, bool, str | None]:
    """Mirror AIOrchestrator._prepare_episode's post-analyzer safeguards."""
    if active_episode is None and action == "CREATE":
        return "CREATE", True, None
    if active_episode is not None and action not in {"CONTINUE", "CLOSE"}:
        return "CONTINUE", False, active_episode.episode_id
    if active_episode is None and action != "CREATE":
        return "NONE", False, None
    return action, False, active_episode.episode_id if active_episode is not None else None


def per_class_metrics(expected: list[str], predicted: list[str]) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for label in ACTIONS:
        tp = sum(1 for y, p in zip(expected, predicted) if y == label and p == label)
        fp = sum(1 for y, p in zip(expected, predicted) if y != label and p == label)
        fn = sum(1 for y, p in zip(expected, predicted) if y == label and p != label)
        support = sum(1 for y in expected if y == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return report


def build_metrics(results: list[dict[str, Any]], total_samples: int) -> dict[str, Any]:
    completed = [r for r in results if r.get("status") == "completed" and r.get("predicted_action") in ACTIONS]
    expected = [r["expected_action"] for r in completed]
    predicted = [r["predicted_action"] for r in completed]
    correct = sum(1 for y, p in zip(expected, predicted) if y == p)
    class_report = per_class_metrics(expected, predicted) if completed else {}
    macro_f1 = (
        sum(float(class_report[label]["f1"]) for label in ACTIONS) / len(ACTIONS)
        if class_report else 0.0
    )

    confusion = {actual: {pred: 0 for pred in ACTIONS} for actual in ACTIONS}
    for actual, pred in zip(expected, predicted):
        confusion[actual][pred] += 1

    raw_valid = [r for r in completed if r.get("raw_model_action") in ACTIONS]
    raw_correct = sum(
        1 for r in raw_valid if r["raw_model_action"] == r["expected_action"]
    )
    by_difficulty: dict[str, dict[str, int | float]] = {}
    by_category: dict[str, dict[str, int | float]] = {}
    for dimension in ("difficulty", "category"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in completed:
            buckets[str(row.get(dimension, "unknown"))].append(row)
        output = {}
        for name, rows in sorted(buckets.items()):
            hits = sum(r["predicted_action"] == r["expected_action"] for r in rows)
            output[name] = {
                "correct": hits,
                "total": len(rows),
                "accuracy": round(hits / len(rows), 4) if rows else 0.0,
            }
        if dimension == "difficulty":
            by_difficulty = output
        else:
            by_category = output

    latencies = [
        float(r["usage"].get("duration_seconds", 0))
        for r in completed if isinstance(r.get("usage"), dict)
    ]
    return {
        "total_samples": total_samples,
        "completed_samples": len(completed),
        "failed_samples": sum(1 for r in results if r.get("status") != "completed"),
        "accuracy": round(correct / len(completed), 4) if completed else None,
        "correct": correct,
        "macro_f1": round(macro_f1, 4) if completed else None,
        "raw_model_action_accuracy": round(raw_correct / len(raw_valid), 4) if raw_valid else None,
        "raw_model_action_samples": len(raw_valid),
        "per_action": class_report,
        "confusion_matrix": confusion,
        "accuracy_by_difficulty": by_difficulty,
        "accuracy_by_category": by_category,
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
    }


def save_report(path: Path, results: list[dict[str, Any]], total_samples: int, model: str, rotator) -> None:
    report = {
        "benchmark": "TurnAnalyzer episode lifecycle benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source_logic": {
            "prompt": "services.ai.prompts.turn_analyzer_prompt.PROMPT",
            "analyzer": "services.ai.turn_analyzer.TurnAnalyzer.analyze/build_prompt/_validate_episode",
            "orchestrator": "AIOrchestrator._prepare_episode post-analyzer safeguards",
        },
        "metrics": build_metrics(results, total_samples),
        "api_usage": {
            "successful_requests": rotator.total_requests,
            "prompt_tokens": rotator.total_prompt_tokens,
            "completion_tokens": rotator.total_completion_tokens,
            "total_tokens": rotator.total_prompt_tokens + rotator.total_completion_tokens,
            "key_rotations": rotator.rotation_count,
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark production TurnAnalyzer episode decisions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Input benchmark JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Detailed results JSON")
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N samples")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between successful calls")
    args = parser.parse_args()

    if not args.dataset.is_absolute():
        args.dataset = (Path.cwd() / args.dataset).resolve()
    if not args.output.is_absolute():
        args.output = (Path.cwd() / args.output).resolve()
    if not args.dataset.is_file():
        parser.error(f"Dataset not found: {args.dataset}")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        parser.error("Dataset must be a non-empty JSON array.")
    if args.limit is not None:
        dataset = dataset[: max(args.limit, 0)]
    if not dataset:
        parser.error("No samples selected.")

    for sample in dataset:
        if not isinstance(sample, dict) or not isinstance(sample.get("input"), dict):
            parser.error("Each dataset item must contain an input object.")
        if sample.get("ground_truth", {}).get("action") not in ACTIONS:
            parser.error(f"Invalid ground-truth action in {sample.get('sample_id', 'unknown sample')}.")

    load_environment()
    keys = acquire_initial_key(parse_key_list())
    # services.llm_provider constructs a module-level Groq client at import time.
    # Set a valid first key before importing TurnAnalyzer, then replace only the
    # generate_response symbol used by TurnAnalyzer with the rotating wrapper.
    os.environ["GROQ_API_KEY"] = keys[0]
    os.environ["GROQ_MODEL"] = args.model
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))

    # TurnAnalyzer imports MOOD_LABELS/MIN_CONFIDENCE from mood_detector.py,
    # which in this project imports `models` and initializes the database
    # engine as a module side effect. The decision benchmark never reads or
    # writes ORM rows, so provide only the EpisodeMood symbol required at import
    # time. The real mood_detector module and its constants are still imported.
    if "models" not in sys.modules:
        models_import_shim = types.ModuleType("models")
        models_import_shim.EpisodeMood = object
        sys.modules["models"] = models_import_shim

    try:
        from services.ai import turn_analyzer as turn_analyzer_module
    except Exception as exc:
        raise RuntimeError(
            "Could not import the backend TurnAnalyzer. Run this script from the "
            "backend project with its requirements installed."
        ) from exc

    rotator = RotatingGroqGenerator(keys, args.model)
    turn_analyzer_module.generate_response = rotator
    analyzer = turn_analyzer_module.TurnAnalyzer()

    results: list[dict[str, Any]] = []
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(dataset)} | Model: {args.model} | Configured keys: {len(keys)}")
    print("Using the backend's TurnAnalyzer prompt, context construction, and validators.\n")

    try:
        for index, sample in enumerate(dataset, start=1):
            sample_input = sample["input"]
            active_episode = make_episode_object(sample_input.get("active_episode"))
            previous_moods = make_history_objects(sample_input.get("previous_moods"), "mood")
            previous_reflections = make_history_objects(sample_input.get("previous_reflections"), "reflection")

            # Call the same production analyzer method used by AIOrchestrator.
            decisions = analyzer.analyze(
                active_episode=active_episode,
                latest_message=sample_input.get("latest_message", ""),
                conversation=sample_input.get("recent_context") or [],
                previous_moods=previous_moods,
                previous_reflections=previous_reflections,
            )

            raw_payload = parse_raw_json(rotator.last_raw_response)
            raw_episode = raw_payload.get("episode") if isinstance(raw_payload, dict) else None
            raw_action = (
                str(raw_episode.get("action", "")).strip().upper()
                if isinstance(raw_episode, dict) else None
            )
            if raw_action not in ACTIONS:
                raw_action = None

            analyzer_action = decisions.get("episode", {}).get("action", "NONE")
            orchestrator_action, would_create, resulting_episode_id = apply_orchestrator_action(
                analyzer_action, active_episode,
            )
            usage = dict(rotator.last_usage)
            call_error = rotator.last_error

            row = {
                "sample_id": sample.get("sample_id", f"S{index:03d}"),
                "category": sample.get("category"),
                "difficulty": sample.get("difficulty"),
                "status": "failed" if call_error else "completed",
                "expected_action": sample["ground_truth"]["action"],
                "raw_model_action": raw_action,
                "turn_analyzer_action": analyzer_action,
                "predicted_action": orchestrator_action if not call_error else None,
                "expected_title": sample["ground_truth"].get("title"),
                "predicted_title": decisions.get("episode", {}).get("title"),
                "reason": decisions.get("episode", {}).get("reason"),
                "confidence": decisions.get("episode", {}).get("confidence"),
                "would_create_episode": would_create,
                "resulting_episode_id": resulting_episode_id,
                "mood": decisions.get("mood"),
                "reflection": decisions.get("reflection"),
                "usage": usage,
                "error": call_error,
                "raw_response": (rotator.last_raw_response or "")[:3000],
            }
            results.append(row)
            save_report(args.output, results, len(dataset), args.model, rotator)

            if call_error:
                print(f"[{index:02d}/{len(dataset):02d}] {row['sample_id']} API ERROR: {call_error}")
                print("Stopping after the provider error. Completed results have been saved.")
                break

            marker = "OK" if orchestrator_action == row["expected_action"] else "MISS"
            latency = usage.get("duration_seconds", "?")
            print(
                f"[{index:02d}/{len(dataset):02d}] {row['sample_id']} "
                f"{marker} expected={row['expected_action']} predicted={orchestrator_action} "
                f"raw={raw_action or 'INVALID'} time={latency}s"
            )
            if args.delay > 0 and index < len(dataset):
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\nBenchmark stopped. Saving completed samples...")
    finally:
        save_report(args.output, results, len(dataset), args.model, rotator)

    metrics = build_metrics(results, len(dataset))
    print("\n========== EPISODE BENCHMARK SUMMARY ==========")
    print(f"Completed: {metrics['completed_samples']}/{metrics['total_samples']}")
    if metrics["accuracy"] is not None:
        print(f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Raw model action accuracy: {metrics['raw_model_action_accuracy']}")
        print("Per-action metrics:")
        for action in ACTIONS:
            item = metrics["per_action"][action]
            print(
                f"  {action:8s} P={item['precision']:.3f} "
                f"R={item['recall']:.3f} F1={item['f1']:.3f} "
                f"support={item['support']}"
            )
    print(
        f"Tokens: prompt={rotator.total_prompt_tokens}, "
        f"completion={rotator.total_completion_tokens}, "
        f"key rotations={rotator.rotation_count}"
    )
    print(f"Detailed results: {args.output}")
    return 0 if metrics["completed_samples"] == len(dataset) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
