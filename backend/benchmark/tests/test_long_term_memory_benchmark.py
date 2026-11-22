import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from benchmark.long_term_memory_benchmark import (
    DEFAULT_DATASET,
    BenchmarkAborted,
    build_judge_prompt,
    call_with_recovery,
    dataset_fingerprint,
    evaluate_rules,
    is_key_or_quota_error,
    normalize_judge_result,
    parse_json_response,
    production_input,
    rotate_backend_key,
    run_benchmark,
    token_similarity,
    validate_dataset,
)


class LongTermMemoryBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Stub only external integration boundaries. The production ORM and
        # LongTermMemory Python source are still imported from the real backend.
        groq_stub = types.ModuleType("groq")
        class StubGroq:
            def __init__(self, api_key=None, **_kwargs):
                self.api_key = api_key
        groq_stub.Groq = StubGroq
        sys.modules["groq"] = groq_stub

        from sqlalchemy.orm import declarative_base
        database_stub = types.ModuleType("database")
        database_stub.Base = declarative_base()
        sys.modules["database"] = database_stub

        for name in list(sys.modules):
            if name == "models" or name.startswith("services"):
                del sys.modules[name]

        cls.ltm_module = importlib.import_module("services.ai.long_term_memory")
        cls.models = importlib.import_module("models")

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name in {"models", "database", "groq"} or name.startswith("services"):
                del sys.modules[name]

    def setUp(self):
        self.dataset = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
        validate_dataset(self.dataset)

    def case(self, sample_id):
        return next(item for item in self.dataset["samples"] if item["sample_id"] == sample_id)

    @staticmethod
    def make_evaluation_sample(
        *, category="insufficient_evidence_noise", previous_memory="", expected_memory="",
        mood_observations=None, reflections=None, validated_patterns=None,
    ):
        return {
            "sample_id": "LTM_TEST",
            "category": category,
            "validated_patterns": validated_patterns or [],
            "current_episode": {"episode_id": "EP_TEST", "summary": "A synthetic test episode."},
            "mood_observations": mood_observations or [],
            "reflections": reflections or [],
            "previous_memory": previous_memory,
            "expected": {"memory": expected_memory},
        }

    def test_dataset_has_exact_distribution_and_valid_shapes(self):
        self.assertEqual(self.dataset["total_samples"], 10)
        self.assertEqual(self.dataset["benchmark_name"], "Long-Term Memory Benchmark")
        self.assertEqual(self.dataset["version"], "1.2")
        self.assertNotIn("cases", self.dataset)
        self.assertEqual(len(self.dataset["samples"]), 10)
        validate_dataset(self.dataset)

    def test_original_schema_does_not_require_created_at(self):
        # Explicitly remove optional timestamps and prove the flat schema remains valid.
        import copy
        without_timestamps = copy.deepcopy(self.dataset)
        for sample in without_timestamps["samples"]:
            for mood in sample["mood_observations"]:
                mood.pop("created_at", None)
            for reflection in sample["reflections"]:
                reflection.pop("created_at", None)
        validate_dataset(without_timestamps)

    def test_validator_rejects_new_cases_input_schema(self):
        invalid = {
            "benchmark_name": "Long-Term Memory Benchmark",
            "version": "1.2",
            "total_samples": 10,
            "cases": [],
        }
        with self.assertRaises(ValueError):
            validate_dataset(invalid)

    def test_parse_json_accepts_plain_json_and_fence(self):
        self.assertEqual(parse_json_response('{"memory": ""}'), {"memory": ""})
        self.assertEqual(parse_json_response('```json\n{"memory": "- fact"}\n```'), {"memory": "- fact"})

    def test_parse_json_rejects_invalid_json(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_json_response("not json")

    def test_similarity_understands_delay_paraphrases(self):
        score = token_similarity(
            "The user repeatedly postpones starting important work.",
            "The user has a recurring tendency to procrastinate on important tasks.",
        )
        self.assertGreaterEqual(score, 0.38)

    def test_similarity_rejects_explicit_negation_conflict(self):
        score = token_similarity(
            "The user prefers working late at night.",
            "The user prefers early mornings and does not prefer late-night work.",
        )
        self.assertEqual(score, 0.0)

    def test_correction_rule_detects_replaced_stale_fact(self):
        case = self.make_evaluation_sample(
            category="memory_correction",
            previous_memory="- The user is a violinist.",
            expected_memory="- The user owns a green bicycle.",
        )
        result = evaluate_rules(case, case["expected"]["memory"])
        self.assertEqual(result["fact_recall"], 1.0)
        self.assertEqual(result["stale_previous_facts"], 1)
        self.assertEqual(result["stale_previous_facts_omitted"], 1)
        self.assertEqual(result["correction_accuracy"], 1.0)

    def test_empty_expected_memory_penalizes_new_claims(self):
        case = self.make_evaluation_sample(expected_memory="", previous_memory="")
        empty_result = evaluate_rules(case, "")
        hallucinated_result = evaluate_rules(case, "- The user has a stable anxiety condition.")
        self.assertEqual(empty_result["fact_precision"], 1.0)
        self.assertEqual(hallucinated_result["fact_precision"], 0.0)
        self.assertEqual(hallucinated_result["hallucination_rate_proxy"], 1.0)

    def test_failed_generation_is_not_given_factuality_scores(self):
        case = self.make_evaluation_sample(expected_memory="", previous_memory="")
        result = evaluate_rules(case, None, generation_status="failed")
        self.assertIsNone(result["fact_precision"])
        self.assertIsNone(result["fact_recall"])
        self.assertIsNone(result["heuristic_composite"])

    def test_single_episode_mood_is_not_promoted_to_memory(self):
        case = self.make_evaluation_sample(
            expected_memory="",
            previous_memory="",
            mood_observations=[{"mood": "fear", "confidence": 0.9, "evidence": "I feel anxious."}],
        )
        result = evaluate_rules(case, "- The user is anxious and has a stable anxiety problem.")
        self.assertEqual(result["mood_handling_accuracy"], 0.0)
        self.assertGreaterEqual(len(result["transient_mood_trait_leaks"]), 1)

    def test_reflection_supported_expected_fact_is_counted(self):
        case = self.make_evaluation_sample(
            expected_memory="- The user finds short walks helpful after work.",
            reflections=[{"reflection": "The user finds short walks helpful after work."}],
        )
        result = evaluate_rules(case, case["expected"]["memory"])
        self.assertEqual(result["reflection_utilization"], 1.0)

    def test_resume_requires_existing_checkpoint(self):
        provider, fake_module, orm_models = self._fake_runner_dependencies()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_benchmark(
                    self.dataset,
                    run_dir=Path(directory) / "missing-run",
                    provider=provider,
                    ltm_module=fake_module,
                    orm_models=orm_models,
                    resume=True,
                )

    def test_judge_result_schema_validation(self):
        valid = {
            "scores": {
                name: 4 for name in (
                    "factual_accuracy", "completeness", "retention", "correction_handling",
                    "pattern_handling", "conservatism", "format_compliance", "overall",
                )
            },
            "unsupported_claims": [],
            "missed_expected_facts": [],
            "retained_or_corrected_facts": [],
            "rationale": "Supported and concise.",
        }
        normalized = normalize_judge_result(json.dumps(valid))
        self.assertEqual(normalized["scores"]["overall"], 4.0)
        valid["scores"]["overall"] = 6
        with self.assertRaises(ValueError):
            normalize_judge_result(json.dumps(valid))
        valid["scores"]["overall"] = 4
        del valid["unsupported_claims"]
        with self.assertRaises(ValueError):
            normalize_judge_result(json.dumps(valid))

    def test_judge_prompt_contains_sample_and_expected_memory(self):
        sample = self.dataset["samples"][0]
        prompt = build_judge_prompt(sample, "- The user prefers morning study.")
        self.assertIn(sample["sample_id"], prompt)
        self.assertIn("expected_memory", prompt)
        self.assertIn("generated_memory", prompt)

    def test_user_provided_flat_sample_shape_is_accepted(self):
        import copy
        sample = copy.deepcopy(self.dataset["samples"][0])
        sample.update({
            "sample_id": "LTM_SCHEMA_CHECK",
            "category": "memory_consolidation",
            "validated_patterns": [{
                "pattern_id": "PAT_001",
                "category": "behavior",
                "description": "The user repeatedly postpones academic tasks, resulting in deadline-related anxiety.",
                "confidence": 0.92,
                "evidence": [
                    {"episode_id": "EP_001", "quote": "The user repeatedly postponed starting the assignment.", "reason": "Evidence of task avoidance."},
                    {"episode_id": "EP_003", "quote": "The user delayed starting the project until the deadline was close.", "reason": "Shows recurrence of task avoidance."},
                ],
                "status": "ACTIVE",
                "resolution_evidence": None,
            }],
            "current_episode": {
                "episode_id": "EP_003",
                "summary": "The user postponed starting a project until the deadline was close. They reported feeling anxious and guilty about delaying the work.",
            },
            "mood_observations": [{
                "mood": "fear", "confidence": 0.86,
                "evidence": "I am anxious about the approaching deadline.",
            }],
            "reflections": [{
                "reflection": "The user recognizes that postponing tasks increases deadline-related stress."
            }],
            "previous_memory": "- The user values academic achievement and completing responsibilities.",
            "expected": {"memory": "- The user values academic achievement and completing responsibilities.\n- The user has repeatedly reported difficulty initiating academic tasks.\n- Delaying academic tasks has been associated with deadline-related anxiety."},
        })
        dataset_with_example = copy.deepcopy(self.dataset)
        dataset_with_example["samples"][0] = sample
        validate_dataset(dataset_with_example)
        db, episode = production_input(sample, self.models.EpisodeMood, self.models.EpisodeReflection)
        self.assertEqual(episode.episode_id, "EP_003")
        self.assertIn(self.models.EpisodeMood, db._rows)
        self.assertEqual(db._rows[self.models.EpisodeMood][0].created_at, None)

    def test_key_quota_error_detection(self):
        for status in (401, 403, 429):
            with self.subTest(status=status):
                error = RuntimeError("Groq request rejected")
                error.status_code = status
                self.assertTrue(is_key_or_quota_error(error))
        self.assertTrue(is_key_or_quota_error(RuntimeError("insufficient_quota")))
        self.assertFalse(is_key_or_quota_error(ValueError("invalid output format")))

    def test_real_rotation_helper_replaces_provider_client(self):
        provider = types.SimpleNamespace(client=None)
        rotate_backend_key(provider, "replacement-key")
        self.assertEqual(provider.client.api_key, "replacement-key")

    def test_dataset_fingerprint_is_stable_and_sensitive_to_changes(self):
        import copy
        original = self.dataset
        reordered = {key: original[key] for key in reversed(list(original))}
        self.assertEqual(dataset_fingerprint(original), dataset_fingerprint(reordered))
        changed = copy.deepcopy(original)
        changed["samples"][0]["expected"]["memory"] += "\n- changed fact"
        self.assertNotEqual(dataset_fingerprint(original), dataset_fingerprint(changed))

    def test_key_rotation_retries_failed_operation(self):
        class FakeProvider:
            current_key = "old"
        provider = FakeProvider()
        calls = {"n": 0}
        def operation():
            calls["n"] += 1
            if provider.current_key == "old":
                error = RuntimeError("quota exceeded")
                error.status_code = 429
                raise error
            return "ok"
        def rotate(_provider, key):
            provider.current_key = key
        result = call_with_recovery(
            operation,
            description="test",
            provider=provider,
            prompt_for_key=lambda _prompt: "new-key",
            rotate_key=rotate,
            sleep=lambda _delay: None,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 2)

    def test_production_memory_method_builds_prompt_and_returns_generated_memory(self):
        case = next(sample for sample in self.dataset["samples"] if sample["validated_patterns"])
        db, episode = production_input(case, self.models.EpisodeMood, self.models.EpisodeReflection)
        captured = {}
        response = json.dumps({"memory": case["expected"]["memory"]})
        with patch.object(self.ltm_module, "generate_response", side_effect=lambda prompt, **kwargs: captured.update({"prompt": prompt, "kwargs": kwargs}) or response):
            instance = self.ltm_module.LongTermMemory.__new__(self.ltm_module.LongTermMemory)
            generated = instance._generate_memory(
                db=db,
                episode=episode,
                previous_memory=case["previous_memory"],
                validated_patterns=case["validated_patterns"],
            )
        self.assertEqual(generated, case["expected"]["memory"])
        self.assertIn("VALIDATED PATTERNS", captured["prompt"])
        self.assertIn(case["validated_patterns"][0]["pattern_id"], captured["prompt"])
        self.assertEqual(captured["kwargs"]["max_completion_tokens"], 2048)
        self.assertEqual(captured["kwargs"]["reasoning_effort"], "low")
        self.assertEqual(captured["kwargs"]["response_format"], {"type": "json_object"})

    def test_production_memory_rejects_more_than_eight_bullets(self):
        memory = "\n".join(f"- fact {index}" for index in range(9))
        with self.assertRaises(ValueError):
            self.ltm_module.LongTermMemory._validate_memory_shape(memory)

    def test_production_memory_rejects_non_bullet_line(self):
        with self.assertRaises(ValueError):
            self.ltm_module.LongTermMemory._validate_memory_shape("A heading\n- a fact")

    def _fake_runner_dependencies(self, fail_judge=False):
        dataset = self.dataset
        memory_by_episode = {
            case["current_episode"]["episode_id"]: case["expected"]["memory"]
            for case in dataset["samples"]
        }
        judge_payload = {
            "scores": {
                name: 4 for name in (
                    "factual_accuracy", "completeness", "retention", "correction_handling",
                    "pattern_handling", "conservatism", "format_compliance", "overall",
                )
            },
            "unsupported_claims": [],
            "missed_expected_facts": [],
            "retained_or_corrected_facts": [],
            "rationale": "synthetic test judge",
        }

        class FakeProvider:
            MODEL = "openai/gpt-oss-120b-test"
            def __init__(self):
                self.calls = 0
            def generate_response(self, contents, **kwargs):
                self.calls += 1
                if isinstance(contents, str) and contents.startswith("GEN:"):
                    episode_id = contents.split(":", 1)[1]
                    return json.dumps({"memory": memory_by_episode[episode_id]})
                if fail_judge:
                    error = RuntimeError("429 Too Many Requests")
                    error.status_code = 429
                    raise error
                return json.dumps(judge_payload)

        class FakeLTM:
            def _generate_memory(_self, db, episode, previous_memory, validated_patterns):
                raw = fake_module.generate_response(f"GEN:{episode.episode_id}")
                return json.loads(raw)["memory"]

        fake_provider = FakeProvider()
        fake_module = types.SimpleNamespace(LongTermMemory=FakeLTM, generate_response=None)
        class DummyMood:
            pass
        class DummyReflection:
            pass
        return fake_provider, fake_module, (DummyMood, DummyReflection)

    def test_full_runner_writes_separate_outputs_and_resume_skips_completed_work(self):
        provider, fake_module, orm_models = self._fake_runner_dependencies()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            state = run_benchmark(
                self.dataset,
                run_dir=run_dir,
                provider=provider,
                ltm_module=fake_module,
                orm_models=orm_models,
                prompt_for_key=lambda _prompt: "new-key",
            )
            self.assertEqual(len(state["generated"]), 10)
            self.assertEqual(len(state["judged"]), 10)
            self.assertEqual(json.loads((run_dir / "generated_memories.json").read_text())["results"].__len__(), 10)
            self.assertTrue((run_dir / "rule_based_results.json").exists())
            self.assertTrue((run_dir / "llm_judge_results.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "report.txt").exists())
            calls_after_first_run = provider.calls

            resumed = run_benchmark(
                self.dataset,
                run_dir=run_dir,
                provider=provider,
                ltm_module=fake_module,
                orm_models=orm_models,
                resume=True,
                prompt_for_key=lambda _prompt: "new-key",
            )
            self.assertEqual(provider.calls, calls_after_first_run)
            self.assertEqual(len(resumed["generated"]), 10)
            self.assertEqual(len(resumed["judged"]), 10)

    def test_resume_rejects_changed_dataset_or_model(self):
        import copy
        provider, fake_module, orm_models = self._fake_runner_dependencies()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_benchmark(
                self.dataset, run_dir=run_dir, provider=provider,
                ltm_module=fake_module, orm_models=orm_models,
            )

            changed_dataset = copy.deepcopy(self.dataset)
            changed_dataset["samples"][0]["expected"]["memory"] += "\n- changed fact"
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_benchmark(
                    changed_dataset, run_dir=run_dir, provider=provider,
                    ltm_module=fake_module, orm_models=orm_models, resume=True,
                )

            provider.MODEL = "different-model"
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_benchmark(
                    self.dataset, run_dir=run_dir, provider=provider,
                    ltm_module=fake_module, orm_models=orm_models, resume=True,
                )

    def test_generated_sample_is_checkpointed_before_judge_key_prompt(self):
        provider, fake_module, orm_models = self._fake_runner_dependencies(fail_judge=True)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with self.assertRaises(BenchmarkAborted):
                run_benchmark(
                    self.dataset,
                    run_dir=run_dir,
                    provider=provider,
                    ltm_module=fake_module,
                    orm_models=orm_models,
                    prompt_for_key=lambda _prompt: "",
                )
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
            first_sample_id = self.dataset["samples"][0]["sample_id"]
            self.assertEqual(checkpoint["generated"][first_sample_id]["generation_status"], "success")
            self.assertNotIn(first_sample_id, checkpoint["judged"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
