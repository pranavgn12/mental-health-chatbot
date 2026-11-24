import importlib.util
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "long_term_memory_benchmark.py"
spec = importlib.util.spec_from_file_location("ltm_benchmark", SCRIPT)
bench = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(bench)


class FakeChromaEngine:
    def score_paragraphs(self, expected, generated):
        if not expected and not generated:
            return {"score": 1.0, "expected_to_generated": 1.0, "generated_to_expected": 1.0,
                    "expected_paragraphs": 0, "generated_paragraphs": 0}
        if not expected or not generated:
            return {"score": 0.0, "expected_to_generated": 0.0, "generated_to_expected": 0.0,
                    "expected_paragraphs": len(expected), "generated_paragraphs": len(generated)}
        vocab = ["study", "morning", "walk", "project", "completed"]
        def encode(texts):
            vectors = []
            for text in texts:
                words = set(__import__("re").sub(r"[^a-z]+", " ", text.lower()).split())
                vector = np.array([1.0 if term in words else 0.0 for term in vocab], dtype=float)
                norm = np.linalg.norm(vector)
                vectors.append(vector / norm if norm else vector)
            return np.vstack(vectors)
        e, g = encode(expected), encode(generated)
        sim = e @ g.T
        a, b = float(sim.max(axis=1).mean()), float(sim.max(axis=0).mean())
        return {"score": (a+b)/2, "expected_to_generated": a, "generated_to_expected": b,
                "expected_paragraphs": len(expected), "generated_paragraphs": len(generated)}


class FakeGroq:
    def __init__(self, api_key=None):
        self.api_key = api_key


class FakeProvider:
    def __init__(self):
        self.client = None


class BenchmarkTests(unittest.TestCase):
    def test_normalizes_new_cases_schema(self):
        data = {
            "benchmark_name": "LTM",
            "cases": [{
                "case_id": "LTM_001",
                "input": {
                    "episode_summary": "The user completed a project.",
                    "mood_observations": [{"mood": "relieved", "confidence": 0.9, "evidence": "It is done."}],
                    "reflections": [{"reflection": "Planning helped."}],
                    "validated_patterns": [],
                    "previous_memory": None,
                },
                "expected_output": {"memory": "The user completed a project."},
                "evaluation_criteria": {
                    "must_preserve": ["The user completed a project."],
                    "must_not_infer": [],
                    "temporal_constraints": ["Do not describe the project as planned."],
                    "paragraph_expectations": [],
                },
            }],
        }
        cases = bench.validate_dataset(data)
        self.assertEqual(cases[0]["sample_id"], "LTM_001")
        self.assertEqual(cases[0]["current_episode"]["summary"], "The user completed a project.")
        self.assertEqual(cases[0]["previous_memory"], "")
        self.assertEqual(bench.retention_targets(cases[0]), ["The user completed a project."])

    def test_normalizes_project_samples_schema(self):
        data = {"samples": [{
            "sample_id": "LTM_S1",
            "category": "memory_update",
            "current_episode": {"episode_id": "EP1", "summary": "The user began a report."},
            "mood_observations": [],
            "reflections": [],
            "validated_patterns": [],
            "previous_memory": "",
            "expected": {"memory": "The user began a report."},
        }]}
        cases = bench.validate_dataset(data)
        self.assertEqual(len(cases), 1)
        self.assertEqual(bench.retention_targets(cases[0]), ["The user began a report."])

    def test_accepts_legacy_case_input_current_episode(self):
        data = {"cases": [{
            "case_id": "LTM_OLD",
            "input": {"current_episode": {"episode_id": "EP2", "summary": "The user tried a short walk."},
                      "mood_observations": [], "reflections": [], "validated_patterns": [], "previous_memory": ""},
            "expected": {"memory": "The user tried a short walk."},
        }]}
        cases = bench.validate_dataset(data)
        self.assertEqual(cases[0]["current_episode"]["episode_id"], "EP2")

    def test_judge_metric_formulas(self):
        result = {
            "information_retention": [
                {"target": "A", "verdict": "retained"},
                {"target": "B", "verdict": "partial"},
                {"target": "C", "verdict": "missing"},
            ],
            "factual_accuracy": {
                "claims": [
                    {"claim": "supported", "verdict": "supported"},
                    {"claim": "partial", "verdict": "partially_supported"},
                    {"claim": "unsupported", "verdict": "unsupported"},
                ],
                "must_not_infer_violations": [{"constraint": "No diagnosis", "violating_claim": "The user has a diagnosis."}],
            },
            "temporal_update_accuracy": [
                {"constraint": "planned", "verdict": "correct"},
                {"constraint": "completed", "verdict": "incorrect"},
            ],
            "notes": [],
        }
        metrics = bench.calculate_judge_metrics(result)
        self.assertEqual(metrics["information_retention"], 0.5)
        self.assertEqual(metrics["factual_accuracy"], 0.5)
        self.assertEqual(metrics["temporal_update_accuracy"], 0.5)

    def test_prohibited_inference_rate_is_reported_separately(self):
        case = bench.validate_dataset({"cases": [{
            "case_id": "LTM_SAFETY",
            "input": {"episode_summary": "The user felt anxious before a presentation.",
                      "mood_observations": [], "reflections": [], "validated_patterns": [],
                      "previous_memory": ""},
            "expected_output": {"memory": "The user felt anxious before a presentation."},
            "evaluation_criteria": {
                "must_preserve": ["The user felt anxious before a presentation."],
                "must_not_infer": ["Do not infer a diagnosis.", "Do not infer a recurring condition."],
                "temporal_constraints": [], "paragraph_expectations": []
            }
        }]})[0]
        result = {
            "information_retention": [{"target": "The user felt anxious before a presentation.", "verdict": "retained"}],
            "factual_accuracy": {
                "claims": [{"claim": "The user felt anxious before a presentation.", "verdict": "supported"}],
                "must_not_infer_violations": [{"constraint": "Do not infer a diagnosis.",
                                               "violating_claim": "The user has an anxiety disorder."}]
            },
            "temporal_update_accuracy": [], "notes": []
        }
        metrics = bench.calculate_judge_metrics(result, case)
        self.assertEqual(metrics["factual_accuracy"], 1.0)
        self.assertEqual(metrics["must_not_infer_violation_rate"], 0.5)
        self.assertEqual(metrics["counts"]["must_not_infer_constraints"], 2)

    def test_bge_paragraph_similarity(self):
        model = FakeChromaEngine()
        result = bench.paragraph_similarity_score(
            "The user studies in the morning.\n\nThe user completed the project.",
            "The user likes morning study.\n\nThe project was completed.",
            model,
        )
        self.assertGreaterEqual(result["score"], 0.5)
        self.assertEqual(result["expected_paragraphs"], 2)
        self.assertEqual(result["generated_paragraphs"], 2)

    def test_empty_embedding_cases(self):
        model = FakeChromaEngine()
        self.assertEqual(bench.paragraph_similarity_score("", "", model)["score"], 1.0)
        self.assertEqual(bench.paragraph_similarity_score("expected", "", model)["score"], 0.0)

    def test_key_rotation_retries_same_operation(self):
        provider = FakeProvider()
        manager = bench.APIKeyManager(provider, FakeGroq, ["key-one", "key-two"])
        attempts = {"count": 0}
        def operation():
            attempts["count"] += 1
            if provider.client.api_key == "key-one":
                exc = RuntimeError("429 rate limit")
                exc.status_code = 429
                raise exc
            return "success"
        value = bench.call_with_key_rotation(operation, "unit test", manager)
        self.assertEqual(value, "success")
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(provider.client.api_key, "key-two")


    def test_prompts_for_replacement_key_after_pool_is_exhausted(self):
        from unittest.mock import patch
        provider = FakeProvider()
        manager = bench.APIKeyManager(provider, FakeGroq, ["first-key"])
        failure = RuntimeError("429 Too Many Requests")
        failure.status_code = 429
        with patch.object(bench.getpass, "getpass", return_value="replacement-key"):
            manager.rotate("generation", failure)
        self.assertEqual(provider.client.api_key, "replacement-key")
        self.assertEqual(manager.current_key, "replacement-key")

    def test_key_error_detection(self):
        exc = RuntimeError("quota exceeded")
        self.assertTrue(bench.is_key_or_quota_error(exc))
        self.assertFalse(bench.is_key_or_quota_error(ValueError("invalid JSON")))


    def test_chroma_engine_uses_temporary_collection_and_exact_cosine(self):
        import types
        from unittest.mock import patch
        calls = {"created": None, "deleted": None}
        class Collection:
            def add(self, **kwargs):
                calls["added"] = kwargs
            def get(self, **kwargs):
                ids = kwargs["ids"]
                vectors = {
                    "expected_0": [1.0, 0.0],
                    "expected_1": [0.0, 1.0],
                    "generated_0": [0.8, 0.6],
                }
                return {"ids": ids, "embeddings": np.asarray([vectors[item_id] for item_id in ids], dtype=np.float32)}
        class Client:
            def create_collection(self, **kwargs):
                calls["created"] = kwargs
                return Collection()
            def delete_collection(self, **kwargs):
                calls["deleted"] = kwargs
        chromadb = types.ModuleType("chromadb")
        chromadb.EphemeralClient = lambda: Client()
        utils = types.ModuleType("chromadb.utils")
        embedding_functions = types.ModuleType("chromadb.utils.embedding_functions")
        class EmbeddingFunction:
            def __init__(self, model_name):
                calls["model_name"] = model_name
        embedding_functions.SentenceTransformerEmbeddingFunction = EmbeddingFunction
        with patch.dict("sys.modules", {"chromadb": chromadb, "chromadb.utils": utils,
                                          "chromadb.utils.embedding_functions": embedding_functions}):
            engine = bench.ChromaSimilarityEngine("BAAI/bge-small-en-v1.5")
            result = engine.score_paragraphs(["e1", "e2"], ["g1"])
        self.assertAlmostEqual(result["expected_to_generated"], 0.7)
        self.assertAlmostEqual(result["generated_to_expected"], 0.8)
        self.assertAlmostEqual(result["score"], 0.75)
        self.assertEqual(calls["created"]["metadata"], {"hnsw:space": "cosine"})
        self.assertEqual(calls["model_name"], "BAAI/bge-small-en-v1.5")
        self.assertEqual(calls["deleted"]["name"], calls["created"]["name"])

    def test_judge_result_validation_checks_target_counts(self):
        case = bench.validate_dataset({"cases": [{
            "case_id": "LTM_J", "input": {"episode_summary": "The user completed a task.",
                "mood_observations": [], "reflections": [], "validated_patterns": [], "previous_memory": ""},
            "expected_output": {"memory": "The user completed a task."},
            "evaluation_criteria": {"must_preserve": ["The user completed a task."],
                "must_not_infer": [], "temporal_constraints": ["Do not call it planned."], "paragraph_expectations": []}
        }]})[0]
        evaluation = {
            "information_retention": [{"target": "The user completed a task.", "verdict": "retained", "evidence": "It is stated as complete."}],
            "factual_accuracy": {"claims": [{"claim": "The user completed a task.", "verdict": "supported", "evidence": "episode summary"}], "must_not_infer_violations": []},
            "temporal_update_accuracy": [{"constraint": "Do not call it planned.", "verdict": "correct", "evidence": "It is complete."}],
            "notes": []
        }
        checked = bench.validate_judge_result(evaluation, case)
        self.assertEqual(bench.calculate_judge_metrics(checked)["information_retention"], 1.0)

    def test_rejects_duplicate_case_ids(self):
        case = {
            "case_id": "DUP",
            "input": {"episode_summary": "Some summary.", "mood_observations": [], "reflections": [],
                      "validated_patterns": [], "previous_memory": ""},
            "expected_output": {"memory": ""},
        }
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bench.validate_dataset({"cases": [case, case]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
