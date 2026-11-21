import importlib.util
import math
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "knowledge_rag_benchmark.py"
spec = importlib.util.spec_from_file_location("knowledge_rag_benchmark", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class FakeJudge:
    model = "test-model"

    def __init__(self, response):
        self.response = response
        self.last_messages = None

    def complete(self, messages, **kwargs):
        self.last_messages = messages
        return self.response


class BenchmarkTests(unittest.TestCase):
    def test_dataset_format_and_automatic_ids(self):
        name, questions = bench.normalize_dataset({
            "dataset_name": "knowledge_rag_benchmark",
            "questions": [{
                "question": "Why train staff?",
                "reference_answer": "To maintain competency.",
                "evidence_phrase": "Training maintains competency.",
                "source_document": "manual.pdf",
            }, {
                "question": "Why supervise staff?",
                "reference_answer": "To support quality.",
                "evidence_phrase": "Supervision supports quality.",
                "source_document": "manual.pdf",
            }],
        })
        self.assertEqual(name, "knowledge_rag_benchmark")
        self.assertEqual([q["question_id"] for q in questions], ["Q001", "Q002"])
        self.assertEqual(questions[0]["evidence_phrases"][0]["phrase"], "Training maintains competency.")

    def test_metrics(self):
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(bench.precision_at_k(retrieved, relevant, 3), 2 / 3)
        self.assertAlmostEqual(bench.recall_at_k(retrieved, relevant, 2), 0.5)
        self.assertAlmostEqual(bench.reciprocal_rank(retrieved, relevant, 3), 1.0)
        self.assertGreater(bench.ndcg_at_k(retrieved, {"a": 3, "b": 2}, 3), 0)

    def test_parse_json_with_fence(self):
        result = bench.parse_judge_json('```json\n{"grade": 3, "rationale": "supported"}\n```')
        self.assertEqual(result["grade"], 3)

    def test_chunk_judge_grade(self):
        fake = FakeJudge('{"grade":2,"supports_answer":true,"source_match":true,"rationale":"supporting evidence"}')
        question = {
            "question": "Why train?", "reference_answer": "Maintain skills.",
            "evidence_phrases": [{"phrase": "Training maintains skills."}],
            "source_document": "manual.pdf",
        }
        candidate = {"content": "Staff training maintains skills.", "metadata": {"source": "manual.pdf"}}
        result = bench.llm_judge_chunk(fake, question, candidate)
        self.assertEqual(result["relevance_grade"], 2)
        self.assertTrue(result["source_match"])

    def test_answer_judge_grade(self):
        fake = FakeJudge('{"grade":3,"label":"correct","context_supported":true,"rationale":"equivalent answer"}')
        question = {"question": "Why train?", "reference_answer": "Maintain skills."}
        result = bench.llm_judge_answer(fake, question, "Training keeps skills current.", [
            {"source": "manual.pdf", "page_start": 1, "page_end": 1, "content": "Training keeps skills current."}
        ])
        self.assertEqual(result["grade"], 3)
        self.assertEqual(result["label"], "correct")

    def test_context_error_classifier(self):
        self.assertTrue(bench.GroqJudge._is_context_error(RuntimeError("maximum context length exceeded")))
        self.assertTrue(bench.GroqJudge._is_context_error(RuntimeError("output token limit reached")))
        self.assertFalse(bench.GroqJudge._is_context_error(RuntimeError("invalid API key")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
