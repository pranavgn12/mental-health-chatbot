"""Offline unit tests for the benchmark script bundled beside this file."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Prefer the explicitly rewritten script in the shared /mnt/data workspace;
# inside the distributable package the canonical file is knowledge_rag_benchmark.py.
if (HERE / "knowledge_rag_benchmark_v3.py").exists():
    import knowledge_rag_benchmark_v3 as bench
elif (HERE / "knowledge_rag_benchmark_rewritten.py").exists():
    import knowledge_rag_benchmark_rewritten as bench
else:
    import knowledge_rag_benchmark as bench


class FakeCollection:
    name = "knowledge"

    def __init__(self, metadata_rows=None, query_response=None):
        self.metadata_rows = metadata_rows or []
        self.query_response = query_response or {
            "ids": [["chunk-1"]],
            "documents": [["This chunk supports the answer."]],
            "metadatas": [[{"source": "Example\nDocument.pdf", "page_start": 3, "page_end": 3}]],
            "distances": [[0.1]],
        }
        self.query_calls = []

    def count(self):
        return len(self.metadata_rows) or 3

    def get(self, include=None, limit=None):
        return {"metadatas": self.metadata_rows}

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        if "where" in kwargs:
            requested = kwargs["where"].get("source")
            matches = [m for m in self.metadata_rows if m.get("source") == requested]
            if not matches:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            return {
                "ids": [["chunk-1"]],
                "documents": [["This chunk supports the answer."]],
                "metadatas": [[matches[0]]],
                "distances": [[0.1]],
            }
        return self.query_response


class BenchmarkTests(unittest.TestCase):
    def test_normalize_source_handles_embedded_newlines(self):
        self.assertEqual(
            bench.normalize_source_label("Psychological\ninterventions\nimplementation\nmanual.pdf"),
            bench.normalize_source_label("Psychological interventions implementation manual.pdf"),
        )

    def test_normalize_dataset_supports_evidence_phrase_and_auto_ids(self):
        raw = {
            "dataset_name": "tiny",
            "questions": [{
                "question": "Why?",
                "reference_answer": "Because.",
                "evidence_phrase": "The exact supporting sentence.",
                "source_document": "source.pdf",
            }],
        }
        name, questions = bench.normalize_dataset(raw)
        self.assertEqual(name, "tiny")
        self.assertEqual(questions[0]["question_id"], "Q001")
        self.assertEqual(questions[0]["evidence_phrases"][0]["phrase"], "The exact supporting sentence.")
        self.assertEqual(questions[0]["evidence_phrases"][0]["source_document"], "source.pdf")

    def test_source_catalog_resolves_newline_metadata_and_uses_raw_where_value(self):
        stored = "Psychological\ninterventions\nimplementation\nmanual.pdf"
        collection = FakeCollection(metadata_rows=[{"source": stored}])
        question = {
            "question_id": "Q1",
            "question": "Why?",
            "reference_answer": "Because.",
            "source_document": "Psychological interventions implementation manual.pdf",
            "evidence_phrases": [{"evidence_id": "E001", "phrase": "supporting text", "source_document": None}],
        }
        catalog = bench.build_source_catalog(collection)
        candidates, report = bench.build_candidate_pool(collection, question, top_n=5, source_catalog=catalog)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(collection.query_calls[0]["where"], {"source": stored})
        self.assertEqual(report["status"], "matched")
        self.assertFalse(report["used_global_fallback"])

    def test_unknown_source_uses_global_fallback_instead_of_zero_candidates(self):
        collection = FakeCollection(metadata_rows=[{"source": "actual_source.pdf"}])
        question = {
            "question_id": "Q1",
            "question": "Why?",
            "reference_answer": "Because.",
            "source_document": "example_document.pdf",
            "evidence_phrases": [{"evidence_id": "E001", "phrase": "supporting text", "source_document": None}],
        }
        candidates, report = bench.build_candidate_pool(collection, question, top_n=5)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(report["used_global_fallback"])
        self.assertEqual(report["status"], "unmatched_source_global_fallback")
        self.assertNotIn("where", collection.query_calls[0])


    def test_matched_source_with_no_hits_tries_global_fallback(self):
        class NoHitFilteredCollection(FakeCollection):
            def query(self, **kwargs):
                self.query_calls.append(kwargs)
                if "where" in kwargs:
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
                return self.query_response

        stored = "Actual\nsource.pdf"
        collection = NoHitFilteredCollection(metadata_rows=[{"source": stored}])
        question = {
            "question_id": "Q1",
            "question": "Why?",
            "reference_answer": "Because.",
            "source_document": "Actual source.pdf",
            "evidence_phrases": [{"evidence_id": "E001", "phrase": "supporting text", "source_document": None}],
        }
        candidates, report = bench.build_candidate_pool(collection, question, top_n=5)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(collection.query_calls), 2)
        self.assertIn("where", collection.query_calls[0])
        self.assertNotIn("where", collection.query_calls[1])
        self.assertEqual(report["status"], "matched_source_global_fallback")

    def test_metrics_report_every_question_but_aggregate_only_supported_ground_truth(self):
        questions = [
            {"question_id": "Q1", "question": "supported"},
            {"question_id": "Q2", "question": "no positives"},
            {"question_id": "Q3", "question": "no candidates"},
            {"question_id": "Q4", "question": "partial"},
        ]
        records = [
            {"question_id": "Q1", "retrieved_chunks": [{"chroma_id": "a"}, {"chroma_id": "x"}]},
            {"question_id": "Q2", "retrieved_chunks": [{"chroma_id": "b"}]},
            {"question_id": "Q3", "retrieved_chunks": []},
            {"question_id": "Q4", "retrieved_chunks": [{"chroma_id": "d"}]},
        ]
        state = {"annotations": {
            "Q1": {"decisions": {"a": {"relevance_grade": 3}, "b": {"relevance_grade": 1}}, "candidate_count": 2, "review_complete": True},
            "Q2": {"decisions": {"b": {"relevance_grade": 1}}, "candidate_count": 1, "review_complete": True},
            "Q3": {"decisions": {}, "candidate_count": 0, "review_complete": True},
            "Q4": {"decisions": {"d": {"relevance_grade": 2}}, "candidate_count": 2, "review_complete": False},
        }}
        rows, summary = bench.evaluate_retrieval(questions, records, state, [1, 2])
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary["retrieval_evaluable_question_count"], 1)
        self.assertEqual(summary["evaluation_status_counts"], {
            "evaluated_with_positive_ground_truth": 1,
            "completed_weak_only_no_binary_relevant": 1,
            "no_candidates_found": 1,
            "incomplete_ground_truth_review": 1,
        })
        self.assertAlmostEqual(rows[0]["precision@1"], 1.0)
        self.assertIsNone(rows[1]["precision@1"])
        self.assertIsNotNone(rows[1]["ndcg@2"])  # Grade 1 contributes to graded nDCG.
        self.assertEqual(summary["graded_ndcg_evaluable_question_count"], 2)
        self.assertAlmostEqual(
            summary["macro_average_on_evaluable_questions"]["ndcg@2"],
            (rows[0]["ndcg@2"] + rows[1]["ndcg@2"]) / 2,
        )
        self.assertIsNone(rows[2]["mrr@2"])
        self.assertIsNone(rows[3]["ndcg@2"])

    def test_answer_benchmark_is_opt_in_by_default(self):
        args = bench.parse_args(["--dataset", "questions.json"])
        self.assertFalse(args.run_answer_benchmark)
        args = bench.parse_args(["--dataset", "questions.json", "--run-answer-benchmark"])
        self.assertTrue(args.run_answer_benchmark)

    def test_question_fingerprint_changes_when_evidence_or_answer_changes(self):
        base = {
            "question": "Why?", "reference_answer": "Because.",
            "source_document": "a.pdf",
            "evidence_phrases": [{"phrase": "Exact phrase", "source_document": "a.pdf"}],
        }
        changed = dict(base)
        changed["reference_answer"] = "A different answer."
        self.assertNotEqual(bench.question_fingerprint(base), bench.question_fingerprint(changed))
        changed = dict(base)
        changed["evidence_phrases"] = [{"phrase": "Different exact phrase", "source_document": "a.pdf"}]
        self.assertNotEqual(bench.question_fingerprint(base), bench.question_fingerprint(changed))

    def test_metrics_do_not_silently_drop_questions_when_retrieval_record_missing(self):
        questions = [
            {"question_id": "Q1", "question": "present"},
            {"question_id": "Q2", "question": "missing"},
        ]
        state = {"annotations": {
            "Q1": {"decisions": {"a": {"relevance_grade": 3}}, "candidate_ids": ["a"], "candidate_count": 1, "review_complete": True},
            "Q2": {"decisions": {"b": {"relevance_grade": 3}}, "candidate_ids": ["b"], "candidate_count": 1, "review_complete": True},
        }}
        rows, summary = bench.evaluate_retrieval(
            questions, [{"question_id": "Q1", "retrieved_chunks": [{"chroma_id": "a"}]}], state, [1]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["evaluation_status"], "retrieval_record_missing")
        self.assertTrue(rows[1]["retrieval_record_missing"])
        self.assertEqual(summary["retrieval_record_missing_count"], 1)

    def test_completed_candidate_review_is_reused_without_requery_or_llm(self):
        import tempfile

        class NoQueryCollection(FakeCollection):
            def get(self, include=None, limit=None):
                raise AssertionError("cached review should not rebuild the source catalog")

            def query(self, **kwargs):
                raise AssertionError("cached review should not re-query evidence phrases")

        class NoCallJudge:
            def complete(self, *args, **kwargs):
                raise AssertionError("cached review should not spend Groq tokens")

        question = {
            "question_id": "Q001", "question": "Why?", "reference_answer": "Because.",
            "source_document": "source.pdf",
            "evidence_phrases": [{"evidence_id": "E001", "phrase": "exact phrase", "source_document": "source.pdf"}],
        }
        collection = NoQueryCollection(metadata_rows=[{"source": "source.pdf"}])
        state = {"annotations": {"Q001": {
            "question_fingerprint": bench.question_fingerprint(question),
            "candidate_discovery_version": bench.CANDIDATE_DISCOVERY_VERSION,
            "candidate_top_n": 15, "candidate_collection_count": 1,
            "review_complete": True, "candidate_ids": ["c1"], "candidate_count": 1,
            "decisions": {"c1": {"relevance_grade": 3}}, "relevant_chunks": {},
        }}}
        with tempfile.TemporaryDirectory() as tmp:
            bench.review_candidates([question], collection, Path(tmp), state, 15, 100, NoCallJudge())

    def test_batched_judgement_uses_one_request_and_maps_ids(self):
        class DummyJudge:
            model = "test-model"

            def __init__(self):
                self.calls = 0

            def complete(self, messages, **kwargs):
                self.calls += 1
                return (
                    '{"judgements":['
                    '{"chroma_id":"c1","grade":3,"supports_answer":true,"source_match":true,"rationale":"direct"},'
                    '{"chroma_id":"c2","grade":1,"supports_answer":false,"source_match":true,"rationale":"weak"}'
                    ']}'
                )

        judge = DummyJudge()
        question = {
            "question": "Why?", "reference_answer": "Because.",
            "source_document": "source.pdf",
            "evidence_phrases": [{"phrase": "support", "source_document": "source.pdf"}],
        }
        candidates = [
            {"chroma_id": "c1", "content": "first", "metadata": {"source": "source.pdf"}},
            {"chroma_id": "c2", "content": "second", "metadata": {"source": "source.pdf"}},
        ]
        result = bench.llm_judge_chunks_batch(judge, question, candidates)
        self.assertEqual(judge.calls, 1)
        self.assertEqual(result["c1"]["relevance_grade"], 3)
        self.assertEqual(result["c2"]["relevance_grade"], 1)

    def test_compact_text_limits_prompt_size(self):
        value = "A" * 10000
        shortened = bench.compact_text(value, 1000)
        self.assertLessEqual(len(shortened), 1100)
        self.assertIn("content omitted", shortened)

    def test_answer_embedding_is_cached_and_cpu_by_default(self):
        calls = []

        class DummySentenceTransformer:
            def __init__(self, model_name, device):
                calls.append((model_name, device))

            def encode(self, values, **kwargs):
                return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)

        fake_module = types.SimpleNamespace(SentenceTransformer=DummySentenceTransformer)
        bench._load_sentence_model.cache_clear()
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            first = bench.answer_embedding_similarity("a", "b", "fake-model")
            second = bench.answer_embedding_similarity("c", "d", "fake-model")
        self.assertEqual(first, 0.0)
        self.assertEqual(second, 0.0)
        self.assertEqual(calls, [("fake-model", "cpu")])
        bench._load_sentence_model.cache_clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
