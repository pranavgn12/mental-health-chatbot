import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from io import BytesIO
from urllib.error import HTTPError

# Keep the tests independent from a configured Groq account.
provider_stub = types.ModuleType("services.llm_provider")
provider_stub.generate_response = lambda *args, **kwargs: "{}"
provider_stub.generate_response_stream = lambda *args, **kwargs: iter(())
sys.modules.setdefault("services.llm_provider", provider_stub)

from services.ai.rag.query_analyzer import QueryAnalyzer
from services.ai.rag.rag_pipeline import KnowledgeRAGPipeline
from services.ai.rag.relevance_evaluator import RelevanceEvaluator
from services.ai.rag.context_selector import ContextSelector
from services.ai.rag.web_retriever import WebRetriever


class FakeAnalyzer:
    def __init__(self, decision):
        self.decision = decision

    def analyze(self, query):
        return self.decision


class FakeGenerator:
    def generate(self, query):
        return [query, f"{query} complementary evidence"]


class FakeRetriever:
    def __init__(self, source_type, configured=True, results=None):
        self.source_type = source_type
        self.configured = configured
        self.results = results

    def is_configured(self):
        return self.configured

    def retrieve(self, queries):
        if self.results is not None:
            return self.results
        return [{
            "id": f"{self.source_type}_1",
            "source_type": self.source_type,
            "title": f"{self.source_type} source",
            "source": f"{self.source_type} source",
            "url": "https://example.org/reference" if self.source_type == "web" else None,
            "content": f"Relevant {self.source_type} evidence about the user question. " * 8,
            "relevance": 0.9,
        }]


class FakeEvaluator:
    def evaluate(self, query, candidates):
        return [
            {**candidate, "relevance": 0.9, "evaluation_reason": "test"}
            for candidate in candidates
        ]


class RAGPipelineTests(unittest.TestCase):
    def test_skips_retrieval_when_analyzer_says_no_rag(self):
        local = FakeRetriever("local")
        web = FakeRetriever("web")
        pipeline = KnowledgeRAGPipeline(
            query_analyzer=FakeAnalyzer({
                "needs_rag": False,
                "use_local": False,
                "use_web": False,
                "reason": "casual conversation",
            }),
            query_generator=FakeGenerator(),
            local_retriever=local,
            web_retriever=web,
        )

        result = pipeline.retrieve("Hello!")

        self.assertFalse(result["used"])
        self.assertIsNone(result["context"])
        self.assertEqual(result["search_queries"], [])
        self.assertEqual(result["retrieval_status"], "not_needed")

    def test_retrieves_from_both_sources_and_formats_context(self):
        pipeline = KnowledgeRAGPipeline(
            query_analyzer=FakeAnalyzer({
                "needs_rag": True,
                "use_local": True,
                "use_web": True,
                "reason": "current knowledge question",
            }),
            query_generator=FakeGenerator(),
            local_retriever=FakeRetriever("local"),
            web_retriever=FakeRetriever("web"),
            relevance_evaluator=FakeEvaluator(),
            context_selector=ContextSelector(),
        )

        result = pipeline.retrieve("How can I manage stress?")

        self.assertTrue(result["used"])
        self.assertEqual(len(result["search_queries"]), 2)
        self.assertEqual(result["search_queries"][0], "How can I manage stress?")
        self.assertEqual(result["retrieved_count"], 2)
        self.assertEqual(result["selected_count"], 2)
        self.assertIn("LOCAL", result["context"])
        self.assertIn("WEB", result["context"])
        self.assertIn("https://example.org/reference", result["context"])
        self.assertEqual(result["retrieval_status"], "success")
        self.assertEqual(result["web"]["result_count"], 1)

    def test_missing_tavily_key_is_reported(self):
        pipeline = KnowledgeRAGPipeline(
            query_analyzer=FakeAnalyzer({
                "needs_rag": True,
                "use_local": False,
                "use_web": True,
                "reason": "explicit web request",
            }),
            query_generator=FakeGenerator(),
            web_retriever=FakeRetriever("web", configured=False),
            relevance_evaluator=FakeEvaluator(),
        )

        result = pipeline.retrieve("Search the internet for current guidance")

        self.assertFalse(result["used"])
        self.assertEqual(result["retrieval_status"], "failed")
        self.assertFalse(result["web"]["configured"])
        self.assertTrue(any("TAVILY_API_KEY" in error for error in result["errors"]))

    def test_explicit_web_request_overrides_no_rag_model_decision(self):
        model_decision = json.dumps({
            "needs_rag": False,
            "use_local": False,
            "use_web": False,
            "reason": "model said not needed",
        })
        with patch(
            "services.ai.rag.query_analyzer.generate_response",
            return_value=model_decision,
        ):
            result = QueryAnalyzer().analyze(
                "Please search the internet for the latest research on student stress."
            )

        self.assertTrue(result["needs_rag"])
        self.assertTrue(result["use_web"])
        self.assertEqual(result["routing_method"], "llm+web-intent-guard")

    def test_time_sensitive_request_routes_to_web(self):
        model_decision = json.dumps({
            "needs_rag": False,
            "use_local": False,
            "use_web": False,
            "reason": "model said not needed",
        })
        with patch(
            "services.ai.rag.query_analyzer.generate_response",
            return_value=model_decision,
        ):
            result = QueryAnalyzer().analyze("What are the latest 2026 guidelines?")

        self.assertTrue(result["needs_rag"])
        self.assertTrue(result["use_web"])

    def test_immediate_danger_is_not_forced_into_web_rag(self):
        model_decision = json.dumps({
            "needs_rag": False,
            "use_local": False,
            "use_web": False,
            "reason": "urgent situation",
        })
        with patch(
            "services.ai.rag.query_analyzer.generate_response",
            return_value=model_decision,
        ):
            result = QueryAnalyzer().analyze(
                "I want to kill myself right now; please search online."
            )

        self.assertFalse(result["needs_rag"])
        self.assertFalse(result["use_web"])

    def test_evaluator_with_unknown_ids_falls_back_instead_of_dropping_evidence(self):
        candidates = [{
            "id": "web_1",
            "source_type": "web",
            "title": "Example",
            "url": "https://example.org",
            "content": "Useful source content",
            "search_score": 0.9,
        }]
        with patch(
            "services.ai.rag.relevance_evaluator.generate_response",
            return_value=json.dumps({"evaluations": [{"id": "wrong_id", "useful": True}]}),
        ):
            evaluated = RelevanceEvaluator().evaluate("question", candidates)

        self.assertEqual(len(evaluated), 1)
        self.assertEqual(evaluated[0]["id"], "web_1")

    def test_active_orchestrator_is_wired_to_multisource_pipeline(self):
        backend_root = Path(__file__).resolve().parents[1]
        source = (backend_root / "services/ai/orchestrator.py").read_text(encoding="utf-8")

        self.assertIn("from services.ai.rag import KnowledgeRAGPipeline", source)
        self.assertIn("self.knowledge_rag = KnowledgeRAGPipeline()", source)
        self.assertEqual(source.count("rag_result = self._retrieve_knowledge(latest_message)"), 2)
        self.assertIn('"rag": rag_result', source)


class WebRetrieverTests(unittest.TestCase):
    def test_parses_tavily_search_results(self):
        response_payload = {
            "results": [{
                "title": "A useful source",
                "url": "https://example.org/article",
                "content": "A detailed result with enough content to use as evidence.",
                "published_date": "2026-11-01",
                "score": 0.92,
            }]
        }

        class FakeResponse(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        with patch(
            "services.ai.rag.web_retriever.urlopen",
            return_value=FakeResponse(json.dumps(response_payload).encode("utf-8")),
        ):
            results = WebRetriever(api_key="test-key")._search_one("test query")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_type"], "web")
        self.assertEqual(results[0]["url"], "https://example.org/article")
        self.assertEqual(results[0]["published_date"], "2026-11-01")

    def test_tavily_http_error_is_not_silently_swallowed(self):
        error = HTTPError(
            "https://api.tavily.com/search",
            401,
            "Unauthorized",
            hdrs=None,
            fp=BytesIO(b"unauthorized"),
        )
        with patch(
            "services.ai.rag.web_retriever.urlopen",
            side_effect=error,
        ):
            with self.assertRaisesRegex(RuntimeError, "Tavily HTTP 401"):
                WebRetriever(api_key="test-key")._search_one("test query")

    def test_missing_tavily_key_fails_with_clear_message(self):
        with self.assertRaisesRegex(RuntimeError, "TAVILY_API_KEY"):
            WebRetriever(api_key="").retrieve(["test query"])



if __name__ == "__main__":
    unittest.main()
