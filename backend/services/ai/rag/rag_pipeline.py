import logging
from concurrent.futures import ThreadPoolExecutor

from .query_analyzer import QueryAnalyzer
from .query_generator import QueryGenerator
from .local_retriever import LocalRetriever
from .web_retriever import WebRetriever
from .relevance_evaluator import RelevanceEvaluator
from .context_selector import ContextSelector

logger = logging.getLogger(__name__)


class KnowledgeRAGPipeline:
    """
    Conditional multi-source RAG:
      query analysis -> query generation -> parallel local/web retrieval
      -> relevance evaluation -> context selection.
    """

    def __init__(
        self,
        query_analyzer=None,
        query_generator=None,
        local_retriever=None,
        web_retriever=None,
        relevance_evaluator=None,
        context_selector=None,
    ):
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.query_generator = query_generator or QueryGenerator()
        self.local_retriever = local_retriever or LocalRetriever()
        self.web_retriever = web_retriever or WebRetriever()
        self.relevance_evaluator = relevance_evaluator or RelevanceEvaluator()
        self.context_selector = context_selector or ContextSelector()

    def retrieve(self, user_query: str) -> dict:
        """Return structured retrieval evidence and diagnostics, not a final answer."""
        decision = self.query_analyzer.analyze(user_query) or {}
        use_local = bool(decision.get("needs_rag") and decision.get("use_local"))
        use_web = bool(decision.get("needs_rag") and decision.get("use_web"))

        web_configured = None
        if use_web:
            checker = getattr(self.web_retriever, "is_configured", None)
            web_configured = bool(checker()) if callable(checker) else True

        result = {
            "used": False,
            "decision": decision,
            "search_queries": [],
            "context": None,
            "sources": [],
            "retrieved_count": 0,
            "selected_count": 0,
            "retrieval_status": "not_needed",
            "errors": [],
            "web": {
                "requested": use_web,
                "configured": web_configured,
                "result_count": 0,
            },
            "local": {
                "requested": use_local,
                "result_count": 0,
            },
        }

        if not decision.get("needs_rag") or not (use_local or use_web):
            return result

        queries = self.query_generator.generate(user_query)
        result["search_queries"] = queries
        if not queries:
            result["retrieval_status"] = "no_search_queries"
            result["errors"].append("query_generation: no valid search queries")
            return result

        candidates = []
        errors = []
        tasks = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            if use_local:
                tasks["local"] = executor.submit(
                    self.local_retriever.retrieve, queries
                )

            if use_web:
                if web_configured is False:
                    errors.append("web: TAVILY_API_KEY is not configured")
                else:
                    tasks["web"] = executor.submit(
                        self.web_retriever.retrieve, queries
                    )

            for source, future in tasks.items():
                try:
                    retrieved = future.result()
                    if not isinstance(retrieved, list):
                        raise TypeError("retriever must return a list")
                    candidates.extend(retrieved)
                    result[source]["result_count"] = len(retrieved)
                except Exception as exc:
                    logger.exception("%s retrieval failed", source)
                    errors.append(f"{source}: {type(exc).__name__}: {str(exc)[:180]}")

        result["retrieved_count"] = len(candidates)
        if not candidates:
            result["errors"] = errors
            result["retrieval_status"] = "failed" if errors else "no_results"
            return result

        evaluated = self.relevance_evaluator.evaluate(user_query, candidates)
        selected = self.context_selector.select(evaluated)
        context = self.context_selector.format_context(selected)

        result.update({
            "used": bool(context),
            "context": context,
            "sources": [
                {
                    "source_type": item.get("source_type"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "published_date": item.get("published_date"),
                    "relevance": item.get("relevance"),
                }
                for item in selected
            ],
            "selected_count": len(selected),
            "errors": errors,
            "retrieval_status": (
                "success" if context and not errors
                else "partial" if context
                else "no_relevant_results"
            ),
        })
        return result
