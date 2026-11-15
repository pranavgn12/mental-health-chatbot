import logging
import os
from concurrent.futures import ThreadPoolExecutor

from .query_analyzer import QueryAnalyzer
from .query_generator import QueryGenerator
from .local_retriever import LocalRetriever
from .web_retriever import WebRetriever
from .relevance_evaluator import RelevanceEvaluator
from .context_selector import ContextSelector

logger = logging.getLogger("rag_pipeline")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [RAG] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)


def _env_bool(name, default=False):
    value = os.getenv(name)
    return default if value is None else value.strip().casefold() in {"1", "true", "yes", "on"}


class KnowledgeRAGPipeline:
    """Hybrid knowledge RAG. Existing retrieve() result/reference keys are preserved."""

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
        logger.info("Knowledge RAG pipeline initialized.")

    def retrieve(self, user_query: str) -> dict:
        logger.info("=" * 60)
        logger.info("Starting knowledge retrieval; query length=%d", len(user_query or ""))

        try:
            decision = self.query_analyzer.analyze(user_query) or {}
        except Exception:
            logger.exception("Query analysis failed.")
            raise

        use_local = bool(decision.get("needs_rag") and decision.get("use_local"))
        use_web = bool(decision.get("needs_rag") and decision.get("use_web") and _env_bool("RAG_ENABLE_WEB", False))
        logger.info("Routing: needs_rag=%s local=%s web=%s method=%s reason=%s",
                    decision.get("needs_rag"), use_local, use_web,
                    decision.get("routing_method"), decision.get("reason"))

        web_configured = None
        if use_web:
            checker = getattr(self.web_retriever, "is_configured", None)
            web_configured = bool(checker()) if callable(checker) else True

        # Preserve the established result schema. New local citation fields are
        # added to each existing source item; original source keys stay unchanged.
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
            "web": {"requested": use_web, "configured": web_configured, "result_count": 0},
            "local": {"requested": use_local, "result_count": 0},
        }

        if not decision.get("needs_rag") or not (use_local or use_web):
            logger.info("Retrieval skipped: no source selected.")
            return result

        try:
            # Warm the existing BM25 index once and expose its dynamically
            # discovered reference vocabulary to query generation.
            get_catalog = getattr(self.local_retriever, "get_identifier_catalog", None)
            known_identifiers = None
            if use_local and callable(get_catalog):
                try:
                    known_identifiers = get_catalog()
                except Exception:
                    # Let the actual retrieval stage report DB/index failures;
                    # query generation can still proceed with the original text.
                    logger.exception("Could not load corpus identifier catalogue; using generic query parsing.")
            try:
                queries = self.query_generator.generate(
                    user_query, known_identifiers=known_identifiers
                )
            except TypeError:
                # Preserve compatibility with injected/custom query generators
                # that still implement generate(user_query) only.
                queries = self.query_generator.generate(user_query)
        except Exception as exc:
            logger.exception("Query generation failed.")
            result["retrieval_status"] = "no_search_queries"
            result["errors"].append(f"query_generation: {type(exc).__name__}")
            return result
        result["search_queries"] = queries
        if not queries:
            result["retrieval_status"] = "no_search_queries"
            return result
        logger.info("Search queries: %s", queries)

        candidates, errors, tasks = [], [], {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            if use_local:
                tasks["local"] = executor.submit(self.local_retriever.retrieve, queries)
            if use_web:
                if web_configured is False:
                    errors.append("web: TAVILY_API_KEY is not configured")
                else:
                    tasks["web"] = executor.submit(self.web_retriever.retrieve, queries)
            for source, future in tasks.items():
                try:
                    retrieved = future.result()
                    if not isinstance(retrieved, list):
                        raise TypeError("Retriever must return a list")
                    candidates.extend(retrieved)
                    result[source]["result_count"] = len(retrieved)
                    logger.info("%s retrieval returned %d candidates", source, len(retrieved))
                except Exception as exc:
                    logger.exception("%s retrieval failed", source)
                    errors.append(f"{source}: {type(exc).__name__}: {str(exc)[:180]}")

        result["retrieved_count"] = len(candidates)
        if not candidates:
            result["errors"] = errors
            result["retrieval_status"] = "failed" if errors else "no_results"
            return result

        try:
            evaluated = self.relevance_evaluator.evaluate(user_query, candidates)
            selected = self.context_selector.select(evaluated, query=user_query)
            context = self.context_selector.format_context(selected)
        except Exception:
            logger.exception("Evaluation or context selection failed.")
            raise

        # Keep all original source fields; include optional local page/section
        # provenance so downstream citation rendering can use accurate refs.
        sources = []
        for item in selected:
            sources.append({
                "source_type": item.get("source_type"),
                "title": item.get("title"),
                "url": item.get("url"),
                "published_date": item.get("published_date"),
                "relevance": item.get("relevance"),
                "source": item.get("source"),
                "topic": item.get("topic"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "heading_path": item.get("heading_path"),
                "module": item.get("module"),
                "section": item.get("section"),
                "chunk": item.get("chunk"),
                "content_type": item.get("content_type"),
            })

        result.update({
            "used": bool(context),
            "context": context,
            "sources": sources,
            "selected_count": len(selected),
            "errors": errors,
            "retrieval_status": (
                "success" if context and not errors
                else "partial" if context
                else "no_relevant_results"
            ),
        })
        logger.info("RAG complete: status=%s retrieved=%d selected=%d",
                    result["retrieval_status"], result["retrieved_count"], result["selected_count"])
        return result
