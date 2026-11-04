
import logging
from concurrent.futures import ThreadPoolExecutor

from .query_analyzer import QueryAnalyzer
from .query_generator import QueryGenerator
from .local_retriever import LocalRetriever
from .web_retriever import WebRetriever
from .relevance_evaluator import RelevanceEvaluator
from .context_selector import ContextSelector


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logger = logging.getLogger("rag_pipeline")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [RAG] %(message)s",
            datefmt="%H:%M:%S"
        )
    )

    logger.addHandler(console_handler)


class KnowledgeRAGPipeline:
    """
    Conditional multi-source RAG:

        Query Analysis
             |
        Query Generation
             |
        +----+----+
        |         |
      Local      Web
     Retrieval  Retrieval
        |         |
        +----+----+
             |
     Relevance Evaluation
             |
      Context Selection
             |
       Final Context
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

        logger.info("Knowledge RAG pipeline initialized.")

    def retrieve(self, user_query: str) -> dict:
        """
        Return structured retrieval evidence and diagnostics,
        not a final answer.
        """

        logger.info("=" * 60)
        logger.info("Starting knowledge retrieval.")
        logger.info("Input query length: %d characters", len(user_query or ""))

        # =====================================================
        # 1. QUERY ANALYSIS
        # =====================================================

        logger.info("Stage 1/6: Analyzing query.")

        try:
            decision = self.query_analyzer.analyze(user_query) or {}

        except Exception:
            logger.exception("Query analysis failed.")
            raise

        use_local = bool(
            decision.get("needs_rag") and decision.get("use_local")
        )

        use_web = bool(
            decision.get("needs_rag") and decision.get("use_web")
        )

        logger.info(
            "Routing decision: needs_rag=%s, local=%s, web=%s",
            decision.get("needs_rag"),
            use_local,
            use_web
        )

        logger.info(
            "Routing method: %s",
            decision.get("routing_method", "unknown")
        )

        logger.info(
            "Routing reason: %s",
            decision.get("reason", "No reason provided")
        )

        # =====================================================
        # 2. INITIALIZE RESULT
        # =====================================================

        web_configured = None

        if use_web:
            checker = getattr(
                self.web_retriever,
                "is_configured",
                None
            )

            web_configured = (
                bool(checker()) if callable(checker) else True
            )

            logger.info(
                "Tavily API configuration: %s",
                "AVAILABLE" if web_configured else "MISSING"
            )

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

        # =====================================================
        # 3. CHECK WHETHER RETRIEVAL IS REQUIRED
        # =====================================================

        if not decision.get("needs_rag") or not (use_local or use_web):

            logger.info(
                "Retrieval skipped. No local or web knowledge required."
            )

            logger.info("Retrieval status: not_needed")
            logger.info("=" * 60)

            return result

        # =====================================================
        # 4. QUERY GENERATION
        # =====================================================

        logger.info("Stage 2/6: Generating search queries.")

        try:
            queries = self.query_generator.generate(user_query)

        except Exception:
            logger.exception("Query generation failed.")

            result["retrieval_status"] = "no_search_queries"
            result["errors"].append(
                "query_generation: generation failed"
            )

            logger.info("=" * 60)
            return result

        result["search_queries"] = queries

        logger.info(
            "Generated %d search queries.",
            len(queries)
        )

        for index, query in enumerate(queries, start=1):
            logger.info(
                "Search query %d: %s",
                index,
                query
            )

        if not queries:

            logger.warning("No valid search queries generated.")

            result["retrieval_status"] = "no_search_queries"
            result["errors"].append(
                "query_generation: no valid search queries"
            )

            logger.info("=" * 60)
            return result

        # =====================================================
        # 5. PARALLEL LOCAL AND WEB RETRIEVAL
        # =====================================================

        logger.info("Stage 3/6: Starting retrieval.")

        candidates = []
        errors = []
        tasks = {}

        with ThreadPoolExecutor(max_workers=2) as executor:

            # ---------------------------------------------
            # LOCAL RETRIEVAL
            # ---------------------------------------------

            if use_local:

                logger.info(
                    "Starting local ChromaDB retrieval."
                )

                tasks["local"] = executor.submit(
                    self.local_retriever.retrieve,
                    queries
                )

            # ---------------------------------------------
            # WEB RETRIEVAL
            # ---------------------------------------------

            if use_web:

                if web_configured is False:

                    error_message = (
                        "web: TAVILY_API_KEY is not configured"
                    )

                    logger.error(error_message)
                    errors.append(error_message)

                else:

                    logger.info(
                        "Starting Tavily web search."
                    )

                    tasks["web"] = executor.submit(
                        self.web_retriever.retrieve,
                        queries
                    )

            # ---------------------------------------------
            # COLLECT RESULTS
            # ---------------------------------------------

            for source, future in tasks.items():

                try:

                    logger.info(
                        "Waiting for %s retrieval results.",
                        source
                    )

                    retrieved = future.result()

                    if not isinstance(retrieved, list):

                        raise TypeError(
                            "Retriever must return a list."
                        )

                    candidates.extend(retrieved)

                    result[source]["result_count"] = len(retrieved)

                    logger.info(
                        "%s retrieval completed: %d results.",
                        source.upper(),
                        len(retrieved)
                    )

                except Exception as exc:

                    logger.exception(
                        "%s retrieval failed.",
                        source.upper()
                    )

                    errors.append(
                        f"{source}: {type(exc).__name__}: "
                        f"{str(exc)[:180]}"
                    )

        result["retrieved_count"] = len(candidates)

        logger.info(
            "Total retrieved candidates: %d",
            len(candidates)
        )

        logger.info(
            "Local results: %d | Web results: %d",
            result["local"]["result_count"],
            result["web"]["result_count"]
        )

        # =====================================================
        # 6. HANDLE EMPTY RESULTS
        # =====================================================

        if not candidates:

            result["errors"] = errors

            result["retrieval_status"] = (
                "failed" if errors else "no_results"
            )

            logger.warning(
                "No retrieval candidates were found."
            )

            if errors:
                logger.error(
                    "Retrieval errors: %s",
                    errors
                )

            logger.info(
                "Final retrieval status: %s",
                result["retrieval_status"]
            )

            logger.info("=" * 60)

            return result

        # =====================================================
        # 7. RELEVANCE EVALUATION
        # =====================================================

        logger.info(
            "Stage 4/6: Evaluating relevance of %d candidates.",
            len(candidates)
        )

        try:

            evaluated = self.relevance_evaluator.evaluate(
                user_query,
                candidates
            )

        except Exception:

            logger.exception(
                "Relevance evaluation failed."
            )

            raise

        logger.info(
            "Relevance evaluation completed: %d candidates retained.",
            len(evaluated)
        )

        # =====================================================
        # 8. CONTEXT SELECTION
        # =====================================================

        logger.info(
            "Stage 5/6: Selecting relevant context."
        )

        try:

            selected = self.context_selector.select(evaluated)

            context = self.context_selector.format_context(selected)

        except Exception:

            logger.exception(
                "Context selection or formatting failed."
            )

            raise

        logger.info(
            "Selected %d sources for final context.",
            len(selected)
        )

        logger.info(
            "Final context length: %d characters.",
            len(context or "")
        )

        # =====================================================
        # 9. BUILD FINAL RESULT
        # =====================================================

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

                    # Local retrieval metadata
                    "source": item.get("source"),
                    "topic": item.get("topic"),
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

        # =====================================================
        # 10. FINAL DIAGNOSTICS
        # =====================================================

        logger.info("Stage 6/6: Retrieval completed.")

        logger.info(
            "Final status: %s",
            result["retrieval_status"]
        )

        logger.info(
            "Knowledge used: %s",
            result["used"]
        )

        logger.info(
            "Sources selected: %d",
            result["selected_count"]
        )

        logger.info(
            "Errors encountered: %d",
            len(errors)
        )

        for index, source in enumerate(result["sources"], start=1):

            logger.info(
                "Source %d: type=%s, title=%s",
                index,
                source.get("source_type"),
                source.get("title")
            )

        if errors:

            logger.warning(
                "Retrieval completed with errors: %s",
                errors
            )

        logger.info("Knowledge RAG retrieval finished.")
        logger.info("=" * 60)

        return result

