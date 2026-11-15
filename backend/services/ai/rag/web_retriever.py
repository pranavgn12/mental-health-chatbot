import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
TAVILY_ENDPOINT = "https://api.tavily.com/search"


class WebRetriever:
    """Tavily-backed web search; invoked only when RAG_ENABLE_WEB=true."""

    def __init__(self, api_key=None, timeout=18, max_results=5):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout = timeout
        self.max_results = max(1, int(max_results))

    def is_configured(self):
        return bool(self.api_key)

    def _search_one(self, query):
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "advanced"),
            "topic": "general",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        request = Request(
            TAVILY_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "MentalHealthChatbot/1.0"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Tavily HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError("Tavily network error") from exc
        except (TimeoutError, ValueError) as exc:
            raise RuntimeError(f"Tavily {type(exc).__name__}") from exc

        if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
            raise RuntimeError("Tavily returned an invalid results payload")
        results = []
        for item in data.get("results", []):
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or item.get("raw_content") or "").strip()
            url = (item.get("url") or "").strip()
            if not content or not url.startswith(("https://", "http://")):
                continue
            results.append({
                "id": "",
                "source_type": "web",
                "title": (item.get("title") or url)[:300],
                "source": item.get("title") or url,
                "url": url,
                "content": content[:5000],
                "published_date": item.get("published_date"),
                "search_score": item.get("score"),
                "retrieval_query": query,
            })
        return results

    def retrieve(self, queries: list[str]) -> list[dict]:
        if not self.is_configured():
            raise RuntimeError("TAVILY_API_KEY is not configured")
        queries = list(dict.fromkeys(q.strip() for q in (queries or []) if isinstance(q, str) and q.strip()))
        if not queries:
            return []
        all_results, failures = [], []
        with ThreadPoolExecutor(max_workers=min(2, len(queries))) as executor:
            futures = {executor.submit(self._search_one, query): query for query in queries}
            for future in as_completed(futures):
                query = futures[future]
                try:
                    all_results.extend(future.result())
                except Exception as exc:
                    failures.append(f"{type(exc).__name__} for one query")
                    logger.exception("Tavily search failed for query %r", query)
        if not all_results and failures:
            raise RuntimeError("All Tavily searches failed: " + "; ".join(failures))
        unique = {}
        for item in all_results:
            unique.setdefault(item["url"], item)
        results = list(unique.values())
        for index, item in enumerate(results, start=1):
            item["id"] = f"web_{index}"
        return results
