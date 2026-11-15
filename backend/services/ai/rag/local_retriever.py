import logging
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from .identifier_utils import extract_identifiers, identifier_in_text

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DB_DIR = BACKEND_DIR / "vector_db"
COLLECTION_NAME = os.getenv("KNOWLEDGE_COLLECTION", "knowledge")
EMBEDDING_MODEL = os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_TOP_K = int(os.getenv("KNOWLEDGE_TOP_K", "12"))
RRF_K = int(os.getenv("KNOWLEDGE_RRF_K", "60"))

# Tokenizer keeps compact alphanumeric document references together.
_TOKEN_RE = re.compile(r"[a-z]+\d+[a-z0-9]*|[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)


def _tokens(text):
    raw_tokens = [token.casefold() for token in _TOKEN_RE.findall(text or "")]
    # Add canonical document-reference tokens so variants such as RFC-9110 and
    # RFC9110, or "Table 4" and TABLE#4, share a lexical search term.
    canonical_ids = [
        identifier.casefold().replace("#", "")
        for identifier in extract_identifiers(text or "")
    ]
    return raw_tokens + canonical_ids


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    db_path = Path(os.getenv("CHROMA_DB_DIR", str(DEFAULT_DB_DIR))).expanduser()
    if not db_path.is_absolute():
        db_path = (BACKEND_DIR / db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"ChromaDB directory does not exist: {db_path}")

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)


class _BM25Index:
    """Small dependency-free BM25 index built from the already indexed Chroma collection."""

    def __init__(self, ids, documents, metadatas, k1=1.5, b=0.75):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.k1 = k1
        self.b = b
        self.tokens = [_tokens(doc) for doc in documents]
        self.lengths = [len(row) for row in self.tokens]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.doc_freq = Counter()
        for row in self.tokens:
            self.doc_freq.update(set(row))
        self.n_docs = len(self.tokens)
        self.postings = defaultdict(list)
        # Corpus-derived ID vocabulary: no fixed prefix list or domain-specific
        # assumptions. These identifiers are used to validate query IDs.
        self.identifier_catalog = set()
        for document, metadata in zip(documents, metadatas):
            self.identifier_catalog.update(extract_identifiers(document))
            # Structural metadata can contain references not repeated in the
            # chunk body (e.g. a section identifier supplied by the parser).
            for key, value in (metadata or {}).items():
                if isinstance(value, str):
                    self.identifier_catalog.update(extract_identifiers(value))
                    # Use field names as labels for structured references, e.g.
                    # metadata.section="2.1" becomes SECTION#2.1.
                    if key.casefold() in {
                        "section", "subsection", "chapter", "table", "figure",
                        "recommendation", "equation", "appendix", "annex",
                        "article", "clause", "requirement", "control", "question",
                    }:
                        self.identifier_catalog.update(extract_identifiers(f"{key} {value}"))
        for doc_index, row in enumerate(self.tokens):
            counts = Counter(row)
            for token, frequency in counts.items():
                self.postings[token].append((doc_index, frequency))

    def search(self, query, top_k):
        query_tokens = _tokens(query)
        if not query_tokens or not self.n_docs:
            return []
        query_counts = Counter(query_tokens)
        scores = defaultdict(float)
        for token, qtf in query_counts.items():
            df = self.doc_freq.get(token, 0)
            if not df:
                continue
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for doc_index, tf in self.postings.get(token, []):
                dl = self.lengths[doc_index]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
                scores[doc_index] += idf * ((tf * (self.k1 + 1.0)) / denom) * min(qtf, 2)
        return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]


@lru_cache(maxsize=1)
def _get_bm25_index():
    collection = _get_collection()
    count = collection.count()
    if count <= 0:
        return _BM25Index([], [], [])

    # The collection is loaded once per process. This means the existing
    # ChromaDB index is reused; no separate ingestion or re-indexing is needed.
    data = collection.get(include=["documents", "metadatas"], limit=count)
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or [{} for _ in documents]
    return _BM25Index(ids, documents, metadatas)


class LocalRetriever:
    """Hybrid retrieval over the existing collection: Chroma vector + in-memory BM25."""

    def get_identifier_catalog(self):
        """Return generic identifiers discovered in the currently indexed corpus."""
        return frozenset(_get_bm25_index().identifier_catalog)

    def retrieve(self, queries: list[str], top_k: int = DEFAULT_TOP_K) -> list[dict]:
        queries = list(dict.fromkeys(
            q.strip() for q in (queries or []) if isinstance(q, str) and q.strip()
        ))
        if not queries:
            return []

        collection = _get_collection()
        count = collection.count()
        if count <= 0:
            return []

        top_k = max(1, min(int(top_k), count))
        bm25 = _get_bm25_index()
        candidates = {}
        fused_scores = defaultdict(float)
        vector_distances = {}
        bm25_scores = {}
        match_methods = defaultdict(set)
        metadata_by_id = {}
        content_by_id = {}
        source_by_id = {}

        def add_ranked(rows, method):
            for rank, row in enumerate(rows, start=1):
                chroma_id = row["chroma_id"]
                fused_scores[chroma_id] += 1.0 / (RRF_K + rank)
                match_methods[chroma_id].add(method)
                content_by_id[chroma_id] = row["content"]
                metadata_by_id[chroma_id] = row["metadata"]
                source_by_id[chroma_id] = row["source"]
                if row.get("distance") is not None:
                    old = vector_distances.get(chroma_id)
                    if old is None or row["distance"] < old:
                        vector_distances[chroma_id] = row["distance"]
                if row.get("bm25_score") is not None:
                    bm25_scores[chroma_id] = max(bm25_scores.get(chroma_id, 0.0), row["bm25_score"])

        for query_index, query in enumerate(queries):
            vector_rows = []
            vector_result = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            result_ids = (vector_result.get("ids") or [[]])[0]
            result_docs = (vector_result.get("documents") or [[]])[0]
            result_metas = (vector_result.get("metadatas") or [[]])[0]
            result_distances = (vector_result.get("distances") or [[]])[0]
            for i, chroma_id in enumerate(result_ids):
                content = result_docs[i] if i < len(result_docs) else None
                if not content or not content.strip():
                    continue
                metadata = result_metas[i] if i < len(result_metas) and result_metas[i] else {}
                distance = result_distances[i] if i < len(result_distances) else None
                vector_rows.append({
                    "chroma_id": chroma_id,
                    "content": content.strip(),
                    "metadata": metadata,
                    "source": metadata.get("source", "local knowledge base"),
                    "distance": float(distance) if distance is not None else None,
                })
            add_ranked(vector_rows, "vector")

            lexical_rows = []
            for doc_index, score in bm25.search(query, top_k):
                content = bm25.documents[doc_index]
                if not content or not content.strip():
                    continue
                metadata = bm25.metadatas[doc_index] or {}
                lexical_rows.append({
                    "chroma_id": bm25.ids[doc_index],
                    "content": content.strip(),
                    "metadata": metadata,
                    "source": metadata.get("source", "local knowledge base"),
                    "bm25_score": float(score),
                })
            add_ranked(lexical_rows, "bm25")

        # Exact corpus-known document identifiers receive a modest deterministic
        # boost. The catalogue is extracted from indexed text, not hard-coded.
        requested_ids = set(extract_identifiers(
            " ".join(queries), known_identifiers=bm25.identifier_catalog
        ))
        if requested_ids:
            for chroma_id, content in content_by_id.items():
                content_ids = set(extract_identifiers(content))
                if requested_ids.intersection(content_ids):
                    fused_scores[chroma_id] += 0.025

        if not fused_scores:
            return []

        max_score = max(fused_scores.values()) or 1.0
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        for chroma_id in ranked_ids:
            metadata = metadata_by_id[chroma_id] or {}
            content = content_by_id[chroma_id]
            base_id = str(chroma_id)
            candidate_id = "local_" + re.sub(r"[^A-Za-z0-9_-]", "_", base_id)[:48]
            candidate = {
                "id": candidate_id,
                "source_type": "local",
                "title": metadata.get("document_title") or metadata.get("source") or "Local knowledge source",
                "source": metadata.get("source", "local knowledge base"),
                "url": None,
                "content": content,
                "topic": metadata.get("topic"),
                "distance": vector_distances.get(chroma_id),
                "retrieval_query": None,
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "heading_path": metadata.get("heading_path"),
                "module": metadata.get("module"),
                "section": metadata.get("section"),
                "chunk": metadata.get("chunk", metadata.get("chunk_index")),
                "content_type": metadata.get("content_type", "text"),
                "chroma_id": base_id,
                "retrieval_score": fused_scores[chroma_id] / max_score,
                "bm25_score": bm25_scores.get(chroma_id),
                "retrieval_methods": sorted(match_methods[chroma_id]),
                "exact_id_match": bool(requested_ids and any(
                    identifier_in_text(identifier, content) for identifier in requested_ids
                )),
            }
            candidates[candidate_id] = candidate

        return sorted(candidates.values(), key=lambda item: item["retrieval_score"], reverse=True)[:max(top_k * 2, 16)]
