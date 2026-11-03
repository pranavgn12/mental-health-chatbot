import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DB_DIR = BACKEND_DIR / "vector_db"
COLLECTION_NAME = os.getenv("KNOWLEDGE_COLLECTION", "knowledge")
EMBEDDING_MODEL = os.getenv(
    "KNOWLEDGE_EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)
DISTANCE_THRESHOLD = float(os.getenv("KNOWLEDGE_DISTANCE_THRESHOLD", "0.60"))


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    db_path = Path(os.getenv("CHROMA_DB_DIR", str(DEFAULT_DB_DIR))).expanduser()
    if not db_path.is_absolute():
        db_path = (BACKEND_DIR / db_path).resolve()

    if not db_path.exists():
        raise FileNotFoundError(f"ChromaDB directory does not exist: {db_path}")

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )


class LocalRetriever:
    """Queries the existing ChromaDB 'knowledge' collection."""

    def retrieve(self, queries: list[str], top_k: int = 4) -> list[dict]:
        queries = [q.strip() for q in (queries or []) if isinstance(q, str) and q.strip()]
        if not queries:
            return []

        collection = _get_collection()
        result = collection.query(
            query_texts=queries,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        found = {}
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        distances = result.get("distances") or []

        for query_index, query_docs in enumerate(documents):
            query_metas = metadatas[query_index] if query_index < len(metadatas) else []
            query_distances = distances[query_index] if query_index < len(distances) else []

            for index, content in enumerate(query_docs or []):
                if not content or not content.strip():
                    continue
                distance = (
                    float(query_distances[index])
                    if index < len(query_distances) and query_distances[index] is not None
                    else None
                )
                if distance is not None and distance > DISTANCE_THRESHOLD:
                    continue

                metadata = query_metas[index] if index < len(query_metas) else {}
                source = metadata.get("source", "local knowledge base")
                chunk = metadata.get("chunk", index)
                key = (source, str(chunk), content[:160])

                candidate = {
                    "id": f"local_{len(found) + 1}",
                    "source_type": "local",
                    "title": source,
                    "source": source,
                    "url": None,
                    "content": content.strip(),
                    "topic": metadata.get("topic"),
                    "distance": distance,
                    "retrieval_query": query_index,
                }

                previous = found.get(key)
                if previous is None or (
                    distance is not None
                    and (previous["distance"] is None or distance < previous["distance"])
                ):
                    found[key] = candidate

        return list(found.values())
