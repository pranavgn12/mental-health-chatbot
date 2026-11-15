"""Inspect existing ChromaDB chunks for WHO mhGAP recommendation table content.

Run from the backend directory:
    python -m path.to.knowledge_rag_v2.inspect_knowledge_tables --source "WHO mhGAP.pdf"

Adjust CHROMA_DB_DIR, KNOWLEDGE_COLLECTION and KNOWLEDGE_EMBEDDING_MODEL
using the same environment variables as the application.
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect indexed WHO recommendation chunks.")
    parser.add_argument("--source", default="WHO mhGAP.pdf")
    parser.add_argument("--ids", nargs="+", default=["DEP1", "DEP2", "DEP3", "DEP4"])
    args = parser.parse_args()

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    base = Path(__file__).resolve().parents[3]
    db_path = Path(os.getenv("CHROMA_DB_DIR", str(base / "vector_db"))).expanduser()
    if not db_path.is_absolute():
        db_path = (base / db_path).resolve()
    embedding = SentenceTransformerEmbeddingFunction(
        model_name=os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(
        name=os.getenv("KNOWLEDGE_COLLECTION", "knowledge"),
        embedding_function=embedding,
    )

    count = collection.count()
    data = collection.get(include=["documents", "metadatas"], limit=count)
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or [{} for _ in docs]
    found_any = False

    for recommendation_id in args.ids:
        print("\n" + "=" * 90)
        print(f"SEARCHING FOR {recommendation_id}")
        print("=" * 90)
        matches = []
        for chroma_id, document, metadata in zip(ids, docs, metas):
            metadata = metadata or {}
            if args.source and metadata.get("source") != args.source:
                continue
            if recommendation_id.casefold() in (document or "").casefold():
                matches.append((chroma_id, document, metadata))
        if not matches:
            print("NOT FOUND in matching source chunks.")
            continue
        found_any = True
        for chroma_id, document, metadata in matches:
            print(f"\nChroma ID: {chroma_id}")
            for key in ("source", "page_start", "page_end", "heading_path", "module", "section", "chunk", "content_type"):
                if key in metadata:
                    print(f"{key}: {metadata[key]}")
            print("\n--- FULL STORED CHUNK ---\n")
            print(document)
            print("\n--- END CHUNK ---")
    if not found_any:
        print("\nNo requested recommendation IDs were found. Check --source spelling or inspect the source metadata.")


if __name__ == "__main__":
    main()
