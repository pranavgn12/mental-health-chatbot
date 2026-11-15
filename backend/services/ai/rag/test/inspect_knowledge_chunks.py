"""Inspect arbitrary indexed knowledge chunks and their provenance.

Run from the backend directory, for example:
    python -m your_package.inspect_knowledge_chunks --source "paper.pdf" --ids "Table 2" REF-3
    python -m your_package.inspect_knowledge_chunks --show-identifiers

Uses the same ChromaDB environment variables as LocalRetriever.
"""

import argparse
import os
from pathlib import Path

from .identifier_utils import extract_identifiers, identifier_in_text


def _connect():
    import chromadb

    base = Path(__file__).resolve().parents[3]
    db_path = Path(os.getenv("CHROMA_DB_DIR", str(base / "vector_db"))).expanduser()
    if not db_path.is_absolute():
        db_path = (base / db_path).resolve()
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=os.getenv("KNOWLEDGE_COLLECTION", "knowledge"))
    return collection


def main():
    parser = argparse.ArgumentParser(description="Inspect chunks in the local knowledge collection.")
    parser.add_argument("--source", help="Only inspect chunks from this source filename.")
    parser.add_argument("--ids", nargs="+", help="Document identifiers, e.g. REF-3, Section 2.1, Table 4.")
    parser.add_argument("--search", help="Substring to find in chunk text.")
    parser.add_argument("--show-identifiers", action="store_true", help="Print the corpus-derived identifier catalogue.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum chunks to print (default: 20).")
    args = parser.parse_args()

    collection = _connect()
    count = collection.count()
    if count == 0:
        print("The knowledge collection is empty.")
        return

    data = collection.get(include=["documents", "metadatas"], limit=count)
    chroma_ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or [{} for _ in documents]

    if args.show_identifiers:
        catalogue = set()
        for document in documents:
            catalogue.update(extract_identifiers(document or ""))
        print(f"Discovered {len(catalogue)} identifiers:")
        for identifier in sorted(catalogue):
            print(identifier)
        if not args.ids and not args.search:
            return

    requested = []
    for raw_id in args.ids or []:
        extracted = extract_identifiers(raw_id)
        requested.extend(extracted or [raw_id.strip().upper()])
    requested = list(dict.fromkeys(requested))

    printed = 0
    for chroma_id, document, metadata in zip(chroma_ids, documents, metadatas):
        metadata = metadata or {}
        document = document or ""
        if args.source and metadata.get("source") != args.source:
            continue
        if requested and not any(identifier_in_text(item, document) for item in requested):
            continue
        if args.search and args.search.casefold() not in document.casefold():
            continue

        printed += 1
        print("\n" + "=" * 88)
        print(f"Chunk {printed} | Chroma ID: {chroma_id}")
        for key in (
            "source", "document_title", "page_start", "page_end", "heading_path",
            "module", "section", "chunk", "content_type", "file_type",
        ):
            if key in metadata:
                print(f"{key}: {metadata[key]}")
        print("\n--- FULL STORED CHUNK ---\n")
        print(document)
        if printed >= max(1, args.limit):
            break

    if not printed:
        print("No matching chunks found. Check the source filename, identifier spelling, or stored metadata.")


if __name__ == "__main__":
    main()
