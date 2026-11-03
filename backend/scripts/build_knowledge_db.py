"""Build/update the local ChromaDB knowledge collection.

Run from the backend directory:
    python scripts/build_knowledge_db.py

To intentionally rebuild the complete collection:
    python scripts/build_knowledge_db.py --rebuild
"""

import argparse
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import markdown
from bs4 import BeautifulSoup

BACKEND_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BACKEND_DIR / "knowledge"
DB_DIR = BACKEND_DIR / "vector_db"
COLLECTION_NAME = "knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 1200
OVERLAP = 180


def markdown_to_text(source: str) -> str:
    html = markdown.markdown(source)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n", strip=True)


def chunk_text(text: str) -> list[str]:
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a paragraph/sentence boundary where possible.
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start + CHUNK_SIZE // 2, end),
            )
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(start + 1, end - OVERLAP)

    return chunks


def stable_id(source_name: str, chunk_index: int, content: str) -> str:
    raw = f"{source_name}:{chunk_index}:{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the entire knowledge collection.",
    )
    args = parser.parse_args()

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {KNOWLEDGE_DIR}")

    source_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not source_files:
        raise RuntimeError(f"No Markdown files found in {KNOWLEDGE_DIR}")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing knowledge collection (--rebuild).")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "l2"},
    )

    total_chunks = 0
    for file in source_files:
        print(f"Processing {file.name}")
        text = markdown_to_text(file.read_text(encoding="utf-8"))
        chunks = chunk_text(text)

        # Remove stale chunks for this source before inserting its current chunks.
        collection.delete(where={"source": file.name})

        ids, documents, metadatas = [], [], []
        for index, chunk in enumerate(chunks):
            ids.append(stable_id(file.name, index, chunk))
            documents.append(chunk)
            metadatas.append({
                "topic": file.stem,
                "source": file.name,
                "chunk": index,
            })

        if ids:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
        total_chunks += len(ids)

    print("\nKnowledge database updated.")
    print(f"Markdown files: {len(source_files)}")
    print(f"Chunks indexed: {total_chunks}")
    print(f"Collection count: {collection.count()}")
    print(f"Database path: {DB_DIR}")


if __name__ == "__main__":
    main()
