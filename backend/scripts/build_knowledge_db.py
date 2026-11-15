"""
Build/update the local ChromaDB knowledge collection from PDF and Markdown files.

Run from the backend directory:
    python scripts/build_knowledge_db.py

To intentionally rebuild the complete collection:
    python scripts/build_knowledge_db.py --rebuild

Supported input formats:
    .pdf
    .md

PDFs and Markdown files are converted into DoclingDocument objects and chunked
with Docling's HybridChunker. Chunk text includes its heading context, while
metadata preserves source, heading path, content type, and PDF page numbers
when available.
"""

import argparse
import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from transformers import AutoTokenizer


BACKEND_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BACKEND_DIR / "knowledge"
DB_DIR = BACKEND_DIR / "vector_db"

COLLECTION_NAME = "knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Keep below the embedding model's 512-token sequence limit to leave room for
# heading context and special tokens added during embedding.
CHUNK_MAX_TOKENS = 450

SUPPORTED_EXTENSIONS = {".pdf", ".md"}


def stable_id(source_name: str, chunk_index: int, content: str) -> str:
    """Generate a deterministic ID for a source/chunk pair."""
    raw = f"{source_name}:{chunk_index}:{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_chunk_metadata(chunk: Any, source_file: Path, chunk_index: int) -> dict:
    """Extract Chroma-compatible metadata from a Docling chunk."""
    headings = list(getattr(chunk.meta, "headings", None) or [])
    doc_items = list(getattr(chunk.meta, "doc_items", None) or [])

    page_numbers = set()
    labels = set()

    for item in doc_items:
        label = getattr(item, "label", None)
        if label is not None:
            labels.add(str(getattr(label, "value", label)).lower())

        for provenance in (getattr(item, "prov", None) or []):
            page_no = getattr(provenance, "page_no", None)
            if page_no is not None:
                page_numbers.add(int(page_no))

    # A chunk can contain multiple item types. Give tables priority so they
    # can be identified later during retrieval or prompt construction.
    if any("table" in label for label in labels):
        content_type = "table"
    elif any("list" in label for label in labels):
        content_type = "list"
    else:
        content_type = "text"

    heading_path = " > ".join(str(h) for h in headings if str(h).strip())

    metadata = {
        "source": source_file.name,
        "topic": str(headings[-1]) if headings else source_file.stem,
        "document_title": source_file.stem,
        "heading_path": heading_path or "Unspecified",
        "module": str(headings[0]) if headings else "Unspecified",
        "section": str(headings[-1]) if headings else "Unspecified",
        "chunk": int(chunk_index),
        "content_type": content_type,
        "file_type": source_file.suffix.lower().lstrip("."),
    }

    if page_numbers:
        metadata["page_start"] = min(page_numbers)
        metadata["page_end"] = max(page_numbers)

    return metadata


def create_docling_components():
    """
    Create reusable Docling converter and chunker.

    The chunker's tokenizer is aligned with the same Hugging Face model used
    by ChromaDB's SentenceTransformerEmbeddingFunction.
    """
    converter = DocumentConverter()

    hf_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    docling_tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=CHUNK_MAX_TOKENS,
    )

    chunker = HybridChunker(
        tokenizer=docling_tokenizer,
        merge_peers=True,
    )

    return converter, chunker


def process_document(
    source_file: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
) -> tuple[list[str], list[str], list[dict]]:
    """Convert one PDF/Markdown file and return IDs, texts, and metadata."""
    print(f"Processing {source_file.name}...")

    conversion_result = converter.convert(source=str(source_file))
    doc = conversion_result.document

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for index, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        # contextualize() adds the document heading hierarchy to the chunk.
        # This is the text that should be embedded and stored for retrieval.
        content = chunker.contextualize(chunk=chunk).strip()

        if not content:
            continue

        ids.append(stable_id(source_file.name, index, content))
        documents.append(content)
        metadatas.append(get_chunk_metadata(chunk, source_file, index))

    # Release conversion resources after all chunks have been extracted.
    input_resource = getattr(conversion_result, "input", None)
    if input_resource is not None:
        try:
            input_resource.release()
        except Exception:
            pass

    if not documents:
        raise RuntimeError(
            f"No usable chunks were extracted from {source_file.name}. "
            "The existing indexed version of this source has been left untouched."
        )

    print(f"  Extracted {len(documents)} chunks.")
    return ids, documents, metadatas


def main():
    parser = argparse.ArgumentParser(
        description="Build/update the local ChromaDB knowledge collection."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the entire knowledge collection.",
    )
    args = parser.parse_args()

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {KNOWLEDGE_DIR}"
        )

    source_files = sorted(
        file
        for file in KNOWLEDGE_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not source_files:
        raise RuntimeError(
            f"No PDF or Markdown files found in {KNOWLEDGE_DIR}"
        )

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

    # Reuse one converter and chunker for all documents. This avoids repeatedly
    # loading Docling's models for every PDF/Markdown file.
    converter, chunker = create_docling_components()

    total_chunks = 0

    for source_file in source_files:
        # Process first. If parsing fails, the existing source chunks remain
        # in ChromaDB because deletion happens only after successful extraction.
        ids, documents, metadatas = process_document(
            source_file=source_file,
            converter=converter,
            chunker=chunker,
        )

        # Preserve the original incremental-update behavior: replace only the
        # chunks belonging to the source currently being processed.
        collection.delete(where={"source": source_file.name})

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        total_chunks += len(ids)

    print("\nKnowledge database updated.")
    print(f"Source files processed: {len(source_files)}")
    print(f"Chunks indexed this run: {total_chunks}")
    print(f"Collection count: {collection.count()}")
    print(f"Database path: {DB_DIR}")


if __name__ == "__main__":
    main()
