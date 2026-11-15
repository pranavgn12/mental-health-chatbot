# Knowledge RAG v2 — corpus-aware edition

A replacement for the existing knowledge RAG package. It keeps the same module-oriented layout, the same `KnowledgeRAGPipeline.retrieve(user_query)` entry point, and the existing top-level result and `sources` keys. Additional local provenance fields are optional additions; the existing reference fields are preserved.

## What's included

- Hybrid retrieval over the existing ChromaDB collection: vector similarity plus a dependency-free BM25 index.
- Reciprocal Rank Fusion (RRF) combines vector and lexical rankings without treating incompatible raw scores as directly comparable.
- No fixed L2 distance cutoff by default. Distances remain available as diagnostics.
- **Generic, corpus-aware identifier handling.** At process startup, the retriever scans indexed chunks and builds an identifier catalogue. It recognises document-specific alphanumeric identifiers (for example `REF-12`, `RFC9110`, `Q3`) and labelled references (for example `Section 2.1`, `Table 4`, `Figure 3`, `Recommendation 5`). It does not contain a WHO/mhGAP prefix list.
- Numbered ranges with a shared prefix/label can be expanded, but when a corpus catalogue is available only identifiers actually present in the indexed collection are expanded and boosted.
- Exact identifier coverage is included in deterministic relevance scoring and context selection.
- Context selection uses full-content hashes for deduplication and does not silently truncate selected chunks.
- Local page, section, source, and content-type metadata flow through to the context and `sources` objects.
- Web search is disabled unless `RAG_ENABLE_WEB=true`.

## Existing collection compatibility

No re-indexing is required just to use the new identifier detection or hybrid retriever. The BM25 index and identifier catalogue are built from the documents already stored in ChromaDB. The configured collection path, collection name and embedding model must match the values used by the existing indexer.

After adding documents or changing the collection, restart the backend process so the in-memory BM25 index and identifier catalogue are refreshed.

## Installation

```bash
pip install chromadb sentence-transformers python-dotenv
```

No separate BM25 package is required. Docling is only needed when ingesting or rebuilding PDF documents, not for runtime retrieval.

## Integration

Replace the files in your existing RAG package with the contents of this directory, preserving the package directory name and call sites:

```python
from your_existing_package import KnowledgeRAGPipeline

rag = KnowledgeRAGPipeline()
result = rag.retrieve(user_query)
```

`retrieve()` preserves the existing top-level keys:

`used`, `decision`, `search_queries`, `context`, `sources`, `retrieved_count`, `selected_count`, `retrieval_status`, `errors`, `web`, `local`.

Existing source keys remain:

`source_type`, `title`, `url`, `published_date`, `relevance`, `source`, `topic`.

Additional optional local fields:

`page_start`, `page_end`, `heading_path`, `module`, `section`, `chunk`, `content_type`.

## Environment variables

```env
KNOWLEDGE_COLLECTION=knowledge
KNOWLEDGE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHROMA_DB_DIR=vector_db
KNOWLEDGE_TOP_K=12
KNOWLEDGE_RRF_K=60
RAG_ENABLE_WEB=false
RAG_LLM_RELEVANCE_ENABLED=false
RAG_MAX_QUERIES=8
RAG_MAX_CONTEXT_ITEMS=10
RAG_MAX_CONTEXT_CHARS=16000
```

## Inspect the existing indexed corpus

Print the identifiers discovered from all indexed chunks:

```bash
python -m your_existing_package.inspect_knowledge_chunks --show-identifiers
```

Inspect chunks containing specific identifiers or references:

```bash
python -m your_existing_package.inspect_knowledge_chunks --source "paper.pdf" --ids "Table 2" REF-3 "Section 4.1"
```

Inspect chunks containing a text phrase:

```bash
python -m your_existing_package.inspect_knowledge_chunks --search "confidence interval" --limit 10
```

The older module name `inspect_knowledge_tables` remains as a compatibility alias, but the inspector is now document-agnostic.

## Notes and limitations

- The identifier catalogue is heuristic and corpus-derived, not a semantic document parser. It is designed to identify common alphanumeric codes and labelled structural references. Unusual reference formats may require adding a pattern to `identifier_utils.py`; this does not require a hard-coded document-specific prefix list.
- If the PDF ingestion stage dropped table cells or failed to preserve their row association, retrieval cannot restore the missing information. Inspect the stored chunks before attributing an answer error to retrieval.
- The in-memory BM25/index catalogue is cached for the life of the process. Restart after re-indexing.
- This is a retrieval component, not a clinical decision system. Generated answers should remain grounded in retrieved evidence and continue to use the application's separate safety handling.
