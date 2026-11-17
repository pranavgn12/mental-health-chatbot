#!/usr/bin/env python3
"""Build/update a Docling + HybridChunker + BGE + ChromaDB knowledge index.

Place this file at ``backend/scripts/build_knowledge_db.py`` and run from
``backend/``:

    python scripts/build_knowledge_db.py --dry-run
    python scripts/build_knowledge_db.py
    python scripts/build_knowledge_db.py --rebuild

Design goals
------------
* General-purpose PDF/Markdown ingestion; no document-specific page regexes.
* Docling owns layout, reading order, OCR and table structure extraction.
* HybridChunker uses the same HF tokenizer as the embedding model.
* Tables use Markdown serialization (with repeated headers) rather than the
  default triplet serialization, which can introduce row ordinals and obscure
  column relationships for tables without row headers.
* Every final chunk is token-counted with the exact embedding tokenizer.
* Oversized final chunks are split without truncating source text.
* Page furniture is not deleted by guessing from numeric text. Docling keeps
  furniture separate from the body; the body is chunked and furniture is
  reported for diagnostics.
* Full chunk text is written to JSONL for review; reports include all warnings.
* Incremental updates upsert new chunks before deleting stale chunks.
* --rebuild does not delete the existing collection until every source has
  extracted and passed validation.

The Transformers warning about an oversized sequence can be emitted while
HybridChunker counts an entire table before splitting it. It is not proof that
that oversized table was sent to the embedding model. The final token audit in
this script is the authoritative check for stored chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BACKEND_DIR / "knowledge"
DB_DIR = BACKEND_DIR / "vector_db"
REPORT_DIR = BACKEND_DIR / "knowledge_extraction_reports"

COLLECTION_NAME = "knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MAX_TOKENS = 512
TARGET_CHUNK_TOKENS = 450  # Includes special tokens and contextualized headings.
CHROMA_BATCH_SIZE = 256
SUPPORTED_EXTENSIONS = {".pdf", ".md"}

MULTISPACE_RE = re.compile(r"[ \t]+")
MULTIBLANK_RE = re.compile(r"\n{3,}")
MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
CHECKBOX_RE = re.compile(r"(?:\[[ xX]?\]|☐|☑|☒|□|■)")
TOC_TRIPLET_ARTIFACT_RE = re.compile(r",\s*\d{1,4}\s*=\s*(?:[ivxlcdm]+|\d{1,4})\b", re.IGNORECASE)
CALLOUT_TITLE_RE = re.compile(
    r"^\s*(?:key messages?|key points?|important|note|warning|caution|box\s+\d+(?:\.\d+)?|tip|clinical tip|practice point)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dependency/version helpers
# ---------------------------------------------------------------------------


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def stable_id(source_name: str, chunk_index: int, content: str) -> str:
    """Stable source/chunk ID suitable for ChromaDB."""
    raw = f"{source_name}:{chunk_index}:{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def item_label(item: Any) -> str:
    label = getattr(item, "label", None)
    return str(getattr(label, "value", label) or "").casefold().replace("_", "").replace(" ", "")


def item_text(item: Any, doc: Any = None) -> str:
    value = getattr(item, "text", None)
    if value:
        return str(value).strip()
    exporter = getattr(item, "export_to_markdown", None)
    if callable(exporter) and doc is not None:
        try:
            return str(exporter(doc=doc)).strip()
        except Exception:
            pass
    return ""


def iter_doc_items(doc: Any, root: Any = None) -> Iterable[Any]:
    """Iterate Docling items while tolerating small API differences."""
    iterator = getattr(doc, "iterate_items", None)
    if callable(iterator):
        try:
            kwargs = {"root": root} if root is not None else {}
            for entry in iterator(**kwargs):
                yield entry[0] if isinstance(entry, tuple) else entry
            return
        except TypeError:
            try:
                for entry in iterator():
                    yield entry[0] if isinstance(entry, tuple) else entry
                return
            except Exception:
                pass
        except Exception:
            pass

    # Fallback for simple/older document models.
    attributes = ("texts", "tables", "pictures", "groups") if root is None else ("texts", "tables", "pictures", "groups")
    seen: set[int] = set()
    for attribute in attributes:
        for item in getattr(doc, attribute, None) or []:
            if id(item) not in seen:
                seen.add(id(item))
                yield item


def get_page_count(doc: Any) -> int:
    count = getattr(doc, "num_pages", None)
    if isinstance(count, int):
        return count
    pages = getattr(doc, "pages", None)
    try:
        return len(pages)
    except Exception:
        return 0


def get_item_pages(item: Any) -> set[int]:
    pages: set[int] = set()
    for provenance in getattr(item, "prov", None) or []:
        page_no = getattr(provenance, "page_no", None)
        try:
            if page_no is not None:
                pages.add(int(page_no))
        except (TypeError, ValueError):
            continue
    return pages


def inspect_furniture(doc: Any) -> dict[str, Any]:
    """Inspect Docling's furniture layer without deleting text by its value.

    Docling normally excludes furniture from body iteration. We explicitly ask
    for both layers here so the report can show what was detected. A label-based
    footer/header safeguard is still useful for versions where an item is
    incorrectly attached to a body group.
    """
    iterator = getattr(doc, "iterate_items", None)
    if not callable(iterator):
        return {
            "available": False,
            "items_detected": 0,
            "page_headers": 0,
            "page_footers": 0,
            "note": "This Docling version does not expose iterate_items; no numeric text was removed heuristically.",
        }

    try:
        from docling_core.types.doc.document import ContentLayer
        entries = iterator(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE})
    except (ImportError, TypeError, AttributeError):
        try:
            entries = iterator()
        except Exception as exc:
            return {
                "available": False,
                "items_detected": 0,
                "page_headers": 0,
                "page_footers": 0,
                "note": f"Furniture inspection unavailable ({exc}); no numeric text was removed heuristically.",
            }

    labels: Counter[str] = Counter()
    furniture_items = 0
    for entry in entries:
        item = entry[0] if isinstance(entry, tuple) else entry
        label = item_label(item)
        layer = getattr(item, "content_layer", None)
        layer_value = str(getattr(layer, "value", layer) or "").casefold()
        if layer_value == "furniture" or label in {"pageheader", "pagefooter"}:
            furniture_items += 1
            labels[label] += 1

    return {
        "available": True,
        "items_detected": furniture_items,
        "page_headers": labels.get("pageheader", 0),
        "page_footers": labels.get("pagefooter", 0),
        "note": "Docling furniture items are reported separately. No arbitrary standalone numbers or content lines are deleted.",
    }


# ---------------------------------------------------------------------------
# Text normalization and exact token accounting
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Conservative whitespace cleanup; preserve numbers and hyphens."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")  # soft hyphen only; visible hyphens remain.
    # Collapse repeated horizontal whitespace but preserve line boundaries.
    text = "\n".join(MULTISPACE_RE.sub(" ", line).rstrip() for line in text.split("\n"))
    text = MULTIBLANK_RE.sub("\n\n", text)
    return text.strip()


def count_tokens(tokenizer: Any, text: str) -> int:
    """Count tokens as the embedding tokenizer will see them (incl. specials)."""
    if not text:
        return 0

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)
    except Exception:
        pass

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return len(encode(text, add_special_tokens=True))
        except TypeError:
            return len(encode(text))

    raise TypeError("Tokenizer must be callable or expose encode(text).")


def _split_into_whitespace_atoms(text: str) -> list[str]:
    """Split while retaining every whitespace character, for lossless packing."""
    return re.findall(r"\S+\s*", text)


def _split_oversized_atom(atom: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Rare fallback for a single very long token/URL; split at character edges."""
    if count_tokens(tokenizer, atom) <= max_tokens:
        return [atom]

    pieces: list[str] = []
    remaining = atom
    while remaining:
        low, high = 1, len(remaining)
        best = 0
        while low <= high:
            mid = (low + high) // 2
            if count_tokens(tokenizer, remaining[:mid]) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best <= 0:
            raise ValueError(
                "The tokenizer cannot fit even one character within the configured token budget."
            )
        pieces.append(remaining[:best])
        remaining = remaining[best:]
    return pieces


def split_plain_text_to_budget(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Losslessly split arbitrary text; never truncate tokens or characters."""
    text = text.strip()
    if not text:
        return []
    if count_tokens(tokenizer, text) <= max_tokens:
        return [text]

    atoms = _split_into_whitespace_atoms(text)
    chunks: list[str] = []
    current = ""

    for atom in atoms:
        if count_tokens(tokenizer, atom) > max_tokens:
            if current:
                chunks.append(current.rstrip())
                current = ""
            long_pieces = _split_oversized_atom(atom, tokenizer, max_tokens)
            chunks.extend(piece for piece in long_pieces[:-1])
            current = long_pieces[-1]
            continue

        candidate = current + atom
        if current and count_tokens(tokenizer, candidate) > max_tokens:
            chunks.append(current.rstrip())
            current = atom
        else:
            current = candidate

    if current.strip():
        chunks.append(current.rstrip())

    # Defensive postcondition: the function must never return an overflow.
    for chunk in chunks:
        actual = count_tokens(tokenizer, chunk)
        if actual > max_tokens:
            raise RuntimeError(
                f"Internal chunk-splitting failure: {actual} tokens exceeds {max_tokens}."
            )
    return chunks


def split_markdown_table_to_budget(
    text: str,
    tokenizer: Any,
    max_tokens: int,
) -> list[str] | None:
    """Split Markdown tables by rows and repeat table headers in every part.

    Returns None when the input is not a recognizable Markdown table.
    Wide/oversized individual rows fall back to labelled row-cell text so the
    row key and column name remain attached to each split value.
    """
    lines = text.splitlines()
    header_index = None
    for index in range(len(lines) - 1):
        if "|" in lines[index] and MARKDOWN_TABLE_SEPARATOR_RE.match(lines[index + 1]):
            header_index = index
            break
    if header_index is None:
        return None

    prefix_lines = lines[:header_index]
    header_line = lines[header_index]
    separator_line = lines[header_index + 1]
    row_lines = [line for line in lines[header_index + 2:] if "|" in line]
    suffix_lines = [line for line in lines[header_index + 2:] if "|" not in line and line.strip()]
    preamble = "\n".join(prefix_lines).strip()
    fixed = "\n".join(part for part in (preamble, header_line, separator_line) if part)
    chunks: list[str] = []
    current_rows: list[str] = []

    def render(rows: Sequence[str]) -> str:
        return "\n".join(part for part in (fixed, *rows) if part).strip()

    for row_number, row in enumerate(row_lines, start=1):
        candidate = render([*current_rows, row])
        if count_tokens(tokenizer, candidate) <= max_tokens:
            current_rows.append(row)
            continue

        if current_rows:
            chunks.append(render(current_rows))
            current_rows = []

        single = render([row])
        if count_tokens(tokenizer, single) <= max_tokens:
            current_rows = [row]
            continue

        # Preserve the row identity and column label when one cell is too long.
        header_cells = [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", header_line.strip().strip("|"))]
        row_cells = [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", row.strip().strip("|"))]
        row_key = (row_cells[0] if row_cells else "") or f"row {row_number}"
        cell_parts: list[str] = []
        for column_index, value in enumerate(row_cells):
            if not value:
                continue
            column = header_cells[column_index] if column_index < len(header_cells) else f"column {column_index + 1}"
            cell_prefix = f"{preamble}\nRow key: {row_key}\nColumn: {column}\nValue:".strip()
            prefix_tokens = count_tokens(tokenizer, cell_prefix)
            value_budget = max_tokens - prefix_tokens - 1
            if value_budget < 1:
                raise ValueError(
                    "Table title/context plus row and column labels exceed the token budget; "
                    "refusing to split away structural context."
                )
            value_chunks = split_plain_text_to_budget(value, tokenizer, value_budget)
            cell_parts.extend(f"{cell_prefix} {part}" for part in value_chunks)
        if not cell_parts:
            raise ValueError("A Markdown table row exceeded the token limit but could not be represented safely.")
        chunks.extend(cell_parts)

    if current_rows:
        chunks.append(render(current_rows))
    if suffix_lines and chunks:
        # Attach a small trailing note to the final segment if it still fits.
        suffix = "\n".join(suffix_lines)
        candidate = chunks[-1] + "\n" + suffix
        if count_tokens(tokenizer, candidate) <= max_tokens:
            chunks[-1] = candidate
        else:
            chunks.extend(split_plain_text_to_budget(suffix, tokenizer, max_tokens))

    if not chunks:
        return split_plain_text_to_budget(text, tokenizer, max_tokens)

    for chunk in chunks:
        if count_tokens(tokenizer, chunk) > max_tokens:
            raise RuntimeError("Markdown table splitter returned an oversized chunk.")
    return chunks


def split_final_chunk(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    content_type: str,
) -> list[str]:
    if count_tokens(tokenizer, text) <= max_tokens:
        return [text]
    if content_type in {"table", "toc", "form"}:
        table_parts = split_markdown_table_to_budget(text, tokenizer, max_tokens)
        if table_parts is not None:
            return table_parts
    return split_plain_text_to_budget(text, tokenizer, max_tokens)


# ---------------------------------------------------------------------------
# Docling chunk construction and metadata
# ---------------------------------------------------------------------------


def chunk_item_labels(chunk: Any) -> set[str]:
    meta = getattr(chunk, "meta", None)
    items = list(getattr(meta, "doc_items", None) or [])
    return {item_label(item) for item in items if item_label(item)}


def get_chunk_metadata(chunk: Any, source_file: Path, chunk_index: int) -> dict[str, Any]:
    meta = getattr(chunk, "meta", None)
    headings = list(getattr(meta, "headings", None) or [])
    captions = list(getattr(meta, "captions", None) or [])
    doc_items = list(getattr(meta, "doc_items", None) or [])

    page_numbers: set[int] = set()
    labels: set[str] = set()
    for item in doc_items:
        label = item_label(item)
        if label:
            labels.add(label)
        page_numbers.update(get_item_pages(item))

    if "documentindex" in labels:
        content_type = "toc"
    elif any(label in {"form", "checkboxselected", "checkboxunselected", "keyvalueregion"} for label in labels):
        content_type = "form"
    elif "table" in labels:
        content_type = "table"
    elif "picture" in labels or "chart" in labels:
        content_type = "figure"
    elif "listitem" in labels or "orderedlist" in labels or "unorderedlist" in labels:
        content_type = "list"
    else:
        content_type = "text"

    heading_values = [str(value).strip() for value in headings if str(value).strip()]
    caption_values = [str(value).strip() for value in captions if str(value).strip()]
    metadata: dict[str, Any] = {
        "source": source_file.name,
        "topic": heading_values[-1] if heading_values else source_file.stem,
        "document_title": source_file.stem,
        "heading_path": " > ".join(heading_values) or "Unspecified",
        "module": heading_values[0] if heading_values else "Unspecified",
        "section": heading_values[-1] if heading_values else "Unspecified",
        "chunk": int(chunk_index),
        "content_type": content_type,
        "file_type": source_file.suffix.lower().lstrip("."),
        "doc_item_labels": ",".join(sorted(labels)) or "unknown",
    }
    if caption_values:
        metadata["captions"] = " | ".join(caption_values)[:1000]
    if page_numbers:
        metadata["page_start"] = min(page_numbers)
        metadata["page_end"] = max(page_numbers)
        metadata["page_numbers"] = ",".join(str(page) for page in sorted(page_numbers))
    return metadata


def is_furniture_only_chunk(chunk: Any) -> bool:
    labels = chunk_item_labels(chunk)
    return bool(labels) and labels.issubset({"pageheader", "pagefooter"})


def looks_like_form_table(content: str, labels: set[str]) -> bool:
    if any(label in {"form", "checkboxselected", "checkboxunselected", "keyvalueregion"} for label in labels):
        return True
    return len(CHECKBOX_RE.findall(content)) >= 2


def detect_callout_subtype(content: str, metadata: dict[str, Any]) -> bool:
    """Identify common callout labels without assuming a fixed PDF layout."""
    candidates = [metadata.get("heading_path", ""), metadata.get("captions", "")]
    candidates.extend(line.strip() for line in content.splitlines()[:10] if line.strip())
    return any(CALLOUT_TITLE_RE.match(str(value)) for value in candidates if value)


def assess_chunk_quality(content: str, metadata: dict[str, Any], token_count: int) -> list[str]:
    warnings: list[str] = []
    if len(content.strip()) < 40:
        warnings.append("very_short_chunk_review_context")
    if token_count > TARGET_CHUNK_TOKENS:
        warnings.append("final_chunk_exceeds_target_token_budget")
    if metadata.get("content_type") == "table":
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        has_markdown_header = any(
            "|" in lines[index] and index + 1 < len(lines) and MARKDOWN_TABLE_SEPARATOR_RE.match(lines[index + 1])
            for index in range(len(lines))
        )
        if not has_markdown_header:
            warnings.append("table_not_serialized_as_recognizable_markdown_review_structure")
    if TOC_TRIPLET_ARTIFACT_RE.search(content):
        warnings.append("possible_triplet_serialization_artifact_in_contents_or_index")
    if metadata.get("content_type") == "figure" and not metadata.get("captions"):
        warnings.append("figure_chunk_has_no_extracted_caption_or_text_description")
    if metadata.get("content_subtype") == "callout" and len(content.strip()) < 100:
        warnings.append("callout_chunk_is_short_review_title_body_association")
    if looks_like_form_table(content, set(metadata.get("doc_item_labels", "").split(","))):
        warnings.append("form_like_content_review_fields_and_checkbox_relationships")
    return warnings


def _build_table_serializer_provider():
    """Create a Docling serializer provider that emits Markdown tables."""
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import (
        MarkdownParams,
        MarkdownTableSerializer,
    )

    class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            try:
                params = MarkdownParams(compact_tables=True)
            except TypeError:
                params = MarkdownParams()
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=params,
            )

    return MarkdownTableSerializerProvider()


def create_docling_components():
    """Create converter, HybridChunker and the exact embedding tokenizer."""
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    converter = DocumentConverter()  # Docling defaults include OCR/table structure.
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, use_fast=True)
    docling_tokenizer = HuggingFaceTokenizer(
        tokenizer=embedding_tokenizer,
        max_tokens=TARGET_CHUNK_TOKENS,
    )

    try:
        serializer_provider = _build_table_serializer_provider()
    except (ImportError, AttributeError, TypeError) as exc:
        raise RuntimeError(
            "This builder requires Docling Core's MarkdownTableSerializer support. "
            "Upgrade docling-core/docling to a version exposing "
            "ChunkingDocSerializer and MarkdownTableSerializer. "
            f"Original import error: {exc}"
        ) from exc
    table_serializer = "markdown_compact"

    kwargs = {
        "tokenizer": docling_tokenizer,
        "merge_peers": True,
        "repeat_table_header": True,
    }
    if serializer_provider is not None:
        kwargs["serializer_provider"] = serializer_provider

    try:
        chunker = HybridChunker(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "The installed Docling Core does not support the required HybridChunker "
            "configuration (including repeated Markdown table headers). Upgrade "
            "docling/docling-core to a compatible current version; the builder will "
            "not silently disable table-header repetition. "
            f"Original error: {exc}"
        ) from exc

    return converter, chunker, embedding_tokenizer, table_serializer


# ---------------------------------------------------------------------------
# PDF-level diagnostics (optional PyMuPDF; never changes source content)
# ---------------------------------------------------------------------------


def _wrap_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def inspect_pdf_preflight(source_file: Path) -> dict[str, Any]:
    """Diagnose pages likely to need visual review, when PyMuPDF is installed.

    The page-rotation recommendation is based on the dominant *visible* text
    direction after applying the page's existing /Rotate value. A correction is
    proposed only when at least 55% of the page's text characters share the same
    non-horizontal cardinal orientation. This avoids rotating pages merely
    because they contain vertical side labels.
    """
    if source_file.suffix.lower() != ".pdf":
        return {"available": False, "reason": "not_a_pdf"}
    try:
        import fitz
    except ImportError:
        return {"available": False, "reason": "pymupdf_not_installed"}

    result: dict[str, Any] = {
        "available": True,
        "engine": f"PyMuPDF {getattr(fitz, 'VersionBind', 'unknown')}",
        "page_count": 0,
        "predominantly_rotated_text_pages": [],
        "likely_image_only_pages": [],
        "probable_multi_column_pages": [],
    }
    try:
        opened_pdf = fitz.open(str(source_file))
    except Exception as exc:
        result["error"] = str(exc)
        return result

    with opened_pdf as pdf:
        result["page_count"] = len(pdf)
        for page_index, page in enumerate(pdf, start=1):
            try:
                page_dict = page.get_text("dict")
                orientation_weights: Counter[int] = Counter()
                text_blocks = []
                total_chars = 0

                for block in page_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    block_text_parts = []
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(str(span.get("text", "")) for span in spans).strip()
                        if not line_text:
                            continue
                        weight = len(line_text)
                        total_chars += weight
                        block_text_parts.append(line_text)
                        dx, dy = line.get("dir", (1.0, 0.0))
                        raw_angle = math.degrees(math.atan2(dy, dx))
                        visible_angle = _wrap_angle(raw_angle + float(page.rotation))
                        cardinal = int(round(visible_angle / 90.0) * 90)
                        cardinal = int(_wrap_angle(cardinal))
                        orientation_weights[cardinal] += weight
                    if block_text_parts:
                        bbox = block.get("bbox", (0, 0, 0, 0))
                        text_blocks.append((bbox, " ".join(block_text_parts)))

                images_count = len(page.get_images(full=True))
                if total_chars < 40 and images_count > 0:
                    result["likely_image_only_pages"].append(page_index)

                if total_chars:
                    dominant_angle, dominant_weight = orientation_weights.most_common(1)[0]
                    non_horizontal_ratio = 1.0 - (orientation_weights.get(0, 0) / total_chars)
                    if dominant_angle != 0 and dominant_weight / total_chars >= 0.55:
                        correction = int((-dominant_angle) % 360)
                        result["predominantly_rotated_text_pages"].append({
                            "page": page_index,
                            "rotated_text_ratio": round(dominant_weight / total_chars, 3),
                            "dominant_angle_degrees": dominant_angle,
                            "current_page_rotation": int(page.rotation),
                            "suggested_additional_rotation_degrees": correction,
                        })

                # Conservative multi-column candidate: two sizeable text blocks
                # occupy distinct horizontal bands and overlap vertically.
                width = float(page.rect.width)
                candidates = []
                for bbox, block_text in text_blocks:
                    x0, y0, x1, y1 = bbox
                    if len(block_text) >= 100 and (x1 - x0) <= width * 0.72:
                        candidates.append((x0, y0, x1, y1, len(block_text)))
                found_columns = False
                for i, left in enumerate(candidates):
                    for right in candidates[i + 1:]:
                        overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
                        min_height = max(1.0, min(left[3] - left[1], right[3] - right[1]))
                        horizontally_separate = left[2] <= right[0] or right[2] <= left[0]
                        if horizontally_separate and overlap / min_height >= 0.25:
                            found_columns = True
                            break
                    if found_columns:
                        break
                if found_columns:
                    result["probable_multi_column_pages"].append(page_index)
            except Exception as exc:
                result.setdefault("page_diagnostic_errors", []).append({
                    "page": page_index,
                    "error": str(exc),
                })
    return result


def create_rotation_normalized_pdf(
    source_file: Path,
    preflight: dict[str, Any],
    temp_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Create a temporary copy with only confidently rotated pages corrected."""
    corrections = preflight.get("predominantly_rotated_text_pages", [])
    if not corrections or not preflight.get("available"):
        return source_file, []

    try:
        import fitz
    except ImportError:
        return source_file, []

    corrected: list[dict[str, Any]] = []
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"{source_file.stem}.orientation-normalized.pdf"
    with fitz.open(str(source_file)) as pdf:
        for entry in corrections:
            page_number = int(entry["page"])
            correction = int(entry["suggested_additional_rotation_degrees"])
            if correction not in {90, 180, 270}:
                continue
            page = pdf[page_number - 1]
            old_rotation = int(page.rotation)
            new_rotation = (old_rotation + correction) % 360
            page.set_rotation(new_rotation)
            corrected.append({
                "page": page_number,
                "previous_rotation": old_rotation,
                "new_rotation": new_rotation,
                "applied_rotation_degrees": correction,
            })
        if corrected:
            pdf.save(str(output_path), garbage=4, deflate=True)

    return (output_path if corrected else source_file), corrected


# ---------------------------------------------------------------------------
# Reporting and extraction
# ---------------------------------------------------------------------------


def get_docling_page_coverage(metadatas: Sequence[dict[str, Any]], page_count: int) -> dict[str, Any]:
    pages: set[int] = set()
    for metadata in metadatas:
        serialized = metadata.get("page_numbers")
        if isinstance(serialized, str) and serialized.strip():
            for value in serialized.split(","):
                try:
                    pages.add(int(value))
                except ValueError:
                    continue
            continue
        start = metadata.get("page_start")
        if isinstance(start, int):
            pages.add(start)
    valid_pages = sorted(page for page in pages if page > 0)
    missing = [page for page in range(1, page_count + 1) if page not in pages]
    return {
        "pages_with_chunks": valid_pages,
        "pages_without_chunks": missing,
        "first_page_with_chunk": valid_pages[0] if valid_pages else None,
        "last_page_with_chunk": valid_pages[-1] if valid_pages else None,
        "coverage_ratio": round(len(set(valid_pages)) / page_count, 4) if page_count else None,
    }


def select_sample_indices(metadatas: Sequence[dict[str, Any]], maximum: int = 16) -> list[int]:
    if not metadatas:
        return []
    selected: set[int] = {0, len(metadatas) - 1}
    if len(metadatas) > 2:
        selected.update(round(i * (len(metadatas) - 1) / min(maximum - 1, len(metadatas) - 1)) for i in range(min(maximum, len(metadatas))))
    # Include one representative of every structure type.
    seen_types: set[str] = set()
    for index, metadata in enumerate(metadatas):
        content_type = str(metadata.get("content_type", "text"))
        if content_type not in seen_types:
            selected.add(index)
            seen_types.add(content_type)
    return sorted(selected)


def process_document(
    source_file: Path,
    converter: Any,
    chunker: Any,
    embedding_tokenizer: Any,
    table_serializer: str = "markdown_compact",
) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    """Convert, chunk, normalize, split, validate and report one source file."""
    print(f"\nProcessing {source_file.name}...")
    pdf_preflight = inspect_pdf_preflight(source_file)
    temp_context = tempfile.TemporaryDirectory(prefix="knowledge_pdf_normalization_")
    temp_dir = Path(temp_context.name)
    conversion_result = None

    try:
        try:
            conversion_path, applied_rotations = create_rotation_normalized_pdf(
                source_file, pdf_preflight, temp_dir
            )
        except Exception as exc:
            # Preserve the source and continue with the unmodified PDF; expose
            # the failure in diagnostics rather than aborting silently.
            conversion_path, applied_rotations = source_file, []
            pdf_preflight.setdefault("rotation_normalization_errors", []).append(str(exc))
        if applied_rotations:
            print(f"  Corrected page orientation on {len(applied_rotations)} page(s) for extraction.")
        conversion_result = converter.convert(source=str(conversion_path))
        doc = conversion_result.document
        page_count = get_page_count(doc)
        furniture_report = inspect_furniture(doc)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        token_counts: list[int] = []
        duplicate_map: dict[str, list[int]] = defaultdict(list)
        raw_chunk_count = 0
        skipped_furniture_chunks = 0
        excluded_toc_chunks = 0
        excluded_toc_samples: list[dict[str, Any]] = []

        for raw_index, chunk in enumerate(chunker.chunk(dl_doc=doc)):
            raw_chunk_count += 1
            if is_furniture_only_chunk(chunk):
                skipped_furniture_chunks += 1
                continue

            content = normalize_text(chunker.contextualize(chunk))
            if not content:
                continue

            metadata = get_chunk_metadata(chunk, source_file, raw_index)
            labels = chunk_item_labels(chunk)
            # Contents/index entries are navigation metadata, not knowledge.
            # Keep a sample in the report but do not embed them into the RAG.
            if metadata["content_type"] == "toc":
                excluded_toc_chunks += 1
                if len(excluded_toc_samples) < 12:
                    excluded_toc_samples.append({
                        "source_chunk": raw_index,
                        "page_numbers": sorted(set().union(*(get_item_pages(item) for item in getattr(getattr(chunk, "meta", None), "doc_items", []) or []))),
                        "text": content,
                    })
                continue
            if metadata["content_type"] == "table" and looks_like_form_table(content, labels):
                metadata["content_subtype"] = "form"
            elif detect_callout_subtype(content, metadata):
                metadata["content_subtype"] = "callout"

            final_parts = split_final_chunk(
                content,
                tokenizer=embedding_tokenizer,
                max_tokens=TARGET_CHUNK_TOKENS,
                content_type=metadata["content_type"],
            )

            for split_index, final_content in enumerate(final_parts):
                final_content = normalize_text(final_content)
                if not final_content:
                    continue
                token_count = count_tokens(embedding_tokenizer, final_content)
                if token_count > EMBEDDING_MAX_TOKENS:
                    raise RuntimeError(
                        f"Final chunk is {token_count} tokens (> {EMBEDDING_MAX_TOKENS}) in "
                        f"{source_file.name}, source chunk {raw_index}, split {split_index}. "
                        "Refusing to index truncated content."
                    )

                final_metadata = dict(metadata)
                final_metadata["chunk"] = len(documents)
                final_metadata["source_chunk"] = raw_index
                final_metadata["split_part"] = split_index
                final_metadata["token_count"] = token_count

                final_index = len(documents)
                chunk_id = stable_id(source_file.name, final_index, final_content)
                ids.append(chunk_id)
                documents.append(final_content)
                metadatas.append(final_metadata)
                token_counts.append(token_count)
                duplicate_map[hashlib.sha256(final_content.encode("utf-8")).hexdigest()].append(final_index)

                chunk_warnings = assess_chunk_quality(final_content, final_metadata, token_count)
                if chunk_warnings:
                    warnings.append({
                        "chunk": final_index,
                        "source_chunk": raw_index,
                        "split_part": split_index,
                        "page_start": final_metadata.get("page_start"),
                        "page_end": final_metadata.get("page_end"),
                        "content_type": final_metadata.get("content_type"),
                        "content_subtype": final_metadata.get("content_subtype"),
                        "heading_path": final_metadata.get("heading_path"),
                        "token_count": token_count,
                        "warnings": chunk_warnings,
                        "text": final_content,
                    })

        if not documents:
            raise RuntimeError(
                f"No usable chunks were extracted from {source_file.name}. Existing indexed source was left untouched."
            )

        page_coverage = get_docling_page_coverage(metadatas, page_count)
        type_counts = dict(Counter(metadata["content_type"] for metadata in metadatas))
        exact_duplicates = [
            {"chunk_indices": indices, "count": len(indices)}
            for indices in duplicate_map.values()
            if len(indices) > 1
        ]
        triplet_artifact_chunks = [
            index for index, content in enumerate(documents)
            if TOC_TRIPLET_ARTIFACT_RE.search(content)
        ]
        overflow_chunks = [
            {"chunk": index, "tokens": token_count}
            for index, token_count in enumerate(token_counts)
            if token_count > EMBEDDING_MAX_TOKENS
        ]

        document_level_warnings = [
            *([
                {"type": "predominantly_rotated_text", **entry}
                for entry in pdf_preflight.get("predominantly_rotated_text_pages", [])
            ]),
            *([
                {"type": "likely_image_only_page", "page": page}
                for page in pdf_preflight.get("likely_image_only_pages", [])
            ]),
            *([
                {"type": "probable_multi_column_layout", "page": page}
                for page in pdf_preflight.get("probable_multi_column_pages", [])
            ]),
        ]

        report = {
            "report_schema_version": 2,
            "source": source_file.name,
            "source_pages_detected": page_count,
            "raw_docling_chunks": raw_chunk_count,
            "chunks_created": len(documents),
            "chunk_type_counts": type_counts,
            "table_serializer": table_serializer,
            "target_chunk_tokens": TARGET_CHUNK_TOKENS,
            "embedding_model_max_tokens": EMBEDDING_MAX_TOKENS,
            "token_statistics": {
                "min": min(token_counts) if token_counts else None,
                "median": statistics.median(token_counts) if token_counts else None,
                "mean": round(statistics.mean(token_counts), 2) if token_counts else None,
                "max": max(token_counts) if token_counts else None,
                "chunks_over_target": sum(value > TARGET_CHUNK_TOKENS for value in token_counts),
                "chunks_over_model_limit": len(overflow_chunks),
                "overflow_chunks": overflow_chunks,
            },
            "page_coverage": page_coverage,
            "furniture": furniture_report,
            "pdf_preflight": pdf_preflight,
            "applied_page_rotations": applied_rotations,
            "document_level_warnings": document_level_warnings,
            "cleanup": {
                "heuristic_page_number_deletions": 0,
                "heuristic_header_footer_deletions": 0,
                "note": "No arbitrary standalone numbers or content lines are deleted. Docling furniture is handled separately where available.",
            },
            "skipped_furniture_only_chunks": skipped_furniture_chunks,
            "excluded_toc_chunks": excluded_toc_chunks,
            "excluded_toc_samples": excluded_toc_samples,
            "exact_duplicate_chunks": exact_duplicates,
            "possible_triplet_serialization_artifact_chunks": triplet_artifact_chunks,
            "warnings": warnings,
            "review_notes": [
                "A Transformers sequence-length warning during HybridChunker table measurement can be a false alarm; final stored chunk token counts below are authoritative.",
                "Markdown table serialization preserves visible row/column relationships better than triplet serialization for tables without row headers.",
                "Markdown cannot fully represent merged-cell spans; inspect complex merged headers and forms in the source PDF.",
                "Pages with predominantly rotated text are normalized in a temporary PDF copy before Docling extraction; the source PDF is never modified.",
                "Docling's reading order is retained; multi-column pages and callout title/body association still need representative visual review.",
                "Pages without chunks are reported individually; a blank or image-only page is not automatically an extraction failure.",
            ],
            "sample_chunks": [
                {
                    "chunk": index,
                    "page_start": metadatas[index].get("page_start"),
                    "page_end": metadatas[index].get("page_end"),
                    "content_type": metadatas[index].get("content_type"),
                    "content_subtype": metadatas[index].get("content_subtype"),
                    "heading_path": metadatas[index].get("heading_path"),
                    "token_count": token_counts[index],
                    "text": documents[index],
                }
                for index in select_sample_indices(metadatas)
            ],
        }

        print(
            f"  Extracted {len(documents)} final chunks from {raw_chunk_count} Docling chunks "
            f"({type_counts.get('table', 0)} tables, {type_counts.get('list', 0)} lists, "
            f"{type_counts.get('figure', 0)} figures, "
            f"{sum(1 for item in metadatas if item.get('content_subtype') == 'form')} form-like chunks; "
            f"excluded {excluded_toc_chunks} contents/index chunks)."
        )
        print(
            f"  Final token counts: min={report['token_statistics']['min']}, "
            f"median={report['token_statistics']['median']}, max={report['token_statistics']['max']} "
            f"(target {TARGET_CHUNK_TOKENS}, model limit {EMBEDDING_MAX_TOKENS})."
        )
        print(
            f"  Page coverage: {len(page_coverage['pages_with_chunks'])}/{page_count or '?'} "
            f"pages; {len(page_coverage['pages_without_chunks'])} pages have no chunks."
        )
        if applied_rotations:
            print(f"  Applied page rotations: {applied_rotations}")
        if warnings or document_level_warnings:
            print(f"  Quality review: {len(warnings)} chunk warning(s), {len(document_level_warnings)} page/layout warning(s).")
        else:
            print("  Quality review: no heuristic warnings.")

        return ids, documents, metadatas, report
    finally:
        input_resource = getattr(conversion_result, "input", None) if conversion_result is not None else None
        if input_resource is not None:
            try:
                input_resource.release()
            except Exception:
                pass
        temp_context.cleanup()


def write_report(report: dict[str, Any], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", report["source"])
    path = report_dir / f"{safe_name}.extraction_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_chunk_dump(source_name: str, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[dict[str, Any]], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name)
    path = report_dir / f"{safe_name}.chunks.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for chunk_id, content, metadata in zip(ids, documents, metadatas):
            handle.write(json.dumps({"id": chunk_id, "metadata": metadata, "text": content}, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Safer ChromaDB writes
# ---------------------------------------------------------------------------


def batched_indices(length: int, batch_size: int = CHROMA_BATCH_SIZE):
    for start in range(0, length, batch_size):
        yield start, min(length, start + batch_size)


def upsert_source(collection: Any, source_name: str, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
    """Upsert new chunks first; remove stale old IDs only after successful upsert."""
    existing = collection.get(where={"source": source_name}, include=["metadatas"])
    old_ids = set(existing.get("ids") or [])
    for start, end in batched_indices(len(ids)):
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    stale_ids = sorted(old_ids - set(ids))
    if stale_ids:
        for start, end in batched_indices(len(stale_ids)):
            collection.delete(ids=stale_ids[start:end])


def add_records(collection: Any, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
    for start, end in batched_indices(len(ids)):
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build/update the local Knowledge RAG collection with validated Docling chunks."
    )
    parser.add_argument("--rebuild", action="store_true", help="Recreate the complete knowledge collection after all documents pass extraction.")
    parser.add_argument("--dry-run", action="store_true", help="Extract, validate and write reports without modifying ChromaDB.")
    parser.add_argument("--knowledge-dir", type=Path, default=KNOWLEDGE_DIR, help=f"Document directory (default: {KNOWLEDGE_DIR}).")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR, help=f"Report directory (default: {REPORT_DIR}).")
    parser.add_argument("--no-chunk-dump", action="store_true", help="Do not write full chunk JSONL audit files.")
    args = parser.parse_args(argv)

    knowledge_dir = args.knowledge_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    source_files = sorted(
        file for file in knowledge_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not source_files:
        raise RuntimeError(f"No PDF or Markdown files found in {knowledge_dir}")

    converter, chunker, embedding_tokenizer, table_serializer = create_docling_components()

    # Extract and validate all sources first. In --rebuild mode this prevents a
    # conversion failure from deleting the current production collection.
    prepared: list[tuple[Path, list[str], list[str], list[dict[str, Any]], dict[str, Any]]] = []
    for source_file in source_files:
        ids, documents, metadatas, report = process_document(
            source_file=source_file,
            converter=converter,
            chunker=chunker,
            embedding_tokenizer=embedding_tokenizer,
            table_serializer=table_serializer,
        )
        report_path = write_report(report, report_dir)
        print(f"  Report: {report_path}")
        if not args.no_chunk_dump:
            dump_path = write_chunk_dump(source_file.name, ids, documents, metadatas, report_dir)
            print(f"  Full chunk audit: {dump_path}")
        prepared.append((source_file, ids, documents, metadatas, report))

    total_chunks = sum(len(entry[1]) for entry in prepared)
    total_warnings = sum(
        len(entry[4]["warnings"]) + len(entry[4].get("document_level_warnings", []))
        for entry in prepared
    )

    if args.dry_run:
        print("\nDry run complete. ChromaDB was not modified.")
    else:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        DB_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(DB_DIR))
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

        if args.rebuild:
            # All documents have been converted and validated before this point.
            try:
                client.delete_collection(COLLECTION_NAME)
                print("Deleted previous collection after all source extraction succeeded (--rebuild).")
            except Exception:
                pass
            collection = client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "l2"},
            )
            for _, ids, documents, metadatas, _ in prepared:
                add_records(collection, ids, documents, metadatas)
        else:
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "l2"},
            )
            for source_file, ids, documents, metadatas, _ in prepared:
                upsert_source(collection, source_file.name, ids, documents, metadatas)

        print(f"Collection count: {collection.count()}")
        print(f"Database path: {DB_DIR}")

    print("\nExtraction completed.")
    print(f"Source files processed: {len(prepared)}")
    print(f"Chunks prepared: {total_chunks}")
    print(f"Chunks flagged for review: {total_warnings}")
    print(f"Reports directory: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
