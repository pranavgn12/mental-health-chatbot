"""Dependency-free regression tests for build_knowledge_db.py helpers.

Run:
    python -m unittest -v test_build_knowledge_db.py

These tests exercise token splitting, table serialization post-processing,
metadata/provenance, conservative cleanup, and safe incremental Chroma writes.
They intentionally do not pretend to integration-test Docling/ChromaDB when
those runtime dependencies are not installed in the test environment.
"""
from __future__ import annotations

import re
import tempfile
import unittest
import importlib.util
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
SCRIPT_CANDIDATES = [HERE.parent / "build_knowledge_db.py", HERE.parents[1] / "scripts" / "build_knowledge_db.py"]
for candidate in SCRIPT_CANDIDATES:
    if candidate.exists():
        sys.path.insert(0, str(candidate.parent))
        break

import build_knowledge_db as builder


class FakeTokenizer:
    """Deterministic tokenizer mock: one token per non-whitespace atom + specials."""

    def __call__(self, text, add_special_tokens=True, truncation=False, return_attention_mask=False):
        tokens = re.findall(r"\S+", text)
        if add_special_tokens:
            tokens = ["<s>", *tokens, "</s>"]
        return {"input_ids": tokens}

    def encode(self, text, add_special_tokens=True):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]


class CharTokenizer(FakeTokenizer):
    def __call__(self, text, add_special_tokens=True, truncation=False, return_attention_mask=False):
        tokens = list(text)
        if add_special_tokens:
            tokens = ["<s>", *tokens, "</s>"]
        return {"input_ids": tokens}


class FakeItem:
    def __init__(self, label="text", page_numbers=(1,)):
        self.label = label
        self.prov = [SimpleNamespace(page_no=p) for p in page_numbers]


class FakeChunk:
    def __init__(self, text, labels=("text",), pages=(1,), headings=("Section",), captions=()):
        self.text = text
        self.meta = SimpleNamespace(
            doc_items=[FakeItem(label, pages) for label in labels],
            headings=list(headings),
            captions=list(captions),
        )


class FakeChunker:
    def __init__(self, chunks):
        self._chunks = chunks

    def chunk(self, dl_doc):
        return iter(self._chunks)

    def contextualize(self, chunk):
        heading = " > ".join(chunk.meta.headings)
        return f"{heading}\n{chunk.text}" if heading else chunk.text


class FakeDoc:
    def __init__(self, pages=3):
        self.pages = [object() for _ in range(pages)]
        self.furniture = object()

    def iterate_items(self, root=None, included_content_layers=None):
        return iter([])


class FakeConverter:
    def __init__(self, doc):
        self.doc = doc

    def convert(self, source):
        return SimpleNamespace(document=self.doc, input=None)


class FakeCollection:
    def __init__(self):
        self.records = {}
        self.calls = []

    def get(self, where=None, include=None):
        source = (where or {}).get("source")
        ids = [key for key, value in self.records.items() if value["metadata"].get("source") == source]
        return {"ids": ids, "metadatas": [self.records[key]["metadata"] for key in ids]}

    def upsert(self, ids, documents, metadatas):
        self.calls.append(("upsert", list(ids)))
        for item_id, document, metadata in zip(ids, documents, metadatas):
            self.records[item_id] = {"document": document, "metadata": metadata}

    def delete(self, ids):
        self.calls.append(("delete", list(ids)))
        for item_id in ids:
            self.records.pop(item_id, None)


class TokenSplittingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_plain_text_split_never_exceeds_budget_and_preserves_words(self):
        source = " ".join(f"word{i}" for i in range(137))
        chunks = builder.split_plain_text_to_budget(source, self.tokenizer, 19)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(builder.count_tokens(self.tokenizer, c) <= 19 for c in chunks))
        recovered = " ".join(" ".join(chunks).split())
        self.assertEqual(recovered, source)

    def test_single_long_atom_is_split_without_loss(self):
        source = "A" * 200
        char_tokenizer = CharTokenizer()
        chunks = builder.split_plain_text_to_budget(source, char_tokenizer, 12)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), source)
        self.assertTrue(all(builder.count_tokens(char_tokenizer, c) <= 12 for c in chunks))

    def test_markdown_table_split_repeats_headers_and_keeps_each_row(self):
        rows = [f"| Treatment {i} | Population {i} |" for i in range(1, 16)]
        source = "Table: Recommendations\n| Treatment | Population |\n|---|---|\n" + "\n".join(rows)
        chunks = builder.split_markdown_table_to_budget(source, self.tokenizer, 18)
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("| Treatment | Population |", chunk)
            self.assertIn("|---|---|", chunk)
            self.assertLessEqual(builder.count_tokens(self.tokenizer, chunk), 18)
        combined = "\n".join(chunks)
        for row in rows:
            self.assertIn(row, combined)

    def test_oversized_table_row_keeps_row_and_column_labels(self):
        long_value = " ".join(f"detail{i}" for i in range(80))
        source = (
            "Table: Example\n| ID | Description |\n|---|---|\n"
            f"| item-42 | {long_value} |"
        )
        chunks = builder.split_markdown_table_to_budget(source, self.tokenizer, 18)
        self.assertTrue(chunks)
        self.assertTrue(all(builder.count_tokens(self.tokenizer, c) <= 18 for c in chunks))
        self.assertTrue(all("Row key: item-42" in c for c in chunks))
        value_chunks = [c for c in chunks if "Value: detail" in c]
        self.assertTrue(value_chunks)
        self.assertTrue(all("Column: Description" in c for c in value_chunks))
        recovered = " ".join(chunks)
        for word in long_value.split():
            self.assertIn(word, recovered)


class NormalizationAndMetadataTests(unittest.TestCase):
    def test_normalization_preserves_numbers_and_hyphens(self):
        source = "Step 1\n300 mg\nlow-, middle-\nand high-income\n\n\nNext"
        normalized = builder.normalize_text(source)
        self.assertIn("Step 1", normalized)
        self.assertIn("300 mg", normalized)
        self.assertIn("middle-\nand", normalized)
        self.assertNotIn("middleand", normalized)
        self.assertNotIn("\n\n\n", normalized)

    def test_metadata_preserves_exact_pages_not_a_range_guess(self):
        chunk = FakeChunk("content", pages=(2, 5), headings=("A", "B"))
        metadata = builder.get_chunk_metadata(chunk, Path("example.pdf"), 3)
        self.assertEqual(metadata["page_numbers"], "2,5")
        coverage = builder.get_docling_page_coverage([metadata], 5)
        self.assertEqual(coverage["pages_with_chunks"], [2, 5])
        self.assertEqual(coverage["pages_without_chunks"], [1, 3, 4])

    def test_table_and_toc_types_are_distinguished(self):
        table = FakeChunk("| A | B |\n|---|---|\n|x|y|", labels=("table",))
        toc = FakeChunk("| section | page |", labels=("document_index",))
        self.assertEqual(builder.get_chunk_metadata(table, Path("a.pdf"), 0)["content_type"], "table")
        self.assertEqual(builder.get_chunk_metadata(toc, Path("a.pdf"), 1)["content_type"], "toc")

    def test_form_checkbox_detection(self):
        self.assertTrue(builder.looks_like_form_table("☐ Yes  ☐ No", {"table"}))
        self.assertFalse(builder.looks_like_form_table("ordinary table", {"table"}))

    def test_callout_and_missing_figure_caption_are_flagged(self):
        metadata = {"heading_path": "Chapter 1", "captions": "", "content_type": "text"}
        self.assertTrue(builder.detect_callout_subtype("KEY MESSAGES\nImportant guidance", metadata))
        figure_warnings = builder.assess_chunk_quality(
            "[IMAGE]", {"content_type": "figure", "captions": ""}, 4
        )
        self.assertIn("figure_chunk_has_no_extracted_caption_or_text_description", figure_warnings)

    def test_old_triplet_toc_artifact_is_flagged_not_deleted(self):
        content = "Contents\nPreface, 1 = ix"
        warnings = builder.assess_chunk_quality(content, {"content_type": "text"}, 12)
        self.assertIn("possible_triplet_serialization_artifact_in_contents_or_index", warnings)
        self.assertIn("1 = ix", content)

    def test_page_number_cleanup_is_not_guess_based(self):
        # The new cleaner intentionally preserves all textual numbers.
        text = builder.normalize_text("1\n2\nPage 3\n300 mg")
        self.assertEqual(text, "1\n2\nPage 3\n300 mg")


class DocumentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_process_document_splits_final_contextualized_text_and_audits_tokens(self):
        body = " ".join(f"clinicalword{i}" for i in range(1000))
        chunk = FakeChunk(body, pages=(1, 2), headings=("Chapter", "Section"))
        doc = FakeDoc(pages=3)
        ids, docs, metas, report = builder.process_document(
            source_file=Path("fixture.pdf"),
            converter=FakeConverter(doc),
            chunker=FakeChunker([chunk]),
            embedding_tokenizer=self.tokenizer,
        )
        self.assertGreater(len(docs), 1)
        self.assertEqual(len(ids), len(docs))
        self.assertTrue(all(m["token_count"] <= builder.TARGET_CHUNK_TOKENS for m in metas))
        self.assertEqual(report["token_statistics"]["chunks_over_model_limit"], 0)
        self.assertEqual(report["page_coverage"]["pages_with_chunks"], [1, 2])
        self.assertEqual(report["page_coverage"]["pages_without_chunks"], [3])
        text = " ".join(docs)
        for word in ("clinicalword0", "clinicalword499", "clinicalword999"):
            self.assertIn(word, text)

    def test_furniture_only_chunks_are_skipped(self):
        chunk = FakeChunk("Header title", labels=("page_header",), pages=(1,))
        self.assertTrue(builder.is_furniture_only_chunk(chunk))

    def test_explicit_document_index_chunks_are_excluded_from_embeddings(self):
        toc = FakeChunk("Contents\nSection 1 ........ 12", labels=("document_index",), pages=(1,))
        body = FakeChunk("This is substantive body content that should be embedded.", pages=(2,))
        ids, documents, metadatas, report = builder.process_document(
            source_file=Path("fixture.pdf"),
            converter=FakeConverter(FakeDoc(pages=2)),
            chunker=FakeChunker([toc, body]),
            embedding_tokenizer=self.tokenizer,
        )
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(documents), 1)
        self.assertIn("substantive body content", documents[0])
        self.assertEqual(report["excluded_toc_chunks"], 1)
        self.assertEqual(len(report["excluded_toc_samples"]), 1)


class PdfPreflightTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF is optional")
    def test_rotated_page_is_detected_and_normalized_without_mutating_source(self):
        import fitz

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "layout_fixture.pdf"
            pdf = fitz.open()
            normal = pdf.new_page(width=595, height=842)
            normal.insert_text((50, 80), "A normal horizontal paragraph with enough text to establish normal reading direction.")
            # A vertical side label alone should not force page rotation.
            normal.insert_text((20, 100), "SIDE", rotate=90)

            rotated = pdf.new_page(width=595, height=842)
            for index, value in enumerate((
                "ROTATED TABLE HEADER ONE",
                "ROTATED TABLE HEADER TWO",
                "ROW ONE DATA VALUE",
                "ROW TWO DATA VALUE",
                "ROW THREE DATA VALUE",
            )):
                rotated.insert_text((150 + index * 70, 760), value, rotate=90)
            pdf.save(source)
            pdf.close()

            before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            preflight = builder.inspect_pdf_preflight(source)
            rotated_pages = [entry["page"] for entry in preflight["predominantly_rotated_text_pages"]]
            self.assertEqual(rotated_pages, [2])

            normalized_path, applied = builder.create_rotation_normalized_pdf(
                source, preflight, Path(temp) / "normalized"
            )
            self.assertEqual(applied[0]["page"], 2)
            self.assertEqual(applied[0]["applied_rotation_degrees"], 90)
            self.assertNotEqual(normalized_path, source)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before_hash)
            with fitz.open(normalized_path) as normalized:
                self.assertEqual(normalized[1].rotation, 90)



class ChromaWriteTests(unittest.TestCase):
    def test_upsert_happens_before_stale_ids_are_deleted(self):
        collection = FakeCollection()
        collection.records["old"] = {"document": "old", "metadata": {"source": "doc.pdf"}}
        builder.upsert_source(
            collection,
            "doc.pdf",
            ["new"],
            ["new document"],
            [{"source": "doc.pdf"}],
        )
        self.assertEqual([call[0] for call in collection.calls], ["upsert", "delete"])
        self.assertIn("new", collection.records)
        self.assertNotIn("old", collection.records)


if __name__ == "__main__":
    unittest.main(verbosity=2)
