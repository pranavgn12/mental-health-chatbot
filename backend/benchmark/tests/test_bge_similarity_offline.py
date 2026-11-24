import json
import re
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

from benchmark.bge_similarity_offline import (
    load_cases,
    run,
    pair_chunk_coverage,
    split_memory_paragraphs,
    token_aware_chunks,
)


class FakeTokenizer:
    """Whitespace tokenizer with realistic special-token and offset behavior."""
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=False, truncation=False,
                 return_offsets_mapping=False):
        matches = list(re.finditer(r"\S+", text))
        ids = list(range(100, 100 + len(matches)))
        offsets = [(m.start(), m.end()) for m in matches]
        if add_special_tokens:
            ids = [1] + ids + [2]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class OfflineBGETests(unittest.TestCase):
    def test_short_paragraph_stays_single_chunk(self):
        text = "one two three"
        chunks = token_aware_chunks(text, FakeTokenizer(), max_seq_length=8, overlap=1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)
        self.assertLessEqual(chunks[0]["model_tokens"], 8)

    def test_long_paragraph_is_chunked_and_every_chunk_fits(self):
        text = " ".join(f"word{i}" for i in range(23))
        chunks = token_aware_chunks(text, FakeTokenizer(), max_seq_length=8, overlap=2)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["model_tokens"] <= 8 for chunk in chunks))
        self.assertEqual(chunks[0]["start_token"], 0)
        self.assertEqual(chunks[-1]["end_token"], 23)
        # Adjacent chunks overlap and the source token spans have no gaps.
        for left, right in zip(chunks, chunks[1:]):
            self.assertGreaterEqual(left["end_token"], right["start_token"])

    def test_chunk_overlap_validation(self):
        with self.assertRaises(ValueError):
            token_aware_chunks("a b c", FakeTokenizer(), max_seq_length=8, overlap=6)

    def test_pair_chunk_coverage_is_cosine_for_single_chunk(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        self.assertAlmostEqual(pair_chunk_coverage(a, b, np), 1.0)

    def test_pair_chunk_coverage_handles_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        self.assertAlmostEqual(pair_chunk_coverage(a, b, np), 0.0)

    def test_split_memory_paragraphs_compatible_with_benchmark(self):
        self.assertEqual(
            split_memory_paragraphs("First paragraph.\n\nSecond paragraph."),
            ["First paragraph.", "Second paragraph."],
        )


    def test_full_offline_run_encodes_all_chunks_once_and_writes_reports(self):
        class FakeModel:
            instance = None

            def __init__(self, model_name, device):
                self.model_name = model_name
                self.device = device
                self.max_seq_length = 8
                self.tokenizer = FakeTokenizer()
                self.encode_calls = 0
                self.encoded_text_count = 0
                FakeModel.instance = self

            def encode(self, texts, **kwargs):
                self.encode_calls += 1
                self.encoded_text_count = len(texts)
                # Deterministic finite vectors; no external model/API required.
                return np.asarray([[len(text.split()) + 1.0, 1.0, 0.5] for text in texts], dtype=np.float32)

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeModel)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.json"
            results_dir = root / "run"
            results_dir.mkdir()
            generated_path = results_dir / "generated_memories.json"
            long_text = " ".join(f"fact{i}" for i in range(23))
            dataset_path.write_text(json.dumps({"samples": [{
                "sample_id": "LTM-01", "expected": {"memory": long_text}
            }]}), encoding="utf-8")
            generated_path.write_text(json.dumps({"results": {"LTM-01": {
                "status": "completed", "memory": long_text
            }}}), encoding="utf-8")
            args = Namespace(
                dataset=dataset_path, results_dir=results_dir, generated_file=None,
                output_dir=None, model="fake-bge", device="cpu", batch_size=16,
                chunk_overlap=2, limit=None, quiet=True,
            )
            with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                output_dir = run(args)
            self.assertEqual(FakeModel.instance.encode_calls, 1)
            self.assertGreater(FakeModel.instance.encoded_text_count, 2)
            result_path = output_dir / "bge_similarity_offline_results.json"
            summary_path = output_dir / "bge_similarity_offline_summary.json"
            self.assertTrue(result_path.is_file())
            self.assertTrue(summary_path.is_file())
            report = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(report["truncation_safeguard"]["all_embedding_chunks_within_model_limit"])
            self.assertEqual(report["truncation_safeguard"]["paragraphs_split_into_multiple_chunks"], 2)

    def test_loads_original_samples_and_generated_results_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = tmp / "dataset.json"
            generated = tmp / "generated_memories.json"
            dataset.write_text(json.dumps({"samples": [{
                "sample_id": "LTM-01", "expected": {"memory": "Reference memory."}
            }]}), encoding="utf-8")
            generated.write_text(json.dumps({"results": {"LTM-01": {
                "status": "completed", "memory": "Generated memory."
            }}}), encoding="utf-8")
            cases, results = load_cases(dataset, generated)
            self.assertEqual(cases[0]["sample_id"], "LTM-01")
            self.assertEqual(cases[0]["expected_memory"], "Reference memory.")
            self.assertEqual(results["LTM-01"]["memory"], "Generated memory.")

    def test_loads_new_cases_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = tmp / "dataset.json"
            generated = tmp / "generated.json"
            dataset.write_text(json.dumps({"cases": [{
                "case_id": "LTM-NEW", "expected_output": {"memory": "Expected."}
            }]}), encoding="utf-8")
            generated.write_text(json.dumps({"results": {"LTM-NEW": {
                "status": "completed", "memory": "Actual."
            }}}), encoding="utf-8")
            cases, _ = load_cases(dataset, generated)
            self.assertEqual(cases[0]["sample_id"], "LTM-NEW")
            self.assertEqual(cases[0]["expected_memory"], "Expected.")


if __name__ == "__main__":
    unittest.main()
